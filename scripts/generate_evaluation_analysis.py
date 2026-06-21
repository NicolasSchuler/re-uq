"""Generate headless publication artifacts from a completed benchmark run."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu


PAPER_RESULT_FIELDS = [
    "model",
    "task",
    "uq_method",
    "accuracy",
    "f1_or_macro_f1",
    "over_commitment",
    "brier",
    "ece",
    "auroc",
    "error_detection_auroc",
    "selective_error_defer_10",
    "selective_error_defer_20",
    "monotonicity_violations",
    "monotonicity_strict_violations",
    "monotonicity_tolerance",
    "monotonicity_mean_max_increase",
    "monotonicity_max_increase",
    "unsupported_mandatory_acceptance_90",
    "high_conf_overcommit_all_90",
    "high_conf_overcommit_overcommittable_90",
    "weak_recall",
    "weak_strengthening_90",
    "over_commitment_severity_all",
    "over_commitment_severity_given_overcommitment",
    "text_modality_parse_coverage",
    "heuristic_text_modality_rate",
    "label_text_consistency",
    "text_over_commitment",
    "strict_text_over_commitment",
    "label_correct_text_overcommit_90",
    "strengthening_recall",
    "false_preserve_rate",
    "evidence_phrase_source_rate",
    "parse_failure_rate",
]


def task3_audit_modes_in_rows(rows: list[dict[str, Any]]) -> set[str]:
    modes: set[str] = set()
    for row in rows:
        raw_mode = str(row.get("task3_audit_mode", "")).strip()
        if raw_mode:
            modes.add(eu.normalize_task3_audit_mode(raw_mode))
        else:
            modes.add(eu.LEGACY_TASK3_AUDIT_MODE)
    return modes


def require_task3_audit_mode(rows: list[dict[str, Any]], requested_mode: str) -> None:
    requested_mode = eu.normalize_task3_audit_mode(requested_mode)
    modes = task3_audit_modes_in_rows(rows)
    if modes == {requested_mode}:
        return
    raise ValueError(
        "Task 3 audit mode mismatch. "
        f"Requested {requested_mode!r}, found {sorted(modes)!r}. "
        "Official Task 3 analysis requires blind rows; legacy rows are anchored diagnostics."
    )


def load_task3_items_for_analysis(
    root: Path,
    dataset_id: str,
    variant: str,
    source_run_id: str,
    task3_rows: list[dict[str, Any]],
    model: str | None,
    audit_mode: str,
) -> tuple[list[dict[str, Any]], Path | None]:
    reconstructed = eu.task3_items_from_raw_rows(task3_rows)
    if reconstructed:
        return reconstructed, None

    models = sorted(
        {
            str(row.get("task2_model") or row.get("model") or "").strip()
            for row in task3_rows
            if str(row.get("task2_model") or row.get("model") or "").strip()
        }
    )
    selected_model = model or (models[0] if len(models) == 1 else "")
    if selected_model:
        run_specific_path = eu.task3_verification_items_path(
            root,
            dataset_id,
            variant,
            source_run_id,
            selected_model,
            audit_mode,
        )
        if run_specific_path.exists():
            return eu.read_csv_rows(run_specific_path), run_specific_path

    legacy_path = eu.artifact_path(root / "data/processed/task3_verification_items.csv", dataset_id, variant)
    if eu.normalize_task3_audit_mode(audit_mode) == eu.LEGACY_TASK3_AUDIT_MODE and legacy_path.exists():
        return eu.read_csv_rows(legacy_path), legacy_path
    return [], None


def parse_status_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("parse_status", "")) for row in rows))


def parse_failure_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return math.nan
    return sum(1 for row in rows if str(row.get("parse_status", "")) != "ok") / len(rows)


def require_parse_quality(name: str, rows: list[dict[str, Any]], max_failure_rate: float) -> None:
    rate = parse_failure_rate(rows)
    if math.isnan(rate):
        raise ValueError(f"{name} has no raw rows.")
    if rate > max_failure_rate:
        raise ValueError(
            f"{name} parse failure rate {rate:.3f} exceeds threshold {max_failure_rate:.3f}: "
            f"{parse_status_counts(rows)}"
        )


def require_probability_confidence(name: str, rows: list[dict[str, Any]]) -> None:
    bad_rows: list[str] = []
    for row in rows:
        if str(row.get("parse_status", "")) != "ok" or not isinstance(row.get("parsed_json"), dict):
            continue
        if not eu.prompt_version_uses_confidence_0_1(row.get("prompt_version")):
            continue
        parsed = row["parsed_json"]
        if eu.parse_confidence(parsed.get("confidence"), eu.CONFIDENCE_SCALE_0_1) is None:
            bad_rows.append(f"{row.get('run_id')}:{row.get('task')}:{row.get('item_id')}")
    if bad_rows:
        sample = ", ".join(bad_rows[:5])
        raise ValueError(f"{name} contains v2 rows without probability-scale confidence: {sample}")


def require_registry_complete(
    registry_path: Path,
    run_id: str,
    *,
    model: str | None = None,
    profile_id: str | None = None,
) -> list[dict[str, str]]:
    rows = eu.read_csv_rows(registry_path) if registry_path.exists() else []
    matches = [row for row in rows if str(row.get("run_id", "")) == str(run_id)]
    if model:
        matches = [row for row in matches if str(row.get("model", "")) == model]
    if profile_id:
        matches = [row for row in matches if str(row.get("profile_id", "")) == profile_id]
    if not matches:
        raise ValueError(f"No registry row found for run_id={run_id!r} in {registry_path}.")
    incomplete = [row for row in matches if str(row.get("status", "")) != "complete"]
    if incomplete:
        statuses = sorted({str(row.get("status", "")) for row in incomplete})
        raise ValueError(f"Run {run_id!r} is not complete in {registry_path}: statuses={statuses}")
    return matches


def require_construct_review_complete(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"Weak-modality construct review is missing: {path}")
    status = eu.weak_modality_construct_review_status(eu.read_csv_rows(path))
    if not status["valid"]:
        raise ValueError(f"Weak-modality construct review is incomplete or disagreed: {status}")


def ci_rows_for_scores(scores: list[dict[str, Any]], iterations: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, grouped_rows in eu.grouped(scores, ["model", "task", "uq_method"]).items():
        model, task, uq_method = key

        def acc_metric(sample_rows: list[dict[str, Any]], task: str = str(task)) -> float:
            return eu.task_accuracy(sample_rows, task)

        def brier_metric(sample_rows: list[dict[str, Any]], task: str = str(task)) -> float:
            return eu.brier_score(
                [int(row["y_true"]) for row in sample_rows],
                eu.calibration_probabilities(sample_rows, task),
            )

        acc_point, acc_low, acc_high = eu.bootstrap_seed_metric(grouped_rows, acc_metric, iterations=iterations)
        brier_point, brier_low, brier_high = eu.bootstrap_seed_metric(grouped_rows, brier_metric, iterations=iterations)
        row = {
            "model": model,
            "task": task,
            "uq_method": uq_method,
            "accuracy": acc_point,
            "accuracy_ci_low": acc_low,
            "accuracy_ci_high": acc_high,
            "brier": brier_point,
            "brier_ci_low": brier_low,
            "brier_ci_high": brier_high,
        }
        row.update(eu.headline_risk_ci_fields(grouped_rows, str(task), iterations=iterations))
        rows.append(row)
    return rows


def write_result_notes_template(path: Path) -> None:
    notes = [
        "# Result Notes for IST Manuscript",
        "",
        "## Observations",
        "- Observation: <grounded result from metrics_summary.csv>.",
        "- Observation: <grounded result from task1_p_yes_by_modality.svg>.",
        "- Observation: <grounded high-confidence over-commitment result>.",
        "- Observation: <grounded blind Task 3 text-audit result, if run>.",
        "",
        "## Interpretation",
        "- Hypothesis: <what the observed pattern may imply>.",
        "",
        "## Caveats",
        "- Controlled variants are synthetic minimal pairs.",
        "- Confidence values are verbalized or consistency-derived, not direct internal model uncertainty.",
        "- ACSE-inspired semantic entropy is a five-sample triage signal unless thresholds are calibrated on held-out task data.",
        "- Report exact provider, profile, endpoint family, model ID, prompt version, and run ID.",
        "",
        "## Recommended Next Step",
        "- Recommendation: <best follow-up experiment or paper edit>.",
    ]
    path.write_text("\n".join(notes) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-facing evaluation artifacts from completed raw outputs.")
    parser.add_argument("--dataset", default=eu.DATASET_NICE)
    parser.add_argument("--variant", default="must")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task3-run-id")
    parser.add_argument(
        "--task3-audit-mode",
        choices=eu.TASK3_AUDIT_MODES + [eu.LEGACY_TASK3_AUDIT_MODE],
        default=eu.OFFICIAL_TASK3_AUDIT_MODE,
    )
    parser.add_argument("--model")
    parser.add_argument("--profile")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--expected-stochastic-samples", type=int, default=5)
    parser.add_argument("--max-parse-failure-rate", type=float, default=0.02)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-registry-check", action="store_true")
    parser.add_argument("--skip-construct-review-check", action="store_true")
    args = parser.parse_args()

    root = eu.project_root()
    dataset_id = eu.normalize_dataset_id(args.dataset)
    variant = eu.normalize_benchmark_variant(args.variant)
    output_dir = args.output_dir or root / "outputs" / f"evaluation_{dataset_id}_{variant}_{eu.safe_identifier(args.run_id)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = eu.artifact_path(root / "data/processed/benchmark_items.csv", dataset_id, variant)
    raw_path = eu.model_outputs_raw_path(root, dataset_id, variant)
    registry_path = eu.run_registry_path(root, dataset_id, variant)
    construct_review_path = root / "docs/weak_modality_construct_review.csv"
    task3_legacy_items_path = eu.artifact_path(root / "data/processed/task3_verification_items.csv", dataset_id, variant)
    task3_path = eu.task3_raw_path(root, dataset_id, variant)
    task3_audit_mode = eu.normalize_task3_audit_mode(args.task3_audit_mode)

    benchmark = eu.read_csv_rows(benchmark_path)
    if not benchmark:
        raise ValueError(f"Benchmark file has no rows: {benchmark_path}")
    selected_run_id, raw_rows = eu.select_run_rows(eu.read_jsonl(raw_path), run_id=args.run_id, prefix=None)
    if selected_run_id != args.run_id or not raw_rows:
        raise ValueError(f"No raw rows found for run_id={args.run_id!r} in {raw_path}.")
    if args.model:
        raw_rows = [row for row in raw_rows if str(row.get("model", "")) == args.model]
    if args.profile:
        raw_rows = [row for row in raw_rows if str(row.get("profile_id", "")) == args.profile]
    if not raw_rows:
        raise ValueError("No raw rows remain after applying model/profile filters.")

    if not args.skip_registry_check:
        require_registry_complete(registry_path, args.run_id, model=args.model, profile_id=args.profile)
    if not args.skip_construct_review_check:
        require_construct_review_complete(construct_review_path)
    require_parse_quality("Task 1/2 run", raw_rows, args.max_parse_failure_rate)
    require_probability_confidence("Task 1/2 run", raw_rows)

    result_benchmark = eu.benchmark_rows_with_current_raw_outputs(benchmark, raw_rows)
    stale_item_count = len(benchmark) - len(result_benchmark)
    if stale_item_count and not args.allow_partial:
        raise ValueError(
            f"{stale_item_count} benchmark items do not have current raw prompts in run {args.run_id!r}. "
            "Use --allow-partial only for diagnostics."
        )

    task3_items: list[dict[str, Any]] = []
    task3_rows: list[dict[str, Any]] = []
    task3_items_artifact_path: Path | None = None
    if args.task3_run_id:
        _, task3_rows = eu.select_run_rows(eu.read_jsonl(task3_path), run_id=args.task3_run_id, prefix=None)
        if args.model:
            task3_rows = [row for row in task3_rows if str(row.get("model", "")) == args.model]
        if args.profile:
            task3_rows = [row for row in task3_rows if str(row.get("profile_id", "")) == args.profile]
        if not task3_rows:
            raise ValueError(f"No Task 3 raw rows found for run_id={args.task3_run_id!r}.")
        require_task3_audit_mode(task3_rows, task3_audit_mode)
        task3_items, task3_items_artifact_path = load_task3_items_for_analysis(
            root,
            dataset_id,
            variant,
            args.run_id,
            task3_rows,
            args.model,
            task3_audit_mode,
        )
        task3_items = [row for row in task3_items if str(row.get("task2_run_id", "")) == args.run_id]
        if not task3_items:
            raise ValueError(
                f"No Task 3 items found for source run_id={args.run_id!r}; "
                "new Task 3 rows should contain item provenance, or the run-specific item CSV must exist."
            )
        if not args.skip_registry_check:
            require_registry_complete(eu.task3_registry_path(root, dataset_id, variant), args.task3_run_id, model=args.model, profile_id=args.profile)
        require_parse_quality("Task 3 run", task3_rows, args.max_parse_failure_rate)
        require_probability_confidence("Task 3 run", task3_rows)

    scores = eu.build_uq_scores(result_benchmark, raw_rows)
    task3_scores = eu.build_task3_scores(task3_items, task3_rows) if task3_items and task3_rows else []
    baseline_scores = eu.build_rule_baseline_scores(result_benchmark)
    scores.extend(task3_scores)
    scores.extend(baseline_scores)
    acse_normalized_rows = eu.acse_normalized_score_rows(scores)
    acse_calibration_rows = eu.acse_calibration_diagnostic_rows(acse_normalized_rows)
    summary = eu.metric_summary_by_model_task_method(scores)
    ci_rows = ci_rows_for_scores(scores, iterations=max(1, int(args.bootstrap_iterations)))

    eu.write_csv_rows(output_dir / "uq_scores.csv", scores)
    eu.write_csv_rows(
        output_dir / "acse_semantic_normalized_scores.csv",
        acse_normalized_rows,
        fieldnames=eu.ACSE_NORMALIZED_SCORE_FIELDS,
    )
    eu.write_csv_rows(
        output_dir / "acse_semantic_calibration.csv",
        acse_calibration_rows,
        fieldnames=eu.ACSE_CALIBRATION_FIELDS,
    )
    eu.write_csv_rows(output_dir / "metrics_summary.csv", summary)
    eu.write_csv_rows(output_dir / "bootstrap_seed_ci.csv", ci_rows)
    (output_dir / "acse_semantic_calibration.md").write_text(
        eu.markdown_table(acse_calibration_rows, eu.ACSE_CALIBRATION_FIELDS)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics_summary.md").write_text(eu.markdown_table(summary, PAPER_RESULT_FIELDS) + "\n", encoding="utf-8")
    (output_dir / "bootstrap_seed_ci.md").write_text(
        eu.markdown_table(
            ci_rows,
            [
                "model",
                "task",
                "uq_method",
                "accuracy",
                "accuracy_ci_low",
                "accuracy_ci_high",
                "brier",
                "brier_ci_low",
                "brier_ci_high",
                "unsupported_mandatory_acceptance_80_ci_low",
                "unsupported_mandatory_acceptance_80_ci_high",
                "high_conf_overcommit_overcommittable_80_ci_low",
                "high_conf_overcommit_overcommittable_80_ci_high",
                "weak_strengthening_80_ci_low",
                "weak_strengthening_80_ci_high",
                "label_correct_text_overcommit_80_ci_low",
                "label_correct_text_overcommit_80_ci_high",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "paper_results_table.md").write_text(eu.markdown_table(summary, PAPER_RESULT_FIELDS) + "\n", encoding="utf-8")
    eu.write_task1_modality_svg(scores, output_dir / "task1_p_yes_by_modality.svg")
    eu.write_qualitative_overcommitment_examples(scores, result_benchmark, output_dir, limit=5, threshold=0.80)
    eu.write_uq_method_inventory(output_dir)
    write_result_notes_template(output_dir / "result_notes_template.md")

    provenance = {
        "created_at_utc": eu.utc_now_iso(),
        "dataset_id": dataset_id,
        "benchmark_variant": variant,
        "run_id": args.run_id,
        "task3_run_id": args.task3_run_id or "",
        "task3_audit_mode": task3_audit_mode if args.task3_run_id else "",
        "task3_audit_modes_observed": sorted(task3_audit_modes_in_rows(task3_rows)) if task3_rows else [],
        "model_filter": args.model or "",
        "profile_filter": args.profile or "",
        "benchmark_items": len(benchmark),
        "scored_benchmark_items": len(result_benchmark),
        "stale_item_count": stale_item_count,
        "raw_rows": len(raw_rows),
        "task3_rows": len(task3_rows),
        "score_rows": len(scores),
        "acse_normalized_rows": len(acse_normalized_rows),
        "acse_calibration_rows": len(acse_calibration_rows),
        "summary_rows": len(summary),
        "task1_task2_parse_status": parse_status_counts(raw_rows),
        "task3_parse_status": parse_status_counts(task3_rows),
        "bootstrap_iterations": max(1, int(args.bootstrap_iterations)),
        "expected_stochastic_samples": int(args.expected_stochastic_samples),
        "artifacts": [
            eu.artifact_metadata(benchmark_path, root=root),
            eu.artifact_metadata(raw_path, root=root),
            eu.artifact_metadata(registry_path, root=root),
            eu.artifact_metadata(construct_review_path, root=root),
            eu.artifact_metadata(task3_items_artifact_path or task3_legacy_items_path, root=root),
            eu.artifact_metadata(task3_path, root=root),
            eu.artifact_metadata(root / "prompts/mandatory_entailment.txt", root=root),
            eu.artifact_metadata(root / "prompts/modality_extraction.txt", root=root),
            eu.artifact_metadata(root / "prompts/modality_verification.txt", root=root),
            eu.artifact_metadata(root / "prompts/modality_verification_declared.txt", root=root),
        ],
    }
    (output_dir / "provenance_manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "run_id": args.run_id,
                "task3_run_id": args.task3_run_id or "",
                "task3_audit_mode": task3_audit_mode if args.task3_run_id else "",
                "score_rows": len(scores),
                "acse_calibration_rows": len(acse_calibration_rows),
                "summary_rows": len(summary),
                "stale_item_count": stale_item_count,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
