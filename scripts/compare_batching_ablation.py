"""Compare deterministic Task 2 request sizes and sibling composition.

The final primary protocol is `single`. Archived `grouped` and `shuffled`
names mean 16 items; other sizes are explicit, e.g. `grouped_4`. Shuffled
requests separate siblings, while grouped requests follow benchmark order.
Every delta compares the same jointly eligible exact items, resampled by
capability, against an explicitly named baseline. These intervals condition
on the saved generations and do not model cross-capability request dependence.
Use `--baseline-arm grouped` when reproducing historical 16-item comparisons.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu


DEFAULT_DATASET_ID = "mlm_tapt"
DEFAULT_VARIANT = "must"
DEFAULT_RUN_GROUP_ID = "provider-matrix-v2-2026-05"
DEFAULT_OUTPUT_PREFIX = Path("outputs/batching_ablation_summary")
DEFAULT_BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260518

# Archived 16-item names remain stable; other sizes must not collapse into them.
ARM_GROUPED = "grouped"
ARM_SHUFFLED = "shuffled"
ARM_SINGLE = "single"
STRATA = ("all", "weak_intent")
METRICS: tuple[tuple[str, Callable[[list[dict[str, Any]]], float]], ...] = (
    ("label_accuracy", lambda rows: eu.task_accuracy(rows, "task2")),
    ("strict_text_strengthening", lambda rows: eu.text_strengthening_rate(rows, True)),
    ("broad_text_strengthening", lambda rows: eu.text_strengthening_rate(rows, False)),
)

ARM_FIELDS = [
    "model",
    "arm",
    "stratum",
    "run_id",
    "batch_size",
    "batch_order",
    "n",
    "n_failed",
    "n_text_readable",
    "label_accuracy",
    "strict_text_strengthening",
    "strict_text_strengthening_ci_low",
    "strict_text_strengthening_ci_high",
    "strict_text_strengthening_seed_ci_low",
    "strict_text_strengthening_seed_ci_high",
    "broad_text_strengthening",
    "broad_text_strengthening_ci_low",
    "broad_text_strengthening_ci_high",
    "broad_text_strengthening_seed_ci_low",
    "broad_text_strengthening_seed_ci_high",
    "bootstrap_ci_cluster_field",
    "weak_strict_text_strengthening_90",
]
DELTA_FIELDS = [
    "model",
    "arm",
    "stratum",
    "metric",
    "baseline_arm",
    "baseline_value",
    "arm_value",
    "full_arm_baseline_descriptive",
    "full_arm_comparison_descriptive",
    "delta",
    "delta_ci_low",
    "delta_ci_high",
    "delta_cluster_field",
    "n_delta_clusters",
    "n_baseline",
    "n_arm",
    "n_complete_pairs",
    "n_excluded_single_arm",
    "n_matched_capabilities",
    "n_excluded_ineligible_items",
    "n_duplicate_identities",
    "n_missing_identity_rows",
    "n_baseline_failed_items",
    "n_comparison_failed_items",
    "n_baseline_unclassified_items",
    "n_comparison_unclassified_items",
    "n_baseline_eligible_items",
    "n_comparison_eligible_items",
    "unavailable_reason",
    "delta_ci_unavailable_reason",
    "cluster_note",
]


def registry_arm(row: Any) -> str:
    """Which arm a registry row belongs to, from its batching plan.

    A registry row written before `batch_order` existed is a grouped paper run;
    that is the same reading `select_cell_runs` applies to the archive.
    """
    return eu.batching_arm_name(
        row.get("batch_size", 0),
        str(row.get("batch_order", "") or eu.DEFAULT_BATCH_ORDER),
    )


def select_arm_runs(
    registry_rows: list[dict[str, Any]],
    *,
    run_group_id: str,
    include_smoke: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Latest complete, fully covered run per (model, arm)."""
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in registry_rows:
        if str(row.get("run_group_id", "")) != run_group_id:
            continue
        if str(row.get("status", "")) != "complete":
            continue
        if not include_smoke and str(row.get("run_id", "")).startswith("smoke-"):
            continue
        try:
            coverage = float(row.get("deterministic_item_coverage", 0) or 0)
        except (TypeError, ValueError):
            coverage = 0.0
        if coverage < 1.0:
            continue
        key = (str(row.get("model", "")), registry_arm(row))
        current = selected.get(key)
        order = (str(row.get("started_at_utc", "")), str(row.get("run_id", "")))
        if current is None or order > (
            str(current.get("started_at_utc", "")),
            str(current.get("run_id", "")),
        ):
            selected[key] = row
    return selected


def task2_scores(
    benchmark: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    *,
    sampling_plan: eu.SamplingPlan,
) -> list[dict[str, Any]]:
    """Deterministic Task 2 score rows of one arm."""
    deterministic = [
        row
        for row in eu.dedupe_raw_rows(raw_rows)
        if row.get("task") == "task2" and row.get("sample_kind") == "deterministic"
    ]
    by_id = {str(item["item_id"]): item for item in benchmark}
    for row in deterministic:
        item = by_id.get(str(row.get("item_id", "")))
        if item is None or not eu.raw_record_matches_benchmark_item(row, item):
            raise ValueError(
                f"Raw source differs from current benchmark: {row.get('item_id')}"
            )
    scores = eu.build_uq_scores(benchmark, deterministic, sampling_plan=sampling_plan)
    by_item = {str(row["item_id"]): row for row in scores}
    observations = []
    for raw in deterministic:
        item = by_id[str(raw["item_id"])]
        score = by_item.get(str(raw["item_id"]))
        observations.append(
            {
                **(score or {}),
                "item_id": item["item_id"],
                "source_identity": f"{item.get('source_corpus', '')}::{item['seed_id']}",
                "seed_id": item["seed_id"],
                "model": raw.get("model", ""),
                "run_id": raw.get("run_id", ""),
                "batch_id": raw.get("batch_id", ""),
                "dataset_id": raw.get("dataset_id", ""),
                "benchmark_variant": raw.get("benchmark_variant", "must"),
                "task": "task2",
                "source_modality": item["source_modality"],
                "gold_modality": item["source_modality"],
                "response_status": "ok"
                if score
                else raw.get("parse_status") or "missing_score",
            }
        )
    return observations


def stratum_rows(rows: list[dict[str, Any]], stratum: str) -> list[dict[str, Any]]:
    if stratum == "all":
        return rows
    if stratum == "weak_intent":
        return [
            row for row in rows if str(row.get("source_modality", "")) == "nice_to_have"
        ]
    raise ValueError(f"Unknown stratum: {stratum}")


def _weak_strict_90(rows: list[dict[str, Any]]) -> float | str:
    readable = [
        row for row in rows if str(row.get("text_modality_parse_status", "")) == "ok"
    ]
    if not readable:
        return ""
    strengthened = sum(
        1
        for row in rows
        if eu.is_truthy_strict(row.get("strict_text_high_conf_overcommit_90"))
    )
    return strengthened / len(readable)


def _finite(value: float) -> float | str:
    return "" if isinstance(value, float) and math.isnan(value) else value


def arm_row(
    model: str,
    arm: str,
    stratum: str,
    registry_row: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    # Failed observations (request, output-format or parse failures) carry no
    # prediction; they are counted, not scored. `n` stays the planned total.
    answered = [row for row in rows if row.get("response_status", "ok") == "ok"]
    ci = eu.text_over_commitment_ci_fields(
        answered, iterations=bootstrap_samples, seed=BOOTSTRAP_SEED
    )
    return {
        "model": model,
        "arm": arm,
        "stratum": stratum,
        "run_id": str(registry_row.get("run_id", "")),
        "batch_size": registry_row.get("batch_size", ""),
        "batch_order": registry_row.get("batch_order", ""),
        "n": len(rows),
        "n_failed": len(rows) - len(answered),
        "n_text_readable": ci["text_over_commitment_n_denominator"],
        "label_accuracy": eu.task_accuracy(answered, "task2") if answered else "",
        "strict_text_strengthening": ci["strict_text_over_commitment"],
        "strict_text_strengthening_ci_low": ci["strict_text_over_commitment_ci_low"],
        "strict_text_strengthening_ci_high": ci["strict_text_over_commitment_ci_high"],
        "strict_text_strengthening_seed_ci_low": ci[
            "strict_text_over_commitment_seed_ci_low"
        ],
        "strict_text_strengthening_seed_ci_high": ci[
            "strict_text_over_commitment_seed_ci_high"
        ],
        "broad_text_strengthening": ci["text_over_commitment"],
        "broad_text_strengthening_ci_low": ci["text_over_commitment_ci_low"],
        "broad_text_strengthening_ci_high": ci["text_over_commitment_ci_high"],
        "broad_text_strengthening_seed_ci_low": ci["text_over_commitment_seed_ci_low"],
        "broad_text_strengthening_seed_ci_high": ci[
            "text_over_commitment_seed_ci_high"
        ],
        "bootstrap_ci_cluster_field": ci.get("bootstrap_ci_cluster_field", ""),
        "weak_strict_text_strengthening_90": _weak_strict_90(answered)
        if stratum == "weak_intent"
        else "",
    }


def delta_rows(
    model: str,
    arm: str,
    stratum: str,
    grouped: list[dict[str, Any]],
    other: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    baseline_arm: str = ARM_GROUPED,
) -> list[dict[str, Any]]:
    """`arm - baseline` per metric, paired by item and resampled by seed."""
    rows: list[dict[str, Any]] = []
    for metric_name, metric in METRICS:

        def eligible(row, name=metric_name):
            return (
                row.get("response_status", "ok") == "ok"
                and row.get("pred_modality") in eu.MODALITIES
                and (
                    name == "label_accuracy"
                    or row.get("text_modality_parse_status") == "ok"
                )
            )

        a, b, counts = eu.exact_item_metric_pairs(grouped, other, eligible)
        paired = eu.bootstrap_seed_metric_delta(
            a,
            b,
            metric,
            # Capability-conditional sensitivity interval: changing request
            # partitions can induce dependence across capabilities. This does
            # not estimate variability over new request compositions/reruns.
            cluster_field=eu.BOOTSTRAP_CLUSTER_FALLBACK_FIELD,
            pair_field="exact_pair_id",
            iterations=bootstrap_samples,
            seed=BOOTSTRAP_SEED,
        )
        rows.append(
            {
                "model": model,
                "arm": arm,
                "stratum": stratum,
                "metric": metric_name,
                "baseline_arm": baseline_arm,
                "baseline_value": _finite(metric(a)) if a else "",
                "arm_value": _finite(metric(b)) if b else "",
                "full_arm_baseline_descriptive": _finite(
                    metric([r for r in grouped if eligible(r)])
                ),
                "full_arm_comparison_descriptive": _finite(
                    metric([r for r in other if eligible(r)])
                ),
                "delta": _finite(paired.delta),
                "delta_ci_low": _finite(paired.ci_low)
                if paired.n_clusters >= 2
                else "",
                "delta_ci_high": _finite(paired.ci_high)
                if paired.n_clusters >= 2
                else "",
                "delta_cluster_field": paired.cluster_field,
                "n_delta_clusters": paired.n_clusters,
                "n_baseline": len(grouped),
                "n_arm": len(other),
                "n_complete_pairs": counts["n_matched_items"],
                "n_excluded_single_arm": counts["n_unmatched_items"],
                **{
                    key: counts[key]
                    for key in (
                        "n_matched_capabilities",
                        "n_excluded_ineligible_items",
                        "n_duplicate_identities",
                        "n_missing_identity_rows",
                    )
                },
                **{
                    f"n_{arm_name}_{suffix}": counts[f"n_{source}_{suffix}"]
                    for arm_name, source in (
                        ("baseline", "bare"),
                        ("comparison", "document"),
                    )
                    for suffix in (
                        "failed_items",
                        "unclassified_items",
                        "eligible_items",
                    )
                },
                "unavailable_reason": "" if a else "no jointly eligible exact items",
                "delta_ci_unavailable_reason": "fewer than two capabilities"
                if paired.n_clusters < 2
                else "bootstrap disabled"
                if bootstrap_samples <= 0
                else "",
                "cluster_note": "capability-conditional; cross-capability request dependence not modelled",
            }
        )
    return rows


def build_tables(
    benchmark: list[dict[str, Any]],
    registry_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    *,
    run_group_id: str,
    include_smoke: bool,
    bootstrap_samples: int,
    sampling_plan: eu.SamplingPlan,
    baseline_arm: str = ARM_SINGLE,
) -> dict[str, Any]:
    selected = select_arm_runs(
        registry_rows, run_group_id=run_group_id, include_smoke=include_smoke
    )
    raw_by_run: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        raw_by_run.setdefault(
            (str(row.get("run_id", "")), str(row.get("model", ""))), []
        ).append(row)

    scores_by_arm: dict[tuple[str, str], list[dict[str, Any]]] = {}
    provenance: list[dict[str, Any]] = []
    for (model, arm), registry_row in sorted(selected.items()):
        run_id = str(registry_row["run_id"])
        scores_by_arm[(model, arm)] = task2_scores(
            benchmark,
            raw_by_run.get((run_id, model), []),
            sampling_plan=sampling_plan,
        )
        provenance.append(
            {
                "model": model,
                "arm": arm,
                "run_id": run_id,
                "run_group_id": run_group_id,
                "started_at_utc": registry_row.get("started_at_utc", ""),
                "batch_size": registry_row.get("batch_size", ""),
                "batch_order": registry_row.get("batch_order", ""),
                "notes": registry_row.get("notes", ""),
                "task2_rows": len(scores_by_arm[(model, arm)]),
            }
        )

    arm_table: list[dict[str, Any]] = []
    delta_table: list[dict[str, Any]] = []
    for model in sorted({model for model, _ in scores_by_arm}):
        arms = sorted(arm for m, arm in scores_by_arm if m == model)
        # Keep the reference visible first, even when it is absent (deltas
        # then explicitly report no matched items rather than inventing zeros).
        arms = [baseline_arm, *[arm for arm in arms if arm != baseline_arm]]
        for stratum in STRATA:
            per_arm = {
                arm: stratum_rows(scores_by_arm.get((model, arm), []), stratum)
                for arm in arms
            }
            for arm in arms:
                if (model, arm) not in scores_by_arm:
                    continue
                arm_table.append(
                    arm_row(
                        model,
                        arm,
                        stratum,
                        selected[(model, arm)],
                        per_arm[arm],
                        bootstrap_samples=bootstrap_samples,
                    )
                )
            for arm in arms:
                if arm == baseline_arm or (model, arm) not in scores_by_arm:
                    continue
                delta_table.extend(
                    delta_rows(
                        model,
                        arm,
                        stratum,
                        per_arm[baseline_arm],
                        per_arm[arm],
                        bootstrap_samples=bootstrap_samples,
                        baseline_arm=baseline_arm,
                    )
                )
    return {
        "arms": arm_table,
        "deltas": delta_table,
        "provenance": provenance,
        "baseline_arm": baseline_arm,
    }


def write_outputs(tables: dict[str, Any], output_prefix: Path) -> dict[str, Path]:
    baseline_arm = tables.get("baseline_arm", ARM_GROUPED)
    csv_path = output_prefix.with_suffix(".csv")
    delta_csv_path = output_prefix.with_name(f"{output_prefix.name}_deltas.csv")
    md_path = output_prefix.with_suffix(".md")
    provenance_path = output_prefix.with_name(f"{output_prefix.name}_provenance.json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    eu.write_csv_rows(csv_path, tables["arms"], fieldnames=ARM_FIELDS)
    eu.write_csv_rows(delta_csv_path, tables["deltas"], fieldnames=DELTA_FIELDS)
    lines = [
        "# Batching Ablation Summary",
        "",
        "Deterministic Task 2 rows of one cell under different request compositions.",
        "`single` means one item; `grouped`/`shuffled` mean 16 items, and",
        "other sizes have a suffix (e.g. `grouped_4`). Shuffled requests never",
        "contain two variants of one capability. Exact sizes are in the arm table.",
        f"Reference arm: `{baseline_arm}`. Strata: `all` and `weak_intent`.",
        "",
        "Per-arm CIs are the usual request-clustered bootstraps with the",
        "seed-clustered pair alongside. The **deltas** are resampled by seed,",
        "conditional on these generations and request compositions; they do not",
        "model cross-capability request dependence or variability across reruns.",
        "Pairs are exact source items eligible for the metric in both arms",
        "(`n_complete_pairs`). Unmatched, ineligible, failed, unclassified and",
        "ambiguous identities are counted separately. Full-arm rates are descriptive.",
        "",
        "An interval containing zero is not evidence of invariance. Grouped",
        "rates are protocol-specific estimates, not bounds on another protocol.",
        "",
        "## Arms",
        "",
        eu.markdown_table(tables["arms"], ARM_FIELDS),
        "",
        f"## Deltas (arm - {baseline_arm})",
        "",
        eu.markdown_table(tables["deltas"], DELTA_FIELDS),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    eu.write_json(
        provenance_path,
        {
            "generated_at_utc": eu.utc_now_iso(),
            "baseline_arm": baseline_arm,
            "runs": tables["provenance"],
        },
    )
    return {
        "csv": csv_path,
        "deltas_csv": delta_csv_path,
        "markdown": md_path,
        "provenance": provenance_path,
    }


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=DEFAULT_DATASET_ID)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--run-group-id", default=DEFAULT_RUN_GROUP_ID)
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument(
        "--baseline-arm",
        default=ARM_SINGLE,
        help="Reference arm (single by default; grouped for archived 16-item studies).",
    )
    parser.add_argument(
        "--run-id", action="append", help="Only compare these run IDs (repeatable)."
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Cluster-bootstrap resamples (0 disables the CIs).",
    )
    parser.add_argument(
        "--stochastic-samples",
        type=int,
        default=0,
        help="Stochastic samples per item the compared runs planned (arms are "
        "deterministic-only by default).",
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args(argv)

    root = eu.project_root()
    dataset_id = eu.normalize_dataset_id(args.dataset)
    variant = eu.normalize_benchmark_variant(args.variant)
    benchmark = eu.read_csv_rows(
        eu.artifact_path(
            root / "data/processed/benchmark_items.csv", dataset_id, variant
        )
    )
    registry_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for smoke in [False, True] if args.include_smoke else [False]:
        registry_path = eu.run_registry_path(root, dataset_id, variant, smoke=smoke)
        if registry_path.exists():
            registry_rows.extend(eu.read_csv_rows(registry_path))
        raw_path = eu.model_outputs_raw_path(root, dataset_id, variant, smoke=smoke)
        if raw_path.exists():
            raw_rows.extend(eu.read_jsonl(raw_path))
    if args.run_id:
        selected = set(args.run_id)
        registry_rows = [row for row in registry_rows if row.get("run_id") in selected]
        raw_rows = [row for row in raw_rows if row.get("run_id") in selected]
    tables = build_tables(
        benchmark,
        registry_rows,
        raw_rows,
        run_group_id=args.run_group_id,
        include_smoke=args.include_smoke,
        bootstrap_samples=args.bootstrap_samples,
        sampling_plan=eu.SamplingPlan(stochastic_samples=args.stochastic_samples),
        baseline_arm=args.baseline_arm,
    )
    output_prefix = (
        args.output_prefix
        if args.output_prefix.is_absolute()
        else root / args.output_prefix
    )
    paths = write_outputs(tables, output_prefix)
    print(
        f"{len(tables['provenance'])} arm runs, {len(tables['arms'])} arm rows, "
        f"{len(tables['deltas'])} delta rows -> {paths['markdown']}"
    )
    return paths


if __name__ == "__main__":
    main()
