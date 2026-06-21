"""Summarize completed provider/model runs for a configured run group."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu


SUMMARY_FIELDS = [
    "run_group_id",
    "run_id",
    "provider_id",
    "profile_id",
    "dataset_id",
    "benchmark_variant",
    "model",
    "task",
    "uq_method",
    "n",
    "accuracy",
    "over_commitment",
    "high_conf_overcommit_90",
    "weak_recall",
    "weak_strengthening_90",
    "text_modality_accuracy",
    "label_text_consistency",
    "parse_failure_rate",
]


def completed_registry_rows(
    registry_rows: list[dict[str, Any]],
    run_group_id: str,
    include_smoke: bool,
    exclude_model_prefixes: list[str] | None = None,
) -> list[dict[str, Any]]:
    exclude_model_prefixes = exclude_model_prefixes or []
    rows = []
    for row in registry_rows:
        if str(row.get("run_group_id", "")) != run_group_id:
            continue
        if str(row.get("status", "")) != "complete":
            continue
        if not include_smoke and str(row.get("run_id", "")).startswith("smoke-"):
            continue
        model = str(row.get("model", ""))
        if any(model.startswith(prefix) for prefix in exclude_model_prefixes):
            continue
        rows.append(row)
    return rows


def annotate_summary(row: dict[str, Any], registry_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_group_id": registry_row.get("run_group_id", ""),
        "run_id": registry_row.get("run_id", ""),
        "provider_id": registry_row.get("provider_id", ""),
        "profile_id": registry_row.get("profile_id", ""),
        "dataset_id": registry_row.get("dataset_id", ""),
        "benchmark_variant": registry_row.get("benchmark_variant", ""),
        **row,
    }


def write_matrix_outputs(rows: list[dict[str, Any]], status_rows: list[dict[str, Any]], output_prefix: Path) -> dict[str, Path]:
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    status_path = output_prefix.with_name(f"{output_prefix.name}_registry_status.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    eu.write_csv_rows(csv_path, rows)
    eu.write_csv_rows(status_path, status_rows, fieldnames=eu.RUN_REGISTRY_FIELDS)

    lines = ["# Run Matrix Summary", ""]
    if rows:
        frame = pd.DataFrame.from_records(rows)
        for field in SUMMARY_FIELDS:
            if field not in frame.columns:
                frame[field] = ""
        lines.append(frame.loc[:, SUMMARY_FIELDS].to_markdown(index=False, floatfmt=".3f"))
    else:
        lines.append("_No completed runs found for this run group._")
    lines.extend(["", "## Registry Status", ""])
    if status_rows:
        status_frame = pd.DataFrame.from_records(status_rows)
        display_fields = [
            "run_id",
            "provider_id",
            "profile_id",
            "model",
            "status",
            "expected_records",
            "observed_records",
            "parse_success_rate",
            "stochastic_complete_item_rate",
        ]
        for field in display_fields:
            if field not in status_frame.columns:
                status_frame[field] = ""
        lines.append(status_frame.loc[:, display_fields].to_markdown(index=False, floatfmt=".3f"))
    else:
        lines.append("_No registry rows found._")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path, "registry_status": status_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed provider/model runs in a run config group.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--variant")
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument(
        "--exclude-model-prefix",
        action="append",
        default=[],
        help="Exclude completed runs whose model id starts with this prefix. Can be repeated.",
    )
    args = parser.parse_args()

    root = eu.project_root()
    run_config = eu.load_run_config(args.config)
    datasets = [eu.normalize_dataset_id(args.dataset)] if args.dataset else list(run_config["datasets"])
    variants = [eu.normalize_benchmark_variant(args.variant)] if args.variant else list(run_config["benchmark_variants"])
    run_group_id = run_config["run_group_id"]

    for dataset_id in datasets:
        for variant in variants:
            registry_path = eu.run_registry_path(root, dataset_id, variant)
            registry_rows = eu.read_csv_rows(registry_path) if registry_path.exists() else []
            complete_rows = completed_registry_rows(
                registry_rows,
                run_group_id,
                include_smoke=args.include_smoke,
                exclude_model_prefixes=args.exclude_model_prefix,
            )
            benchmark_path = eu.artifact_path(root / "data/processed/benchmark_items.csv", dataset_id, variant)
            raw_path = eu.model_outputs_raw_path(root, dataset_id, variant)
            benchmark = eu.read_csv_rows(benchmark_path)
            all_raw_rows = eu.read_jsonl(raw_path)
            summary_rows: list[dict[str, Any]] = []
            ensemble_raw_rows: list[dict[str, Any]] = []

            for registry_row in complete_rows:
                run_id = registry_row["run_id"]
                raw_rows = [
                    row
                    for row in all_raw_rows
                    if str(row.get("run_id", "")) == str(run_id)
                    and str(row.get("model", "")) == str(registry_row.get("model", ""))
                ]
                result_benchmark = eu.benchmark_rows_with_current_raw_outputs(benchmark, raw_rows)
                scores = eu.build_uq_scores(result_benchmark, raw_rows)
                for row in eu.metric_summary_by_model_task_method(scores):
                    summary_rows.append(annotate_summary(row, registry_row))
                ensemble_raw_rows.extend(raw_rows)

            ensemble_scores = eu.build_run_group_ensemble_disagreement_scores(
                benchmark,
                ensemble_raw_rows,
                run_group_id=run_group_id,
            )
            if ensemble_scores:
                registry_stub = {
                    "run_group_id": run_group_id,
                    "run_id": f"{run_group_id}:ensemble",
                    "provider_id": "ensemble",
                    "profile_id": "run_group",
                    "dataset_id": dataset_id,
                    "benchmark_variant": variant,
                }
                for row in eu.metric_summary_by_model_task_method(ensemble_scores):
                    summary_rows.append(annotate_summary(row, registry_stub))

            output_prefix = eu.artifact_path(root / "outputs/run_matrix_summary.csv", dataset_id, variant).with_suffix("")
            paths = write_matrix_outputs(summary_rows, registry_rows, output_prefix)
            print(f"{dataset_id}/{variant}: wrote {paths['markdown']} and {paths['csv']}")


if __name__ == "__main__":
    main()
