"""Compare the bare and document arms of the context ablation (TODO section B).

Reads the `pure` run registry and raw outputs, picks the latest complete run
per (model, item_context), scores each arm exactly as the run-matrix
comparison does (deterministic Task 2 rows), and writes one table:

* one row per model x arm x stratum (`all`, `weak_intent`, `marker_M`,
  `marker_O`) with n, declared-label accuracy, strict and broad text
  strengthening with seed-clustered CIs, and the README-style weak-intent
  rate where the stratum is weak-intent;
* one delta row per model x stratum (`document - bare`) with a *paired*
  seed-clustered bootstrap CI (`eu.bootstrap_seed_metric_delta`).

Outputs `outputs/context_ablation_summary.{csv,md}` and a provenance JSON
listing the run ids and resolved-config digests behind every row. Nothing
here touches the paper-facing exports; see docs/context_ablation.md.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu


DATASET_ID = eu.DATASET_PURE
VARIANT = "must"
DEFAULT_RUN_GROUP_ID = "context-ablation-2026-09"
DEFAULT_OUTPUT_PREFIX = Path("outputs/context_ablation_summary")
DEFAULT_BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260518
STRATA = ("all", "weak_intent", "marker_M", "marker_O")
METRICS: tuple[tuple[str, Callable[[list[dict[str, Any]]], float]], ...] = (
    ("label_accuracy", lambda rows: eu.task_accuracy(rows, "task2")),
    ("strict_text_strengthening", lambda rows: eu.text_strengthening_rate(rows, True)),
    ("broad_text_strengthening", lambda rows: eu.text_strengthening_rate(rows, False)),
)
ARM_FIELDS = [
    "model",
    "item_context",
    "stratum",
    "run_id",
    "n",
    "n_text_readable",
    "label_accuracy",
    "strict_text_strengthening",
    "strict_text_strengthening_ci_low",
    "strict_text_strengthening_ci_high",
    "broad_text_strengthening",
    "broad_text_strengthening_ci_low",
    "broad_text_strengthening_ci_high",
    "weak_strict_text_strengthening_90",
]
DELTA_FIELDS = [
    "model",
    "stratum",
    "metric",
    "bare",
    "document",
    "delta",
    "delta_ci_low",
    "delta_ci_high",
    "n_bare",
    "n_document",
]


def select_arm_runs(
    registry_rows: list[dict[str, Any]],
    *,
    run_group_id: str,
    include_smoke: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Latest complete, fully covered run per (model, item_context)."""
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
        # Registry rows written before the knob existed have a blank column;
        # blank is the bare paper condition, exactly as for batch_order.
        arm = eu.normalize_item_context(row.get("item_context"))
        key = (str(row.get("model", "")), arm)
        current = selected.get(key)
        order = (str(row.get("started_at_utc", "")), str(row.get("run_id", "")))
        if current is None or order > (
            str(current.get("started_at_utc", "")),
            str(current.get("run_id", "")),
        ):
            selected[key] = row
    return selected


def task2_scores(
    benchmark: list[dict[str, Any]], raw_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Deterministic Task 2 score rows with the item's author marker joined on."""
    marker_by_item = {
        str(row["item_id"]): str(row.get("context_marker", "")) for row in benchmark
    }
    current = eu.benchmark_rows_with_current_raw_outputs(benchmark, raw_rows)
    scores = eu.build_uq_scores(current, raw_rows)
    return [
        {**row, "context_marker": marker_by_item.get(str(row["item_id"]), "")}
        for row in scores
        if str(row.get("task", "")) == "task2"
        and str(row.get("uq_method", "")) == "verbalized_confidence"
    ]


def stratum_rows(rows: list[dict[str, Any]], stratum: str) -> list[dict[str, Any]]:
    if stratum == "all":
        return rows
    if stratum == "weak_intent":
        return [r for r in rows if str(r.get("gold_modality", "")) == "nice_to_have"]
    if stratum.startswith("marker_"):
        marker = stratum.removeprefix("marker_")
        return [r for r in rows if str(r.get("context_marker", "")) == marker]
    raise ValueError(f"Unknown stratum: {stratum}")


def _weak_strict_90(rows: list[dict[str, Any]]) -> float | str:
    readable = [r for r in rows if str(r.get("text_modality_parse_status", "")) == "ok"]
    if not readable:
        return ""
    strengthened = sum(
        1
        for r in rows
        if eu.is_truthy_strict(r.get("strict_text_high_conf_overcommit_90"))
    )
    return strengthened / len(readable)


def arm_row(
    model: str,
    arm: str,
    stratum: str,
    run_id: str,
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    ci = eu.text_over_commitment_ci_fields(
        rows, iterations=bootstrap_samples, seed=BOOTSTRAP_SEED
    )
    return {
        "model": model,
        "item_context": arm,
        "stratum": stratum,
        "run_id": run_id,
        "n": len(rows),
        "n_text_readable": ci["text_over_commitment_n_denominator"],
        "label_accuracy": eu.task_accuracy(rows, "task2") if rows else "",
        "strict_text_strengthening": ci["strict_text_over_commitment"],
        "strict_text_strengthening_ci_low": ci["strict_text_over_commitment_ci_low"],
        "strict_text_strengthening_ci_high": ci["strict_text_over_commitment_ci_high"],
        "broad_text_strengthening": ci["text_over_commitment"],
        "broad_text_strengthening_ci_low": ci["text_over_commitment_ci_low"],
        "broad_text_strengthening_ci_high": ci["text_over_commitment_ci_high"],
        "weak_strict_text_strengthening_90": _weak_strict_90(rows)
        if stratum == "weak_intent"
        else "",
    }


def _finite(value: float) -> float | str:
    return "" if isinstance(value, float) and math.isnan(value) else value


def delta_rows(
    model: str,
    stratum: str,
    bare: list[dict[str, Any]],
    document: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name, metric in METRICS:
        if bare and document:
            point, low, high = eu.bootstrap_seed_metric_delta(
                bare,
                document,
                metric,
                iterations=bootstrap_samples,
                seed=BOOTSTRAP_SEED,
            )
        else:
            point = low = high = math.nan
        rows.append(
            {
                "model": model,
                "stratum": stratum,
                "metric": metric_name,
                "bare": _finite(metric(bare)) if bare else "",
                "document": _finite(metric(document)) if document else "",
                "delta": _finite(point),
                "delta_ci_low": _finite(low),
                "delta_ci_high": _finite(high),
                "n_bare": len(bare),
                "n_document": len(document),
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
            benchmark, raw_by_run.get((run_id, model), [])
        )
        provenance.append(
            {
                "model": model,
                "item_context": arm,
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
    models = sorted({model for model, _ in scores_by_arm})
    for model in models:
        arms = {
            arm: scores_by_arm.get((model, arm), [])
            for arm in (eu.ITEM_CONTEXT_BARE, eu.ITEM_CONTEXT_DOCUMENT)
        }
        for stratum in STRATA:
            per_arm = {arm: stratum_rows(rows, stratum) for arm, rows in arms.items()}
            for arm, rows in per_arm.items():
                if (model, arm) not in scores_by_arm:
                    continue
                arm_table.append(
                    arm_row(
                        model,
                        arm,
                        stratum,
                        str(selected[(model, arm)]["run_id"]),
                        rows,
                        bootstrap_samples=bootstrap_samples,
                    )
                )
            delta_table.extend(
                delta_rows(
                    model,
                    stratum,
                    per_arm[eu.ITEM_CONTEXT_BARE],
                    per_arm[eu.ITEM_CONTEXT_DOCUMENT],
                    bootstrap_samples=bootstrap_samples,
                )
            )
    return {"arms": arm_table, "deltas": delta_table, "provenance": provenance}


def write_outputs(tables: dict[str, Any], output_prefix: Path) -> dict[str, Path]:
    csv_path = output_prefix.with_suffix(".csv")
    delta_csv_path = output_prefix.with_name(f"{output_prefix.name}_deltas.csv")
    md_path = output_prefix.with_suffix(".md")
    provenance_path = output_prefix.with_name(f"{output_prefix.name}_provenance.json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    eu.write_csv_rows(csv_path, tables["arms"], fieldnames=ARM_FIELDS)
    eu.write_csv_rows(delta_csv_path, tables["deltas"], fieldnames=DELTA_FIELDS)
    lines = [
        "# Context Ablation Summary",
        "",
        "Deterministic Task 2 rows of the `pure` cell, bare vs document context.",
        "Strata: `all`, `weak_intent` (source modality nice_to_have), `marker_M` /",
        "`marker_O` (the author's marker on the seed requirement). CIs are",
        "seed-clustered bootstraps; delta CIs are paired over the same seeds.",
        "",
        "## Arms",
        "",
        eu.markdown_table(tables["arms"], ARM_FIELDS),
        "",
        "## Deltas (document - bare)",
        "",
        eu.markdown_table(tables["deltas"], DELTA_FIELDS),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    provenance_path.write_text(
        json.dumps(
            {
                "generated_at_utc": eu.utc_now_iso(),
                "dataset_id": DATASET_ID,
                "benchmark_variant": VARIANT,
                "runs": tables["provenance"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "csv": csv_path,
        "deltas_csv": delta_csv_path,
        "markdown": md_path,
        "provenance": provenance_path,
    }


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-group-id", default=DEFAULT_RUN_GROUP_ID)
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Seed-clustered bootstrap resamples (0 disables the CIs).",
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args(argv)

    root = eu.project_root()
    benchmark = eu.read_csv_rows(
        eu.artifact_path(root / "data/processed/benchmark_items.csv", DATASET_ID)
    )
    registry_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for smoke in [False, True] if args.include_smoke else [False]:
        registry_path = eu.run_registry_path(root, DATASET_ID, VARIANT, smoke=smoke)
        if registry_path.exists():
            registry_rows.extend(eu.read_csv_rows(registry_path))
        raw_path = eu.model_outputs_raw_path(root, DATASET_ID, VARIANT, smoke=smoke)
        if raw_path.exists():
            raw_rows.extend(eu.read_jsonl(raw_path))
    tables = build_tables(
        benchmark,
        registry_rows,
        raw_rows,
        run_group_id=args.run_group_id,
        include_smoke=args.include_smoke,
        bootstrap_samples=args.bootstrap_samples,
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
