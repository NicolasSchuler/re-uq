"""Compare the batching arms of the request-composition ablation (TODO A).

Every archived run sent 16 benchmark items per request, in `seed x variant`
order, so each Task 2 request carried all four modality variants of the same
four seeds. The prompt says to evaluate each item independently, but the
minimal-pair contrast sat inside the context window: no reported number is free
of that confound until it is measured.

Three arms answer the same items with the same seeds and parameters and differ
only in how the requests were composed:

* `grouped` -- the paper condition, 16 items per request, consecutive seeds;
* `shuffled` -- 16 items per request, but never two source variants of one seed
  in the same request (a constrained shuffle derived from the run seed);
* `single` -- one item per request, the frozen single-item prompt.

This is the context ablation's sibling (`scripts/compare_context_ablation.py`)
and writes the same two tables: per arm, and `arm - grouped` deltas with a
paired cluster bootstrap. Two differences follow from what is being varied.
The pairing unit is the **item**, not the seed: the arms disagree about which
items share a request, so a seed is not a unit both arms measure the same way.
And the delta is resampled by **seed** rather than by request, because the
request is exactly what the ablation changes -- a `single` arm request holds
one item, a `grouped` one holds sixteen, and resampling those as if they were
the same unit would compare two different cluster sizes.

    .venv/bin/python scripts/compare_batching_ablation.py

Outputs `outputs/batching_ablation_summary.{csv,md}`, a `_deltas.csv`, and a
provenance JSON naming the run behind every row.
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
PAPER_BATCH_SIZE = 16

#: Arm ids, in reporting order. `grouped` is the baseline every delta is
#: measured against because it is the condition the archived runs used.
ARM_GROUPED = "grouped"
ARM_SHUFFLED = "shuffled"
ARM_SINGLE = "single"
ARMS = (ARM_GROUPED, ARM_SHUFFLED, ARM_SINGLE)
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
    "grouped",
    "arm_value",
    "delta",
    "delta_ci_low",
    "delta_ci_high",
    "delta_cluster_field",
    "n_delta_clusters",
    "n_grouped",
    "n_arm",
    "n_complete_pairs",
    "n_excluded_single_arm",
]


def registry_arm(row: Any) -> str:
    """Which arm a registry row belongs to, from its batching plan.

    A registry row written before `batch_order` existed is a grouped paper run;
    that is the same reading `select_cell_runs` applies to the archive.
    """
    try:
        batch_size = int(row.get("batch_size", 0) or 0)
    except (TypeError, ValueError):
        batch_size = 0
    if batch_size == 1:
        return ARM_SINGLE
    order = str(row.get("batch_order", "") or eu.DEFAULT_BATCH_ORDER)
    return ARM_SHUFFLED if order == eu.BATCH_ORDER_SHUFFLED else ARM_GROUPED


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
    current = eu.benchmark_rows_with_current_raw_outputs(benchmark, raw_rows)
    scores = eu.build_uq_scores(current, raw_rows, sampling_plan=sampling_plan)
    return [
        row
        for row in scores
        if str(row.get("task", "")) == "task2"
        and str(row.get("uq_method", "")) == "verbalized_confidence"
    ]


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
    ci = eu.text_over_commitment_ci_fields(
        rows, iterations=bootstrap_samples, seed=BOOTSTRAP_SEED
    )
    return {
        "model": model,
        "arm": arm,
        "stratum": stratum,
        "run_id": str(registry_row.get("run_id", "")),
        "batch_size": registry_row.get("batch_size", ""),
        "batch_order": registry_row.get("batch_order", ""),
        "n": len(rows),
        "n_text_readable": ci["text_over_commitment_n_denominator"],
        "label_accuracy": eu.task_accuracy(rows, "task2") if rows else "",
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
        "weak_strict_text_strengthening_90": _weak_strict_90(rows)
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
) -> list[dict[str, Any]]:
    """`arm - grouped` per metric, paired by item and resampled by seed."""
    rows: list[dict[str, Any]] = []
    for metric_name, metric in METRICS:
        paired = eu.bootstrap_seed_metric_delta(
            grouped,
            other,
            metric,
            # The request is what this ablation varies, so it cannot also be
            # the resampling unit; the seed is the coarsest unit both arms
            # measure identically. Pairing is by item because the arms do not
            # agree on which items share a request.
            cluster_field=eu.BOOTSTRAP_CLUSTER_FALLBACK_FIELD,
            pair_field="item_id",
            iterations=bootstrap_samples,
            seed=BOOTSTRAP_SEED,
        )
        rows.append(
            {
                "model": model,
                "arm": arm,
                "stratum": stratum,
                "metric": metric_name,
                "grouped": _finite(metric(grouped)) if grouped else "",
                "arm_value": _finite(metric(other)) if other else "",
                "delta": _finite(paired.delta),
                "delta_ci_low": _finite(paired.ci_low),
                "delta_ci_high": _finite(paired.ci_high),
                "delta_cluster_field": paired.cluster_field,
                "n_delta_clusters": paired.n_clusters,
                "n_grouped": len(grouped),
                "n_arm": len(other),
                "n_complete_pairs": paired.n_complete_pairs,
                "n_excluded_single_arm": paired.n_excluded_single_arm,
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
        for stratum in STRATA:
            per_arm = {
                arm: stratum_rows(scores_by_arm.get((model, arm), []), stratum)
                for arm in ARMS
            }
            for arm in ARMS:
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
            for arm in (ARM_SHUFFLED, ARM_SINGLE):
                if (model, arm) not in scores_by_arm:
                    continue
                delta_table.extend(
                    delta_rows(
                        model,
                        arm,
                        stratum,
                        per_arm[ARM_GROUPED],
                        per_arm[arm],
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
        "# Batching Ablation Summary",
        "",
        "Deterministic Task 2 rows of one cell under three request compositions:",
        "`grouped` (the paper condition, 16 items per request in seed order),",
        "`shuffled` (16 items, never two variants of one seed together), and",
        "`single` (one item per request). Strata: `all` and `weak_intent`.",
        "",
        "Per-arm CIs are the usual request-clustered bootstraps with the",
        "seed-clustered pair alongside. The **deltas** are resampled by seed,",
        "not by request: the request is what the arms vary, so it cannot also",
        "be the unit of independence. Pairs are items answered in both arms",
        "(`n_complete_pairs`); items answered in only one are excluded and",
        "counted (`n_excluded_single_arm`).",
        "",
        "If a delta's interval covers zero, the grouped numbers stand as",
        "reported. If it does not, the grouped numbers are a bound and the",
        "paper has to say in which direction.",
        "",
        "## Arms",
        "",
        eu.markdown_table(tables["arms"], ARM_FIELDS),
        "",
        "## Deltas (arm - grouped)",
        "",
        eu.markdown_table(tables["deltas"], DELTA_FIELDS),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    eu.write_json(
        provenance_path,
        {
            "generated_at_utc": eu.utc_now_iso(),
            "paper_batch_size": PAPER_BATCH_SIZE,
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
    tables = build_tables(
        benchmark,
        registry_rows,
        raw_rows,
        run_group_id=args.run_group_id,
        include_smoke=args.include_smoke,
        bootstrap_samples=args.bootstrap_samples,
        sampling_plan=eu.SamplingPlan(stochastic_samples=args.stochastic_samples),
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
