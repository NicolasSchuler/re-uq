"""Compare document context against bare input on exact, jointly eligible items.

Full-arm descriptions and matched-cohort estimates are exported separately.
Request connected components preserve dependence in both arms; capability
intervals are retained as sensitivity estimates. See `docs/context_ablation.md`.
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


DATASET_ID = eu.DATASET_PURE
VARIANT = "must"
DEFAULT_RUN_GROUP_ID = "context-ablation-2026-09"
DEFAULT_OUTPUT_PREFIX = Path("outputs/context_ablation_summary")
DEFAULT_BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260518
# The ablation runs draw the paper's five stochastic samples per item. Only the
# deterministic Task 2 rows are compared here, but scoring still declares the
# plan those runs were executed under rather than inferring it.
DEFAULT_STOCHASTIC_SAMPLES = 5
MODALITIES = tuple(eu.MODALITIES)
STRATA = (
    "all",
    "weak_intent",
    "marker_M",
    "marker_O",
    *(f"modality_{m}" for m in MODALITIES),
    *(f"marker_{marker}/modality_{m}" for marker in ("M", "O") for m in MODALITIES),
)
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
    "stratum",
    "metric",
    "bare",
    "document",
    "delta",
    "delta_ci_low",
    "delta_ci_high",
    "delta_seed_ci_low",
    "delta_seed_ci_high",
    "delta_cluster_field",
    "n_delta_clusters",
    "n_bare",
    "n_document",
    "n_complete_pairs",
    "n_excluded_single_arm",
]


ARM_FIELDS += ["estimate_cohort", "n_failed_items", "n_unclassified_items"]
DELTA_FIELDS += [
    "full_arm_bare_descriptive",
    "full_arm_document_descriptive",
    "cluster_note",
    "unavailable_reason",
    "delta_ci_unavailable_reason",
    "delta_seed_ci_unavailable_reason",
    "n_matched_items",
    "n_matched_capabilities",
    "n_unmatched_items",
    "n_duplicate_identities",
    "n_excluded_ineligible_items",
    "n_missing_identity_rows",
] + [
    f"n_{arm}_{kind}"
    for arm in ("bare", "document")
    for kind in (
        "duplicate_rows",
        "failed_items",
        "unclassified_items",
        "eligible_items",
    )
]


def select_arm_runs(
    registry_rows: list[dict[str, Any]],
    *,
    run_group_id: str,
    include_smoke: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Return the latest complete, fully covered run per (model, item_context).

    Args:
        registry_rows: Run-registry rows to choose from; rows of another
            dataset or benchmark variant than the `pure` `must` cell are
            skipped.
        run_group_id: Only rows of this run group are considered.
        include_smoke: Whether runs whose ID starts with `smoke-` may be
            selected.

    Returns:
        The selected registry row keyed by `(model, item_context)`, where a
        blank `item_context` reads as bare. A run qualifies when its status is
        `complete` and its `deterministic_item_coverage` is at least 1; among
        qualifying runs the latest `started_at_utc` wins, with the run ID as
        tie-breaker.
    """
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in registry_rows:
        if (
            row.get("benchmark_variant", VARIANT) != VARIANT
            or row.get("dataset_id", DATASET_ID) != DATASET_ID
        ):
            continue
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
    benchmark: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    *,
    sampling_plan: eu.SamplingPlan,
) -> list[dict[str, Any]]:
    """Return deterministic Task 2 score rows with the item's author marker joined on.

    Each raw row is resolved to its current benchmark item by the recorded
    source statement and modality (narrowed by requirement ID, original
    requirement and corpus when recorded), falling back to the item ID for
    records without source fields.

    Args:
        benchmark: Current benchmark items of the `pure` cell.
        raw_rows: Raw model outputs of one run; only deterministic Task 2 rows
            are scored, after deduplicating retries.
        sampling_plan: Sampling plan passed to `eu.build_uq_scores`.

    Returns:
        One observation per deterministic raw row, carrying its score fields
        when the row could be scored, a `source_identity` of the form
        `corpus::requirement`, the gold modality, the `context_marker`, and a
        `response_status` that is `ok` for scored rows and the parse status (or
        `missing_score`) otherwise.

    Raises:
        ValueError: If a raw row's source matches no item or several items, its
            item ID is unknown, or its source differs from the current benchmark
            item.
    """
    by_id = {str(r["item_id"]): r for r in benchmark}
    by_source = {}
    for item in benchmark:
        by_source.setdefault(
            (item["source_statement"], item["source_modality"]), []
        ).append(item)
    # Resolve every raw row to its current benchmark item first, then score
    # the arm in one pass: build_uq_scores resolves the sampling plan and
    # dedupes per call, which is far too much to repeat per row.
    resolved: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for raw in eu.dedupe_raw_rows(raw_rows):
        if raw.get("task") != "task2" or raw.get("sample_kind") != "deterministic":
            continue
        # Prefer the recorded source even when an arm-local ID happens to name
        # a different current item (for example after arm-local renumbering).
        if raw.get("source_statement"):
            candidates = by_source.get(
                (raw["source_statement"], raw.get("source_modality")), []
            )
            for field in (
                "context_requirement_id",
                "original_requirement",
                "source_corpus",
            ):
                if raw.get(field):
                    candidates = [
                        item for item in candidates if item.get(field) == raw[field]
                    ]
            if len(candidates) != 1:
                raise ValueError(
                    "Unmatched or ambiguous raw source identity: "
                    + str(raw.get("item_id"))
                )
            item = candidates[0]
        else:
            # Legacy fixtures/records without durable source fields still pass
            # the current benchmark prompt check below.
            item = by_id.get(str(raw.get("item_id", "")))
            if item is None:
                raise ValueError(
                    "Missing raw source identity: " + str(raw.get("item_id"))
                )
        canonical = {**raw, "item_id": item["item_id"], "seed_id": item["seed_id"]}
        if (
            raw.get("source_statement")
            and raw["source_statement"] != item["source_statement"]
        ) or not eu.raw_record_matches_benchmark_item(canonical, item):
            raise ValueError(
                "Raw source differs from current benchmark: " + str(raw.get("item_id"))
            )
        resolved.append((raw, item, canonical))
    # Score only the deterministic observations: no stochastic embedding work.
    # Rows are keyed by the current item, and one raw row per item per arm is
    # what dedupe_raw_rows leaves, so the score joins back one-to-one.
    scored = eu.build_uq_scores(
        [item for _, item, _ in resolved],
        [canonical for _, _, canonical in resolved],
        sampling_plan=sampling_plan,
    )
    score_by_item = {str(row["item_id"]): row for row in scored}
    observations = []
    for raw, item, _canonical in resolved:
        score = score_by_item.get(str(item["item_id"]))
        identity = (
            str(
                item.get("source_corpus")
                or item.get("context_document")
                or item.get("source_dataset", "")
            )
            + "::"
            + str(item.get("context_requirement_id") or item["seed_id"])
        )
        observations.append(
            {
                **(score or {}),
                "item_id": raw["item_id"],
                "source_identity": identity,
                "seed_id": identity,
                "source_statement": item["source_statement"],
                "model": raw.get("model", ""),
                "run_id": raw.get("run_id", ""),
                "dataset_id": DATASET_ID,
                "benchmark_variant": raw.get("benchmark_variant", VARIANT),
                "gold_modality": item["source_modality"],
                "context_marker": item.get("context_marker", ""),
                "response_status": "ok"
                if score
                else (
                    raw.get("parse_status")
                    if raw.get("parse_status") != "ok"
                    else "missing_score"
                ),
            }
        )
    return observations


def stratum_rows(rows: list[dict[str, Any]], stratum: str) -> list[dict[str, Any]]:
    """Restrict observation rows to one analysis stratum.

    Args:
        rows: Observation rows of one arm.
        stratum: `all` keeps every row, `weak_intent` the rows with gold
            modality `nice_to_have`, `marker_<marker>` the rows with that author
            marker, and `modality_<modality>` the rows with that gold modality.
            Two strata joined by `/` are applied in turn.

    Returns:
        The rows in the stratum (the input list itself for `all`).

    Raises:
        ValueError: If `stratum` is not one of these forms.
    """
    if "/" in stratum:
        first, second = stratum.split("/", 1)
        return stratum_rows(stratum_rows(rows, first), second)
    if stratum.startswith("modality_"):
        return [
            r
            for r in rows
            if r.get("gold_modality") == stratum.removeprefix("modality_")
        ]
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
    """Build the full-arm descriptive row of one context arm in one stratum.

    Args:
        model: Model name written to the row.
        arm: Item-context arm (`bare` or `document`) written to the row.
        stratum: Stratum name; the weak-intent strict strengthening rate at
            confidence 0.90 is only filled for `weak_intent`.
        run_id: Run ID of the arm, written to the row.
        rows: Observation rows of the arm in this stratum, failed ones included.
        bootstrap_samples: Cluster-bootstrap resamples for the intervals.

    Returns:
        A row with the `ARM_FIELDS` columns, computed over the rows that are
        eligible for label accuracy. An interval is left blank when fewer than
        two clusters (or, for the seed interval, two capabilities) are readable.
    """
    observed = rows
    rows = [r for r in rows if metric_eligible(r, "label_accuracy")]
    ci = eu.text_over_commitment_ci_fields(
        rows, iterations=bootstrap_samples, seed=BOOTSTRAP_SEED
    )
    readable = [r for r in rows if r.get("text_modality_parse_status") == "ok"]
    resolved = ci.get("bootstrap_ci_cluster_field", "seed_id")
    n_clusters = len({r.get(resolved) for r in readable})
    n_capabilities = len({r.get("seed_id") for r in readable})
    for key in ci:
        if key.endswith(("_ci_low", "_ci_high")) and (
            ("_seed_ci_" in key and n_capabilities < 2)
            or ("_seed_ci_" not in key and n_clusters < 2)
        ):
            ci[key] = ""
    return {
        "model": model,
        "item_context": arm,
        "stratum": stratum,
        "run_id": run_id,
        "estimate_cohort": "full_arm_descriptive",
        "n": len(observed),
        "n_failed_items": len(observed) - len(rows),
        "n_unclassified_items": sum(
            r.get("text_modality_parse_status") != "ok" for r in rows
        ),
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


def _finite(value: float) -> float | str:
    return "" if isinstance(value, float) and math.isnan(value) else value


def metric_eligible(row, metric):
    """Return whether an observation row can be scored for a metric.

    Args:
        row: Observation row from `task2_scores`.
        metric: Metric name from `METRICS`; every metric except
            `label_accuracy` also needs a readable text modality.

    Returns:
        True when the response succeeded, the predicted modality is one of
        `MODALITIES` and, for the text strengthening metrics, the text modality
        parsed.
    """
    valid = (
        row.get("response_status", "ok") == "ok"
        and row.get("pred_modality") in MODALITIES
    )
    return valid and (
        metric == "label_accuracy" or row.get("text_modality_parse_status") == "ok"
    )


def delta_rows(
    model: str,
    stratum: str,
    bare: list[dict[str, Any]],
    document: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    """Compute `document - bare` per metric on the jointly eligible exact items.

    Args:
        model: Model name written to each row.
        stratum: Stratum name written to each row.
        bare: Observation rows of the bare arm in this stratum.
        document: Observation rows of the document arm in this stratum.
        bootstrap_samples: Bootstrap resamples for the delta intervals; zero or
            fewer marks them as disabled.

    Returns:
        One row per metric in `METRICS` with the `DELTA_FIELDS` columns: both
        arms on the matched cohort and in full, the delta with an interval
        clustered on connected request components and a seed-clustered
        sensitivity interval, and the pairing and exclusion counts.
    """
    rows: list[dict[str, Any]] = []
    for metric_name, metric in METRICS:

        def eligible(r, name=metric_name):
            return metric_eligible(r, name)

        matched_bare, matched_document, counts = eu.exact_item_metric_pairs(
            bare, document, eligible
        )
        matched_bare, matched_document, cluster, cluster_note = (
            eu.paired_request_clusters(matched_bare, matched_document)
        )
        paired = eu.bootstrap_seed_metric_delta(
            matched_bare,
            matched_document,
            metric,
            pair_field="exact_pair_id",
            cluster_field=cluster,
            iterations=bootstrap_samples,
            seed=BOOTSTRAP_SEED,
        )
        by_seed = eu.bootstrap_seed_metric_delta(
            matched_bare,
            matched_document,
            metric,
            pair_field="exact_pair_id",
            cluster_field="seed_id",
            iterations=bootstrap_samples,
            seed=BOOTSTRAP_SEED,
        )
        rows.append(
            {
                "model": model,
                "stratum": stratum,
                "metric": metric_name,
                "bare": _finite(metric(matched_bare)) if matched_bare else "",
                "full_arm_bare_descriptive": _finite(
                    metric([r for r in bare if eligible(r)])
                ),
                "document": _finite(metric(matched_document))
                if matched_document
                else "",
                "full_arm_document_descriptive": _finite(
                    metric([r for r in document if eligible(r)])
                ),
                "cluster_note": cluster_note,
                "unavailable_reason": ""
                if matched_bare
                else "no jointly eligible exact items",
                "delta_ci_unavailable_reason": "fewer than two clusters"
                if paired.n_clusters < 2
                else "bootstrap disabled"
                if bootstrap_samples <= 0
                else "",
                "delta_seed_ci_unavailable_reason": "fewer than two capabilities"
                if by_seed.n_clusters < 2
                else "bootstrap disabled"
                if bootstrap_samples <= 0
                else "",
                **counts,
                "delta": _finite(paired.delta),
                "delta_ci_low": _finite(paired.ci_low)
                if paired.n_clusters >= 2
                else "",
                "delta_ci_high": _finite(paired.ci_high)
                if paired.n_clusters >= 2
                else "",
                "delta_seed_ci_low": _finite(by_seed.ci_low)
                if by_seed.n_clusters >= 2
                else "",
                "delta_seed_ci_high": _finite(by_seed.ci_high)
                if by_seed.n_clusters >= 2
                else "",
                "delta_cluster_field": paired.cluster_field,
                "n_delta_clusters": paired.n_clusters,
                "n_bare": len(bare),
                "n_document": len(document),
                "n_complete_pairs": counts["n_matched_items"],
                "n_excluded_single_arm": counts["n_unmatched_items"],
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
    """Build the arm, delta and provenance tables of the context ablation.

    Args:
        benchmark: Current benchmark items of the `pure` cell.
        registry_rows: Run-registry rows; `select_arm_runs` picks one run per
            model and item context from them.
        raw_rows: Raw model outputs of all candidate runs.
        run_group_id: Run group whose runs are compared.
        include_smoke: Whether smoke runs may be selected.
        bootstrap_samples: Bootstrap resamples for the per-arm and delta
            intervals.
        sampling_plan: Sampling plan the compared runs used.

    Returns:
        A dict with `arms` (rows of `ARM_FIELDS`), `deltas` (rows of
        `DELTA_FIELDS`, one set per model and stratum) and `provenance` (one
        entry per selected run).
    """
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
    """Write the context ablation tables as CSV, Markdown and provenance JSON.

    Args:
        tables: Output of `build_tables`.
        output_prefix: Path prefix; the arm table goes to `<prefix>.csv`, the
            deltas to `<prefix>_deltas.csv`, the report to `<prefix>.md` and the
            run provenance to `<prefix>_provenance.json`.

    Returns:
        The written paths under the keys `csv`, `deltas_csv`, `markdown` and
        `provenance`.
    """
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
        "Full-arm estimates are descriptive; deltas and both arm values use the same",
        "jointly eligible exact source-item cohort. Counts are items except explicitly",
        "named capability, row, or cluster counts. Legacy n_complete_pairs counts items.",
        "Request intervals use connected components of both arms' request partitions;",
        "missing IDs fall back to capabilities. Seed intervals are sensitivity estimates.",
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
    eu.write_json(
        provenance_path,
        {
            "generated_at_utc": eu.utc_now_iso(),
            "dataset_id": DATASET_ID,
            "benchmark_variant": VARIANT,
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
    """Compare the bare and document arms of the `pure` cell from the command line.

    Args:
        argv: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        The written output paths, as returned by `write_outputs`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-group-id", default=DEFAULT_RUN_GROUP_ID)
    parser.add_argument("--include-smoke", action="store_true")
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
        default=DEFAULT_STOCHASTIC_SAMPLES,
        help="Stochastic samples per item the compared runs planned.",
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
        if eu.raw_store_exists(raw_path):
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
