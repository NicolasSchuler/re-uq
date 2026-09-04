"""Regenerate the paper-facing per-cell snapshots and per-model disaggregations.

The per-cell snapshot CSVs that ``scripts/aggregate_paper_headline_metrics.py``
reads (``paper_task2_text_drift_metrics.csv`` and
``paper_text_drift_confidence_and_stability.csv``) were shipped as static files
whose generating code was never committed. This module regenerates them from the
local raw model outputs plus the benchmark CSVs, and additionally exports two
disaggregated tables that the pooled headline hides:

* ``paper_per_model_modality_table.csv`` -- model x dataset x variant x source
  modality, with strict/broad strengthening counts, rates, seed-clustered
  bootstrap CIs, the p>=0.90 high-confidence share, the agreement denominator,
  and mean generated-requirement length.
* ``paper_per_model_headline.csv`` -- the same columns per model, pooled over
  the requested cells.

Scoring unit and aggregation rules are documented in ``docs/aggregation.md``.

Run selection: for each (dataset, variant) cell the run registry CSV is read and
only runs compatible with the pinned paper group, task set, coverage, sampling,
and batching plan are kept, dropping smoke runs by default and models matching
``--exclude-model-prefix`` (``azure.`` by default). When a model has several
compatible runs in a cell the most recent one (by ``started_at_utc``, then
``run_id``) wins. A chosen run that contributed no raw rows is an error
(``--allow-missing-raw`` downgrades it to a warning and drops the model), so a
stale registry row can never pass an empty cell off as a measured one. The
chosen run ids are recorded in
``outputs/paper_snapshot_provenance.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover - invocation-path fallback
    from scripts import eval_utils as eu


DEFAULT_CELLS = [
    ("mlm_tapt", "must"),
    ("mlm_tapt", "shall"),
    ("nice", "must"),
    ("nice", "shall"),
]
DEFAULT_MODELS = [
    "glm-4.5-air",
    "glm-4.7",
    "glm-5",
    "glm-5-turbo",
    "glm-5.1",
    "kit.gemma4-31b-it",
]
DEFAULT_EXCLUDE_MODEL_PREFIXES = ["azure."]
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_PAPER_RUN_GROUP_ID = "provider-matrix-2026-05"
DEFAULT_PAPER_BATCH_ORDER = "grouped"
DEFAULT_PAPER_BATCH_SIZE = 16
# Every official run drew 5 stochastic samples per item; a group with fewer is
# an incomplete measurement, not a smaller one.
DEFAULT_EXPECTED_STOCHASTIC_SAMPLES = 5
BOOTSTRAP_SEED = 20260518
HIGH_CONFIDENCE_THRESHOLD = 0.90

TASK2_SNAPSHOT_NAME = "paper_task2_text_drift_metrics.csv"
CONFIDENCE_SNAPSHOT_NAME = "paper_text_drift_confidence_and_stability.csv"
REGENERATED_SUFFIX = "_regenerated"
PER_MODEL_MODALITY_NAME = "paper_per_model_modality_table.csv"
PER_MODEL_HEADLINE_NAME = "paper_per_model_headline.csv"
HEADLINE_CI_NAME = "paper_headline_bootstrap_ci.csv"
PROVENANCE_NAME = "paper_snapshot_provenance.json"


TASK2_SNAPSHOT_FIELDS = [
    "dataset",
    "variant",
    "models",
    "n",
    "label_accuracy",
    "macro_f1",
    "label_over_commitment",
    "label_under_commitment",
    "label_high_conf_overcommit_90_all",
    "label_high_conf_overcommit_90_overcommittable",
    "weak_recall",
    "weak_strengthening_90_label",
    "brier",
    "ece",
    "text_modality_parse_coverage",
    "heuristic_text_modality_rate",
    "label_text_consistency",
    "text_over_commitment",
    "strict_text_over_commitment",
    "text_under_commitment",
    "text_high_conf_overcommit_80",
    "text_high_conf_overcommit_90",
    "label_correct_text_overcommit_80",
    "label_correct_text_overcommit_90",
    "parse_failure_rate",
    # Weak stakeholder-intent headline (README 29.8%): the strict, high-confidence
    # strengthening rate over `nice_to_have` sources, on both denominators.
    "weak_n",
    "weak_n_readable",
    "weak_strict_text_strengthening_90",
    "weak_strict_text_strengthening_90_all_weak",
    # Coverage-adjusted accounting for the two headline text-drift rates.
    "text_over_commitment_n_numerator",
    "text_over_commitment_n_denominator",
    "text_over_commitment_n_unknown_excluded",
    "text_over_commitment_lower_bound",
    "text_over_commitment_upper_bound",
    "strict_text_over_commitment_n_numerator",
    "strict_text_over_commitment_n_denominator",
    "strict_text_over_commitment_n_unknown_excluded",
    "strict_text_over_commitment_lower_bound",
    "strict_text_over_commitment_upper_bound",
    "text_modality_negated_rate",
    "text_modality_multi_modal_rate",
    # Answer-length / bloat diagnostics.
    "mean_requirement_word_count",
    "mean_source_word_count",
    "mean_length_ratio",
    "mean_completion_tokens",
    "length_tercile_bounds",
    "strengthening_rate_by_length_tercile",
    "mean_requirement_word_count_by_source_modality",
]

CONFIDENCE_SNAPSHOT_FIELDS = [
    "dataset",
    "variant",
    "n",
    "broad_text_oc_n",
    "broad_text_oc_conf_ge_90",
    "broad_text_oc_mean_confidence",
    "strict_text_oc_n",
    "strict_text_oc_conf_ge_90",
    "strict_text_oc_mean_confidence",
    "strict_text_oc_mean_modality_consistency_uncertainty",
    "strict_text_oc_unanimous_modality_samples",
    "strict_text_oc_mean_predictive_entropy",
    "strict_text_oc_mean_variation_ratio",
    # Repeated-sample agreement is computed only over items where every
    # stochastic sample parsed; the excluded count is reported alongside.
    "agreement_n_complete",
    "agreement_n_incomplete_excluded",
]

PER_MODEL_FIELDS = [
    "model",
    "dataset",
    "variant",
    "source_modality",
    "n_items",
    "n_valid",
    "n_parse_failures",
    "broad_strengthening_n",
    "broad_strengthening_denominator",
    "broad_strengthening_rate",
    "broad_strengthening_ci_low",
    "broad_strengthening_ci_high",
    "strict_strengthening_n",
    "strict_strengthening_denominator",
    "strict_strengthening_rate",
    "strict_strengthening_ci_low",
    "strict_strengthening_ci_high",
    "strict_high_conf_share_90",
    "agreement_n_complete",
    "agreement_n_incomplete_excluded",
    "mean_requirement_word_count",
]


# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------


def select_cell_runs(
    registry_rows: list[dict[str, Any]],
    models: Iterable[str],
    exclude_model_prefixes: Iterable[str] = DEFAULT_EXCLUDE_MODEL_PREFIXES,
    include_smoke: bool = False,
    *,
    run_group_id: str | None = None,
    benchmark_item_count: int = 0,
    expected_stochastic_samples: int = DEFAULT_EXPECTED_STOCHASTIC_SAMPLES,
    expected_batch_order: str | None = None,
    expected_batch_size: int | None = None,
    run_prefixes: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Pick the latest compatible registry run for each cohort model."""
    cohort = list(models)
    excluded = list(exclude_model_prefixes)
    chosen: dict[str, dict[str, Any]] = {}
    for row in registry_rows:
        model = str(row.get("model", ""))
        run_id = str(row.get("run_id", ""))
        if str(row.get("status", "")) != "complete":
            continue
        if not include_smoke and run_id.startswith("smoke-"):
            continue
        if any(model.startswith(prefix) for prefix in excluded):
            continue
        if cohort and model not in cohort:
            continue
        if run_prefixes is not None and not eu.run_id_matches_prefix(
            run_id, run_prefixes
        ):
            continue
        if run_group_id is not None:
            issues = eu.registry_row_compatibility_issues(
                row,
                run_group_id=run_group_id,
                benchmark_item_count=benchmark_item_count,
                expected_stochastic_samples=expected_stochastic_samples,
                required_tasks=("task1", "task2"),
                exact_tasks=("task1", "task2"),
                expected_batch_order=expected_batch_order,
                expected_batch_size=expected_batch_size,
                allow_partial_benchmark=include_smoke and eu.is_smoke_run_id(run_id),
                # Historical paper rows predate this registry column. The pinned
                # run group, exact task/coverage plan, and batch size still keep
                # them distinct from newer batching ablations.
                allow_missing_batch_order=True,
            )
            if issues:
                continue
        current = chosen.get(model)
        sort_key = (str(row.get("started_at_utc", "")), run_id)
        if current is None or sort_key > (
            str(current.get("started_at_utc", "")),
            str(current.get("run_id", "")),
        ):
            chosen[model] = dict(row)
    return chosen


def stream_raw_rows(path: Path, run_ids: set[str]) -> Iterator[dict[str, Any]]:
    """Yield raw records whose ``run_id`` is in ``run_ids``.

    The raw files hold 60-70k records each, so the filter is applied while
    streaming instead of materializing the whole file.
    """
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            # Cheap substring prefilter before paying for json.loads.
            if not any(run_id in line for run_id in run_ids):
                continue
            record = json.loads(line)
            if str(record.get("run_id", "")) in run_ids:
                yield record


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | str:
    usable = [value for value in values if value is not None and not math.isnan(value)]
    return sum(usable) / len(usable) if usable else ""


def _numeric(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def task2_deterministic_rows(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in scores
        if str(row.get("task", "")) == "task2"
        and str(row.get("uq_method", "")) == "verbalized_confidence"
    ]


def task2_deterministic_raw_rows(
    raw_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The raw records the deterministic Task 2 score rows are built from."""
    return [
        row
        for row in raw_rows
        if str(row.get("task", "")) == "task2"
        and str(row.get("sample_kind", "")) == "deterministic"
    ]


def task2_parse_failure_rate(raw_rows: Iterable[dict[str, Any]]) -> float:
    """Share of deterministic Task 2 responses that did not parse.

    A response that fails to parse never produces a score row, so this rate is
    only visible in the raw records; deriving it from the score rows would
    divide failures by the successes that survived them.
    """
    rows = task2_deterministic_raw_rows(raw_rows)
    if not rows:
        return 0.0
    return sum(1 for row in rows if str(row.get("parse_status", "")) != "ok") / len(
        rows
    )


#: Key every paper-facing join between a deterministic observation and its
#: repeated-sample row uses.
PaperJoinKey = tuple[str, str, str, str]


def stamp_cell_identity(
    scores: list[dict[str, Any]], dataset: str, variant: str
) -> list[dict[str, Any]]:
    """Record which cell a score row came from, in place.

    ``build_uq_scores`` scores one (dataset, variant) cell at a time and its
    rows carry no cell identity, because the cell is implied by the raw file
    they were read from. The exporter pools four cells, so it stamps the
    identity here (using the same spelling as the run registry) before anything
    is pooled or joined; see :func:`paper_join_key`.
    """
    dataset_id = eu.normalize_dataset_id(dataset)
    benchmark_variant = eu.normalize_benchmark_variant(variant)
    for row in scores:
        row["dataset_id"] = dataset_id
        row["benchmark_variant"] = benchmark_variant
    return scores


def paper_join_key(row: Mapping[str, Any]) -> PaperJoinKey:
    """``(model, dataset, variant, item)`` -- the identity a paper join needs.

    Benchmark item ids are reused between the ``must`` and ``shall`` variants of
    a family (all 720 ids overlap), so a ``(model, item)`` key silently collapses
    the four-cell matrix onto one cell: a pooled agreement number kept one row
    where four were measured, and a deterministic row could be joined to another
    cell's repeated samples. Provider and profile are deliberately NOT part of
    this key -- one paper cell selects exactly one run per model, and pooling
    across providers is what the per-model tables are for.
    """
    identity = eu.ObservationIdentity.from_raw_row(row)
    return (identity.model, identity.dataset_id, identity.variant, identity.item_id)


def stochastic_rows_by_method(
    scores: list[dict[str, Any]], uq_method: str
) -> dict[PaperJoinKey, dict[str, Any]]:
    return {
        paper_join_key(row): row
        for row in scores
        if str(row.get("task", "")) == "task2"
        and str(row.get("uq_method", "")) == uq_method
    }


def agreement_for_strict_rows(
    strict_rows: Iterable[dict[str, Any]],
    consistency: dict[PaperJoinKey, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Agreement metrics over every strict deterministic item.

    ``build_uq_scores`` cannot emit a distribution for a group with zero valid
    stochastic samples. Such an absent row is still an incomplete measurement
    and must be counted in the excluded population.
    """
    strict_keys = {paper_join_key(row) for row in strict_rows}
    present = [consistency[key] for key in strict_keys if key in consistency]
    metrics = eu.repeated_sample_agreement_metrics(present)
    metrics["agreement_n_incomplete_excluded"] += len(strict_keys) - len(present)
    complete = [
        row for row in present if eu.is_truthy_strict(row.get("stochastic_complete"))
    ]
    return metrics, complete


def weak_intent_snapshot_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Weak stakeholder-intent strengthening, the README's 29.8% headline.

    Weak rows are the deterministic Task 2 rows whose *source* modality is
    ``nice_to_have``. The published rate divides the strict, p>=0.90
    strengthening count by the weak rows whose generated text yielded a
    modality at all (``weak_n_readable``); ``_all_weak`` charges the unreadable
    rows to the denominator instead, bounding the rate from below.
    """
    weak_rows = [
        row for row in rows if str(row.get("gold_modality", "")) == "nice_to_have"
    ]
    readable = [
        row
        for row in weak_rows
        if str(row.get("text_modality_parse_status", "")) == "ok"
    ]
    # Strict strengthening already requires a parsed text modality, so the
    # numerator is the same under both denominators.
    strengthened = sum(
        1
        for row in weak_rows
        if eu.is_truthy_strict(row.get("strict_text_high_conf_overcommit_90"))
    )
    return {
        "weak_n": len(weak_rows),
        "weak_n_readable": len(readable),
        "weak_strict_text_strengthening_90": strengthened / len(readable)
        if readable
        else "",
        "weak_strict_text_strengthening_90_all_weak": strengthened / len(weak_rows)
        if weak_rows
        else "",
    }


def task2_snapshot_row(
    dataset: str,
    variant: str,
    models: int,
    rows: list[dict[str, Any]],
    raw_rows: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Rebuild one row of ``paper_task2_text_drift_metrics.csv``."""
    y_true = [int(row["y_true"]) for row in rows]
    calibration = eu.calibration_probabilities(rows, "task2")
    gold = [str(row["gold_modality"]) for row in rows]
    pred = [str(row["pred_modality"]) for row in rows]
    over_metrics = eu.overcommitment_summary_metrics(rows)
    text_metrics = eu.text_modality_summary_metrics(rows)
    bloat_metrics = eu.length_bloat_metrics(rows)
    row = {
        "dataset": dataset,
        "variant": variant,
        "models": models,
        "n": len(rows),
        "label_accuracy": eu.task_accuracy(rows, "task2"),
        "macro_f1": eu.macro_f1_score(gold, pred, eu.MODALITIES),
        "label_over_commitment": over_metrics["over_commitment"],
        "label_under_commitment": over_metrics["under_commitment"],
        "label_high_conf_overcommit_90_all": eu.task2_high_confidence_overcommitment_rate(
            rows, 0.90, "all"
        ),
        "label_high_conf_overcommit_90_overcommittable": eu.task2_high_confidence_overcommitment_rate(
            rows, 0.90, "overcommittable"
        ),
        "weak_recall": eu.weak_modality_recall(rows),
        "weak_strengthening_90_label": eu.weak_strengthening_rate(rows, 0.90),
        "brier": eu.brier_score(y_true, calibration),
        "ece": eu.ece_score(y_true, calibration),
        "parse_failure_rate": task2_parse_failure_rate(raw_rows),
        **weak_intent_snapshot_fields(rows),
    }
    row.update(
        {
            key: value
            for key, value in text_metrics.items()
            if key in TASK2_SNAPSHOT_FIELDS
        }
    )
    row.update(
        {
            key: value
            for key, value in bloat_metrics.items()
            if key in TASK2_SNAPSHOT_FIELDS
        }
    )
    return {field: row.get(field, "") for field in TASK2_SNAPSHOT_FIELDS}


def confidence_snapshot_row(
    dataset: str,
    variant: str,
    det_rows: list[dict[str, Any]],
    scores: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rebuild one row of ``paper_text_drift_confidence_and_stability.csv``.

    ``n`` is the coverage-adjusted denominator: deterministic Task 2 rows whose
    generated text yielded a modality. Stability columns join the deterministic
    row to the repeated-sample score rows for the same (model, item).
    """
    text_rows = [
        row
        for row in det_rows
        if str(row.get("text_modality_parse_status", "")) == "ok"
    ]
    broad_rows = [
        row for row in text_rows if eu.is_truthy_strict(row.get("text_overcommit"))
    ]
    strict_rows = [
        row
        for row in text_rows
        if eu.is_truthy_strict(row.get("strict_text_overcommit"))
    ]

    consistency = stochastic_rows_by_method(scores, "modality_consistency")
    entropy = stochastic_rows_by_method(scores, "predictive_entropy")
    variation = stochastic_rows_by_method(scores, "variation_ratio")

    agreement, complete_consistency = agreement_for_strict_rows(
        strict_rows, consistency
    )
    complete_keys = {paper_join_key(row) for row in complete_consistency}

    return {
        "dataset": dataset,
        "variant": variant,
        "n": len(text_rows),
        "broad_text_oc_n": len(broad_rows),
        "broad_text_oc_conf_ge_90": _share_at_least(
            broad_rows, HIGH_CONFIDENCE_THRESHOLD
        ),
        "broad_text_oc_mean_confidence": _mean(
            [float(row["confidence"]) for row in broad_rows]
        ),
        "strict_text_oc_n": len(strict_rows),
        "strict_text_oc_conf_ge_90": _share_at_least(
            strict_rows, HIGH_CONFIDENCE_THRESHOLD
        ),
        "strict_text_oc_mean_confidence": _mean(
            [float(row["confidence"]) for row in strict_rows]
        ),
        "strict_text_oc_mean_modality_consistency_uncertainty": _mean(
            [float(row["uncertainty_score"]) for row in complete_consistency]
        ),
        "strict_text_oc_unanimous_modality_samples": agreement[
            "repeated_sample_unanimity"
        ],
        "strict_text_oc_mean_predictive_entropy": _mean(
            [
                float(entropy[key]["uncertainty_score"])
                for key in complete_keys
                if key in entropy
            ]
        ),
        "strict_text_oc_mean_variation_ratio": _mean(
            [
                float(variation[key]["uncertainty_score"])
                for key in complete_keys
                if key in variation
            ]
        ),
        "agreement_n_complete": agreement["agreement_n_complete"],
        "agreement_n_incomplete_excluded": agreement["agreement_n_incomplete_excluded"],
    }


def _share_at_least(rows: list[dict[str, Any]], threshold: float) -> float | str:
    if not rows:
        return ""
    return sum(
        1 for row in rows if float(row.get("confidence", 0.0)) >= threshold
    ) / len(rows)


def per_model_row(
    model: str,
    dataset: str,
    variant: str,
    source_modality: str,
    det_rows: list[dict[str, Any]],
    consistency: dict[PaperJoinKey, dict[str, Any]],
    n_items: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """One disaggregated row: counts, rates, seed-clustered CIs, and length."""
    text_rows = [
        row
        for row in det_rows
        if str(row.get("text_modality_parse_status", "")) == "ok"
    ]
    strict_rows = [
        row
        for row in text_rows
        if eu.is_truthy_strict(row.get("strict_text_overcommit"))
    ]
    broad_rows = [
        row for row in text_rows if eu.is_truthy_strict(row.get("text_overcommit"))
    ]
    ci_fields = eu.text_over_commitment_ci_fields(
        det_rows,
        iterations=bootstrap_samples,
        seed=BOOTSTRAP_SEED,
    )
    agreement, _ = agreement_for_strict_rows(strict_rows, consistency)
    word_counts = [_numeric(row.get("requirement_word_count")) for row in det_rows]
    return {
        "model": model,
        "dataset": dataset,
        "variant": variant,
        "source_modality": source_modality,
        "n_items": n_items,
        "n_valid": len(det_rows),
        "n_parse_failures": max(0, n_items - len(det_rows)),
        "broad_strengthening_n": len(broad_rows),
        "broad_strengthening_denominator": len(text_rows),
        "broad_strengthening_rate": ci_fields.get("text_over_commitment", ""),
        "broad_strengthening_ci_low": ci_fields.get("text_over_commitment_ci_low", ""),
        "broad_strengthening_ci_high": ci_fields.get(
            "text_over_commitment_ci_high", ""
        ),
        "strict_strengthening_n": len(strict_rows),
        "strict_strengthening_denominator": len(text_rows),
        "strict_strengthening_rate": ci_fields.get("strict_text_over_commitment", ""),
        "strict_strengthening_ci_low": ci_fields.get(
            "strict_text_over_commitment_ci_low", ""
        ),
        "strict_strengthening_ci_high": ci_fields.get(
            "strict_text_over_commitment_ci_high", ""
        ),
        "strict_high_conf_share_90": _share_at_least(
            strict_rows, HIGH_CONFIDENCE_THRESHOLD
        ),
        "agreement_n_complete": agreement["agreement_n_complete"],
        "agreement_n_incomplete_excluded": agreement["agreement_n_incomplete_excluded"],
        "mean_requirement_word_count": _mean(
            [value for value in word_counts if value is not None]
        ),
    }


# ---------------------------------------------------------------------------
# Cell scoring
# ---------------------------------------------------------------------------


def drop_runs_without_raw_rows(
    chosen: dict[str, dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    registry_path: Path | str,
    raw_path: Path | str,
    allow_missing_raw: bool,
) -> dict[str, dict[str, Any]]:
    """Refuse to score a registry run that contributed no raw rows.

    A stale ``status == complete`` registry row whose raw records were pruned,
    renamed, or never written would otherwise silently produce an empty cell
    that still looks like a full model x cell measurement.
    """
    observed = {str(row.get("run_id", "")) for row in raw_rows}
    missing = {
        model: str(row["run_id"])
        for model, row in chosen.items()
        if str(row["run_id"]) not in observed
    }
    if not missing:
        return chosen
    detail = ", ".join(f"{model}={run_id}" for model, run_id in sorted(missing.items()))
    message = (
        f"Registry {registry_path} selects complete runs with no rows in {raw_path}: {detail}. "
        "Re-run those cells or pass --allow-missing-raw to drop the models instead."
    )
    if not allow_missing_raw:
        raise ValueError(message)
    eu.logger.warning(
        "%s", {"warning": "run_without_raw_rows_dropped", "detail": message}
    )
    return {model: row for model, row in chosen.items() if model not in missing}


def score_cell(
    root: Path,
    dataset: str,
    variant: str,
    models: list[str],
    exclude_model_prefixes: list[str],
    include_smoke: bool = False,
    expected_stochastic_samples: int | None = DEFAULT_EXPECTED_STOCHASTIC_SAMPLES,
    allow_missing_raw: bool = False,
    run_group_id: str | None = DEFAULT_PAPER_RUN_GROUP_ID,
    expected_batch_order: str | None = DEFAULT_PAPER_BATCH_ORDER,
    expected_batch_size: int | None = DEFAULT_PAPER_BATCH_SIZE,
) -> dict[str, Any]:
    """Load one (dataset, variant) cell and return its scores plus provenance."""
    benchmark_path = eu.artifact_path(
        root / "data/processed/benchmark_items.csv", dataset, variant
    )
    benchmark = eu.read_csv_rows(benchmark_path)
    smoke_flags = [False, True] if include_smoke else [False]
    registry_paths = [
        eu.run_registry_path(root, dataset, variant, smoke=smoke)
        for smoke in smoke_flags
    ]
    raw_paths = [
        eu.model_outputs_raw_path(root, dataset, variant, smoke=smoke)
        for smoke in smoke_flags
    ]
    registry_rows = [
        row
        for path in registry_paths
        if path.exists()
        for row in eu.read_csv_rows(path)
    ]
    full_prefix = "full" if variant == "must" else f"full-{variant}"
    smoke_prefix = "smoke" if variant == "must" else f"smoke-{variant}"
    chosen = select_cell_runs(
        registry_rows,
        models,
        exclude_model_prefixes=exclude_model_prefixes,
        include_smoke=include_smoke,
        run_group_id=run_group_id,
        benchmark_item_count=len(benchmark),
        expected_stochastic_samples=int(expected_stochastic_samples or 0),
        expected_batch_order=expected_batch_order,
        expected_batch_size=expected_batch_size,
        run_prefixes=[full_prefix, smoke_prefix] if include_smoke else [full_prefix],
    )
    missing_models = [
        model
        for model in models
        if model not in chosen
        and not any(model.startswith(p) for p in exclude_model_prefixes)
    ]
    if missing_models and not allow_missing_raw:
        raise ValueError(
            "No compatible complete paper run found for model(s): "
            f"{', '.join(sorted(missing_models))}. Expected run_group_id={run_group_id!r}, "
            f"tasks=task1,task2, stochastic_samples={int(expected_stochastic_samples or 0)}, "
            f"batch_order={expected_batch_order!r}, batch_size={expected_batch_size!r}."
        )
    if missing_models:
        eu.logger.warning(
            "%s",
            {
                "warning": "models_without_compatible_runs_dropped",
                "models": sorted(missing_models),
                "run_group_id": run_group_id,
            },
        )
    run_ids = {str(row["run_id"]) for row in chosen.values()}
    raw_rows = (
        [
            row
            for path in raw_paths
            if path.exists()
            for row in stream_raw_rows(path, run_ids)
        ]
        if run_ids
        else []
    )
    registry_label = ", ".join(str(path) for path in registry_paths)
    raw_label = ", ".join(str(path) for path in raw_paths)
    chosen = drop_runs_without_raw_rows(
        chosen, raw_rows, registry_label, raw_label, allow_missing_raw
    )
    scored_benchmark = eu.benchmark_rows_with_current_raw_outputs(benchmark, raw_rows)
    sampling_plan = eu.SamplingPlan(
        stochastic_samples=int(expected_stochastic_samples or 0)
    )
    scores = (
        eu.build_uq_scores(scored_benchmark, raw_rows, sampling_plan=sampling_plan)
        if raw_rows
        else []
    )
    # Stamp the cell before the caller pools four of them; `paper_join_key`
    # reads it back.
    stamp_cell_identity(scores, dataset, variant)
    return {
        "dataset": dataset,
        "variant": variant,
        "registry_path": str(registry_paths[0]),
        "registry_paths": [str(path) for path in registry_paths],
        "benchmark_path": str(benchmark_path),
        "raw_path": str(raw_paths[0]),
        "raw_paths": [str(path) for path in raw_paths],
        "run_ids": {model: str(row["run_id"]) for model, row in sorted(chosen.items())},
        "n_benchmark_items": len(scored_benchmark),
        "n_raw_rows": len(raw_rows),
        "expected_stochastic_samples": expected_stochastic_samples,
        "sampling_plan": sampling_plan,
        "scores": scores,
        "raw_rows": raw_rows,
    }


def markdown_for_rows(title: str, rows: list[dict[str, Any]], fields: list[str]) -> str:
    return "\n".join([f"# {title}", "", eu.markdown_table(rows, fields), ""])


def export_tables(
    root: Path,
    cells: list[tuple[str, str]],
    models: list[str],
    exclude_model_prefixes: list[str],
    output_dir: Path,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    overwrite_snapshots: bool = False,
    include_smoke: bool = False,
    expected_stochastic_samples: int | None = DEFAULT_EXPECTED_STOCHASTIC_SAMPLES,
    allow_missing_raw: bool = False,
    run_group_id: str | None = DEFAULT_PAPER_RUN_GROUP_ID,
    expected_batch_order: str | None = DEFAULT_PAPER_BATCH_ORDER,
    expected_batch_size: int | None = DEFAULT_PAPER_BATCH_SIZE,
) -> dict[str, Any]:
    """Regenerate every paper-facing table for ``cells`` and write them out."""
    output_dir.mkdir(parents=True, exist_ok=True)
    task2_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    per_model_rows: list[dict[str, Any]] = []
    provenance_cells: list[dict[str, Any]] = []
    pooled_det_rows: list[dict[str, Any]] = []
    pooled_by_model: dict[str, list[dict[str, Any]]] = {}
    pooled_consistency: dict[PaperJoinKey, dict[str, Any]] = {}
    pooled_items_by_model: dict[str, int] = {}

    for dataset, variant in cells:
        cell = score_cell(
            root,
            dataset,
            variant,
            models,
            exclude_model_prefixes,
            include_smoke=include_smoke,
            expected_stochastic_samples=expected_stochastic_samples,
            allow_missing_raw=allow_missing_raw,
            run_group_id=run_group_id,
            expected_batch_order=expected_batch_order,
            expected_batch_size=expected_batch_size,
        )
        scores = cell.pop("scores")
        # Raw rows carry the parse failures that never became score rows.
        raw_rows = cell.pop("raw_rows")
        # The plan itself is not JSON; its provenance is the source it declares.
        cell["sampling_plan_source"] = cell.pop("sampling_plan").source
        det_rows = task2_deterministic_rows(scores)
        consistency = stochastic_rows_by_method(scores, "modality_consistency")
        # Index the deterministic rows once; the per-model and per-modality
        # slices below would otherwise rescan them models * (1 + 4) times.
        rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        rows_by_model_modality: dict[tuple[str, str], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for row in det_rows:
            model = str(row.get("model", ""))
            rows_by_model[model].append(row)
            rows_by_model_modality[(model, str(row.get("source_modality", "")))].append(
                row
            )
        cell_models = sorted(rows_by_model)
        task2_rows.append(
            task2_snapshot_row(dataset, variant, len(cell_models), det_rows, raw_rows)
        )
        confidence_rows.append(
            confidence_snapshot_row(dataset, variant, det_rows, scores)
        )

        items_per_model = cell["n_benchmark_items"]
        for model in cell_models:
            pooled_by_model.setdefault(model, []).extend(rows_by_model[model])
            pooled_items_by_model[model] = (
                pooled_items_by_model.get(model, 0) + items_per_model
            )
            for modality in eu.MODALITIES:
                modality_rows = rows_by_model_modality.get((model, modality), [])
                if not modality_rows:
                    continue
                per_model_rows.append(
                    per_model_row(
                        model,
                        dataset,
                        variant,
                        modality,
                        modality_rows,
                        consistency,
                        n_items=_items_for_modality(items_per_model),
                        bootstrap_samples=bootstrap_samples,
                    )
                )
        pooled_det_rows.extend(det_rows)
        pooled_consistency.update(consistency)
        provenance_cells.append(cell)

    headline_rows = [
        per_model_row(
            model,
            "all",
            "all",
            "all",
            rows,
            pooled_consistency,
            n_items=pooled_items_by_model.get(model, len(rows)),
            bootstrap_samples=bootstrap_samples,
        )
        for model, rows in sorted(pooled_by_model.items())
    ]

    pooled_ci = eu.text_over_commitment_ci_fields(
        pooled_det_rows,
        iterations=bootstrap_samples,
        seed=BOOTSTRAP_SEED,
    )
    headline_ci_rows = [
        {
            "headline_key": "broad_text_strengthening",
            "value": pooled_ci.get("text_over_commitment", ""),
            "ci_low": pooled_ci.get("text_over_commitment_ci_low", ""),
            "ci_high": pooled_ci.get("text_over_commitment_ci_high", ""),
            "n_numerator": pooled_ci.get("text_over_commitment_n_numerator", ""),
            "n_denominator": pooled_ci.get("text_over_commitment_n_denominator", ""),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        {
            "headline_key": "strict_text_strengthening",
            "value": pooled_ci.get("strict_text_over_commitment", ""),
            "ci_low": pooled_ci.get("strict_text_over_commitment_ci_low", ""),
            "ci_high": pooled_ci.get("strict_text_over_commitment_ci_high", ""),
            "n_numerator": pooled_ci.get("strict_text_over_commitment_n_numerator", ""),
            "n_denominator": pooled_ci.get(
                "strict_text_over_commitment_n_denominator", ""
            ),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    ]

    suffix = "" if overwrite_snapshots else REGENERATED_SUFFIX
    task2_path = output_dir / TASK2_SNAPSHOT_NAME.replace(".csv", f"{suffix}.csv")
    confidence_path = output_dir / CONFIDENCE_SNAPSHOT_NAME.replace(
        ".csv", f"{suffix}.csv"
    )
    modality_path = output_dir / PER_MODEL_MODALITY_NAME
    headline_path = output_dir / PER_MODEL_HEADLINE_NAME
    headline_ci_path = output_dir / HEADLINE_CI_NAME
    provenance_path = output_dir / PROVENANCE_NAME

    eu.write_csv_rows(task2_path, task2_rows, fieldnames=TASK2_SNAPSHOT_FIELDS)
    eu.write_csv_rows(
        confidence_path, confidence_rows, fieldnames=CONFIDENCE_SNAPSHOT_FIELDS
    )
    eu.write_csv_rows(modality_path, per_model_rows, fieldnames=PER_MODEL_FIELDS)
    eu.write_csv_rows(headline_path, headline_rows, fieldnames=PER_MODEL_FIELDS)
    eu.write_csv_rows(headline_ci_path, headline_ci_rows)
    modality_path.with_suffix(".md").write_text(
        markdown_for_rows("Per-Model Modality Table", per_model_rows, PER_MODEL_FIELDS),
        encoding="utf-8",
    )
    headline_path.with_suffix(".md").write_text(
        markdown_for_rows("Per-Model Headline", headline_rows, PER_MODEL_FIELDS),
        encoding="utf-8",
    )
    provenance = {
        "generated_at_utc": eu.utc_now_iso(),
        "script": "scripts/export_paper_tables.py",
        "models_cohort": models,
        "exclude_model_prefixes": exclude_model_prefixes,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "expected_stochastic_samples": expected_stochastic_samples,
        "sampling_plan_source": eu.SAMPLING_PLAN_SOURCE_PLANNED,
        "run_group_id": run_group_id,
        "expected_batch_order": expected_batch_order,
        "expected_batch_size": expected_batch_size,
        "allow_missing_raw": allow_missing_raw,
        "cells": provenance_cells,
        "outputs": {
            "task2_snapshot": str(task2_path),
            "confidence_snapshot": str(confidence_path),
            "per_model_modality_table": str(modality_path),
            "per_model_headline": str(headline_path),
            "headline_bootstrap_ci": str(headline_ci_path),
        },
    }
    eu.write_json(provenance_path, provenance)
    return {
        "task2_rows": task2_rows,
        "confidence_rows": confidence_rows,
        "per_model_rows": per_model_rows,
        "headline_rows": headline_rows,
        "headline_ci_rows": headline_ci_rows,
        "paths": {
            "task2": task2_path,
            "confidence": confidence_path,
            "per_model_modality": modality_path,
            "per_model_headline": headline_path,
            "headline_ci": headline_ci_path,
            "provenance": provenance_path,
        },
    }


def _items_for_modality(items_per_model: int) -> int:
    """Benchmark items per model for one source modality in a cell."""
    return items_per_model // len(eu.MODALITIES) if items_per_model else 0


def parse_cells(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return list(DEFAULT_CELLS)
    cells = []
    for value in values:
        dataset, _, variant = str(value).partition("/")
        if not dataset or not variant:
            raise ValueError(f"Cell must be given as dataset/variant, got {value!r}")
        cells.append(
            (eu.normalize_dataset_id(dataset), eu.normalize_benchmark_variant(variant))
        )
    return cells


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cell",
        action="append",
        help="dataset/variant cell, repeatable (default: all four).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_MODELS),
        help="Official model cohort to pool over.",
    )
    parser.add_argument(
        "--exclude-model-prefix",
        action="append",
        default=None,
        help="Model id prefix to exclude (default: azure.).",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES
    )
    parser.add_argument(
        "--overwrite-snapshots",
        action="store_true",
        help="Write the canonical snapshot names instead of the *_regenerated variants.",
    )
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument(
        "--expected-stochastic-samples",
        type=int,
        default=DEFAULT_EXPECTED_STOCHASTIC_SAMPLES,
        help="Stochastic samples every item is expected to have (default: 5).",
    )
    parser.add_argument(
        "--allow-missing-raw",
        action="store_true",
        help="Warn and drop models whose selected run has no raw rows instead of failing.",
    )
    parser.add_argument(
        "--run-group-id",
        default=DEFAULT_PAPER_RUN_GROUP_ID,
        help="Compatible paper run group to select.",
    )
    parser.add_argument(
        "--expected-batch-order",
        default=DEFAULT_PAPER_BATCH_ORDER,
        choices=sorted(eu.BATCH_ORDERS),
        help="Batch order required for selected paper runs.",
    )
    parser.add_argument(
        "--expected-batch-size",
        type=int,
        default=DEFAULT_PAPER_BATCH_SIZE,
        help="Batch size required for selected paper runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = eu.project_root()
    output_dir = args.output_dir or (root / "outputs")
    result = export_tables(
        root,
        parse_cells(args.cell),
        list(args.models),
        args.exclude_model_prefix or list(DEFAULT_EXCLUDE_MODEL_PREFIXES),
        output_dir,
        bootstrap_samples=args.bootstrap_samples,
        overwrite_snapshots=args.overwrite_snapshots,
        include_smoke=args.include_smoke,
        expected_stochastic_samples=args.expected_stochastic_samples,
        allow_missing_raw=args.allow_missing_raw,
        run_group_id=args.run_group_id,
        expected_batch_order=args.expected_batch_order,
        expected_batch_size=args.expected_batch_size,
    )
    for name, path in result["paths"].items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
