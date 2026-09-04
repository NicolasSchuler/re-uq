"""Write the manuscript's ``numbers.tex`` macro file from the paper artifacts.

Every number the manuscript reports is printed through a LaTeX macro defined in
``manuscript/numbers.tex``. Those values used to be typed in by hand from
``outputs/paper_*.csv``. This exporter regenerates the whole file from the
artifacts, so a rerun of the pipeline propagates into the prose and tables
without anyone transcribing a digit.

Usage::

    .venv/bin/python scripts/export_paper_numbers.py                     # dry run
    .venv/bin/python scripts/export_paper_numbers.py --output manuscript/numbers.tex

The default ``--output`` is ``outputs/paper_numbers.tex`` so the default
invocation never touches the (gitignored) manuscript tree. When the output file
already exists the exporter prints a macro-level diff -- added, removed, changed
-- so a rerun's effect on the paper is visible before the file is used.

Formatting conventions (matching the hand-written file)
-------------------------------------------------------

* Percentages: one decimal, without the percent sign (``8.6``).
* AUROC / AUPRC: three decimals (``0.764``).
* Counts: thousands separators from five digits up (``16,448``), none below
  (``1412``) -- the typographic convention the submitted draft used.
* Ranges: ``a--b`` (LaTeX en dash), always min then max.

Aggregation
-----------

``docs/aggregation.md`` is the specification; this module only chooses which of
its two published conventions each macro uses, and every choice is annotated in
the emitted file:

* **Pooled** (``sum(numerators) / sum(denominators)``) for every strengthening
  rate, no-cue share, Task 1 control rate and per-model figure. This is the
  primary convention of ``docs/aggregation.md`` section 5, and it is the only
  one that keeps the per-model table's ``All`` column equal to the sum of its
  parts.
* **Unweighted macro over the four cells** for the three headline quantities
  ``docs/aggregation.md`` section 5 publishes that way (the p>=0.90
  high-confidence share and repeated-sample agreement), for the blind-audit
  recall / false-preserve headlines, and for every AUROC -- an AUROC has no
  numerator and denominator to pool, so a mean over cells is the only defined
  aggregate. Ranges over cells and over models are always ranges of the
  correspondingly pooled per-cell / per-model values.
* The weak-intent headline is a single named cell (``mlm_tapt/must``), as
  documented; ``numWeakStrictRange`` is the range of the same quantity over the
  four cells.

Model macro keys
----------------

Plain LaTeX control words are letters only, so a model id cannot appear
verbatim in a macro name. The six cohort models use the keys the manuscript
already prints (``GlmFourFiveAir``, ``GlmFourSeven``, ``GlmFive``,
``GlmFiveTurbo``, ``GlmFiveOne``, ``Gemma``). Any other model id is
transliterated: the id is split into runs of letters and single digits,
non-alphanumerics are dropped, each letter run is capitalized, and each digit
becomes its English name (``mistral-7b-instruct`` -> ``MistralSevenBInstruct``).
Unknown models keep their raw id as the display label and are listed after the
hosted cohort.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

try:
    import eval_utils as eu
    import export_paper_tables
except ModuleNotFoundError:  # pragma: no cover - invocation-path fallback
    from scripts import eval_utils as eu, export_paper_tables


EXPORTER_VERSION = "1.0.0"
DEFAULT_OUTPUT_NAME = "paper_numbers.tex"

# --- Input artifacts ---------------------------------------------------------

HEADLINE_METRICS = "paper_headline_metrics.csv"
HEADLINE_BOOTSTRAP_CI = "paper_headline_bootstrap_ci.csv"
PER_MODEL_HEADLINE = "paper_per_model_headline.csv"
MODALITY_TABLE = "paper_per_model_modality_table.csv"
TASK1_CONTROL = "paper_task1_control_metrics.csv"
TASK2_TEXT_DRIFT = "paper_task2_text_drift_metrics.csv"
CONFIDENCE_STABILITY = "paper_text_drift_confidence_and_stability.csv"
BLIND_AUDIT = "paper_task3_blind_audit_metrics.csv"
BLIND_MODEL_SUMMARY = "blind_task3_model_summary.csv"
BLIND_ANALYSIS_SUMMARY = "blind_task3_analysis_summary.csv"
UQ_TEXT_DRIFT_OVERALL = "paper_uq_text_drift_metrics_overall.csv"
ACSE_SUMMARY = "acse_text_drift_uncertainty_summary.csv"
ACSE_BY_MODEL = "acse_text_drift_uncertainty_by_model.csv"
PROBE_GRID = "embedding_diagnostic/probe_grid_summary.csv"
SNAPSHOT_PROVENANCE = "paper_snapshot_provenance.json"

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    HEADLINE_METRICS: ("headline_key", "value_pooled"),
    HEADLINE_BOOTSTRAP_CI: (
        "headline_key",
        "value",
        "ci_low",
        "ci_high",
        "n_numerator",
        "n_denominator",
    ),
    PER_MODEL_HEADLINE: (
        "model",
        "n_items",
        "broad_strengthening_n",
        "broad_strengthening_denominator",
        "strict_strengthening_n",
        "strict_strengthening_denominator",
        "strict_high_conf_share_90",
    ),
    MODALITY_TABLE: (
        "model",
        "dataset",
        "variant",
        "source_modality",
        "n_items",
        "broad_strengthening_n",
        "broad_strengthening_denominator",
        "strict_strengthening_n",
        "strict_strengthening_denominator",
        "mean_requirement_word_count",
    ),
    TASK1_CONTROL: (
        "dataset",
        "variant",
        "n",
        "accuracy",
        "unsupported_mandatory_acceptance_90",
    ),
    TASK2_TEXT_DRIFT: (
        "dataset",
        "variant",
        "n",
        "label_accuracy",
        "label_over_commitment",
        "text_modality_parse_coverage",
    ),
    CONFIDENCE_STABILITY: (
        "dataset",
        "variant",
        "strict_text_oc_n",
        "strict_text_oc_conf_ge_90",
        "strict_text_oc_unanimous_modality_samples",
    ),
    BLIND_AUDIT: (
        "dataset",
        "variant",
        "strict_gold_strengthening_n",
        "strict_strengthening_recall",
        "strict_false_preserve_rate",
    ),
    BLIND_MODEL_SUMMARY: (
        "dataset",
        "variant",
        "model",
        "task1_n",
        "task1_unsupported_mandatory_acceptance_90",
        "task2_weak_strict_text_strengthening_90",
        "task3_strict_strengthened_n",
        "task3_strict_strengthening_recall",
    ),
    BLIND_ANALYSIS_SUMMARY: ("dataset", "variant"),
    UQ_TEXT_DRIFT_OVERALL: ("uq_method", "strict_text_oc_auroc"),
    ACSE_SUMMARY: ("dataset", "variant", "strict_text_oc_auroc"),
    ACSE_BY_MODEL: ("dataset", "variant", "model", "strict_text_oc_auroc"),
    PROBE_GRID: (
        "feature_backend",
        "text_variant",
        "group_mode",
        "scope",
        "target",
        "auroc_mean",
        "auprc_mean",
    ),
}

# The weak-intent headline lives in the Task 2 snapshot once
# scripts/export_paper_tables.py has recomputed it, and in the blind Task 3
# summary otherwise. At least one of the two must carry it.
WEAK_HEADLINE_COLUMN = "weak_strict_text_strengthening_90"

# docs/aggregation.md section 5: the weak-intent headline is this single cell,
# not an aggregate.
WEAK_HEADLINE_CELL = ("mlm_tapt", "must")

SOURCE_CONDITIONS = ("mandatory", "recommended", "optional", "nice_to_have")
# Macro-name suffix per source condition; "nice_to_have" reads as "weak intent"
# in the manuscript.
CONDITION_KEYS = {
    "mandatory": "Mandatory",
    "recommended": "Recommended",
    "optional": "Optional",
    "nice_to_have": "Weak",
}

# --- Model naming ------------------------------------------------------------

MODEL_KEYS = {
    "glm-4.5-air": "GlmFourFiveAir",
    "glm-4.7": "GlmFourSeven",
    "glm-5": "GlmFive",
    "glm-5-turbo": "GlmFiveTurbo",
    "glm-5.1": "GlmFiveOne",
    "kit.gemma4-31b-it": "Gemma",
}
MODEL_LABELS = {
    "glm-4.5-air": "GLM-4.5-Air",
    "glm-4.7": "GLM-4.7",
    "glm-5": "GLM-5",
    "glm-5-turbo": "GLM-5-Turbo",
    "glm-5.1": "GLM-5.1",
    "kit.gemma4-31b-it": "Gemma-4-31B",
}
DIGIT_WORDS = (
    "Zero",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
)

# The macro names the submitted draft's hand-written numbers.tex defined. Pinned
# so a refactor cannot silently drop a macro the manuscript still prints.
SUBMITTED_DRAFT_MACROS = (
    "numSeedsPerDataset",
    "numSeeds",
    "numConditions",
    "numItems",
    "numItemsPerCell",
    "numCells",
    "numModels",
    "numBatchSize",
    "numStochasticSamples",
    "numTaskTwoAnswers",
    "numTaskTwoReadable",
    "numTaskOneAcc",
    "numTaskOneAccRange",
    "numTaskOneUpgrade",
    "numTaskOneUpgradeRange",
    "numLabelAcc",
    "numLabelStrengthening",
    "numStrictOverall",
    "numStrictOverallCI",
    "numStrictOverallNum",
    "numStrictOverallDen",
    "numBroadOverall",
    "numBroadOverallCI",
    "numBroadOverallNum",
    "numWeakStrict",
    "numWeakStrictRange",
    "numWeakStrictNum",
    "numWeakStrictDen",
    "numNoModalShare",
    "numWeakWords",
    "numOtherWords",
    "numHighConfShare",
    "numHighConfShareRange",
    "numSampleAgreement",
    "numMeaningVarAUROC",
    "numMeaningVarAUROCRange",
    "numVerbConfAUROC",
    "numEmbGlobalAUROC",
    "numEmbGlobalAUPRC",
    "numEmbSampledAUROC",
    "numEmbSampledAUPRC",
    "numEmbWithinRecommended",
    "numEmbWithinOptional",
    "numEmbWithinWeak",
    "numEmbLeakControl",
    "numEmbContextLevel",
    "numEmbContextDataset",
    "numBlindRecall",
    "numBlindRecallRange",
    "numBlindMissed",
    "numBlindMissedRange",
    "numNoCueShare",
    "numNoCueShareRange",
    "numNoCueShareModels",
    "numStrictOverallRange",
    "numBroadOverallRange",
    "numBroadModelsRange",
    "numWeakStrictModelsRange",
    "numHighConfModelsRange",
    "numBlindRecallModelsRange",
)


class PaperNumbersError(RuntimeError):
    """A required artifact, column, row or cell is missing or unusable."""


class Macro(NamedTuple):
    """One ``\\newcommand`` line: macro name, rendered value, trailing comment."""

    name: str
    value: str
    comment: str = ""


def model_key(model_id: str) -> str:
    """Letters-only macro key for ``model_id`` (see the module docstring)."""
    known = MODEL_KEYS.get(model_id)
    if known:
        return known
    parts = [
        DIGIT_WORDS[int(chunk)]
        if chunk.isdigit()
        else chunk[:1].upper() + chunk[1:].lower()
        for chunk in re.findall(r"[A-Za-z]+|[0-9]", model_id)
    ]
    key = "".join(parts)
    if not key:
        raise PaperNumbersError(
            f"model id {model_id!r} has no alphanumeric characters, so it cannot "
            "become a LaTeX macro name"
        )
    return key


def model_label(model_id: str) -> str:
    """Display label for the per-model table; unknown ids are used as-is."""
    return MODEL_LABELS.get(model_id, model_id)


def order_models(models: Iterable[str]) -> list[str]:
    """Hosted cohort first (in its pinned order), then any other model by id."""
    present = list(dict.fromkeys(models))
    cohort = [model for model in export_paper_tables.DEFAULT_MODELS if model in present]
    extra = sorted(model for model in present if model not in set(cohort))
    return cohort + extra


# --- Formatting --------------------------------------------------------------


def fmt_percent(value: float) -> str:
    """A rate in [0, 1] as a percentage with one decimal, no percent sign."""
    return f"{value * 100:.1f}"


def fmt_auroc(value: float) -> str:
    return f"{value:.3f}"


def fmt_count(value: float) -> str:
    """Counts carry thousands separators from five digits up, as in the draft."""
    if not math.isfinite(value):
        # round(nan) raises a bare ValueError; this says which artifact is blank.
        raise PaperNumbersError(
            f"a count is not finite ({value!r}); a source cell is blank"
        )
    count = round(value)
    return f"{count:,}" if abs(count) >= 10_000 else str(count)


def fmt_words(value: float) -> str:
    return f"{value:.1f}"


NAN_TOKEN = re.compile(r"(?<![A-Za-z0-9])nan(?![A-Za-z0-9])", re.IGNORECASE)


def nan_macros(macros: Mapping[str, str]) -> list[str]:
    """Macro names whose rendered value carries a non-finite number.

    ``_number`` returns NaN for a blank cell by design and the scalar formatters
    render it as the literal string ``nan``, so a missing artifact column would
    otherwise reach numbers.tex as ``\\numStrictOverallCI{nan--nan}``. Checking
    the rendered macros catches every such path at once.
    """
    return sorted(name for name, value in macros.items() if NAN_TOKEN.search(value))


def fmt_range(values: Sequence[float], formatter: Any, label: str) -> str:
    usable = [value for value in values if math.isfinite(value)]
    if not usable:
        raise PaperNumbersError(f"{label}: no finite values to build a range from")
    return f"{formatter(min(usable))}--{formatter(max(usable))}"


# --- Numeric helpers ---------------------------------------------------------


def _number(row: Mapping[str, Any], column: str, source: str) -> float:
    """Parse one cell, accepting both ``0.298`` and the legacy ``29.8%`` form."""
    raw = str(row.get(column, "")).strip()
    if not raw:
        return math.nan
    scale = 1.0
    if raw.endswith("%"):
        raw = raw[:-1].strip()
        scale = 100.0
    try:
        return float(raw) / scale
    except ValueError as exc:
        raise PaperNumbersError(
            f"{source}: column {column!r} is not numeric: {row.get(column)!r}"
        ) from exc


def _pooled(pairs: Iterable[tuple[float, float]], label: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for value, weight in pairs:
        if not (math.isfinite(value) and math.isfinite(weight)):
            continue
        numerator += value
        denominator += weight
    if denominator <= 0:
        raise PaperNumbersError(f"{label}: pooled denominator is zero")
    return numerator / denominator


def _macro_over(values: Iterable[float], label: str) -> float:
    usable = [value for value in values if math.isfinite(value)]
    if not usable:
        raise PaperNumbersError(f"{label}: no finite values to average")
    return sum(usable) / len(usable)


def _cell_id(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row["dataset"]), str(row["variant"]))


def _cell_name(cell: tuple[str, str]) -> str:
    return f"{cell[0]}/{cell[1]}"


def _ordered(values: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


# --- Loading and validation --------------------------------------------------


class Artifacts:
    """The validated input artifacts, as row lists keyed by file name."""

    def __init__(self, outputs_dir: Path) -> None:
        self.outputs_dir = outputs_dir
        self.paths: dict[str, Path] = {}
        self.rows: dict[str, list[dict[str, str]]] = {}
        missing: list[str] = []
        for name in REQUIRED_COLUMNS:
            path = outputs_dir / name
            if not path.is_file():
                missing.append(name)
                continue
            self.paths[name] = path
        if missing:
            raise PaperNumbersError(
                "missing required artifact(s) under "
                f"{outputs_dir}: {', '.join(sorted(missing))}. Run "
                "scripts/export_paper_tables.py and "
                "scripts/aggregate_paper_headline_metrics.py first."
            )
        for name, columns in REQUIRED_COLUMNS.items():
            frame = eu.read_csv_frame(self.paths[name])
            absent = [column for column in columns if column not in frame.columns]
            if absent:
                raise PaperNumbersError(
                    f"{name}: missing required column(s) {', '.join(absent)}"
                )
            rows = frame.to_dict(orient="records")
            if not rows:
                raise PaperNumbersError(f"{name}: no data rows")
            self.rows[name] = rows

    def __getitem__(self, name: str) -> list[dict[str, str]]:
        return self.rows[name]

    def columns(self, name: str) -> set[str]:
        return set(self.rows[name][0])

    def source_digests(self) -> list[tuple[str, str]]:
        return [(name, eu.sha256_file(self.paths[name])) for name in REQUIRED_COLUMNS]

    def stale_provenance(self) -> tuple[Path | None, list[str]]:
        """The provenance file and the inputs newer than it (empty when fresh)."""
        provenance = self.outputs_dir / SNAPSHOT_PROVENANCE
        if not provenance.is_file():
            return None, []
        stamp = provenance.stat().st_mtime
        newer = [
            name
            for name, path in sorted(self.paths.items())
            if path.stat().st_mtime > stamp
        ]
        return provenance, newer


def _one(rows: Sequence[Mapping[str, Any]], label: str) -> Mapping[str, Any]:
    if len(rows) != 1:
        raise PaperNumbersError(f"{label}: expected exactly one row, found {len(rows)}")
    return rows[0]


def _probe_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    text_variant: str,
    group_mode: str,
    scope: str,
    target: str,
) -> Mapping[str, Any]:
    """The single embedding-probe row for one (variant, grouping, scope, target)."""
    matches = [
        row
        for row in rows
        if str(row["feature_backend"]) == "mlx"
        and str(row["text_variant"]) == text_variant
        and str(row["group_mode"]) == group_mode
        and str(row["scope"]) == scope
        and str(row["target"]) == target
    ]
    label = (
        f"{PROBE_GRID}: feature_backend=mlx, text_variant={text_variant}, "
        f"group_mode={group_mode}, scope={scope}, target={target}"
    )
    return _one(matches, label)


# --- Per-model and per-cell aggregation --------------------------------------


class Strengthening:
    """Pooled strict/broad strengthening and no-cue shares over one row set."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], label: str) -> None:
        self.label = label
        self.items = 0.0
        self.denominator = 0.0
        self.strict_n = 0.0
        self.broad_n = 0.0
        for row in rows:
            self.items += _number(row, "n_items", label)
            self.denominator += _number(row, "broad_strengthening_denominator", label)
            self.strict_n += _number(row, "strict_strengthening_n", label)
            self.broad_n += _number(row, "broad_strengthening_n", label)
        self.strict_denominator = sum(
            _number(row, "strict_strengthening_denominator", label) for row in rows
        )

    @property
    def strict(self) -> float:
        return _pooled([(self.strict_n, self.strict_denominator)], self.label)

    @property
    def broad(self) -> float:
        return _pooled([(self.broad_n, self.denominator)], self.label)

    @property
    def no_cue(self) -> float:
        return _pooled([(self.items - self.denominator, self.items)], self.label)


def _group(
    rows: Sequence[Mapping[str, Any]], key: Any
) -> dict[Any, list[Mapping[str, Any]]]:
    grouped: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(key(row), []).append(row)
    return grouped


# --- Macro construction ------------------------------------------------------


def benchmark_design(outputs_dir: Path) -> dict[str, dict[str, Any]]:
    """The frozen benchmark design, keyed by dataset id.

    The seed and item counts the paper reports are otherwise derived from
    observed rows, so a seed missing from every model in every cell divides
    evenly and passes the guards below unnoticed. These manifests record what
    the benchmark was built to, which is the only way to see that.
    """
    design: dict[str, dict[str, Any]] = {}
    for path in sorted(outputs_dir.glob("benchmark_manifest*.json")):
        metadata = json.loads(path.read_text(encoding="utf-8")).get("metadata", {})
        dataset_id = str(metadata.get("dataset_id", "")).strip()
        if dataset_id:
            design[dataset_id] = metadata
    return design


def _check_against_design(
    outputs_dir: Path,
    datasets: Sequence[str],
    conditions: Sequence[str],
    seeds_per_dataset: int,
    warnings: list[str],
) -> None:
    design = benchmark_design(outputs_dir)
    for dataset in datasets:
        metadata = design.get(dataset)
        if metadata is None:
            warnings.append(
                f"no benchmark manifest for dataset {dataset!r} under {outputs_dir}; "
                "the seed and item counts are unchecked"
            )
            continue
        planned_seeds = int(metadata.get("seed_count", 0) or 0)
        if planned_seeds and planned_seeds != seeds_per_dataset:
            warnings.append(
                f"{MODALITY_TABLE}: {dataset} contributes {seeds_per_dataset} seeds, "
                f"but its benchmark manifest declares {planned_seeds}; the reported "
                "seed and item counts describe the answers on disk, not the design"
            )
        planned = [str(value) for value in metadata.get("source_modalities", [])]
        if planned and sorted(planned) != sorted(conditions):
            warnings.append(
                f"{MODALITY_TABLE}: {dataset} source conditions "
                f"{sorted(conditions)} differ from the manifest's {sorted(planned)}"
            )


def _benchmark_block(artifacts: Artifacts, warnings: list[str]) -> list[Macro]:
    rows = artifacts[MODALITY_TABLE]
    models = _ordered(str(row["model"]) for row in rows)
    cells = _ordered(_cell_id(row) for row in rows)
    datasets = _ordered(cell[0] for cell in cells)
    conditions = _ordered(str(row["source_modality"]) for row in rows)
    expected = len(models) * len(cells) * len(conditions)
    if len(rows) != expected:
        raise PaperNumbersError(
            f"{MODALITY_TABLE}: expected a complete "
            f"{len(models)} x {len(cells)} x {len(conditions)} grid "
            f"({expected} rows), found {len(rows)}"
        )
    overall = Strengthening(rows, MODALITY_TABLE)
    answers = round(overall.items)
    per_cell_block = len(models) * len(cells)
    if answers % per_cell_block:
        raise PaperNumbersError(
            f"{MODALITY_TABLE}: {answers} answers is not divisible by "
            f"{per_cell_block} model x cell blocks"
        )
    items_per_cell = answers // per_cell_block
    if items_per_cell % len(conditions):
        raise PaperNumbersError(
            f"{MODALITY_TABLE}: {items_per_cell} items per cell is not divisible "
            f"by {len(conditions)} source conditions"
        )
    seeds_per_dataset = items_per_cell // len(conditions)
    seeds = seeds_per_dataset * len(datasets)
    _check_against_design(
        artifacts.outputs_dir, datasets, conditions, seeds_per_dataset, warnings
    )
    return [
        Macro("numSeedsPerDataset", str(seeds_per_dataset)),
        Macro("numSeeds", str(seeds), f"{len(datasets)} datasets"),
        Macro("numConditions", str(len(conditions))),
        Macro("numItems", fmt_count(seeds * len(conditions))),
        Macro("numItemsPerCell", fmt_count(items_per_cell)),
        Macro("numCells", str(len(cells))),
        Macro("numModels", str(len(models))),
        Macro("numBatchSize", str(export_paper_tables.DEFAULT_PAPER_BATCH_SIZE)),
        Macro(
            "numStochasticSamples",
            str(export_paper_tables.DEFAULT_EXPECTED_STOCHASTIC_SAMPLES),
        ),
        Macro(
            "numTaskTwoAnswers",
            fmt_count(answers),
            f"{len(models)} models x {len(cells)} cells x {items_per_cell} items",
        ),
        Macro(
            "numTaskTwoReadable",
            fmt_count(overall.denominator),
            "answers with a readable text modality",
        ),
    ]


def _headline_rate(
    artifacts: Artifacts, key: str
) -> tuple[float, float, float, int, int, tuple[float, float] | None]:
    """Pooled value, CI bounds, counts and optional seed-clustered CI."""
    rows = [
        row
        for row in artifacts[HEADLINE_BOOTSTRAP_CI]
        if str(row["headline_key"]) == key
    ]
    row = _one(rows, f"{HEADLINE_BOOTSTRAP_CI}: headline_key={key}")
    seed_ci: tuple[float, float] | None = None
    if {"seed_ci_low", "seed_ci_high"} <= artifacts.columns(HEADLINE_BOOTSTRAP_CI):
        low = _number(row, "seed_ci_low", HEADLINE_BOOTSTRAP_CI)
        high = _number(row, "seed_ci_high", HEADLINE_BOOTSTRAP_CI)
        if math.isfinite(low) and math.isfinite(high):
            seed_ci = (low, high)
    return (
        _number(row, "value", HEADLINE_BOOTSTRAP_CI),
        _number(row, "ci_low", HEADLINE_BOOTSTRAP_CI),
        _number(row, "ci_high", HEADLINE_BOOTSTRAP_CI),
        round(_number(row, "n_numerator", HEADLINE_BOOTSTRAP_CI)),
        round(_number(row, "n_denominator", HEADLINE_BOOTSTRAP_CI)),
        seed_ci,
    )


def _weak_source(artifacts: Artifacts) -> tuple[str, list[dict[str, str]]]:
    """The snapshot carrying the conf>=0.90 weak-intent column, preferring Task 2."""
    for name in (TASK2_TEXT_DRIFT, BLIND_ANALYSIS_SUMMARY):
        rows = artifacts[name]
        if WEAK_HEADLINE_COLUMN in artifacts.columns(name) and all(
            str(row.get(WEAK_HEADLINE_COLUMN, "")).strip() for row in rows
        ):
            return name, rows
    raise PaperNumbersError(
        f"column {WEAK_HEADLINE_COLUMN!r} is absent (or blank) in both "
        f"{TASK2_TEXT_DRIFT} and {BLIND_ANALYSIS_SUMMARY}; one of them must carry "
        "the weak-intent headline"
    )


def _rq1_block(artifacts: Artifacts, warnings: list[str]) -> list[Macro]:
    modality = artifacts[MODALITY_TABLE]
    by_cell = _group(modality, _cell_id)
    by_model = _group(modality, lambda row: str(row["model"]))
    models = order_models(by_model)

    cell_stats = {
        cell: Strengthening(rows, f"{MODALITY_TABLE} {_cell_name(cell)}")
        for cell, rows in by_cell.items()
    }
    model_stats = {
        model: Strengthening(rows, f"{MODALITY_TABLE} {model}")
        for model, rows in by_model.items()
    }

    task1 = artifacts[TASK1_CONTROL]
    accuracy_pairs = [
        (
            _number(row, "accuracy", TASK1_CONTROL) * _number(row, "n", TASK1_CONTROL),
            _number(row, "n", TASK1_CONTROL),
        )
        for row in task1
    ]
    upgrade_pairs = [
        (
            _number(row, "unsupported_mandatory_acceptance_90", TASK1_CONTROL)
            * _number(row, "n", TASK1_CONTROL),
            _number(row, "n", TASK1_CONTROL),
        )
        for row in task1
    ]
    blind_models = _group(artifacts[BLIND_MODEL_SUMMARY], lambda row: str(row["model"]))
    upgrade_by_model = [
        _pooled(
            [
                (
                    _number(
                        row,
                        "task1_unsupported_mandatory_acceptance_90",
                        BLIND_MODEL_SUMMARY,
                    )
                    * _number(row, "task1_n", BLIND_MODEL_SUMMARY),
                    _number(row, "task1_n", BLIND_MODEL_SUMMARY),
                )
                for row in rows
            ],
            f"{BLIND_MODEL_SUMMARY} {model} task1 upgrade",
        )
        for model, rows in blind_models.items()
    ]

    task2 = artifacts[TASK2_TEXT_DRIFT]
    label_pairs = [
        (
            _number(row, "label_accuracy", TASK2_TEXT_DRIFT)
            * _number(row, "n", TASK2_TEXT_DRIFT),
            _number(row, "n", TASK2_TEXT_DRIFT),
        )
        for row in task2
    ]
    label_oc_pairs = [
        (
            _number(row, "label_over_commitment", TASK2_TEXT_DRIFT)
            * _number(row, "n", TASK2_TEXT_DRIFT),
            _number(row, "n", TASK2_TEXT_DRIFT),
        )
        for row in task2
    ]
    coverage_no_cue = [
        1.0 - _number(row, "text_modality_parse_coverage", TASK2_TEXT_DRIFT)
        for row in task2
    ]
    for row in task2:
        cell = _cell_id(row)
        stats = cell_stats.get(cell)
        if stats is None:
            continue
        derived = 1.0 - stats.no_cue
        declared = _number(row, "text_modality_parse_coverage", TASK2_TEXT_DRIFT)
        if abs(derived - declared) > 1e-6:
            warnings.append(
                f"{TASK2_TEXT_DRIFT} {_cell_name(cell)}: text_modality_parse_coverage "
                f"{declared:.6f} disagrees with {MODALITY_TABLE} {derived:.6f}"
            )

    strict_value, strict_low, strict_high, strict_num, strict_den, strict_seed = (
        _headline_rate(artifacts, "strict_text_strengthening")
    )
    broad_value, broad_low, broad_high, broad_num, broad_den, broad_seed = (
        _headline_rate(artifacts, "broad_text_strengthening")
    )
    _cross_check_headlines(
        artifacts,
        warnings,
        {
            "strict_text_strengthening": (strict_value, strict_num, strict_den),
            "broad_text_strengthening": (broad_value, broad_num, broad_den),
        },
        cell_stats,
    )
    _cross_check_per_model(artifacts, warnings, model_stats)

    weak_source_name, weak_rows = _weak_source(artifacts)
    weak_by_cell = {
        _cell_id(row): _number(row, WEAK_HEADLINE_COLUMN, weak_source_name)
        for row in weak_rows
    }
    if WEAK_HEADLINE_CELL not in weak_by_cell:
        raise PaperNumbersError(
            f"{weak_source_name}: the weak-intent headline cell "
            f"{_cell_name(WEAK_HEADLINE_CELL)} is absent, so "
            "\\numWeakStrict cannot be derived"
        )
    weak_rate = weak_by_cell[WEAK_HEADLINE_CELL]
    weak_denominator = sum(
        _number(row, "strict_strengthening_denominator", MODALITY_TABLE)
        for row in by_cell[WEAK_HEADLINE_CELL]
        if str(row["source_modality"]) == "nice_to_have"
    )
    weak_models = [
        _pooled(
            [
                (
                    _number(
                        row,
                        "task2_weak_strict_text_strengthening_90",
                        BLIND_MODEL_SUMMARY,
                    )
                    * _weak_denominator(modality, model, _cell_id(row)),
                    _weak_denominator(modality, model, _cell_id(row)),
                )
                for row in rows
            ],
            f"{BLIND_MODEL_SUMMARY} {model} weak strict",
        )
        for model, rows in blind_models.items()
    ]

    weak_words = _pooled(
        [
            (
                _number(row, "mean_requirement_word_count", MODALITY_TABLE)
                * _number(row, "n_items", MODALITY_TABLE),
                _number(row, "n_items", MODALITY_TABLE),
            )
            for row in modality
            if str(row["source_modality"]) == "nice_to_have"
        ],
        "weak requirement word count",
    )
    other_words = _pooled(
        [
            (
                _number(row, "mean_requirement_word_count", MODALITY_TABLE)
                * _number(row, "n_items", MODALITY_TABLE),
                _number(row, "n_items", MODALITY_TABLE),
            )
            for row in modality
            if str(row["source_modality"]) != "nice_to_have"
        ],
        "other requirement word count",
    )

    no_cue_pooled = _pooled(
        [
            (stats.items - stats.denominator, stats.items)
            for stats in cell_stats.values()
        ],
        "pooled no-cue share",
    )

    macros = [
        Macro(
            "numTaskOneAcc",
            fmt_percent(_pooled(accuracy_pairs, "Task 1 accuracy")),
            "pooled over cells, item-weighted",
        ),
        Macro(
            "numTaskOneAccRange",
            fmt_range(
                [_number(row, "accuracy", TASK1_CONTROL) for row in task1],
                fmt_percent,
                "Task 1 accuracy range",
            ),
        ),
        Macro(
            "numTaskOneUpgrade",
            fmt_percent(_pooled(upgrade_pairs, "Task 1 upgrade")),
            "unsupported mandatory acceptance, conf >= 0.90",
        ),
        Macro(
            "numTaskOneUpgradeRange",
            fmt_range(
                [
                    _number(row, "unsupported_mandatory_acceptance_90", TASK1_CONTROL)
                    for row in task1
                ],
                fmt_percent,
                "Task 1 upgrade range",
            ),
        ),
        Macro(
            "numTaskOneUpgradeModels",
            fmt_range(upgrade_by_model, fmt_percent, "Task 1 upgrade by model"),
            f"per model, pooled over cells ({BLIND_MODEL_SUMMARY})",
        ),
        Macro(
            "numLabelAcc", fmt_percent(_pooled(label_pairs, "Task 2 label accuracy"))
        ),
        Macro(
            "numLabelStrengthening",
            fmt_percent(_pooled(label_oc_pairs, "Task 2 label over-commitment")),
        ),
        Macro(
            "numStrictOverall", fmt_percent(strict_value), "pooled, seed-clustered CI"
        ),
        Macro(
            "numStrictOverallCI",
            f"{fmt_percent(strict_low)}--{fmt_percent(strict_high)}",
        ),
    ]
    if strict_seed is not None:
        macros.append(
            Macro(
                "numStrictOverallSeedCI",
                f"{fmt_percent(strict_seed[0])}--{fmt_percent(strict_seed[1])}",
                "secondary, seed-clustered",
            )
        )
    macros += [
        Macro("numStrictOverallNum", fmt_count(strict_num)),
        Macro("numStrictOverallDen", fmt_count(strict_den)),
        Macro(
            "numStrictOverallRange",
            fmt_range(
                [stats.strict for stats in cell_stats.values()],
                fmt_percent,
                "strict range over cells",
            ),
            "over cells",
        ),
        Macro(
            "numStrictModelsRange",
            fmt_range(
                [model_stats[model].strict for model in models],
                fmt_percent,
                "strict range over models",
            ),
            "over models",
        ),
        Macro("numBroadOverall", fmt_percent(broad_value), "pooled"),
        Macro(
            "numBroadOverallCI", f"{fmt_percent(broad_low)}--{fmt_percent(broad_high)}"
        ),
    ]
    if broad_seed is not None:
        macros.append(
            Macro(
                "numBroadOverallSeedCI",
                f"{fmt_percent(broad_seed[0])}--{fmt_percent(broad_seed[1])}",
                "secondary, seed-clustered",
            )
        )
    macros += [
        Macro("numBroadOverallNum", fmt_count(broad_num)),
        Macro(
            "numBroadOverallRange",
            fmt_range(
                [stats.broad for stats in cell_stats.values()],
                fmt_percent,
                "broad range over cells",
            ),
            "over cells",
        ),
        Macro(
            "numBroadModelsRange",
            fmt_range(
                [model_stats[model].broad for model in models],
                fmt_percent,
                "broad range over models",
            ),
            "over models",
        ),
        Macro(
            "numWeakStrict",
            fmt_percent(weak_rate),
            f"{_cell_name(WEAK_HEADLINE_CELL)} cell, conf >= 0.90",
        ),
        Macro(
            "numWeakStrictRange",
            fmt_range(
                list(weak_by_cell.values()), fmt_percent, "weak strict over cells"
            ),
            "over cells",
        ),
        Macro("numWeakStrictNum", fmt_count(weak_rate * weak_denominator)),
        Macro("numWeakStrictDen", fmt_count(weak_denominator), "readable weak answers"),
        Macro(
            "numWeakStrictModelsRange",
            fmt_range(weak_models, fmt_percent, "weak strict over models"),
            "per model, pooled over cells",
        ),
        Macro(
            "numNoCueShare",
            fmt_percent(no_cue_pooled),
            f"{fmt_count(sum(stats.items - stats.denominator for stats in cell_stats.values()))}"
            f" / {fmt_count(sum(stats.items for stats in cell_stats.values()))}"
            " answers without a readable modal cue",
        ),
        Macro(
            "numNoCueShareRange",
            fmt_range(coverage_no_cue, fmt_percent, "no-cue share over cells"),
            "over cells",
        ),
        Macro(
            "numNoCueShareModels",
            fmt_range(
                [model_stats[model].no_cue for model in models],
                fmt_percent,
                "no-cue share over models",
            ),
            "over models",
        ),
        Macro(
            "numNoModalShare",
            fmt_percent(no_cue_pooled),
            "deprecated alias of \\numNoCueShare (submitted-draft macro name)",
        ),
        Macro("numWeakWords", fmt_words(weak_words), "nice_to_have sources"),
        Macro("numOtherWords", fmt_words(other_words), "the other three conditions"),
    ]
    return macros


def _weak_denominator(
    modality: Sequence[Mapping[str, Any]], model: str, cell: tuple[str, str]
) -> float:
    """Readable weak-intent answers for one model in one cell."""
    return sum(
        _number(row, "strict_strengthening_denominator", MODALITY_TABLE)
        for row in modality
        if str(row["model"]) == model
        and _cell_id(row) == cell
        and str(row["source_modality"]) == "nice_to_have"
    )


def _cross_check_headlines(
    artifacts: Artifacts,
    warnings: list[str],
    headlines: Mapping[str, tuple[float, int, int]],
    cell_stats: Mapping[tuple[str, str], Strengthening],
) -> None:
    """Warn when the three independent pooled sources disagree."""
    pooled = {
        str(row["headline_key"]): _number(row, "value_pooled", HEADLINE_METRICS)
        for row in artifacts[HEADLINE_METRICS]
    }
    derived = {
        "strict_text_strengthening": (
            sum(stats.strict_n for stats in cell_stats.values()),
            sum(stats.strict_denominator for stats in cell_stats.values()),
        ),
        "broad_text_strengthening": (
            sum(stats.broad_n for stats in cell_stats.values()),
            sum(stats.denominator for stats in cell_stats.values()),
        ),
    }
    for key, (value, numerator, denominator) in headlines.items():
        declared = pooled.get(key, math.nan)
        if math.isfinite(declared) and abs(declared - value) > 1e-6:
            warnings.append(
                f"{HEADLINE_METRICS} {key}: value_pooled {declared:.8f} disagrees "
                f"with {HEADLINE_BOOTSTRAP_CI} value {value:.8f}"
            )
        table_num, table_den = derived[key]
        if (round(table_num), round(table_den)) != (numerator, denominator):
            warnings.append(
                f"{HEADLINE_BOOTSTRAP_CI} {key}: {numerator}/{denominator} disagrees "
                f"with {MODALITY_TABLE} {round(table_num)}/{round(table_den)}"
            )


def _cross_check_per_model(
    artifacts: Artifacts,
    warnings: list[str],
    model_stats: Mapping[str, Strengthening],
) -> None:
    for row in artifacts[PER_MODEL_HEADLINE]:
        model = str(row["model"])
        stats = model_stats.get(model)
        if stats is None:
            warnings.append(
                f"{PER_MODEL_HEADLINE}: model {model} is absent from {MODALITY_TABLE}"
            )
            continue
        declared = round(_number(row, "strict_strengthening_n", PER_MODEL_HEADLINE))
        if declared != round(stats.strict_n):
            warnings.append(
                f"{PER_MODEL_HEADLINE} {model}: strict_strengthening_n {declared} "
                f"disagrees with {MODALITY_TABLE} {round(stats.strict_n)}"
            )


def _rq2_block(artifacts: Artifacts) -> list[Macro]:
    confidence = artifacts[CONFIDENCE_STABILITY]
    high_conf = [
        _number(row, "strict_text_oc_conf_ge_90", CONFIDENCE_STABILITY)
        for row in confidence
    ]
    unanimous = [
        _number(row, "strict_text_oc_unanimous_modality_samples", CONFIDENCE_STABILITY)
        for row in confidence
    ]
    high_conf_models = [
        _number(row, "strict_high_conf_share_90", PER_MODEL_HEADLINE)
        for row in artifacts[PER_MODEL_HEADLINE]
    ]

    acse_cells = [
        _number(row, "strict_text_oc_auroc", ACSE_SUMMARY)
        for row in artifacts[ACSE_SUMMARY]
    ]
    acse_by_model = _group(artifacts[ACSE_BY_MODEL], lambda row: str(row["model"]))
    acse_models = [
        _macro_over(
            [_number(row, "strict_text_oc_auroc", ACSE_BY_MODEL) for row in rows],
            f"{ACSE_BY_MODEL} {model}",
        )
        for model, rows in acse_by_model.items()
    ]

    verbalized = _one(
        [
            row
            for row in artifacts[UQ_TEXT_DRIFT_OVERALL]
            if str(row["uq_method"]) == "verbalized_confidence"
        ],
        f"{UQ_TEXT_DRIFT_OVERALL}: uq_method=verbalized_confidence",
    )

    return [
        Macro(
            "numHighConfShare",
            fmt_percent(_macro_over(high_conf, "high-confidence share")),
            "unweighted macro over cells (docs/aggregation.md section 5)",
        ),
        Macro(
            "numHighConfShareRange",
            fmt_range(high_conf, fmt_percent, "high-confidence share over cells"),
            "over cells",
        ),
        Macro(
            "numHighConfModelsRange",
            fmt_range(
                high_conf_models, fmt_percent, "high-confidence share over models"
            ),
            "over models",
        ),
        Macro(
            "numSampleAgreement",
            fmt_percent(_macro_over(unanimous, "repeated-sample agreement")),
        ),
        Macro(
            "numMeaningVarAUROC",
            fmt_auroc(_macro_over(acse_cells, "meaning-variation AUROC")),
            "macro over cells; an AUROC has no pooled form",
        ),
        Macro(
            "numMeaningVarAUROCRange",
            fmt_range(acse_cells, fmt_auroc, "meaning-variation AUROC over cells"),
            "over cells",
        ),
        Macro(
            "numMeaningVarAUROCModels",
            fmt_range(acse_models, fmt_auroc, "meaning-variation AUROC over models"),
            "over models, macro over each model's cells",
        ),
        Macro(
            "numVerbConfAUROC",
            fmt_auroc(
                _number(verbalized, "strict_text_oc_auroc", UQ_TEXT_DRIFT_OVERALL)
            ),
        ),
    ]


def _rq3_block(artifacts: Artifacts) -> list[Macro]:
    probe = artifacts[PROBE_GRID]
    deterministic = "deterministic_strict_text_overcommit"
    target = _probe_row(
        probe,
        text_variant="reqonly",
        group_mode="seed",
        scope="global",
        target=deterministic,
    )
    sampled = _probe_row(
        probe,
        text_variant="reqonly",
        group_mode="seed",
        scope="global",
        target="sample_strict_text_overcommit",
    )
    leak = _probe_row(
        probe,
        text_variant="prefixed",
        group_mode="seed",
        scope="global",
        target=deterministic,
    )
    context_level = _probe_row(
        probe,
        text_variant="reqonly",
        group_mode="item",
        scope="global",
        target="source_modality",
    )
    context_dataset = _probe_row(
        probe,
        text_variant="reqonly",
        group_mode="item",
        scope="global",
        target="dataset_variant",
    )
    within = {
        condition: _probe_row(
            probe,
            text_variant="reqonly",
            group_mode="seed",
            scope=f"source_modality={condition}",
            target=deterministic,
        )
        for condition in ("recommended", "optional", "nice_to_have")
    }

    audit = artifacts[BLIND_AUDIT]
    recall = [_number(row, "strict_strengthening_recall", BLIND_AUDIT) for row in audit]
    preserve = [
        _number(row, "strict_false_preserve_rate", BLIND_AUDIT) for row in audit
    ]
    blind_models = _group(artifacts[BLIND_MODEL_SUMMARY], lambda row: str(row["model"]))
    recall_models = [
        _pooled(
            [
                (
                    _number(
                        row, "task3_strict_strengthening_recall", BLIND_MODEL_SUMMARY
                    )
                    * _number(row, "task3_strict_strengthened_n", BLIND_MODEL_SUMMARY),
                    _number(row, "task3_strict_strengthened_n", BLIND_MODEL_SUMMARY),
                )
                for row in rows
                if math.isfinite(
                    _number(
                        row, "task3_strict_strengthening_recall", BLIND_MODEL_SUMMARY
                    )
                )
            ],
            f"{BLIND_MODEL_SUMMARY} {model} blind recall",
        )
        for model, rows in blind_models.items()
    ]

    def auroc(row: Mapping[str, Any]) -> str:
        return fmt_auroc(_number(row, "auroc_mean", PROBE_GRID))

    def auprc(row: Mapping[str, Any]) -> str:
        return fmt_auroc(_number(row, "auprc_mean", PROBE_GRID))

    return [
        Macro(
            "numEmbGlobalAUROC", auroc(target), "reqonly / seed-grouped, all sources"
        ),
        Macro("numEmbGlobalAUPRC", auprc(target)),
        Macro("numEmbSampledAUROC", auroc(sampled)),
        Macro("numEmbSampledAUPRC", auprc(sampled)),
        Macro("numEmbWithinRecommended", auroc(within["recommended"])),
        Macro("numEmbWithinOptional", auroc(within["optional"])),
        Macro("numEmbWithinWeak", auroc(within["nice_to_have"])),
        Macro("numEmbLeakControl", auroc(leak), "same probe on label-prefixed text"),
        Macro("numEmbContextLevel", auroc(context_level), "source modal force"),
        Macro(
            "numEmbContextDataset", auroc(context_dataset), "source dataset x variant"
        ),
        Macro(
            "numBlindRecall",
            fmt_percent(_macro_over(recall, "blind strict recall")),
            "strict gold strengthening, macro over cells",
        ),
        Macro(
            "numBlindRecallRange",
            fmt_range(recall, fmt_percent, "blind strict recall over cells"),
            "over cells",
        ),
        Macro(
            "numBlindRecallModelsRange",
            fmt_range(recall_models, fmt_percent, "blind strict recall over models"),
            "per model, pooled over cells",
        ),
        Macro(
            "numBlindMissed",
            fmt_percent(_macro_over(preserve, "blind strict false preserve")),
            "strict-strengthened answers called preserved",
        ),
        Macro(
            "numBlindMissedRange",
            fmt_range(preserve, fmt_percent, "blind false preserve over cells"),
            "over cells",
        ),
    ]


def _per_model_blocks(artifacts: Artifacts) -> tuple[list[Macro], list[Macro], str]:
    modality = artifacts[MODALITY_TABLE]
    by_model = _group(modality, lambda row: str(row["model"]))
    models = order_models(by_model)

    pooled: list[Macro] = []
    conditions: list[Macro] = []
    table_rows: list[str] = []
    for model in models:
        key = model_key(model)
        stats = Strengthening(by_model[model], f"{MODALITY_TABLE} {model}")
        pooled += [
            Macro(f"numStrict{key}", fmt_percent(stats.strict), model),
            Macro(f"numBroad{key}", fmt_percent(stats.broad)),
            Macro(f"numNoCue{key}", fmt_percent(stats.no_cue)),
        ]
        cells: dict[str, tuple[str, str]] = {}
        for condition in SOURCE_CONDITIONS:
            rows = [
                row
                for row in by_model[model]
                if str(row["source_modality"]) == condition
            ]
            if not rows:
                raise PaperNumbersError(
                    f"{MODALITY_TABLE}: model {model} has no {condition} rows"
                )
            condition_stats = Strengthening(
                rows, f"{MODALITY_TABLE} {model}/{condition}"
            )
            suffix = CONDITION_KEYS[condition]
            strict = fmt_percent(condition_stats.strict)
            broad = fmt_percent(condition_stats.broad)
            cells[condition] = (strict, broad)
            conditions += [
                Macro(f"numStrict{key}{suffix}", strict),
                Macro(f"numBroad{key}{suffix}", broad),
            ]
        columns = [model_label(model), fmt_percent(stats.no_cue)]
        # Mandatory sources cannot be strengthened, so the table omits them.
        for condition in ("recommended", "optional", "nice_to_have"):
            columns += list(cells[condition])
        columns += [fmt_percent(stats.strict), fmt_percent(stats.broad)]
        table_rows.append(" & ".join(columns) + r" \\")
    return pooled, conditions, "\n".join(table_rows)


def build_blocks(
    artifacts: Artifacts,
) -> tuple[list[tuple[str, list[Macro]]], list[str]]:
    """The ordered macro blocks and any non-fatal consistency warnings."""
    warnings: list[str] = []
    pooled, conditions, table_body = _per_model_blocks(artifacts)
    blocks = [
        ("Benchmark", _benchmark_block(artifacts, warnings)),
        ("RQ1: preservation (all cells)", _rq1_block(artifacts, warnings)),
        ("RQ2: uncertainty signals on strengthened outputs", _rq2_block(artifacts)),
        ("RQ3: detectors", _rq3_block(artifacts)),
        ("Per-model strengthening (pooled over cells)", pooled),
        ("Per-model x source condition (strict / broad)", conditions),
        (
            "Per-model table body (Table 3)",
            [Macro("numTableThreeRows", table_body, "rows only, hosted cohort first")],
        ),
    ]
    return blocks, warnings


def macros_from_blocks(blocks: Sequence[tuple[str, list[Macro]]]) -> dict[str, str]:
    """Flatten the blocks, failing closed on a duplicate macro name."""
    macros: dict[str, str] = {}
    for _, block in blocks:
        for macro in block:
            if macro.name in macros:
                raise PaperNumbersError(f"duplicate macro name \\{macro.name}")
            if not macro.name.isalpha():
                raise PaperNumbersError(
                    f"macro name {macro.name!r} is not letters-only, so plain LaTeX "
                    "cannot define it"
                )
            macros[macro.name] = macro.value
    return macros


# --- Rendering and diffing ---------------------------------------------------

RULE_WIDTH = 79


def render(
    blocks: Sequence[tuple[str, list[Macro]]],
    digests: Sequence[tuple[str, str]],
    outputs_dir: Path,
) -> str:
    lines = [
        "%% numbers.tex -- every number reported in the manuscript, in one place.",
        "%%",
        "%% GENERATED FILE -- do not edit by hand.",
        f"%% Generated by scripts/export_paper_numbers.py v{EXPORTER_VERSION}",
        f"%% at {eu.utc_now_iso()} from {outputs_dir}",
        "%%",
        "%% Naming: \\num<What><Scope>. Percentages are given without the percent",
        "%% sign, with one decimal. AUROC/AUPRC use three decimals. Counts carry a",
        "%% thousands separator from five digits up. Ranges are min--max.",
        "%% Aggregation conventions: docs/aggregation.md.",
        "%%",
        "%% Source artifacts (SHA-256):",
    ]
    width = max(len(name) for name, _ in digests)
    lines += [f"%%   {name:<{width}}  {digest}" for name, digest in digests]

    for title, block in blocks:
        if not block:
            continue
        prefix = f"% --- {title} "
        lines += ["", prefix + "-" * max(RULE_WIDTH - len(prefix), 1)]
        for macro in block:
            definition = f"\\newcommand{{\\{macro.name}}}{{{macro.value}}}"
            lines.append(
                f"{definition}  % {macro.comment}" if macro.comment else definition
            )
    return "\n".join(lines) + "\n"


MACRO_PATTERN = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{")


def parse_macros(text: str) -> dict[str, str]:
    """Macro name -> value for every ``\\newcommand`` in ``text``.

    Values may span lines and contain balanced braces, so the closing brace is
    found by counting rather than by a regular expression.
    """
    macros: dict[str, str] = {}
    for match in MACRO_PATTERN.finditer(text):
        depth = 1
        index = match.end()
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        macros[match.group(1)] = text[match.end() : index - 1]
    return macros


def diff_macros(before: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
    """Human-readable added / removed / changed lines, added and removed last."""
    changed = [
        f"  {'changed':<9}{name:<34}{before[name]} -> {after[name]}"
        for name in after
        if name in before and before[name] != after[name]
    ]
    added = [
        f"  {'added':<9}{name:<34}{after[name]}" for name in after if name not in before
    ]
    removed = [
        f"  {'removed':<9}{name:<34}{before[name]}"
        for name in before
        if name not in after
    ]
    return changed + added + removed


# --- CLI ---------------------------------------------------------------------


def build_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write the manuscript's numbers.tex macro file from the paper artifacts."
        )
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=root / "outputs",
        help="Directory holding the paper_*.csv artifacts (default: outputs/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs" / DEFAULT_OUTPUT_NAME,
        help=(
            "Where to write the macro file. The default stays inside outputs/ so "
            "the manuscript is only touched when it is named explicitly."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail instead of writing when any consistency warning fires or any "
            "macro carries a non-finite value. Use this for the manuscript."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    root = eu.project_root()
    args = build_parser(root).parse_args(argv)
    try:
        artifacts = Artifacts(args.outputs_dir)
        blocks, warnings = build_blocks(artifacts)
        macros = macros_from_blocks(blocks)
        text = render(blocks, artifacts.source_digests(), args.outputs_dir)
    except PaperNumbersError as exc:
        raise SystemExit(f"export_paper_numbers: {exc}") from exc

    provenance, newer = artifacts.stale_provenance()
    if provenance is None:
        print(f"note: {SNAPSHOT_PROVENANCE} is absent; skipped the freshness check.")
    elif newer:
        print(
            f"warning: {SNAPSHOT_PROVENANCE} is older than {len(newer)} input(s): "
            f"{', '.join(newer)}. Rerun scripts/export_paper_tables.py."
        )
    offenders = nan_macros(macros)
    if offenders:
        warnings.append(
            "non-finite value in "
            + ", ".join(offenders)
            + "; the artifact column those macros read is blank"
        )
    for warning in warnings:
        print(f"warning: {warning}")
    if warnings and args.strict:
        # Reported before anything is written, so --strict never leaves a
        # half-trusted numbers.tex behind.
        print(
            f"export_paper_numbers: {len(warnings)} warning(s) under --strict; "
            f"{args.output} not written."
        )
        return 1

    output = Path(args.output)
    before = (
        parse_macros(output.read_text(encoding="utf-8")) if output.is_file() else {}
    )
    eu.atomic_write_text(output, text)
    print(f"Wrote {output} ({len(macros)} macros)")
    if before:
        lines = diff_macros(before, macros)
        print(f"Diff vs the previous {output.name}:")
        print("\n".join(lines) if lines else "  (no macro added, removed or changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
