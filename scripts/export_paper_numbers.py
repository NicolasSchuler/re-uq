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
# Task 1, the detector AUROCs and the blind-audit rates used to come from
# `paper_task1_control_metrics.csv`, `paper_task3_blind_audit_metrics.csv`,
# `blind_task3_model_summary.csv`, `blind_task3_analysis_summary.csv`,
# `paper_uq_text_drift_metrics_overall.csv` and the two
# `acse_text_drift_uncertainty_*` files -- static 2026-05 snapshots that no
# code in this repository writes, so a rerun could not refresh them. They are
# replaced by the generated per-model RQ table.
PER_MODEL_RQ = "paper_per_model_rq_table.csv"
TASK2_TEXT_DRIFT = "paper_task2_text_drift_metrics.csv"
CONFIDENCE_STABILITY = "paper_text_drift_confidence_and_stability.csv"
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
    PER_MODEL_RQ: (
        "model",
        "dataset",
        "variant",
        "task1_n",
        "task1_accuracy",
        "task1_unsupported_acceptance_90_rate",
        "task2_n_readable",
        "task2_strict_strengthening_n",
        "task2_strict_strengthening_rate",
        "task2_broad_strengthening_rate",
        "task2_weak_n_readable",
        "task2_weak_strict_strengthening_rate",
        "task2_weak_strict_high_conf_90_n",
        "task2_weak_strict_high_conf_90_denominator",
        "task2_weak_strict_high_conf_90_rate",
        "task2_strict_high_conf_90_rate",
        "task2_strict_agreement_rate",
        "task2_meaning_variation_auroc",
        "task2_verbalized_confidence_auroc",
        "task3_strict_flagged_rate",
        "task3_strict_called_preserved_rate",
    ),
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


#: Characters a raw model id can carry that LaTeX would otherwise eat. Local
#: ids look like `qwen/qwen3.5-9b`, so an unescaped label breaks the build.
LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    return "".join(LATEX_ESCAPES.get(character, character) for character in text)


def model_label(model_id: str) -> str:
    """Display label for the per-model tables; unknown ids are escaped as-is."""
    label = MODEL_LABELS.get(model_id)
    return label if label is not None else latex_escape(model_id)


def order_models(models: Iterable[str], cohort: Sequence[str] = ()) -> list[str]:
    """Hosted cohort first (in `cohort` order), then any other model by id.

    `cohort` comes from `paper_snapshot_provenance.json`, which the tables
    exporter writes with the cohort it actually selected, so changing the models
    of a rerun needs no code change here. It falls back to the pinned default
    cohort only when the provenance file predates that field.
    """
    present = list(dict.fromkeys(models))
    order = list(cohort) or list(export_paper_tables.DEFAULT_MODELS)
    hosted = [model for model in order if model in present]
    extra = sorted(model for model in present if model not in set(hosted))
    return hosted + extra


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

        self.hosted_models, self.local_models = self._cohort_split()

    def _cohort_split(self) -> tuple[list[str], list[str]]:
        """Hosted and local model ids, as the tables exporter recorded them.

        Read from `paper_snapshot_provenance.json` rather than hardcoded, so a
        rerun with a different cohort needs no change here. Older provenance
        files have neither field and fall back to the pinned default cohort.
        """
        path = self.outputs_dir / SNAPSHOT_PROVENANCE
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return [], []
        if not isinstance(payload, Mapping):
            return [], []
        hosted = [str(model) for model in payload.get("models_hosted", []) or []]
        local = [str(model) for model in payload.get("models_local", []) or []]
        return hosted, local

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


class RqTable:
    """Row lookup for the generated per-model RQ table.

    Three row kinds, all keyed on the same two ``"all"`` sentinels the
    per-model headline already uses: one per model (pooled over cells), one per
    cell (pooled over models), and one grand row. Every RQ1/RQ2/RQ3 macro reads
    one of them, so a rate and the range printed beside it can never come from
    different denominators.
    """

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self.by_model: dict[str, Mapping[str, Any]] = {}
        self.by_cell: dict[tuple[str, str], Mapping[str, Any]] = {}
        self.grand: Mapping[str, Any] | None = None
        for row in rows:
            model, dataset, variant = (
                str(row["model"]),
                str(row["dataset"]),
                str(row["variant"]),
            )
            if model != "all" and (dataset, variant) == ("all", "all"):
                self.by_model[model] = row
            elif model == "all" and (dataset, variant) != ("all", "all"):
                self.by_cell[(dataset, variant)] = row
            elif (model, dataset, variant) == ("all", "all", "all"):
                self.grand = row
        if self.grand is None or not self.by_model or not self.by_cell:
            raise PaperNumbersError(
                f"{PER_MODEL_RQ}: expected per-model rows, per-cell rows and one "
                "pooled row (model=dataset=variant='all')"
            )

    def pooled(self, column: str) -> float:
        return _number(self.grand or {}, column, PER_MODEL_RQ)

    def over_cells(self, column: str) -> list[float]:
        return [_number(row, column, PER_MODEL_RQ) for row in self.by_cell.values()]

    def over_models(self, column: str) -> list[float]:
        return [_number(row, column, PER_MODEL_RQ) for row in self.by_model.values()]

    def cell(self, cell: tuple[str, str], column: str) -> float:
        row = self.by_cell.get(cell)
        if row is None:
            raise PaperNumbersError(
                f"{PER_MODEL_RQ}: cell {_cell_name(cell)} is absent"
            )
        return _number(row, column, PER_MODEL_RQ)


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


CLUSTER_LABEL = {
    eu.DEFAULT_BOOTSTRAP_CLUSTER_FIELD: "request-clustered",
    eu.BOOTSTRAP_CLUSTER_FALLBACK_FIELD: "seed-clustered",
}


def cluster_note(cluster_field: str) -> str:
    """How the primary interval was resampled, as the artifact recorded it.

    A fixed string would go stale the moment the resampling unit changes, and a
    slice that quietly fell back to the seed would still claim the request.
    """
    if not cluster_field:
        return "pooled, resampling unit not recorded"
    label = CLUSTER_LABEL.get(cluster_field, f"{cluster_field}-clustered")
    return f"pooled, {label} CI"


class HeadlineRate(NamedTuple):
    """One pooled headline rate, both cluster bootstraps, and the unit used."""

    value: float
    ci_low: float
    ci_high: float
    n_numerator: int
    n_denominator: int
    seed_ci: tuple[float, float] | None
    cluster_field: str


def _headline_rate(artifacts: Artifacts, key: str) -> HeadlineRate:
    rows = [
        row
        for row in artifacts[HEADLINE_BOOTSTRAP_CI]
        if str(row["headline_key"]) == key
    ]
    row = _one(rows, f"{HEADLINE_BOOTSTRAP_CI}: headline_key={key}")
    columns = artifacts.columns(HEADLINE_BOOTSTRAP_CI)
    seed_ci: tuple[float, float] | None = None
    if {"seed_ci_low", "seed_ci_high"} <= columns:
        low = _number(row, "seed_ci_low", HEADLINE_BOOTSTRAP_CI)
        high = _number(row, "seed_ci_high", HEADLINE_BOOTSTRAP_CI)
        if math.isfinite(low) and math.isfinite(high):
            seed_ci = (low, high)
    return HeadlineRate(
        value=_number(row, "value", HEADLINE_BOOTSTRAP_CI),
        ci_low=_number(row, "ci_low", HEADLINE_BOOTSTRAP_CI),
        ci_high=_number(row, "ci_high", HEADLINE_BOOTSTRAP_CI),
        n_numerator=round(_number(row, "n_numerator", HEADLINE_BOOTSTRAP_CI)),
        n_denominator=round(_number(row, "n_denominator", HEADLINE_BOOTSTRAP_CI)),
        seed_ci=seed_ci,
        # Snapshots written before the migration have no such column.
        cluster_field=str(row.get("ci_cluster_field", "")).strip()
        if "ci_cluster_field" in columns
        else "",
    )


def _rq1_block(artifacts: Artifacts, warnings: list[str]) -> list[Macro]:
    modality = artifacts[MODALITY_TABLE]
    by_cell = _group(modality, _cell_id)
    by_model = _group(modality, lambda row: str(row["model"]))
    models = order_models(by_model, artifacts.hosted_models + artifacts.local_models)

    cell_stats = {
        cell: Strengthening(rows, f"{MODALITY_TABLE} {_cell_name(cell)}")
        for cell, rows in by_cell.items()
    }
    model_stats = {
        model: Strengthening(rows, f"{MODALITY_TABLE} {model}")
        for model, rows in by_model.items()
    }

    rq = RqTable(artifacts[PER_MODEL_RQ])
    upgrade_by_model = rq.over_models("task1_unsupported_acceptance_90_rate")
    weak_models = rq.over_models("task2_weak_strict_high_conf_90_rate")
    weak_rate = rq.cell(WEAK_HEADLINE_CELL, "task2_weak_strict_high_conf_90_rate")
    weak_numerator = rq.cell(WEAK_HEADLINE_CELL, "task2_weak_strict_high_conf_90_n")
    weak_denominator = rq.cell(
        WEAK_HEADLINE_CELL, "task2_weak_strict_high_conf_90_denominator"
    )

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

    strict = _headline_rate(artifacts, "strict_text_strengthening")
    broad = _headline_rate(artifacts, "broad_text_strengthening")
    strict_value, strict_low, strict_high = strict.value, strict.ci_low, strict.ci_high
    strict_num, strict_den, strict_seed = (
        strict.n_numerator,
        strict.n_denominator,
        strict.seed_ci,
    )
    broad_value, broad_low, broad_high = broad.value, broad.ci_low, broad.ci_high
    broad_num, broad_den, broad_seed = (
        broad.n_numerator,
        broad.n_denominator,
        broad.seed_ci,
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
    _cross_check_rq_table(artifacts, warnings, rq, strict, broad)

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
            fmt_percent(rq.pooled("task1_accuracy")),
            "pooled over cells and models",
        ),
        Macro(
            "numTaskOneAccRange",
            fmt_range(
                rq.over_cells("task1_accuracy"), fmt_percent, "Task 1 accuracy range"
            ),
        ),
        Macro(
            "numTaskOneUpgrade",
            fmt_percent(rq.pooled("task1_unsupported_acceptance_90_rate")),
            "unsupported mandatory acceptance, conf >= 0.90",
        ),
        Macro(
            "numTaskOneUpgradeCI",
            f"{fmt_percent(rq.pooled('task1_unsupported_acceptance_90_ci_low'))}--"
            f"{fmt_percent(rq.pooled('task1_unsupported_acceptance_90_ci_high'))}",
            cluster_note(str(rq.grand.get("task1_ci_cluster_field", ""))),
        ),
        Macro(
            "numTaskOneUpgradeRange",
            fmt_range(
                rq.over_cells("task1_unsupported_acceptance_90_rate"),
                fmt_percent,
                "Task 1 upgrade range",
            ),
        ),
        Macro(
            "numTaskOneUpgradeModels",
            fmt_range(upgrade_by_model, fmt_percent, "Task 1 upgrade by model"),
            f"per model, pooled over cells ({PER_MODEL_RQ})",
        ),
        Macro(
            "numLabelAcc", fmt_percent(_pooled(label_pairs, "Task 2 label accuracy"))
        ),
        Macro(
            "numLabelStrengthening",
            fmt_percent(_pooled(label_oc_pairs, "Task 2 label over-commitment")),
        ),
        Macro(
            "numStrictOverall",
            fmt_percent(strict_value),
            cluster_note(strict.cluster_field),
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
        Macro(
            "numBroadOverall",
            fmt_percent(broad_value),
            cluster_note(broad.cluster_field),
        ),
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
                rq.over_cells("task2_weak_strict_high_conf_90_rate"),
                fmt_percent,
                "weak strict over cells",
            ),
            "over cells",
        ),
        Macro("numWeakStrictNum", fmt_count(weak_numerator)),
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


def _cross_check_rq_table(
    artifacts: Artifacts,
    warnings: list[str],
    rq: RqTable,
    strict: HeadlineRate,
    broad: HeadlineRate,
) -> None:
    """The two generated headline paths must agree on the same numbers.

    `paper_headline_bootstrap_ci.csv` and the RQ table's pooled row are written
    by the same exporter run from the same score rows, so a disagreement means
    one of the two slices is wrong -- worth a warning rather than a silent
    preference for whichever the macro happens to read.
    """
    for label, headline, column in (
        ("strict", strict, "task2_strict_strengthening"),
        ("broad", broad, "task2_broad_strengthening"),
    ):
        pooled = rq.pooled(f"{column}_rate")
        if math.isfinite(pooled) and abs(pooled - headline.value) > 1e-9:
            warnings.append(
                f"{PER_MODEL_RQ}: pooled {label} strengthening {pooled:.6f} "
                f"disagrees with {HEADLINE_BOOTSTRAP_CI} {headline.value:.6f}"
            )
        declared = round(rq.pooled(f"{column}_denominator"))
        if declared != headline.n_denominator:
            warnings.append(
                f"{PER_MODEL_RQ}: pooled {label} denominator {declared} disagrees "
                f"with {HEADLINE_BOOTSTRAP_CI} {headline.n_denominator}"
            )
    _ = artifacts


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
    """Signal macros. Every rate is the pooled one, with its clustered CI.

    These used to be unweighted macros over the four cells, taken from static
    snapshots. The RQ table computes them over the same rows the bootstrap
    resamples, so the pooled value and the range beside it share a denominator.
    """
    rq = RqTable(artifacts[PER_MODEL_RQ])
    cluster = cluster_note(str(rq.grand.get("task2_ci_cluster_field", "")))
    return [
        Macro(
            "numHighConfShare",
            fmt_percent(rq.pooled("task2_strict_high_conf_90_rate")),
            cluster,
        ),
        Macro(
            "numHighConfShareCI",
            f"{fmt_percent(rq.pooled('task2_strict_high_conf_90_ci_low'))}--"
            f"{fmt_percent(rq.pooled('task2_strict_high_conf_90_ci_high'))}",
        ),
        Macro(
            "numHighConfShareRange",
            fmt_range(
                rq.over_cells("task2_strict_high_conf_90_rate"),
                fmt_percent,
                "high-confidence share over cells",
            ),
            "over cells",
        ),
        Macro(
            "numHighConfModelsRange",
            fmt_range(
                rq.over_models("task2_strict_high_conf_90_rate"),
                fmt_percent,
                "high-confidence share over models",
            ),
            "over models",
        ),
        Macro(
            "numSampleAgreement",
            fmt_percent(rq.pooled("task2_strict_agreement_rate")),
            "unanimous repeated samples, complete groups only",
        ),
        Macro(
            "numMeaningVarAUROC",
            fmt_auroc(rq.pooled("task2_meaning_variation_auroc")),
            cluster,
        ),
        Macro(
            "numMeaningVarAUROCCI",
            f"{fmt_auroc(rq.pooled('task2_meaning_variation_auroc_ci_low'))}--"
            f"{fmt_auroc(rq.pooled('task2_meaning_variation_auroc_ci_high'))}",
        ),
        Macro(
            "numMeaningVarAUROCRange",
            fmt_range(
                rq.over_cells("task2_meaning_variation_auroc"),
                fmt_auroc,
                "meaning-variation AUROC over cells",
            ),
            "over cells",
        ),
        Macro(
            "numMeaningVarAUROCModels",
            fmt_range(
                rq.over_models("task2_meaning_variation_auroc"),
                fmt_auroc,
                "meaning-variation AUROC over models",
            ),
            "over models",
        ),
        Macro(
            "numVerbConfAUROC",
            fmt_auroc(rq.pooled("task2_verbalized_confidence_auroc")),
            "verbalized confidence as a strengthening detector",
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

    rq = RqTable(artifacts[PER_MODEL_RQ])

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
            fmt_percent(rq.pooled("task3_strict_flagged_rate")),
            cluster_note(str(rq.grand.get("task3_ci_cluster_field", ""))),
        ),
        Macro(
            "numBlindRecallCI",
            f"{fmt_percent(rq.pooled('task3_strict_flagged_ci_low'))}--"
            f"{fmt_percent(rq.pooled('task3_strict_flagged_ci_high'))}",
        ),
        Macro(
            "numBlindRecallRange",
            fmt_range(
                rq.over_cells("task3_strict_flagged_rate"),
                fmt_percent,
                "blind strict recall over cells",
            ),
            "over cells",
        ),
        Macro(
            "numBlindRecallModelsRange",
            fmt_range(
                rq.over_models("task3_strict_flagged_rate"),
                fmt_percent,
                "blind strict recall over models",
            ),
            "per model, pooled over cells",
        ),
        Macro(
            "numBlindMissed",
            fmt_percent(rq.pooled("task3_strict_called_preserved_rate")),
            "strict-strengthened answers called preserved",
        ),
        Macro(
            "numBlindMissedRange",
            fmt_range(
                rq.over_cells("task3_strict_called_preserved_rate"),
                fmt_percent,
                "blind false preserve over cells",
            ),
            "over cells",
        ),
    ]


def fmt_pct_ci(rate: float, low: float, high: float) -> str:
    """``8.6 [7.6, 9.6]``, or ``8.6 ---`` for a degenerate interval.

    The dash case is tested on the *formatted* bounds, not the raw floats: an
    interval that collapses to within a rounding step would otherwise print as
    ``[0.0, 0.0]``, which reads like a measured width rather than none. A rate
    that does not exist at all (no rows) prints a bare dash.
    """
    if not math.isfinite(rate):
        return "---"
    value = fmt_percent(rate)
    if not (math.isfinite(low) and math.isfinite(high)):
        return f"{value} ---"
    low_text, high_text = fmt_percent(low), fmt_percent(high)
    if low_text == high_text:
        return f"{value} ---"
    return f"{value} [{low_text}, {high_text}]"


def fmt_auroc_ci(value: float, low: float, high: float) -> str:
    """``0.768 [0.742, 0.791]``, with the same degenerate and absent cases."""
    if not math.isfinite(value):
        return "---"
    text = fmt_auroc(value)
    if not (math.isfinite(low) and math.isfinite(high)):
        return f"{text} ---"
    low_text, high_text = fmt_auroc(low), fmt_auroc(high)
    if low_text == high_text:
        return f"{text} ---"
    return f"{text} [{low_text}, {high_text}]"


def render_grouped_table_body(
    hosted_rows: Sequence[str],
    local_rows: Sequence[str],
    all_row: str,
    n_columns: int,
) -> str:
    r"""Everything between a table's header rule and its bottom rule.

    Hosted models, then local models (a `\placeholder` row while none have
    run), then the pooled row after an inner rule. Shared by all three per-model
    tables so their bodies cannot drift apart.
    """
    group = f"\\multicolumn{{{n_columns}}}{{@{{}}l}}"
    placeholder = " & ".join(
        [r"\placeholder{model}"] + [r"\placeholder{}"] * (n_columns - 1)
    )
    lines = [f"{group}{{\\textit{{Hosted}}}} \\\\", *hosted_rows]
    lines += [
        r"\addlinespace[0.3em]",
        f"{group}{{\\textit{{Local (llama.cpp)}}}} \\\\",
        *(local_rows or [placeholder + r" \\"]),
        r"\midrule",
        all_row,
    ]
    return "\n".join(lines)


def _rq_table_rows(rq: RqTable, models: Sequence[str]) -> tuple[list[str], list[str]]:
    """The `tab:rq1` and `tab:rq23` cells for each of `models`, in table order."""
    rq_one: list[str] = []
    rq_two_three: list[str] = []
    for model in models:
        row = rq.by_model.get(model)
        if row is None:
            raise PaperNumbersError(f"{PER_MODEL_RQ}: model {model} has no pooled row")
        rq_one.append(_rq_one_row(model_label(model), row))
        rq_two_three.append(_rq_two_three_row(model_label(model), row))
    return rq_one, rq_two_three


def _rate_cell(row: Mapping[str, Any], name: str) -> str:
    return fmt_pct_ci(
        _number(row, f"{name}_rate", PER_MODEL_RQ),
        _number(row, f"{name}_ci_low", PER_MODEL_RQ),
        _number(row, f"{name}_ci_high", PER_MODEL_RQ),
    )


def _rq_one_row(label: str, row: Mapping[str, Any]) -> str:
    """Model | Task 1 N, rate | Task 2 N, strict, broad | weak N, strict."""
    columns = [
        label,
        fmt_count(
            _number(row, "task1_unsupported_acceptance_90_denominator", PER_MODEL_RQ)
        ),
        _rate_cell(row, "task1_unsupported_acceptance_90"),
        fmt_count(_number(row, "task2_n_readable", PER_MODEL_RQ)),
        _rate_cell(row, "task2_strict_strengthening"),
        _rate_cell(row, "task2_broad_strengthening"),
        fmt_count(_number(row, "task2_weak_n_readable", PER_MODEL_RQ)),
        _rate_cell(row, "task2_weak_strict_strengthening"),
    ]
    return " & ".join(columns) + r" \\"


def _rq_two_three_row(label: str, row: Mapping[str, Any]) -> str:
    """Model | strengthened N | confidence | meaning variation | blind check."""
    columns = [
        label,
        fmt_count(_number(row, "task2_strict_strengthening_n", PER_MODEL_RQ)),
        _rate_cell(row, "task2_strict_high_conf_90"),
        fmt_auroc_ci(
            _number(row, "task2_meaning_variation_auroc", PER_MODEL_RQ),
            _number(row, "task2_meaning_variation_auroc_ci_low", PER_MODEL_RQ),
            _number(row, "task2_meaning_variation_auroc_ci_high", PER_MODEL_RQ),
        ),
        _rate_cell(row, "task3_strict_flagged"),
        _rate_cell(row, "task3_strict_called_preserved"),
    ]
    return " & ".join(columns) + r" \\"


def _rq_table_blocks(artifacts: Artifacts) -> list[Macro]:
    """The two RQ table bodies, hosted then local then the pooled row."""
    rq = RqTable(artifacts[PER_MODEL_RQ])
    hosted = order_models(
        [model for model in rq.by_model if model not in set(artifacts.local_models)],
        artifacts.hosted_models,
    )
    local = order_models(
        [model for model in rq.by_model if model in set(artifacts.local_models)],
        artifacts.local_models,
    )
    hosted_one, hosted_two_three = _rq_table_rows(rq, hosted)
    local_one, local_two_three = _rq_table_rows(rq, local)
    grand = rq.grand or {}
    return [
        Macro(
            "numTableRqOneRows",
            render_grouped_table_body(
                hosted_one, local_one, _rq_one_row("All models", grand), 8
            ),
            "tab:rq1 body",
        ),
        Macro(
            "numTableRqTwoThreeRows",
            render_grouped_table_body(
                hosted_two_three,
                local_two_three,
                _rq_two_three_row("All models", grand),
                6,
            ),
            "tab:rq23 body",
        ),
    ]


def _per_model_blocks(artifacts: Artifacts) -> tuple[list[Macro], list[Macro], str]:
    modality = artifacts[MODALITY_TABLE]
    by_model = _group(modality, lambda row: str(row["model"]))
    models = order_models(by_model, artifacts.hosted_models + artifacts.local_models)

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
        ("RQ table bodies (tab:rq1, tab:rq23)", _rq_table_blocks(artifacts)),
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
