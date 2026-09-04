"""Tests for scripts/export_paper_numbers.py.

The exporter turns the untracked ``outputs/paper_*.csv`` artifacts into the
manuscript's ``numbers.tex`` macro file, so the real inputs are not available to
CI. These tests therefore build a miniature artifact set -- two models, two
cells, four source conditions -- whose every aggregate is hand-computable, and
assert the exact macro strings the exporter must emit.

The fixture is deliberately scaled so that some counts cross the thousands
separator boundary (32,000 answers) while others stay below it (3440
strengthened answers), which is exactly the formatting split the manuscript
uses.
"""

from __future__ import annotations

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import eval_utils as eu, export_paper_numbers as exporter

# --- Fixture -----------------------------------------------------------------

CELLS = [("mlm_tapt", "must"), ("nice", "shall")]
CONDITIONS = ["mandatory", "recommended", "optional", "nice_to_have"]
N_ITEMS_PER_ROW = 2000

# (model, cell) -> readable-text denominator of every condition row in the cell.
COVERAGE = {
    ("glm-5.1", ("mlm_tapt", "must")): 1600,
    ("glm-5.1", ("nice", "shall")): 1800,
    ("kit.gemma4-31b-it", ("mlm_tapt", "must")): 2000,
    ("kit.gemma4-31b-it", ("nice", "shall")): 2000,
}

# (model, cell) -> condition -> (strict_n, broad_n)
STRENGTHENING = {
    ("glm-5.1", ("mlm_tapt", "must")): {
        "mandatory": (0, 0),
        "recommended": (160, 320),
        "optional": (80, 240),
        "nice_to_have": (400, 800),
    },
    ("glm-5.1", ("nice", "shall")): {
        "mandatory": (0, 0),
        "recommended": (180, 360),
        "optional": (180, 180),
        "nice_to_have": (540, 900),
    },
    ("kit.gemma4-31b-it", ("mlm_tapt", "must")): {
        "mandatory": (0, 0),
        "recommended": (200, 400),
        "optional": (100, 200),
        "nice_to_have": (1000, 1200),
    },
    ("kit.gemma4-31b-it", ("nice", "shall")): {
        "mandatory": (0, 0),
        "recommended": (0, 200),
        "optional": (0, 200),
        "nice_to_have": (600, 1000),
    },
}

# Weak sources draw longer generated requirements than the other three
# conditions, which is what the numWeakWords / numOtherWords pair reports.
WEAK_WORDS = 20.0
OTHER_WORDS = 15.0

# (model, cell) -> (task1 upgrade rate, weak strict rate, strict gold n, recall)
BLIND_MODEL = {
    ("glm-5.1", ("mlm_tapt", "must")): (0.02, 0.25, 200, 0.5),
    ("glm-5.1", ("nice", "shall")): (0.04, 0.30, 100, 0.4),
    ("kit.gemma4-31b-it", ("mlm_tapt", "must")): (0.0, 0.50, 200, 0.7),
    ("kit.gemma4-31b-it", ("nice", "shall")): (0.0, 0.60, 100, 0.9),
}

# (model, cell) -> (task1 upgrade rate, blind strict recall).
RQ_MODEL = {
    ("glm-5.1", ("mlm_tapt", "must")): (0.02, 0.5),
    ("glm-5.1", ("nice", "shall")): (0.04, 0.4),
    ("kit.gemma4-31b-it", ("mlm_tapt", "must")): (0.0, 0.7),
    ("kit.gemma4-31b-it", ("nice", "shall")): (0.0, 0.9),
}
# cell -> shares that vary by cell, so a pooled value and the range beside it
# are different numbers in the fixture too.
CELL_SHARES = {
    ("mlm_tapt", "must"): {
        "task1_accuracy": 0.99,
        "high_conf": 0.99,
        "agreement": 0.65,
        "meaning_variation_auroc": 0.70,
        "verbalized_confidence_auroc": 0.76,
        "called_preserved": 0.44,
    },
    ("nice", "shall"): {
        "task1_accuracy": 0.97,
        "high_conf": 0.97,
        "agreement": 0.61,
        "meaning_variation_auroc": 0.80,
        "verbalized_confidence_auroc": 0.82,
        "called_preserved": 0.48,
    },
}
TASK1_ITEMS_PER_CELL = 1000

DETERMINISTIC = "deterministic_strict_text_overcommit"
SAMPLED = "sample_strict_text_overcommit"

# feature_backend, text_variant, group_mode, scope, target, auroc, auprc
PROBE_ROWS = [
    ("mlx", "reqonly", "seed", "global", DETERMINISTIC, 0.7, 0.17),
    ("mlx", "reqonly", "seed", "global", SAMPLED, 0.744, 0.276),
    (
        "mlx",
        "reqonly",
        "seed",
        "source_modality=recommended",
        DETERMINISTIC,
        0.613,
        0.03,
    ),
    ("mlx", "reqonly", "seed", "source_modality=optional", DETERMINISTIC, 0.612, 0.04),
    (
        "mlx",
        "reqonly",
        "seed",
        "source_modality=nice_to_have",
        DETERMINISTIC,
        0.62,
        0.39,
    ),
    ("mlx", "prefixed", "seed", "global", DETERMINISTIC, 0.822, 0.267),
    ("mlx", "reqonly", "item", "global", "source_modality", 0.8376, 0.694),
    ("mlx", "reqonly", "item", "global", "dataset_variant", 0.7276, 0.341),
    # Decoy rows the selectors must not pick up.
    ("tfidf", "reqonly", "seed", "global", DETERMINISTIC, 0.9, 0.9),
    ("mlx", "reqonly", "item", "global", DETERMINISTIC, 0.69, 0.165),
]


#: A model the fixture has no hand-written counts for (a local model, say)
#: reuses this cohort member's, so tests can add one without a second table.
FALLBACK_FIXTURE_MODEL = "kit.gemma4-31b-it"


def _coverage(model: str, cell: tuple[str, str]) -> int:
    return COVERAGE.get((model, cell), COVERAGE[(FALLBACK_FIXTURE_MODEL, cell)])


def _strengthening(model: str, cell: tuple[str, str]) -> dict[str, tuple[int, int]]:
    return STRENGTHENING.get(
        (model, cell), STRENGTHENING[(FALLBACK_FIXTURE_MODEL, cell)]
    )


def _rq_model(model: str, cell: tuple[str, str]) -> tuple[float, float]:
    return RQ_MODEL.get((model, cell), RQ_MODEL[(FALLBACK_FIXTURE_MODEL, cell)])


def _modality_rows(models: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in models:
        for cell in CELLS:
            denominator = _coverage(model, cell)
            for condition in CONDITIONS:
                strict_n, broad_n = _strengthening(model, cell)[condition]
                rows.append(
                    {
                        "model": model,
                        "dataset": cell[0],
                        "variant": cell[1],
                        "source_modality": condition,
                        "n_items": N_ITEMS_PER_ROW,
                        "broad_strengthening_n": broad_n,
                        "broad_strengthening_denominator": denominator,
                        "broad_strengthening_rate": broad_n / denominator,
                        "strict_strengthening_n": strict_n,
                        "strict_strengthening_denominator": denominator,
                        "strict_strengthening_rate": strict_n / denominator,
                        "mean_requirement_word_count": (
                            WEAK_WORDS if condition == "nice_to_have" else OTHER_WORDS
                        ),
                    }
                )
    return rows


def _per_model_headline_rows(models: list[str]) -> list[dict[str, object]]:
    high_conf = {"glm-5.1": 0.9, "kit.gemma4-31b-it": 1.0}
    rows: list[dict[str, object]] = []
    for model in models:
        items = denominator = strict_n = broad_n = 0
        for cell in CELLS:
            for condition in CONDITIONS:
                items += N_ITEMS_PER_ROW
                denominator += _coverage(model, cell)
                strict, broad = _strengthening(model, cell)[condition]
                strict_n += strict
                broad_n += broad
        rows.append(
            {
                "model": model,
                "dataset": "all",
                "variant": "all",
                "source_modality": "all",
                "n_items": items,
                "broad_strengthening_n": broad_n,
                "broad_strengthening_denominator": denominator,
                "strict_strengthening_n": strict_n,
                "strict_strengthening_denominator": denominator,
                "strict_high_conf_share_90": high_conf.get(model, 1.0),
            }
        )
    return rows


def _cell_counts(model, cell):
    """(strict, broad, readable, weak strict, weak readable) for one model-cell."""
    coverage = _coverage(model, cell)
    counts = _strengthening(model, cell)
    strict = sum(strict_n for strict_n, _ in counts.values())
    broad = sum(broad_n for _, broad_n in counts.values())
    return (
        strict,
        broad,
        coverage * len(CONDITIONS),
        counts["nice_to_have"][0],
        coverage,
    )


def _weighted(pairs):
    """Item-weighted mean of (value, weight) pairs, 0.0 when nothing weighs."""
    total = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / total if total else 0.0


def _rq_row(model, dataset, variant, models, cells):
    """One row of the per-model RQ table, derived from the modality fixture.

    Counts come from `STRENGTHENING`/`COVERAGE`, so the pooled row agrees with
    `paper_headline_bootstrap_ci.csv` and the exporter's cross-check stays
    silent; the cell-varying shares come from `CELL_SHARES`.
    """
    counts = [(m, cell, _cell_counts(m, cell)) for m in models for cell in cells]
    strict = sum(cell_counts[0] for _, _, cell_counts in counts)
    broad = sum(cell_counts[1] for _, _, cell_counts in counts)
    readable = sum(cell_counts[2] for _, _, cell_counts in counts)
    weak_strict = sum(cell_counts[3] for _, _, cell_counts in counts)
    weak_readable = sum(cell_counts[4] for _, _, cell_counts in counts)
    task1_items = TASK1_ITEMS_PER_CELL * len(models) * len(cells)

    def by_cell(key, weight_index):
        return _weighted(
            [
                (CELL_SHARES[cell][key], cell_counts[weight_index])
                for _, cell, cell_counts in counts
            ]
        )

    row = {
        "model": model,
        "dataset": dataset,
        "variant": variant,
        "n_models_pooled": len(models),
        "n_cells_pooled": len(cells),
        "bootstrap_samples": 1000,
        "bootstrap_seed": 20260518,
        "task1_ci_cluster_field": "batch_id",
        "task2_ci_cluster_field": "batch_id",
        "task3_ci_cluster_field": "batch_id",
        "task1_n": task1_items,
        "task1_accuracy": _weighted(
            [
                (CELL_SHARES[cell]["task1_accuracy"], TASK1_ITEMS_PER_CELL)
                for _, cell, _ in counts
            ]
        ),
        "task2_n": readable,
        "task2_n_readable": readable,
        "task2_weak_n": weak_readable,
        "task2_weak_n_readable": weak_readable,
        "task3_n": strict,
        "task3_strict_joined_n": strict,
        "task3_strict_unaudited_n": 0,
        "task2_strict_agreement_incomplete_excluded": 0,
    }
    for name, key in (
        ("task2_meaning_variation_auroc", "meaning_variation_auroc"),
        ("task2_verbalized_confidence_auroc", "verbalized_confidence_auroc"),
    ):
        value = by_cell(key, 2)
        row.update(
            {
                name: value,
                f"{name}_n": readable,
                f"{name}_n_positive": strict,
                f"{name}_ci_low": value - 0.03,
                f"{name}_ci_high": value + 0.03,
                f"{name}_seed_ci_low": value - 0.02,
                f"{name}_seed_ci_high": value + 0.02,
            }
        )

    upgrade = _weighted(
        [(_rq_model(m, cell)[0], TASK1_ITEMS_PER_CELL) for m, cell, _ in counts]
    )
    recall = _weighted(
        [(_rq_model(m, cell)[1], cell_counts[0]) for m, cell, cell_counts in counts]
    )
    rates = {
        "task1_unsupported_acceptance_90": (
            round(upgrade * task1_items),
            task1_items,
        ),
        "task2_no_cue": (0, readable),
        "task2_strict_strengthening": (strict, readable),
        "task2_broad_strengthening": (broad, readable),
        "task2_weak_strict_strengthening": (weak_strict, weak_readable),
        "task2_weak_strict_high_conf_90": (weak_strict, weak_readable),
        "task2_strict_high_conf_90": (by_cell("high_conf", 0) * strict, strict),
        "task2_strict_agreement": (by_cell("agreement", 0) * strict, strict),
        "task3_strict_flagged": (recall * strict, strict),
        "task3_strict_called_preserved": (
            by_cell("called_preserved", 0) * strict,
            strict,
        ),
    }
    for name, (numerator, denominator) in rates.items():
        rate = numerator / denominator if denominator else ""
        row.update(
            {
                f"{name}_n": round(numerator),
                f"{name}_denominator": denominator,
                f"{name}_rate": rate,
                f"{name}_ci_low": max(0.0, rate - 0.01) if rate != "" else "",
                f"{name}_ci_high": min(1.0, rate + 0.01) if rate != "" else "",
                f"{name}_seed_ci_low": max(0.0, rate - 0.005) if rate != "" else "",
                f"{name}_seed_ci_high": min(1.0, rate + 0.005) if rate != "" else "",
            }
        )
    return row


def _rq_rows(models):
    """Per-model rows, per-cell rows, and the pooled row."""
    rows = [_rq_row(model, "all", "all", [model], CELLS) for model in models]
    rows += [
        _rq_row("all", dataset, variant, models, [(dataset, variant)])
        for dataset, variant in CELLS
    ]
    rows.append(_rq_row("all", "all", "all", models, CELLS))
    return rows


def write_fixture(outputs_dir: Path, models: list[str] | None = None) -> None:
    """Write a complete miniature artifact set under ``outputs_dir``."""
    models = models or ["glm-5.1", "kit.gemma4-31b-it"]
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # The frozen design the seed and item counts are cross-checked against:
    # 16 rows x 2000 items / (2 models x 2 cells) / 4 conditions = 2000 seeds.
    for dataset in ["mlm_tapt", "nice"]:
        suffix = "" if dataset == "nice" else f"_{dataset}"
        (outputs_dir / f"benchmark_manifest{suffix}.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "dataset_id": dataset,
                        "seed_count": 2000,
                        "source_modalities": CONDITIONS,
                    }
                }
            ),
            encoding="utf-8",
        )

    eu.write_csv_rows(outputs_dir / exporter.MODALITY_TABLE, _modality_rows(models))
    eu.write_csv_rows(
        outputs_dir / exporter.PER_MODEL_HEADLINE, _per_model_headline_rows(models)
    )
    eu.write_csv_rows(
        outputs_dir / exporter.HEADLINE_BOOTSTRAP_CI,
        [
            {
                "headline_key": "strict_text_strengthening",
                "value": 3440 / 29600,
                "ci_low": 0.11,
                "ci_high": 0.122,
                "n_numerator": 3440,
                "n_denominator": 29600,
            },
            {
                "headline_key": "broad_text_strengthening",
                "value": 6000 / 29600,
                "ci_low": 0.19,
                "ci_high": 0.215,
                "n_numerator": 6000,
                "n_denominator": 29600,
            },
        ],
    )
    eu.write_csv_rows(
        outputs_dir / exporter.HEADLINE_METRICS,
        [
            {
                "headline_key": "strict_text_strengthening",
                "value_pooled": 3440 / 29600,
                "value_macro_over_cells": 0.1167,
                "per_cell_values": "0.1347|0.0987",
            },
            {
                "headline_key": "broad_text_strengthening",
                "value_pooled": 6000 / 29600,
                "value_macro_over_cells": 0.2031,
                "per_cell_values": "0.2194|0.1868",
            },
        ],
    )
    eu.write_csv_rows(
        outputs_dir / exporter.TASK2_TEXT_DRIFT,
        [
            {
                "dataset": "mlm_tapt",
                "variant": "must",
                "n": 16000,
                "label_accuracy": 1.0,
                "label_over_commitment": 0.0,
                "text_modality_parse_coverage": 0.9,
            },
            {
                "dataset": "nice",
                "variant": "shall",
                "n": 16000,
                "label_accuracy": 1.0,
                "label_over_commitment": 0.0,
                "text_modality_parse_coverage": 0.95,
            },
        ],
    )
    eu.write_csv_rows(
        outputs_dir / exporter.CONFIDENCE_STABILITY,
        [
            {
                "dataset": "mlm_tapt",
                "variant": "must",
                "strict_text_oc_n": 1940,
                "strict_text_oc_conf_ge_90": 1.0,
                "strict_text_oc_unanimous_modality_samples": 1.0,
            },
            {
                "dataset": "nice",
                "variant": "shall",
                "strict_text_oc_n": 1500,
                "strict_text_oc_conf_ge_90": 0.9,
                "strict_text_oc_unanimous_modality_samples": 1.0,
            },
        ],
    )
    eu.write_csv_rows(outputs_dir / exporter.PER_MODEL_RQ, _rq_rows(models))
    eu.write_csv_rows(
        outputs_dir / exporter.PROBE_GRID,
        [
            {
                "feature_backend": backend,
                "text_variant": variant,
                "group_mode": group,
                "scope": scope,
                "target": target,
                "model": "hgb",
                "auroc_mean": auroc,
                "auprc_mean": auprc,
            }
            for backend, variant, group, scope, target, auroc, auprc in PROBE_ROWS
        ],
    )

    # Written last: `stale_provenance` compares mtimes against this file.
    eu.write_json(
        outputs_dir / exporter.SNAPSHOT_PROVENANCE,
        {"models_hosted": exporter.order_models(models), "models_local": []},
    )


def _drop_column(path: Path, column: str) -> None:
    rows = eu.read_csv_rows(path)
    for row in rows:
        row.pop(column, None)
    eu.write_csv_rows(path, rows)


class ExporterFixtureTest(unittest.TestCase):
    """Shared temp-directory fixture: one exporter run over the mini artifacts."""

    def setUp(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.outputs = self.root / "outputs"
        self.output = self.root / "numbers.tex"
        write_fixture(self.outputs)

    def run_exporter(self, *extra: str) -> tuple[int, str]:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = exporter.main(
                [
                    "--outputs-dir",
                    str(self.outputs),
                    "--output",
                    str(self.output),
                    *extra,
                ]
            )
        return code, stdout.getvalue()

    def export(self, *extra: str) -> tuple[dict[str, str], str]:
        code, printed = self.run_exporter(*extra)
        self.assertEqual(code, 0, printed)
        return exporter.parse_macros(self.output.read_text(encoding="utf-8")), printed


class HeadlineMacroTest(ExporterFixtureTest):
    def test_pooled_strict_and_broad_rates(self) -> None:
        macros, _ = self.export()
        # 3440 / 29600 = 11.62%, 6000 / 29600 = 20.27%.
        self.assertEqual(macros["numStrictOverall"], "11.6")
        self.assertEqual(macros["numBroadOverall"], "20.3")
        self.assertEqual(macros["numStrictOverallCI"], "11.0--12.2")
        self.assertEqual(macros["numBroadOverallCI"], "19.0--21.5")

    def test_counts_use_the_manuscript_thousands_convention(self) -> None:
        macros, _ = self.export()
        # Five digits and up get a separator; four-digit counts do not.
        self.assertEqual(macros["numStrictOverallDen"], "29,600")
        self.assertEqual(macros["numTaskTwoAnswers"], "32,000")
        self.assertEqual(macros["numItems"], "16,000")
        self.assertEqual(macros["numStrictOverallNum"], "3440")
        self.assertEqual(macros["numBroadOverallNum"], "6000")
        self.assertEqual(macros["numItemsPerCell"], "8000")

    def test_cell_and_model_ranges(self) -> None:
        macros, _ = self.export()
        # Cells: 1940/14400 = 13.5% and 1500/15200 = 9.9%.
        self.assertEqual(macros["numStrictOverallRange"], "9.9--13.5")
        self.assertEqual(macros["numBroadOverallRange"], "18.7--21.9")
        # Models: 1540/13600 = 11.3% and 1900/16000 = 11.9%.
        self.assertEqual(macros["numStrictModelsRange"], "11.3--11.9")
        self.assertEqual(macros["numBroadModelsRange"], "20.0--20.6")

    def test_no_cue_share_pooled_cells_and_models(self) -> None:
        macros, _ = self.export()
        # (32000 - 29600) / 32000 = 7.5%.
        self.assertEqual(macros["numNoCueShare"], "7.5")
        self.assertEqual(macros["numNoCueShareRange"], "5.0--10.0")
        self.assertEqual(macros["numNoCueShareModels"], "0.0--15.0")
        # Deprecated alias kept for the submitted draft's macro name.
        self.assertEqual(macros["numNoModalShare"], macros["numNoCueShare"])

    def test_task1_control_and_per_model_upgrade_range(self) -> None:
        macros, _ = self.export()
        self.assertEqual(macros["numTaskOneAcc"], "98.0")
        self.assertEqual(macros["numTaskOneAccRange"], "97.0--99.0")
        self.assertEqual(macros["numTaskOneUpgrade"], "1.5")
        self.assertEqual(macros["numTaskOneUpgradeCI"], "0.5--2.5")
        self.assertEqual(macros["numTaskOneUpgradeRange"], "1.0--2.0")
        self.assertEqual(macros["numTaskOneUpgradeModels"], "0.0--3.0")
        self.assertEqual(macros["numLabelAcc"], "100.0")
        self.assertEqual(macros["numLabelStrengthening"], "0.0")

    def test_weak_intent_headline_uses_the_named_cell(self) -> None:
        macros, _ = self.export()
        self.assertEqual(macros["numWeakStrict"], "38.9")
        self.assertEqual(macros["numWeakStrictRange"], "30.0--38.9")
        self.assertEqual(macros["numWeakStrictDen"], "3600")
        self.assertEqual(macros["numWeakStrictNum"], "1400")
        # Per model, pooled over cells by the readable weak denominator.
        self.assertEqual(macros["numWeakStrictModelsRange"], "27.6--40.0")

    def test_uncertainty_and_detector_macros(self) -> None:
        macros, _ = self.export()
        # Pooled over the rows the bootstrap resamples, not a macro over cells.
        self.assertEqual(macros["numHighConfShare"], "98.1")
        self.assertEqual(macros["numHighConfShareRange"], "97.0--99.0")
        self.assertEqual(macros["numHighConfModelsRange"], "97.8--98.4")
        self.assertEqual(macros["numSampleAgreement"], "63.3")
        self.assertEqual(macros["numMeaningVarAUROC"], "0.751")
        self.assertEqual(macros["numMeaningVarAUROCRange"], "0.700--0.800")
        self.assertEqual(macros["numMeaningVarAUROCModels"], "0.750--0.753")
        self.assertEqual(macros["numVerbConfAUROC"], "0.791")
        self.assertEqual(macros["numEmbGlobalAUROC"], "0.700")
        self.assertEqual(macros["numEmbGlobalAUPRC"], "0.170")
        self.assertEqual(macros["numEmbSampledAUROC"], "0.744")
        self.assertEqual(macros["numEmbSampledAUPRC"], "0.276")
        self.assertEqual(macros["numEmbWithinRecommended"], "0.613")
        self.assertEqual(macros["numEmbWithinOptional"], "0.612")
        self.assertEqual(macros["numEmbWithinWeak"], "0.620")
        self.assertEqual(macros["numEmbLeakControl"], "0.822")
        self.assertEqual(macros["numEmbContextLevel"], "0.838")
        self.assertEqual(macros["numEmbContextDataset"], "0.728")

    def test_blind_audit_macros(self) -> None:
        macros, _ = self.export()
        self.assertEqual(macros["numBlindRecall"], "61.9")
        self.assertEqual(macros["numBlindRecallCI"], "60.9--62.9")
        self.assertEqual(macros["numBlindRecallRange"], "60.0--63.4")
        self.assertEqual(macros["numBlindMissed"], "45.7")
        self.assertEqual(macros["numBlindMissedRange"], "44.0--48.0")
        self.assertEqual(macros["numBlindRecallModelsRange"], "44.2--76.3")

    def test_requirement_word_counts(self) -> None:
        macros, _ = self.export()
        self.assertEqual(macros["numWeakWords"], "20.0")
        self.assertEqual(macros["numOtherWords"], "15.0")

    def test_benchmark_shape_macros(self) -> None:
        macros, _ = self.export()
        self.assertEqual(macros["numModels"], "2")
        self.assertEqual(macros["numCells"], "2")
        self.assertEqual(macros["numConditions"], "4")
        self.assertEqual(macros["numSeeds"], "4000")
        self.assertEqual(macros["numSeedsPerDataset"], "2000")
        self.assertEqual(macros["numTaskTwoReadable"], "29,600")
        self.assertEqual(macros["numBatchSize"], "16")
        self.assertEqual(macros["numStochasticSamples"], "5")


class PerModelMacroTest(ExporterFixtureTest):
    def test_per_model_condition_macros(self) -> None:
        macros, _ = self.export()
        # glm-5.1 recommended: (160 + 180) / (1600 + 1800) = 10.0% strict.
        self.assertEqual(macros["numStrictGlmFiveOneRecommended"], "10.0")
        self.assertEqual(macros["numBroadGlmFiveOneRecommended"], "20.0")
        self.assertEqual(macros["numStrictGlmFiveOneOptional"], "7.6")
        self.assertEqual(macros["numBroadGlmFiveOneOptional"], "12.4")
        self.assertEqual(macros["numStrictGlmFiveOneWeak"], "27.6")
        self.assertEqual(macros["numBroadGemmaWeak"], "55.0")
        self.assertEqual(macros["numStrictGemmaMandatory"], "0.0")

    def test_per_model_pooled_and_no_cue_macros(self) -> None:
        macros, _ = self.export()
        self.assertEqual(macros["numStrictGlmFiveOne"], "11.3")
        self.assertEqual(macros["numBroadGlmFiveOne"], "20.6")
        self.assertEqual(macros["numStrictGemma"], "11.9")
        self.assertEqual(macros["numNoCueGlmFiveOne"], "15.0")
        self.assertEqual(macros["numNoCueGemma"], "0.0")

    def test_table_three_rows_shape_and_order(self) -> None:
        macros, _ = self.export()
        body = macros["numTableThreeRows"]
        rows = [row.strip() for row in body.split(r"\\") if row.strip()]
        self.assertEqual(len(rows), 2)
        # Hosted cohort order, not alphabetical by display label.
        self.assertEqual(
            rows[0],
            "GLM-5.1 & 15.0 & 10.0 & 20.0 & 7.6 & 12.4 & 27.6 & 50.0 & 11.3 & 20.6",
        )
        self.assertTrue(rows[1].startswith("Gemma-4-31B &"), rows[1])
        for row in rows:
            self.assertEqual(len(row.split("&")), 10, row)
        self.assertTrue(body.rstrip().endswith(r"\\"))

    def test_unknown_model_gets_a_transliterated_key_and_raw_label(self) -> None:
        self.assertEqual(
            exporter.model_key("mistral-7b-instruct"), "MistralSevenBInstruct"
        )
        self.assertEqual(
            exporter.model_key("llama.cpp/qwen3-32b"), "LlamaCppQwenThreeThreeTwoB"
        )
        self.assertEqual(
            exporter.model_label("mistral-7b-instruct"), "mistral-7b-instruct"
        )
        # Every cohort key is the one the manuscript already uses.
        self.assertEqual(exporter.model_key("glm-4.5-air"), "GlmFourFiveAir")
        self.assertEqual(exporter.model_key("glm-4.7"), "GlmFourSeven")
        self.assertEqual(exporter.model_key("glm-5"), "GlmFive")
        self.assertEqual(exporter.model_key("glm-5-turbo"), "GlmFiveTurbo")
        self.assertEqual(exporter.model_key("glm-5.1"), "GlmFiveOne")
        self.assertEqual(exporter.model_key("kit.gemma4-31b-it"), "Gemma")

    def test_unknown_model_is_exported_after_the_hosted_cohort(self) -> None:
        write_fixture(self.outputs, models=["kit.gemma4-31b-it", "glm-5.1"])
        macros, _ = self.export()
        rows = [
            row.strip()
            for row in macros["numTableThreeRows"].split(r"\\")
            if row.strip()
        ]
        self.assertTrue(rows[0].startswith("GLM-5.1 &"), rows[0])


class OutputShapeTest(ExporterFixtureTest):
    def test_header_records_version_time_and_source_hashes(self) -> None:
        self.export()
        text = self.output.read_text(encoding="utf-8")
        self.assertIn("GENERATED FILE", text)
        self.assertIn(exporter.EXPORTER_VERSION, text)
        self.assertIn("scripts/export_paper_numbers.py", text)
        self.assertIn(eu.sha256_file(self.outputs / exporter.MODALITY_TABLE), text)
        self.assertIn(exporter.MODALITY_TABLE, text)

    def test_every_submitted_draft_macro_is_still_emitted(self) -> None:
        macros, _ = self.export()
        missing = sorted(set(exporter.SUBMITTED_DRAFT_MACROS) - set(macros))
        self.assertEqual(missing, [])

    def test_macro_names_are_unique_and_letters_only(self) -> None:
        macros, _ = self.export()
        for name in macros:
            self.assertTrue(name.isalpha(), name)
        text = self.output.read_text(encoding="utf-8")
        self.assertEqual(text.count("\\newcommand{"), len(macros))

    def test_seed_clustered_cis_are_optional(self) -> None:
        macros, _ = self.export()
        self.assertNotIn("numStrictOverallSeedCI", macros)

        path = self.outputs / exporter.HEADLINE_BOOTSTRAP_CI
        rows = eu.read_csv_rows(path)
        for row in rows:
            row["seed_ci_low"] = "0.10"
            row["seed_ci_high"] = "0.13"
        eu.write_csv_rows(path, rows)
        macros, _ = self.export()
        self.assertEqual(macros["numStrictOverallSeedCI"], "10.0--13.0")
        self.assertEqual(macros["numBroadOverallSeedCI"], "10.0--13.0")

    def test_diff_summary_reports_added_changed_and_removed(self) -> None:
        self.export()
        text = self.output.read_text(encoding="utf-8")
        text = text.replace(
            "\\newcommand{\\numStrictOverall}{11.6}",
            "\\newcommand{\\numStrictOverall}{9.9}",
        ).replace(
            "\\newcommand{\\numBroadOverall}{20.3}",
            "\\newcommand{\\numLegacyThing}{1.0}",
        )
        self.output.write_text(text, encoding="utf-8")

        _, printed = self.export()
        self.assertIn("changed  numStrictOverall", printed)
        self.assertIn("9.9 -> 11.6", printed)
        self.assertIn("added    numBroadOverall", printed)
        self.assertIn("removed  numLegacyThing", printed)

    def test_default_output_never_touches_the_manuscript(self) -> None:
        self.assertEqual(exporter.DEFAULT_OUTPUT_NAME, "paper_numbers.tex")


class ValidationTest(ExporterFixtureTest):
    def test_missing_artifact_fails_closed(self) -> None:
        (self.outputs / exporter.PER_MODEL_RQ).unlink()
        with self.assertRaises(SystemExit) as caught:
            self.run_exporter()
        self.assertIn(exporter.PER_MODEL_RQ, str(caught.exception))
        self.assertFalse(self.output.exists())

    def test_missing_column_fails_closed(self) -> None:
        _drop_column(
            self.outputs / exporter.MODALITY_TABLE, "strict_strengthening_denominator"
        )
        with self.assertRaises(SystemExit) as caught:
            self.run_exporter()
        message = str(caught.exception)
        self.assertIn(exporter.MODALITY_TABLE, message)
        self.assertIn("strict_strengthening_denominator", message)

    def test_missing_rq_column_fails_closed(self) -> None:
        _drop_column(
            self.outputs / exporter.PER_MODEL_RQ,
            "task2_weak_strict_high_conf_90_rate",
        )
        with self.assertRaises(SystemExit) as caught:
            self.run_exporter()
        self.assertIn("task2_weak_strict_high_conf_90_rate", str(caught.exception))

    def test_rq_table_without_a_pooled_row_fails_closed(self) -> None:
        path = self.outputs / exporter.PER_MODEL_RQ
        rows = [
            row
            for row in eu.read_csv_rows(path)
            if (row["model"], row["dataset"]) != ("all", "all")
        ]
        eu.write_csv_rows(path, rows)
        with self.assertRaises(SystemExit) as caught:
            self.run_exporter()
        self.assertIn("pooled row", str(caught.exception))

    def test_ambiguous_probe_selection_fails_closed(self) -> None:
        path = self.outputs / exporter.PROBE_GRID
        rows = eu.read_csv_rows(path)
        rows.append(dict(rows[0]))
        eu.write_csv_rows(path, rows)
        with self.assertRaises(SystemExit) as caught:
            self.run_exporter()
        self.assertIn(exporter.PROBE_GRID, str(caught.exception))

    def test_missing_headline_cell_fails_closed(self) -> None:
        path = self.outputs / exporter.PER_MODEL_RQ
        rows = [
            row
            for row in eu.read_csv_rows(path)
            if not (row["model"] == "all" and row["dataset"] == "mlm_tapt")
        ]
        eu.write_csv_rows(path, rows)
        with self.assertRaises(SystemExit) as caught:
            self.run_exporter()
        self.assertIn("mlm_tapt/must", str(caught.exception))

    def test_stale_provenance_warns_but_still_writes(self) -> None:
        provenance = self.outputs / exporter.SNAPSHOT_PROVENANCE
        eu.write_json(provenance, {"generated_at_utc": "2020-01-01T00:00:00Z"})
        os.utime(provenance, (10_000.0, 10_000.0))
        _, printed = self.export()
        self.assertIn("warning", printed.lower())
        self.assertIn(exporter.SNAPSHOT_PROVENANCE, printed)


class FormatterGuardTest(unittest.TestCase):
    """`_number` returns NaN for a blank cell, so the formatters have to say so."""

    def test_a_non_finite_count_names_itself_instead_of_raising_value_error(
        self,
    ) -> None:
        with self.assertRaises(exporter.PaperNumbersError):
            exporter.fmt_count(float("nan"))

    def test_nan_macros_finds_every_rendered_non_finite_value(self) -> None:
        found = exporter.nan_macros(
            {
                "numA": "11.6",
                "numB": "nan--nan",
                "numC": "NaN",
                "numD": "0.707",
                "numE": "kit.nanogpt & 12 & 3.4",
            }
        )
        # numE holds a model id, not a value: "nan" inside a word is not a number.
        self.assertEqual(found, ["numB", "numC"])


class ClusterProvenanceTest(ExporterFixtureTest):
    """The annotation must come from the artifact, never from a fixed string."""

    def _set_cluster_field(self, value: str | None) -> None:
        path = self.outputs / exporter.HEADLINE_BOOTSTRAP_CI
        rows = eu.read_csv_rows(path)
        for row in rows:
            if value is None:
                row.pop("ci_cluster_field", None)
            else:
                row["ci_cluster_field"] = value
        eu.write_csv_rows(path, rows)

    def _annotation(self, macro: str) -> str:
        for line in self.output.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"\\newcommand{{\\{macro}}}"):
                return line.split("%", 1)[1].strip() if "%" in line else ""
        raise AssertionError(f"{macro} not found")

    def test_a_request_clustered_snapshot_says_so(self) -> None:
        self._set_cluster_field(eu.DEFAULT_BOOTSTRAP_CLUSTER_FIELD)
        self.export()
        self.assertEqual(
            self._annotation("numStrictOverall"), "pooled, request-clustered CI"
        )
        self.assertEqual(
            self._annotation("numBroadOverall"), "pooled, request-clustered CI"
        )

    def test_a_slice_that_fell_back_to_the_seed_is_not_mislabelled(self) -> None:
        self._set_cluster_field(eu.BOOTSTRAP_CLUSTER_FALLBACK_FIELD)
        self.export()
        self.assertEqual(
            self._annotation("numStrictOverall"), "pooled, seed-clustered CI"
        )

    def test_a_snapshot_without_the_column_claims_nothing(self) -> None:
        self._set_cluster_field(None)
        self.export()
        self.assertEqual(
            self._annotation("numStrictOverall"), "pooled, resampling unit not recorded"
        )

    def test_an_unknown_unit_is_reported_verbatim(self) -> None:
        self._set_cluster_field("run_id")
        self.export()
        self.assertEqual(
            self._annotation("numStrictOverall"), "pooled, run_id-clustered CI"
        )


class StrictModeTest(ExporterFixtureTest):
    """--strict is what the manuscript path uses: never write a doubtful number."""

    def _blank_headline_cis(self) -> None:
        path = self.outputs / exporter.HEADLINE_BOOTSTRAP_CI
        rows = eu.read_csv_rows(path)
        for row in rows:
            row["ci_low"] = ""
            row["ci_high"] = ""
        eu.write_csv_rows(path, rows)

    def test_a_blank_ci_column_reaches_numbers_tex_as_nan_by_default(self) -> None:
        """The default path is unchanged, so the warning is the only signal."""
        self._blank_headline_cis()
        code, printed = self.run_exporter()
        self.assertEqual(code, 0, printed)
        macros = exporter.parse_macros(self.output.read_text(encoding="utf-8"))
        self.assertEqual(macros["numStrictOverallCI"], "nan--nan")
        self.assertIn("numStrictOverallCI", printed)
        self.assertIn("non-finite", printed)

    def test_strict_refuses_to_write_a_nan_macro(self) -> None:
        self._blank_headline_cis()
        code, printed = self.run_exporter("--strict")
        self.assertEqual(code, 1)
        self.assertIn("numStrictOverallCI", printed)
        self.assertFalse(self.output.exists(), "nothing may be written under --strict")

    def test_strict_leaves_an_existing_numbers_tex_untouched(self) -> None:
        self.assertEqual(self.run_exporter()[0], 0)
        good = self.output.read_text(encoding="utf-8")
        self._blank_headline_cis()
        self.assertEqual(self.run_exporter("--strict")[0], 1)
        self.assertEqual(self.output.read_text(encoding="utf-8"), good)

    def test_a_clean_fixture_is_silent_under_strict(self) -> None:
        code, printed = self.run_exporter("--strict")
        self.assertEqual(code, 0, printed)
        self.assertNotIn("warning:", printed)


class BenchmarkDesignCrossCheckTest(ExporterFixtureTest):
    """Seed and item counts come from observed rows; the design says what was planned."""

    def _set_seed_count(self, dataset: str, seed_count: int) -> None:
        suffix = "" if dataset == "nice" else f"_{dataset}"
        path = self.outputs / f"benchmark_manifest{suffix}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["metadata"]["seed_count"] = seed_count
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_a_seed_missing_from_every_cell_is_reported(self) -> None:
        # The divisibility guards cannot see this: 15992 answers still divide by
        # 4 model-cell blocks and by 4 conditions.
        self._set_seed_count("nice", 2001)
        macros, printed = self.export()
        self.assertEqual(macros["numSeedsPerDataset"], "2000")
        self.assertIn("nice contributes 2000 seeds", printed)
        self.assertIn("declares 2001", printed)

    def test_a_seed_mismatch_is_fatal_under_strict(self) -> None:
        self._set_seed_count("mlm_tapt", 1999)
        code, printed = self.run_exporter("--strict")
        self.assertEqual(code, 1)
        self.assertIn("mlm_tapt", printed)

    def test_an_absent_manifest_says_the_counts_are_unchecked(self) -> None:
        (self.outputs / "benchmark_manifest.json").unlink()
        _, printed = self.export()
        self.assertIn("no benchmark manifest for dataset 'nice'", printed)
        self.assertIn("unchecked", printed)

    def test_conditions_are_checked_against_the_manifest(self) -> None:
        path = self.outputs / "benchmark_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["metadata"]["source_modalities"] = ["mandatory", "recommended"]
        path.write_text(json.dumps(manifest), encoding="utf-8")
        _, printed = self.export()
        self.assertIn("source conditions", printed)

    def test_a_blank_item_count_is_fatal_and_names_the_artifact(self) -> None:
        """Counts never degrade to a warning: they size every derived number."""
        path = self.outputs / exporter.MODALITY_TABLE
        rows = eu.read_csv_rows(path)
        rows[0]["n_items"] = ""
        eu.write_csv_rows(path, rows)
        with self.assertRaises(SystemExit) as caught:
            self.run_exporter()
        self.assertIn(exporter.MODALITY_TABLE, str(caught.exception))
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()


class RqTableMacroTest(ExporterFixtureTest):
    """`\\numTableRqOneRows` / `\\numTableRqTwoThreeRows`: the two table bodies."""

    def _rows(self, body: str) -> list[str]:
        return [line.strip() for line in body.splitlines() if line.strip()]

    def test_rq_one_body_has_the_manuscript_column_order(self) -> None:
        macros, _ = self.export()
        rows = self._rows(macros["numTableRqOneRows"])

        self.assertEqual(rows[0], r"\multicolumn{8}{@{}l}{\textit{Hosted}} \\")
        self.assertEqual(
            rows[1],
            "GLM-5.1 & 2000 & 3.0 [2.0, 4.0] & 13,600 & 11.3 [10.3, 12.3] & "
            r"20.6 [19.6, 21.6] & 3400 & 27.6 [26.6, 28.6] \\",
        )
        # Model, Task 1 N and rate, readable N, strict, broad, weak N, weak strict.
        for row in rows[1:2] + rows[-1:]:
            self.assertEqual(row.count("&"), 7)
        self.assertTrue(rows[-1].startswith("All models &"))
        self.assertEqual(rows[-2], r"\midrule")

    def test_rq_two_three_body_has_the_manuscript_column_order(self) -> None:
        macros, _ = self.export()
        rows = self._rows(macros["numTableRqTwoThreeRows"])

        self.assertEqual(rows[0], r"\multicolumn{6}{@{}l}{\textit{Hosted}} \\")
        # Model, strengthened N, confidence, AUROC, flagged, called preserved.
        self.assertEqual(rows[1].count("&"), 5)
        self.assertIn("0.7", rows[1])
        self.assertTrue(rows[-1].startswith("All models &"))

    def test_hosted_order_follows_the_recorded_cohort(self) -> None:
        write_fixture(self.outputs, models=["kit.gemma4-31b-it", "glm-5.1"])
        eu.write_json(
            self.outputs / exporter.SNAPSHOT_PROVENANCE,
            {"models_hosted": ["kit.gemma4-31b-it", "glm-5.1"], "models_local": []},
        )
        macros, _ = self.export()
        rows = self._rows(macros["numTableRqOneRows"])

        # The provenance order wins over the pinned default cohort.
        self.assertTrue(rows[1].startswith("Gemma-4-31B &"), rows[1])
        self.assertTrue(rows[2].startswith("GLM-5.1 &"), rows[2])

    def test_local_models_are_their_own_group_with_escaped_labels(self) -> None:
        models = ["glm-5.1", "qwen/qwen3.5-9b"]
        write_fixture(self.outputs, models=models)
        eu.write_json(
            self.outputs / exporter.SNAPSHOT_PROVENANCE,
            {"models_hosted": ["glm-5.1"], "models_local": ["qwen/qwen3.5-9b"]},
        )
        macros, _ = self.export()
        rows = self._rows(macros["numTableRqOneRows"])

        self.assertTrue(rows[1].startswith("GLM-5.1 &"))
        self.assertEqual(
            rows[3], r"\multicolumn{8}{@{}l}{\textit{Local (llama.cpp)}} \\"
        )
        self.assertTrue(rows[4].startswith("qwen/qwen3.5-9b &"), rows[4])
        self.assertNotIn(r"\placeholder", macros["numTableRqOneRows"])

    def test_local_group_is_a_placeholder_row_until_a_local_model_runs(self) -> None:
        macros, _ = self.export()
        rows = self._rows(macros["numTableRqOneRows"])

        placeholder = next(row for row in rows if r"\placeholder" in row)
        self.assertEqual(placeholder.count("&"), 7)
        self.assertTrue(placeholder.startswith(r"\placeholder{model}"))

    def test_degenerate_and_absent_intervals_render_a_dash(self) -> None:
        self.assertEqual(exporter.fmt_pct_ci(0.086, 0.076, 0.096), "8.6 [7.6, 9.6]")
        self.assertEqual(exporter.fmt_pct_ci(0.0, 0.0, 0.0), "0.0 ---")
        # Bounds that differ by less than the printed precision are not a width.
        self.assertEqual(exporter.fmt_pct_ci(0.5, 0.50001, 0.50004), "50.0 ---")
        self.assertEqual(exporter.fmt_pct_ci(float("nan"), 0.0, 1.0), "---")
        self.assertEqual(
            exporter.fmt_auroc_ci(0.7685, 0.742, 0.791), "0.768 [0.742, 0.791]"
        )
        self.assertEqual(exporter.fmt_auroc_ci(float("nan"), 0.0, 1.0), "---")

    def test_pooled_rates_are_cross_checked_against_the_headline_snapshot(self) -> None:
        path = self.outputs / exporter.PER_MODEL_RQ
        rows = eu.read_csv_rows(path)
        for row in rows:
            if (row["model"], row["dataset"]) == ("all", "all"):
                row["task2_strict_strengthening_rate"] = "0.42"
        eu.write_csv_rows(path, rows)

        _, printed = self.export()

        self.assertIn("disagrees with", printed)
        self.assertIn(exporter.HEADLINE_BOOTSTRAP_CI, printed)
