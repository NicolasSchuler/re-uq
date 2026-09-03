"""Tests for the paper-facing metric, agreement, bloat, and export code paths.

Covers the text-modality negation guard, the repeated-sample agreement
coverage guard, the coverage-adjusted text over-commitment bounds, the
answer-length/bloat metrics, the run-matrix CI annotation, and
``scripts/export_paper_tables.py`` end to end on a synthetic two-model cell.
"""

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scripts import aggregate_paper_headline_metrics as agg
from scripts import compare_run_matrix as compare_matrix
from scripts import eval_utils as eu
from scripts import export_paper_tables as export
from scripts import generate_evaluation_analysis as analysis_cli


def _seeds():
    return [
        {
            "seed_id": f"S{index:04d}",
            "source_dataset": "NICE",
            "original_requirement": "The system shall export reports.",
            "capability_text_final": f"export report set {index}",
        }
        for index in range(1, 7)
    ]


def _task2_raw(
    item,
    *,
    model,
    run_id,
    requirement,
    sample_kind="deterministic",
    sample_index=0,
    parse_status="ok",
):
    return {
        "run_id": run_id,
        "model": model,
        "task": "task2",
        "item_id": item["item_id"],
        "seed_id": item["seed_id"],
        "source_modality": item["source_modality"],
        "sample_index": sample_index,
        "sample_kind": sample_kind,
        "confidence_scale": "0_1",
        "prompt_version": "v2-conf01",
        "raw_text": "",
        "parsed_json": (
            {
                "requirement": requirement,
                "modality": item["task2_gold_modality"],
                "confidence": 0.95,
            }
            if parse_status == "ok"
            else None
        ),
        "parse_status": parse_status,
        "error": "",
    }


class TextModalityDetectorTest(unittest.TestCase):
    def test_negated_modals_are_not_a_positive_modality(self):
        for text in [
            "The system must not export reports.",
            "The system shall never export reports.",
            "The system can't export reports.",
            "The system cannot export reports.",
            "The system shouldn't export reports.",
            "The system mustn't export reports.",
        ]:
            with self.subTest(text=text):
                diagnostic = eu.requirement_text_modality_diagnostic(text)
                self.assertEqual(diagnostic["text_modality"], "negated")
                self.assertEqual(diagnostic["text_modality_basis"], "negated_modal")

    def test_negated_modal_never_strengthens_and_is_excluded_from_strict(self):
        fields = eu.text_modality_fields(
            "The system must not export reports.", "nice_to_have", "nice_to_have", 0.99
        )
        self.assertEqual(fields["text_modality"], "negated")
        self.assertEqual(fields["text_modality_parse_status"], "unknown")
        self.assertFalse(fields["text_overcommit"])
        self.assertFalse(fields["strict_text_overcommit"])
        self.assertFalse(fields["text_high_conf_overcommit_90"])

    def test_multi_modal_sentence_keeps_priority_and_sets_flag(self):
        diagnostic = eu.requirement_text_modality_diagnostic(
            "The system must export reports and may archive them."
        )
        self.assertEqual(diagnostic["text_modality"], "mandatory")
        self.assertEqual(diagnostic["text_modality_basis"], "explicit_modal")
        self.assertTrue(diagnostic["text_modality_multi_modal"])
        self.assertEqual(diagnostic["text_modality_modals_found"], ["may", "must"])

    def test_single_modal_is_not_flagged_multi_modal(self):
        diagnostic = eu.requirement_text_modality_diagnostic(
            "The system SHOULD export reports."
        )
        self.assertEqual(diagnostic["text_modality"], "recommended")
        self.assertFalse(diagnostic["text_modality_multi_modal"])
        self.assertEqual(diagnostic["text_modality_modals_found"], ["should"])

    def test_heuristic_system_verb_case_is_unchanged(self):
        diagnostic = eu.requirement_text_modality_diagnostic(
            "The system exports reports."
        )
        self.assertEqual(diagnostic["text_modality"], "mandatory")
        self.assertEqual(diagnostic["text_modality_basis"], "heuristic_system_verb")
        self.assertFalse(diagnostic["text_modality_multi_modal"])
        self.assertEqual(diagnostic["text_modality_modals_found"], [])

    def test_backward_compatible_keys_are_present(self):
        diagnostic = eu.requirement_text_modality_diagnostic("")
        self.assertEqual(
            set(diagnostic),
            {
                "text_modality",
                "text_modality_basis",
                "text_modality_multi_modal",
                "text_modality_modals_found",
            },
        )


class RepeatedSampleAgreementTest(unittest.TestCase):
    def _row(self, item_id, valid_n, total_n, distribution):
        return {
            "task": "task2",
            "uq_method": "modality_consistency",
            "item_id": item_id,
            "valid_n": valid_n,
            "total_n": total_n,
            "stochastic_complete": valid_n == total_n,
            "uncertainty_measure": "variation_ratio",
            "uncertainty_score": 1.0 - max(distribution.values()),
            "label_distribution": eu.label_distribution_json(distribution),
        }

    def test_incomplete_sample_sets_are_excluded_from_agreement(self):
        rows = [
            self._row("a", 5, 5, {"mandatory": 1.0}),
            self._row("b", 5, 5, {"mandatory": 0.6, "optional": 0.4}),
            # valid_n < total_n: excluded from both numerator and denominator.
            self._row("c", 3, 5, {"mandatory": 1.0}),
        ]
        metrics = eu.repeated_sample_agreement_metrics(rows)
        self.assertEqual(metrics["agreement_n_complete"], 2)
        self.assertEqual(metrics["agreement_n_incomplete_excluded"], 1)
        self.assertAlmostEqual(metrics["repeated_sample_unanimity"], 0.5)
        self.assertAlmostEqual(metrics["mean_repeated_sample_agreement"], 0.8)

    def test_all_incomplete_reports_empty_rate_and_excluded_count(self):
        metrics = eu.repeated_sample_agreement_metrics(
            [
                self._row("a", 4, 5, {"mandatory": 1.0}),
                self._row("b", 1, 5, {"optional": 1.0}),
            ]
        )
        self.assertEqual(metrics["repeated_sample_unanimity"], "")
        self.assertEqual(metrics["agreement_n_complete"], 0)
        self.assertEqual(metrics["agreement_n_incomplete_excluded"], 2)

    def test_score_rows_carry_stochastic_complete(self):
        benchmark = eu.build_benchmark_items(_seeds()[:1])
        item = [row for row in benchmark if row["source_modality"] == "nice_to_have"][0]
        raw_rows = [
            _task2_raw(
                item,
                model="m1",
                run_id="r1",
                requirement="The system must export reports.",
            )
        ]
        for sample_index in range(5):
            raw_rows.append(
                _task2_raw(
                    item,
                    model="m1",
                    run_id="r1",
                    requirement="The system must export reports.",
                    sample_kind="stochastic",
                    sample_index=sample_index,
                    parse_status="ok" if sample_index < 4 else "invalid_json",
                )
            )
        scores = eu.build_uq_scores(benchmark, raw_rows)
        stochastic = [
            row for row in scores if row["uq_method"] == "modality_consistency"
        ]
        self.assertEqual(len(stochastic), 1)
        self.assertEqual(stochastic[0]["valid_n"], 4)
        self.assertEqual(stochastic[0]["total_n"], 5)
        self.assertFalse(stochastic[0]["stochastic_complete"])
        metrics = eu.repeated_sample_agreement_metrics(scores)
        self.assertEqual(metrics["agreement_n_complete"], 0)
        self.assertEqual(metrics["agreement_n_incomplete_excluded"], 1)

    def test_min_valid_samples_drops_thin_stochastic_groups(self):
        benchmark = eu.build_benchmark_items(_seeds()[:1])
        item = [row for row in benchmark if row["source_modality"] == "optional"][0]
        raw_rows = [
            _task2_raw(
                item,
                model="m1",
                run_id="r1",
                requirement="The system may export reports.",
                sample_kind="stochastic",
                sample_index=index,
                parse_status="ok" if index < 2 else "invalid_json",
            )
            for index in range(5)
        ]
        self.assertTrue(
            any(
                row["uq_method"] == "modality_consistency"
                for row in eu.build_uq_scores(benchmark, raw_rows)
            )
        )
        guarded = eu.build_uq_scores(benchmark, raw_rows, min_valid_samples=3)
        self.assertFalse(
            any(row["uq_method"] == "modality_consistency" for row in guarded)
        )


class CoverageAdjustedBoundsTest(unittest.TestCase):
    def _row(self, status, *, broad=False, strict=False):
        return {
            "task": "task2",
            "text_modality_parse_status": status,
            "text_overcommit": broad,
            "strict_text_overcommit": strict,
            "y_true": 1,
            "confidence": 0.95,
        }

    def test_bounds_bracket_the_published_rate(self):
        rows = [
            self._row("ok", broad=True, strict=True),
            self._row("ok", broad=True),
            self._row("ok"),
            self._row("unknown"),
        ]
        metrics = eu.text_modality_summary_metrics(rows)
        self.assertAlmostEqual(metrics["text_over_commitment"], 2 / 3)
        self.assertEqual(metrics["text_over_commitment_n_numerator"], 2)
        self.assertEqual(metrics["text_over_commitment_n_denominator"], 3)
        self.assertEqual(metrics["text_over_commitment_n_unknown_excluded"], 1)
        self.assertAlmostEqual(metrics["text_over_commitment_lower_bound"], 0.5)
        self.assertAlmostEqual(metrics["text_over_commitment_upper_bound"], 0.75)
        self.assertAlmostEqual(metrics["strict_text_over_commitment"], 1 / 3)
        self.assertEqual(metrics["strict_text_over_commitment_n_unknown_excluded"], 1)
        self.assertAlmostEqual(metrics["strict_text_over_commitment_lower_bound"], 0.25)
        self.assertAlmostEqual(metrics["strict_text_over_commitment_upper_bound"], 0.5)
        self.assertLessEqual(
            metrics["text_over_commitment_lower_bound"], metrics["text_over_commitment"]
        )
        self.assertGreaterEqual(
            metrics["text_over_commitment_upper_bound"], metrics["text_over_commitment"]
        )

    def test_heuristic_rate_is_reported_next_to_the_broad_metric(self):
        rows = [
            {
                **self._row("ok", broad=True),
                "text_modality_basis": "heuristic_system_verb",
            },
            {**self._row("ok"), "text_modality_basis": "explicit_modal"},
        ]
        metrics = eu.text_modality_summary_metrics(rows)
        self.assertAlmostEqual(metrics["heuristic_text_modality_rate"], 0.5)
        self.assertAlmostEqual(metrics["text_over_commitment"], 0.5)


class LengthBloatMetricsTest(unittest.TestCase):
    def test_score_rows_carry_length_fields(self):
        benchmark = eu.build_benchmark_items(_seeds()[:1])
        item = [row for row in benchmark if row["source_modality"] == "nice_to_have"][0]
        raw = _task2_raw(
            item, model="m1", run_id="r1", requirement="The system must export reports."
        )
        raw["usage_completion_tokens"] = 42
        score = eu.build_uq_scores(benchmark, [raw])[0]
        self.assertEqual(score["requirement_word_count"], 5)
        self.assertEqual(
            score["source_word_count"], eu.word_count(item["source_statement"])
        )
        self.assertAlmostEqual(score["length_ratio"], 5 / score["source_word_count"])
        self.assertEqual(score["completion_tokens"], 42)

    def test_runner_recorded_word_count_wins_over_recomputation(self):
        benchmark = eu.build_benchmark_items(_seeds()[:1])
        item = benchmark[0]
        raw = _task2_raw(
            item, model="m1", run_id="r1", requirement="The system must export reports."
        )
        raw["requirement_word_count"] = 11
        score = eu.build_uq_scores(benchmark, [raw])[0]
        self.assertEqual(score["requirement_word_count"], 11)

    def test_tercile_breakdown_reports_strict_and_broad_per_bucket(self):
        rows = []
        for index in range(9):
            rows.append(
                {
                    "task": "task2",
                    "source_modality": "nice_to_have",
                    "requirement_word_count": index + 1,
                    "source_word_count": 10,
                    "length_ratio": (index + 1) / 10,
                    "text_modality_parse_status": "ok",
                    # Only the shortest tercile strengthens.
                    "text_overcommit": index < 3,
                    "strict_text_overcommit": index < 2,
                }
            )
        metrics = eu.length_bloat_metrics(rows)
        self.assertEqual(metrics["length_tercile_1_n"], 3)
        self.assertAlmostEqual(metrics["length_tercile_1_text_over_commitment"], 1.0)
        self.assertAlmostEqual(
            metrics["length_tercile_1_strict_text_over_commitment"], 2 / 3
        )
        self.assertAlmostEqual(metrics["length_tercile_3_text_over_commitment"], 0.0)
        self.assertIn("t1:n=3", metrics["strengthening_rate_by_length_tercile"])
        self.assertAlmostEqual(metrics["mean_requirement_word_count"], 5.0)
        self.assertAlmostEqual(metrics["mean_length_ratio"], 0.5)
        self.assertAlmostEqual(metrics["mean_requirement_word_count_nice_to_have"], 5.0)
        self.assertIn(
            "nice_to_have=5.00",
            metrics["mean_requirement_word_count_by_source_modality"],
        )

    def test_summary_exposes_length_and_agreement_columns(self):
        benchmark = eu.build_benchmark_items(_seeds()[:2])
        raw_rows = [
            _task2_raw(
                item,
                model="m1",
                run_id="r1",
                requirement="The system must export reports.",
            )
            for item in benchmark
        ]
        summary = eu.metric_summary_by_model_task_method(
            eu.build_uq_scores(benchmark, raw_rows)
        )
        row = [entry for entry in summary if entry["task"] == "task2"][0]
        for field in [
            "mean_requirement_word_count",
            "mean_length_ratio",
            "strengthening_rate_by_length_tercile",
            "mean_requirement_word_count_by_source_modality",
            "agreement_n_complete",
            "agreement_n_incomplete_excluded",
            "text_over_commitment_n_numerator",
            "text_over_commitment_lower_bound",
        ]:
            self.assertIn(field, row)


class RunMatrixCiTest(unittest.TestCase):
    def test_annotate_text_drift_cis_adds_counts_and_bounds(self):
        benchmark = eu.build_benchmark_items(_seeds())
        raw_rows = [
            _task2_raw(
                item,
                model="m1",
                run_id="r1",
                requirement="The system must export reports.",
            )
            for item in benchmark
        ]
        scores = eu.build_uq_scores(benchmark, raw_rows)
        summary = eu.metric_summary_by_model_task_method(scores)
        compare_matrix.annotate_text_drift_cis(summary, scores, bootstrap_samples=25)
        row = [entry for entry in summary if entry["task"] == "task2"][0]
        self.assertGreater(row["strict_text_over_commitment_n_numerator"], 0)
        self.assertEqual(
            row["strict_text_over_commitment_n_denominator"], len(benchmark)
        )
        self.assertLessEqual(
            row["strict_text_over_commitment_ci_low"],
            row["strict_text_over_commitment"],
        )
        self.assertGreaterEqual(
            row["strict_text_over_commitment_ci_high"],
            row["strict_text_over_commitment"],
        )
        for field in [
            "text_over_commitment",
            "text_over_commitment_ci_low",
            "strict_text_over_commitment_ci_high",
        ]:
            self.assertIn(field, compare_matrix.SUMMARY_FIELDS)


class SmokeTreeRoutingTest(unittest.TestCase):
    """Smoke/fake runs live in data/processed/smoke/ and must be opted into."""

    def _write_registry_and_raw(
        self, root: Path, *, smoke: bool, run_id: str, model: str
    ) -> None:
        eu.write_csv_rows(
            eu.run_registry_path(root, "nice", "must", smoke=smoke),
            [
                {
                    "run_id": run_id,
                    "run_group_id": "g1",
                    "model": model,
                    "status": "complete",
                    "provider_id": "p",
                    "profile_id": "p",
                    "dataset_id": "nice",
                    "benchmark_variant": "must",
                }
            ],
        )
        raw_path = eu.model_outputs_raw_path(root, "nice", "must", smoke=smoke)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark = eu.build_benchmark_items(_seeds()[:1])
        with raw_path.open("w", encoding="utf-8") as handle:
            for item in benchmark:
                handle.write(
                    json.dumps(
                        _task2_raw(
                            item,
                            model=model,
                            run_id=run_id,
                            requirement="The system must export reports.",
                        )
                    )
                    + "\n"
                )

    def test_include_smoke_reads_the_smoke_tree(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_registry_and_raw(root, smoke=False, run_id="full-1", model="m1")
            self._write_registry_and_raw(root, smoke=True, run_id="smoke-1", model="m2")

            registry_rows, raw_rows = compare_matrix.load_registry_and_raw_rows(
                root, "nice", "must"
            )
            self.assertEqual({row["run_id"] for row in registry_rows}, {"full-1"})
            self.assertEqual({row["run_id"] for row in raw_rows}, {"full-1"})

            registry_rows, raw_rows = compare_matrix.load_registry_and_raw_rows(
                root, "nice", "must", include_smoke=True
            )
            self.assertEqual(
                {row["run_id"] for row in registry_rows}, {"full-1", "smoke-1"}
            )
            self.assertEqual({row["run_id"] for row in raw_rows}, {"full-1", "smoke-1"})
            # The startswith("smoke-") guard still gates which runs are summarized.
            self.assertEqual(
                {
                    row["run_id"]
                    for row in compare_matrix.completed_registry_rows(
                        registry_rows, "g1", include_smoke=False
                    )
                },
                {"full-1"},
            )
            self.assertEqual(
                {
                    row["run_id"]
                    for row in compare_matrix.completed_registry_rows(
                        registry_rows, "g1", include_smoke=True
                    )
                },
                {"full-1", "smoke-1"},
            )

    def test_analysis_paths_follow_the_run_id_into_the_smoke_tree(self):
        root = Path("/tmp/does-not-need-to-exist")
        smoke_raw = eu.model_outputs_raw_path(root, "nice", "must", run_id="smoke-1")
        full_raw = eu.model_outputs_raw_path(root, "nice", "must", run_id="full-1")
        self.assertEqual(smoke_raw.parent.name, eu.SMOKE_TREE_DIRNAME)
        self.assertNotEqual(full_raw.parent.name, eu.SMOKE_TREE_DIRNAME)
        self.assertEqual(
            eu.run_registry_path(root, "nice", "must", run_id="smoke-1").parent.name,
            eu.SMOKE_TREE_DIRNAME,
        )
        self.assertEqual(
            eu.task3_raw_path(root, "nice", "must", run_id="task3-smoke-1").parent.name,
            eu.SMOKE_TREE_DIRNAME,
        )
        self.assertTrue(eu.is_smoke_run_id("task3-smoke-1"))
        self.assertFalse(eu.is_smoke_run_id("task3-full-1"))
        self.assertEqual(
            eu.task3_raw_path(root, "nice", "must", run_id="smoke-task3-1").parent.name,
            eu.SMOKE_TREE_DIRNAME,
        )

    def test_analysis_cli_reads_the_smoke_tree_for_a_smoke_run_id(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark = eu.build_benchmark_items(_seeds()[:1])
            eu.write_csv_rows(
                eu.artifact_path(
                    root / "data/processed/benchmark_items.csv", "nice", "must"
                ),
                benchmark,
            )
            # Only the smoke tree holds the run; the full-run file is absent.
            raw_path = eu.model_outputs_raw_path(root, "nice", "must", run_id="smoke-1")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(
                "\n".join(
                    json.dumps(
                        _task2_raw(
                            item,
                            model="m1",
                            run_id="smoke-1",
                            requirement="The system must export reports.",
                        )
                    )
                    for item in benchmark
                ),
                encoding="utf-8",
            )
            argv = [
                "generate_evaluation_analysis.py",
                "--dataset",
                "nice",
                "--variant",
                "must",
                "--run-id",
                "smoke-1",
                "--output-dir",
                str(root / "outputs" / "analysis"),
                "--skip-manifest-check",
                "--skip-registry-check",
                "--skip-construct-review-check",
                "--allow-partial",
                "--bootstrap-iterations",
                "5",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(analysis_cli.eu, "project_root", return_value=root),
                redirect_stdout(io.StringIO()),
            ):
                analysis_cli.main()
            self.assertTrue((root / "outputs" / "analysis" / "uq_scores.csv").exists())


class ExportPaperTablesTest(unittest.TestCase):
    def _write_cell(self, root: Path, dataset: str, variant: str) -> list[dict]:
        benchmark = eu.build_benchmark_items(_seeds())
        benchmark_path = eu.artifact_path(
            root / "data/processed/benchmark_items.csv", dataset, variant
        )
        eu.write_csv_rows(benchmark_path, benchmark)
        compatible = {
            "run_group_id": export.DEFAULT_PAPER_RUN_GROUP_ID,
            "tasks": "task1,task2",
            "expected_records": len(benchmark) * 2 * 6,
            "observed_records": len(benchmark) * 2 * 6,
            "deterministic_item_coverage": 1.0,
            "stochastic_complete_item_rate": 1.0,
            "batch_size": export.DEFAULT_PAPER_BATCH_SIZE,
            "batch_order": export.DEFAULT_PAPER_BATCH_ORDER,
        }
        registry_rows = [
            {
                **compatible,
                "run_id": "full-old",
                "model": "m1",
                "status": "complete",
                "started_at_utc": "2026-01-01T00:00:00Z",
            },
            {
                **compatible,
                "run_id": "full-new",
                "model": "m1",
                "status": "complete",
                "started_at_utc": "2026-02-01T00:00:00Z",
            },
            {
                **compatible,
                "run_id": "full-m2",
                "model": "m2",
                "status": "complete",
                "started_at_utc": "2026-02-01T00:00:00Z",
            },
            {
                **compatible,
                "run_id": "smoke-1",
                "model": "m1",
                "status": "complete",
                "started_at_utc": "2026-03-01T00:00:00Z",
            },
            {
                **compatible,
                "run_id": "full-azure",
                "model": "azure.gpt",
                "status": "complete",
                "started_at_utc": "2026-03-01T00:00:00Z",
            },
            {
                **compatible,
                "run_id": "full-partial",
                "model": "m3",
                "status": "running",
                "started_at_utc": "2026-03-01T00:00:00Z",
            },
        ]
        eu.write_csv_rows(eu.run_registry_path(root, dataset, variant), registry_rows)

        raw_path = eu.model_outputs_raw_path(root, dataset, variant)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", encoding="utf-8") as handle:
            for item in benchmark:
                # m1 always writes a mandatory requirement (strengthens weak sources);
                # m2 echoes the source statement.
                for model, run_id, requirement in [
                    ("m1", "full-new", "The system must export reports."),
                    ("m1", "full-old", "The system exports reports."),
                    ("m2", "full-m2", item["source_statement"]),
                ]:
                    handle.write(
                        json.dumps(
                            _task2_raw(
                                item,
                                model=model,
                                run_id=run_id,
                                requirement=requirement,
                            )
                        )
                        + "\n"
                    )
                for sample_index in range(3):
                    handle.write(
                        json.dumps(
                            _task2_raw(
                                item,
                                model="m1",
                                run_id="full-new",
                                requirement="The system must export reports.",
                                sample_kind="stochastic",
                                sample_index=sample_index,
                            )
                        )
                        + "\n"
                    )
        return benchmark

    def test_select_cell_runs_prefers_latest_and_drops_smoke_azure_incomplete(self):
        rows = [
            {
                "run_id": "full-old",
                "model": "m1",
                "status": "complete",
                "started_at_utc": "2026-01-01T00:00:00Z",
            },
            {
                "run_id": "full-new",
                "model": "m1",
                "status": "complete",
                "started_at_utc": "2026-02-01T00:00:00Z",
            },
            {
                "run_id": "smoke-x",
                "model": "m1",
                "status": "complete",
                "started_at_utc": "2026-09-01T00:00:00Z",
            },
            {
                "run_id": "full-a",
                "model": "azure.gpt",
                "status": "complete",
                "started_at_utc": "2026-02-01T00:00:00Z",
            },
            {
                "run_id": "full-r",
                "model": "m2",
                "status": "running",
                "started_at_utc": "2026-02-01T00:00:00Z",
            },
        ]
        chosen = export.select_cell_runs(rows, ["m1", "m2", "azure.gpt"])
        self.assertEqual(set(chosen), {"m1"})
        self.assertEqual(chosen["m1"]["run_id"], "full-new")

    def test_select_cell_runs_ignores_newer_incompatible_ablation(self):
        compatible = {
            "run_group_id": export.DEFAULT_PAPER_RUN_GROUP_ID,
            "tasks": "task1,task2",
            "status": "complete",
            "expected_records": 120,
            "observed_records": 120,
            "deterministic_item_coverage": 1.0,
            "stochastic_complete_item_rate": 1.0,
            "batch_order": "grouped",
            "batch_size": 16,
        }
        rows = [
            {
                **compatible,
                "run_id": "full-paper",
                "model": "m1",
                "started_at_utc": "2026-01-01T00:00:00Z",
            },
            {
                **compatible,
                "run_id": "full-ablation",
                "run_group_id": "batching-ablation",
                "model": "m1",
                "tasks": "task2",
                "expected_records": 10,
                "observed_records": 10,
                "stochastic_complete_item_rate": 0.0,
                "batch_order": "shuffled",
                "started_at_utc": "2026-09-01T00:00:00Z",
            },
        ]
        chosen = export.select_cell_runs(
            rows,
            ["m1"],
            run_group_id=export.DEFAULT_PAPER_RUN_GROUP_ID,
            benchmark_item_count=10,
            expected_stochastic_samples=5,
            expected_batch_order="grouped",
            expected_batch_size=16,
            run_prefixes=["full"],
        )
        self.assertEqual(chosen["m1"]["run_id"], "full-paper")

    def test_absent_stochastic_group_is_counted_as_incomplete(self):
        strict_rows = [
            {"model": "m1", "item_id": "a"},
            {"model": "m1", "item_id": "b"},
        ]
        consistency = {
            ("m1", "a"): {
                "model": "m1",
                "item_id": "a",
                "task": "task2",
                "uq_method": "modality_consistency",
                "valid_n": 5,
                "total_n": 5,
                "stochastic_complete": True,
                "uncertainty_score": 0.0,
                "label_distribution": eu.label_distribution_json({"mandatory": 1.0}),
            }
        }
        agreement, complete = export.agreement_for_strict_rows(strict_rows, consistency)
        self.assertEqual(agreement["agreement_n_complete"], 1)
        self.assertEqual(agreement["agreement_n_incomplete_excluded"], 1)
        self.assertEqual(agreement["repeated_sample_unanimity"], 1.0)
        self.assertEqual(len(complete), 1)

    def test_stream_raw_rows_filters_by_run_id(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps({"run_id": run_id, "item_id": run_id})
                    for run_id in ["keep-1", "drop-1", "keep-2"]
                ),
                encoding="utf-8",
            )
            rows = list(export.stream_raw_rows(path, {"keep-1", "keep-2"}))
            self.assertEqual([row["run_id"] for row in rows], ["keep-1", "keep-2"])

    def test_score_cell_include_smoke_reads_registry_and_raw_smoke_trees(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark = self._write_cell(root, "mlm_tapt", "must")
            smoke_registry = eu.run_registry_path(root, "mlm_tapt", "must", smoke=True)
            eu.write_csv_rows(
                smoke_registry,
                [
                    {
                        "run_id": "smoke-latest",
                        "run_group_id": export.DEFAULT_PAPER_RUN_GROUP_ID,
                        "model": "m1",
                        "tasks": "task1,task2",
                        "status": "complete",
                        "expected_records": 2 * 2 * 6,
                        "observed_records": 2 * 2 * 6,
                        "deterministic_item_coverage": 1.0,
                        "stochastic_complete_item_rate": 1.0,
                        "batch_size": 16,
                        "batch_order": "grouped",
                        "started_at_utc": "2026-09-01T00:00:00Z",
                    }
                ],
            )
            smoke_raw = eu.model_outputs_raw_path(root, "mlm_tapt", "must", smoke=True)
            for item in benchmark[:2]:
                eu.append_jsonl(
                    smoke_raw,
                    _task2_raw(
                        item,
                        model="m1",
                        run_id="smoke-latest",
                        requirement="The system must export reports.",
                    ),
                )

            cell = export.score_cell(
                root,
                "mlm_tapt",
                "must",
                ["m1"],
                ["azure."],
                include_smoke=True,
            )
            self.assertEqual(cell["run_ids"], {"m1": "smoke-latest"})
            self.assertEqual(
                {row["run_id"] for row in cell["raw_rows"]}, {"smoke-latest"}
            )
            self.assertEqual(len(cell["registry_paths"]), 2)
            self.assertEqual(len(cell["raw_paths"]), 2)

    def test_export_tables_writes_snapshots_and_disaggregations(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark = self._write_cell(root, "mlm_tapt", "must")
            output_dir = root / "outputs"
            result = export.export_tables(
                root,
                [("mlm_tapt", "must")],
                ["m1", "m2"],
                ["azure."],
                output_dir,
                bootstrap_samples=25,
            )

            task2_row = result["task2_rows"][0]
            self.assertEqual(task2_row["models"], 2)
            # Two models x 24 benchmark items, the superseded run is excluded.
            self.assertEqual(task2_row["n"], 2 * len(benchmark))
            self.assertGreater(task2_row["strict_text_over_commitment"], 0)
            for field in [
                "text_over_commitment_n_numerator",
                "text_over_commitment_lower_bound",
                "heuristic_text_modality_rate",
                "mean_requirement_word_count",
                "mean_length_ratio",
                "strengthening_rate_by_length_tercile",
            ]:
                self.assertIn(field, task2_row)

            confidence_row = result["confidence_rows"][0]
            self.assertEqual(
                confidence_row["n"],
                task2_row["text_over_commitment_n_denominator"],
            )
            self.assertEqual(
                confidence_row["strict_text_oc_n"],
                task2_row["strict_text_over_commitment_n_numerator"],
            )
            self.assertIn("agreement_n_complete", confidence_row)
            self.assertIn("agreement_n_incomplete_excluded", confidence_row)

            per_model = result["per_model_rows"]
            self.assertEqual({row["model"] for row in per_model}, {"m1", "m2"})
            self.assertEqual(
                {row["source_modality"] for row in per_model}, set(eu.MODALITIES)
            )
            weak_m1 = [
                row
                for row in per_model
                if row["model"] == "m1" and row["source_modality"] == "nice_to_have"
            ][0]
            self.assertEqual(
                weak_m1["strict_strengthening_n"],
                weak_m1["strict_strengthening_denominator"],
            )
            self.assertLessEqual(
                weak_m1["strict_strengthening_ci_low"],
                weak_m1["strict_strengthening_rate"],
            )
            self.assertEqual(weak_m1["n_items"], len(benchmark) // len(eu.MODALITIES))

            headline = {row["model"]: row for row in result["headline_rows"]}
            self.assertEqual(set(headline), {"m1", "m2"})
            self.assertEqual(headline["m1"]["n_valid"], len(benchmark))

            for path in result["paths"].values():
                self.assertTrue(Path(path).exists())
            self.assertTrue((output_dir / "paper_per_model_modality_table.md").exists())
            self.assertTrue((output_dir / "paper_per_model_headline.md").exists())

            provenance = json.loads(
                (output_dir / export.PROVENANCE_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(
                provenance["cells"][0]["run_ids"], {"m1": "full-new", "m2": "full-m2"}
            )
            self.assertEqual(provenance["models_cohort"], ["m1", "m2"])
            self.assertEqual(provenance["bootstrap_samples"], 25)
            self.assertEqual(provenance["expected_stochastic_samples"], 5)
            self.assertFalse(provenance["allow_missing_raw"])

    def test_weak_intent_headline_is_exported_next_to_the_other_rates(self):
        """The 29.8%-style weak headline is derived, not an orphaned snapshot."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark = self._write_cell(root, "mlm_tapt", "must")
            result = export.export_tables(
                root,
                [("mlm_tapt", "must")],
                ["m1", "m2"],
                ["azure."],
                root / "outputs",
                bootstrap_samples=5,
            )
            row = result["task2_rows"][0]
            for field in [
                "weak_n",
                "weak_n_readable",
                "weak_strict_text_strengthening_90",
                "weak_strict_text_strengthening_90_all_weak",
            ]:
                self.assertIn(field, export.TASK2_SNAPSHOT_FIELDS)
                self.assertIn(field, row)

            weak_items = len(
                [
                    item
                    for item in benchmark
                    if item["source_modality"] == "nice_to_have"
                ]
            )
            # Both models answer every weak item; m1 always strengthens to
            # "must" at confidence 0.95, m2 echoes the weak source statement.
            self.assertEqual(row["weak_n"], 2 * weak_items)
            self.assertEqual(row["weak_n_readable"], 2 * weak_items)
            self.assertAlmostEqual(row["weak_strict_text_strengthening_90"], 0.5)
            self.assertAlmostEqual(
                row["weak_strict_text_strengthening_90_all_weak"], 0.5
            )

            # The aggregator prefers this column over the blind Task 3 CSV. The
            # stability columns of this tiny cell are empty, so the aggregator
            # is fed a complete synthetic confidence row for the same cell.
            blind_rows = [
                {
                    "dataset": "mlm_tapt",
                    "variant": "must",
                    "weak_strict_text_strengthening_90": "29.8%",
                }
            ]
            confidence_rows = [
                {
                    "dataset": "mlm_tapt",
                    "variant": "must",
                    "n": str(row["text_over_commitment_n_denominator"]),
                    "broad_text_oc_n": str(row["text_over_commitment_n_numerator"]),
                    "strict_text_oc_n": str(
                        row["strict_text_over_commitment_n_numerator"]
                    ),
                    "strict_text_oc_conf_ge_90": "1.0",
                    "strict_text_oc_unanimous_modality_samples": "1.0",
                }
            ]
            headline = {
                headline_row["headline_key"]: headline_row
                for headline_row in agg.build_headline_rows(
                    result["task2_rows"], confidence_rows, blind_rows
                )
            }["weak_strict_text_strengthening_90"]
            self.assertEqual(
                headline["source_csv"], "paper_task2_text_drift_metrics.csv"
            )
            self.assertEqual(float(headline["per_cell_values"]), 0.5)
            self.assertEqual(headline["cell_n"], str(2 * weak_items))

    def test_parse_failure_rate_comes_from_the_raw_rows(self):
        """A response that never parsed has no score row, only a raw row."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark = self._write_cell(root, "mlm_tapt", "must")
            # One extra deterministic Task 2 raw row that failed to parse.
            raw_path = eu.model_outputs_raw_path(root, "mlm_tapt", "must")
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        _task2_raw(
                            benchmark[0],
                            model="m2",
                            run_id="full-m2",
                            requirement="",
                            parse_status="invalid_json",
                        )
                    )
                    + "\n"
                )
            result = export.export_tables(
                root,
                [("mlm_tapt", "must")],
                ["m1", "m2"],
                ["azure."],
                root / "outputs",
                bootstrap_samples=5,
            )
            row = result["task2_rows"][0]
            # 3 deterministic Task 2 raw rows per item (m1 new, m1 old, m2) but
            # only the two selected runs stream in, so 2 x 24 + 1 failure.
            self.assertAlmostEqual(
                row["parse_failure_rate"], 1 / (2 * len(benchmark) + 1)
            )
            self.assertGreater(row["parse_failure_rate"], 0)

    def test_parse_failure_rate_is_zero_without_failures(self):
        self.assertEqual(
            export.task2_parse_failure_rate(
                [
                    {
                        "task": "task2",
                        "sample_kind": "deterministic",
                        "parse_status": "ok",
                    },
                    # Stochastic and Task 1 rows never enter the denominator.
                    {
                        "task": "task2",
                        "sample_kind": "stochastic",
                        "parse_status": "invalid_json",
                    },
                    {
                        "task": "task1",
                        "sample_kind": "deterministic",
                        "parse_status": "truncated",
                    },
                ]
            ),
            0.0,
        )

    def test_a_selected_run_without_raw_rows_is_an_error(self):
        """A stale complete registry row must not yield a silently empty cell."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark = self._write_cell(root, "mlm_tapt", "must")
            registry_path = eu.run_registry_path(root, "mlm_tapt", "must")
            rows = eu.read_csv_rows(registry_path)
            rows.append(
                {
                    "run_group_id": export.DEFAULT_PAPER_RUN_GROUP_ID,
                    "run_id": "full-ghost",
                    "model": "m9",
                    "tasks": "task1,task2",
                    "status": "complete",
                    "expected_records": len(benchmark) * 2 * 6,
                    "observed_records": len(benchmark) * 2 * 6,
                    "deterministic_item_coverage": 1.0,
                    "stochastic_complete_item_rate": 1.0,
                    "batch_size": export.DEFAULT_PAPER_BATCH_SIZE,
                    "batch_order": export.DEFAULT_PAPER_BATCH_ORDER,
                    "started_at_utc": "2026-04-01T00:00:00Z",
                }
            )
            eu.write_csv_rows(registry_path, rows)

            with self.assertRaises(ValueError) as context:
                export.score_cell(
                    root, "mlm_tapt", "must", ["m1", "m2", "m9"], ["azure."]
                )
            message = str(context.exception)
            self.assertIn("full-ghost", message)
            self.assertIn(str(registry_path), message)
            self.assertIn("--allow-missing-raw", message)

            # The flag downgrades it to a warning and drops the model.
            cell = export.score_cell(
                root,
                "mlm_tapt",
                "must",
                ["m1", "m2", "m9"],
                ["azure."],
                allow_missing_raw=True,
            )
            self.assertEqual(sorted(cell["run_ids"]), ["m1", "m2"])
            self.assertNotIn("m9", cell["run_ids"])

            result = export.export_tables(
                root,
                [("mlm_tapt", "must")],
                ["m1", "m2", "m9"],
                ["azure."],
                root / "outputs",
                bootstrap_samples=5,
                allow_missing_raw=True,
            )
            self.assertEqual(
                {row["model"] for row in result["headline_rows"]}, {"m1", "m2"}
            )
            provenance = json.loads(
                (root / "outputs" / export.PROVENANCE_NAME).read_text(encoding="utf-8")
            )
            self.assertTrue(provenance["allow_missing_raw"])

    def test_expected_stochastic_samples_is_threaded_into_the_scores(self):
        """The cell has 3 stochastic samples; the official runs drew 5."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_cell(root, "mlm_tapt", "must")
            complete_at_three = export.score_cell(
                root,
                "mlm_tapt",
                "must",
                ["m1", "m2"],
                ["azure."],
                expected_stochastic_samples=3,
                run_group_id=None,
            )
            complete_at_five = export.score_cell(
                root,
                "mlm_tapt",
                "must",
                ["m1", "m2"],
                ["azure."],
                expected_stochastic_samples=5,
                run_group_id=None,
            )

            def complete_count(cell):
                return sum(
                    1
                    for row in cell["scores"]
                    if str(row.get("uq_method", "")) == "modality_consistency"
                    and eu.is_truthy_strict(row.get("stochastic_complete"))
                )

            self.assertGreater(complete_count(complete_at_three), 0)
            self.assertEqual(complete_count(complete_at_five), 0)
            self.assertEqual(
                export.build_parser().parse_args([]).expected_stochastic_samples,
                export.DEFAULT_EXPECTED_STOCHASTIC_SAMPLES,
            )

    def test_headline_rows_accept_bootstrap_cis(self):
        task2_rows = [
            {
                "dataset": "mlm_tapt",
                "variant": "must",
                "text_over_commitment": "0.2",
                "strict_text_over_commitment": "0.1",
            }
        ]
        confidence_rows = [
            {
                "dataset": "mlm_tapt",
                "variant": "must",
                "n": "100",
                "broad_text_oc_n": "20",
                "strict_text_oc_n": "10",
                "strict_text_oc_conf_ge_90": "1.0",
                "strict_text_oc_unanimous_modality_samples": "1.0",
            }
        ]
        blind_rows = [
            {
                "dataset": "mlm_tapt",
                "variant": "must",
                "weak_strict_text_strengthening_90": "29.8%",
            }
        ]
        ci_rows = [
            {
                "headline_key": "strict_text_strengthening",
                "ci_low": 0.05,
                "ci_high": 0.15,
                "bootstrap_samples": 1000,
            }
        ]
        rows = {
            row["headline_key"]: row
            for row in agg.build_headline_rows(
                task2_rows, confidence_rows, blind_rows, ci_rows
            )
        }
        self.assertEqual(rows["strict_text_strengthening"]["value_ci_low"], 0.05)
        self.assertEqual(rows["strict_text_strengthening"]["value_ci_high"], 0.15)
        self.assertEqual(rows["strict_text_strengthening"]["bootstrap_samples"], 1000)
        self.assertEqual(rows["broad_text_strengthening"]["value_ci_low"], "")


if __name__ == "__main__":
    unittest.main()
