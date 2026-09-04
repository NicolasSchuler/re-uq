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

from scripts import (
    aggregate_paper_headline_metrics as agg,
    compare_context_ablation as context_ablation,
    compare_run_matrix as compare_matrix,
    eval_utils as eu,
    export_paper_tables as export,
    generate_evaluation_analysis as analysis_cli,
)

# Deterministic-only fixtures plan no repeated samples, so the plan denominator
# is inert and every group scores exactly the rows the run wrote.
NO_STOCHASTIC_PLAN = eu.SamplingPlan(stochastic_samples=0)
FIVE_SAMPLE_PLAN = eu.SamplingPlan(stochastic_samples=5)


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
    batch_id=None,
):
    record = {
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
    # Left off by default so the shared fixtures keep exercising the
    # seed-clustered fallback; batched fixtures opt in explicitly.
    if batch_id is not None:
        record["batch_id"] = batch_id
    return record


def _task1_raw(item, *, model, run_id, decision="yes", confidence=0.95, batch_id=None):
    """One Task 1 entailment answer for `item`.

    `decision="yes"` on a non-mandatory source is the unsupported acceptance
    the RQ1 table counts; `y_true` is derived from the item by the scorer.
    """
    record = {
        "run_id": run_id,
        "model": model,
        "task": "task1",
        "item_id": item["item_id"],
        "seed_id": item["seed_id"],
        "source_modality": item["source_modality"],
        "sample_index": 0,
        "sample_kind": "deterministic",
        "confidence_scale": "0_1",
        "prompt_version": "v2-conf01",
        "raw_text": "",
        "parsed_json": {
            "decision": decision,
            "confidence": confidence,
            "brief_reason": "fixture",
        },
        "parse_status": "ok",
        "error": "",
    }
    if batch_id is not None:
        record["batch_id"] = batch_id
    return record


def _task3_raw(
    item, *, model, run_id, relation, audit_mode="blind", confidence=0.9, batch_id=None
):
    """One blind-audit verdict on `model`'s own Task 2 answer for `item`.

    The `item_id` is the one `eval_utils.build_task3_verification_items`
    generates, so the exporter's recomputed items and these rows join.
    """
    record = {
        "run_id": run_id,
        "model": model,
        "task": "task3",
        "item_id": (
            f"{item['item_id']}__task3__{eu.safe_identifier(model)}"
            f"__{eu.safe_identifier(audit_mode)}"
        ),
        "seed_id": item["seed_id"],
        "source_modality": item["source_modality"],
        "sample_index": 0,
        "sample_kind": "deterministic",
        "confidence_scale": "0_1",
        "prompt_version": "v2-conf01",
        "raw_text": "",
        "parsed_json": {
            "relation": relation,
            "confidence": confidence,
            "evidence_phrase": "",
            "brief_reason": "fixture",
        },
        "parse_status": "ok",
        "error": "",
    }
    if batch_id is not None:
        record["batch_id"] = batch_id
    return record


def _batched_task2_raw(benchmark, *, model, run_id, requirement, seeds_per_request=2):
    """Task 2 raw rows batched the way the archived runs sent them.

    Consecutive whole seeds share one request, so a request contains all four
    source conditions of each of its seeds.
    """
    by_seed = {}
    for item in benchmark:
        by_seed.setdefault(item["seed_id"], []).append(item)
    rows = []
    for index, seed_id in enumerate(sorted(by_seed)):
        request = index // seeds_per_request
        rows.extend(
            _task2_raw(
                item,
                model=model,
                run_id=run_id,
                requirement=requirement,
                batch_id=f"{run_id}:{model}:task2:deterministic:0:{request}",
            )
            for item in by_seed[seed_id]
        )
    return rows


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
        item = next(
            row for row in benchmark if row["source_modality"] == "nice_to_have"
        )
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
        scores = eu.build_uq_scores(benchmark, raw_rows, sampling_plan=FIVE_SAMPLE_PLAN)
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
        item = next(row for row in benchmark if row["source_modality"] == "optional")
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
                for row in eu.build_uq_scores(
                    benchmark, raw_rows, sampling_plan=FIVE_SAMPLE_PLAN
                )
            )
        )
        guarded = eu.build_uq_scores(
            benchmark, raw_rows, min_valid_samples=3, sampling_plan=FIVE_SAMPLE_PLAN
        )
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
        item = next(
            row for row in benchmark if row["source_modality"] == "nice_to_have"
        )
        raw = _task2_raw(
            item, model="m1", run_id="r1", requirement="The system must export reports."
        )
        raw["usage_completion_tokens"] = 42
        score = eu.build_uq_scores(benchmark, [raw], sampling_plan=NO_STOCHASTIC_PLAN)[
            0
        ]
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
        score = eu.build_uq_scores(benchmark, [raw], sampling_plan=NO_STOCHASTIC_PLAN)[
            0
        ]
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
            eu.build_uq_scores(benchmark, raw_rows, sampling_plan=NO_STOCHASTIC_PLAN)
        )
        row = next(entry for entry in summary if entry["task"] == "task2")
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
        scores = eu.build_uq_scores(
            benchmark, raw_rows, sampling_plan=NO_STOCHASTIC_PLAN
        )
        summary = eu.metric_summary_by_model_task_method(scores)
        compare_matrix.annotate_text_drift_cis(summary, scores, bootstrap_samples=25)
        row = next(entry for entry in summary if entry["task"] == "task2")
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

    def test_annotate_text_drift_cis_reports_both_clusters(self):
        benchmark = eu.build_benchmark_items(_seeds())
        raw_rows = _batched_task2_raw(
            benchmark,
            model="m1",
            run_id="r1",
            requirement="The system must export reports.",
        )
        scores = eu.build_uq_scores(
            benchmark, raw_rows, sampling_plan=NO_STOCHASTIC_PLAN
        )

        summary = eu.metric_summary_by_model_task_method(scores)
        compare_matrix.annotate_text_drift_cis(summary, scores, bootstrap_samples=50)
        row = next(entry for entry in summary if entry["task"] == "task2")

        self.assertEqual(row["bootstrap_ci_cluster_field"], "batch_id")
        self.assertGreaterEqual(
            row["strict_text_over_commitment_ci_high"]
            - row["strict_text_over_commitment_ci_low"],
            row["strict_text_over_commitment_seed_ci_high"]
            - row["strict_text_over_commitment_seed_ci_low"],
        )
        for field in [
            "bootstrap_ci_cluster_field",
            "text_over_commitment_seed_ci_low",
            "text_over_commitment_seed_ci_high",
            "strict_text_over_commitment_seed_ci_low",
            "strict_text_over_commitment_seed_ci_high",
        ]:
            self.assertIn(field, compare_matrix.SUMMARY_FIELDS)


class ContextAblationTableTest(unittest.TestCase):
    """The ablation table pairs the two arms on the same items and seeds."""

    @staticmethod
    def _pure_benchmark():
        seeds = [
            {
                **seed,
                "source_dataset": "PURE",
                "context_document": "Fixture FRS",
                "context_legend": "(M) mandatory, (O) optional",
                "context_section": "1 Fixture",
                "context_requirement_id": f"1.{index}",
                # Seeds 1-4 are mandatory-marked, 5-6 optional-marked.
                "context_marker": "M" if index <= 4 else "O",
                "context_before": "",
                "context_after": "",
            }
            for index, seed in enumerate(_seeds(), start=1)
        ]
        return eu.build_benchmark_items(
            seeds, passthrough_fields=eu.PURE_CONTEXT_FIELDS
        )

    @staticmethod
    def _registry_row(run_id, model, item_context, started, **overrides):
        return {
            "run_id": run_id,
            "run_group_id": "context-ablation-2026-09",
            "model": model,
            "dataset_id": "pure",
            "benchmark_variant": "must",
            "status": "complete",
            "deterministic_item_coverage": "1.0",
            "started_at_utc": started,
            "item_context": item_context,
            "batch_size": "16",
            "batch_order": "grouped",
            "notes": f"resolved_config_sha={run_id}",
            **overrides,
        }

    def _raw_rows(
        self, benchmark, run_id, model, *, strengthen_weak, seeds_per_request=None
    ):
        """One arm's raw rows, optionally batched `seeds_per_request` per request."""
        request_of = {
            seed_id: index // (seeds_per_request or 1)
            for index, seed_id in enumerate(sorted({r["seed_id"] for r in benchmark}))
        }
        rows = []
        for item in benchmark:
            requirement = item["source_statement"].replace("MAY", "may")
            if strengthen_weak and item["source_modality"] == "nice_to_have":
                # A strict strengthening: explicit modal, high confidence.
                requirement = f"The system must {item['capability_text']}."
            rows.append(
                _task2_raw(
                    item,
                    model=model,
                    run_id=run_id,
                    requirement=requirement,
                    batch_id=None
                    if seeds_per_request is None
                    else (
                        f"{run_id}:{model}:task2:deterministic:0:"
                        f"{request_of[item['seed_id']]}"
                    ),
                )
            )
        return rows

    def test_select_arm_runs_takes_latest_complete_and_reads_blank_as_bare(self):
        rows = [
            self._registry_row("old-bare", "m1", "", "2026-09-01T00:00:00Z"),
            self._registry_row("new-bare", "m1", "bare", "2026-09-02T00:00:00Z"),
            self._registry_row("doc", "m1", "document", "2026-09-02T00:00:00Z"),
            self._registry_row(
                "partial",
                "m1",
                "document",
                "2026-09-03T00:00:00Z",
                deterministic_item_coverage="0.5",
            ),
            self._registry_row(
                "running", "m1", "document", "2026-09-03T00:00:00Z", status="running"
            ),
            self._registry_row(
                "other-group",
                "m1",
                "document",
                "2026-09-03T00:00:00Z",
                run_group_id="provider-matrix-2026-05",
            ),
            self._registry_row("smoke-x", "m1", "document", "2026-09-04T00:00:00Z"),
        ]
        selected = context_ablation.select_arm_runs(
            rows, run_group_id="context-ablation-2026-09", include_smoke=False
        )
        self.assertEqual(
            {key: row["run_id"] for key, row in selected.items()},
            {("m1", "bare"): "new-bare", ("m1", "document"): "doc"},
        )
        with_smoke = context_ablation.select_arm_runs(
            rows, run_group_id="context-ablation-2026-09", include_smoke=True
        )
        self.assertEqual(with_smoke[("m1", "document")]["run_id"], "smoke-x")

    def test_build_tables_reports_arms_strata_and_paired_deltas(self):
        benchmark = self._pure_benchmark()
        registry = [
            self._registry_row("bare-1", "m1", "bare", "2026-09-02T00:00:00Z"),
            self._registry_row("doc-1", "m1", "document", "2026-09-02T00:00:00Z"),
        ]
        raw = self._raw_rows(
            benchmark, "bare-1", "m1", strengthen_weak=False
        ) + self._raw_rows(benchmark, "doc-1", "m1", strengthen_weak=True)

        tables = context_ablation.build_tables(
            benchmark,
            registry,
            raw,
            run_group_id="context-ablation-2026-09",
            include_smoke=False,
            bootstrap_samples=50,
            sampling_plan=NO_STOCHASTIC_PLAN,
        )

        arms = {(r["item_context"], r["stratum"]): r for r in tables["arms"]}
        self.assertEqual(
            set(arms),
            {(a, s) for a in ("bare", "document") for s in context_ablation.STRATA},
        )
        self.assertEqual(arms[("bare", "all")]["n"], len(benchmark))
        self.assertEqual(arms[("bare", "weak_intent")]["n"], 6)
        self.assertEqual(arms[("bare", "marker_M")]["n"], 16)
        self.assertEqual(arms[("bare", "marker_O")]["n"], 8)
        self.assertEqual(arms[("bare", "all")]["label_accuracy"], 1.0)
        self.assertEqual(
            arms[("bare", "weak_intent")]["strict_text_strengthening"], 0.0
        )
        self.assertEqual(
            arms[("document", "weak_intent")]["strict_text_strengthening"], 1.0
        )
        self.assertEqual(
            arms[("document", "weak_intent")]["weak_strict_text_strengthening_90"], 1.0
        )
        self.assertEqual(arms[("bare", "all")]["weak_strict_text_strengthening_90"], "")

        deltas = {(r["stratum"], r["metric"]): r for r in tables["deltas"]}
        self.assertEqual(
            len(deltas), len(context_ablation.STRATA) * len(context_ablation.METRICS)
        )
        weak = deltas[("weak_intent", "strict_text_strengthening")]
        self.assertEqual(
            (weak["bare"], weak["document"], weak["delta"]), (0.0, 1.0, 1.0)
        )
        self.assertEqual((weak["delta_ci_low"], weak["delta_ci_high"]), (1.0, 1.0))
        self.assertEqual((weak["n_bare"], weak["n_document"]), (6, 6))
        overall = deltas[("all", "strict_text_strengthening")]
        self.assertAlmostEqual(overall["delta"], 6 / 24)
        self.assertGreater(overall["delta_ci_low"], 0.0)
        self.assertEqual(deltas[("all", "label_accuracy")]["delta"], 0.0)

        self.assertEqual(
            [
                (p["item_context"], p["run_id"], p["task2_rows"])
                for p in tables["provenance"]
            ],
            [("bare", "bare-1", 24), ("document", "doc-1", 24)],
        )

        with TemporaryDirectory() as tmpdir:
            paths = context_ablation.write_outputs(
                tables, Path(tmpdir) / "context_ablation_summary"
            )
            markdown = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("## Deltas (document - bare)", markdown)
            self.assertIn("weak_intent", markdown)
            self.assertEqual(len(eu.read_csv_rows(paths["csv"])), len(tables["arms"]))
            self.assertEqual(
                len(eu.read_csv_rows(paths["deltas_csv"])), len(tables["deltas"])
            )
            provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
            self.assertEqual(provenance["dataset_id"], "pure")
            self.assertIn("resolved_config_sha=doc-1", provenance["runs"][1]["notes"])

    def test_delta_rows_report_the_complete_pair_cohort(self):
        benchmark = self._pure_benchmark()
        registry = [
            self._registry_row("bare-1", "m1", "bare", "2026-09-02T00:00:00Z"),
            self._registry_row("doc-1", "m1", "document", "2026-09-02T00:00:00Z"),
        ]
        bare = self._raw_rows(benchmark, "bare-1", "m1", strengthen_weak=False)
        document = self._raw_rows(benchmark, "doc-1", "m1", strengthen_weak=True)
        # The document arm never answered the last seed, so that seed has no pair.
        orphan_seed = benchmark[-1]["seed_id"]
        document = [row for row in document if row["seed_id"] != orphan_seed]

        tables = context_ablation.build_tables(
            benchmark,
            registry,
            bare + document,
            run_group_id="context-ablation-2026-09",
            include_smoke=False,
            bootstrap_samples=50,
            sampling_plan=NO_STOCHASTIC_PLAN,
        )

        deltas = {(r["stratum"], r["metric"]): r for r in tables["deltas"]}
        overall = deltas[("all", "label_accuracy")]
        seeds = {row["seed_id"] for row in benchmark}
        self.assertEqual(overall["n_complete_pairs"], len(seeds) - 1)
        self.assertEqual(overall["n_excluded_single_arm"], 1)
        self.assertIn("n_complete_pairs", context_ablation.DELTA_FIELDS)

    def test_ablation_rows_report_request_and_seed_clustered_intervals(self):
        benchmark = self._pure_benchmark()
        registry = [
            self._registry_row("bare-1", "m1", "bare", "2026-09-02T00:00:00Z"),
            self._registry_row("doc-1", "m1", "document", "2026-09-02T00:00:00Z"),
        ]
        # Three seeds per request, so a request is strictly coarser than a seed
        # and the two arms carry different request ids for the same seeds.
        bare = self._raw_rows(
            benchmark, "bare-1", "m1", strengthen_weak=False, seeds_per_request=3
        )
        document = self._raw_rows(
            benchmark, "doc-1", "m1", strengthen_weak=True, seeds_per_request=3
        )

        tables = context_ablation.build_tables(
            benchmark,
            registry,
            bare + document,
            run_group_id="context-ablation-2026-09",
            include_smoke=False,
            bootstrap_samples=100,
            sampling_plan=NO_STOCHASTIC_PLAN,
        )

        arms = {(r["item_context"], r["stratum"]): r for r in tables["arms"]}
        deltas = {(r["stratum"], r["metric"]): r for r in tables["deltas"]}
        arm = arms[("document", "all")]
        delta = deltas[("all", "strict_text_strengthening")]

        self.assertEqual(arm["bootstrap_ci_cluster_field"], "batch_id")
        self.assertGreaterEqual(
            arm["strict_text_strengthening_ci_high"]
            - arm["strict_text_strengthening_ci_low"],
            arm["strict_text_strengthening_seed_ci_high"]
            - arm["strict_text_strengthening_seed_ci_low"],
        )
        # Pairing stays by seed; only the resampling unit is the request.
        self.assertEqual(delta["delta_cluster_field"], "batch_id")
        self.assertEqual(delta["n_delta_clusters"], 2)
        self.assertEqual(delta["n_complete_pairs"], 6)
        self.assertGreaterEqual(
            delta["delta_ci_high"] - delta["delta_ci_low"],
            delta["delta_seed_ci_high"] - delta["delta_seed_ci_low"],
        )
        for field in (
            "bootstrap_ci_cluster_field",
            "strict_text_strengthening_seed_ci_low",
            "broad_text_strengthening_seed_ci_high",
        ):
            self.assertIn(field, context_ablation.ARM_FIELDS)
        for field in (
            "delta_cluster_field",
            "n_delta_clusters",
            "delta_seed_ci_low",
            "delta_seed_ci_high",
        ):
            self.assertIn(field, context_ablation.DELTA_FIELDS)

    def test_missing_arm_yields_empty_deltas_not_a_crash(self):
        benchmark = self._pure_benchmark()
        registry = [self._registry_row("bare-1", "m1", "bare", "2026-09-02T00:00:00Z")]
        raw = self._raw_rows(benchmark, "bare-1", "m1", strengthen_weak=False)
        tables = context_ablation.build_tables(
            benchmark,
            registry,
            raw,
            run_group_id="context-ablation-2026-09",
            include_smoke=False,
            bootstrap_samples=10,
            sampling_plan=NO_STOCHASTIC_PLAN,
        )
        self.assertEqual({r["item_context"] for r in tables["arms"]}, {"bare"})
        self.assertTrue(all(r["delta"] == "" for r in tables["deltas"]))
        self.assertTrue(all(r["n_document"] == 0 for r in tables["deltas"]))


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


class SamplingPlanThreadingTest(unittest.TestCase):
    """AB#2: a run with 3 of 5 planned samples must read 3/5 in every caller."""

    def _write_cell(self, root: Path, *, run_id: str, observed_samples: int) -> list:
        benchmark = eu.build_benchmark_items(_seeds()[:2])
        eu.write_csv_rows(
            eu.artifact_path(
                root / "data/processed/benchmark_items.csv", "nice", "must"
            ),
            benchmark,
        )
        raw_path = eu.model_outputs_raw_path(root, "nice", "must")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", encoding="utf-8") as handle:
            for item in benchmark:
                handle.write(
                    json.dumps(
                        _task2_raw(
                            item,
                            model="m1",
                            run_id=run_id,
                            requirement="The system must export reports.",
                        )
                    )
                    + "\n"
                )
                for sample_index in range(observed_samples):
                    handle.write(
                        json.dumps(
                            _task2_raw(
                                item,
                                model="m1",
                                run_id=run_id,
                                requirement="The system must export reports.",
                                sample_kind="stochastic",
                                sample_index=sample_index,
                            )
                        )
                        + "\n"
                    )
        return benchmark

    def test_analysis_cli_threads_the_declared_plan_into_the_scores(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_cell(root, run_id="full-1", observed_samples=3)
            argv = [
                "generate_evaluation_analysis.py",
                "--dataset",
                "nice",
                "--variant",
                "must",
                "--run-id",
                "full-1",
                "--expected-stochastic-samples",
                "5",
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

            rows = [
                row
                for row in eu.read_csv_rows(
                    root / "outputs" / "analysis" / "uq_scores.csv"
                )
                if row["uq_method"] == "modality_consistency"
            ]
            provenance = json.loads(
                (root / "outputs" / "analysis" / "provenance_manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(rows)
        self.assertEqual({row["valid_n"] for row in rows}, {"3"})
        self.assertEqual({row["total_n"] for row in rows}, {"5"})
        self.assertEqual({row["stochastic_complete"] for row in rows}, {"False"})
        self.assertEqual(provenance["sampling_plan_source"], "planned")
        self.assertEqual(provenance["expected_stochastic_samples"], 5)

    def _run_matrix(self, root: Path, planned_samples: int) -> list:
        config_path = root / f"run_config_{planned_samples}.json"
        config_path.write_text(
            json.dumps(
                {
                    "run_group_id": "plan-group",
                    "datasets": ["nice"],
                    "benchmark_variants": ["must"],
                    "stochastic": {
                        "temperature": 0.7,
                        "top_p": 1.0,
                        "samples": planned_samples,
                    },
                    "profiles": [
                        {
                            "profile_id": "fake",
                            "provider_id": "fake",
                            "base_url": "http://127.0.0.1:1234/v1",
                            "api_key_env": "LOCAL_OPENAI_API_KEY",
                            "models": ["m1"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        argv = [
            "compare_run_matrix.py",
            "--config",
            str(config_path),
            "--bootstrap-samples",
            "5",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(compare_matrix.eu, "project_root", return_value=root),
            redirect_stdout(io.StringIO()),
        ):
            compare_matrix.main()
        summary_path = eu.artifact_path(
            root / "outputs/run_matrix_summary.csv", "nice", "must"
        )
        return [
            row
            for row in eu.read_csv_rows(summary_path)
            if row["uq_method"] == "modality_consistency"
        ]

    def test_run_matrix_threads_the_run_config_plan_into_the_scores(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark = self._write_cell(root, run_id="full-1", observed_samples=3)
            eu.write_csv_rows(
                eu.run_registry_path(root, "nice", "must"),
                [
                    {
                        "run_id": "full-1",
                        "run_group_id": "plan-group",
                        "model": "m1",
                        "dataset_id": "nice",
                        "benchmark_variant": "must",
                        "status": "complete",
                    }
                ],
            )

            planned_five = self._run_matrix(root, 5)
            planned_three = self._run_matrix(root, 3)

        self.assertTrue(planned_five)
        # Five planned, three written: every group is incomplete and excluded
        # from the agreement denominator.
        self.assertEqual({row["agreement_n_complete"] for row in planned_five}, {"0"})
        self.assertEqual(
            {row["agreement_n_incomplete_excluded"] for row in planned_five},
            {str(len(benchmark))},
        )
        self.assertEqual(
            {row["agreement_n_complete"] for row in planned_three},
            {str(len(benchmark))},
        )


class CellFixture:
    """Writes one synthetic paper cell: registry, raw rows, and its audits.

    Shared by the export tests and the RQ-table tests rather than inherited
    between them, so neither class re-runs the other's cases.
    """

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
                # Task 1: both models accept every candidate, so the weak-source
                # rows are unsupported acceptances at p >= 0.90.
                for model, run_id in [("m1", "full-new"), ("m2", "full-m2")]:
                    handle.write(
                        json.dumps(_task1_raw(item, model=model, run_id=run_id)) + "\n"
                    )
        self._write_task3_cell(root, dataset, variant, benchmark)
        return benchmark

    def _write_task3_cell(
        self, root: Path, dataset: str, variant: str, benchmark: list[dict]
    ) -> None:
        """Blind audits of the Task 2 runs `_write_cell` selects.

        m1 flags its own strengthening, m2 calls everything preserved; the
        registry `notes` name the audited Task 2 run, which is what
        `select_task3_cell_runs` pins on.
        """
        prefix = "task3" if variant == "must" else f"task3-{variant}"
        audits = [
            ("m1", "full-new", f"{prefix}-m1", "strengthens"),
            ("m2", "full-m2", f"{prefix}-m2", "preserves"),
        ]
        eu.write_csv_rows(
            eu.task3_registry_path(root, dataset, variant),
            [
                {
                    "run_id": run_id,
                    "run_group_id": export.DEFAULT_PAPER_RUN_GROUP_ID,
                    "model": model,
                    "status": "complete",
                    "tasks": "task3",
                    "expected_records": len(benchmark) * 6,
                    "observed_records": len(benchmark) * 6,
                    "deterministic_item_coverage": 1.0,
                    "stochastic_complete_item_rate": 1.0,
                    "batch_size": export.DEFAULT_PAPER_BATCH_SIZE,
                    "batch_order": export.DEFAULT_PAPER_BATCH_ORDER,
                    "started_at_utc": "2026-02-02T00:00:00Z",
                    "notes": f"mode=full; audit_mode=blind; source_run_id={source}",
                }
                for model, source, run_id, _ in audits
            ],
        )
        raw_path = eu.task3_raw_path(root, dataset, variant)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", encoding="utf-8") as handle:
            for item in benchmark:
                for model, _, run_id, relation in audits:
                    handle.write(
                        json.dumps(
                            _task3_raw(
                                item, model=model, run_id=run_id, relation=relation
                            )
                        )
                        + "\n"
                    )


class ExportPaperTablesTest(CellFixture, unittest.TestCase):
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
            {
                "model": "m1",
                "dataset_id": "nice",
                "benchmark_variant": "must",
                "item_id": "a",
            },
            {
                "model": "m1",
                "dataset_id": "nice",
                "benchmark_variant": "must",
                "item_id": "b",
            },
        ]
        consistency = {
            ("m1", "nice", "must", "a"): {
                "model": "m1",
                "dataset_id": "nice",
                "benchmark_variant": "must",
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
            weak_m1 = next(
                row
                for row in per_model
                if row["model"] == "m1" and row["source_modality"] == "nice_to_have"
            )
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

            # Both cluster bootstraps reach the per-model and headline tables.
            for field in [
                "strengthening_ci_cluster_field",
                "broad_strengthening_seed_ci_low",
                "broad_strengthening_seed_ci_high",
                "strict_strengthening_seed_ci_low",
                "strict_strengthening_seed_ci_high",
            ]:
                self.assertIn(field, export.PER_MODEL_FIELDS)
                self.assertIn(field, weak_m1)
            headline_ci = {
                row["headline_key"]: row for row in result["headline_ci_rows"]
            }
            strict_ci = headline_ci["strict_text_strengthening"]
            self.assertIn("seed_ci_low", strict_ci)
            self.assertIn("seed_ci_high", strict_ci)
            self.assertIn("ci_cluster_field", strict_ci)

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

            def consistency_rows(cell):
                return [
                    row
                    for row in cell["scores"]
                    if str(row.get("uq_method", "")) == "modality_consistency"
                ]

            planned_at_five = consistency_rows(complete_at_five)
            self.assertTrue(planned_at_five)
            # Three samples were written, five were planned: 3/5, not 3/3.
            self.assertEqual({row["valid_n"] for row in planned_at_five}, {3})
            self.assertEqual({row["total_n"] for row in planned_at_five}, {5})
            self.assertEqual(
                {row["sampling_plan_source"] for row in planned_at_five}, {"planned"}
            )
            self.assertEqual(
                {row["total_n"] for row in consistency_rows(complete_at_three)}, {3}
            )
            self.assertEqual(
                complete_at_five["sampling_plan"],
                eu.SamplingPlan(stochastic_samples=5),
            )
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
                "seed_ci_low": 0.08,
                "seed_ci_high": 0.12,
                "ci_cluster_field": "batch_id",
                "bootstrap_samples": 1000,
            }
        ]
        rows = {
            row["headline_key"]: row
            for row in agg.build_headline_rows(
                task2_rows, confidence_rows, blind_rows, ci_rows
            )
        }
        strict = rows["strict_text_strengthening"]
        self.assertEqual(strict["value_ci_low"], 0.05)
        self.assertEqual(strict["value_ci_high"], 0.15)
        self.assertEqual(strict["bootstrap_samples"], 1000)
        # The request-clustered interval is the reported one; the narrower
        # seed-clustered pair travels alongside it.
        self.assertEqual(strict["value_seed_ci_low"], 0.08)
        self.assertEqual(strict["value_seed_ci_high"], 0.12)
        self.assertEqual(strict["value_ci_cluster_field"], "batch_id")
        self.assertEqual(rows["broad_text_strengthening"]["value_ci_low"], "")
        self.assertEqual(rows["broad_text_strengthening"]["value_seed_ci_low"], "")
        self.assertEqual(rows["broad_text_strengthening"]["value_ci_cluster_field"], "")


class BenchmarkGroundTruthDocTest(unittest.TestCase):
    """The generated ground-truth document must render the shipped tables."""

    def test_document_renders_every_template_and_worked_example(self):
        from scripts import export_benchmark_ground_truth as ground_truth

        root = Path(__file__).resolve().parents[1]
        document = ground_truth.build_document(root, nice_seed="S0001", mlm_seed=None)
        self.assertIn("# Benchmark Ground Truth", document)
        # All four fixed templates, verbatim, in both dataset sections.
        for template, occurrences in (
            ("The system MUST {capability}.", 2),
            ("The system SHOULD {capability}.", 2),
            ("The system MAY {capability}.", 2),
            # The weak template appears twice per section: once as the
            # main condition and once as the `probe_useful_if` variant, which
            # is identical to it by design.
            ("It would be useful if the system could {capability}.", 4),
        ):
            self.assertEqual(document.count(template), occurrences)
        # The worked example carries the structural gold labels.
        self.assertIn("`S0001_mandatory`", document)
        self.assertIn("`nice_to_have`", document)
        # The SHALL robustness variant is labelled distinctly, and the
        # weak phrasing probes and the validation trail are present.
        self.assertIn("(SHALL cell)", document)
        self.assertIn("probe_future_enhancement", document)
        self.assertIn("Validation trail", document)

    def test_cli_writes_the_document_to_a_custom_path(self):
        from scripts import export_benchmark_ground_truth as ground_truth

        with TemporaryDirectory() as tmp:
            output = ground_truth.main(["--output", str(Path(tmp) / "doc.md")])
            self.assertTrue(output.exists())
            self.assertIn(
                "# Benchmark Ground Truth", output.read_text(encoding="utf-8")
            )

    def test_non_first_worked_seed_uses_the_matching_shall_row(self):
        from scripts import export_benchmark_ground_truth as ground_truth

        root = Path(__file__).resolve().parents[1]
        nice_seed = eu.read_csv_rows(root / "data/processed/seeds_selected.csv")[1][
            "seed_id"
        ]
        mlm_seed = eu.read_csv_rows(
            root / "data/processed/seeds_selected_mlm_tapt.csv"
        )[1]["seed_id"]
        document = ground_truth.build_document(root, nice_seed, mlm_seed)

        for seed_id in (nice_seed, mlm_seed):
            self.assertIn(f"`{seed_id}_mandatory (SHALL cell)`", document)

    def test_validation_trail_enumerates_every_benchmark_cell(self):
        from scripts import export_benchmark_ground_truth as ground_truth

        root = Path(__file__).resolve().parents[1]
        document = ground_truth.build_document(root, nice_seed="S0001", mlm_seed=None)
        for artifact in (
            "data/processed/seeds_review.csv",
            "data/processed/seeds_review_mlm_tapt.csv",
            "outputs/benchmark_statements_review.csv",
            "outputs/benchmark_statements_review_shall.csv",
            "outputs/benchmark_statements_review_mlm_tapt.csv",
            "outputs/benchmark_statements_review_mlm_tapt_shall.csv",
            "outputs/benchmark_manifest.json",
            "outputs/benchmark_manifest_mlm_tapt.json",
        ):
            self.assertIn(f"`{artifact}`", document)
        for artifact in (
            "outputs/benchmark_statements_review.csv",
            "outputs/benchmark_statements_review_shall.csv",
            "outputs/benchmark_statements_review_mlm_tapt.csv",
            "outputs/benchmark_statements_review_mlm_tapt_shall.csv",
        ):
            self.assertIn(f"[`{artifact}`](../{artifact}) | 180 |", document)
        for artifact in (
            "outputs/benchmark_manifest.json",
            "outputs/benchmark_manifest_mlm_tapt.json",
        ):
            self.assertIn(f"[`{artifact}`](../{artifact}) | 10 |", document)


if __name__ == "__main__":
    unittest.main()


class PerModelRqTableTest(CellFixture, unittest.TestCase):
    """`paper_per_model_rq_table.csv`: the source of `tab:rq1` / `tab:rq23`.

    Reuses the two-model cell fixture, where m1 rewrites every source as
    mandatory (so every non-mandatory source is strengthened) and m2 echoes the
    source (so none is), and both models audit their own answers blind.
    """

    def _export(self, root, **kwargs):
        return export.export_tables(
            root,
            [("mlm_tapt", "must")],
            ["m1", "m2"],
            ["azure."],
            root / "outputs",
            bootstrap_samples=kwargs.pop("bootstrap_samples", 25),
            **kwargs,
        )

    def _rows(self, result):
        return {
            (row["model"], row["dataset"], row["variant"]): row
            for row in result["rq_rows"]
        }

    def test_row_set_is_one_per_model_plus_one_per_cell_plus_all(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_cell(root, "mlm_tapt", "must")
            self._write_cell(root, "nice", "must")
            result = export.export_tables(
                root,
                [("mlm_tapt", "must"), ("nice", "must")],
                ["m1", "m2"],
                ["azure."],
                root / "outputs",
                bootstrap_samples=25,
            )

            keys = set(self._rows(result))
            self.assertEqual(
                keys,
                {
                    ("m1", "all", "all"),
                    ("m2", "all", "all"),
                    ("all", "mlm_tapt", "must"),
                    ("all", "nice", "must"),
                    ("all", "all", "all"),
                },
            )
            # The model x cell interior is deliberately absent.
            self.assertNotIn(("m1", "mlm_tapt", "must"), keys)
            grand = self._rows(result)[("all", "all", "all")]
            self.assertEqual(grand["n_models_pooled"], 2)
            self.assertEqual(grand["n_cells_pooled"], 2)

    def test_counts_match_the_slice_definitions(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark = self._write_cell(root, "mlm_tapt", "must")
            rows = self._rows(self._export(root))
            m1 = rows[("m1", "all", "all")]

            # 24 items = 6 seeds x 4 source conditions; m1 answers all of them.
            self.assertEqual(m1["task2_n"], len(benchmark))
            self.assertEqual(m1["task2_n_readable"], len(benchmark))
            self.assertEqual(m1["task2_weak_n"], len(benchmark) // 4)
            self.assertEqual(m1["task2_weak_n_readable"], len(benchmark) // 4)
            # Every non-mandatory source is rewritten as mandatory.
            self.assertEqual(
                m1["task2_strict_strengthening_n"], 3 * len(benchmark) // 4
            )
            self.assertEqual(
                m1["task2_strict_strengthening_denominator"], len(benchmark)
            )
            self.assertEqual(m1["task2_no_cue_n"], 0)
            # Task 1 accepts every candidate; the weak-source items are the
            # unsupported acceptances.
            self.assertEqual(m1["task1_n"], len(benchmark))
            self.assertEqual(
                m1["task1_unsupported_acceptance_90_n"],
                m1["task1_unsupported_acceptance_90_denominator"],
            )
            self.assertEqual(m1["task1_accuracy"], 0.25)

            m2 = rows[("m2", "all", "all")]
            self.assertEqual(m2["task2_strict_strengthening_n"], 0)
            self.assertEqual(m2["task2_strict_strengthening_rate"], 0.0)

    def test_weak_strict_and_weak_strict_high_conf_are_separate_columns(self):
        """`tab:rq1` prints the ungated rate; `\\numWeakStrict` is the 0.90 one."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark = self._write_cell(root, "mlm_tapt", "must")
            # Re-write m1's weak answers with a confidence below the gate.
            raw_path = eu.model_outputs_raw_path(root, "mlm_tapt", "must")
            records = [json.loads(line) for line in raw_path.read_text().splitlines()]
            for record in records:
                if (
                    record["model"] == "m1"
                    and record["run_id"] == "full-new"
                    and record["task"] == "task2"
                    and record["source_modality"] == "nice_to_have"
                    and isinstance(record.get("parsed_json"), dict)
                ):
                    record["parsed_json"]["confidence"] = 0.5
            raw_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n"
            )

            m1 = self._rows(self._export(root))[("m1", "all", "all")]

            weak_items = len(benchmark) // 4
            self.assertEqual(m1["task2_weak_strict_strengthening_n"], weak_items)
            self.assertEqual(m1["task2_weak_strict_high_conf_90_n"], 0)
            self.assertNotEqual(
                m1["task2_weak_strict_strengthening_rate"],
                m1["task2_weak_strict_high_conf_90_rate"],
            )

    def test_task3_verdicts_are_joined_to_the_strict_strengthened_answers(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark = self._write_cell(root, "mlm_tapt", "must")
            rows = self._rows(self._export(root))
            m1 = rows[("m1", "all", "all")]

            # The denominator is m1's strict-strengthened answers, not all of
            # its Task 3 verdicts.
            self.assertEqual(
                m1["task3_strict_flagged_denominator"],
                m1["task2_strict_strengthening_n"],
            )
            self.assertEqual(m1["task3_strict_unaudited_n"], 0)
            # m1 flags every one of them; m2 has nothing to flag.
            self.assertEqual(m1["task3_strict_flagged_rate"], 1.0)
            self.assertEqual(m1["task3_strict_called_preserved_rate"], 0.0)
            self.assertEqual(rows[("m2", "all", "all")]["task3_strict_flagged_n"], 0)
            self.assertEqual(3 * len(benchmark) // 4, m1["task3_strict_joined_n"])

    def test_no_task3_leaves_the_audit_columns_blank(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_cell(root, "mlm_tapt", "must")
            m1 = self._rows(self._export(root, include_task3=False))[
                ("m1", "all", "all")
            ]

            self.assertEqual(m1["task3_n"], 0)
            self.assertEqual(m1["task3_strict_flagged_rate"], "")
            self.assertEqual(m1["task3_strict_flagged_ci_low"], "")
            # The Task 2 columns are unaffected.
            self.assertGreater(m1["task2_strict_strengthening_n"], 0)

    def test_missing_task3_audit_fails_closed(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_cell(root, "mlm_tapt", "must")
            eu.task3_registry_path(root, "mlm_tapt", "must").unlink()

            with self.assertRaises(ValueError) as caught:
                self._export(root)
            self.assertIn("No compatible complete Task 3 audit", str(caught.exception))

            # ...and can be downgraded deliberately.
            rows = self._rows(self._export(root, allow_missing_task3=True))
            self.assertEqual(rows[("m1", "all", "all")]["task3_n"], 0)

    def test_task3_audit_of_another_task2_run_is_not_selected(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_cell(root, "mlm_tapt", "must")
            registry_path = eu.task3_registry_path(root, "mlm_tapt", "must")
            registry_rows = eu.read_csv_rows(registry_path)
            for row in registry_rows:
                row["notes"] = "mode=full; audit_mode=blind; source_run_id=full-old"
            eu.write_csv_rows(registry_path, registry_rows)

            with self.assertRaises(ValueError) as caught:
                self._export(root)
            self.assertIn("source_run_id", str(caught.exception))

    def test_task3_rows_auditing_another_model_are_refused(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark = self._write_cell(root, "mlm_tapt", "must")
            # m1's audit run answers about m2's item ids.
            raw_path = eu.task3_raw_path(root, "mlm_tapt", "must")
            records = [json.loads(line) for line in raw_path.read_text().splitlines()]
            for record in records:
                if record["model"] == "m1":
                    record["item_id"] = record["item_id"].replace("__m1__", "__m2__")
            raw_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n"
            )

            with self.assertRaises(ValueError) as caught:
                self._export(root)
            self.assertIn("audit another model", str(caught.exception))
            self.assertTrue(benchmark)

    def test_task3_rows_from_a_different_generation_are_refused(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_cell(root, "mlm_tapt", "must")
            raw_path = eu.task3_raw_path(root, "mlm_tapt", "must")
            records = [json.loads(line) for line in raw_path.read_text().splitlines()]
            records[0]["item_id"] = "S9999_mandatory__task3__m1__blind"
            raw_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n"
            )

            with self.assertRaises(ValueError) as caught:
                self._export(root)
            self.assertIn("different Task 2 generation", str(caught.exception))

    def test_request_clustered_interval_is_primary_when_rows_carry_a_batch_id(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_cell(root, "mlm_tapt", "must")
            seed_clustered = self._rows(self._export(root))[("m1", "all", "all")]
            # The shared fixture writes no batch_id, so the primary interval
            # falls back to the seed and the two pairs coincide.
            self.assertEqual(seed_clustered["task2_ci_cluster_field"], "seed_id")
            self.assertEqual(
                seed_clustered["task2_strict_strengthening_ci_low"],
                seed_clustered["task2_strict_strengthening_seed_ci_low"],
            )

    def test_provenance_records_the_cohort_split_and_the_audit_runs(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_cell(root, "mlm_tapt", "must")
            result = self._export(root, local_models=["m2"])
            provenance = json.loads(
                result["paths"]["provenance"].read_text(encoding="utf-8")
            )

            self.assertEqual(provenance["models_hosted"], ["m1"])
            self.assertEqual(provenance["models_local"], ["m2"])
            self.assertEqual(provenance["task3_audit_mode"], "blind")
            cell = provenance["cells"][0]
            self.assertEqual(
                cell["task3_run_ids"], {"m1": "task3-m1", "m2": "task3-m2"}
            )
            self.assertEqual(cell["n_task3_raw_rows"], 2 * cell["n_benchmark_items"])
            self.assertIn("per_model_rq_table", provenance["outputs"])

    def test_rq_table_is_written_with_its_markdown_mirror(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_cell(root, "mlm_tapt", "must")
            result = self._export(root)
            path = result["paths"]["per_model_rq_table"]

            self.assertTrue(path.exists())
            self.assertTrue(path.with_suffix(".md").exists())
            written = eu.read_csv_rows(path)
            self.assertEqual(len(written), len(result["rq_rows"]))
            self.assertEqual(list(written[0]), export.PER_MODEL_RQ_FIELDS)


def _rq_score_row(
    item_id,
    *,
    model="m1",
    seed_id="S0001",
    batch_id=None,
    source_modality="nice_to_have",
    strict=False,
    confidence=0.95,
    parse_status="ok",
):
    """A deterministic Task 2 score row, as `build_uq_scores` emits one."""
    row = {
        "model": model,
        "task": "task2",
        "uq_method": "verbalized_confidence",
        "item_id": item_id,
        "seed_id": seed_id,
        "dataset_id": "mlm_tapt",
        "benchmark_variant": "must",
        "source_modality": source_modality,
        "gold_modality": source_modality,
        "text_modality_parse_status": parse_status,
        "strict_text_overcommit": strict,
        "text_overcommit": strict,
        "strict_text_high_conf_overcommit_90": bool(strict and confidence >= 0.90),
        "confidence": confidence,
        "uncertainty_score": 1.0 - confidence,
    }
    if batch_id is not None:
        row["batch_id"] = batch_id
    return row


class RqSliceMetricsTest(unittest.TestCase):
    """The slice joins and detector metrics behind the per-model RQ table."""

    def _slices(self, rows, *, meaning_variation=None, task3=None, consistency=None):
        return export.rq_metric_slices(
            [],
            rows,
            consistency or {},
            meaning_variation or {},
            task3 or [],
        )

    def test_meaning_variation_auroc_ranks_strengthened_answers_by_uncertainty(self):
        rows = [
            _rq_score_row(f"i{index}", strict=index < 4, seed_id=f"S{index:04d}")
            for index in range(8)
        ]
        # Strengthened answers vary more in meaning than faithful ones.
        meaning_variation = {
            export.paper_join_key(row): {
                "uncertainty_score": 0.9
                if eu.is_truthy_strict(row["strict_text_overcommit"])
                else 0.1
            }
            for row in rows
        }
        slices = self._slices(rows, meaning_variation=meaning_variation)
        row = export.per_model_rq_row(
            "m1",
            "all",
            "all",
            slices,
            n_models_pooled=1,
            n_cells_pooled=1,
            bootstrap_samples=0,
        )

        self.assertEqual(row["task2_meaning_variation_auroc"], 1.0)
        self.assertEqual(row["task2_meaning_variation_auroc_n"], 8)
        self.assertEqual(row["task2_meaning_variation_auroc_n_positive"], 4)
        self.assertEqual(
            row["task2_meaning_variation_auroc"],
            eu.auroc_score([1, 1, 1, 1, 0, 0, 0, 0], [0.9] * 4 + [0.1] * 4),
        )

    def test_verbalized_confidence_auroc_scores_one_minus_confidence(self):
        rows = [
            _rq_score_row(
                f"i{index}",
                strict=index < 4,
                # Strengthened answers are asserted slightly less confidently.
                confidence=0.80 if index < 4 else 0.99,
                seed_id=f"S{index:04d}",
            )
            for index in range(8)
        ]
        meaning_variation = {
            export.paper_join_key(row): {"uncertainty_score": 0.5} for row in rows
        }
        row = export.per_model_rq_row(
            "m1",
            "all",
            "all",
            self._slices(rows, meaning_variation=meaning_variation),
            n_models_pooled=1,
            n_cells_pooled=1,
            bootstrap_samples=0,
        )

        # Higher uncertainty must mean more strengthening, so a less confident
        # strengthened answer is a correct ranking, not an inverted one.
        self.assertEqual(row["task2_verbalized_confidence_auroc"], 1.0)
        self.assertEqual(row["task2_meaning_variation_auroc"], 0.5)

    def test_auroc_is_blank_when_every_answer_has_the_same_label(self):
        rows = [
            _rq_score_row(f"i{index}", strict=True, seed_id=f"S{index:04d}")
            for index in range(4)
        ]
        meaning_variation = {
            export.paper_join_key(row): {"uncertainty_score": 0.5} for row in rows
        }
        row = export.per_model_rq_row(
            "m1",
            "all",
            "all",
            self._slices(rows, meaning_variation=meaning_variation),
            n_models_pooled=1,
            n_cells_pooled=1,
            bootstrap_samples=0,
        )

        self.assertEqual(row["task2_meaning_variation_auroc"], "")
        self.assertEqual(row["task2_meaning_variation_auroc_n"], 4)
        self.assertEqual(row["task2_meaning_variation_auroc_n_positive"], 4)

    def test_task3_verdicts_cluster_on_the_audited_task2_request(self):
        rows = [
            _rq_score_row("i0", strict=True, batch_id="r1"),
            _rq_score_row("i1", strict=True, batch_id="r1"),
        ]
        task3 = [
            {
                "model": "m1",
                "task": "task3",
                "uq_method": "verbalized_confidence",
                "dataset_id": "mlm_tapt",
                "benchmark_variant": "must",
                "source_item_id": "i0",
                "gold_relation": "strengthens",
                "pred_relation": "strengthens",
                # The audit's own request must NOT become the cluster.
                "batch_id": "audit-request",
            },
            {
                "model": "m1",
                "task": "task3",
                "uq_method": "verbalized_confidence",
                "dataset_id": "mlm_tapt",
                "benchmark_variant": "must",
                "source_item_id": "i1",
                "gold_relation": "strengthens",
                "pred_relation": "preserves",
                "batch_id": "audit-request",
            },
        ]
        slices = self._slices(rows, task3=task3)

        self.assertEqual(
            [row["batch_id"] for row in slices["task3_strict"]], ["r1", "r1"]
        )
        row = export.per_model_rq_row(
            "m1",
            "all",
            "all",
            slices,
            n_models_pooled=1,
            n_cells_pooled=1,
            bootstrap_samples=10,
        )
        self.assertEqual(row["task3_ci_cluster_field"], "batch_id")
        self.assertEqual(row["task3_strict_flagged_n"], 1)
        self.assertEqual(row["task3_strict_called_preserved_n"], 1)
        self.assertEqual(row["task3_strict_flagged_denominator"], 2)

    def test_unaudited_strengthened_answers_are_counted_not_hidden(self):
        rows = [
            _rq_score_row("i0", strict=True, seed_id="S0001"),
            _rq_score_row("i1", strict=True, seed_id="S0002"),
        ]
        task3 = [
            {
                "model": "m1",
                "task": "task3",
                "uq_method": "verbalized_confidence",
                "dataset_id": "mlm_tapt",
                "benchmark_variant": "must",
                "source_item_id": "i0",
                "gold_relation": "strengthens",
                "pred_relation": "strengthens",
            }
        ]
        row = export.per_model_rq_row(
            "m1",
            "all",
            "all",
            self._slices(rows, task3=task3),
            n_models_pooled=1,
            n_cells_pooled=1,
            bootstrap_samples=0,
        )

        self.assertEqual(row["task3_strict_joined_n"], 1)
        self.assertEqual(row["task3_strict_unaudited_n"], 1)

    def test_request_clustered_interval_is_primary_and_seed_clustered_is_kept(self):
        # Two requests, each holding two whole seeds; strengthening is
        # all-or-none per request, which the seed clustering cannot see.
        rows = [
            _rq_score_row(
                f"i{index}",
                seed_id=f"S{index:04d}",
                batch_id="r1" if index < 4 else "r2",
                strict=index < 4,
            )
            for index in range(8)
        ]
        row = export.per_model_rq_row(
            "m1",
            "all",
            "all",
            self._slices(rows),
            n_models_pooled=1,
            n_cells_pooled=1,
            bootstrap_samples=200,
        )

        self.assertEqual(row["task2_ci_cluster_field"], "batch_id")
        request_width = (
            row["task2_strict_strengthening_ci_high"]
            - row["task2_strict_strengthening_ci_low"]
        )
        seed_width = (
            row["task2_strict_strengthening_seed_ci_high"]
            - row["task2_strict_strengthening_seed_ci_low"]
        )
        self.assertGreater(request_width, seed_width)

    def test_rows_without_a_batch_id_fall_back_to_seed_clustering(self):
        rows = [
            _rq_score_row(f"i{index}", seed_id=f"S{index:04d}", strict=index < 4)
            for index in range(8)
        ]
        row = export.per_model_rq_row(
            "m1",
            "all",
            "all",
            self._slices(rows),
            n_models_pooled=1,
            n_cells_pooled=1,
            bootstrap_samples=50,
        )

        self.assertEqual(row["task2_ci_cluster_field"], "seed_id")
        self.assertEqual(
            row["task2_strict_strengthening_ci_low"],
            row["task2_strict_strengthening_seed_ci_low"],
        )

    def test_weak_intent_rate_gates_the_numerator_not_the_denominator(self):
        rows = [
            _rq_score_row("i0", strict=True, confidence=0.95),
            _rq_score_row("i1", strict=True, confidence=0.50),
            _rq_score_row("i2", strict=False, confidence=0.95),
            _rq_score_row("i3", strict=True, source_modality="optional"),
        ]

        self.assertEqual(eu.weak_intent_strict_strengthening_rate(rows), 2 / 3)
        self.assertEqual(eu.weak_intent_strict_strengthening_rate(rows, 0.90), 1 / 3)

    def test_unreadable_answers_are_excluded_from_the_strengthening_denominator(self):
        rows = [
            _rq_score_row("i0", strict=True),
            _rq_score_row("i1", parse_status="unknown"),
        ]
        row = export.per_model_rq_row(
            "m1",
            "all",
            "all",
            self._slices(rows),
            n_models_pooled=1,
            n_cells_pooled=1,
            bootstrap_samples=0,
        )

        self.assertEqual(row["task2_n"], 2)
        self.assertEqual(row["task2_n_readable"], 1)
        self.assertEqual(row["task2_strict_strengthening_denominator"], 1)
        self.assertEqual(row["task2_no_cue_n"], 1)
        self.assertEqual(row["task2_no_cue_rate"], 0.5)
