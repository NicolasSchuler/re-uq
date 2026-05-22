import json
import math
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts import eval_utils as eu
from scripts import compare_run_matrix as compare_matrix
from scripts import evaluate_external_ai_probe as external_eval
from scripts import generate_evaluation_analysis as analysis_cli
from scripts import run_experiment_from_config as run_config_cli
from scripts import run_task3_verification_from_config as task3_cli
from scripts import show_run_progress
from scripts import structured_outputs as so


class EvalUtilsTest(unittest.TestCase):
    def _instructor_task1_jobs(self, seed_count=1):
        seeds = [
            {
                "seed_id": f"S{index + 1:04d}",
                "source_dataset": "NICE",
                "original_requirement": f"The system shall export report {index + 1}.",
                "capability_text_final": f"export report {index + 1}",
            }
            for index in range(seed_count)
        ]
        benchmark = eu.build_benchmark_items(seeds)
        mandatory_items = [row for row in benchmark if row["source_modality"] == "mandatory"]
        return eu.planned_completion_jobs(
            mandatory_items,
            tasks=["task1"],
            model="m1",
            host="http://localhost:1234/v1",
            run_id="full-1",
            prompt_version="v2-instructor-conf01",
            task1_template=eu.load_prompt("prompts/mandatory_entailment.txt"),
            task2_template=eu.load_prompt("prompts/modality_extraction.txt"),
            deterministic={"temperature": 0.0, "top_p": 1.0, "samples": 1},
            stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 0},
            max_tokens=64,
            timeout_s=30,
            api_key_env="LOCAL_OPENAI_API_KEY",
            structured_output="instructor",
            extra_body={"thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}},
            validation_retries=3,
        )

    def test_benchmark_labels(self):
        seeds = [
            {
                "seed_id": "S0001",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": "export reports",
            }
        ]
        items = eu.build_benchmark_items(seeds)
        self.assertEqual(len(items), 4)
        by_modality = {row["source_modality"]: row for row in items}
        self.assertEqual(by_modality["mandatory"]["task1_gold_decision"], "yes")
        self.assertEqual(by_modality["recommended"]["task1_gold_decision"], "no")
        self.assertEqual(by_modality["optional"]["task2_gold_modality"], "optional")
        self.assertGreater(by_modality["mandatory"]["ordinal_strength"], by_modality["nice_to_have"]["ordinal_strength"])

    def test_benchmark_statement_review_export(self):
        seeds = [
            {
                "seed_id": "S0001",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": "export reports",
            }
        ]
        benchmark = eu.build_benchmark_items(seeds)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_paths = eu.write_benchmark_statement_review(benchmark, tmpdir)
            frame = eu.benchmark_statement_review_frame(benchmark)

            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["MUST source"], "The system MUST export reports.")
            self.assertEqual(frame.iloc[0]["Nice-to-have source"], "It would be useful if the system could export reports.")
            self.assertTrue(export_paths["markdown"].exists())
            self.assertTrue(export_paths["csv"].exists())

    def test_shall_benchmark_uses_shall_but_keeps_labels(self):
        seeds = [
            {
                "seed_id": "S0001",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": "export reports",
            }
        ]
        items = eu.build_benchmark_items(seeds, mandatory_keyword="SHALL")
        self.assertEqual(len(items), 4)
        self.assertEqual(len({row["item_id"] for row in items}), 4)
        by_modality = {row["source_modality"]: row for row in items}
        self.assertEqual(by_modality["mandatory"]["source_statement"], "The system SHALL export reports.")
        self.assertEqual(by_modality["mandatory"]["candidate_requirement"], "The system SHALL export reports.")
        self.assertEqual(by_modality["mandatory"]["task1_gold_decision"], "yes")
        self.assertEqual(by_modality["recommended"]["task1_gold_decision"], "no")

    def test_weak_modality_probe_generation(self):
        seeds = [
            {
                "seed_id": f"S{i:04d}",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": f"export reports {i}",
            }
            for i in range(1, 21)
        ]

        items = eu.build_weak_modality_probe_items(seeds)

        self.assertEqual(len(items), 80)
        self.assertEqual(len({row["item_id"] for row in items}), 80)
        self.assertEqual({row["task2_gold_modality"] for row in items}, {"nice_to_have"})
        self.assertEqual({row["source_modality"] for row in items}, {"nice_to_have"})
        self.assertEqual({row["template_id"] for row in items}, {row["template_id"] for row in eu.WEAK_MODALITY_PROBE_TEMPLATES})
        for row in items:
            text = row["source_statement"].lower()
            self.assertNotIn("nice_to_have", text)
            self.assertNotIn("nice-to-have", text)
            self.assertTrue(text.endswith("."))

    def test_weak_modality_sanity_validation(self):
        rows = eu.weak_modality_template_sanity_rows()
        incomplete = eu.weak_modality_sanity_status(rows)
        self.assertFalse(incomplete["valid"])
        self.assertEqual(set(incomplete["incomplete_template_ids"]), {row["template_id"] for row in eu.WEAK_MODALITY_PROBE_TEMPLATES})

        agreed = [{**row, "weaker_than_should": "yes", "reviewer": "r1"} for row in rows]
        self.assertTrue(eu.weak_modality_sanity_status(agreed)["valid"])

        disagreed = list(agreed)
        disagreed[0] = {**disagreed[0], "weaker_than_should": "no"}
        status = eu.weak_modality_sanity_status(disagreed)
        self.assertFalse(status["valid"])
        self.assertEqual(status["disagreeing_template_ids"], [disagreed[0]["template_id"]])

    def test_weak_modality_construct_review_requires_two_agreeing_reviewers(self):
        rows = eu.weak_modality_construct_review_rows()
        self.assertEqual(len(rows), 8)
        self.assertEqual(set(rows[0]), set(eu.WEAK_MODALITY_CONSTRUCT_REVIEW_FIELDS))

        incomplete = eu.weak_modality_construct_review_status(rows)
        self.assertFalse(incomplete["valid"])
        self.assertEqual(
            set(incomplete["incomplete_template_ids"]),
            {row["template_id"] for row in eu.WEAK_MODALITY_PROBE_TEMPLATES},
        )

        agreed = [{**row, "weaker_than_should": "yes"} for row in rows]
        self.assertTrue(eu.weak_modality_construct_review_status(agreed)["valid"])

        one_reviewer_only = [row for row in agreed if row["reviewer_id"] == "R1"]
        one_status = eu.weak_modality_construct_review_status(one_reviewer_only)
        self.assertFalse(one_status["valid"])
        self.assertEqual(
            set(one_status["insufficient_template_ids"]),
            {row["template_id"] for row in eu.WEAK_MODALITY_PROBE_TEMPLATES},
        )

        disagreed = list(agreed)
        disagreed[0] = {**disagreed[0], "weaker_than_should": "no"}
        status = eu.weak_modality_construct_review_status(disagreed)
        self.assertFalse(status["valid"])
        self.assertEqual(status["disagreeing_template_ids"], [disagreed[0]["template_id"]])

    def test_requirement_text_modality_parser(self):
        cases = [
            ("The system SHALL export reports.", "mandatory"),
            ("The system SHOULD export reports.", "recommended"),
            ("The system MAY export reports.", "optional"),
            ("The system could export reports.", "optional"),
            ("The system can export reports.", "optional"),
            ("It would be nice if the system could export reports.", "nice_to_have"),
            ("The system exports reports.", "mandatory"),
            ("System provides export reports.", "mandatory"),
        ]

        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(eu.requirement_text_modality(text), expected)

    def test_text_modality_overcommitment_fields(self):
        optional_to_shall = eu.text_modality_fields("The system shall export reports.", "optional", "optional", 0.98)
        nice_to_could = eu.text_modality_fields("The system could export reports.", "nice_to_have", "nice_to_have", 0.94)
        weak_phrase = eu.text_modality_fields(
            "It would be nice if the system could export reports.",
            "nice_to_have",
            "nice_to_have",
            0.94,
        )

        self.assertTrue(optional_to_shall["text_overcommit"])
        self.assertTrue(nice_to_could["text_overcommit"])
        self.assertFalse(weak_phrase["text_overcommit"])
        self.assertEqual(weak_phrase["text_modality"], "nice_to_have")

    def test_manifest_hash_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "rows.csv"
            prompt_path = root / "prompt.txt"
            manifest_path = root / "manifest.json"
            eu.write_csv_rows(csv_path, [{"id": "1"}, {"id": "2"}])
            prompt_path.write_text("prompt\n", encoding="utf-8")

            manifest = eu.write_benchmark_manifest([csv_path, prompt_path], manifest_path, root=root, metadata={"kind": "test"})
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["metadata"]["kind"], "test")
            self.assertEqual(loaded["artifacts"][0]["rows"], 2)
            self.assertEqual(loaded["artifacts"][0]["sha256"], eu.sha256_file(csv_path))
            self.assertEqual(loaded["artifacts"][1]["rows"], "")

    def test_rule_based_parser_maps_modalities(self):
        cases = {
            "The system MUST export reports.": "mandatory",
            "The system SHALL export reports.": "mandatory",
            "The system SHOULD export reports.": "recommended",
            "The system MAY export reports.": "optional",
            "It would be useful if the system could export reports.": "nice_to_have",
        }
        for text, expected in cases.items():
            self.assertEqual(eu.rule_based_source_modality(text), expected)

    def test_rule_baseline_scores_are_perfect_on_controlled_benchmark(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        scores = eu.build_rule_baseline_scores(benchmark)
        summary = eu.metric_summary_by_model_task_method(scores)

        self.assertEqual(len(scores), 8)
        self.assertEqual({row["model"] for row in scores}, {eu.RULE_BASELINE_MODEL})
        self.assertTrue(all(row["accuracy"] == 1.0 for row in summary))
        self.assertTrue(all(row["uq_method"] == eu.RULE_BASELINE_METHOD for row in summary))

    def test_parse_task1_response(self):
        raw = 'Some preface {"decision": "yes", "confidence": 87, "brief_reason": "MUST matches"}'
        parsed, status = eu.parse_task_response("task1", raw)
        self.assertEqual(status, "ok")
        self.assertEqual(parsed["decision"], "yes")
        self.assertEqual(parsed["confidence"], 87.0)

    def test_instructor_response_models_enforce_confidence_probability(self):
        parsed = so.Task2Response.model_validate(
            {"requirement": "The system MAY export reports.", "modality": "optional", "confidence": 0.95}
        )

        self.assertEqual(parsed.confidence, 0.95)
        for bad_confidence in ["0.95", -0.1, 1.1, 95]:
            with self.subTest(confidence=bad_confidence):
                with self.assertRaises(Exception):
                    so.Task2Response.model_validate(
                        {
                            "requirement": "The system MAY export reports.",
                            "modality": "optional",
                            "confidence": bad_confidence,
                        }
                    )

    def test_confidence_probability_handles_legacy_and_instructor_rows(self):
        legacy = {"parsed_json": {"confidence": 90.0}}
        prompt_v2_row = {
            "prompt_version": "v2-conf01",
            "parsed_json": {"confidence": 0.9},
        }
        prompt_contract_row = {
            "parsed_json": {"confidence": 0.85},
            "output_contract_version": so.PROMPT_OUTPUT_CONTRACT_VERSION,
        }
        instructor_row = {
            "parsed_json": {"confidence": 0.9},
            "output_contract_version": so.INSTRUCTOR_OUTPUT_CONTRACT_VERSION,
            "confidence_scale": so.INSTRUCTOR_CONFIDENCE_SCALE,
        }
        instructor_contract_only = {
            "parsed_json": {"confidence": 0.8},
            "output_contract_version": so.INSTRUCTOR_OUTPUT_CONTRACT_VERSION,
        }
        instructor_scale_only = {
            "parsed_json": {"confidence": 0.7},
            "confidence_scale": so.INSTRUCTOR_CONFIDENCE_SCALE,
        }

        self.assertEqual(eu.confidence_probability(legacy), 0.9)
        self.assertEqual(eu.confidence_probability(prompt_v2_row), 0.9)
        self.assertEqual(eu.confidence_probability(prompt_contract_row), 0.85)
        self.assertEqual(eu.confidence_probability(instructor_row), 0.9)
        self.assertEqual(eu.confidence_probability(instructor_contract_only), 0.8)
        self.assertEqual(eu.confidence_probability(instructor_scale_only), 0.7)

    def test_v2_raw_records_mark_probability_confidence_for_non_instructor_runs(self):
        item = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )[0]

        record = eu.build_raw_record(
            run_id="full-1",
            model="m1",
            host="http://localhost:8000/v1",
            task="task1",
            item=item,
            sample_index=0,
            sample_kind="deterministic",
            temperature=0.0,
            top_p=1.0,
            prompt_version="v2-conf01",
            prompt="prompt",
            completion={
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.9, "brief_reason": "ok"}',
                "latency_s": 0.01,
                "error": "",
            },
            structured_output="json_schema",
        )

        self.assertEqual(record["parse_status"], "ok")
        self.assertEqual(record["confidence_scale"], eu.CONFIDENCE_SCALE_0_1)
        self.assertEqual(record["output_contract_version"], so.PROMPT_OUTPUT_CONTRACT_VERSION)
        self.assertEqual(eu.confidence_probability(record), 0.9)

    def test_raw_records_infer_probability_scale_from_prompt_contract(self):
        item = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )[0]

        record = eu.build_raw_record(
            run_id="full-1",
            model="m1",
            host="http://localhost:8000/v1",
            task="task1",
            item=item,
            sample_index=0,
            sample_kind="deterministic",
            temperature=0.0,
            top_p=1.0,
            prompt_version="v1",
            prompt='Return JSON only: {"confidence": 0.0-1.0}\nUse confidence as a decimal probability.',
            completion={
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.9, "brief_reason": "ok"}',
                "latency_s": 0.01,
                "error": "",
            },
        )

        self.assertEqual(record["parse_status"], "ok")
        self.assertEqual(record["confidence_scale"], eu.CONFIDENCE_SCALE_0_1)
        self.assertEqual(record["output_contract_version"], so.PROMPT_OUTPUT_CONTRACT_VERSION)
        self.assertEqual(eu.confidence_probability(record), 0.9)

    def test_v2_raw_records_reject_percentage_confidence_without_legacy_marker(self):
        item = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )[0]

        record = eu.build_raw_record(
            run_id="full-1",
            model="m1",
            host="http://localhost:8000/v1",
            task="task1",
            item=item,
            sample_index=0,
            sample_kind="deterministic",
            temperature=0.0,
            top_p=1.0,
            prompt_version="v2-conf01",
            prompt="prompt",
            completion={
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 95, "brief_reason": "bad scale"}',
                "latency_s": 0.01,
                "error": "",
            },
        )

        self.assertEqual(record["parse_status"], "invalid_confidence")
        self.assertIsNotNone(record["parsed_json"])

    def test_task1_selected_label_confidence_sets_p_yes_for_yes_and_no(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        item_by_modality = {row["source_modality"]: row for row in benchmark}
        raw_rows = [
            {
                "run_id": "full-1",
                "model": "m1",
                "task": "task1",
                "item_id": item_by_modality["mandatory"]["item_id"],
                "sample_kind": "deterministic",
                "sample_index": 0,
                "parse_status": "ok",
                "parsed_json": {"decision": "yes", "confidence": 0.8, "brief_reason": "mandatory"},
                "output_contract_version": so.INSTRUCTOR_OUTPUT_CONTRACT_VERSION,
                "confidence_scale": so.INSTRUCTOR_CONFIDENCE_SCALE,
            },
            {
                "run_id": "full-1",
                "model": "m1",
                "task": "task1",
                "item_id": item_by_modality["optional"]["item_id"],
                "sample_kind": "deterministic",
                "sample_index": 0,
                "parse_status": "ok",
                "parsed_json": {"decision": "no", "confidence": 0.7, "brief_reason": "optional"},
                "output_contract_version": so.INSTRUCTOR_OUTPUT_CONTRACT_VERSION,
                "confidence_scale": so.INSTRUCTOR_CONFIDENCE_SCALE,
            },
        ]

        scores = eu.build_uq_scores(benchmark, raw_rows)
        score_by_item = {row["item_id"]: row for row in scores}

        self.assertEqual(score_by_item[item_by_modality["mandatory"]["item_id"]]["confidence"], 0.8)
        self.assertEqual(score_by_item[item_by_modality["mandatory"]["item_id"]]["p_yes"], 0.8)
        self.assertEqual(score_by_item[item_by_modality["optional"]["item_id"]]["confidence"], 0.7)
        self.assertAlmostEqual(score_by_item[item_by_modality["optional"]["item_id"]]["p_yes"], 0.3)

    def test_auto_capability_text_strips_quotes_and_boilerplate(self):
        self.assertEqual(
            eu.auto_capability_text("'The system shall refresh the display every 60 seconds.'"),
            "refresh the display every 60 seconds",
        )
        self.assertEqual(
            eu.auto_capability_text('"The application must export reports as CSV."'),
            "export reports as CSV",
        )
        self.assertEqual(
            eu.auto_capability_text("'The system shall interface with CampusConnect's central server.'"),
            "interface with CampusConnect's central server",
        )
        self.assertEqual(
            eu.auto_capability_text("The TMT Observatory shall monitor subsystem performance."),
            "monitor subsystem performance",
        )
        self.assertEqual(
            eu.auto_capability_text("The system shall be able to display a summary which will include cohort progress."),
            "display a summary which will include cohort progress",
        )

    def test_refresh_capability_suggestions_preserves_manual_edits(self):
        rows = [
            {
                "original_requirement": "'The system shall refresh the display every 60 seconds.'",
                "capability_text_auto": "'The system shall refresh the display every 60 seconds.'",
                "include": "yes",
                "capability_text_final": "'The system shall refresh the display every 60 seconds.'",
            },
            {
                "original_requirement": "'The system shall display Events or Activities.'",
                "capability_text_auto": "'The system shall display Events or Activities.'",
                "include": "yes",
                "capability_text_final": "show events and activities",
            },
        ]
        refreshed, updated = eu.refresh_capability_suggestions(rows)
        self.assertEqual(updated, 1)
        self.assertEqual(refreshed[0]["capability_text_final"], "refresh the display every 60 seconds")
        self.assertEqual(refreshed[1]["capability_text_final"], "show events and activities")

    def test_write_included_capability_review(self):
        rows = [
            {
                "seed_id": "S0001",
                "original_requirement": "'The system shall refresh the display every 60 seconds.'",
                "capability_text_final": "refresh the display every 60 seconds.",
                "include": "yes",
            },
            {
                "seed_id": "S0002",
                "original_requirement": "'The product shall reject invalid input.'",
                "capability_text_final": "reject invalid input",
                "include": "no",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            review_path = Path(tmpdir) / "seeds_review.csv"
            eu.write_csv_rows(review_path, rows)
            export_paths = eu.write_included_capability_review(review_path, Path(tmpdir) / "outputs")
            frame = eu.included_capability_review_frame(review_path)

            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["Final capability text"], "refresh the display every 60 seconds")
            self.assertTrue(export_paths["markdown"].exists())
            self.assertTrue(export_paths["csv"].exists())

    def test_write_csv_rows_if_changed_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.csv"
            candidate_path = Path(tmpdir) / "rows_candidate.csv"
            eu.write_csv_rows(path, [{"id": "1", "value": "reviewed"}])

            result = eu.write_csv_rows_if_changed(
                path,
                [{"id": "1", "value": "regenerated"}],
                candidate_path=candidate_path,
            )

            self.assertEqual(result["status"], "candidate_written")
            self.assertEqual(eu.read_csv_rows(path)[0]["value"], "reviewed")
            self.assertEqual(eu.read_csv_rows(candidate_path)[0]["value"], "regenerated")

    def test_write_csv_rows_if_changed_reports_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.csv"
            rows = [{"id": "1", "value": "reviewed"}]
            eu.write_csv_rows(path, rows)

            result = eu.write_csv_rows_if_changed(path, rows)

            self.assertEqual(result["status"], "unchanged")
            self.assertFalse((Path(tmpdir) / "rows_candidate.csv").exists())

    def test_automatic_filter_rejects_internal_sentence_boundary(self):
        original = "'The product shall ensure that it can only be accessed by authorized users. The product will be able to distinguish between authorized and unauthorized users in all access attempts'"
        capability = eu.auto_capability_text(original)
        include, reason = eu.automatic_filter(original, capability)
        self.assertFalse(include)
        self.assertIn("multi_sentence", reason)

    def test_automatic_filter_rejects_stranded_preposition_capability(self):
        include, reason = eu.automatic_filter(
            "The system shall interface with CampusConnect's central server.",
            "with CampusConnect's central server",
        )
        self.assertFalse(include)
        self.assertIn("stranded_preposition", reason)

    def test_automatic_filter_rejects_residual_modal_capability(self):
        include, reason = eu.automatic_filter(
            "The system shall display a summary which will include cohort progress.",
            "display a summary which will include cohort progress",
        )
        self.assertFalse(include)
        self.assertIn("residual_modal_in_capability", reason)
        include, reason = eu.automatic_filter(
            "There shall be possibility to power the equipment set from the vehicle alternator.",
            "be possibility to power the equipment set from the vehicle alternator",
        )
        self.assertFalse(include)
        self.assertIn("residual_modal_in_capability", reason)

    def test_mlm_tapt_filter_rejects_fragments_and_messy_rows(self):
        cases = {
            "Initialization Data": "too_short",
            "The input data shall include: 1. Name 2. Address": "list_or_heading_marker",
            "The algorithm shall produce the output. NOTE 1 applies": "multi_sentence",
            "The scheduler shall set the interval timer. The display shall refresh": "multi_sentence",
            "The field shall match Table 8-1.": "table_or_figure_reference",
        }
        for requirement, expected_reason in cases.items():
            with self.subTest(requirement=requirement):
                include, reason = eu.mlm_tapt_filter(requirement, eu.auto_capability_text(requirement), source_corpus="source_WEB")
                self.assertFalse(include)
                self.assertIn(expected_reason, reason)

    def test_mlm_tapt_filter_accepts_clean_requirement(self):
        requirement = "The scheduler shall set the interval timer."
        include, reason = eu.mlm_tapt_filter(requirement, eu.auto_capability_text(requirement), source_corpus="source_WEB")

        self.assertTrue(include, reason)

    def test_make_mlm_tapt_seed_candidates_preserves_source_excludes_pure_and_dedupes(self):
        rows = [
            {"source": "alpha_WEB", "reqs": "The scheduler shall set the interval timer."},
            {"source": "alpha_WEB", "reqs": "The scheduler shall set the interval timer."},
            {"source": "beta_PURE", "reqs": "The report shall include diagnostic data."},
            {"source": "gamma_WEB", "reqs": "Initialization Data"},
            {"source": "delta_WEB", "reqs": "The display should show recent alerts."},
            {"source": "epsilon_WEB", "reqs": "Users may export reports."},
        ]

        candidates = eu.make_mlm_tapt_seed_candidates(rows, target_count=2, seed=123)
        selected = [row for row in candidates if row["include"] == "yes"]

        self.assertEqual(len(candidates), 5)
        self.assertEqual(len(selected), 2)
        self.assertEqual({row["source_dataset"] for row in candidates}, {"mlm_tapt"})
        self.assertIn("source_corpus", candidates[0])
        pure = [row for row in candidates if row["source_corpus"] == "beta_PURE"][0]
        self.assertEqual(pure["auto_include"], "no")
        self.assertIn("excluded_source", pure["auto_exclusion_reason"])

    def test_weighted_sample_candidate_indices_is_deterministic_and_caps_sources(self):
        candidates = []
        for source in ["a", "b", "c", "d", "e", "f"]:
            for index in range(50):
                candidates.append({"auto_include": "yes", "source_corpus": source, "id": f"{source}{index}"})

        first = eu.weighted_sample_candidate_indices(candidates, target_count=180, seed=42, source_cap=30)
        second = eu.weighted_sample_candidate_indices(candidates, target_count=180, seed=42, source_cap=30)
        counts = {}
        for index in first:
            source = candidates[index]["source_corpus"]
            counts[source] = counts.get(source, 0) + 1

        self.assertEqual(first, second)
        self.assertEqual(len(first), 180)
        self.assertLessEqual(max(counts.values()), 30)

    def test_artifact_path_combines_dataset_and_variant_suffixes(self):
        base = Path("data/processed/benchmark_items.csv")

        self.assertEqual(eu.artifact_path(base, "nice"), base)
        self.assertEqual(eu.artifact_path(base, "mlm_tapt"), Path("data/processed/benchmark_items_mlm_tapt.csv"))
        self.assertEqual(
            eu.artifact_path(base, "mlm_tapt", "shall"),
            Path("data/processed/benchmark_items_mlm_tapt_shall.csv"),
        )

    def test_parse_task2_response(self):
        raw = '{"requirement": "The system SHOULD export reports.", "modality": "should", "confidence": 80}'
        parsed, status = eu.parse_task_response("task2", raw)
        self.assertEqual(status, "ok")
        self.assertEqual(parsed["modality"], "recommended")

    def test_parse_task3_response_and_relation_aliases(self):
        raw = '{"relation": "stronger", "confidence": 82, "evidence_phrase": "MAY", "brief_reason": "upgraded"}'
        parsed, status = eu.parse_task_response("task3", raw)

        self.assertEqual(status, "ok")
        self.assertEqual(parsed["relation"], "strengthens")
        self.assertEqual(eu.normalize_relation("same modality"), "preserves")
        self.assertEqual(eu.normalize_relation("content mismatch"), "content_changed")

        _, status = eu.parse_task_response("task3", '{"relation":"preserves","confidence":50}')
        self.assertEqual(status, "missing_fields")

        _, status = eu.parse_task_response("task3", '{"relation":"unclear","confidence":50,"evidence_phrase":"MAY"}')
        self.assertEqual(status, "invalid_label")

    def test_task3_gold_relation_from_ordinal_modality(self):
        self.assertEqual(eu.task3_gold_relation("nice_to_have", "recommended"), "strengthens")
        self.assertEqual(eu.task3_gold_relation("optional", "mandatory"), "strengthens")
        self.assertEqual(eu.task3_gold_relation("mandatory", "optional"), "weakens")
        self.assertEqual(eu.task3_gold_relation("recommended", "recommended"), "preserves")

    def test_metrics(self):
        y_true = [1, 0, 1, 0]
        p = [0.9, 0.2, 0.8, 0.1]
        self.assertLess(eu.brier_score(y_true, p), 0.05)
        self.assertEqual(eu.auroc_score(y_true, p), 1.0)
        self.assertFalse(math.isnan(eu.ece_score(y_true, p)))
        rank_strength = [1.0, 0.67, 0.33, 0.0]
        recoded_strength = [1.0, 0.75, 0.33, 0.0]
        p_yes = [1.0, 0.05, 0.0, 0.05]
        self.assertEqual(eu.spearman_corr(rank_strength, p_yes), eu.spearman_corr(recoded_strength, p_yes))
        self.assertNotEqual(eu.pearson_corr(rank_strength, p_yes), eu.pearson_corr(recoded_strength, p_yes))

    def test_distribution_uncertainty_helpers(self):
        distribution = eu.label_distribution(["yes", "yes", "no"], ["yes", "no"])

        self.assertEqual(distribution, {"yes": 2 / 3, "no": 1 / 3})
        self.assertEqual(eu.majority_label({"yes": 0.5, "no": 0.5}, ["yes", "no"]), "yes")
        self.assertAlmostEqual(eu.variation_ratio(distribution), 1 / 3)
        self.assertGreater(eu.normalized_predictive_entropy(distribution), 0.0)
        self.assertLess(eu.normalized_predictive_entropy(distribution), 1.0)

    def test_monotonicity_violation(self):
        rows = [
            {"seed_id": "S1", "source_modality": "mandatory", "p_yes": 0.9},
            {"seed_id": "S1", "source_modality": "recommended", "p_yes": 0.6},
            {"seed_id": "S1", "source_modality": "optional", "p_yes": 0.3},
            {"seed_id": "S1", "source_modality": "nice_to_have", "p_yes": 0.1},
            {"seed_id": "S2", "source_modality": "mandatory", "p_yes": 0.9},
            {"seed_id": "S2", "source_modality": "recommended", "p_yes": 0.94},
            {"seed_id": "S2", "source_modality": "optional", "p_yes": 0.2},
            {"seed_id": "S2", "source_modality": "nice_to_have", "p_yes": 0.1},
            {"seed_id": "S3", "source_modality": "mandatory", "p_yes": 0.9},
            {"seed_id": "S3", "source_modality": "recommended", "p_yes": 0.96},
            {"seed_id": "S3", "source_modality": "optional", "p_yes": 0.2},
            {"seed_id": "S3", "source_modality": "nice_to_have", "p_yes": 0.1},
        ]
        diagnostics = eu.monotonicity_violation_diagnostics(rows)

        self.assertEqual(eu.monotonicity_violation_rate(rows), 1 / 3)
        self.assertEqual(eu.monotonicity_violation_rate(rows, tolerance=0.0), 2 / 3)
        self.assertEqual(diagnostics["monotonicity_violations"], 1 / 3)
        self.assertEqual(diagnostics["monotonicity_strict_violations"], 2 / 3)
        self.assertEqual(diagnostics["monotonicity_tolerance"], eu.MONOTONICITY_TOLERANCE)
        self.assertAlmostEqual(diagnostics["monotonicity_mean_max_increase"], (0.0 + 0.04 + 0.06) / 3)
        self.assertAlmostEqual(diagnostics["monotonicity_max_increase"], 0.06)

    def test_high_confidence_overcommitment_metrics(self):
        task1_rows = [
            {"task": "task1", "y_true": 0, "p_yes": 0.95},
            {"task": "task1", "y_true": 0, "p_yes": 0.20},
            {"task": "task1", "y_true": 1, "p_yes": 0.95},
        ]
        task2_rows = [
            {"task": "task2", "gold_modality": "optional", "pred_modality": "mandatory", "confidence": 0.90},
            {"task": "task2", "gold_modality": "recommended", "pred_modality": "optional", "confidence": 0.95},
        ]

        self.assertEqual(eu.high_confidence_overcommitment_rate(task1_rows, "task1", 0.80), 0.5)
        self.assertEqual(eu.high_confidence_overcommitment_rate(task2_rows, "task2", 0.80), 0.5)

    def test_task2_prompt_sensitivity_summary_counts_nice_to_have_upgrade(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        item = [row for row in benchmark if row["source_modality"] == "nice_to_have"][0]
        raw_rows = [
            {
                "run_id": "r1-default",
                "model": "m1:default",
                "task": "task2",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "parsed_json": {"requirement": "The system should export reports.", "modality": "recommended", "confidence": 95.0},
                "parse_status": "ok",
            },
            {
                "run_id": "r1-labels-only",
                "model": "m1:labels_only",
                "task": "task2",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "parsed_json": {"requirement": "It would be useful if the system could export reports.", "modality": "nice_to_have", "confidence": 90.0},
                "parse_status": "ok",
            },
        ]

        summary = eu.task2_prompt_sensitivity_summary(benchmark, raw_rows)
        by_model = {row["model"]: row for row in summary}

        self.assertEqual(by_model["m1:default"]["nice_to_have_accuracy"], 0.0)
        self.assertEqual(by_model["m1:default"]["nice_to_have_to_recommended_rate"], 1.0)
        self.assertEqual(by_model["m1:default"]["high_conf_overcommit_90"], 1.0)
        self.assertEqual(by_model["m1:labels_only"]["nice_to_have_accuracy"], 1.0)
        self.assertEqual(by_model["m1:labels_only"]["over_commitment"], 0.0)

    def test_weak_modality_probe_summary_counts_overcommitment(self):
        seeds = [
            {
                "seed_id": "S0001",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": "export reports",
            }
        ]
        items = eu.build_weak_modality_probe_items(seeds)
        item = [row for row in items if row["template_id"] == "useful_if"][0]
        raw_record = eu.build_raw_record(
            run_id="weak-probe-r1",
            model="m1",
            host="http://localhost:8000/v1",
            task="task2",
            item=item,
            sample_index=0,
            sample_kind="deterministic",
            temperature=0.0,
            top_p=1.0,
            prompt_version="v1",
            prompt="prompt",
            completion={
                "ok": True,
                "raw_text": '{"requirement":"The system should export reports.","modality":"recommended","confidence":95}',
                "latency_s": 0.1,
                "error": "",
            },
        )

        summary = eu.weak_modality_probe_summary(items, [raw_record])

        self.assertEqual(raw_record["template_id"], "useful_if")
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["template_id"], "useful_if")
        self.assertEqual(summary[0]["valid_n"], 1)
        self.assertEqual(summary[0]["accuracy"], 0.0)
        self.assertEqual(summary[0]["to_recommended_rate"], 1.0)
        self.assertEqual(summary[0]["over_commitment"], 1.0)
        self.assertEqual(summary[0]["high_conf_overcommit_90"], 1.0)

    def test_weak_modality_probe_summary_uses_instructor_confidence_scale(self):
        seeds = [
            {
                "seed_id": "S0001",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": "export reports",
            }
        ]
        items = eu.build_weak_modality_probe_items(seeds)
        item = [row for row in items if row["template_id"] == "useful_if"][0]
        raw_record = {
            "run_id": "weak-probe-r1",
            "model": "m1",
            "task": "task2",
            "item_id": item["item_id"],
            "seed_id": item["seed_id"],
            "source_modality": item["source_modality"],
            "sample_kind": "deterministic",
            "sample_index": 0,
            "parse_status": "ok",
            "parsed_json": {
                "requirement": "The system should export reports.",
                "modality": "recommended",
                "confidence": 0.95,
            },
            "output_contract_version": so.INSTRUCTOR_OUTPUT_CONTRACT_VERSION,
            "confidence_scale": so.INSTRUCTOR_CONFIDENCE_SCALE,
        }

        summary = eu.weak_modality_probe_summary(items, [raw_record])

        self.assertEqual(summary[0]["high_conf_overcommit_90"], 1.0)
        self.assertEqual(summary[0]["mean_confidence"], 0.95)

    def test_qualitative_examples_sorted_by_risk(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        optional_item = [row for row in benchmark if row["source_modality"] == "optional"][0]
        nice_item = [row for row in benchmark if row["source_modality"] == "nice_to_have"][0]
        scores = [
            {
                "model": "m1",
                "task": "task1",
                "uq_method": "verbalized_confidence",
                "item_id": optional_item["item_id"],
                "seed_id": optional_item["seed_id"],
                "source_modality": optional_item["source_modality"],
                "y_true": 0,
                "p_yes": 0.85,
            },
            {
                "model": "m1",
                "task": "task1",
                "uq_method": "verbalized_confidence",
                "item_id": nice_item["item_id"],
                "seed_id": nice_item["seed_id"],
                "source_modality": nice_item["source_modality"],
                "y_true": 0,
                "p_yes": 0.95,
            },
        ]
        examples = eu.qualitative_overcommitment_examples(scores, benchmark, limit=2, threshold=0.80)

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["risk_score"], 0.95)
        self.assertEqual(examples[0]["source_modality"], "nice_to_have")

    def test_build_uq_scores_and_summary(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        raw_rows = []
        for item in benchmark:
            decision = "yes" if item["source_modality"] == "mandatory" else "no"
            raw_rows.append(
                {
                    "run_id": "r1",
                    "model": "m1",
                    "host": "http://localhost:8000/v1",
                    "task": "task1",
                    "item_id": item["item_id"],
                    "seed_id": item["seed_id"],
                    "source_modality": item["source_modality"],
                    "sample_index": 0,
                    "sample_kind": "deterministic",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "prompt_version": "v1",
                    "raw_text": "",
                    "parsed_json": {"decision": decision, "confidence": 90.0, "brief_reason": ""},
                    "parse_status": "ok",
                    "latency_s": 0.1,
                    "error": "",
                }
            )
        scores = eu.build_uq_scores(benchmark, raw_rows)
        summary = eu.metric_summary_by_model_task_method(scores)
        self.assertEqual(len(scores), 4)
        self.assertEqual(summary[0]["accuracy"], 1.0)
        self.assertTrue(all("uncertainty_score" in row for row in scores))

    def test_stale_raw_prompt_rows_are_excluded_from_scores(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0088",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall interface with CampusConnect's central server.",
                    "capability_text_final": "interface with CampusConnect's central server",
                }
            ]
        )
        item = [row for row in benchmark if row["source_modality"] == "mandatory"][0]
        raw = {
            "run_id": "r1",
            "model": "m1",
            "host": "http://localhost:8000/v1",
            "task": "task1",
            "item_id": item["item_id"],
            "seed_id": item["seed_id"],
            "source_modality": item["source_modality"],
            "sample_index": 0,
            "sample_kind": "deterministic",
            "temperature": 0.0,
            "top_p": 1.0,
            "prompt_version": "v1",
            "raw_text": "",
            "parsed_json": {"decision": "yes", "confidence": 90.0, "brief_reason": ""},
            "parse_status": "ok",
            "latency_s": 0.1,
            "error": "",
            "prompt": "Source statement:\n\"The system MUST with CampusConnect's central server.\"",
        }

        self.assertEqual(eu.benchmark_rows_with_current_raw_outputs(benchmark, [raw]), [])
        self.assertEqual(eu.build_uq_scores(benchmark, [raw]), [])

        fresh = {**raw, "prompt": f"Source statement:\n\"{item['source_statement']}\""}
        self.assertEqual(len(eu.benchmark_rows_with_current_raw_outputs(benchmark, [fresh])), 1)
        self.assertEqual(len(eu.build_uq_scores(benchmark, [fresh])), 1)

    def test_build_task3_items_scores_and_summary(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        source_item = [row for row in benchmark if row["source_modality"] == "nice_to_have"][0]
        task2_raw = [
            {
                "run_id": "full-r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task2",
                "item_id": source_item["item_id"],
                "seed_id": source_item["seed_id"],
                "source_modality": source_item["source_modality"],
                "sample_index": 0,
                "sample_kind": "deterministic",
                "temperature": 0.0,
                "top_p": 1.0,
                "prompt_version": "v1",
                "raw_text": "",
                "parsed_json": {"requirement": "The system SHOULD export reports.", "modality": "recommended", "confidence": 95.0},
                "parse_status": "ok",
                "latency_s": 0.1,
                "error": "",
            }
        ]
        task3_items = eu.build_task3_verification_items(benchmark, task2_raw)
        item = task3_items[0]

        self.assertEqual(len(task3_items), 1)
        self.assertEqual(item["source_item_id"], source_item["item_id"])
        self.assertEqual(item["task2_modality"], "recommended")
        self.assertEqual(item["task3_gold_relation"], "strengthens")

        raw_rows = [
            {
                "run_id": "task3-r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task3",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "sample_index": 0,
                "sample_kind": "deterministic",
                "temperature": 0.0,
                "top_p": 1.0,
                "prompt_version": "v1:task3",
                "raw_text": "",
                "parsed_json": {
                    "relation": "preserves",
                    "confidence": 0.9,
                    "evidence_phrase": "It would be useful if",
                    "brief_reason": "missed upgrade",
                },
                "parse_status": "ok",
                "latency_s": 0.1,
                "error": "",
                "output_contract_version": so.INSTRUCTOR_OUTPUT_CONTRACT_VERSION,
                "confidence_scale": so.INSTRUCTOR_CONFIDENCE_SCALE,
            },
            {
                "run_id": "task3-r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task3",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "sample_index": 0,
                "sample_kind": "stochastic",
                "temperature": 0.7,
                "top_p": 1.0,
                "prompt_version": "v1:task3",
                "raw_text": "",
                "parsed_json": {"relation": "strengthens", "confidence": 80.0, "evidence_phrase": "It would be useful if"},
                "parse_status": "ok",
                "latency_s": 0.1,
                "error": "",
            },
            {
                "run_id": "task3-r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task3",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "sample_index": 1,
                "sample_kind": "stochastic",
                "temperature": 0.7,
                "top_p": 1.0,
                "prompt_version": "v1:task3",
                "raw_text": "",
                "parsed_json": {"relation": "strengthens", "confidence": 75.0, "evidence_phrase": "useful"},
                "parse_status": "ok",
                "latency_s": 0.1,
                "error": "",
            },
            {
                "run_id": "task3-r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task3",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "sample_index": 2,
                "sample_kind": "stochastic",
                "temperature": 0.7,
                "top_p": 1.0,
                "prompt_version": "v1:task3",
                "raw_text": "",
                "parsed_json": None,
                "parse_status": "invalid_json",
                "latency_s": 0.1,
                "error": "",
            },
        ]

        scores = eu.build_task3_scores(task3_items, raw_rows)
        summary = eu.metric_summary_by_model_task_method(scores)
        by_method = {row["uq_method"]: row for row in summary}

        self.assertEqual({row["task"] for row in scores}, {"task3"})
        verbalized_score = [row for row in scores if row["uq_method"] == "verbalized_confidence"][0]
        self.assertEqual(verbalized_score["confidence"], 0.9)
        self.assertEqual(by_method["verbalized_confidence"]["accuracy"], 0.0)
        self.assertEqual(by_method["verbalized_confidence"]["f1_or_macro_f1"], 0.0)
        self.assertEqual(by_method["verbalized_confidence"]["strengthening_recall"], 0.0)
        self.assertEqual(by_method["verbalized_confidence"]["false_preserve_rate"], 1.0)
        self.assertEqual(by_method["verbalized_confidence"]["evidence_phrase_source_rate"], 1.0)
        self.assertEqual(by_method["relation_consistency"]["accuracy"], 1.0)
        self.assertEqual(by_method["relation_consistency"]["f1_or_macro_f1"], 1.0)
        self.assertAlmostEqual(by_method["relation_consistency"]["parse_failure_rate"], 1 / 3)

    def test_task2_summary_accuracy_uses_modality_labels(self):
        scores = [
            {
                "model": "m1",
                "task": "task2",
                "uq_method": "verbalized_confidence",
                "item_id": "S0001_nice_to_have",
                "seed_id": "S0001",
                "source_modality": "nice_to_have",
                "ordinal_strength": 0,
                "numeric_strength": 0.0,
                "valid_n": 1,
                "total_n": 1,
                "parse_failures": 0,
                "y_true": 0,
                "y_pred": 0,
                "p_yes": "",
                "confidence": 0.95,
                "uncertainty_score": 0.05,
                "uncertainty_measure": "one_minus_confidence",
                "label_distribution": "",
                "gold_modality": "nice_to_have",
                "pred_modality": "recommended",
            },
            {
                "model": "m1",
                "task": "task2",
                "uq_method": "verbalized_confidence",
                "item_id": "S0001_optional",
                "seed_id": "S0001",
                "source_modality": "optional",
                "ordinal_strength": 1,
                "numeric_strength": 0.33,
                "valid_n": 1,
                "total_n": 1,
                "parse_failures": 0,
                "y_true": 1,
                "y_pred": 1,
                "p_yes": "",
                "confidence": 0.90,
                "uncertainty_score": 0.10,
                "uncertainty_measure": "one_minus_confidence",
                "label_distribution": "",
                "gold_modality": "optional",
                "pred_modality": "optional",
            },
        ]

        summary = eu.metric_summary_by_model_task_method(scores)
        acc_point, _, _ = eu.bootstrap_seed_metric(
            scores,
            lambda sample_rows: eu.task_accuracy(sample_rows, "task2"),
            iterations=10,
        )

        self.assertEqual(summary[0]["accuracy"], 0.5)
        self.assertEqual(acc_point, 0.5)
        self.assertEqual(summary[0]["f1_or_macro_f1"], eu.macro_f1_score(["nice_to_have", "optional"], ["recommended", "optional"], eu.MODALITIES))
        self.assertEqual(summary[0]["over_commitment"], 0.5)
        self.assertEqual(summary[0]["over_commitment_severity"], 1.0)
        self.assertEqual(summary[0]["over_commitment_severity_all"], 1.0)
        self.assertEqual(summary[0]["over_commitment_severity_given_overcommitment"], 2.0)
        self.assertEqual(summary[0]["weak_recall"], 0.0)
        self.assertEqual(summary[0]["weak_strengthening_90"], 1.0)
        self.assertEqual(summary[0]["high_conf_overcommit_all_90"], 0.5)
        self.assertEqual(summary[0]["high_conf_overcommit_overcommittable_90"], 0.5)

    def test_high_confidence_denominators_are_explicit(self):
        task1_rows = [
            {"task": "task1", "y_true": 0, "p_yes": 0.95},
            {"task": "task1", "y_true": 0, "p_yes": 0.40},
            {"task": "task1", "y_true": 1, "p_yes": 0.99},
        ]
        task2_rows = [
            {"task": "task2", "gold_modality": "nice_to_have", "pred_modality": "recommended", "confidence": 0.95},
            {"task": "task2", "gold_modality": "optional", "pred_modality": "optional", "confidence": 0.95},
            {"task": "task2", "gold_modality": "mandatory", "pred_modality": "mandatory", "confidence": 0.95},
        ]

        self.assertEqual(eu.unsupported_mandatory_acceptance_rate(task1_rows, 0.90), 0.5)
        self.assertEqual(eu.high_confidence_overcommitment_rate(task1_rows, "task1", 0.90), 0.5)
        self.assertEqual(eu.task2_high_confidence_overcommitment_rate(task2_rows, 0.90, denominator="all"), 1 / 3)
        self.assertEqual(eu.task2_high_confidence_overcommitment_rate(task2_rows, 0.90, denominator="overcommittable"), 0.5)
        self.assertEqual(eu.weak_strengthening_rate(task2_rows, 0.90), 1.0)

    def test_task2_deterministic_scores_include_text_modality_fields(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        optional_item = [row for row in benchmark if row["source_modality"] == "optional"][0]
        raw_rows = [
            {
                "run_id": "r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task2",
                "item_id": optional_item["item_id"],
                "seed_id": optional_item["seed_id"],
                "source_modality": optional_item["source_modality"],
                "sample_index": 0,
                "sample_kind": "deterministic",
                "temperature": 0.0,
                "top_p": 1.0,
                "prompt_version": "v1",
                "raw_text": "",
                "parsed_json": {"requirement": "The system SHALL export reports.", "modality": "optional", "confidence": 98.0},
                "parse_status": "ok",
                "latency_s": 0.1,
                "error": "",
            }
        ]

        scores = eu.build_uq_scores(benchmark, raw_rows)
        summary = eu.metric_summary_by_model_task_method(scores)

        self.assertEqual(scores[0]["text_modality"], "mandatory")
        self.assertFalse(scores[0]["label_text_consistent"])
        self.assertTrue(scores[0]["text_overcommit"])
        self.assertEqual(summary[0]["accuracy"], 1.0)
        self.assertEqual(summary[0]["text_modality_accuracy"], 0.0)
        self.assertEqual(summary[0]["text_modality_accuracy_all"], 0.0)
        self.assertEqual(summary[0]["text_modality_parse_coverage"], 1.0)
        self.assertEqual(summary[0]["text_high_conf_overcommit_90"], 1.0)

    def test_text_modality_summary_reports_coverage_and_all_row_accuracy(self):
        rows = [
            {
                "text_modality_parse_status": "ok",
                "text_modality_correct": True,
                "label_text_consistent": True,
                "text_overcommit": False,
                "text_undercommit": False,
                "text_high_conf_overcommit_80": False,
                "text_high_conf_overcommit_90": False,
            },
            {
                "text_modality_parse_status": "unknown",
                "text_modality_correct": False,
                "label_text_consistent": False,
                "text_overcommit": False,
                "text_undercommit": False,
                "text_high_conf_overcommit_80": False,
                "text_high_conf_overcommit_90": False,
            },
        ]

        metrics = eu.text_modality_summary_metrics(rows)

        self.assertEqual(metrics["text_modality_accuracy"], 1.0)
        self.assertEqual(metrics["text_modality_accuracy_all"], 0.5)
        self.assertEqual(metrics["text_modality_parse_coverage"], 0.5)

        empty_metrics = eu.text_modality_summary_metrics([{"text_modality_parse_status": ""}])
        self.assertEqual(empty_metrics["text_modality_accuracy"], "")
        self.assertEqual(empty_metrics["text_modality_accuracy_all"], "")
        self.assertEqual(empty_metrics["text_modality_parse_coverage"], "")

    def test_external_evaluator_counts_label_text_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gold_path = root / "gold.csv"
            output_path = root / "outputs.jsonl"
            eu.write_csv_rows(
                gold_path,
                [
                    {
                        "external_item_id": "EXT0001",
                        "source_kind": "main_benchmark",
                        "original_item_id": "S0001_optional",
                        "seed_id": "S0001",
                        "source_condition": "optional",
                        "source_modality": "optional",
                        "task2_gold_modality": "optional",
                        "capability_text": "export reports",
                        "source_statement": "The system MAY export reports.",
                    }
                ],
            )
            output_path.write_text(
                json.dumps(
                    {
                        "external_item_id": "EXT0001",
                        "requirement": "The system SHALL export reports.",
                        "modality": "optional",
                        "confidence": 0.98,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            scored, validation = external_eval.evaluate_outputs(output_path, gold_path)
            summary = external_eval.source_condition_summary(scored)

            self.assertEqual(validation["parse_errors"], 0)
            self.assertEqual(validation["confidence_scale"], eu.CONFIDENCE_SCALE_0_1)
            self.assertTrue(validation["paper_ready"])
            self.assertIn("raw_output_sha256", validation)
            self.assertTrue(bool(scored.iloc[0]["correct"]))
            self.assertFalse(bool(scored.iloc[0]["label_text_consistent"]))
            self.assertFalse(bool(scored.iloc[0]["text_modality_correct"]))
            self.assertTrue(bool(scored.iloc[0]["text_overcommit"]))
            self.assertEqual(summary.iloc[0]["accuracy"], 1.0)
            self.assertEqual(summary.iloc[0]["text_modality_accuracy"], 0.0)

    def test_external_evaluator_rejects_duplicate_external_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gold_path = root / "gold.csv"
            output_path = root / "outputs.jsonl"
            eu.write_csv_rows(
                gold_path,
                [
                    {
                        "external_item_id": "EXT0001",
                        "source_kind": "main_benchmark",
                        "original_item_id": "S0001_optional",
                        "seed_id": "S0001",
                        "source_condition": "optional",
                        "source_modality": "optional",
                        "task2_gold_modality": "optional",
                        "capability_text": "export reports",
                        "source_statement": "The system MAY export reports.",
                    }
                ],
            )
            duplicate_row = {
                "external_item_id": "EXT0001",
                "requirement": "The system MAY export reports.",
                "modality": "optional",
                "confidence": 0.9,
            }
            output_path.write_text(
                json.dumps(duplicate_row) + "\n" + json.dumps({**duplicate_row, "confidence": 0.8}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate external_item_id.*EXT0001"):
                external_eval.evaluate_outputs(output_path, gold_path)

    def test_external_comparison_skips_scored_items_without_current_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = [
                {
                    "correct": True,
                    "overcommit": False,
                    "high_conf_overcommit_90": False,
                    "text_modality_correct": True,
                    "label_text_consistency": True,
                    "text_overcommit": False,
                    "text_high_conf_overcommit_90": False,
                }
            ]
            eu.write_csv_rows(root / "current_model_scored_items.csv", rows)
            eu.write_csv_rows(root / "stale_model_scored_items.csv", rows)
            (root / "current_model_evaluation.md").write_text(
                "- Paper-ready under current contract: no\n",
                encoding="utf-8",
            )

            report_path = external_eval.write_external_comparison_report(root)

            self.assertIsNotNone(report_path)
            text = report_path.read_text(encoding="utf-8")
            self.assertIn("current_model", text)
            self.assertIn("paper_ready", text)
            self.assertNotIn("stale_model", text)

    def test_external_evaluator_uses_confidence_probability_scale(self):
        self.assertTrue(external_eval.valid_probability_confidence(0.95))
        self.assertFalse(external_eval.valid_probability_confidence(95))
        self.assertFalse(external_eval.valid_probability_confidence("95%"))

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gold_path = root / "gold.csv"
            output_path = root / "outputs.jsonl"
            eu.write_csv_rows(
                gold_path,
                [
                    {
                        "external_item_id": "EXT0001",
                        "source_kind": "main_benchmark",
                        "original_item_id": "S0001_optional",
                        "seed_id": "S0001",
                        "source_condition": "optional",
                        "source_modality": "optional",
                        "task2_gold_modality": "optional",
                        "capability_text": "export reports",
                        "source_statement": "The system MAY export reports.",
                    },
                    {
                        "external_item_id": "EXT0002",
                        "source_kind": "main_benchmark",
                        "original_item_id": "S0001_recommended",
                        "seed_id": "S0001",
                        "source_condition": "recommended",
                        "source_modality": "recommended",
                        "task2_gold_modality": "recommended",
                        "capability_text": "export reports",
                        "source_statement": "The system SHOULD export reports.",
                    },
                ],
            )
            output_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "external_item_id": "EXT0001",
                                "requirement": "The system MAY export reports.",
                                "modality": "optional",
                                "confidence": 0.95,
                            }
                        ),
                        json.dumps(
                            {
                                "external_item_id": "EXT0002",
                                "requirement": "The system SHOULD export reports.",
                                "modality": "recommended",
                                "confidence": 95,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            scored, validation = external_eval.evaluate_outputs(output_path, gold_path)

            self.assertEqual(validation["invalid_confidence_count"], 1)
            self.assertFalse(validation["paper_ready"])
            self.assertIn("invalid_confidence", validation["paper_ready_blockers"])
            self.assertEqual(float(scored.loc[scored["external_item_id"].eq("EXT0001"), "confidence_num"].iloc[0]), 0.95)

    def test_stochastic_uq_scores(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        item = benchmark[0]
        raw_rows = [
            {
                "run_id": "r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task1",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "sample_index": index,
                "sample_kind": "stochastic",
                "temperature": 0.7,
                "top_p": 1.0,
                "prompt_version": "v1",
                "raw_text": "",
                "parsed_json": {"decision": "yes" if index < 4 else "no", "confidence": 70.0, "brief_reason": ""},
                "parse_status": "ok",
                "latency_s": 0.1,
                "error": "",
            }
            for index in range(5)
        ]
        scores = eu.build_uq_scores(benchmark, raw_rows)
        by_method = {row["uq_method"]: row for row in scores}

        self.assertEqual(set(by_method), {"label_self_consistency", "predictive_entropy", "variation_ratio"})
        self.assertAlmostEqual(by_method["label_self_consistency"]["p_yes"], 0.8)
        self.assertAlmostEqual(by_method["variation_ratio"]["uncertainty_score"], 0.2)
        self.assertEqual(by_method["predictive_entropy"]["uncertainty_measure"], "normalized_entropy")
        self.assertGreater(by_method["predictive_entropy"]["uncertainty_score"], by_method["variation_ratio"]["uncertainty_score"])

    def test_stochastic_parse_failures_are_retained_in_score_counts(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        item = benchmark[0]
        raw_rows = [
            {
                "run_id": "r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task1",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "sample_index": index,
                "sample_kind": "stochastic",
                "temperature": 0.7,
                "top_p": 1.0,
                "prompt_version": "v1",
                "raw_text": "",
                "parsed_json": {"decision": "yes", "confidence": 70.0, "brief_reason": ""},
                "parse_status": "ok",
                "latency_s": 0.1,
                "error": "",
            }
            for index in range(4)
        ]
        raw_rows.append({**raw_rows[0], "sample_index": 4, "parsed_json": None, "parse_status": "invalid_json"})

        scores = eu.build_uq_scores(benchmark, raw_rows)

        self.assertTrue(all(row["valid_n"] == 4 for row in scores))
        self.assertTrue(all(row["total_n"] == 5 for row in scores))
        self.assertTrue(all(row["parse_failures"] == 1 for row in scores))

    def test_ensemble_disagreement_scores_task1(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        item = benchmark[0]
        raw_rows = []
        for model, decision in [("m1", "yes"), ("m2", "no")]:
            raw_rows.append(
                {
                    "run_id": "r1",
                    "model": model,
                    "host": "http://localhost:8000/v1",
                    "task": "task1",
                    "item_id": item["item_id"],
                    "seed_id": item["seed_id"],
                    "source_modality": item["source_modality"],
                    "sample_index": 0,
                    "sample_kind": "deterministic",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "prompt_version": "v1",
                    "raw_text": "",
                    "parsed_json": {"decision": decision, "confidence": 80.0, "brief_reason": ""},
                    "parse_status": "ok",
                    "latency_s": 0.1,
                    "error": "",
                }
            )

        scores = eu.build_uq_scores(benchmark, raw_rows)
        ensemble = [row for row in scores if row["uq_method"] == "model_ensemble_disagreement"]

        self.assertEqual(len(ensemble), 1)
        self.assertEqual(ensemble[0]["model"], "ensemble:2_models")
        self.assertEqual(ensemble[0]["y_pred"], 1)
        self.assertAlmostEqual(ensemble[0]["p_yes"], 0.5)
        self.assertAlmostEqual(ensemble[0]["uncertainty_score"], 0.5)

    def test_ensemble_disagreement_scores_task2(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        item = benchmark[2]
        raw_rows = []
        for model, modality in [("m1", "optional"), ("m2", "mandatory"), ("m3", "mandatory")]:
            raw_rows.append(
                {
                    "run_id": "r1",
                    "model": model,
                    "host": "http://localhost:8000/v1",
                    "task": "task2",
                    "item_id": item["item_id"],
                    "seed_id": item["seed_id"],
                    "source_modality": item["source_modality"],
                    "sample_index": 0,
                    "sample_kind": "deterministic",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "prompt_version": "v1",
                    "raw_text": "",
                    "parsed_json": {"requirement": "The system MUST export reports.", "modality": modality, "confidence": 80.0},
                    "parse_status": "ok",
                    "latency_s": 0.1,
                    "error": "",
                }
            )

        scores = eu.build_uq_scores(benchmark, raw_rows)
        ensemble = [row for row in scores if row["uq_method"] == "model_ensemble_disagreement"]

        self.assertEqual(len(ensemble), 1)
        self.assertEqual(ensemble[0]["pred_modality"], "mandatory")
        self.assertEqual(ensemble[0]["gold_modality"], "optional")
        self.assertAlmostEqual(ensemble[0]["confidence"], 2 / 3)

    def test_error_detection_auroc_uses_uncertainty_score(self):
        rows = [
            {"task": "task1", "y_true": 1, "y_pred": 1, "uncertainty_score": 0.1},
            {"task": "task1", "y_true": 0, "y_pred": 1, "uncertainty_score": 0.9},
            {"task": "task1", "y_true": 0, "y_pred": 0, "uncertainty_score": 0.2},
            {"task": "task1", "y_true": 1, "y_pred": 0, "uncertainty_score": 0.8},
        ]

        self.assertEqual(eu.error_detection_auroc(rows, "task1"), 1.0)

    def test_selective_deferral_reports_retained_error_rate(self):
        rows = [
            {"task": "task1", "y_true": 1, "y_pred": 1, "uncertainty_score": 0.1},
            {"task": "task1", "y_true": 0, "y_pred": 1, "uncertainty_score": 0.9},
            {"task": "task1", "y_true": 0, "y_pred": 0, "uncertainty_score": 0.2},
            {"task": "task1", "y_true": 1, "y_pred": 1, "uncertainty_score": 0.3},
        ]

        metrics = eu.selective_deferral_metrics(rows, "task1", defer_fractions=(0.25,))

        self.assertEqual(metrics["selective_coverage_defer_25"], 0.75)
        self.assertEqual(metrics["selective_error_defer_25"], 0.0)

    def test_headline_risk_ci_fields_cover_task1_and_task2_metrics(self):
        task1_rows = [
            {"seed_id": "S0001", "task": "task1", "y_true": 0, "p_yes": 0.95},
            {"seed_id": "S0002", "task": "task1", "y_true": 0, "p_yes": 0.40},
            {"seed_id": "S0003", "task": "task1", "y_true": 1, "p_yes": 0.99},
        ]
        task2_rows = [
            {
                "seed_id": "S0001",
                "task": "task2",
                "gold_modality": "nice_to_have",
                "pred_modality": "recommended",
                "confidence": 0.95,
            },
            {
                "seed_id": "S0002",
                "task": "task2",
                "gold_modality": "optional",
                "pred_modality": "optional",
                "confidence": 0.95,
            },
        ]

        task1_fields = eu.headline_risk_ci_fields(task1_rows, "task1", iterations=10)
        task2_fields = eu.headline_risk_ci_fields(task2_rows, "task2", iterations=10)

        self.assertIn("unsupported_mandatory_acceptance_90_ci_low", task1_fields)
        self.assertIn("unsupported_mandatory_acceptance_90_ci_high", task1_fields)
        self.assertIn("high_conf_overcommit_overcommittable_90_ci_low", task2_fields)
        self.assertIn("weak_strengthening_90_ci_high", task2_fields)

    def test_response_logprob_tokens_are_detected(self):
        response_json = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"token": " yes", "logprob": -0.1, "top_logprobs": []},
                        ]
                    }
                }
            ]
        }

        tokens = eu.response_logprob_tokens(response_json)

        self.assertTrue(eu.completion_has_logprobs(response_json))
        self.assertEqual(tokens[0]["token"], " yes")
        self.assertEqual(tokens[0]["logprob"], -0.1)

    def test_responses_logprob_tokens_are_detected(self):
        response_json = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "yes",
                            "logprobs": [
                                {"token": "yes", "logprob": -0.2, "top_logprobs": [{"token": "no", "logprob": -2.0}]},
                            ],
                        }
                    ],
                }
            ]
        }

        tokens = eu.response_logprob_tokens(response_json)

        self.assertTrue(eu.completion_has_logprobs(response_json))
        self.assertEqual(eu.responses_output_text(response_json), "yes")
        self.assertEqual(tokens[0]["token"], "yes")
        self.assertEqual(tokens[0]["top_logprobs"][0]["token"], "no")

    def test_responses_endpoint_url(self):
        self.assertEqual(eu.responses_endpoint_url("http://localhost:1234/v1"), "http://localhost:1234/v1/responses")
        self.assertEqual(eu.responses_endpoint_url("http://localhost:1234"), "http://localhost:1234/v1/responses")
        self.assertEqual(eu.responses_endpoint_url("http://localhost:1234/v1/responses"), "http://localhost:1234/v1/responses")

    def test_resolve_llm_concurrency_uses_config_and_env_override(self):
        self.assertEqual(eu.resolve_llm_concurrency({"llm": {"concurrency": 3}}, env={}), 3)
        self.assertEqual(eu.resolve_llm_concurrency({"llm": {"concurrency": 3}}, env={"LLM_CONCURRENCY": "7"}), 7)
        with self.assertRaises(ValueError):
            eu.resolve_llm_concurrency({"llm": {"concurrency": 3}}, env={"LLM_CONCURRENCY": "0"})
        with self.assertRaises(ValueError):
            eu.resolve_llm_concurrency({"llm": {"concurrency": "nope"}}, env={})

    def test_run_config_parsing_and_manual_server_guard(self):
        config = eu.normalize_run_config(
            {
                "run_group_id": "group1",
                "datasets": ["nice", "mlm_tapt"],
                "benchmark_variants": ["must"],
                "logging": {"progress_every_records": 5, "warn_parse_failure_rate": 0.1},
                "profiles": [
                    {
                        "profile_id": "local",
                        "provider_id": "llama_cpp",
                        "base_url": "http://127.0.0.1:1234/v1/",
                        "api_key_env": "LOCAL_OPENAI_API_KEY",
                        "models": ["m1", "m2"],
                        "requires_manual_server": True,
                        "batch_size": 4,
                    },
                    {
                        "profile_id": "zai",
                        "provider_id": "zai",
                        "base_url": "https://api.z.ai/api/coding/paas/v4/",
                        "api_key_env": "ZAI_API_KEY",
                        "models": ["glm-5.1"],
                        "json_mode": True,
                        "extra_body": {"thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}},
                    },
                ],
            }
        )

        self.assertEqual(config["profiles"][0]["base_url"], "http://127.0.0.1:1234/v1")
        self.assertEqual(config["profiles"][0]["batch_size"], 4)
        self.assertEqual(config["logging"]["progress_every_records"], 5)
        self.assertEqual(config["logging"]["warn_parse_failure_rate"], 0.1)
        self.assertEqual(config["profiles"][1]["base_url"], "https://api.z.ai/api/coding/paas/v4")
        self.assertEqual(config["profiles"][1]["response_format"], None)
        self.assertEqual(config["profiles"][1]["structured_output"], "json_object")
        self.assertEqual(config["profiles"][1]["extra_body"]["thinking"]["type"], "disabled")
        with self.assertRaises(ValueError):
            eu.validate_manual_server_profile(config["profiles"][0])

        selected = eu.filter_run_profiles(config, profile_id="local", model="m2")
        eu.validate_manual_server_profile(selected[0])
        self.assertEqual(selected[0]["models"], ["m2"])

    def test_load_run_config_accepts_trailing_commas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "current_run.json"
            config_path.write_text(
                """
{
  "run_group_id": "group1",
  "datasets": ["nice",],
  "benchmark_variants": ["must",],
  "profiles": [
    {
      "profile_id": "zai",
      "provider_id": "zai",
      "base_url": "https://api.z.ai/api/coding/paas/v4",
      "api_key_env": "ZAI_API_KEY",
      "models": ["glm-5.1",],
      "json_mode": true,
      "structured_output": "json_object",
    },
  ],
}
""",
                encoding="utf-8",
            )

            config = eu.load_run_config(config_path)

        self.assertEqual(config["datasets"], ["nice"])
        self.assertEqual(config["profiles"][0]["models"], ["glm-5.1"])
        self.assertEqual(config["profiles"][0]["structured_output"], "json_object")

    def test_run_logging_config_override_precedence(self):
        logging_config = eu.normalize_run_logging_config(
            {"progress_every_records": 50, "write_progress_csv": False},
            overrides={"progress_every_records": 10, "warn_after_records": 0, "write_event_jsonl": False},
        )

        self.assertEqual(logging_config["progress_every_records"], 10)
        self.assertEqual(logging_config["warn_after_records"], 0)
        self.assertFalse(logging_config["write_progress_csv"])
        self.assertFalse(logging_config["write_event_jsonl"])

    def test_provider_request_metadata_and_extra_body_are_preserved(self):
        seeds = [
            {
                "seed_id": "S0001",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": "export reports",
            }
        ]
        benchmark = eu.build_benchmark_items(seeds)
        captured = {}
        jobs = eu.planned_completion_jobs(
            benchmark[:1],
            tasks=["task1"],
            model="glm-5.1",
            host="https://api.z.ai/api/coding/paas/v4",
            run_id="full-1",
            prompt_version="v1",
            task1_template=eu.load_prompt("prompts/mandatory_entailment.txt"),
            task2_template=eu.load_prompt("prompts/modality_extraction.txt"),
            deterministic={"temperature": 0.0, "top_p": 1.0, "samples": 1},
            stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 0},
            max_tokens=64,
            timeout_s=30,
            api_key_env="ZAI_API_KEY",
            provider_id="zai",
            profile_id="zai",
            run_group_id="group1",
            json_mode=True,
            extra_body={"thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}},
        )

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.8, "brief_reason": "ok"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        record = eu.run_completion_job(jobs[0], completion_fn=fake_completion)

        self.assertEqual(captured["extra_body"]["thinking"]["type"], "disabled")
        self.assertEqual(record["provider_id"], "zai")
        self.assertEqual(record["profile_id"], "zai")
        self.assertEqual(record["run_group_id"], "group1")
        self.assertTrue(record["json_mode"])
        self.assertEqual(record["request_extra_body"]["response_format"]["type"], "json_object")

    def test_provider_preflight_does_not_duplicate_extra_body_response_format(self):
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 100, "brief_reason": "probe"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        preflight = eu.provider_preflight(
            host="https://api.z.ai/api/coding/paas/v4",
            model="glm-5.1",
            api_key_env="ZAI_API_KEY",
            timeout_s=30,
            json_mode=True,
            extra_body={"thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}},
            completion_fn=fake_completion,
        )

        self.assertTrue(preflight["ok"])
        self.assertIsNone(captured["response_format"])
        self.assertEqual(captured["extra_body"]["response_format"]["type"], "json_object")

    def test_json_schema_structured_output_uses_task_schema(self):
        profile = eu.normalize_provider_profile(
            {
                "profile_id": "institutional",
                "provider_id": "institutional_llm",
                "base_url": "https://institutional-llm.example.invalid/api/v1",
                "api_key_env": "INSTITUTIONAL_LLM_API_KEY",
                "models": ["azure.gpt-4.1-mini"],
                "json_schema": True,
            }
        )
        seeds = [
            {
                "seed_id": "S0001",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": "export reports",
            }
        ]
        benchmark = eu.build_benchmark_items(seeds)
        jobs = eu.planned_completion_jobs(
            benchmark[:1],
            tasks=["task2"],
            model="azure.gpt-4.1-mini",
            host=profile["base_url"],
            run_id="full-1",
            prompt_version="v1",
            task1_template=eu.load_prompt("prompts/mandatory_entailment.txt"),
            task2_template=eu.load_prompt("prompts/modality_extraction.txt"),
            deterministic={"temperature": 0.0, "top_p": 1.0, "samples": 1},
            stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 0},
            max_tokens=64,
            timeout_s=30,
            api_key_env=profile["api_key_env"],
            json_mode=profile["json_mode"],
            structured_output=profile["structured_output"],
            response_format=profile["response_format"],
            extra_body=profile["extra_body"],
        )

        response_format = jobs[0]["response_format"]
        self.assertEqual(profile["structured_output"], "json_schema")
        self.assertEqual(response_format["type"], "json_schema")
        schema = response_format["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["modality"]["enum"], ["mandatory", "recommended", "optional", "nice_to_have"])
        self.assertTrue(response_format["json_schema"]["strict"])

    def test_json_schema_batch_uses_results_schema(self):
        seeds = [
            {
                "seed_id": "S0001",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": "export reports",
            },
            {
                "seed_id": "S0002",
                "source_dataset": "NICE",
                "original_requirement": "The system shall print invoices.",
                "capability_text_final": "print invoices",
            },
        ]
        benchmark = eu.build_benchmark_items(seeds)
        jobs = eu.planned_completion_jobs(
            [row for row in benchmark if row["source_modality"] == "mandatory"],
            tasks=["task1"],
            model="m1",
            host="http://localhost:1234/v1",
            run_id="full-1",
            prompt_version="v1",
            task1_template=eu.load_prompt("prompts/mandatory_entailment.txt"),
            task2_template=eu.load_prompt("prompts/modality_extraction.txt"),
            deterministic={"temperature": 0.0, "top_p": 1.0, "samples": 1},
            stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 0},
            max_tokens=64,
            timeout_s=30,
            api_key_env="LOCAL_OPENAI_API_KEY",
            json_mode=True,
            structured_output="json_schema",
        )
        captured = {}

        def fake_batch_completion(**kwargs):
            captured.update(kwargs)
            items = json.loads(kwargs["prompt"].split("Items:\n", 1)[1])
            return {
                "ok": True,
                "raw_text": json.dumps(
                    {
                        "results": [
                            {
                                "request_index": item["request_index"],
                                "decision": "yes",
                                "confidence": 0.8,
                                "brief_reason": "batched",
                            }
                            for item in items
                        ]
                    }
                ),
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(eu.run_completion_jobs(jobs, max_workers=1, completion_fn=fake_batch_completion, batch_size=2))

        schema = captured["response_format"]["json_schema"]["schema"]
        self.assertIn("results", schema["properties"])
        self.assertIn("request_index", schema["properties"]["results"]["items"]["properties"])
        self.assertEqual({row["structured_output"] for row in records}, {"json_schema"})
        self.assertEqual({row["response_format"]["json_schema"]["name"] for row in records}, {"re_uq_task1_batch"})

    def test_json_schema_replaces_extra_body_response_format(self):
        response_format, extra_body = eu.resolve_response_format_args(
            "task1",
            structured_output="json_schema",
            extra_body={"thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}},
        )

        self.assertIsNone(response_format)
        self.assertEqual(extra_body["thinking"]["type"], "disabled")
        self.assertEqual(extra_body["response_format"]["type"], "json_schema")

    def test_instructor_profile_strips_response_format_and_sets_defaults(self):
        profile = eu.normalize_provider_profile(
            {
                "profile_id": "zai",
                "provider_id": "zai",
                "base_url": "https://api.z.ai/api/coding/paas/v4",
                "models": ["glm-5.1"],
                "structured_output": "instructor",
                "extra_body": {"thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}},
            }
        )

        self.assertEqual(profile["structured_output"], "instructor")
        self.assertEqual(profile["instructor_mode"], "json")
        self.assertEqual(profile["validation_retries"], 2)
        self.assertEqual(profile["fallback_batch_size"], 1)
        self.assertEqual(profile["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertIsNone(profile["response_format"])

    def test_instructor_completion_passes_response_model_retries_mode_and_clean_extra_body(self):
        import instructor

        captured = {}

        class ParsedResponse:
            def model_dump(self, mode):
                captured["dump_mode"] = mode
                return {"decision": "yes", "confidence": 0.82, "brief_reason": "valid"}

        class DummyCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return ParsedResponse()

        class DummyChat:
            completions = DummyCompletions()

        class DummyInstructorClient:
            chat = DummyChat()

        with mock.patch("scripts.eval_utils.OpenAI") as openai_cls, mock.patch(
            "instructor.from_openai", return_value=DummyInstructorClient()
        ) as from_openai:
            result = eu.instructor_completion(
                host="http://localhost:1234/v1",
                model="m1",
                prompt="prompt",
                temperature=0.0,
                top_p=1.0,
                max_tokens=64,
                timeout_s=30,
                api_key_env="LOCAL_OPENAI_API_KEY",
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}},
                task="task1",
                instructor_mode="json",
                validation_retries=3,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(json.loads(result["raw_text"])["confidence"], 0.82)
        self.assertEqual(captured["response_model"], so.Task1Response)
        self.assertEqual(captured["max_retries"], 3)
        self.assertEqual(captured["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("response_format", captured)
        self.assertEqual(captured["dump_mode"], "json")
        openai_cls.assert_called_once()
        self.assertEqual(from_openai.call_args.kwargs["mode"], instructor.Mode.JSON)

    def test_instructor_completion_validation_retry_failure_is_marked_for_parser(self):
        class InstructorValidationFailure(Exception):
            pass

        class DummyCompletions:
            def create(self, **kwargs):
                exc = InstructorValidationFailure("bad model output")
                exc.last_completion = '{"decision":"yes","confidence":95,"brief_reason":"bad scale"}'
                raise exc

        class DummyChat:
            completions = DummyCompletions()

        class DummyInstructorClient:
            chat = DummyChat()

        with mock.patch("scripts.eval_utils.OpenAI"), mock.patch(
            "instructor.from_openai", return_value=DummyInstructorClient()
        ):
            result = eu.instructor_completion(
                host="http://localhost:1234/v1",
                model="m1",
                prompt="prompt",
                temperature=0.0,
                top_p=1.0,
                task="task1",
                validation_retries=2,
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["parse_status_override"], "instructor_validation_error")
        self.assertIn("confidence", result["raw_text"])

    def test_instructor_single_item_writes_validated_contract_markers(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        jobs = eu.planned_completion_jobs(
            benchmark[:1],
            tasks=["task1"],
            model="m1",
            host="http://localhost:1234/v1",
            run_id="full-1",
            prompt_version="v2-instructor-conf01",
            task1_template=eu.load_prompt("prompts/mandatory_entailment.txt"),
            task2_template=eu.load_prompt("prompts/modality_extraction.txt"),
            deterministic={"temperature": 0.0, "top_p": 1.0, "samples": 1},
            stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 0},
            max_tokens=64,
            timeout_s=30,
            api_key_env="LOCAL_OPENAI_API_KEY",
            structured_output="instructor",
            extra_body={"thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}},
        )
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.8, "brief_reason": "ok"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        record = eu.run_completion_job(jobs[0], completion_fn=fake_completion)

        self.assertEqual(captured["task"], "task1")
        self.assertFalse(captured["batched"])
        self.assertEqual(captured["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(record["parse_status"], "ok")
        self.assertEqual(record["parsed_json"]["confidence"], 0.8)
        self.assertEqual(record["output_contract_version"], so.INSTRUCTOR_OUTPUT_CONTRACT_VERSION)
        self.assertEqual(record["confidence_scale"], so.INSTRUCTOR_CONFIDENCE_SCALE)

    def test_instructor_batch_partial_results_fall_back_unbatched(self):
        seeds = [
            {
                "seed_id": "S0001",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": "export reports",
            },
            {
                "seed_id": "S0002",
                "source_dataset": "NICE",
                "original_requirement": "The system shall print invoices.",
                "capability_text_final": "print invoices",
            },
        ]
        benchmark = eu.build_benchmark_items(seeds)
        jobs = eu.planned_completion_jobs(
            [row for row in benchmark if row["source_modality"] == "mandatory"],
            tasks=["task1"],
            model="m1",
            host="http://localhost:1234/v1",
            run_id="full-1",
            prompt_version="v2-instructor-conf01",
            task1_template=eu.load_prompt("prompts/mandatory_entailment.txt"),
            task2_template=eu.load_prompt("prompts/modality_extraction.txt"),
            deterministic={"temperature": 0.0, "top_p": 1.0, "samples": 1},
            stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 0},
            max_tokens=64,
            timeout_s=30,
            api_key_env="LOCAL_OPENAI_API_KEY",
            structured_output="instructor",
        )
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            if kwargs["batched"]:
                items = json.loads(kwargs["prompt"].split("Items:\n", 1)[1])
                return {
                    "ok": True,
                    "raw_text": json.dumps(
                        {
                            "results": [
                                {
                                    "request_index": items[0]["request_index"],
                                    "decision": "yes",
                                    "confidence": 0.8,
                                    "brief_reason": "batched",
                                }
                            ]
                        }
                    ),
                    "response_json": {},
                    "latency_s": 0.01,
                    "error": "",
                }
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.9, "brief_reason": "fallback"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(eu.run_completion_jobs(jobs, max_workers=1, completion_fn=fake_completion, batch_size=2))

        self.assertEqual([call["batched"] for call in calls], [True, False])
        self.assertEqual(len(records), 2)
        self.assertEqual({record["parse_status"] for record in records}, {"ok"})
        self.assertEqual(records[0]["parsed_json"]["confidence"], 0.8)
        self.assertEqual(records[1]["parsed_json"]["confidence"], 0.9)

    def test_instructor_batch_invalid_item_falls_back_unbatched(self):
        jobs = self._instructor_task1_jobs(seed_count=2)
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            if kwargs["batched"]:
                items = json.loads(kwargs["prompt"].split("Items:\n", 1)[1])
                return {
                    "ok": True,
                    "raw_text": json.dumps(
                        {
                            "results": [
                                {
                                    "request_index": items[0]["request_index"],
                                    "decision": "yes",
                                    "confidence": 0.8,
                                    "brief_reason": "valid batch item",
                                },
                                {
                                    "request_index": items[1]["request_index"],
                                    "decision": "yes",
                                    "confidence": 95,
                                    "brief_reason": "bad scale",
                                },
                            ]
                        }
                    ),
                    "response_json": {},
                    "latency_s": 0.01,
                    "error": "",
                }
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.91, "brief_reason": "fallback"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(eu.run_completion_jobs(jobs, max_workers=1, completion_fn=fake_completion, batch_size=2))

        self.assertEqual([call["batched"] for call in calls], [True, False])
        self.assertEqual({record["parse_status"] for record in records}, {"ok"})
        self.assertEqual(records[0]["parsed_json"]["confidence"], 0.8)
        self.assertEqual(records[1]["parsed_json"]["confidence"], 0.91)

    def test_instructor_batch_unknown_request_index_falls_back_unbatched(self):
        jobs = self._instructor_task1_jobs(seed_count=2)
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            if kwargs["batched"]:
                return {
                    "ok": True,
                    "raw_text": json.dumps(
                        {
                            "results": [
                                {
                                    "request_index": 999,
                                    "decision": "yes",
                                    "confidence": 0.8,
                                    "brief_reason": "unknown index",
                                }
                            ]
                        }
                    ),
                    "response_json": {},
                    "latency_s": 0.01,
                    "error": "",
                }
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.77, "brief_reason": "fallback"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(eu.run_completion_jobs(jobs, max_workers=1, completion_fn=fake_completion, batch_size=2))

        self.assertEqual([call["batched"] for call in calls], [True, False, False])
        self.assertEqual({record["parse_status"] for record in records}, {"ok"})
        self.assertEqual([record["parsed_json"]["confidence"] for record in records], [0.77, 0.77])

    def test_instructor_batch_duplicate_request_index_falls_back_unbatched(self):
        jobs = self._instructor_task1_jobs(seed_count=2)
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            if kwargs["batched"]:
                items = json.loads(kwargs["prompt"].split("Items:\n", 1)[1])
                duplicate_index = items[0]["request_index"]
                return {
                    "ok": True,
                    "raw_text": json.dumps(
                        {
                            "results": [
                                {
                                    "request_index": duplicate_index,
                                    "decision": "yes",
                                    "confidence": 0.8,
                                    "brief_reason": "first copy",
                                },
                                {
                                    "request_index": duplicate_index,
                                    "decision": "yes",
                                    "confidence": 0.9,
                                    "brief_reason": "duplicate copy",
                                },
                            ]
                        }
                    ),
                    "response_json": {},
                    "latency_s": 0.01,
                    "error": "",
                }
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.76, "brief_reason": "fallback"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(eu.run_completion_jobs(jobs, max_workers=1, completion_fn=fake_completion, batch_size=2))

        parsed_results, parse_status = eu.parse_batch_completion_results(
            json.dumps(
                {
                    "results": [
                        {"request_index": 1, "decision": "yes", "confidence": 0.8, "brief_reason": "a"},
                        {"request_index": 1, "decision": "yes", "confidence": 0.9, "brief_reason": "b"},
                    ]
                }
            )
        )
        self.assertEqual(parse_status, "duplicate_request_index")
        self.assertEqual(set(parsed_results), {1})
        self.assertEqual([call["batched"] for call in calls], [True, False, False])
        self.assertEqual({record["parse_status"] for record in records}, {"ok"})
        self.assertEqual([record["parsed_json"]["confidence"] for record in records], [0.76, 0.76])

    def test_instructor_malformed_batch_and_failed_fallback_remain_pending(self):
        jobs = self._instructor_task1_jobs(seed_count=2)
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            if kwargs["batched"]:
                return {
                    "ok": True,
                    "raw_text": "not json",
                    "response_json": {},
                    "latency_s": 0.01,
                    "error": "",
                }
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 95, "brief_reason": "bad scale"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(eu.run_completion_jobs(jobs, max_workers=1, completion_fn=fake_completion, batch_size=2))

        self.assertEqual([call["batched"] for call in calls], [True, False, False])
        self.assertEqual({record["parse_status"] for record in records}, {"instructor_validation_error"})
        self.assertEqual(len(eu.pending_completion_jobs(jobs, records, "full-1")), len(jobs))

    def test_instructor_failed_fallback_stays_pending(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        jobs = eu.planned_completion_jobs(
            benchmark[:1],
            tasks=["task1"],
            model="m1",
            host="http://localhost:1234/v1",
            run_id="full-1",
            prompt_version="v2-instructor-conf01",
            task1_template=eu.load_prompt("prompts/mandatory_entailment.txt"),
            task2_template=eu.load_prompt("prompts/modality_extraction.txt"),
            deterministic={"temperature": 0.0, "top_p": 1.0, "samples": 1},
            stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 0},
            max_tokens=64,
            timeout_s=30,
            api_key_env="LOCAL_OPENAI_API_KEY",
            structured_output="instructor",
        )

        def fake_completion(**kwargs):
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 95, "brief_reason": "bad scale"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(eu.run_completion_jobs(jobs, max_workers=1, completion_fn=fake_completion, batch_size=1))

        self.assertEqual(records[0]["parse_status"], "instructor_validation_error")
        self.assertEqual(len(eu.pending_completion_jobs(jobs, records, "full-1")), 1)

    def test_zai_example_uses_coding_plan_endpoint(self):
        config = eu.load_run_config("run_configs/full_matrix.example.json")
        zai_profile = next(profile for profile in config["profiles"] if profile["profile_id"] == "zai")

        self.assertEqual(zai_profile["base_url"], "https://api.z.ai/api/coding/paas/v4")
        self.assertIn("Coding Plan", zai_profile["notes"])

    def test_batched_completion_splits_results_to_raw_records(self):
        seeds = [
            {
                "seed_id": "S0001",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": "export reports",
            },
            {
                "seed_id": "S0002",
                "source_dataset": "NICE",
                "original_requirement": "The system shall print invoices.",
                "capability_text_final": "print invoices",
            },
        ]
        benchmark = eu.build_benchmark_items(seeds)
        jobs = eu.planned_completion_jobs(
            [row for row in benchmark if row["source_modality"] == "mandatory"],
            tasks=["task1"],
            model="m1",
            host="http://localhost:1234/v1",
            run_id="full-1",
            prompt_version="v1",
            task1_template=eu.load_prompt("prompts/mandatory_entailment.txt"),
            task2_template=eu.load_prompt("prompts/modality_extraction.txt"),
            deterministic={"temperature": 0.0, "top_p": 1.0, "samples": 1},
            stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 0},
            max_tokens=64,
            timeout_s=30,
            api_key_env="LOCAL_OPENAI_API_KEY",
        )
        calls = []

        def fake_batch_completion(**kwargs):
            calls.append(kwargs)
            items = json.loads(kwargs["prompt"].split("Items:\n", 1)[1])
            return {
                "ok": True,
                "raw_text": json.dumps(
                    {
                        "results": [
                            {
                                "request_index": item["request_index"],
                                "decision": "yes",
                                "confidence": 0.8,
                                "brief_reason": "batched",
                            }
                            for item in items
                        ]
                    }
                ),
                "response_json": {"batched": True},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(eu.run_completion_jobs(jobs, max_workers=1, completion_fn=fake_batch_completion, batch_size=2))

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["max_tokens"], 128)
        self.assertEqual(len(records), 2)
        self.assertEqual({row["parse_status"] for row in records}, {"ok"})
        self.assertEqual(len({row["batch_id"] for row in records}), 1)
        self.assertTrue(all(row["batch_size"] == 2 for row in records))

    def test_batched_completion_missing_results_are_not_marked_ok(self):
        seeds = [
            {
                "seed_id": "S0001",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": "export reports",
            },
            {
                "seed_id": "S0002",
                "source_dataset": "NICE",
                "original_requirement": "The system shall print invoices.",
                "capability_text_final": "print invoices",
            },
        ]
        benchmark = eu.build_benchmark_items(seeds)
        jobs = eu.planned_completion_jobs(
            [row for row in benchmark if row["source_modality"] == "mandatory"],
            tasks=["task1"],
            model="m1",
            host="http://localhost:1234/v1",
            run_id="full-1",
            prompt_version="v1",
            task1_template=eu.load_prompt("prompts/mandatory_entailment.txt"),
            task2_template=eu.load_prompt("prompts/modality_extraction.txt"),
            deterministic={"temperature": 0.0, "top_p": 1.0, "samples": 1},
            stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 0},
            max_tokens=64,
            timeout_s=30,
            api_key_env="LOCAL_OPENAI_API_KEY",
        )

        def fake_ignored_batch_completion(**kwargs):
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 80, "brief_reason": "single"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(eu.run_completion_jobs(jobs, max_workers=1, completion_fn=fake_ignored_batch_completion, batch_size=2))

        self.assertEqual({row["parse_status"] for row in records}, {"missing_batch_result"})
        self.assertTrue(all(row["parsed_json"] is None for row in records))
        self.assertEqual(len(eu.pending_completion_jobs(jobs, records, "full-1")), len(jobs))

    def test_resume_planning_skips_completed_records(self):
        job = {
            "run_id": "full-1",
            "model": "m1",
            "task": "task1",
            "item_id": "S0001_mandatory",
            "sample_kind": "deterministic",
            "sample_index": 0,
        }
        completed = [{**job, "parse_status": "ok"}]
        failed = [{**job, "parse_status": "request_error"}]

        self.assertEqual(eu.pending_completion_jobs([job], completed, "full-1"), [])
        self.assertEqual(len(eu.pending_completion_jobs([job], failed, "full-1")), 1)

    def test_run_registry_summary_and_upsert(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )[:1]
        raw_rows = []
        for task in ["task1", "task2"]:
            item = benchmark[0]
            raw_rows.append(
                {
                    "run_id": "full-1",
                    "model": "m1",
                    "task": task,
                    "item_id": item["item_id"],
                    "seed_id": item["seed_id"],
                    "source_modality": item["source_modality"],
                    "sample_kind": "deterministic",
                    "sample_index": 0,
                    "parse_status": "ok",
                    "parsed_json": {"decision": "yes", "confidence": 90} if task == "task1" else {"requirement": "The system MUST export reports.", "modality": "mandatory", "confidence": 90},
                    "prompt": item["source_statement"],
                }
            )
        row = eu.run_registry_summary(
            benchmark,
            raw_rows,
            run_id="full-1",
            run_group_id="group1",
            provider_id="local",
            profile_id="local",
            model="m1",
            dataset_id="nice",
            variant="must",
            tasks=["task1", "task2"],
            expected_stochastic_samples=0,
            started_at_utc="2026-05-21T00:00:00Z",
            finished_at_utc="2026-05-21T00:01:00Z",
        )

        self.assertEqual(row["status"], "complete")
        self.assertEqual(row["expected_records"], 2)
        self.assertEqual(row["observed_records"], 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run_registry.csv"
            eu.upsert_run_registry_row(path, row)
            eu.upsert_run_registry_row(path, {**row, "status": "partial"})
            rows = eu.read_csv_rows(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "partial")

    def test_live_run_counters_and_warning_events(self):
        raw_rows = [
            {"run_id": "r1", "request_index": 0, "parse_status": "ok"},
            {"run_id": "r1", "request_index": 1, "parse_status": "invalid_json"},
            {"run_id": "r1", "request_index": 2, "parse_status": "request_error"},
        ]

        counters = eu.live_run_counters(
            raw_rows,
            expected_records=10,
            expected_api_calls=10,
            started_monotonic=0.0,
            now_monotonic=3.0,
        )
        early_warnings = eu.warning_events_for_counters(
            counters,
            {"warn_after_records": 4, "warn_parse_failure_rate": 0.1, "warn_request_error_rate": 0.1},
            set(),
        )
        warnings = eu.warning_events_for_counters(
            counters,
            {"warn_after_records": 3, "warn_parse_failure_rate": 0.1, "warn_request_error_rate": 0.1},
            set(),
        )

        self.assertEqual(counters["observed_records"], 3)
        self.assertEqual(counters["ok_records"], 1)
        self.assertEqual(counters["request_error_records"], 1)
        self.assertEqual(counters["records_per_s"], 1.0)
        self.assertEqual(early_warnings, [])
        self.assertEqual({warning["warning_type"] for warning in warnings}, {"parse_failure_rate", "request_error_rate"})

    def test_run_event_jsonl_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.jsonl"
            eu.append_run_event(path, {"event_type": "start", "run_id": "r1"})
            eu.append_run_event(path, {"event_type": "finish", "run_id": "r1", "observed_records": 1})

            rows = eu.read_jsonl(path)
            self.assertEqual([row["event_type"] for row in rows], ["start", "finish"])
            self.assertTrue(all("created_at_utc" in row for row in rows))

    def test_run_group_ensemble_disagreement_across_run_ids(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        item = benchmark[0]
        raw_rows = [
            {
                "run_id": "full-a",
                "run_group_id": "group1",
                "provider_id": "p1",
                "model": "m1",
                "task": "task2",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "sample_kind": "deterministic",
                "sample_index": 0,
                "parse_status": "ok",
                "parsed_json": {"requirement": item["source_statement"], "modality": "mandatory", "confidence": 90},
                "prompt": item["source_statement"],
            },
            {
                "run_id": "full-b",
                "run_group_id": "group1",
                "provider_id": "p2",
                "model": "m2",
                "task": "task2",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "sample_kind": "deterministic",
                "sample_index": 0,
                "parse_status": "ok",
                "parsed_json": {"requirement": item["source_statement"], "modality": "recommended", "confidence": 90},
                "prompt": item["source_statement"],
            },
        ]

        scores = eu.build_run_group_ensemble_disagreement_scores(benchmark, raw_rows, run_group_id="group1")

        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0]["uq_method"], "model_ensemble_disagreement_run_group")
        self.assertEqual(scores[0]["valid_n"], 2)

    def test_run_matrix_completed_rows_filter_excludes_smoke_by_default(self):
        rows = [
            {"run_group_id": "group1", "run_id": "full-1", "status": "complete"},
            {"run_group_id": "group1", "run_id": "smoke-1", "status": "complete"},
            {"run_group_id": "group1", "run_id": "full-2", "status": "partial"},
            {"run_group_id": "other", "run_id": "full-3", "status": "complete"},
        ]

        selected = compare_matrix.completed_registry_rows(rows, "group1", include_smoke=False)
        selected_with_smoke = compare_matrix.completed_registry_rows(rows, "group1", include_smoke=True)

        self.assertEqual([row["run_id"] for row in selected], ["full-1"])
        self.assertEqual([row["run_id"] for row in selected_with_smoke], ["full-1", "smoke-1"])

    def test_fake_cli_smoke_writes_canonical_jsonl_and_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "prompts").mkdir(parents=True)
            (root / "data/processed").mkdir(parents=True)
            (root / "AGENTS.md").write_text("", encoding="utf-8")
            (root / "docs").mkdir()
            (root / "docs/evaluation.md").write_text("", encoding="utf-8")
            (root / "prompts/mandatory_entailment.txt").write_text(
                Path("prompts/mandatory_entailment.txt").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "prompts/modality_extraction.txt").write_text(
                Path("prompts/modality_extraction.txt").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            benchmark = eu.build_benchmark_items(
                [
                    {
                        "seed_id": "S0001",
                        "source_dataset": "NICE",
                        "original_requirement": "The system shall export reports.",
                        "capability_text_final": "export reports",
                    },
                    {
                        "seed_id": "S0002",
                        "source_dataset": "NICE",
                        "original_requirement": "The system shall print invoices.",
                        "capability_text_final": "print invoices",
                    }
                ]
            )
            eu.write_csv_rows(root / "data/processed/benchmark_items.csv", benchmark)
            config_path = root / "run_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_group_id": "smoke-group",
                        "datasets": ["nice"],
                        "benchmark_variants": ["must"],
                        "stochastic": {"temperature": 0.7, "top_p": 1.0, "samples": 1},
                        "logging": {"progress_every_records": 2, "progress_every_seconds": 999, "warn_after_records": 2},
                        "profiles": [
                            {
                                "profile_id": "fake",
                                "provider_id": "fake",
                                "base_url": "http://127.0.0.1:1234/v1",
                                "api_key_env": "LOCAL_OPENAI_API_KEY",
                                "models": ["fake-model"],
                                "concurrency": 1,
                                "batch_size": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "run_experiment_from_config.py",
                "--config",
                str(config_path),
                "--profile",
                "fake",
                "--mode",
                "smoke",
                "--smoke-items",
                "2",
                "--fake-completion",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(run_config_cli.eu, "project_root", return_value=root),
                redirect_stdout(io.StringIO()),
            ):
                run_config_cli.main()

            rows = eu.read_jsonl(root / "data/processed/model_outputs_raw.jsonl")
            registry = eu.read_csv_rows(root / "data/processed/run_registry.csv")
            progress = eu.read_csv_rows(root / "data/processed/run_progress_live.csv")
            events = eu.read_jsonl(root / "data/processed/run_events.jsonl")
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({row["batch_id"] for row in rows}), 4)
            self.assertEqual(registry[0]["status"], "complete")
            self.assertEqual(registry[0]["run_group_id"], "smoke-group")
            self.assertEqual(registry[0]["batch_size"], "2")
            self.assertEqual(registry[0]["expected_api_calls"], "4")
            self.assertEqual(registry[0]["observed_api_calls"], "4")
            self.assertEqual({row["task"] for row in progress}, {"task1", "task2"})
            self.assertEqual({row["event_type"] for row in events}, {"start", "progress", "finish"})
            finish_event = next(row for row in reversed(events) if row["event_type"] == "finish")
            self.assertEqual(finish_event["pending_jobs"], 0)
            self.assertEqual(finish_event["pending_api_calls"], 0)

    def test_task3_cli_fake_run_writes_diagnostic_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "prompts").mkdir(parents=True)
            (root / "data/processed").mkdir(parents=True)
            (root / "AGENTS.md").write_text("", encoding="utf-8")
            (root / "docs").mkdir()
            (root / "docs/evaluation.md").write_text("", encoding="utf-8")
            (root / "prompts/modality_verification.txt").write_text(
                Path("prompts/modality_verification.txt").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            benchmark = eu.build_benchmark_items(
                [
                    {
                        "seed_id": "S0001",
                        "source_dataset": "NICE",
                        "original_requirement": "The system shall export reports.",
                        "capability_text_final": "export reports",
                    },
                    {
                        "seed_id": "S0002",
                        "source_dataset": "NICE",
                        "original_requirement": "The system shall print invoices.",
                        "capability_text_final": "print invoices",
                    },
                ]
            )
            eu.write_csv_rows(root / "data/processed/benchmark_items.csv", benchmark)
            for item in benchmark:
                eu.append_jsonl(
                    root / "data/processed/model_outputs_raw.jsonl",
                    eu.build_raw_record(
                        run_id="full-source",
                        model="fake-model",
                        host="http://127.0.0.1:1234/v1",
                        task="task2",
                        item=item,
                        sample_index=0,
                        sample_kind="deterministic",
                        temperature=0.0,
                        top_p=1.0,
                        prompt_version="v2-conf01",
                        prompt=item["source_statement"],
                        completion={
                            "ok": True,
                            "raw_text": json.dumps(
                                {
                                    "requirement": item["source_statement"],
                                    "modality": item["task2_gold_modality"],
                                    "confidence": 0.9,
                                }
                            ),
                            "latency_s": 0.0,
                            "error": "",
                        },
                        provider_id="fake",
                        profile_id="fake",
                        run_group_id="group1",
                    ),
                )
            config_path = root / "run_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_group_id": "task3-group",
                        "datasets": ["nice"],
                        "benchmark_variants": ["must"],
                        "stochastic": {"temperature": 0.7, "top_p": 1.0, "samples": 0},
                        "profiles": [
                            {
                                "profile_id": "fake",
                                "provider_id": "fake",
                                "base_url": "http://127.0.0.1:1234/v1",
                                "api_key_env": "LOCAL_OPENAI_API_KEY",
                                "models": ["fake-model"],
                                "concurrency": 1,
                                "batch_size": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "run_task3_verification_from_config.py",
                "--config",
                str(config_path),
                "--profile",
                "fake",
                "--model",
                "fake-model",
                "--dataset",
                "nice",
                "--source-run-id",
                "full-source",
                "--mode",
                "smoke",
                "--smoke-items",
                "2",
                "--fake-completion",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(task3_cli.eu, "project_root", return_value=root),
                redirect_stdout(io.StringIO()),
            ):
                task3_cli.main()

            task3_items = eu.read_csv_rows(root / "data/processed/task3_verification_items.csv")
            task3_rows = eu.read_jsonl(root / "data/processed/model_outputs_raw_task3_verification.jsonl")
            registry = eu.read_csv_rows(root / "data/processed/run_registry_task3_verification.csv")
            progress = eu.read_csv_rows(root / "data/processed/run_progress_live_task3_verification.csv")
            self.assertEqual(len(task3_items), len(benchmark))
            self.assertEqual(len(task3_rows), 2)
            self.assertEqual({row["task"] for row in task3_rows}, {"task3"})
            self.assertEqual({row["parse_status"] for row in task3_rows}, {"ok"})
            self.assertEqual(registry[0]["status"], "complete")
            self.assertEqual(registry[0]["tasks"], "task3")
            self.assertEqual({row["task"] for row in progress}, {"task3"})

    def test_analysis_cli_generates_publication_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "prompts").mkdir(parents=True)
            (root / "data/processed").mkdir(parents=True)
            (root / "docs").mkdir(parents=True)
            for prompt_name in ["mandatory_entailment.txt", "modality_extraction.txt", "modality_verification.txt"]:
                (root / f"prompts/{prompt_name}").write_text(
                    Path(f"prompts/{prompt_name}").read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            review_rows = [{**row, "weaker_than_should": "yes"} for row in eu.weak_modality_construct_review_rows()]
            eu.write_csv_rows(
                root / "docs/weak_modality_construct_review.csv",
                review_rows,
                fieldnames=eu.WEAK_MODALITY_CONSTRUCT_REVIEW_FIELDS,
            )
            benchmark = eu.build_benchmark_items(
                [
                    {
                        "seed_id": "S0001",
                        "source_dataset": "NICE",
                        "original_requirement": "The system shall export reports.",
                        "capability_text_final": "export reports",
                    }
                ]
            )
            eu.write_csv_rows(root / "data/processed/benchmark_items.csv", benchmark)
            task1_template = eu.load_prompt(root / "prompts/mandatory_entailment.txt")
            task2_template = eu.load_prompt(root / "prompts/modality_extraction.txt")
            raw_rows = []
            for item in benchmark:
                for task in ["task1", "task2"]:
                    parsed = (
                        {
                            "decision": item["task1_gold_decision"],
                            "confidence": 0.9,
                            "brief_reason": "test",
                        }
                        if task == "task1"
                        else {
                            "requirement": item["source_statement"],
                            "modality": item["task2_gold_modality"],
                            "confidence": 0.9,
                        }
                    )
                    raw = eu.build_raw_record(
                        run_id="full-analysis",
                        model="fake-model",
                        host="http://127.0.0.1:1234/v1",
                        task=task,
                        item=item,
                        sample_index=0,
                        sample_kind="deterministic",
                        temperature=0.0,
                        top_p=1.0,
                        prompt_version="v2-conf01",
                        prompt=eu.prompt_for_benchmark_task(task, item, task1_template, task2_template),
                        completion={
                            "ok": True,
                            "raw_text": json.dumps(parsed),
                            "latency_s": 0.0,
                            "error": "",
                        },
                        provider_id="fake",
                        profile_id="fake",
                        run_group_id="group1",
                    )
                    raw_rows.append(raw)
                    eu.append_jsonl(root / "data/processed/model_outputs_raw.jsonl", raw)
            registry_row = eu.run_registry_summary(
                benchmark,
                raw_rows,
                run_id="full-analysis",
                run_group_id="group1",
                provider_id="fake",
                profile_id="fake",
                model="fake-model",
                dataset_id="nice",
                variant="must",
                tasks=["task1", "task2"],
                expected_stochastic_samples=0,
                started_at_utc="2026-05-22T00:00:00Z",
                finished_at_utc="2026-05-22T00:01:00Z",
            )
            eu.upsert_run_registry_row(root / "data/processed/run_registry.csv", registry_row)
            output_dir = root / "outputs/final"
            argv = [
                "generate_evaluation_analysis.py",
                "--dataset",
                "nice",
                "--variant",
                "must",
                "--run-id",
                "full-analysis",
                "--model",
                "fake-model",
                "--profile",
                "fake",
                "--output-dir",
                str(output_dir),
                "--bootstrap-iterations",
                "5",
                "--max-parse-failure-rate",
                "0",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(analysis_cli.eu, "project_root", return_value=root),
                redirect_stdout(io.StringIO()),
            ):
                analysis_cli.main()

            self.assertTrue((output_dir / "metrics_summary.csv").exists())
            self.assertTrue((output_dir / "paper_results_table.md").exists())
            self.assertTrue((output_dir / "task1_p_yes_by_modality.svg").exists())
            self.assertTrue((output_dir / "provenance_manifest.json").exists())
            provenance = json.loads((output_dir / "provenance_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["run_id"], "full-analysis")
            self.assertEqual(provenance["stale_item_count"], 0)

    def test_show_run_progress_reads_outputs_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data/processed").mkdir(parents=True)
            (root / "AGENTS.md").write_text("", encoding="utf-8")
            (root / "docs").mkdir()
            (root / "docs/evaluation.md").write_text("", encoding="utf-8")
            benchmark = eu.build_benchmark_items(
                [
                    {
                        "seed_id": "S0001",
                        "source_dataset": "NICE",
                        "original_requirement": "The system shall export reports.",
                        "capability_text_final": "export reports",
                    }
                ]
            )[:1]
            eu.write_csv_rows(root / "data/processed/benchmark_items.csv", benchmark)
            raw_rows = [
                {
                    "run_id": "full-1",
                    "model": "m1",
                    "task": "task1",
                    "item_id": benchmark[0]["item_id"],
                    "seed_id": benchmark[0]["seed_id"],
                    "source_modality": benchmark[0]["source_modality"],
                    "sample_kind": "deterministic",
                    "sample_index": 0,
                    "parse_status": "ok",
                    "parsed_json": {"decision": "yes", "confidence": 90},
                    "prompt": benchmark[0]["source_statement"],
                }
            ]
            for row in raw_rows:
                eu.append_jsonl(root / "data/processed/model_outputs_raw.jsonl", row)
            registry_row = eu.run_registry_summary(
                benchmark,
                raw_rows,
                run_id="full-1",
                run_group_id="group1",
                provider_id="fake",
                profile_id="fake",
                model="m1",
                dataset_id="nice",
                variant="must",
                tasks=["task1"],
                expected_stochastic_samples=0,
                started_at_utc="2026-05-21T00:00:00Z",
            )
            eu.upsert_run_registry_row(root / "data/processed/run_registry.csv", registry_row)

            before = sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())
            output = io.StringIO()
            with redirect_stdout(output):
                show_run_progress.print_progress(root, "nice", "must", "full-1")
            after = sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())

            self.assertEqual(before, after)
            self.assertIn("full-1: records 1/1", output.getvalue())
            self.assertIn("task_progress", output.getvalue())

    def test_show_run_progress_requires_disambiguation_for_reused_run_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data/processed").mkdir(parents=True)
            (root / "AGENTS.md").write_text("", encoding="utf-8")
            (root / "docs").mkdir()
            (root / "docs/evaluation.md").write_text("", encoding="utf-8")
            benchmark = eu.build_benchmark_items(
                [
                    {
                        "seed_id": "S0001",
                        "source_dataset": "NICE",
                        "original_requirement": "The system shall export reports.",
                        "capability_text_final": "export reports",
                    }
                ]
            )[:1]
            eu.write_csv_rows(root / "data/processed/benchmark_items.csv", benchmark)
            raw_rows = [
                {
                    "run_id": "full-1",
                    "profile_id": "p1",
                    "model": "m1",
                    "task": "task1",
                    "item_id": benchmark[0]["item_id"],
                    "seed_id": benchmark[0]["seed_id"],
                    "source_modality": benchmark[0]["source_modality"],
                    "sample_kind": "deterministic",
                    "sample_index": 0,
                    "parse_status": "ok",
                    "parsed_json": {"decision": "yes", "confidence": 90},
                    "prompt": benchmark[0]["source_statement"],
                },
                {
                    "run_id": "full-1",
                    "profile_id": "p2",
                    "model": "m2",
                    "task": "task1",
                    "item_id": benchmark[0]["item_id"],
                    "seed_id": benchmark[0]["seed_id"],
                    "source_modality": benchmark[0]["source_modality"],
                    "sample_kind": "deterministic",
                    "sample_index": 0,
                    "parse_status": "request_error",
                    "parsed_json": {},
                    "prompt": benchmark[0]["source_statement"],
                },
            ]
            for row in raw_rows:
                eu.append_jsonl(root / "data/processed/model_outputs_raw.jsonl", row)
            for profile_id, model in [("p1", "m1"), ("p2", "m2")]:
                registry_row = eu.run_registry_summary(
                    benchmark,
                    raw_rows,
                    run_id="full-1",
                    run_group_id="group1",
                    provider_id="fake",
                    profile_id=profile_id,
                    model=model,
                    dataset_id="nice",
                    variant="must",
                    tasks=["task1"],
                    expected_stochastic_samples=0,
                    started_at_utc="2026-05-21T00:00:00Z",
                )
                eu.upsert_run_registry_row(root / "data/processed/run_registry.csv", registry_row)

            with self.assertRaisesRegex(ValueError, "matches multiple registry rows"):
                show_run_progress.print_progress(root, "nice", "must", "full-1")

            output = io.StringIO()
            with redirect_stdout(output):
                show_run_progress.print_progress(root, "nice", "must", "full-1", model="m2")

            text = output.getvalue()
            self.assertIn("full-1: records 1/1", text)
            self.assertIn("parse_status: {'request_error': 1}", text)

    def test_run_completion_jobs_returns_records_from_fake_completion(self):
        item = {
            "item_id": "S0001_mandatory",
            "seed_id": "S0001",
            "source_modality": "mandatory",
        }
        jobs = [
            {
                "request_index": index,
                "run_id": "r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task1",
                "item": item,
                "sample_index": index,
                "sample_kind": "stochastic",
                "temperature": 0.7,
                "top_p": 1.0,
                "prompt_version": "v1",
                "prompt": f"prompt {index}",
                "max_tokens": 32,
                "timeout_s": 5,
                "api_key_env": "LOCAL_OPENAI_API_KEY",
            }
            for index in range(4)
        ]

        def fake_completion(**kwargs):
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 80, "brief_reason": "ok"}',
                "response_json": {"prompt": kwargs["prompt"]},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(eu.run_completion_jobs(jobs, max_workers=2, completion_fn=fake_completion))

        self.assertEqual(len(records), 4)
        self.assertEqual({row["request_index"] for row in records}, {0, 1, 2, 3})
        self.assertTrue(all(row["parse_status"] == "ok" for row in records))
        self.assertTrue(all(row["parsed_json"]["decision"] == "yes" for row in records))

    def test_select_run_rows_uses_latest_full_run_by_default(self):
        raw_rows = [
            {"run_id": "full-20260518-100000-aaaa", "value": 1},
            {"run_id": "pilot-20260518-110000-bbbb", "value": 2},
            {"run_id": "full-20260518-120000-cccc", "value": 3},
            {"run_id": "full-20260518-120000-cccc", "value": 4},
            {"run_id": "full-shall-20260518-130000-dddd", "value": 5},
        ]
        selected_run_id, rows = eu.select_run_rows(raw_rows, prefix="full")
        self.assertEqual(selected_run_id, "full-20260518-120000-cccc")
        self.assertEqual([row["value"] for row in rows], [3, 4])
        self.assertFalse(eu.run_id_matches_prefix("full-shall-20260518-130000-dddd", "full"))
        self.assertTrue(eu.run_id_matches_prefix("full-shall-20260518-130000-dddd", "full-shall"))

    def test_select_run_rows_honors_explicit_run_id(self):
        raw_rows = [
            {"run_id": "full-20260518-100000-aaaa", "value": 1},
            {"run_id": "full-20260518-120000-cccc", "value": 2},
            {"run_id": "full-shall-20260518-130000-dddd", "value": 3},
        ]
        selected_run_id, rows = eu.select_run_rows(raw_rows, run_id="full-20260518-100000-aaaa", prefix="full")
        self.assertEqual(selected_run_id, "full-20260518-100000-aaaa")
        self.assertEqual([row["value"] for row in rows], [1])

        selected_run_id, rows = eu.select_run_rows(raw_rows, run_id="full-shall-20260518-130000-dddd", prefix="full")
        self.assertEqual(selected_run_id, "full-shall-20260518-130000-dddd")
        self.assertEqual(rows, [])

    def test_calibration_probabilities_task1_use_p_yes(self):
        rows = [
            {"p_yes": 0.1, "confidence": 0.9},
            {"p_yes": 0.8, "confidence": 0.8},
        ]
        self.assertEqual(eu.calibration_probabilities(rows, "task1"), [0.1, 0.8])
        self.assertEqual(eu.calibration_probabilities(rows, "task2"), [0.9, 0.8])

    def test_run_progress_summary_for_partial_outputs(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        item = benchmark[0]
        raw_rows = [
            {
                "run_id": "r1",
                "model": "m1",
                "task": "task1",
                "item_id": item["item_id"],
                "sample_kind": "deterministic",
                "parse_status": "ok",
                "parsed_json": {"decision": "yes", "confidence": 90.0, "brief_reason": ""},
            },
            {
                "run_id": "r1",
                "model": "m1",
                "task": "task1",
                "item_id": item["item_id"],
                "sample_kind": "stochastic",
                "parse_status": "invalid_json",
                "parsed_json": None,
            },
        ]

        progress = eu.run_progress_summary(benchmark, raw_rows, expected_stochastic_samples=5)

        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]["observed_records"], 2)
        self.assertAlmostEqual(progress[0]["parse_success_rate"], 0.5)
        self.assertAlmostEqual(progress[0]["record_completion_rate"], 2 / 24)
        self.assertAlmostEqual(progress[0]["deterministic_item_coverage"], 0.25)

    def test_complete_run_ids_from_progress_can_be_variant_scoped(self):
        progress = []
        for run_id in ["full-20260518-120000-aaaa", "full-shall-20260518-130000-bbbb"]:
            for task in ["task1", "task2"]:
                progress.append(
                    {
                        "run_id": run_id,
                        "task": task,
                        "record_completion_rate": 1.0,
                        "deterministic_item_coverage": 1.0,
                        "stochastic_complete_item_rate": 1.0,
                    }
                )

        self.assertEqual(
            eu.complete_run_ids_from_progress(progress, prefix="full"),
            ["full-20260518-120000-aaaa"],
        )
        self.assertEqual(
            eu.complete_run_ids_from_progress(progress, prefix="full-shall"),
            ["full-shall-20260518-130000-bbbb"],
        )

    def test_run_progress_summary_ignores_rows_for_removed_benchmark_items(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        raw_rows = [
            {
                "run_id": "r1",
                "model": "m1",
                "task": "task1",
                "item_id": "removed_item",
                "sample_kind": "deterministic",
                "parse_status": "ok",
                "parsed_json": {"decision": "yes", "confidence": 90.0, "brief_reason": ""},
            }
        ]

        progress = eu.run_progress_summary(benchmark, raw_rows, expected_stochastic_samples=5)

        self.assertEqual(progress, [])

    def test_write_preliminary_result_snapshot(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        item = benchmark[0]
        raw_rows = [
            {
                "run_id": "r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task1",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "sample_index": 0,
                "sample_kind": "deterministic",
                "temperature": 0.0,
                "top_p": 1.0,
                "prompt_version": "v1",
                "raw_text": "",
                "parsed_json": {"decision": "yes", "confidence": 90.0, "brief_reason": ""},
                "parse_status": "ok",
                "latency_s": 0.1,
                "error": "",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = eu.write_preliminary_result_snapshot(
                benchmark,
                raw_rows,
                tmpdir,
                expected_stochastic_samples=5,
            )

            self.assertGreater(snapshot["score_rows"], 0)
            self.assertGreater(snapshot["summary_rows"], 0)
            self.assertEqual(snapshot["progress_rows"], 1)
            for path in snapshot["paths"].values():
                self.assertTrue(path.exists())
            self.assertIn("Preliminary Results", snapshot["paths"]["table"].read_text(encoding="utf-8"))

    def test_matplotlib_figure_export(self):
        rows = [
            {
                "model": "m1",
                "task": "task1",
                "uq_method": "verbalized_confidence",
                "source_modality": modality,
                "p_yes": value,
            }
            for modality, value in [
                ("mandatory", 0.9),
                ("recommended", 0.6),
                ("optional", 0.3),
                ("nice_to_have", 0.1),
            ]
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "figure.svg"
            eu.write_task1_modality_svg(rows, path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 100)

    def test_uq_method_inventory_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = eu.write_uq_method_inventory(tmpdir)

            self.assertTrue(paths["markdown"].exists())
            self.assertTrue(paths["csv"].exists())
            self.assertIn("predictive_entropy", paths["markdown"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
