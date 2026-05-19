import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts import eval_utils as eu
from scripts import evaluate_external_ai_probe as external_eval


class EvalUtilsTest(unittest.TestCase):
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
                    "confidence": 90.0,
                    "evidence_phrase": "It would be useful if",
                    "brief_reason": "missed upgrade",
                },
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
                        "confidence": 98,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            scored, validation = external_eval.evaluate_outputs(output_path, gold_path)
            summary = external_eval.source_condition_summary(scored)

            self.assertEqual(validation["parse_errors"], 0)
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
                "confidence": 90,
            }
            output_path.write_text(
                json.dumps(duplicate_row) + "\n" + json.dumps({**duplicate_row, "confidence": 80}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate external_item_id.*EXT0001"):
                external_eval.evaluate_outputs(output_path, gold_path)

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
