import math
import tempfile
import unittest
from pathlib import Path

from scripts import eval_utils as eu


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

    def test_parse_task1_response(self):
        raw = 'Some preface {"decision": "yes", "confidence": 87, "brief_reason": "MUST matches"}'
        parsed, status = eu.parse_task_response("task1", raw)
        self.assertEqual(status, "ok")
        self.assertEqual(parsed["decision"], "yes")
        self.assertEqual(parsed["confidence"], 87.0)

    def test_parse_task2_response(self):
        raw = '{"requirement": "The system SHOULD export reports.", "modality": "should", "confidence": 80}'
        parsed, status = eu.parse_task_response("task2", raw)
        self.assertEqual(status, "ok")
        self.assertEqual(parsed["modality"], "recommended")

    def test_metrics(self):
        y_true = [1, 0, 1, 0]
        p = [0.9, 0.2, 0.8, 0.1]
        self.assertLess(eu.brier_score(y_true, p), 0.05)
        self.assertEqual(eu.auroc_score(y_true, p), 1.0)
        self.assertFalse(math.isnan(eu.ece_score(y_true, p)))

    def test_monotonicity_violation(self):
        rows = [
            {"seed_id": "S1", "source_modality": "mandatory", "p_yes": 0.9},
            {"seed_id": "S1", "source_modality": "recommended", "p_yes": 0.6},
            {"seed_id": "S1", "source_modality": "optional", "p_yes": 0.3},
            {"seed_id": "S1", "source_modality": "nice_to_have", "p_yes": 0.1},
            {"seed_id": "S2", "source_modality": "mandatory", "p_yes": 0.9},
            {"seed_id": "S2", "source_modality": "recommended", "p_yes": 0.95},
            {"seed_id": "S2", "source_modality": "optional", "p_yes": 0.2},
            {"seed_id": "S2", "source_modality": "nice_to_have", "p_yes": 0.1},
        ]
        self.assertEqual(eu.monotonicity_violation_rate(rows), 0.5)

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
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0]["uq_method"], "label_self_consistency")
        self.assertAlmostEqual(scores[0]["p_yes"], 0.8)

    def test_select_run_rows_uses_latest_full_run_by_default(self):
        raw_rows = [
            {"run_id": "full-20260518-100000-aaaa", "value": 1},
            {"run_id": "pilot-20260518-110000-bbbb", "value": 2},
            {"run_id": "full-20260518-120000-cccc", "value": 3},
            {"run_id": "full-20260518-120000-cccc", "value": 4},
        ]
        selected_run_id, rows = eu.select_run_rows(raw_rows, prefix="full")
        self.assertEqual(selected_run_id, "full-20260518-120000-cccc")
        self.assertEqual([row["value"] for row in rows], [3, 4])

    def test_select_run_rows_honors_explicit_run_id(self):
        raw_rows = [
            {"run_id": "full-20260518-100000-aaaa", "value": 1},
            {"run_id": "full-20260518-120000-cccc", "value": 2},
        ]
        selected_run_id, rows = eu.select_run_rows(raw_rows, run_id="full-20260518-100000-aaaa", prefix="full")
        self.assertEqual(selected_run_id, "full-20260518-100000-aaaa")
        self.assertEqual([row["value"] for row in rows], [1])

    def test_calibration_probabilities_task1_use_p_yes(self):
        rows = [
            {"p_yes": 0.1, "confidence": 0.9},
            {"p_yes": 0.8, "confidence": 0.8},
        ]
        self.assertEqual(eu.calibration_probabilities(rows, "task1"), [0.1, 0.8])
        self.assertEqual(eu.calibration_probabilities(rows, "task2"), [0.9, 0.8])

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


if __name__ == "__main__":
    unittest.main()
