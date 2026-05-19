import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import nbformat

from scripts import eval_utils as eu
from scripts import evaluate_external_ai_probe as external_eval
from scripts import export_external_ai_probe as external_export
from scripts import populate_notebooks


def seed_rows(count=1):
    return [
        {
            "seed_id": f"S{index:04d}",
            "source_dataset": "NICE",
            "original_requirement": "The system shall export reports.",
            "capability_text_final": f"export reports {index}",
        }
        for index in range(1, count + 1)
    ]


def raw_record(
    item,
    *,
    task,
    parsed_json,
    run_id="r1",
    model="m1",
    sample_kind="deterministic",
    parse_status="ok",
):
    record = {
        "run_id": run_id,
        "model": model,
        "host": "http://localhost:8000/v1",
        "task": task,
        "item_id": item["item_id"],
        "seed_id": item["seed_id"],
        "source_modality": item["source_modality"],
        "sample_index": 0,
        "sample_kind": sample_kind,
        "temperature": 0.0,
        "top_p": 1.0,
        "prompt_version": "v1",
        "raw_text": "",
        "parsed_json": parsed_json,
        "parse_status": parse_status,
        "latency_s": 0.1,
        "error": "",
    }
    if "template_id" in item:
        record["template_id"] = item["template_id"]
    return record


class NotebookBoundaryTest(unittest.TestCase):
    NOTEBOOK_BUILDERS = {
        "00_prepare_data.ipynb": populate_notebooks.notebook_00,
        "01_build_modality_benchmark.ipynb": populate_notebooks.notebook_01,
        "02_pilot_local_llms.ipynb": populate_notebooks.notebook_02,
        "02b_weak_modality_robustness_probe.ipynb": populate_notebooks.notebook_02b,
        "03_run_experiments.ipynb": populate_notebooks.notebook_03,
        "03b_run_modality_verification.ipynb": populate_notebooks.notebook_03b,
        "04_compute_uq_and_metrics.ipynb": populate_notebooks.notebook_04,
        "05_analyze_and_export_results.ipynb": populate_notebooks.notebook_05,
    }

    def test_checked_in_notebooks_match_generator_sources(self):
        for name, builder in self.NOTEBOOK_BUILDERS.items():
            with self.subTest(notebook=name):
                actual = nbformat.read(Path("notebooks") / name, as_version=4)
                expected_cells = builder()

                actual_sources = [(cell.cell_type, cell.source) for cell in actual.cells]
                expected_sources = [(cell.cell_type, cell.source) for cell in expected_cells]

                self.assertEqual(actual_sources, expected_sources)

    def test_notebooks_are_clean_execution_artifacts(self):
        for path in sorted(Path("notebooks").glob("*.ipynb")):
            with self.subTest(notebook=str(path)):
                notebook = nbformat.read(path, as_version=4)
                for index, cell in enumerate(notebook.cells, start=1):
                    if cell.cell_type != "code":
                        continue
                    self.assertIsNone(cell.execution_count, f"{path} cell {index} has an execution count")
                    self.assertEqual(cell.outputs, [], f"{path} cell {index} has stored outputs")

    def test_benchmark_manifest_tracks_prompt_inputs_and_metadata(self):
        manifest = json.loads(Path("outputs/benchmark_manifest.json").read_text(encoding="utf-8"))
        artifact_paths = {artifact["path"] for artifact in manifest["artifacts"]}
        required_paths = {
            "prompts/mandatory_entailment.txt",
            "prompts/mandatory_entailment_strict.txt",
            "prompts/modality_extraction.txt",
            "prompts/modality_extraction_labels_only.txt",
            "prompts/modality_verification.txt",
        }

        self.assertTrue(required_paths.issubset(artifact_paths))
        self.assertEqual(manifest["metadata"]["main_benchmark"], "MUST")
        self.assertEqual(manifest["metadata"]["robustness_benchmark"], "SHALL")
        self.assertEqual(manifest["metadata"]["seed_count"], 120)
        self.assertEqual(manifest["metadata"]["source_modalities"], eu.MODALITIES)


class AggregationBoundaryTest(unittest.TestCase):
    def test_task2_prompt_sensitivity_denominator_ignores_non_task2_rows(self):
        benchmark = eu.build_benchmark_items(seed_rows())
        item = [row for row in benchmark if row["source_modality"] == "nice_to_have"][0]
        raw_rows = [
            raw_record(
                item,
                task="task2",
                parsed_json={
                    "requirement": "It would be useful if the system could export reports.",
                    "modality": "nice_to_have",
                    "confidence": 90.0,
                },
            ),
            raw_record(
                item,
                task="task1",
                parsed_json={"decision": "no", "confidence": 90.0, "brief_reason": "weak intent"},
            ),
        ]

        summary = eu.task2_prompt_sensitivity_summary(benchmark, raw_rows)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["n"], 1)
        self.assertEqual(summary[0]["valid_n"], 1)
        self.assertEqual(summary[0]["parse_success_rate"], 1.0)
        self.assertEqual(summary[0]["nice_to_have_accuracy"], 1.0)

    def test_weak_modality_probe_summary_denominator_ignores_non_task2_rows(self):
        probe_items = eu.build_weak_modality_probe_items(seed_rows())
        item = probe_items[0]
        raw_rows = [
            raw_record(
                item,
                task="task2",
                parsed_json={
                    "requirement": "It would be useful if the system could export reports.",
                    "modality": "nice_to_have",
                    "confidence": 90.0,
                },
            ),
            raw_record(
                item,
                task="task1",
                parsed_json={"decision": "no", "confidence": 90.0, "brief_reason": "not mandatory"},
            ),
        ]

        summary = eu.weak_modality_probe_summary(probe_items, raw_rows)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["n"], 1)
        self.assertEqual(summary[0]["valid_n"], 1)
        self.assertEqual(summary[0]["parse_success_rate"], 1.0)
        self.assertEqual(summary[0]["accuracy"], 1.0)


class ParsingAndExternalProbeTest(unittest.TestCase):
    def test_parse_task_response_handles_embedded_braces_and_rejects_bad_fields(self):
        parsed, status = eu.parse_task_response(
            "task1",
            'prefix {"decision":"yes","confidence":88,"brief_reason":"contains {braces} safely"} suffix',
        )
        self.assertEqual(status, "ok")
        self.assertEqual(parsed["brief_reason"], "contains {braces} safely")

        _, status = eu.parse_task_response("task1", '{"decision":"yes","confidence":101,"brief_reason":""}')
        self.assertEqual(status, "invalid_confidence")

        _, status = eu.parse_task_response("task2", '{"modality":"optional","confidence":80}')
        self.assertEqual(status, "missing_fields")

        _, status = eu.parse_task_response("task3", '{"relation":"preserves","confidence":80}')
        self.assertEqual(status, "missing_fields")

    def test_external_evaluator_reports_validation_anomalies(self):
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
                        "source_kind": "weak_modality_probe",
                        "original_item_id": "S0002_weak_future_enhancement",
                        "seed_id": "S0002",
                        "source_condition": "weak_future_enhancement",
                        "source_modality": "nice_to_have",
                        "task2_gold_modality": "nice_to_have",
                        "capability_text": "export reports",
                        "source_statement": "Stakeholders mentioned that the system could export reports as a possible future enhancement.",
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
                                "confidence": 80,
                            }
                        ),
                        json.dumps(
                            {
                                "external_item_id": "EXT9999",
                                "requirement": "The system MUST export reports.",
                                "modality": "should",
                                "confidence": 101,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            scored, validation = external_eval.evaluate_outputs(output_path, gold_path)

            self.assertEqual(validation["output_rows"], 2)
            self.assertEqual(validation["gold_rows"], 2)
            self.assertEqual(validation["duplicate_ids"], 0)
            self.assertEqual(validation["missing_ids"], ["EXT0002"])
            self.assertEqual(validation["extra_ids"], ["EXT9999"])
            self.assertEqual(validation["invalid_label_count"], 1)
            self.assertEqual(validation["invalid_confidence_count"], 1)
            self.assertFalse(math.isnan(float(scored["correct"].mean())))

    def test_external_probe_export_is_blind_deterministic_and_pilot_scoped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config.example.json").write_text(
                json.dumps({"project": {"pilot_seed_count": 2}}),
                encoding="utf-8",
            )
            benchmark = eu.build_benchmark_items(seed_rows(3))
            weak_probe = eu.build_weak_modality_probe_items(seed_rows(3))
            eu.write_csv_rows(root / "data/processed/benchmark_items.csv", benchmark)
            eu.write_csv_rows(root / "data/processed/weak_modality_probe_items.csv", weak_probe)

            inputs, key = external_export.build_external_probe_rows(root)

            source_counts = Counter(row["source_kind"] for row in key)
            self.assertEqual(len(inputs), 14)
            self.assertEqual(len(key), 14)
            self.assertEqual(source_counts["main_benchmark"], 8)
            self.assertEqual(source_counts["weak_modality_probe"], 6)
            self.assertEqual([row["external_item_id"] for row in inputs], [f"EXT{index:04d}" for index in range(1, 15)])
            self.assertEqual([row["external_item_id"] for row in key], [row["external_item_id"] for row in inputs])
            self.assertEqual(set(inputs[0]), {"external_item_id", "source_statement"})
            self.assertNotIn("weak_useful_if", {row["source_condition"] for row in key})
            self.assertEqual({row["seed_id"] for row in key}, {"S0001", "S0002"})


if __name__ == "__main__":
    unittest.main()
