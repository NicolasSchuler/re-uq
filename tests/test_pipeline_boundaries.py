import csv
import io
import json
import math
import multiprocessing
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest import mock

import nbformat

try:
    from helpers import raw_record
except ModuleNotFoundError:  # pragma: no cover - invocation-path fallback
    from tests.helpers import raw_record

from scripts import (
    eval_utils as eu,
    evaluate_external_ai_probe as external_eval,
    export_external_ai_probe as external_export,
    populate_notebooks,
    run_task3_verification_from_config as task3_cli,
    show_run_progress,
)


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


class NotebookBoundaryTest(unittest.TestCase):
    NOTEBOOK_BUILDERS: ClassVar[dict[str, Any]] = {
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

                actual_sources = [
                    (cell.cell_type, cell.source) for cell in actual.cells
                ]
                expected_sources = [
                    (cell.cell_type, cell.source) for cell in expected_cells
                ]

                self.assertEqual(actual_sources, expected_sources)

    def test_notebooks_are_clean_execution_artifacts(self):
        for path in sorted(Path("notebooks").glob("*.ipynb")):
            with self.subTest(notebook=str(path)):
                notebook = nbformat.read(path, as_version=4)
                for index, cell in enumerate(notebook.cells, start=1):
                    if cell.cell_type != "code":
                        continue
                    self.assertIsNone(
                        cell.execution_count,
                        f"{path} cell {index} has an execution count",
                    )
                    self.assertEqual(
                        cell.outputs, [], f"{path} cell {index} has stored outputs"
                    )

    def test_populate_notebooks_help_and_dry_run_do_not_write(self):
        buffer = io.StringIO()
        with self.assertRaises(SystemExit) as context, redirect_stdout(buffer):
            populate_notebooks.main(["--help"])
        self.assertEqual(context.exception.code, 0)
        self.assertIn("--dry-run", buffer.getvalue())

        with tempfile.TemporaryDirectory() as tmpdir:
            notebook_dir = Path(tmpdir) / "notebooks"
            with io.StringIO() as buffer, redirect_stdout(buffer):
                paths = populate_notebooks.main(
                    ["--dry-run", "--notebook-dir", str(notebook_dir)]
                )
                output = buffer.getvalue()
            self.assertIsNone(paths)
            self.assertIn("Would write:", output)
            self.assertFalse(notebook_dir.exists())

    def test_benchmark_manifest_tracks_prompt_inputs_and_metadata(self):
        manifest = json.loads(
            Path("outputs/benchmark_manifest.json").read_text(encoding="utf-8")
        )
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
        self.assertEqual(manifest["metadata"]["seed_count"], 180)
        self.assertEqual(manifest["metadata"]["source_modalities"], eu.MODALITIES)


class PublicationArtifactIntegrityTest(unittest.TestCase):
    BENCHMARK_FILES: ClassVar[list[tuple[str, str]]] = [
        ("data/processed/benchmark_items.csv", "MUST"),
        ("data/processed/benchmark_items_mlm_tapt.csv", "MUST"),
        ("data/processed/benchmark_items_shall.csv", "SHALL"),
        ("data/processed/benchmark_items_mlm_tapt_shall.csv", "SHALL"),
    ]
    WEAK_TEMPLATES: ClassVar[set[str]] = {
        "future_enhancement",
        "low_priority_enhancement",
        "nice_if",
        "useful_if",
    }
    EXTERNAL_MAIN_CONDITIONS: ClassVar[set[str]] = {
        "mandatory",
        "recommended",
        "optional",
        "nice_to_have",
    }
    EXTERNAL_WEAK_CONDITIONS: ClassVar[set[str]] = {
        "weak_future_enhancement",
        "weak_low_priority_enhancement",
        "weak_nice_if",
    }

    def test_checked_benchmark_csvs_preserve_minimal_pair_contract(self):
        for path_text, keyword in self.BENCHMARK_FILES:
            with self.subTest(path=path_text):
                rows = eu.read_csv_rows(path_text)
                self.assertEqual(len(rows), 720)
                self.assertEqual(len({row["item_id"] for row in rows}), 720)
                self.assertEqual(len({row["seed_id"] for row in rows}), 180)
                self.assertEqual(
                    {row["source_modality"] for row in rows}, set(eu.MODALITIES)
                )
                self.assertTrue(
                    all(row["mandatory_keyword"] == keyword for row in rows)
                )

                by_seed: dict[str, list[dict[str, str]]] = {}
                for row in rows:
                    by_seed.setdefault(row["seed_id"], []).append(row)
                    self.assertTrue(row["source_statement"].strip())
                    self.assertTrue(row["candidate_requirement"].strip())
                    self.assertTrue(row["capability_text"].strip())
                    self.assertIn(keyword, row["candidate_requirement"])
                    if row["source_modality"] == "mandatory":
                        self.assertIn(keyword, row["source_statement"])
                    self.assertEqual(
                        row["task1_gold_decision"],
                        "yes" if row["source_modality"] == "mandatory" else "no",
                    )
                    self.assertEqual(
                        row["task1_gold_yes"],
                        "1" if row["source_modality"] == "mandatory" else "0",
                    )
                    self.assertEqual(row["task2_gold_modality"], row["source_modality"])

                for seed_id, seed_rows in by_seed.items():
                    self.assertEqual(len(seed_rows), 4, seed_id)
                    self.assertEqual(
                        {row["source_modality"] for row in seed_rows},
                        set(eu.MODALITIES),
                    )

    def test_checked_weak_modality_probe_rows_are_balanced(self):
        rows = eu.read_csv_rows("data/processed/weak_modality_probe_items.csv")

        self.assertEqual(len(rows), 80)
        self.assertEqual(len({row["item_id"] for row in rows}), 80)
        self.assertEqual(len({row["seed_id"] for row in rows}), 20)
        self.assertEqual({row["template_id"] for row in rows}, self.WEAK_TEMPLATES)
        self.assertEqual({row["source_modality"] for row in rows}, {"nice_to_have"})
        self.assertEqual({row["task2_gold_modality"] for row in rows}, {"nice_to_have"})
        self.assertTrue(all(row["source_statement"].strip() for row in rows))
        self.assertTrue(all(row["capability_text"].strip() for row in rows))

        for seed_id in {row["seed_id"] for row in rows}:
            seed_rows = [row for row in rows if row["seed_id"] == seed_id]
            self.assertEqual(len(seed_rows), 4, seed_id)
            self.assertEqual(
                {row["template_id"] for row in seed_rows}, self.WEAK_TEMPLATES
            )

    def test_checked_external_probe_is_blind_and_balanced(self):
        inputs_path = Path(
            "outputs/external_ai_service_probe/external_task2_inputs.csv"
        )
        with inputs_path.open(newline="", encoding="utf-8") as handle:
            input_reader = csv.DictReader(handle)
            self.assertEqual(
                input_reader.fieldnames, ["external_item_id", "source_statement"]
            )
            inputs = list(input_reader)
        gold = eu.read_csv_rows(
            "outputs/external_ai_service_probe/external_task2_gold_key.csv"
        )

        expected_ids = [f"EXT{index:04d}" for index in range(1, 141)]
        self.assertEqual([row["external_item_id"] for row in inputs], expected_ids)
        self.assertEqual([row["external_item_id"] for row in gold], expected_ids)
        self.assertEqual(len(gold), 140)
        self.assertEqual(len({row["external_item_id"] for row in gold}), 140)
        self.assertTrue(all(row["source_statement"].strip() for row in inputs))

        kind_counts = Counter(row["source_kind"] for row in gold)
        self.assertEqual(kind_counts["main_benchmark"], 80)
        self.assertEqual(kind_counts["weak_modality_probe"], 60)

        main_conditions = Counter(
            row["source_condition"]
            for row in gold
            if row["source_kind"] == "main_benchmark"
        )
        weak_conditions = Counter(
            row["source_condition"]
            for row in gold
            if row["source_kind"] == "weak_modality_probe"
        )
        self.assertEqual(set(main_conditions), self.EXTERNAL_MAIN_CONDITIONS)
        self.assertEqual(set(weak_conditions), self.EXTERNAL_WEAK_CONDITIONS)
        self.assertTrue(all(count == 20 for count in main_conditions.values()))
        self.assertTrue(all(count == 20 for count in weak_conditions.values()))
        self.assertTrue(
            all(row["source_modality"] == row["task2_gold_modality"] for row in gold)
        )

    def test_checked_benchmark_manifests_match_current_artifacts(self):
        expected = {
            "outputs/benchmark_manifest.json": "nice",
            "outputs/benchmark_manifest_mlm_tapt.json": "mlm_tapt",
        }
        required_prompt_paths = {
            "prompts/mandatory_entailment.txt",
            "prompts/mandatory_entailment_strict.txt",
            "prompts/modality_extraction.txt",
            "prompts/modality_extraction_labels_only.txt",
            "prompts/modality_verification.txt",
        }

        for manifest_path, dataset_id in expected.items():
            with self.subTest(path=manifest_path):
                manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                self.assertEqual(manifest["metadata"]["dataset_id"], dataset_id)
                self.assertEqual(manifest["metadata"]["seed_count"], 180)
                self.assertEqual(manifest["metadata"]["main_benchmark"], "MUST")
                self.assertEqual(manifest["metadata"]["robustness_benchmark"], "SHALL")
                self.assertEqual(
                    manifest["metadata"]["source_modalities"], eu.MODALITIES
                )

                artifact_paths = {
                    artifact["path"] for artifact in manifest["artifacts"]
                }
                self.assertTrue(required_prompt_paths.issubset(artifact_paths))
                self.assertIn(
                    eu.artifact_path(
                        "data/processed/benchmark_items.csv", dataset_id
                    ).as_posix(),
                    artifact_paths,
                )
                self.assertIn(
                    eu.artifact_path(
                        "data/processed/benchmark_items.csv", dataset_id, "shall"
                    ).as_posix(),
                    artifact_paths,
                )

                for artifact in manifest["artifacts"]:
                    path = Path(artifact["path"])
                    self.assertTrue(path.exists(), artifact["path"])
                    self.assertTrue(artifact["exists"], artifact["path"])
                    self.assertEqual(
                        artifact["sha256"], eu.sha256_file(path), artifact["path"]
                    )
                    self.assertEqual(
                        artifact["bytes"], path.stat().st_size, artifact["path"]
                    )
                    if path.suffix == ".csv":
                        self.assertEqual(
                            artifact["rows"],
                            len(eu.read_csv_rows(path)),
                            artifact["path"],
                        )


class AggregationBoundaryTest(unittest.TestCase):
    def test_task2_prompt_sensitivity_denominator_ignores_non_task2_rows(self):
        benchmark = eu.build_benchmark_items(seed_rows())
        item = next(
            row for row in benchmark if row["source_modality"] == "nice_to_have"
        )
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
                parsed_json={
                    "decision": "no",
                    "confidence": 90.0,
                    "brief_reason": "weak intent",
                },
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
                parsed_json={
                    "decision": "no",
                    "confidence": 90.0,
                    "brief_reason": "not mandatory",
                },
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

        _, status = eu.parse_task_response(
            "task1", '{"decision":"yes","confidence":101,"brief_reason":""}'
        )
        self.assertEqual(status, "invalid_confidence")

        _, status = eu.parse_task_response(
            "task2", '{"modality":"optional","confidence":80}'
        )
        self.assertEqual(status, "missing_fields")

        _, status = eu.parse_task_response(
            "task3", '{"relation":"preserves","confidence":80}'
        )
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
                                "confidence": 0.8,
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
            eu.write_csv_rows(
                root / "data/processed/weak_modality_probe_items.csv", weak_probe
            )

            inputs, key = external_export.build_external_probe_rows(root)

            source_counts = Counter(row["source_kind"] for row in key)
            self.assertEqual(len(inputs), 14)
            self.assertEqual(len(key), 14)
            self.assertEqual(source_counts["main_benchmark"], 8)
            self.assertEqual(source_counts["weak_modality_probe"], 6)
            self.assertEqual(
                [row["external_item_id"] for row in inputs],
                [f"EXT{index:04d}" for index in range(1, 15)],
            )
            self.assertEqual(
                [row["external_item_id"] for row in key],
                [row["external_item_id"] for row in inputs],
            )
            self.assertEqual(set(inputs[0]), {"external_item_id", "source_statement"})
            self.assertNotIn("weak_useful_if", {row["source_condition"] for row in key})
            self.assertEqual({row["seed_id"] for row in key}, {"S0001", "S0002"})

    def test_external_probe_export_help_and_dry_run_do_not_write(self):
        buffer = io.StringIO()
        with self.assertRaises(SystemExit) as context, redirect_stdout(buffer):
            external_export.main(["--help"])
        self.assertEqual(context.exception.code, 0)
        self.assertIn("--dry-run", buffer.getvalue())

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config.example.json").write_text(
                json.dumps({"project": {"pilot_seed_count": 2}}),
                encoding="utf-8",
            )
            eu.write_csv_rows(
                root / "data/processed/benchmark_items.csv",
                eu.build_benchmark_items(seed_rows(3)),
            )
            eu.write_csv_rows(
                root / "data/processed/weak_modality_probe_items.csv",
                eu.build_weak_modality_probe_items(seed_rows(3)),
            )

            output_dir = root / "outputs" / "external_ai_service_probe"
            with io.StringIO() as buffer, redirect_stdout(buffer):
                external_export.main(["--root", str(root), "--dry-run"])
                output = buffer.getvalue()

            self.assertIn("Would write 14 blind input rows", output)
            self.assertFalse(output_dir.exists())


def _concurrent_append_worker(args):
    """Append many oversized JSON records to a shared file (separate process)."""
    import sys as _sys

    _sys.path.insert(0, "scripts")
    import eval_utils as worker_eu

    path, worker_id, count = args
    payload = "x" * 5000
    for index in range(count):
        worker_eu.append_jsonl(
            Path(path),
            {"worker": worker_id, "index": index, "payload": payload},
        )
    return count


def _concurrent_registry_worker(args):
    import sys as _sys

    _sys.path.insert(0, "scripts")
    import eval_utils as worker_eu

    path, worker_id, count = args
    for index in range(count):
        worker_eu.upsert_run_registry_row(
            Path(path),
            {
                "run_id": f"full-{worker_id}",
                "profile_id": "p1",
                "model": "m1",
                "dataset_id": "nice",
                "benchmark_variant": "must",
                "observed_records": index,
                "notes": "y" * 5000,
            },
        )
    return count


class SharedArtifactConcurrencyTest(unittest.TestCase):
    APPENDS_PER_WORKER = 200

    def test_concurrent_appends_produce_only_valid_json_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model_outputs_raw.jsonl"
            with multiprocessing.get_context("spawn").Pool(2) as pool:
                pool.map(
                    _concurrent_append_worker,
                    [
                        (str(path), 0, self.APPENDS_PER_WORKER),
                        (str(path), 1, self.APPENDS_PER_WORKER),
                    ],
                )

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2 * self.APPENDS_PER_WORKER)
            rows = [json.loads(line) for line in lines]  # raises on any torn line
            self.assertTrue(all(len(row["payload"]) == 5000 for row in rows))
            self.assertEqual(
                Counter(row["worker"] for row in rows),
                Counter({0: self.APPENDS_PER_WORKER, 1: self.APPENDS_PER_WORKER}),
            )

    def test_concurrent_registry_upserts_keep_the_csv_parseable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run_registry.csv"
            with multiprocessing.get_context("spawn").Pool(2) as pool:
                pool.map(
                    _concurrent_registry_worker,
                    [(str(path), 0, 25), (str(path), 1, 25)],
                )

            rows = eu.read_csv_rows(path)
            self.assertEqual({row["run_id"] for row in rows}, {"full-0", "full-1"})
            self.assertEqual(len(rows), 2)
            self.assertFalse(list(path.parent.glob("*.tmp")))


class SmokeArtifactIsolationTest(unittest.TestCase):
    def test_smoke_run_ids_resolve_to_the_parallel_smoke_tree(self):
        root = Path("/tmp/example-root")
        smoke_id = "smoke-20260903-110841-ba264ed9"
        full_id = "full-20260523-014135-78ceaa43"
        self.assertTrue(eu.is_smoke_run_id(smoke_id))
        self.assertFalse(eu.is_smoke_run_id(full_id))
        for helper in (
            eu.model_outputs_raw_path,
            eu.run_registry_path,
            eu.run_events_path,
            eu.run_progress_live_path,
            eu.task3_raw_path,
            eu.task3_registry_path,
            eu.task3_progress_path,
            eu.task3_events_path,
        ):
            smoke_path = helper(root, "mlm_tapt", "must", run_id=smoke_id)
            full_path = helper(root, "mlm_tapt", "must", run_id=full_id)
            self.assertEqual(smoke_path.parent.name, "smoke")
            self.assertEqual(smoke_path.name, full_path.name)
            self.assertEqual(full_path.parent.name, "processed")
            self.assertEqual(helper(root, "mlm_tapt", "must", smoke=True), smoke_path)

    def test_smoke_tree_mapping_is_idempotent(self):
        once = eu.smoke_tree_path(Path("/tmp/x/data/processed/run_registry.csv"))
        self.assertEqual(eu.smoke_tree_path(once), once)

    def test_fake_smoke_task3_run_writes_its_items_csv_into_the_smoke_tree(self):
        """A fake Task 3 run must not overwrite the paper-facing items CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "prompts").mkdir(parents=True)
            (root / "data/processed").mkdir(parents=True)
            (root / "docs").mkdir()
            (root / "AGENTS.md").write_text("", encoding="utf-8")
            (root / "docs/evaluation.md").write_text("", encoding="utf-8")
            (root / "prompts/modality_verification.txt").write_text(
                Path("prompts/modality_verification.txt").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            benchmark = eu.build_benchmark_items(seed_rows(1))
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

            items_root = root / "data/processed/task3_verification_items"
            items_path = eu.task3_verification_items_path(
                root, "nice", "must", "full-source", "fake-model", "blind", smoke=True
            )
            self.assertTrue(items_path.exists(), items_path)
            self.assertEqual(items_path.parent.name, eu.SMOKE_TREE_DIRNAME)
            self.assertEqual(items_path.parent.parent, items_root)
            # The paper-facing items directory holds only the smoke subdirectory.
            self.assertEqual(
                [entry.name for entry in sorted(items_root.iterdir())],
                [eu.SMOKE_TREE_DIRNAME],
            )
            self.assertFalse(
                eu.task3_verification_items_path(
                    root, "nice", "must", "full-source", "fake-model", "blind"
                ).exists()
            )

    def test_task3_dry_run_never_contacts_provider_or_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "prompts").mkdir(parents=True)
            (root / "data/processed").mkdir(parents=True)
            (root / "prompts/modality_verification.txt").write_text(
                Path("prompts/modality_verification.txt").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            benchmark = eu.build_benchmark_items(seed_rows(1))
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
                        run_group_id="source-group",
                    ),
                )
            run_config = eu.normalize_run_config(
                {
                    "run_group_id": "task3-group",
                    "datasets": ["nice"],
                    "benchmark_variants": ["must"],
                    "tasks": ["task1", "task2"],
                    "prompt_version": "v2-conf01",
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
            )
            args = SimpleNamespace(
                profile="fake",
                model="fake-model",
                dataset="nice",
                variant="must",
                source_run_id="full-source",
                mode="smoke",
                audit_mode="blind",
                run_id=None,
                smoke_items=2,
                fake_completion=False,
                dry_run=True,
                allow_partial_source=True,
                progress_every_records=None,
                progress_every_seconds=None,
                warn_after_records=None,
                warn_parse_failure_rate=None,
                warn_request_error_rate=None,
                no_progress_artifacts=False,
                log_level="INFO",
                resolved_config_yaml="",
            )
            with (
                mock.patch.object(task3_cli.eu, "project_root", return_value=root),
                mock.patch.object(
                    task3_cli.eu,
                    "preflight_profile",
                    side_effect=AssertionError("provider called"),
                ),
                mock.patch.object(
                    task3_cli.eu,
                    "run_completion_jobs",
                    side_effect=AssertionError("provider called"),
                ),
                mock.patch.object(
                    task3_cli.eu,
                    "write_csv_rows",
                    side_effect=AssertionError("artifact written"),
                ),
                mock.patch.object(
                    task3_cli.eu,
                    "upsert_run_registry_row",
                    side_effect=AssertionError("artifact written"),
                ),
                mock.patch.object(
                    task3_cli.eu,
                    "append_jsonl",
                    side_effect=AssertionError("artifact written"),
                ),
                mock.patch.object(
                    task3_cli.rp,
                    "write_resolved_config",
                    side_effect=AssertionError("artifact written"),
                ),
            ):
                task3_cli.run_from_config(run_config, args)

            self.assertFalse((root / "data/processed/smoke").exists())
            self.assertFalse(
                (root / "data/processed/task3_verification_items").exists()
            )


class RunnerDiagnosticsTest(unittest.TestCase):
    def test_task3_source_prefixes_accept_smoke_only_in_smoke_mode(self):
        self.assertEqual(task3_cli.source_run_prefixes("must", "full"), ["full"])
        self.assertEqual(
            task3_cli.source_run_prefixes("must", "smoke"), ["full", "smoke"]
        )
        self.assertEqual(
            task3_cli.source_run_prefixes("shall", "smoke"),
            ["full-shall", "smoke-shall"],
        )
        rows = [{"run_id": "smoke-1", "model": "m1"}]
        self.assertEqual(
            eu.select_run_rows(rows, run_id="smoke-1", prefix=["full"]), ("smoke-1", [])
        )
        self.assertEqual(
            eu.select_run_rows(rows, run_id="smoke-1", prefix=["full", "smoke"])[1],
            rows,
        )

    def test_every_task3_smoke_prefix_routes_to_the_smoke_tree(self):
        for variant in ("must", "shall"):
            for audit_mode in eu.TASK3_AUDIT_MODES:
                with self.subTest(variant=variant, audit_mode=audit_mode):
                    run_id = f"{task3_cli.task3_run_prefix('smoke', variant, audit_mode)}-20260903-000000-abc"
                    self.assertTrue(eu.is_smoke_run_id(run_id), run_id)
                    self.assertEqual(
                        eu.task3_raw_path(
                            "/tmp/example-root", variant=variant, run_id=run_id
                        ).parent.name,
                        eu.SMOKE_TREE_DIRNAME,
                    )

    def test_parse_status_histogram_counts_truncated_as_a_failure(self):
        rows = [
            {"parse_status": "ok", "latency_s": 1.0, "usage_completion_tokens": 10},
            {
                "parse_status": "truncated",
                "latency_s": 3.0,
                "retry_count": 2,
                "usage_completion_tokens": 5,
            },
            {"parse_status": "invalid_json", "latency_s": 2.0},
            {"parse_status": "weird_new_status"},
        ]
        histogram = eu.parse_status_histogram(rows)
        self.assertEqual(histogram["truncated"], 1)
        self.assertEqual(histogram["invalid_json"], 1)
        self.assertEqual(histogram["other"], 1)
        self.assertEqual(histogram["missing_batch_result"], 0)
        self.assertTrue(eu.is_parse_failure_status("truncated"))
        self.assertFalse(eu.is_parse_failure_status("ok"))

        quality = eu.run_quality_counters(rows)
        self.assertEqual(quality["retry_total"], 2)
        self.assertEqual(quality["truncated_records"], 1)
        self.assertEqual(quality["usage_completion_tokens"], 15)
        self.assertEqual(quality["latency_p50_s"], 2.0)
        self.assertEqual(quality["latency_p95_s"], 3.0)

        counters = eu.live_run_counters(rows, expected_records=4, expected_api_calls=1)
        self.assertEqual(counters["parse_failure_records"], 3)
        self.assertEqual(counters["truncated_records"], 1)
        # A finished run reports a concrete ETA instead of "unknown".
        self.assertEqual(counters["eta_s"], 0.0)
        self.assertIn("eta 0s", eu.format_live_progress_line("full-1", counters))

    def test_batched_request_diagnostics_are_counted_once(self):
        rows = [
            {
                "batch_id": "batch-1",
                "batch_item_count": 2,
                "parse_status": "ok",
                "latency_s": 4.0,
                "retry_count": 2,
                "usage_completion_tokens": 80,
            },
            {
                "batch_id": "batch-1",
                "batch_item_count": 2,
                "parse_status": "ok",
                "latency_s": 4.0,
                "retry_count": 2,
                "usage_completion_tokens": 80,
            },
            {
                "parse_status": "ok",
                "latency_s": 1.0,
                "retry_count": 1,
                "usage_completion_tokens": 10,
            },
        ]
        quality = eu.run_quality_counters(rows)
        self.assertEqual(quality["retry_total"], 3)
        self.assertEqual(quality["usage_completion_tokens"], 90)
        self.assertEqual(quality["latency_p50_s"], 1.0)
        self.assertEqual(quality["latency_p95_s"], 4.0)
        self.assertEqual(
            eu.answer_length_fields(
                rows[0], {"source_statement": "The system may export."}
            )["completion_tokens"],
            "",
        )

    def test_configure_run_logging_replaces_the_previous_file_handler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first.log"
            second = Path(tmpdir) / "second.log"
            logger = eu.configure_run_logging("INFO", log_path=first)
            logger.info("first-only")
            logger = eu.configure_run_logging("INFO", log_path=second)
            logger.info("second-only")
            for handler in logger.handlers:
                handler.flush()
            eu.configure_run_logging("INFO")

            self.assertIn("first-only", first.read_text(encoding="utf-8"))
            self.assertNotIn("second-only", first.read_text(encoding="utf-8"))
            self.assertIn("second-only", second.read_text(encoding="utf-8"))
            self.assertEqual(
                sum(
                    getattr(handler, "_re_uq_target", None) != "stream"
                    for handler in eu.logger.handlers
                ),
                0,
            )

    def test_registry_compatibility_rejects_deterministic_only_ablation(self):
        row = {
            "run_group_id": "paper-group",
            "tasks": "task1,task2",
            "status": "complete",
            "expected_records": 120,
            "observed_records": 120,
            "deterministic_item_coverage": 1.0,
            "stochastic_complete_item_rate": 1.0,
            "batch_order": "grouped",
            "batch_size": 16,
        }
        kwargs = {
            "run_group_id": "paper-group",
            "benchmark_item_count": 10,
            "expected_stochastic_samples": 5,
            "required_tasks": ("task2",),
            "expected_batch_order": "grouped",
            "expected_batch_size": 16,
        }
        self.assertEqual(eu.registry_row_compatibility_issues(row, **kwargs), [])
        ablation = {
            **row,
            "tasks": "task2",
            "expected_records": 10,
            "observed_records": 10,
            "stochastic_complete_item_rate": 0.0,
            "batch_order": "shuffled",
        }
        issues = eu.registry_row_compatibility_issues(ablation, **kwargs)
        self.assertTrue(any("expected_records" in issue for issue in issues))
        self.assertTrue(any("stochastic coverage" in issue for issue in issues))
        self.assertTrue(any("batch_order" in issue for issue in issues))

    def test_registry_row_carries_run_quality_and_batch_order(self):
        benchmark = eu.build_benchmark_items(seed_rows(1))
        raw_rows = [
            {
                "run_id": "full-1",
                "model": "m1",
                "task": "task2",
                "item_id": benchmark[0]["item_id"],
                "sample_kind": "deterministic",
                "sample_index": 0,
                "parse_status": "truncated",
                "latency_s": 4.0,
                "retry_count": 1,
                "usage_completion_tokens": 42,
            }
        ]
        row = eu.run_registry_summary(
            benchmark,
            raw_rows,
            run_id="full-1",
            run_group_id="g1",
            provider_id="p",
            profile_id="p",
            model="m1",
            dataset_id="nice",
            variant="must",
            tasks=["task2"],
            expected_stochastic_samples=0,
            started_at_utc="2026-09-03T00:00:00Z",
            batch_order="shuffled",
        )
        self.assertEqual(row["batch_order"], "shuffled")
        self.assertEqual(row["retry_total"], 1)
        self.assertEqual(row["truncated_records"], 1)
        self.assertEqual(row["usage_completion_tokens"], 42)
        self.assertEqual(row["latency_p50_s"], 4.0)
        self.assertIn("truncated", row["parse_status_histogram"])
        self.assertEqual(set(row) - set(eu.RUN_REGISTRY_FIELDS), set())

    def test_show_run_progress_scopes_expected_counts_to_the_planned_jobs(self):
        benchmark = eu.build_benchmark_items(seed_rows(10))
        raw_rows = [{"item_id": row["item_id"]} for row in benchmark[:2]]
        # A smoke run: 2 items x 2 tasks x (1 deterministic + 5 stochastic) = 24.
        registry_row = {"tasks": "task1,task2", "expected_records": "24"}
        planned = show_run_progress.planned_benchmark_rows(
            benchmark, registry_row, raw_rows
        )
        self.assertEqual(len(planned), 2)
        summary = eu.run_progress_summary(
            planned,
            [
                {
                    "run_id": "smoke-1",
                    "model": "m1",
                    "task": "task1",
                    "item_id": row["item_id"],
                    "sample_kind": "deterministic",
                    "sample_index": 0,
                    "parse_status": "ok",
                }
                for row in benchmark[:2]
            ],
            expected_stochastic_samples=5,
        )
        self.assertEqual(summary[0]["expected_records"], 12)
        # Without a usable registry row, scope to the item ids the run produced.
        self.assertEqual(
            show_run_progress.planned_benchmark_rows(benchmark, {}, raw_rows),
            benchmark[:2],
        )
        # A full run keeps the whole benchmark.
        self.assertEqual(
            len(
                show_run_progress.planned_benchmark_rows(
                    benchmark, {"tasks": "task1,task2"}, []
                )
            ),
            len(benchmark),
        )
        # A partially written full run must not scope itself to observed ids.
        full_registry_row = {
            "tasks": "task1,task2",
            "expected_records": str(len(benchmark) * 2 * 6),
        }
        self.assertEqual(
            show_run_progress.planned_benchmark_rows(
                benchmark, full_registry_row, raw_rows
            ),
            benchmark,
        )

    def test_watch_reader_only_parses_newly_appended_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model_outputs_raw.jsonl"
            eu.append_jsonl(path, {"run_id": "full-1", "index": 0})
            reader = show_run_progress.RawRowReader()
            self.assertEqual(len(reader.read(path)), 1)
            offset_after_first = reader._offset
            eu.append_jsonl(path, {"run_id": "full-1", "index": 1})
            rows = reader.read(path)
            self.assertEqual([row["index"] for row in rows], [0, 1])
            self.assertGreater(reader._offset, offset_after_first)
            # Unchanged file: no re-read, same rows.
            self.assertEqual(len(reader.read(path)), 2)


if __name__ == "__main__":
    unittest.main()
