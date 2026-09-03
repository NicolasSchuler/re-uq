"""Characterization tests for the provider-matrix runner loops.

``run_experiment_from_config.run_from_config`` fans a single config out over
profiles x models x datasets x benchmark variants. The existing CLI smoke test
in ``test_eval_utils`` only covers a single cell, so it cannot see whether the
per-cell state (run id, output routing, registry identity, live progress) stays
correctly separated once more than one cell runs in the same process.

These tests pin that fan-out behaviour: they assert on the complete artifact
set a fake multi-cell run produces, so a refactor of the runner has to keep
every cell writing the same rows to the same places.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts import eval_utils as eu, run_experiment_from_config as runner

DATASETS = ["nice", "mlm_tapt"]
VARIANTS = ["must", "shall"]
MODELS = ["fake-model-a", "fake-model-b"]
SEEDS = [
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


def _run_config(root: Path) -> Path:
    """Write the matrix run config and return its path."""
    path = root / "run_config.json"
    path.write_text(
        json.dumps(
            {
                "run_group_id": "matrix-group",
                "datasets": DATASETS,
                "benchmark_variants": VARIANTS,
                "stochastic": {"temperature": 0.7, "top_p": 1.0, "samples": 1},
                "logging": {
                    "progress_every_records": 2,
                    "progress_every_seconds": 999,
                    "warn_after_records": 2,
                },
                "profiles": [
                    {
                        "profile_id": "fake",
                        "provider_id": "fake",
                        "base_url": "http://127.0.0.1:1234/v1",
                        "api_key_env": "LOCAL_OPENAI_API_KEY",
                        "models": MODELS,
                        "concurrency": 1,
                        "batch_size": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _scaffold(root: Path) -> None:
    """Create the temp project layout with one benchmark CSV per matrix cell."""
    (root / "prompts").mkdir(parents=True)
    (root / "data/processed").mkdir(parents=True)
    for name in ("mandatory_entailment", "modality_extraction"):
        (root / f"prompts/{name}.txt").write_text(
            Path(f"prompts/{name}.txt").read_text(encoding="utf-8"), encoding="utf-8"
        )
    benchmark = eu.build_benchmark_items(SEEDS)
    for dataset_id in DATASETS:
        for variant in VARIANTS:
            eu.write_csv_rows(
                eu.artifact_path(
                    root / "data/processed/benchmark_items.csv", dataset_id, variant
                ),
                benchmark,
            )


class RunnerMatrixTest(unittest.TestCase):
    """Pin the artifacts a 2x2x2 fake matrix run produces."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)
        _scaffold(self.root)
        args = mock.Mock(
            profile="fake",
            model=None,
            all_models=True,
            dataset=None,
            variant=None,
            task=None,
            mode="smoke",
            run_id=None,
            smoke_items=2,
            fake_completion=True,
            dry_run=False,
            log_level="WARNING",
            progress_every_records=None,
            progress_every_seconds=None,
            warn_after_records=None,
            warn_parse_failure_rate=None,
            warn_request_error_rate=None,
            no_progress_artifacts=False,
            resolved_config_yaml="",
        )
        run_config = eu.load_run_config(_run_config(self.root))
        with (
            mock.patch.object(eu, "project_root", return_value=self.root),
            redirect_stdout(io.StringIO()),
        ):
            runner.run_from_config(run_config, args)

    def _registry_rows(self, dataset_id: str, variant: str) -> list[dict[str, str]]:
        return eu.read_csv_rows(
            eu.run_registry_path(self.root, dataset_id, variant, smoke=True)
        )

    def _raw_rows(self, dataset_id: str, variant: str) -> list[dict[str, object]]:
        return eu.read_jsonl(
            eu.model_outputs_raw_path(self.root, dataset_id, variant, smoke=True)
        )

    def test_every_matrix_cell_produces_one_complete_registry_row(self) -> None:
        for dataset_id in DATASETS:
            for variant in VARIANTS:
                rows = self._registry_rows(dataset_id, variant)
                with self.subTest(dataset=dataset_id, variant=variant):
                    self.assertEqual([row["model"] for row in rows], MODELS)
                    for row in rows:
                        self.assertEqual(row["status"], "complete")
                        self.assertEqual(row["dataset_id"], dataset_id)
                        self.assertEqual(row["benchmark_variant"], variant)
                        self.assertEqual(row["run_group_id"], "matrix-group")
                        self.assertEqual(row["profile_id"], "fake")

    def test_each_cell_gets_a_distinct_run_id(self) -> None:
        run_ids = [
            row["run_id"]
            for dataset_id in DATASETS
            for variant in VARIANTS
            for row in self._registry_rows(dataset_id, variant)
        ]
        self.assertEqual(len(run_ids), 8)
        self.assertEqual(len(set(run_ids)), 8)

    def test_run_id_prefix_encodes_mode_and_variant(self) -> None:
        for dataset_id in DATASETS:
            for variant in VARIANTS:
                expected = "smoke" if variant == "must" else f"smoke-{variant}"
                for row in self._registry_rows(dataset_id, variant):
                    with self.subTest(dataset=dataset_id, variant=variant):
                        self.assertTrue(str(row["run_id"]).startswith(f"{expected}-"))

    def test_raw_rows_are_partitioned_by_dataset_and_variant(self) -> None:
        for dataset_id in DATASETS:
            for variant in VARIANTS:
                rows = self._raw_rows(dataset_id, variant)
                cell_run_ids = {
                    row["run_id"] for row in self._registry_rows(dataset_id, variant)
                }
                with self.subTest(dataset=dataset_id, variant=variant):
                    # 2 items x 2 tasks x 2 models x (1 deterministic + 1
                    # stochastic) sample.
                    self.assertEqual(len(rows), 16)
                    self.assertEqual({str(row["model"]) for row in rows}, set(MODELS))
                    self.assertEqual({str(row["run_id"]) for row in rows}, cell_run_ids)
                    self.assertTrue(
                        all(row["parse_status"] == "ok" for row in rows),
                        "fake completions must all parse",
                    )

    def test_fake_run_never_writes_into_the_paper_facing_tree(self) -> None:
        stray = [
            path
            for path in (self.root / "data/processed").glob("*")
            if path.is_file() and path.name.startswith(("model_outputs", "run_"))
        ]
        self.assertEqual(stray, [], "fake runs must stay inside data/processed/smoke")

    def test_progress_and_event_artifacts_cover_every_cell(self) -> None:
        for dataset_id in DATASETS:
            for variant in VARIANTS:
                events = eu.read_jsonl(
                    eu.run_events_path(self.root, dataset_id, variant, smoke=True)
                )
                progress = eu.read_csv_rows(
                    eu.run_progress_live_path(self.root, dataset_id, variant, smoke=True)
                )
                cell_run_ids = {
                    row["run_id"] for row in self._registry_rows(dataset_id, variant)
                }
                with self.subTest(dataset=dataset_id, variant=variant):
                    self.assertTrue(progress)
                    by_run: dict[str, set[str]] = {}
                    for event in events:
                        by_run.setdefault(str(event["run_id"]), set()).add(
                            str(event["event_type"])
                        )
                    self.assertEqual(set(by_run), cell_run_ids)
                    for run_id, kinds in by_run.items():
                        with self.subTest(run_id=run_id):
                            self.assertIn("start", kinds)
                            self.assertIn("finish", kinds)


if __name__ == "__main__":
    unittest.main()
