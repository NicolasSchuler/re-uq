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
from typing import ClassVar
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


def _run_config(
    root: Path,
    *,
    datasets: list[str] = DATASETS,
    variants: list[str] = VARIANTS,
    models: list[str] = MODELS,
    logging_overrides: dict[str, object] | None = None,
) -> Path:
    """Write a matrix run config and return its path."""
    path = root / "run_config.json"
    path.write_text(
        json.dumps(
            {
                "run_group_id": "matrix-group",
                "datasets": datasets,
                "benchmark_variants": variants,
                "stochastic": {"temperature": 0.7, "top_p": 1.0, "samples": 1},
                "logging": {
                    "progress_every_records": 2,
                    "progress_every_seconds": 999,
                    "warn_after_records": 2,
                    **(logging_overrides or {}),
                },
                "profiles": [
                    {
                        "profile_id": "fake",
                        "provider_id": "fake",
                        "base_url": "http://127.0.0.1:1234/v1",
                        "api_key_env": "LOCAL_OPENAI_API_KEY",
                        "models": models,
                        "concurrency": 1,
                        "batch_size": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _runner_args(**overrides: object) -> mock.Mock:
    """Build the argparse-shaped options object both runner front doors pass."""
    defaults: dict[str, object] = {
        "profile": "fake",
        "model": None,
        "all_models": True,
        "dataset": None,
        "variant": None,
        "task": None,
        "mode": "smoke",
        "run_id": None,
        "smoke_items": 2,
        "fake_completion": True,
        "dry_run": False,
        "log_level": "WARNING",
        "progress_every_records": None,
        "progress_every_seconds": None,
        "warn_after_records": None,
        "warn_parse_failure_rate": None,
        "warn_request_error_rate": None,
        "no_progress_artifacts": False,
        "resolved_config_yaml": "",
    }
    return mock.Mock(**(defaults | overrides))


def _scaffold(root: Path) -> None:
    """Create the temp project layout with one benchmark CSV per matrix cell."""
    (root / "prompts").mkdir(parents=True)
    (root / "data/processed").mkdir(parents=True)
    for name in (
        "mandatory_entailment",
        "modality_extraction",
        "modality_extraction_context",
    ):
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
        args = _runner_args()
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

    def test_every_event_carries_the_full_cell_identity(self) -> None:
        identity = {
            "run_id",
            "run_group_id",
            "dataset_id",
            "benchmark_variant",
            "provider_id",
            "profile_id",
            "model",
        }
        seen = 0
        for dataset_id in DATASETS:
            for variant in VARIANTS:
                events = eu.read_jsonl(
                    eu.run_events_path(self.root, dataset_id, variant, smoke=True)
                )
                self.assertTrue(events)
                for event in events:
                    seen += 1
                    with self.subTest(event_type=event.get("event_type")):
                        # Warning events inherit the same identity block, so a
                        # log line can always be traced back to its run group.
                        self.assertLessEqual(identity, set(event))
                        self.assertEqual(event["run_group_id"], "matrix-group")
        self.assertGreater(seen, 0)

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
                    eu.run_progress_live_path(
                        self.root, dataset_id, variant, smoke=True
                    )
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


class RunnerItemContextTest(unittest.TestCase):
    """The document arm is provenance-visible end to end and Task 2 only."""

    PURE_SEEDS: ClassVar[list[dict[str, str]]] = [
        {
            **seed,
            "context_document": "Fixture FRS",
            "context_legend": "(M) mandatory, (O) optional",
            "context_section": "1 Fixture",
            "context_requirement_id": f"1.{index}",
            "context_marker": "M" if index == 1 else "O",
            "context_before": "",
            "context_after": "",
        }
        for index, seed in enumerate(SEEDS, start=1)
    ]

    def _scaffold_pure(self, root: Path) -> None:
        _scaffold(root)
        eu.write_csv_rows(
            eu.artifact_path(root / "data/processed/benchmark_items.csv", "pure"),
            eu.build_benchmark_items(
                self.PURE_SEEDS, passthrough_fields=eu.PURE_CONTEXT_FIELDS
            ),
        )

    def _run(self, root: Path, item_context: str, task: str | None = "task2") -> None:
        config_path = _run_config(
            root, datasets=["pure"], variants=["must"], models=["fake-model-a"]
        )
        run_config = eu.load_run_config(config_path)
        run_config["item_context"] = item_context
        with (
            mock.patch.object(eu, "project_root", return_value=root),
            redirect_stdout(io.StringIO()),
        ):
            # smoke_items=8 keeps both seeds (4 items each), so both markers run.
            runner.run_from_config(run_config, _runner_args(task=task, smoke_items=8))

    def test_document_arm_requires_task2_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._scaffold_pure(root)
            with self.assertRaises(ValueError):
                self._run(root, "document", task=None)

    def test_both_arms_write_their_context_into_records_and_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._scaffold_pure(root)
            self._run(root, "bare")
            self._run(root, "document")

            registry = eu.read_csv_rows(
                eu.run_registry_path(root, "pure", "must", smoke=True)
            )
            self.assertEqual(
                sorted(row["item_context"] for row in registry), ["bare", "document"]
            )
            self.assertTrue(all(row["status"] == "complete" for row in registry))

            raw = eu.read_jsonl(
                eu.model_outputs_raw_path(root, "pure", "must", smoke=True)
            )
            by_context: dict[str, list[dict[str, object]]] = {}
            for row in raw:
                by_context.setdefault(str(row["item_context"]), []).append(row)
            self.assertEqual(set(by_context), {"bare", "document"})
            self.assertEqual(
                {row["context_marker"] for row in by_context["document"]}, {"M", "O"}
            )
            self.assertTrue(
                all(
                    "Document context" in str(row["prompt"])
                    for row in by_context["document"]
                )
            )
            self.assertTrue(
                all(
                    "Document context" not in str(row["prompt"])
                    for row in by_context["bare"]
                )
            )
            self.assertNotEqual(
                {row["job_config_sha"] for row in by_context["bare"]},
                {row["job_config_sha"] for row in by_context["document"]},
            )
            # Same items, same answers from the offline fixture: the arms differ
            # only in what the model was shown.
            self.assertEqual(
                sorted(str(row["item_id"]) for row in by_context["bare"]),
                sorted(str(row["item_id"]) for row in by_context["document"]),
            )


class RunnerWarningEventTest(unittest.TestCase):
    """Warning events must be traceable to their run group, like progress events."""

    def test_warning_events_carry_the_same_cell_identity_as_progress_events(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _scaffold(root)
            config_path = _run_config(
                root,
                datasets=["nice"],
                variants=["must"],
                models=["fake-model-a"],
                logging_overrides={
                    "warn_after_records": 1,
                    "warn_parse_failure_rate": 0.0,
                },
            )
            unparseable = {
                "ok": True,
                "raw_text": "not json at all",
                "response_json": {},
                "latency_s": 0.0,
                "error": "",
            }
            probe_ok = runner.fake_completion

            def only_the_probe_parses(**kwargs: object) -> dict[str, object]:
                # The provider preflight uses the same completion function and
                # refuses to start on an unparseable reply, so keep it healthy
                # and fail only the benchmark jobs.
                if "probe" in str(kwargs.get("prompt", "")):
                    return probe_ok(**kwargs)
                return unparseable

            with (
                mock.patch.object(eu, "project_root", return_value=root),
                mock.patch.object(
                    runner, "fake_completion", side_effect=only_the_probe_parses
                ),
                redirect_stdout(io.StringIO()),
            ):
                runner.run_from_config(eu.load_run_config(config_path), _runner_args())

            events = eu.read_jsonl(eu.run_events_path(root, "nice", "must", smoke=True))
            warnings = [e for e in events if e.get("event_type") == "warning"]
            self.assertTrue(warnings, "a fully unparseable run must warn")
            for warning in warnings:
                with self.subTest(warning_type=warning.get("warning_type")):
                    self.assertEqual(warning["run_group_id"], "matrix-group")
                    self.assertEqual(warning["dataset_id"], "nice")
                    self.assertEqual(warning["benchmark_variant"], "must")
                    self.assertEqual(warning["model"], "fake-model-a")
                    self.assertEqual(warning["profile_id"], "fake")
                    self.assertEqual(warning["provider_id"], "fake")
                    self.assertTrue(warning["run_id"])


if __name__ == "__main__":
    unittest.main()
