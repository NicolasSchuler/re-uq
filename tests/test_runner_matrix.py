"""Characterization tests for the provider-matrix runner loops and lifecycle.

``run_experiment_from_config.run_from_config`` fans a single config out over
profiles x models x datasets x benchmark variants. The existing CLI smoke test
in ``test_eval_utils`` only covers a single cell, so it cannot see whether the
per-cell state (run id, output routing, registry identity, live progress) stays
correctly separated once more than one cell runs in the same process.

The first half of this module pins that fan-out behaviour: it asserts on the
complete artifact set a fake multi-cell run produces, so a refactor of the
runner has to keep every cell writing the same rows to the same places.

The second half pins the lifecycle protocol both runners now share through
``scripts/runner_lifecycle.py``: the CLI option contract, the resume rules
(a run id must exist and match, and its start time and provenance survive the
resume), terminal-state reconciliation after a failure or a Ctrl-C, and the
per-cell lease that stops a second runner from paying for the same cell twice.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import ClassVar
from unittest import mock

from scripts import (
    eval_utils as eu,
    run_experiment_from_config as runner,
    run_provenance as rp,
    run_task3_verification_from_config as task3_runner,
    runner_lifecycle as rl,
)

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
        # Pinned so registry `notes` do not pick up an auto-created Mock
        # attribute; the Hydra bridge always supplies both provenance fields.
        "resolved_config_sha": "",
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


# =============================================================================
# Shared CLI surface
# =============================================================================

# The option contract both runner CLIs are expected to expose, as
# (flag, default, type name, choices, required, help). `common_runner_parser()`
# is what makes the shared block below literally one declaration; pinning it
# here keeps the two front doors from drifting apart again, and keeps a
# refactor from quietly changing a default or a help string.
CliOption = tuple[str, object, str | None, tuple[str, ...] | None, bool, str | None]

COMMON_CLI_OPTIONS: dict[str, CliOption] = {
    "config": ("--config", None, "Path", None, True, None),
    "profile": ("--profile", None, None, None, False, None),
    "model": ("--model", None, None, None, False, None),
    "dataset": ("--dataset", None, None, None, False, None),
    "variant": ("--variant", None, None, None, False, None),
    "mode": ("--mode", "smoke", None, ("smoke", "full", "resume"), False, None),
    "run_id": ("--run-id", None, None, None, False, None),
    "smoke_items": ("--smoke-items", 2, "int", None, False, None),
    "fake_completion": ("--fake-completion", False, None, None, False, None),
    "progress_every_records": (
        "--progress-every-records",
        None,
        "int",
        None,
        False,
        None,
    ),
    "progress_every_seconds": (
        "--progress-every-seconds",
        None,
        "int",
        None,
        False,
        None,
    ),
    "warn_after_records": ("--warn-after-records", None, "int", None, False, None),
    "warn_parse_failure_rate": (
        "--warn-parse-failure-rate",
        None,
        "float",
        None,
        False,
        None,
    ),
    "warn_request_error_rate": (
        "--warn-request-error-rate",
        None,
        "float",
        None,
        False,
        None,
    ),
    "no_progress_artifacts": (
        "--no-progress-artifacts",
        False,
        None,
        None,
        False,
        None,
    ),
    "log_level": (
        "--log-level",
        "INFO",
        None,
        None,
        False,
        "Logging level for the re_uq logger (default: INFO).",
    ),
}

EXPERIMENT_ONLY_CLI_OPTIONS: dict[str, CliOption] = {
    "task": ("--task", None, None, ("task1", "task2", "both"), False, None),
    "all_models": (
        "--all-models",
        False,
        None,
        None,
        False,
        (
            "Iterate every model of the selected profile(s) sequentially "
            "(ignores --model)."
        ),
    ),
    # `--dry-run` stays with each runner precisely because its help text
    # differs; the Task 3 CLI has never documented it.
    "dry_run": (
        "--dry-run",
        False,
        None,
        None,
        False,
        (
            "Print the planned job/batch/API-call counts and exit without "
            "contacting a provider."
        ),
    ),
}

TASK3_ONLY_CLI_OPTIONS: dict[str, CliOption] = {
    "source_run_id": ("--source-run-id", None, None, None, True, None),
    "audit_mode": (
        "--audit-mode",
        "blind",
        None,
        ("blind", "declared_text", "declared_source"),
        False,
        None,
    ),
    "dry_run": ("--dry-run", False, None, None, False, None),
    "allow_partial_source": (
        "--allow-partial-source",
        False,
        None,
        None,
        False,
        None,
    ),
    "allow_source_profile_mismatch": (
        "--allow-source-profile-mismatch",
        False,
        None,
        None,
        False,
        (
            "Audit Task 2 rows produced under a DIFFERENT provider profile. Off "
            "by default: the audited source profile is recorded in the registry "
            "notes and the mismatch is logged as a warning."
        ),
    ),
}


def _cli_options(parser: argparse.ArgumentParser) -> dict[str, CliOption]:
    """Every option of `parser` except `-h`, as the pinned tuple shape."""
    options: dict[str, CliOption] = {}
    # `_actions` is the only complete view argparse exposes.
    for action in parser._actions:
        if action.dest == "help":
            continue
        options[action.dest] = (
            action.option_strings[0],
            action.default,
            getattr(action.type, "__name__", None) if action.type else None,
            tuple(action.choices) if action.choices else None,
            action.required,
            action.help,
        )
    return options


class RunnerCliParityTest(unittest.TestCase):
    """Both CLIs keep their exact option set, types, defaults, and help text.

    The two parsers now share an `add_help=False` parent, which forces the
    common options to the front of `--help`. Ordering is the only thing that
    may move: everything a caller can actually pass is pinned below.
    """

    def test_experiment_cli_exposes_the_shared_options_plus_its_own(self) -> None:
        self.assertEqual(
            _cli_options(runner.build_parser()),
            COMMON_CLI_OPTIONS | EXPERIMENT_ONLY_CLI_OPTIONS,
        )

    def test_task3_cli_exposes_the_shared_options_plus_its_own(self) -> None:
        self.assertEqual(
            _cli_options(task3_runner.build_parser()),
            COMMON_CLI_OPTIONS | TASK3_ONLY_CLI_OPTIONS,
        )

    def test_the_shared_block_really_comes_from_one_declaration(self) -> None:
        parent = _cli_options(rl.common_runner_parser())
        self.assertEqual(parent, COMMON_CLI_OPTIONS)

    def test_parsing_the_same_argv_gives_both_runners_the_same_values(self) -> None:
        shared = [
            "--config",
            "cfg.json",
            "--profile",
            "p",
            "--model",
            "m",
            "--dataset",
            "nice",
            "--variant",
            "must",
            "--mode",
            "full",
            "--run-id",
            "r1",
            "--smoke-items",
            "7",
            "--fake-completion",
            "--progress-every-records",
            "3",
            "--progress-every-seconds",
            "4",
            "--warn-after-records",
            "5",
            "--warn-parse-failure-rate",
            "0.25",
            "--warn-request-error-rate",
            "0.5",
            "--no-progress-artifacts",
            "--log-level",
            "DEBUG",
            "--dry-run",
        ]
        experiment = runner.build_parser().parse_args(shared)
        task3 = task3_runner.build_parser().parse_args(
            [*shared, "--source-run-id", "full-source"]
        )
        for dest in COMMON_CLI_OPTIONS:
            with self.subTest(option=dest):
                self.assertEqual(getattr(experiment, dest), getattr(task3, dest))
        self.assertTrue(experiment.dry_run)
        self.assertTrue(task3.dry_run)

    def test_help_renders_for_both_runners(self) -> None:
        for parser in (runner.build_parser(), task3_runner.build_parser()):
            text = parser.format_help()
            with self.subTest(prog=parser.prog):
                for flag, *_ in COMMON_CLI_OPTIONS.values():
                    self.assertIn(flag, text)


# =============================================================================
# Lifecycle protocol: resume, failure reconciliation, per-cell leases
# =============================================================================


def _processed_tree(root: Path) -> dict[str, bytes]:
    """Every file under `data/processed`, so a test can prove nothing changed."""
    base = root / "data/processed"
    if not base.exists():
        return {}
    return {
        str(path.relative_to(base)): path.read_bytes()
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


def _dead_pid() -> int:
    """A pid that is guaranteed not to be running: a child we already reaped."""
    process = subprocess.Popen([sys.executable, "-c", ""])
    process.wait()
    return process.pid


class SingleCellRunnerTestCase(unittest.TestCase):
    """One `nice`/`must`/`fake-model-a` cell in a throwaway project root."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)
        _scaffold(self.root)
        self.config = eu.load_run_config(
            _run_config(
                self.root,
                datasets=["nice"],
                variants=["must"],
                models=["fake-model-a"],
            )
        )

    def run_cell(self, **overrides: object) -> None:
        args = _runner_args(all_models=False, model="fake-model-a", **overrides)
        with (
            mock.patch.object(eu, "project_root", return_value=self.root),
            redirect_stdout(io.StringIO()),
        ):
            runner.run_from_config(self.config, args)

    @property
    def registry_path(self) -> Path:
        return eu.run_registry_path(self.root, "nice", "must", smoke=True)

    def registry_row(self) -> dict[str, str]:
        rows = eu.read_csv_rows(self.registry_path)
        self.assertEqual(len(rows), 1, rows)
        return rows[0]

    def lease_path(self, run_id: str) -> Path:
        return rl.cell_lease_path(
            self.root,
            run_id,
            rl.cell_lease_key(
                profile_id="fake",
                model="fake-model-a",
                dataset_id="nice",
                variant="must",
            ),
        )


class RunnerResumeProtocolTest(SingleCellRunnerTestCase):
    """`--mode resume` must name a real, compatible run before anything is written."""

    RUN_ID = "full-resume-fixture"

    def test_resume_with_an_unknown_run_id_fails_before_writing(self) -> None:
        self.run_cell(mode="full", run_id=self.RUN_ID)
        before = _processed_tree(self.root)

        with self.assertRaises(rl.ResumeError) as caught:
            self.run_cell(mode="resume", run_id="full-resume-fixtrue")

        self.assertIn("full-resume-fixtrue", str(caught.exception))
        self.assertIn(self.RUN_ID, str(caught.exception))
        # A typo must not open a log, stamp a resolved config, or add a row.
        self.assertEqual(_processed_tree(self.root), before)

    def test_resume_before_any_run_exists_fails(self) -> None:
        before = _processed_tree(self.root)
        with self.assertRaises(rl.ResumeError) as caught:
            self.run_cell(mode="resume", run_id=self.RUN_ID)
        self.assertIn("no run registry", str(caught.exception))
        self.assertEqual(_processed_tree(self.root), before)

    def test_resume_with_a_different_task_selection_fails(self) -> None:
        self.run_cell(mode="full", run_id=self.RUN_ID)
        before = _processed_tree(self.root)

        with self.assertRaises(rl.ResumeError) as caught:
            self.run_cell(mode="resume", run_id=self.RUN_ID, task="task2")

        self.assertIn("tasks=", str(caught.exception))
        self.assertEqual(_processed_tree(self.root), before)

    def test_resume_preserves_the_original_start_time_and_provenance(self) -> None:
        self.run_cell(
            mode="full",
            run_id=self.RUN_ID,
            resolved_config_yaml="first: composition\n",
            resolved_config_sha="sha-first",
        )
        first = self.registry_row()
        resolved = rp.resolved_config_path(self.root, self.RUN_ID)
        self.assertEqual(resolved.read_text(encoding="utf-8"), "first: composition\n")
        self.assertEqual(first["notes"], "mode=full; resolved_config_sha=sha-first")
        self.assertEqual(first["status"], "complete")

        later = "2099-01-01T00:00:00Z"
        self.assertNotEqual(first["started_at_utc"], later)
        with mock.patch.object(eu, "utc_now_iso", return_value=later):
            self.run_cell(
                mode="resume",
                run_id=self.RUN_ID,
                resolved_config_yaml="second: composition\n",
                resolved_config_sha="sha-second",
            )

        resumed = self.registry_row()
        self.assertEqual(resumed["started_at_utc"], first["started_at_utc"])
        self.assertEqual(resumed["notes"], first["notes"])
        # The clock did move; only the start time is pinned to the first attempt.
        self.assertEqual(resumed["finished_at_utc"], later)
        self.assertEqual(resolved.read_text(encoding="utf-8"), "first: composition\n")
        self.assertEqual(rp.read_resume_log(self.root, self.RUN_ID), [later])

    def test_every_resume_appends_to_the_resumed_at_list(self) -> None:
        self.run_cell(mode="full", run_id=self.RUN_ID)
        for stamp in ("2099-01-01T00:00:00Z", "2099-01-02T00:00:00Z"):
            with mock.patch.object(eu, "utc_now_iso", return_value=stamp):
                self.run_cell(mode="resume", run_id=self.RUN_ID)
        self.assertEqual(
            rp.read_resume_log(self.root, self.RUN_ID),
            ["2099-01-01T00:00:00Z", "2099-01-02T00:00:00Z"],
        )

    def test_a_dry_run_resume_validates_without_recording_anything(self) -> None:
        self.run_cell(mode="full", run_id=self.RUN_ID)
        before = _processed_tree(self.root)
        self.run_cell(mode="resume", run_id=self.RUN_ID, dry_run=True)
        self.assertEqual(_processed_tree(self.root), before)
        self.assertEqual(rp.read_resume_log(self.root, self.RUN_ID), [])


class RunnerFailureReconciliationTest(SingleCellRunnerTestCase):
    """An aborted cell must never be left claiming it is still `running`."""

    RUN_ID = "full-failure-fixture"

    def _run_until(self, side_effect: BaseException) -> None:
        with mock.patch.object(eu, "run_completion_jobs", side_effect=side_effect):
            self.run_cell(mode="full", run_id=self.RUN_ID)

    def test_an_exception_leaves_the_registry_failed(self) -> None:
        with self.assertRaises(RuntimeError):
            self._run_until(RuntimeError("provider exploded"))

        row = self.registry_row()
        self.assertEqual(row["status"], "failed")
        self.assertTrue(row["finished_at_utc"])
        self.assertFalse(self.lease_path(self.RUN_ID).exists())

    def test_a_keyboard_interrupt_leaves_the_registry_interrupted(self) -> None:
        with self.assertRaises(KeyboardInterrupt):
            self._run_until(KeyboardInterrupt())

        row = self.registry_row()
        self.assertEqual(row["status"], "interrupted")
        self.assertTrue(row["finished_at_utc"])
        self.assertFalse(self.lease_path(self.RUN_ID).exists())

    def test_a_reconciled_run_can_still_be_resumed(self) -> None:
        with self.assertRaises(RuntimeError):
            self._run_until(RuntimeError("provider exploded"))
        self.run_cell(mode="resume", run_id=self.RUN_ID)
        self.assertEqual(self.registry_row()["status"], "complete")


class RunnerCellLeaseTest(SingleCellRunnerTestCase):
    """A second runner must not duplicate paid calls for a claimed cell."""

    RUN_ID = "full-lease-fixture"

    def _plant_lease(self, *, pid: int, heartbeat: str) -> Path:
        path = self.lease_path(self.RUN_ID)
        eu.write_json(
            path,
            {
                "run_id": self.RUN_ID,
                "cell_key": "fake-fake-model-a-nice-must",
                "pid": pid,
                "hostname": socket.gethostname(),
                "claimed_at_utc": heartbeat,
                "heartbeat_utc": heartbeat,
            },
        )
        return path

    def test_a_live_lease_refuses_the_cell(self) -> None:
        # This process is unambiguously alive, so it stands in for the runner
        # that already owns the cell.
        planted = self._plant_lease(pid=os.getpid(), heartbeat=eu.utc_now_iso())

        with self.assertRaises(rl.CellLeaseError) as caught:
            self.run_cell(mode="full", run_id=self.RUN_ID)

        self.assertIn(str(os.getpid()), str(caught.exception))
        self.assertIn(self.RUN_ID, str(caught.exception))
        # No rows, and the other runner's lease is left exactly as it was.
        self.assertFalse(
            eu.model_outputs_raw_path(self.root, "nice", "must", smoke=True).exists()
        )
        self.assertEqual(
            json.loads(planted.read_text(encoding="utf-8"))["pid"], os.getpid()
        )

    def test_a_lease_held_by_a_dead_process_is_taken_over(self) -> None:
        self._plant_lease(pid=_dead_pid(), heartbeat=eu.utc_now_iso())

        with self.assertLogs(eu.logger, level="WARNING") as captured:
            self.run_cell(mode="full", run_id=self.RUN_ID)

        self.assertTrue(
            any("taking over stale lease" in line for line in captured.output),
            captured.output,
        )
        self.assertEqual(self.registry_row()["status"], "complete")
        # Released once the cell reached its terminal state.
        self.assertFalse(self.lease_path(self.RUN_ID).exists())

    def test_a_successful_cell_leaves_no_lease_behind(self) -> None:
        self.run_cell(mode="full", run_id=self.RUN_ID)
        self.assertEqual(self.registry_row()["status"], "complete")
        self.assertFalse(self.lease_path(self.RUN_ID).exists())


class CellLeaseStalenessTest(unittest.TestCase):
    """The staleness rule itself: a live pid is not enough, the beat must be recent."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)

    def _lease(self, **overrides: object) -> rl.CellLease:
        return rl.CellLease.for_cell(
            self.root,
            run_id="full-1",
            profile_id="fake",
            model="m",
            dataset_id="nice",
            variant="must",
            **overrides,
        )

    def test_an_expired_heartbeat_is_stale_even_for_a_live_pid(self) -> None:
        lease = self._lease()
        eu.write_json(
            lease.path,
            {
                "run_id": "full-1",
                "cell_key": lease.cell_key,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "claimed_at_utc": "2000-01-01T00:00:00Z",
                "heartbeat_utc": "2000-01-01T00:00:00Z",
            },
        )
        with self.assertLogs(eu.logger, level="WARNING"):
            lease.claim()
        self.assertTrue(lease.held)
        lease.release()
        self.assertFalse(lease.path.exists())

    def test_a_fresh_heartbeat_from_a_live_pid_blocks_a_second_claim(self) -> None:
        first = self._lease()
        first.claim()
        second = self._lease()
        with self.assertRaises(rl.CellLeaseError):
            second.claim()
        first.release()
        # Once released, the cell is free again.
        second.claim()
        self.assertTrue(second.held)
        second.release()

    def test_a_heartbeat_keeps_a_long_running_claim_alive(self) -> None:
        lease = self._lease(stale_after_s=0.0)
        lease.claim()
        # With a zero TTL every heartbeat is already expired, so the lease is
        # takeable -- the point here is that heartbeat rewrites the record
        # rather than dropping the claim.
        lease.heartbeat()
        self.assertTrue(lease.held)
        self.assertEqual(
            json.loads(lease.path.read_text(encoding="utf-8"))["pid"], os.getpid()
        )
        lease.release()
        self.assertFalse(lease.path.exists())


class Task3RunnerLifecycleTest(unittest.TestCase):
    """The Task 3 runner reaches the same lifecycle through its own planning.

    Task 3 has a separate registry file, item population, and provenance notes,
    so its wiring into the shared lifecycle is exercised here rather than only
    through the Task 1/2 runner.
    """

    RUN_ID = "task3-lifecycle-fixture"
    SOURCE_RUN_ID = "full-source"

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)
        (self.root / "prompts").mkdir(parents=True)
        (self.root / "data/processed").mkdir(parents=True)
        (self.root / "prompts/modality_verification.txt").write_text(
            Path("prompts/modality_verification.txt").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        benchmark = eu.build_benchmark_items(SEEDS)
        eu.write_csv_rows(self.root / "data/processed/benchmark_items.csv", benchmark)
        for item in benchmark:
            eu.append_jsonl(
                self.root / "data/processed/model_outputs_raw.jsonl",
                eu.build_raw_record(
                    run_id=self.SOURCE_RUN_ID,
                    model="fake-model-a",
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
                    run_group_id="matrix-group",
                ),
            )
        self.config = eu.load_run_config(
            _run_config(
                self.root,
                datasets=["nice"],
                variants=["must"],
                models=["fake-model-a"],
            )
        )

    def run_cell(self, **overrides: object) -> None:
        args = _runner_args(
            all_models=False,
            model="fake-model-a",
            source_run_id=self.SOURCE_RUN_ID,
            audit_mode="blind",
            allow_partial_source=True,
            allow_source_profile_mismatch=False,
            **overrides,
        )
        with (
            mock.patch.object(eu, "project_root", return_value=self.root),
            redirect_stdout(io.StringIO()),
        ):
            task3_runner.run_from_config(self.config, args)

    def registry_row(self) -> dict[str, str]:
        rows = eu.read_csv_rows(
            eu.task3_registry_path(self.root, "nice", "must", smoke=True)
        )
        self.assertEqual(len(rows), 1, rows)
        return rows[0]

    def test_resume_with_an_unknown_run_id_fails_before_writing(self) -> None:
        self.run_cell(mode="full", run_id=self.RUN_ID)
        before = _processed_tree(self.root)
        with self.assertRaises(rl.ResumeError) as caught:
            self.run_cell(mode="resume", run_id="task3-lifecycle-fixtrue")
        self.assertIn("task3-lifecycle-fixtrue", str(caught.exception))
        # Not even the audit-items CSV is rewritten under a bad resume.
        self.assertEqual(_processed_tree(self.root), before)

    def test_resume_preserves_the_start_time_and_audit_provenance(self) -> None:
        self.run_cell(mode="full", run_id=self.RUN_ID)
        first = self.registry_row()
        self.assertEqual(first["tasks"], "task3")
        self.assertIn("audit_mode=blind", first["notes"])
        self.assertIn(f"source_run_id={self.SOURCE_RUN_ID}", first["notes"])

        later = "2099-01-01T00:00:00Z"
        with mock.patch.object(eu, "utc_now_iso", return_value=later):
            self.run_cell(mode="resume", run_id=self.RUN_ID)

        resumed = self.registry_row()
        self.assertEqual(resumed["started_at_utc"], first["started_at_utc"])
        self.assertEqual(resumed["notes"], first["notes"])
        self.assertEqual(resumed["finished_at_utc"], later)
        self.assertEqual(rp.read_resume_log(self.root, self.RUN_ID), [later])

    def test_an_exception_leaves_the_task3_registry_failed(self) -> None:
        with (
            mock.patch.object(
                eu, "run_completion_jobs", side_effect=RuntimeError("provider exploded")
            ),
            self.assertRaises(RuntimeError),
        ):
            self.run_cell(mode="full", run_id=self.RUN_ID)
        self.assertEqual(self.registry_row()["status"], "failed")


if __name__ == "__main__":
    unittest.main()
