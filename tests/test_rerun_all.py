"""The one-command driver: what it runs, in what order, and what it refuses.

The driver spends two days of provider budget, so the parts worth pinning are
the ones that decide *what* is sent and *what is done with the result*: that a
failed cell is retried as a resume rather than re-requested in full, that the
run id it records is the one this stage wrote and not another stage's run of the
same cell, that a dry run changes nothing, and that the analysis stage refuses
to describe an incomplete cohort.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import eval_utils as eu, rerun_all


class RecordingRunner(rerun_all.Runner):
    """A `Runner` that records commands instead of running them."""

    def __init__(self, root: Path, exit_codes: dict[str, list[int]] | None = None):
        super().__init__(root=root)
        self.commands: list[tuple[str, list[str]]] = []
        self.exit_codes = exit_codes or {}
        self.registry_rows: list[dict[str, object]] = []

    def run(self, argv: list[str], *, label: str) -> int:
        self.commands.append((label, argv))
        codes = self.exit_codes.get(label)
        return codes.pop(0) if codes else 0


def _registry_row(**overrides):
    row = {
        "run_id": "full-1",
        "profile_id": "zai",
        "model": "glm-5.1",
        "dataset_id": "nice",
        "benchmark_variant": "must",
        "tasks": "task1,task2",
        "status": "complete",
        "started_at_utc": "2026-09-05T00:00:00Z",
    }
    row.update(overrides)
    return row


class StateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "outputs" / "rerun_state.json"

    def test_state_round_trips_and_reports_failures(self):
        state = rerun_all.RerunState.load(self.path)
        state.record("cohort:a", "complete", run_id="full-1")
        state.record("cohort:b", "failed", run_id="full-2")

        reloaded = rerun_all.RerunState.load(self.path)

        self.assertTrue(reloaded.done("cohort:a"))
        self.assertEqual(reloaded.run_id("cohort:a"), "full-1")
        self.assertEqual(reloaded.failures(), ["cohort:b"])

    def test_a_dry_run_never_writes_state(self):
        state = rerun_all.RerunState.load(self.path, dry_run=True)
        state.record("cohort:a", "complete", run_id="full-1")

        # In memory only: a printed plan must not make the next real run skip
        # the cells it only pretended to execute.
        self.assertTrue(state.done("cohort:a"))
        self.assertFalse(self.path.exists())

    def test_two_drivers_sharing_one_state_file_keep_each_others_cells(self):
        """One driver per endpoint: a record merges its key, never the snapshot."""
        hosted = rerun_all.RerunState.load(self.path)
        local = rerun_all.RerunState.load(self.path)
        hosted.record("cohort:zai:glm-5.3:nice:must", "complete", run_id="full-1")
        local.record("cohort:local_llama_cpp:q:nice:must", "failed", run_id="full-2")
        hosted.record("task3:zai:glm-5.3:nice:must", "complete", run_id="task3-1")

        reloaded = rerun_all.RerunState.load(self.path)

        self.assertEqual(reloaded.run_id("cohort:zai:glm-5.3:nice:must"), "full-1")
        self.assertEqual(
            reloaded.run_id("cohort:local_llama_cpp:q:nice:must"), "full-2"
        )
        self.assertEqual(reloaded.run_id("task3:zai:glm-5.3:nice:must"), "task3-1")
        # Each driver reports only its own endpoint's failures, read from disk.
        self.assertEqual(hosted.failures(["zai"]), [])
        self.assertEqual(
            hosted.failures(["local_llama_cpp"]), ["cohort:local_llama_cpp:q:nice:must"]
        )
        self.assertEqual(reloaded.failures(), ["cohort:local_llama_cpp:q:nice:must"])


class RunCellTest(unittest.TestCase):
    def setUp(self):
        # The synthetic run begins at the fixture timestamp, independent of wall time.
        clock = patch.object(eu, "utc_now_iso", return_value="2026-09-05T00:00:00Z")
        clock.start()
        self.addCleanup(clock.stop)
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.registry_path = eu.run_registry_path(self.root, "nice", "must")
        self.state = rerun_all.RerunState.load(self.root / "state.json")

    def _write_registry(self, rows):
        eu.write_csv_rows(self.registry_path, rows)

    def _run(self, runner, overrides=None):
        return rerun_all.run_cell_with_retry(
            runner,
            self.state,
            key="cohort:zai:glm-5.1:nice:must",
            overrides=overrides or ["profile=zai", "model=glm-5.1", "mode=full"],
            registry_path=self.registry_path,
            profile_id="zai",
            model="glm-5.1",
            dataset_id="nice",
            variant="must",
            tasks="task1,task2",
        )

    def test_a_completed_cell_is_skipped_on_the_next_invocation(self):
        self._write_registry([_registry_row()])
        runner = RecordingRunner(self.root)

        self.assertTrue(self._run(runner))
        self.assertTrue(self._run(runner))

        self.assertEqual(len(runner.commands), 1)
        self.assertEqual(self.state.run_id("cohort:zai:glm-5.1:nice:must"), "full-1")

    def test_a_failed_cell_is_retried_as_a_resume_of_the_same_run(self):
        self._write_registry([_registry_row(status="failed")])
        runner = RecordingRunner(
            self.root, exit_codes={"cohort:zai:glm-5.1:nice:must": [1]}
        )

        self.assertTrue(self._run(runner))

        self.assertEqual(len(runner.commands), 2)
        retry = runner.commands[1][1]
        self.assertIn("mode=resume", retry)
        self.assertIn("run_id=full-1", retry)
        # Never a second full run: that would re-request everything the first
        # attempt already paid for.
        self.assertNotIn("mode=full", retry)

    def test_a_cell_that_fails_twice_is_recorded_and_the_run_continues(self):
        self._write_registry([_registry_row(status="failed")])
        runner = RecordingRunner(
            self.root,
            exit_codes={
                "cohort:zai:glm-5.1:nice:must": [1],
                "cohort:zai:glm-5.1:nice:must (resume)": [1],
            },
        )

        self.assertFalse(self._run(runner))

        self.assertEqual(self.state.failures(), ["cohort:zai:glm-5.1:nice:must"])

    def test_the_recorded_run_id_is_this_stage_s_run_not_another_stage_s(self):
        # An ablation arm shares (profile, model, dataset, variant) with the
        # cohort cell; only the task set and the launch time tell them apart.
        self._write_registry(
            [
                _registry_row(
                    run_id="full-cohort", started_at_utc="2026-09-05T01:00:00Z"
                ),
                _registry_row(
                    run_id="full-ablation",
                    tasks="task2",
                    started_at_utc="2026-09-05T02:00:00Z",
                ),
            ]
        )

        run_id = rerun_all.latest_run_id(
            self.root,
            profile_id="zai",
            model="glm-5.1",
            dataset_id="nice",
            variant="must",
            tasks="task1,task2",
            since_utc="2026-09-05T00:00:00Z",
            registry_path=self.registry_path,
        )

        self.assertEqual(run_id, "full-cohort")

    def test_an_arm_does_not_claim_a_sibling_arm_s_run(self):
        # The batching arms share profile, model, dataset, variant and task
        # set; only the batching columns tell them apart in the registry.
        self._write_registry(
            [
                _registry_row(
                    run_id="full-grouped",
                    tasks="task2",
                    batch_size="16",
                    batch_order="grouped",
                    item_context="bare",
                    started_at_utc="2026-09-05T02:00:00Z",
                )
            ]
        )
        common = {
            "profile_id": "zai",
            "model": "glm-5.1",
            "dataset_id": "nice",
            "variant": "must",
            "tasks": "task2",
            "since_utc": "2026-09-05T00:00:00Z",
            "registry_path": self.registry_path,
        }

        single = rerun_all.latest_run_id(
            self.root,
            **common,
            arm=rerun_all.arm_columns(
                ["profile.batch_size=1", "profile.batch_order=grouped"]
            ),
        )
        grouped = rerun_all.latest_run_id(
            self.root,
            **common,
            arm=rerun_all.arm_columns(
                ["profile.batch_size=16", "profile.batch_order=grouped"]
            ),
        )

        self.assertEqual(single, "")
        self.assertEqual(grouped, "full-grouped")

    def test_runs_started_before_this_invocation_are_not_claimed(self):
        self._write_registry(
            [_registry_row(run_id="full-old", started_at_utc="2026-09-01T00:00:00Z")]
        )

        run_id = rerun_all.latest_run_id(
            self.root,
            profile_id="zai",
            model="glm-5.1",
            dataset_id="nice",
            variant="must",
            tasks="task1,task2",
            since_utc="2026-09-05T00:00:00Z",
            registry_path=self.registry_path,
        )

        self.assertEqual(run_id, "")

    def test_failed_run_resumes_after_restarting_the_driver(self):
        self._write_registry(
            [_registry_row(run_id="full-prior", started_at_utc="2026-09-01T00:00:00Z")]
        )
        self.state.record("cohort:zai:glm-5.1:nice:must", "failed", run_id="full-prior")
        self.state = rerun_all.RerunState.load(self.state.path)
        runner = RecordingRunner(self.root)
        self.assertTrue(self._run(runner))
        self.assertIn("mode=resume", runner.commands[0][1])
        self.assertIn("run_id=full-prior", runner.commands[0][1])
        self.assertNotIn("mode=full", runner.commands[0][1])

    def test_success_without_a_run_record_is_not_complete(self):
        self.assertFalse(self._run(RecordingRunner(self.root)))
        self.assertEqual(self.state.failures(), ["cohort:zai:glm-5.1:nice:must"])


class ProfileSelectionTest(unittest.TestCase):
    def _rerun(self):
        return {"cohort_profiles": ["zai"], "local_profiles": ["local_llama_cpp"]}

    def test_no_profile_flag_drives_every_profile_in_config_order(self):
        self.assertEqual(
            rerun_all.select_profiles([], self._rerun()), ["zai", "local_llama_cpp"]
        )

    def test_profile_flag_narrows_to_the_named_endpoints(self):
        self.assertEqual(
            rerun_all.select_profiles(["local_llama_cpp"], self._rerun()),
            ["local_llama_cpp"],
        )

    def test_an_unknown_profile_is_an_operator_error(self):
        with self.assertRaises(rerun_all.StageError) as caught:
            rerun_all.select_profiles(["kit_toolbox"], self._rerun())
        self.assertIn("kit_toolbox", str(caught.exception))


class RunnerOverridesTest(unittest.TestCase):
    def test_a_real_run_is_full_and_a_fake_run_is_smoke(self):
        real = rerun_all.Runner(root=Path())
        fake = rerun_all.Runner(root=Path(), fake=True, smoke_items=4)

        self.assertIn("mode=full", real.run_overrides())
        self.assertIn("embedding=qwen3_06b", real.run_overrides())
        self.assertIn("logging.write_request_transcripts=true", real.run_overrides())
        self.assertEqual(real.mode, "full")
        self.assertEqual(
            fake.run_overrides(),
            [
                "mode=smoke",
                "smoke_items=4",
                "fake_completion=true",
                "embedding=tfidf_proxy",
                "logging.write_progress_csv=true",
                "logging.write_event_jsonl=true",
                "logging.write_request_transcripts=true",
            ],
        )
        self.assertEqual(fake.mode, "smoke")


class PreflightTest(unittest.TestCase):
    def test_frozen_primary_protocol_rejects_profile_drift(self):
        with TemporaryDirectory() as tmp:
            config = {
                "datasets": [],
                "variants": [],
                "run_group_id": "g",
                "primary_protocol": {"batch_size": 16, "batch_order": "grouped"},
            }
            profile = {"profile_id": "p", "batch_size": 1, "batch_order": "shuffled"}
            with self.assertRaisesRegex(
                rerun_all.StageError, "primary protocol requires batch_size"
            ):
                rerun_all.stage_preflight(
                    Path(tmp), config, [profile], [("p", "m")], [], fake=True
                )

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "conf").mkdir(parents=True)
        (self.root / "conf/config.yaml").write_text(
            "acse_embedding_backend: tfidf_char_wb_3_5\n", encoding="utf-8"
        )

    def _rerun(self):
        return {
            "run_group_id": "provider-matrix-v2-2026-05",
            "datasets": ["nice"],
            "variants": ["must"],
            "cohort_profiles": ["zai"],
        }

    def test_every_problem_is_reported_at_once(self):
        profile = {
            "profile_id": "zai",
            "api_key_env": "RE_UQ_TEST_KEY_THAT_IS_NOT_SET",
            "models": ["glm-5.1"],
        }

        with self.assertRaises(rerun_all.StageError) as caught:
            rerun_all.stage_preflight(
                self.root, self._rerun(), [profile], [("zai", "glm-5.1")], []
            )
        message = str(caught.exception)

        # The missing key and the missing benchmark, not just the first one.
        self.assertIn("RE_UQ_TEST_KEY_THAT_IS_NOT_SET is not set", message)
        self.assertIn("benchmark missing", message)

    def test_a_manual_server_profile_may_not_carry_several_models(self):
        profile = {
            "profile_id": "local_llama_cpp",
            "api_key_env": "",
            "models": ["a", "b"],
            "requires_manual_server": True,
        }

        with self.assertRaises(rerun_all.StageError) as caught:
            rerun_all.stage_preflight(
                self.root, self._rerun(), [profile], [("local_llama_cpp", "a")], []
            )

        self.assertIn("serves one model at a time", str(caught.exception))

    def test_the_mlx_backend_is_checked_before_any_run(self):
        (self.root / "conf/config.yaml").write_text(
            "acse_embedding_backend: mlx\n", encoding="utf-8"
        )

        from subprocess import CompletedProcess

        with patch.object(
            rerun_all.subprocess,
            "run",
            return_value=CompletedProcess([], 1, "", "No module named mlx_embeddings"),
        ):
            problems = rerun_all.embedding_backend_problems(self.root)
        self.assertEqual(len(problems), 1)
        self.assertIn("mlx_embeddings", problems[0])
        with patch.object(
            rerun_all.subprocess, "run", return_value=CompletedProcess([], 0, "", "")
        ):
            self.assertEqual(rerun_all.embedding_backend_problems(self.root), [])
        self.assertEqual(rerun_all.embedding_backend_problems(self.root, fake=True), [])

    def test_the_mlx_check_is_skipped_when_the_analysis_stage_is_not_requested(self):
        """Generation runs on any host; only the analysis stage needs Metal."""
        (self.root / "conf/config.yaml").write_text(
            "acse_embedding_backend: mlx\n", encoding="utf-8"
        )
        (self.root / "data/processed").mkdir(parents=True)
        (self.root / "data/processed/benchmark_items.csv").write_text(
            "item_id\n", encoding="utf-8"
        )
        profile = {"profile_id": "zai", "api_key_env": "", "models": ["glm-5.3"]}
        from subprocess import CompletedProcess

        with patch.object(
            rerun_all.subprocess,
            "run",
            return_value=CompletedProcess([], 1, "", "No module named mlx_embeddings"),
        ):
            rerun_all.stage_preflight(
                self.root,
                self._rerun(),
                [profile],
                [("zai", "glm-5.3")],
                [],
                analysis=False,
            )
            with self.assertRaises(rerun_all.StageError) as caught:
                rerun_all.stage_preflight(
                    self.root, self._rerun(), [profile], [("zai", "glm-5.3")], []
                )
        self.assertIn("MLX embedding backend is unavailable", str(caught.exception))


class AnalysisGateTest(unittest.TestCase):
    def test_refresh_recomputes_completed_analysis_in_dependency_order(self):
        with TemporaryDirectory() as tmpdir:
            runner = RecordingRunner(Path(tmpdir))
            state = rerun_all.RerunState.load(Path(tmpdir) / "state.json")
            source_key = "cohort:zai:glm:nice:must"
            state.record(source_key, "complete", run_id="full-1")
            state.record("task3:zai:glm:nice:must", "complete", run_id="audit-1")
            config = {
                "run_group_id": "g",
                "datasets": ["nice"],
                "variants": ["must"],
                "analysis": {},
                "ablations": {},
            }
            rerun_all.stage_analysis(runner, state, config, [("zai", "glm")], [])
            first = list(runner.commands)
            runner.commands.clear()
            rerun_all.stage_analysis(runner, state, config, [("zai", "glm")], [])
            self.assertEqual(runner.commands, [])
            rerun_all.stage_analysis(
                runner, state, config, [("zai", "glm")], [], refresh=True
            )
            self.assertEqual(runner.commands, first)
            labels = [key for key, _ in first]
            self.assertLess(
                labels.index("analysis:" + source_key), labels.index("analysis:acse")
            )
            self.assertLess(
                labels.index("analysis:acse"), labels.index("analysis:embedding-probe")
            )
            self.assertEqual(state.run_id(source_key), "full-1")
            self.assertTrue(all(key.startswith("analysis:") for key in labels))

    def test_refresh_rejects_generation_before_loading_configuration(self):
        with patch.object(rerun_all, "load_rerun_config") as load:
            self.assertEqual(rerun_all.main(["--refresh-analysis"]), 2)
            self.assertEqual(
                rerun_all.main(["--refresh-analysis", "--only", "cohort"]), 2
            )
            load.assert_not_called()

    def test_failed_refresh_leaves_downstream_analysis_pending(self):
        with TemporaryDirectory() as tmp:
            state = rerun_all.RerunState.load(Path(tmp) / "state.json")
            source = "cohort:zai:glm:nice:must"
            state.record(source, "complete", run_id="full-1")
            state.record("task3:zai:glm:nice:must", "complete", run_id="audit-1")
            config = {
                "run_group_id": "g",
                "datasets": ["nice"],
                "variants": ["must"],
                "analysis": {},
                "ablations": {},
            }
            runner = RecordingRunner(Path(tmp))
            rerun_all.stage_analysis(runner, state, config, [("zai", "glm")], [])
            runner.exit_codes = {"analysis:acse": [1]}
            with self.assertRaises(rerun_all.StageError):
                rerun_all.stage_analysis(
                    runner, state, config, [("zai", "glm")], [], refresh=True
                )
            self.assertEqual(state.status("analysis:acse"), "failed")
            self.assertEqual(state.status("analysis:paper-tables"), "pending")
            self.assertEqual(state.status("analysis:embedding-probe"), "pending")
            self.assertTrue(state.done(source))

    def test_refresh_uses_recorded_settings_without_reading_live_profiles(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "state.json"
            stored = {
                "rerun": {
                    "run_group_id": "g",
                    "cohort_profiles": ["old"],
                    "local_profiles": [],
                    "datasets": ["nice"],
                    "variants": ["must"],
                    "ablations": {},
                },
                "run_config": {
                    "profiles": [{"profile_id": "old", "models": ["old-model"]}]
                },
                "fake_completion": False,
                "smoke_items": 8,
            }
            eu.write_json(path, {"configuration": stored, "cells": {}})
            before = path.read_bytes()
            with (
                patch.object(eu, "project_root", return_value=root),
                patch.object(rerun_all, "load_profile") as live,
                patch.object(rerun_all, "stage_analysis") as analysis,
            ):
                self.assertEqual(
                    rerun_all.main(
                        [
                            "--only",
                            "analysis",
                            "--refresh-analysis",
                            "--state",
                            str(path),
                            "--dry-run",
                        ]
                    ),
                    0,
                )
            live.assert_not_called()
            self.assertEqual(analysis.call_args.args[3], [("old", "old-model")])
            self.assertTrue(analysis.call_args.kwargs["refresh"])
            self.assertEqual(path.read_bytes(), before)

    def test_the_analysis_stage_refuses_an_incomplete_cohort(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runner = RecordingRunner(root)
            state = rerun_all.RerunState.load(root / "state.json")
            state.record("cohort:zai:glm-5.1:nice:must", "failed")
            rerun = {
                "run_group_id": "provider-matrix-v2-2026-05",
                "datasets": ["nice"],
                "variants": ["must"],
                "analysis": {},
                "ablations": {},
            }

            with self.assertRaises(rerun_all.StageError) as caught:
                rerun_all.stage_analysis(runner, state, rerun, [("zai", "glm-5.1")], [])

            self.assertIn("cohort is incomplete", str(caught.exception))
            self.assertEqual(runner.commands, [])

    def test_the_export_selects_the_rerun_group_and_the_configured_cells(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runner = RecordingRunner(root)
            state = rerun_all.RerunState.load(root / "state.json")
            state.configuration = {
                "run_config": {
                    "profiles": [{"batch_size": 1, "batch_order": "grouped"}]
                }
            }
            state.record("cohort:zai:glm-5.1:nice:must", "complete", run_id="full-1")
            state.record("task3:zai:glm-5.1:nice:must", "complete", run_id="task3-1")
            rerun = {
                "run_group_id": "provider-matrix-v2-2026-05",
                "datasets": ["nice"],
                "variants": ["must"],
                "analysis": {"bootstrap_samples": 10},
                "ablations": {"batching": {"dataset": "nice", "variant": "must"}},
            }

            state.record(
                "cohort:local:qwen/q:nice:must", "complete", run_id="full-local"
            )
            state.record(
                "task3:local:qwen/q:nice:must", "complete", run_id="task3-local"
            )
            rerun_all.stage_analysis(
                runner, state, rerun, [("zai", "glm-5.1")], [("local", "qwen/q")]
            )

            export = next(
                argv
                for label, argv in runner.commands
                if label == "analysis:paper-tables"
            )
            self.assertIn("--run-group-id", export)
            self.assertEqual(
                export[export.index("--run-group-id") + 1],
                "provider-matrix-v2-2026-05",
            )
            self.assertEqual(export[export.index("--cell") + 1], "nice/must")
            self.assertEqual(export[export.index("--expected-batch-size") + 1], "1")
            self.assertIn("--local-model", export)
            self.assertIn("--run-id", export)
            self.assertIn("full-local", export)
            self.assertIn("task3-local", export)
            probe = next(
                argv
                for key, argv in runner.commands
                if key == "analysis:embedding-probe"
            )
            self.assertIn("outputs/rerun/acse_selected_manifest.csv", probe)
            self.assertIn("hgb", probe)
            self.assertEqual(probe[probe.index("--bootstrap-samples") + 1], "10")
            labels = [label for label, _ in runner.commands]
            self.assertLess(
                labels.index("analysis:embedding-probe"),
                labels.index("analysis:figure2"),
            )
            self.assertLess(
                labels.index("analysis:embedding-probe"),
                labels.index("analysis:numbers"),
            )
            # The aggregator must not be asked to regenerate snapshots: that
            # path cannot retarget the run group and would undo the export.
            aggregate = next(
                argv
                for label, argv in runner.commands
                if label == "analysis:headline-metrics"
            )
            self.assertNotIn("--regenerate-snapshots", aggregate)

    def test_a_failed_analysis_step_stops_the_stage(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runner = RecordingRunner(root, exit_codes={"analysis:acse": [1]})
            state = rerun_all.RerunState.load(root / "state.json")
            state.record("cohort:zai:glm-5.1:nice:must", "complete", run_id="full-1")
            state.record("task3:zai:glm-5.1:nice:must", "complete", run_id="task3-1")
            rerun = {
                "run_group_id": "g",
                "datasets": ["nice"],
                "variants": ["must"],
                "analysis": {},
                "ablations": {},
            }

            with self.assertRaises(rerun_all.StageError) as caught:
                rerun_all.stage_analysis(runner, state, rerun, [("zai", "glm-5.1")], [])

            self.assertIn("analysis:acse failed", str(caught.exception))
            # Nothing after it ran.
            self.assertNotIn(
                "analysis:paper-tables", [label for label, _ in runner.commands]
            )


class GeneratedRunConfigTest(unittest.TestCase):
    def test_the_generated_config_is_written_from_conf_not_from_run_configs(self):
        root = eu.project_root()
        rerun = rerun_all.load_rerun_config(root, rerun_all.DEFAULT_RERUN_CONFIG)
        cohort, local = rerun_all.cohort_models(root, rerun)
        profiles = [
            rerun_all.load_profile(root, profile_id)
            for profile_id in list(rerun["cohort_profiles"])
            + list(rerun.get("local_profiles", []))
        ]

        config = rerun_all.generated_run_config(root, rerun, profiles)

        self.assertEqual(config["run_group_id"], rerun["run_group_id"])
        self.assertEqual(config["benchmark_variants"], list(rerun["variants"]))
        self.assertEqual(
            {profile["profile_id"] for profile in config["profiles"]},
            set(rerun["cohort_profiles"]) | set(rerun.get("local_profiles", [])),
        )
        # The cohort is the profiles' model lists, not a hardcoded one.
        self.assertEqual(
            {model for _, model in cohort + local},
            {model for profile in config["profiles"] for model in profile["models"]},
        )
        # Single-item requests are the shared default, including audit sources.
        for profile in config["profiles"]:
            self.assertEqual(int(profile["batch_size"]), 1, profile["profile_id"])

    def test_final_batching_sweep_keeps_size_and_composition_distinct(self):
        rerun = rerun_all.load_rerun_config(
            eu.project_root(), Path("conf/rerun/final.yaml")
        )
        self.assertEqual(
            rerun_all.batching_arm_specs(rerun["ablations"]["batching"]),
            [
                ("single", 1, "grouped"),
                ("grouped_4", 4, "grouped"),
                ("shuffled_4", 4, "shuffled"),
                ("grouped", 16, "grouped"),
                ("shuffled", 16, "shuffled"),
            ],
        )
        self.assertEqual(rerun["primary_protocol"]["batch_size"], 1)
        self.assertEqual(
            rerun["primary_sampling"], {"deterministic": 1, "stochastic": 5}
        )
        self.assertFalse(
            rerun_all.primary_sampling_problems(
                rerun,
                {"deterministic": {"samples": 1}, "stochastic": {"samples": 5}},
            )
        )
        self.assertTrue(
            rerun_all.primary_sampling_problems(
                rerun,
                {"deterministic": {"samples": 1}, "stochastic": {"samples": 3}},
            )
        )
        self.assertEqual(rerun["ablations"]["context"]["batch_size"], 1)
        with self.assertRaisesRegex(rerun_all.StageError, "baseline"):
            rerun_all.batching_arm_specs({"batch_sizes": [4], "baseline_arm": "single"})

    def test_ablation_commands_honor_configured_sizes_including_context(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = RecordingRunner(root)
            state = rerun_all.RerunState(root / "state.json", dry_run=True)
            config = {
                "run_group_id": "final-test",
                "ablations": {
                    "batching": {
                        "models": [{"profile": "p", "model": "m"}],
                        "batch_sizes": [1, 4, 16],
                        "baseline_arm": "single",
                    },
                    "context": {
                        "models": [{"profile": "p", "model": "m"}],
                        "batch_size": 1,
                    },
                },
            }
            with patch.object(rerun_all, "run_cell_with_retry") as run:
                rerun_all.stage_ablations(runner, state, config, root / "config.json")
            calls = [call.kwargs for call in run.call_args_list]
            self.assertEqual(len(calls), 7)
            self.assertEqual(len({c["key"] for c in calls}), 7)
            for call in calls:
                if call["key"].startswith("context:"):
                    self.assertIn("profile.batch_size=1", call["overrides"])


class RerunSafetyTest(unittest.TestCase):
    def test_changed_configuration_does_not_reuse_complete_cells(self):
        with TemporaryDirectory() as tmp:
            state = rerun_all.RerunState(Path(tmp) / "state.json")
            state.bind_configuration({"model": "a", "seed": 1})
            state.record("cohort:a", "complete", run_id="one")
            state = rerun_all.RerunState.load(state.path)
            state.bind_configuration({"model": "a", "seed": 1})
            with self.assertRaises(rerun_all.StageError):
                state.bind_configuration({"model": "a", "seed": 2})
            self.assertEqual(state.run_id("cohort:a"), "one")

    def test_child_stdout_and_stderr_are_saved(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".venv/bin").mkdir(parents=True)
            (root / ".venv/bin/python").symlink_to(sys.executable)
            runner = rerun_all.Runner(root)
            code = runner.run(
                [
                    "-c",
                    "import sys; print('request saved'); print('response saved', file=sys.stderr)",
                ],
                label="test",
            )
            self.assertEqual(code, 0)
            text = (root / "outputs/rerun/logs/test.log").read_text()
            self.assertIn("request saved", text)
            self.assertIn("response saved", text)
            self.assertIn("exit=0", text)

    def test_one_model_finishes_all_provider_stages_before_the_next(self):
        rerun = {
            "run_group_id": "order-test",
            "datasets": ["nice"],
            "variants": ["must"],
            "cohort_profiles": ["zai"],
            "local_profiles": ["local"],
            "ablations": {"batching": {"profiles": ["zai", "local"]}},
        }
        profiles = {
            "zai": {"profile_id": "zai", "models": ["hosted"]},
            "local": {"profile_id": "local", "models": ["local-a", "local-b"]},
        }
        observed = []

        def cohort(runner, state, rerun, models):
            observed.append((models[0][1], "cohort"))

        def audit(runner, state, rerun, config):
            observed.append((config["profiles"][0]["models"][0], "audit"))

        def ablation(runner, state, rerun, path):
            for entry in rerun["ablations"]["batching"]["models"]:
                observed.append((entry["model"], "ablation"))

        with (
            TemporaryDirectory() as tmp,
            patch.object(eu, "project_root", return_value=Path(tmp)),
            patch.object(rerun_all, "load_rerun_config", return_value=rerun),
            patch.object(
                rerun_all, "load_profile", side_effect=lambda root, key: profiles[key]
            ),
            patch.object(
                rerun_all,
                "generated_run_config",
                return_value={"profiles": list(profiles.values())},
            ),
            patch.object(rerun_all, "stage_preflight"),
            patch.object(rerun_all, "stage_cohort", side_effect=cohort),
            patch.object(rerun_all, "stage_task3", side_effect=audit),
            patch.object(rerun_all, "stage_ablations", side_effect=ablation),
            patch.object(rerun_all, "stage_analysis"),
        ):
            self.assertEqual(rerun_all.main(["--dry-run"]), 0)
        self.assertEqual(
            observed,
            [
                ("hosted", "cohort"),
                ("hosted", "audit"),
                ("hosted", "ablation"),
                ("local-a", "cohort"),
                ("local-a", "audit"),
                ("local-a", "ablation"),
                ("local-b", "cohort"),
                ("local-b", "audit"),
            ],
        )

    def test_the_generated_config_covers_the_ablation_datasets(self):
        # The weak probe on `nice` and the context arms on `pure` read the
        # generated config's selector lists, so they must include those
        # datasets even when the cohort only covers `mlm_tapt`.
        root = eu.project_root()
        rerun = rerun_all.load_rerun_config(root, root / "conf/rerun/ablations.yaml")
        profiles = [
            rerun_all.load_profile(root, profile_id)
            for profile_id in list(rerun["cohort_profiles"])
            + list(rerun.get("local_profiles", []))
        ]

        config = rerun_all.generated_run_config(root, rerun, profiles)

        self.assertEqual(config["datasets"], ["mlm_tapt", "pure", "nice"])
        self.assertEqual(config["benchmark_variants"], ["must"])


if __name__ == "__main__":
    unittest.main()
