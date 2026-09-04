"""The one-command driver: what it runs, in what order, and what it refuses.

The driver spends two days of provider budget, so the parts worth pinning are
the ones that decide *what* is sent and *what is done with the result*: that a
failed cell is retried as a resume rather than re-requested in full, that the
run id it records is the one this stage wrote and not another stage's run of the
same cell, that a dry run changes nothing, and that the analysis stage refuses
to describe an incomplete cohort.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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


class RunCellTest(unittest.TestCase):
    def setUp(self):
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


class RunnerOverridesTest(unittest.TestCase):
    def test_a_real_run_is_full_and_a_fake_run_is_smoke(self):
        real = rerun_all.Runner(root=Path())
        fake = rerun_all.Runner(root=Path(), fake=True, smoke_items=4)

        self.assertEqual(real.run_overrides(), ["mode=full"])
        self.assertEqual(real.mode, "full")
        self.assertEqual(
            fake.run_overrides(),
            [
                "mode=smoke",
                "smoke_items=4",
                "fake_completion=true",
                "embedding=tfidf_proxy",
            ],
        )
        self.assertEqual(fake.mode, "smoke")


class PreflightTest(unittest.TestCase):
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

        problems = rerun_all.embedding_backend_problems(self.root)

        # This environment has no mlx-embeddings; the point is that the check
        # happens up front rather than hours into the analysis stage.
        try:
            import mlx_embeddings  # noqa: F401
        except ImportError:
            self.assertEqual(len(problems), 1)
            self.assertIn("mlx-embeddings", problems[0])
        else:  # pragma: no cover - depends on the local environment
            self.assertEqual(problems, [])
        self.assertEqual(rerun_all.embedding_backend_problems(self.root, fake=True), [])


class AnalysisGateTest(unittest.TestCase):
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
            state.record("cohort:zai:glm-5.1:nice:must", "complete", run_id="full-1")
            rerun = {
                "run_group_id": "provider-matrix-v2-2026-05",
                "datasets": ["nice"],
                "variants": ["must"],
                "analysis": {"bootstrap_samples": 10},
                "ablations": {"batching": {"dataset": "nice", "variant": "must"}},
            }

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
            self.assertIn("--local-model", export)
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
        # Every provider run of the rerun is batched the paper way.
        for profile in config["profiles"]:
            self.assertEqual(int(profile["batch_size"]), 16, profile["profile_id"])


if __name__ == "__main__":
    unittest.main()
