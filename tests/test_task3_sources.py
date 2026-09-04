"""Which Task 2 run a Task 3 audit reads.

The resolution used to live in a bash heredoc that only `enqueue_task3_runs.sh`
could call, so nothing tested it and no other driver could reuse it. These
tests pin the rules it encodes: newest wins, only compatible complete Task 2
runs count, and a cell with no source is reported rather than skipped silently.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scripts import eval_utils as eu, task3_sources as ts

RUN_GROUP = "provider-matrix-v2-2026-05"


def _registry_row(**overrides):
    row = {
        "run_id": "full-new",
        "run_group_id": RUN_GROUP,
        "profile_id": "zai",
        "model": "glm-5.1",
        "status": "complete",
        "tasks": "task1,task2",
        "expected_records": 24 * 2 * 6,
        "observed_records": 24 * 2 * 6,
        "deterministic_item_coverage": 1.0,
        "stochastic_complete_item_rate": 1.0,
        "batch_size": 16,
        "batch_order": "grouped",
        "started_at_utc": "2026-02-01T00:00:00Z",
    }
    row.update(overrides)
    return row


class ResolveTask3SourcesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": f"S{index:04d}",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": f"export report set {index}",
                }
                for index in range(1, 7)
            ]
        )
        eu.write_csv_rows(
            eu.artifact_path(
                self.root / "data/processed/benchmark_items.csv", "nice", "must"
            ),
            self.benchmark,
        )

    def _config(self, **overrides):
        config = {
            "run_group_id": RUN_GROUP,
            "datasets": ["nice"],
            "benchmark_variants": ["must"],
            "tasks": ["task1", "task2"],
            "prompt_version": "v2-conf01",
            "batch_order": "grouped",
            "deterministic": {"temperature": 0.0, "top_p": 1.0, "samples": 1},
            "stochastic": {"temperature": 0.7, "top_p": 1.0, "samples": 5},
            "profiles": [
                {
                    "profile_id": "zai",
                    "provider_id": "zai",
                    "base_url": "https://example.invalid/v1",
                    "api_key_env": "ZAI_API_KEY",
                    "models": ["glm-5.1"],
                    "batch_size": 16,
                    "batch_order": "grouped",
                }
            ],
        }
        config.update(overrides)
        return eu.normalize_run_config(config)

    def _write_registry(self, rows):
        eu.write_csv_rows(eu.run_registry_path(self.root, "nice", "must"), rows)

    def test_newest_compatible_run_wins(self):
        self._write_registry(
            [
                _registry_row(run_id="full-old", started_at_utc="2026-01-01T00:00:00Z"),
                _registry_row(run_id="full-new"),
            ]
        )

        sources, gaps = ts.resolve_task3_sources(self._config(), self.root)

        self.assertEqual(gaps, [])
        self.assertEqual(
            sources,
            [ts.Task3Source("nice", "must", "zai", "glm-5.1", "full-new")],
        )

    def test_incompatible_runs_are_not_sources(self):
        for override, why in [
            ({"status": "running"}, "incomplete"),
            ({"run_group_id": "provider-matrix-2026-05"}, "another run group"),
            ({"tasks": "task1"}, "no task2"),
            ({"batch_size": 8}, "another request size"),
            ({"batch_order": "shuffled"}, "another batch order"),
            ({"run_id": "smoke-1"}, "a smoke run"),
        ]:
            with self.subTest(why=why):
                self._write_registry([_registry_row(**override)])
                sources, gaps = ts.resolve_task3_sources(self._config(), self.root)
                self.assertEqual(sources, [], why)
                self.assertEqual(len(gaps), 1)
                self.assertIn("no complete full-* Task 2 run", gaps[0].reason)

    def test_a_cell_without_a_source_is_reported_not_dropped(self):
        self._write_registry([_registry_row(model="glm-5")])

        sources, gaps = ts.resolve_task3_sources(self._config(), self.root)

        self.assertEqual(sources, [])
        self.assertEqual(len(gaps), 1)
        self.assertIn("nice/must zai/glm-5.1", gaps[0].as_line())

    def test_the_shall_variant_reads_its_own_run_prefix(self):
        self.assertEqual(ts.source_run_prefix("must"), "full")
        self.assertEqual(ts.source_run_prefix("shall"), "full-shall")

    def test_skipped_profiles_are_not_resolved(self):
        self._write_registry([_registry_row()])

        sources, gaps = ts.resolve_task3_sources(
            self._config(), self.root, skip_profiles={"zai"}
        )

        self.assertEqual((sources, gaps), ([], []))

    def test_cli_prints_one_line_per_cell(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout

        self._write_registry([_registry_row()])
        config_path = self.root / "run.json"
        eu.write_json(config_path, dict(self._config()))
        out, err = io.StringIO(), io.StringIO()
        with (
            redirect_stdout(out),
            redirect_stderr(err),
            mock.patch.object(eu, "project_root", lambda: self.root),
        ):
            code = ts.main(["--config", str(config_path)])

        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue().strip(), "nice\tmust\tzai\tglm-5.1\tfull-new")
        self.assertEqual(err.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
