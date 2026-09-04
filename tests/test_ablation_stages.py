"""The two ablation stages that had no script: batching and weak phrasing.

`TODO.md` section A left "the three ablation runs themselves and the comparison
table" open, and the weak-phrasing probe lived only in a notebook that bypassed
the runner. These tests pin what the two new entry points do: how a registry
row is read as a batching arm, how the batching delta is paired and resampled
(differently from the context ablation, because the request is the thing being
varied), and that the probe plans the four templates over the pilot seeds and
refuses to send anything until the construct review is complete.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import (
    compare_batching_ablation as batching,
    eval_utils as eu,
    run_weak_modality_probe as probe,
)


def _score_row(item_id, *, seed_id, strict, arm_batch_id):
    return {
        "model": "m1",
        "task": "task2",
        "uq_method": "verbalized_confidence",
        "item_id": item_id,
        "seed_id": seed_id,
        "batch_id": arm_batch_id,
        "source_modality": "nice_to_have",
        "gold_modality": "nice_to_have",
        "pred_modality": "nice_to_have",
        "y_true": "nice_to_have",
        "y_pred": "nice_to_have",
        "text_modality_parse_status": "ok",
        "strict_text_overcommit": strict,
        "text_overcommit": strict,
        "strict_text_high_conf_overcommit_90": strict,
        "confidence": 0.95,
    }


class BatchingArmTest(unittest.TestCase):
    def test_registry_rows_are_read_as_arms_by_their_batching_plan(self):
        cases = [
            ({"batch_size": 16, "batch_order": "grouped"}, batching.ARM_GROUPED),
            ({"batch_size": 16, "batch_order": "shuffled"}, batching.ARM_SHUFFLED),
            ({"batch_size": 1, "batch_order": "grouped"}, batching.ARM_SINGLE),
            # A single-item request has no order to speak of.
            ({"batch_size": 1, "batch_order": "shuffled"}, batching.ARM_SINGLE),
            # Pre-`batch_order` rows are the grouped paper condition.
            ({"batch_size": 16, "batch_order": ""}, batching.ARM_GROUPED),
        ]
        for row, expected in cases:
            with self.subTest(**row):
                self.assertEqual(batching.registry_arm(row), expected)

    def test_only_complete_fully_covered_runs_of_the_group_are_selected(self):
        base = {
            "run_group_id": batching.DEFAULT_RUN_GROUP_ID,
            "model": "m1",
            "status": "complete",
            "deterministic_item_coverage": 1.0,
            "batch_size": 16,
            "batch_order": "grouped",
            "started_at_utc": "2026-02-01T00:00:00Z",
        }
        rows = [
            {**base, "run_id": "full-old", "started_at_utc": "2026-01-01T00:00:00Z"},
            {**base, "run_id": "full-new"},
            {**base, "run_id": "full-partial", "deterministic_item_coverage": 0.5},
            {**base, "run_id": "full-other", "run_group_id": "another-group"},
            {**base, "run_id": "full-running", "status": "running"},
            {**base, "run_id": "smoke-1"},
            {**base, "run_id": "full-shuffled", "batch_order": "shuffled"},
        ]

        selected = batching.select_arm_runs(
            rows,
            run_group_id=batching.DEFAULT_RUN_GROUP_ID,
            include_smoke=False,
        )

        self.assertEqual(
            {key: row["run_id"] for key, row in selected.items()},
            {
                ("m1", batching.ARM_GROUPED): "full-new",
                ("m1", batching.ARM_SHUFFLED): "full-shuffled",
            },
        )

    def test_delta_pairs_by_item_and_resamples_by_seed(self):
        # Grouped: 4 items in one request. Single: the same 4 items, one each.
        grouped = [
            _score_row(
                f"i{index}",
                seed_id=f"S{index}",
                strict=index < 3,
                arm_batch_id="grouped-request",
            )
            for index in range(4)
        ]
        single = [
            _score_row(
                f"i{index}",
                seed_id=f"S{index}",
                strict=index < 1,
                arm_batch_id=f"single-request-{index}",
            )
            for index in range(4)
        ]

        rows = batching.delta_rows(
            "m1", batching.ARM_SINGLE, "all", grouped, single, bootstrap_samples=50
        )
        strict = next(
            row for row in rows if row["metric"] == "strict_text_strengthening"
        )

        # Strengthening drops from 3/4 to 1/4 when the batch is taken apart.
        self.assertEqual(strict["grouped"], 0.75)
        self.assertEqual(strict["arm_value"], 0.25)
        self.assertAlmostEqual(strict["delta"], -0.5)
        # Every item is answered in both arms, and the resampling unit is the
        # seed -- never the request, which is what the arms vary.
        self.assertEqual(strict["n_complete_pairs"], 4)
        self.assertEqual(strict["n_excluded_single_arm"], 0)
        self.assertEqual(strict["delta_cluster_field"], "seed_id")

    def test_items_answered_in_one_arm_only_are_excluded_and_counted(self):
        grouped = [
            _score_row(
                f"i{index}",
                seed_id=f"S{index}",
                strict=True,
                arm_batch_id="grouped-request",
            )
            for index in range(4)
        ]
        single = grouped[:2]

        strict = next(
            row
            for row in batching.delta_rows(
                "m1",
                batching.ARM_SINGLE,
                "all",
                grouped,
                single,
                bootstrap_samples=25,
            )
            if row["metric"] == "strict_text_strengthening"
        )

        self.assertEqual(strict["n_complete_pairs"], 2)
        self.assertEqual(strict["n_excluded_single_arm"], 2)

    def test_write_outputs_emits_both_tables_and_provenance(self):
        with TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "batching_ablation_summary"
            paths = batching.write_outputs(
                {
                    "arms": [
                        batching.arm_row(
                            "m1",
                            batching.ARM_GROUPED,
                            "all",
                            {
                                "run_id": "full-new",
                                "batch_size": 16,
                                "batch_order": "grouped",
                            },
                            [
                                _score_row(
                                    "i0", seed_id="S0", strict=True, arm_batch_id="r1"
                                )
                            ],
                            bootstrap_samples=0,
                        )
                    ],
                    "deltas": [],
                    "provenance": [{"model": "m1", "arm": "grouped"}],
                },
                prefix,
            )

            self.assertEqual(
                list(eu.read_csv_rows(paths["csv"])[0]), batching.ARM_FIELDS
            )
            self.assertTrue(paths["deltas_csv"].exists())
            self.assertIn("arm - grouped", paths["markdown"].read_text())
            provenance = json.loads(paths["provenance"].read_text())
            self.assertEqual(provenance["paper_batch_size"], 16)


class WeakModalityProbeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_smoke_run_ids_route_into_the_smoke_tree(self):
        # A probe smoke run must not append to the paper-facing probe file or
        # overwrite the shipped summary.
        smoke_id = "weak-modality-probe-smoke-1"
        self.assertTrue(eu.is_smoke_run_id(smoke_id))
        self.assertIn(
            "smoke",
            str(probe.probe_raw_path(self.root, "nice", "must", run_id=smoke_id)),
        )
        self.assertEqual(
            probe.probe_summary_dir(self.root, smoke_id),
            self.root / "outputs" / "smoke",
        )
        self.assertEqual(
            probe.probe_summary_dir(self.root, "weak-modality-probe-1"),
            self.root / "outputs",
        )
        self.assertNotIn(
            "smoke",
            str(
                probe.probe_raw_path(
                    self.root, "nice", "must", run_id="weak-modality-probe-1"
                )
            ),
        )

    def test_run_prefix_names_the_variant_and_the_mode(self):
        self.assertEqual(probe.probe_run_prefix("full", "must"), "weak-modality-probe")
        self.assertEqual(
            probe.probe_run_prefix("full", "shall"), "weak-modality-probe-shall"
        )
        self.assertEqual(
            probe.probe_run_prefix("smoke", "must"), "weak-modality-probe-smoke"
        )

    def test_pilot_seeds_follow_the_benchmark_order(self):
        seeds = [
            {
                "seed_id": f"S{index:04d}",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": f"export report set {index}",
            }
            for index in range(1, 5)
        ]
        benchmark = eu.build_benchmark_items(seeds)
        eu.write_csv_rows(
            eu.artifact_path(
                self.root / "data/processed/benchmark_items.csv", "nice", "must"
            ),
            benchmark,
        )
        eu.write_csv_rows(
            eu.artifact_path(self.root / "data/processed/seeds_selected.csv", "nice"),
            list(reversed(seeds)),
        )

        selected = probe.pilot_seeds(self.root, "nice", "must", 2)

        self.assertEqual([row["seed_id"] for row in selected], ["S0001", "S0002"])

    def test_a_missing_pilot_seed_is_an_error_not_a_short_probe(self):
        seeds = [
            {
                "seed_id": f"S{index:04d}",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": f"export report set {index}",
            }
            for index in range(1, 5)
        ]
        eu.write_csv_rows(
            eu.artifact_path(
                self.root / "data/processed/benchmark_items.csv", "nice", "must"
            ),
            eu.build_benchmark_items(seeds),
        )
        eu.write_csv_rows(
            eu.artifact_path(self.root / "data/processed/seeds_selected.csv", "nice"),
            seeds[:1],
        )

        with self.assertRaises(ValueError) as caught:
            probe.pilot_seeds(self.root, "nice", "must", 3)
        self.assertIn("missing 2 pilot seed(s)", str(caught.exception))

    def test_the_construct_review_gates_every_request(self):
        (self.root / "outputs").mkdir(parents=True)
        with self.assertRaises(probe.ProbeSanityError) as caught:
            probe.require_sanity_check(self.root, "nice")
        self.assertIn("weaker than SHOULD", str(caught.exception))

        # Marking every template as weaker than SHOULD opens the gate.
        paths = eu.write_weak_modality_template_sanity_check(self.root / "outputs")
        rows = eu.read_csv_rows(paths["csv"])
        for row in rows:
            row["weaker_than_should"] = "yes"
            row["reviewer"] = "fixture"
        eu.write_csv_rows(paths["csv"], rows, fieldnames=eu.WEAK_MODALITY_SANITY_FIELDS)

        status = probe.require_sanity_check(self.root, "nice")

        self.assertTrue(status["valid"])
        self.assertEqual(
            len(eu.build_weak_modality_probe_items(_probe_seeds())),
            len(_probe_seeds()) * len(eu.WEAK_MODALITY_PROBE_TEMPLATES),
        )


def _probe_seeds():
    return [
        {
            "seed_id": "S0001",
            "source_dataset": "NICE",
            "original_requirement": "The system shall export reports.",
            "capability_text_final": "export reports",
        }
    ]


if __name__ == "__main__":
    unittest.main()
