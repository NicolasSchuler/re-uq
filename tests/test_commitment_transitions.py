"""Frozen selection, deterministic scoring and transition accounting regressions."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scripts import eval_utils as eu, export_commitment_transitions as transitions


class CommitmentTransitionsTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        examples = [
            ("nice_to_have", "The system may export reports."),
            ("optional", "The system must export reports."),
            ("optional", "The system exports reports."),
            ("optional", "The system may export reports."),
            ("mandatory", "The system should export reports."),
            ("recommended", "Export reports."),
            ("recommended", "The system must not export reports."),
            ("nice_to_have", None),
        ]
        self.items, self.raw = [], []
        for index, (source, requirement) in enumerate(examples):
            item = {
                "item_id": str(index),
                "seed_id": str(index),
                "source_modality": source,
                "task2_gold_modality": source,
                "ordinal_strength": eu.ORDINAL_STRENGTH[source],
                "numeric_strength": 0.5,
                "source_statement": f"Stakeholder statement {index}",
            }
            self.items.append(item)
            self.raw.append(
                {
                    "run_id": "frozen",
                    "model": "model",
                    "task": "task2",
                    "sample_kind": "deterministic",
                    "sample_index": 0,
                    "item_id": str(index),
                    "source_modality": source,
                    "prompt": item["source_statement"],
                    "confidence_scale": "0_1",
                    "parse_status": "ok" if requirement else "error",
                    "parsed_json": {
                        "requirement": requirement,
                        "modality": source,
                        "confidence": 0.9,
                    }
                    if requirement
                    else None,
                }
            )
        self.snapshot = {
            "models_cohort": ["model"],
            "cells": [
                {
                    "dataset": "nice",
                    "variant": "must",
                    "run_ids": {"model": "frozen"},
                    "sampling_plan_source": "planned",
                    "expected_stochastic_samples": 5,
                    "n_benchmark_items": len(self.items),
                    "benchmark_path": "benchmark.csv",
                    "raw_path": "raw.jsonl",
                }
            ],
        }

    def load(self):
        path = self.root / "snapshot.json"
        path.write_text(json.dumps(self.snapshot))
        with (
            mock.patch.object(transitions.eu, "read_csv_rows", return_value=self.items),
            mock.patch.object(
                transitions.paper,
                "stream_raw_rows",
                side_effect=lambda path, runs: (
                    row for row in self.raw if row["run_id"] in runs
                ),
            ),
            mock.patch.object(
                eu,
                "semantic_embedding_matrix",
                side_effect=AssertionError("No embeddings allowed"),
            ),
        ):
            return transitions.load_observations(self.root, path)

    def test_all_outcomes_and_pooled_accounting(self):
        observations = self.load()
        self.assertEqual(
            [row["outcome"] for row in observations],
            [
                "strict_up",
                "strict_up",
                "broad_only_up",
                "equal",
                "down",
                "unknown",
                "negated",
                "invalid",
            ],
        )
        self.assertEqual(observations[-1]["source_modality"], "nice_to_have")
        self.assertEqual(observations[-1]["run_id"], "frozen")
        matrix, accounting = transitions.aggregate_observations(observations)
        self.assertEqual(len(matrix), 64)
        pooled = [
            row
            for row in accounting
            if row["model"] == row["dataset"] == row["variant"] == "all"
        ]
        self.assertEqual(sum(row["n_planned"] for row in pooled), 8)
        self.assertEqual(sum(row["n_readable"] for row in pooled), 5)
        optional = next(
            row
            for row in matrix
            if row["model"] == row["dataset"] == row["variant"] == "all"
            and row["source_modality"] == "optional"
            and row["text_modality"] == "mandatory"
        )
        self.assertEqual(
            (optional["n"], optional["n_strict"], optional["n_heuristic"]), (2, 1, 1)
        )
        self.assertAlmostEqual(optional["strict_percentage"], 100 / 3)
        self.assertTrue(
            all(
                row["strict_percentage"] == ""
                for row in matrix
                if row["source_modality"] == "recommended"
            )
        )
        self.assertEqual(
            sum(
                row["n_frame_only"]
                for row in matrix
                if row["model"] == row["dataset"] == row["variant"] == "all"
            ),
            1,
        )

    def test_retries_frozen_selection_and_stochastic_exclusion(self):
        failure = dict(self.raw[0], parse_status="error", parsed_json=None)
        self.raw.insert(0, failure)
        self.raw.append(dict(self.raw[1], sample_kind="stochastic", sample_index=1))
        self.raw.append(dict(self.raw[1], task="task3"))
        self.raw.append(dict(self.raw[1], run_id="newer-unselected", model="other"))
        self.assertEqual(len(self.load()), 8)

    def test_later_failed_retry_keeps_success(self):
        self.raw.append(dict(self.raw[0], parse_status="error", parsed_json=None))
        self.assertEqual(self.load()[0]["outcome"], "strict_up")

    def test_heuristic_equal_and_weak_escalations(self):
        self.raw[4]["parsed_json"]["requirement"] = "The system exports reports."
        observations = self.load()
        self.assertEqual(observations[4]["outcome"], "equal")
        self.assertEqual(
            observations[4]["text_modality_basis"], "heuristic_system_verb"
        )
        for modal in ("should", "must"):
            self.raw[0]["parsed_json"]["requirement"] = (
                f"The system {modal} export reports."
            )
            row = self.load()[0]
            self.assertEqual(row["outcome"], "strict_up")
            self.assertEqual(
                row["strict_text_overcommit_kind"], eu.STRICT_OVERCOMMIT_ESCALATION
            )

    def test_reconciliation_rejects_wrong_denominator_and_count(self):
        _, accounting = transitions.aggregate_observations(self.load())
        published = []
        for row in accounting:
            if row["model"] == "model" and row["dataset"] == "all":
                published.append(
                    {
                        "model": "model",
                        "source_modality": row["source_modality"],
                        "n_items": row["n_planned"],
                        "n_valid": row["n_planned"] - row["n_invalid"],
                        "n_parse_failures": row["n_invalid"],
                        "n_text_unclassified": row["n_unknown"] + row["n_negated"],
                        "strict_strengthening_n": row["n_strict_up"],
                        "strict_strengthening_denominator": row["n_readable"],
                        "broad_strengthening_n": row["n_strict_up"]
                        + row["n_broad_only_up"],
                        "broad_strengthening_denominator": row["n_readable"],
                    }
                )
        published.append(dict(published[0], model="all"))
        with mock.patch.object(eu, "read_csv_rows", return_value=published):
            transitions.reconcile(accounting, Path("unused.csv"))
            for field in ("strict_strengthening_denominator", "strict_strengthening_n"):
                published[0][field] += 1
                with self.assertRaisesRegex(ValueError, "reconciliation mismatch"):
                    transitions.reconcile(accounting, Path("unused.csv"))
                published[0][field] -= 1

    def test_same_item_ids_across_cells_survive(self):
        cell = dict(self.snapshot["cells"][0], variant="shall")
        self.snapshot["cells"].append(cell)
        observations = self.load()
        self.assertEqual(len(observations), 16)
        self.assertEqual(
            len({transitions.paper.paper_join_key(row) for row in observations}), 16
        )

    def test_missing_planned_row_fails(self):
        self.raw.pop()
        with self.assertRaisesRegex(ValueError, "Missing planned"):
            self.load()

    def test_benchmark_prompt_mismatch_fails(self):
        self.raw[0]["prompt"] = "Different source"
        with self.assertRaisesRegex(ValueError, "Raw/benchmark"):
            self.load()

    def test_unexpected_model_for_frozen_run_fails(self):
        self.raw[0]["model"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "Unexpected model/run"):
            self.load()

    def test_wrong_frozen_run_for_model_fails(self):
        self.snapshot["models_cohort"].append("second")
        self.snapshot["cells"][0]["run_ids"]["second"] = "second-run"
        self.raw[0]["run_id"] = "second-run"
        with self.assertRaisesRegex(ValueError, "Unexpected model/run"):
            self.load()

    def test_distinct_provider_duplicate_paper_key_fails(self):
        self.raw.append(dict(self.raw[0], provider_id="other-provider"))
        with self.assertRaisesRegex(ValueError, "Duplicate paper key"):
            self.load()

    def test_compacted_store_reader(self):
        eu.write_csv_rows(self.root / "benchmark.csv", self.items, list(self.items[0]))
        # Exercise the real compacted-store reader while keeping Parquet creation local.
        import pyarrow.parquet as pq

        pq.write_table(eu.raw_rows_to_table(self.raw), self.root / "raw.parquet")
        path = self.root / "snapshot.json"
        path.write_text(json.dumps(self.snapshot))
        observations = transitions.load_observations(self.root, path)
        self.assertEqual(len(observations), 8)


if __name__ == "__main__":
    unittest.main()
