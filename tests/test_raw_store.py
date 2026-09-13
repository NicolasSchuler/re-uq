"""The raw row store: JSONL tail + compacted Parquet sibling.

Invariants: a round trip through Parquet returns the same rows in the same
order with the same keys; readers see both halves; compaction is atomic and
refuses to truncate when the read-back differs.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow.parquet as pq

from scripts import compact_raw_store, eval_utils as eu


def _rows(n: int, run_id: str = "run-1") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(n):
        row: dict[str, object] = {
            "run_id": run_id,
            "task": "task2" if index % 2 else "task1",
            "item_id": f"S{index:04d}_mandatory",
            "sample_index": index % 6,
            "latency_s": 0.5 * index,
            "json_mode": True,
            "parse_status": "ok" if index % 7 else "invalid_json",
            "parsed_json": {"decision": "yes", "confidence": 0.9, "n": index}
            if index % 7
            else None,
            "request_indices": [index],
            "raw_text": f"line {index} with unicode é and a\ttab",
        }
        if index % 3 == 0:
            # Sparse key: absent from two thirds of the rows.
            row["batch_id"] = f"b{index}"
        if index % 5 == 0:
            # Mixed types within one column force JSON encoding.
            row["request_seed"] = index if index % 10 else "seed-str"
        if index % 4 == 0:
            # Sparse key that is sometimes present with a null value: a
            # nullable column alone cannot distinguish that from absence.
            row["usage_total_tokens"] = None if index % 8 else 12
        rows.append(row)
    return rows


class RawStoreRoundTripTest(unittest.TestCase):
    def test_table_round_trip_preserves_rows_keys_and_order(self) -> None:
        rows = _rows(40)
        back = eu.raw_table_to_rows(eu.raw_rows_to_table(rows))
        self.assertEqual(back, rows)
        # Absent keys stay absent; None-valued dense keys stay None.
        self.assertNotIn("batch_id", back[1])
        self.assertIn("batch_id", back[0])
        self.assertIsNone(back[0]["parsed_json"])
        self.assertIsNone(back[4]["usage_total_tokens"])
        self.assertNotIn("usage_total_tokens", back[1])

    def test_mixed_and_nested_columns_are_json_encoded(self) -> None:
        table = eu.raw_rows_to_table(_rows(20))
        json_columns = json.loads(table.schema.metadata[eu.RAW_STORE_JSON_COLUMNS_KEY])
        self.assertIn("parsed_json", json_columns)
        self.assertIn("request_indices", json_columns)
        self.assertIn("request_seed", json_columns)
        self.assertNotIn("latency_s", json_columns)
        self.assertNotIn("json_mode", json_columns)

    def test_foreign_parquet_is_refused(self) -> None:
        import pyarrow as pa

        with self.assertRaisesRegex(ValueError, "raw store"):
            eu.raw_table_to_rows(pa.table({"run_id": ["x"]}))


class CompactionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "model_outputs_raw.jsonl"

    def _append(self, rows: list[dict[str, object]]) -> None:
        for row in rows:
            eu.append_jsonl(self.path, row)

    def test_missing_file_reads_empty_and_compacts_to_nothing(self) -> None:
        self.assertEqual(eu.read_jsonl(self.path), [])
        summary = eu.compact_jsonl(self.path)
        self.assertFalse(summary["compacted"])
        self.assertFalse(eu.raw_store_path(self.path).exists())

    def test_compaction_moves_tail_into_parquet_and_reader_sees_both(self) -> None:
        first = _rows(30, "run-1")
        self._append(first)
        summary = eu.compact_jsonl(self.path)
        self.assertTrue(summary["compacted"])
        self.assertEqual(summary["tail_rows"], 30)
        self.assertEqual(self.path.stat().st_size, 0)
        self.assertTrue(eu.raw_store_path(self.path).exists())
        self.assertEqual(eu.read_jsonl(self.path), first)

        second = _rows(10, "run-2")
        self._append(second)
        self.assertEqual(eu.read_jsonl(self.path), first + second)
        # Existence helpers report the store, not just the tail.
        self.assertTrue(eu.raw_store_exists(self.path))
        self.path.unlink()
        self.assertTrue(eu.raw_store_exists(self.path))
        self.assertEqual(eu.read_jsonl(self.path), first)

    def test_second_compaction_merges_and_keeps_order(self) -> None:
        first, second = _rows(12, "run-1"), _rows(8, "run-2")
        self._append(first)
        eu.compact_jsonl(self.path)
        self._append(second)
        summary = eu.compact_jsonl(self.path)
        self.assertEqual(summary["store_rows_before"], 12)
        self.assertEqual(summary["store_rows_after"], 20)
        self.assertEqual(eu.read_jsonl(self.path), first + second)
        self.assertEqual(pq.read_metadata(eu.raw_store_path(self.path)).num_rows, 20)

    def test_store_read_filters_by_run_and_projects_columns(self) -> None:
        self._append(_rows(12, "run-1") + _rows(6, "run-2"))
        eu.compact_jsonl(self.path)
        subset = eu.read_raw_store(
            self.path,
            columns=["run_id", "item_id", "parsed_json", "nope"],
            run_ids={"run-2"},
        )
        self.assertEqual(len(subset), 6)
        self.assertTrue(all(row["run_id"] == "run-2" for row in subset))
        self.assertEqual(set(subset[1]), {"run_id", "item_id", "parsed_json"})
        # A projection that names a sparse-null column still restores the null.
        projected = eu.read_raw_store(
            self.path, columns=["item_id", "usage_total_tokens"], run_ids={"run-1"}
        )
        self.assertIsNone(projected[4]["usage_total_tokens"])
        self.assertNotIn("usage_total_tokens", projected[1])
        self.assertEqual(subset[1]["parsed_json"]["n"], 1)
        self.assertEqual(eu.read_raw_store(self.path, run_ids=set()), [])

    def test_failed_verification_leaves_both_files_untouched(self) -> None:
        rows = _rows(9)
        self._append(rows)
        before = self.path.read_bytes()
        with (
            mock.patch.object(eu, "_raw_rows_canonical", side_effect=[["a"], ["b"]]),
            self.assertRaisesRegex(RuntimeError, "round trip mismatch"),
        ):
            eu.compact_jsonl(self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(eu.raw_store_path(self.path).exists())
        self.assertEqual(list(self.path.parent.glob("*.tmp-*")), [])

    def test_resume_sees_compacted_rows_as_completed(self) -> None:
        seed = {
            "seed_id": "S0001",
            "source_dataset": "NICE",
            "original_requirement": "The system shall export reports.",
            "capability_text_final": "export reports",
        }
        items = [
            row
            for row in eu.build_benchmark_items([seed])
            if row["source_modality"] == "mandatory"
        ]
        jobs = eu.planned_completion_jobs(
            items,
            tasks=["task1"],
            model="m1",
            host="http://localhost:1234/v1",
            run_id="run-1",
            prompt_version="v2-instructor-conf01",
            task1_template=eu.load_prompt("prompts/mandatory_entailment.txt"),
            task2_template=eu.load_prompt("prompts/modality_extraction.txt"),
            deterministic={"temperature": 0.0, "top_p": 1.0, "samples": 1},
            stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 0},
            max_tokens=64,
            timeout_s=30,
            api_key_env="LOCAL_OPENAI_API_KEY",
            structured_output="instructor",
        )
        self.assertGreater(len(jobs), 0)
        for job in jobs:
            eu.append_jsonl(
                self.path,
                {**job, "run_id": "run-1", "parse_status": "ok"},
            )
        eu.compact_jsonl(self.path)
        pending = eu.pending_completion_jobs(jobs, eu.read_jsonl(self.path), "run-1")
        self.assertEqual(pending, [])


class CompactCliTest(unittest.TestCase):
    def test_cli_compacts_default_globs_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data/processed/logs").mkdir(parents=True)
            raw = root / "data/processed/model_outputs_raw_nice.jsonl"
            transcript = root / "data/processed/logs/full_1.transcript.jsonl"
            for row in _rows(5):
                eu.append_jsonl(raw, row)
                eu.append_jsonl(transcript, {"attempt": 0, **row})
            with mock.patch.object(eu, "project_root", return_value=root):
                self.assertEqual(compact_raw_store.main([]), 0)
            self.assertEqual(raw.stat().st_size, 0)
            self.assertTrue(eu.raw_store_path(transcript).exists())
            self.assertEqual(len(eu.read_jsonl(raw)), 5)
            with mock.patch.object(eu, "project_root", return_value=root):
                self.assertEqual(compact_raw_store.main(["--dry-run"]), 0)


if __name__ == "__main__":
    unittest.main()
