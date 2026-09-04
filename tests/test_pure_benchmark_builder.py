"""End-to-end contracts for the PURE document-context benchmark builder."""

from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts import build_pure_benchmark as pure_builder, eval_utils as eu

REPO_ROOT = Path(__file__).resolve().parents[1]


def _reviewed_seed(index: int, marker: str) -> dict[str, str]:
    return {
        "seed_id": f"PURE-{index:04d}",
        "source_dataset": "PURE",
        "source_corpus": f"document-{index}",
        "original_requirement": f"({marker}) The system shall export report {index}.",
        "capability_text_auto": f"export report {index}",
        "auto_include": "yes",
        "auto_exclusion_reason": "",
        "include": "yes",
        "exclusion_reason": "",
        "capability_text_final": f"export report {index}",
        "context_document": f"Fixture document {index}",
        "context_requirement_id": f"{index}.1",
        "context_marker": marker,
        "context_section": f"{index} Fixture section",
        "context_before": f"Context before requirement {index}",
        "context_after": f"Context after requirement {index}",
        "context_legend": f"Fixture legend {index}: (M) mandatory, (O) optional",
    }


class PureBenchmarkBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)
        (self.root / "prompts").mkdir(parents=True)
        for name in ("modality_extraction.txt", "modality_extraction_context.txt"):
            shutil.copyfile(REPO_ROOT / "prompts" / name, self.root / "prompts" / name)

    def _write_review(self, rows: list[dict[str, str]]) -> None:
        eu.write_csv_rows(
            pure_builder.seeds_review_path(self.root),
            rows,
            fieldnames=eu.seed_review_fields(eu.DATASET_PURE),
        )

    def _build(self, target_count: int) -> dict[str, object]:
        with (
            mock.patch.object(
                pure_builder.eu,
                "load_config",
                return_value={"project": {"target_seed_count": target_count}},
            ),
            redirect_stdout(io.StringIO()),
        ):
            return pure_builder.build_benchmark(self.root)

    def _build_two_seed_fixture(
        self,
    ) -> tuple[dict[str, object], list[dict[str, str]]]:
        self._write_review([_reviewed_seed(1, "M"), _reviewed_seed(2, "O")])
        result = self._build(target_count=2)
        rows = eu.read_csv_rows(pure_builder.benchmark_path(self.root))
        return result, rows

    def test_build_emits_four_unique_modality_variants_per_seed(self) -> None:
        _, rows = self._build_two_seed_fixture()

        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row["item_id"] for row in rows}), 8)
        self.assertEqual({row["source_modality"] for row in rows}, set(eu.MODALITIES))

    def test_build_preserves_each_seeds_document_context_across_variants(self) -> None:
        _, rows = self._build_two_seed_fixture()

        expected_by_seed = {
            "PURE-0001": _reviewed_seed(1, "M"),
            "PURE-0002": _reviewed_seed(2, "O"),
        }
        for seed_id, expected_seed in expected_by_seed.items():
            seed_rows = [row for row in rows if row["seed_id"] == seed_id]
            self.assertEqual(len(seed_rows), 4, seed_id)
            for row in seed_rows:
                for field in eu.PURE_CONTEXT_FIELDS:
                    self.assertEqual(
                        row[field],
                        expected_seed[field],
                        f"{seed_id} variant {row['source_modality']} changed {field}",
                    )

    def test_build_writes_a_verifiable_manifest_with_context_counts(self) -> None:
        result, _ = self._build_two_seed_fixture()

        manifest = result["manifest"]
        self.assertEqual(manifest["metadata"]["marker_counts"], {"M": 1, "O": 1})
        self.assertEqual(
            manifest["metadata"]["documents"],
            {"document-1": 1, "document-2": 1},
        )
        verification = eu.verify_benchmark_manifest(
            pure_builder.manifest_path(self.root), self.root
        )
        self.assertGreater(verification["checked"], 0)
        self.assertEqual(verification["checked"], len(manifest["artifacts"]))
        self.assertEqual(verification["missing"], [])

    def test_build_publishes_selected_seeds_and_review_exports(self) -> None:
        self._build_two_seed_fixture()

        self.assertTrue(pure_builder.seeds_selected_path(self.root).exists())
        self.assertTrue(
            (self.root / "outputs/benchmark_statements_review_pure.csv").exists()
        )
        self.assertTrue(
            (self.root / "outputs/benchmark_statements_review_pure.md").exists()
        )

    def test_invalid_context_marker_stops_benchmark_outputs(self) -> None:
        self._write_review([_reviewed_seed(1, "X")])

        with self.assertRaisesRegex(ValueError, "M/O marker"):
            self._build(target_count=1)

        self.assertFalse(
            pure_builder.benchmark_path(self.root).exists(),
            "an invalid context marker must not produce a benchmark CSV",
        )
        self.assertFalse(
            pure_builder.manifest_path(self.root).exists(),
            "an invalid context marker must not produce a provenance manifest",
        )
        self.assertFalse(
            (self.root / "outputs/benchmark_statements_review_pure.csv").exists()
        )

    @unittest.expectedFailure
    def test_invalid_context_marker_does_not_publish_selected_seeds(self) -> None:
        """Known bug: selected seeds are written before marker validation."""
        self._write_review([_reviewed_seed(1, "X")])

        with self.assertRaisesRegex(ValueError, "M/O marker"):
            self._build(target_count=1)

        self.assertFalse(
            pure_builder.seeds_selected_path(self.root).exists(),
            "invalid reviewed context must not publish selected seeds",
        )


if __name__ == "__main__":
    unittest.main()
