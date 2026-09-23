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

    def test_invalid_context_marker_does_not_publish_selected_seeds(self) -> None:
        """Validate the entire benchmark before publishing selected seeds."""
        self._write_review([_reviewed_seed(1, "X")])

        with self.assertRaisesRegex(ValueError, "M/O marker"):
            self._build(target_count=1)

        self.assertFalse(
            pure_builder.seeds_selected_path(self.root).exists(),
            "invalid reviewed context must not publish selected seeds",
        )

    def test_source_defects_are_rejected_before_any_output(self):
        for original, capability in (
            (
                "ETCS is required to be functional up to 500 km/h.",
                "ETCS is required to be functional up to 500 km/h",
            ),
            ("A dedicated Push-To-Talk button shall be provided.", "be provided"),
            (
                "When the radio changes network, an indication shall be given.",
                "be given",
            ),
        ):
            with self.subTest(original=original):
                seed = _reviewed_seed(1, "M")
                seed.update(
                    original_requirement=original, capability_text_final=capability
                )
                self._write_review([seed])
                with self.assertRaisesRegex(ValueError, "source-based review"):
                    self._build(target_count=1)
                self.assertFalse(pure_builder.seeds_selected_path(self.root).exists())
                self.assertFalse(pure_builder.benchmark_path(self.root).exists())

    def test_corrected_subject_retaining_capability_passes_source_copy_check(self):
        seed = _reviewed_seed(1, "M")
        seed.update(
            original_requirement="A dedicated Push-To-Talk button shall be provided.",
            capability_text_final="provide a dedicated Push-To-Talk button",
        )
        self._write_review([seed])
        _, rows = (
            self._build(target_count=1),
            eu.read_csv_rows(pure_builder.benchmark_path(self.root)),
        )
        for row in rows:
            self.assertEqual(
                eu.requirement_text_modality_diagnostic(row["source_statement"])[
                    "text_modality"
                ],
                row["source_modality"],
            )

    def test_pure_automatic_suggestion_preserves_non_system_subject_for_review(self):
        original = "A dedicated Push-To-Talk button shall be provided."
        self.assertEqual(
            eu.auto_capability_text(original, preserve_subject=True),
            original.rstrip("."),
        )

    def test_existing_benchmark_guard_rejects_changed_reviewed_source(self):
        self._build_two_seed_fixture()
        with mock.patch.object(
            pure_builder.eu,
            "load_config",
            return_value={"project": {"target_seed_count": 2}},
        ):
            pure_builder.validate_existing_benchmark(self.root)
            changed = [_reviewed_seed(1, "M"), _reviewed_seed(2, "O")]
            changed[0]["capability_text_final"] = "export encrypted reports"
            self._write_review(changed)
            with self.assertRaisesRegex(ValueError, "differs from reviewed seeds"):
                pure_builder.validate_existing_benchmark(self.root)

    def test_blank_final_capability_is_not_implicitly_approved(self):
        row = _reviewed_seed(1, "M")
        row["capability_text_final"] = ""
        self._write_review([row])
        with self.assertRaisesRegex(ValueError, "missing explicitly reviewed"):
            self._build(target_count=1)

    def test_revision_proposals_are_complete_unique_and_source_copy_consistent(self):
        proposals = eu.read_csv_rows(REPO_ROOT / "docs/pure_capability_revisions.csv")
        self.assertEqual(len(proposals), 180)
        self.assertEqual(len({r["seed_id"] for r in proposals}), 180)
        seeds = [
            {
                **r,
                "source_dataset": "PURE",
                "original_requirement": "fixture",
                "capability_text_final": r["proposed_capability"],
            }
            for r in proposals
        ]
        for item in eu.build_benchmark_items(seeds):
            self.assertEqual(
                eu.requirement_text_modality_diagnostic(item["source_statement"])[
                    "text_modality"
                ],
                item["source_modality"],
                item["item_id"],
            )

    def test_review_export_never_overwrites_author_judgments(self):
        self._write_review([_reviewed_seed(1, "M")])
        eu.write_csv_rows(
            self.root / "docs/pure_capability_revisions.csv",
            [{"seed_id": "PURE-0001", "proposed_capability": "export report 1"}],
        )
        path = pure_builder.write_capability_review(self.root)["review"]
        rows = eu.read_csv_rows(path)
        rows[0]["author_decision"] = "revise"
        eu.write_csv_rows(path, rows)
        candidate = pure_builder.write_capability_review(self.root)["review"]
        self.assertNotEqual(candidate, path)
        self.assertEqual(eu.read_csv_rows(path)[0]["author_decision"], "revise")

    def test_apply_revisions_requires_review_and_preserves_originals(self):
        seed = _reviewed_seed(1, "M")
        self._write_review([seed])
        path = pure_builder.seeds_review_path(self.root)
        original = path.read_bytes()
        proposal_path = self.root / "docs/pure_capability_revisions.csv"
        proposal = {
            "seed_id": seed["seed_id"],
            "proposed_capability": "export a revised report",
        }
        eu.write_csv_rows(proposal_path, [proposal])
        with self.assertRaisesRegex(ValueError, "explicit accepted decision"):
            pure_builder.apply_capability_revisions(self.root)
        self.assertEqual(path.read_bytes(), original)
        eu.write_csv_rows(proposal_path, [{**proposal, "review_decision": "accepted"}])
        with mock.patch.object(
            pure_builder.eu,
            "load_config",
            return_value={"project": {"target_seed_count": 1}},
        ):
            result = pure_builder.apply_capability_revisions(self.root)
            self.assertEqual(
                pure_builder.apply_capability_revisions(self.root)["status"],
                "unchanged",
            )
        self.assertEqual(
            (result["backup"] / path.relative_to(self.root)).read_bytes(), original
        )
        revised = eu.read_csv_rows(path)[0]
        self.assertEqual(
            revised["capability_text_final"], proposal["proposed_capability"]
        )
        for key in seed.keys() - {"capability_text_final"}:
            self.assertEqual(revised[key], seed[key])

    def test_invalid_reviewed_revision_does_not_write_or_backup(self):
        seed = _reviewed_seed(1, "M")
        self._write_review([seed])
        path = pure_builder.seeds_review_path(self.root)
        original = path.read_bytes()
        eu.write_csv_rows(
            self.root / "docs/pure_capability_revisions.csv",
            [
                {
                    "seed_id": seed["seed_id"],
                    "proposed_capability": "must export a report",
                    "review_decision": "accepted",
                }
            ],
        )
        with (
            mock.patch.object(
                pure_builder.eu,
                "load_config",
                return_value={"project": {"target_seed_count": 1}},
            ),
            self.assertRaisesRegex(ValueError, "review issues"),
        ):
            pure_builder.apply_capability_revisions(self.root)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.root / "outputs/pure_before_capability_review").exists())


if __name__ == "__main__":
    unittest.main()
