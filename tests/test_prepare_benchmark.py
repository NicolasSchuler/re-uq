"""Preparation preserves reviewed inputs and regenerates the frozen study items."""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts import eval_utils as eu, prepare_benchmark as preparation

REPO_ROOT = Path(__file__).resolve().parents[1]


class PrepareBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(REPO_ROOT / "prompts", self.root / "prompts")
        shutil.copy2(
            REPO_ROOT / "config.example.json", self.root / "config.example.json"
        )
        self.config = eu.load_config(self.root / "config.example.json")
        self.config["project"]["target_seed_count"] = 1
        self.config["datasets"]["mlm_tapt_target_seed_count"] = 1

    def review_path(self, dataset="nice"):
        return eu.artifact_path(self.root / "data/processed/seeds_review.csv", dataset)

    def write_review(self, dataset="nice", *, count=1):
        rows = [
            {
                "seed_id": f"S{index:04d}",
                "source_dataset": eu.SOURCE_DATASET_LABELS[dataset],
                "source_corpus": "sample_corpus",
                "original_requirement": "The system shall export a report for each project.",
                "capability_text_auto": "export a report for each project",
                "capability_text_final": f"export reviewed report {index}",
                "include": "yes",
            }
            for index in range(count)
        ]
        eu.write_csv_rows(
            self.review_path(dataset), rows, eu.seed_review_fields(dataset)
        )
        return rows

    def build(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return preparation.build_benchmark(self.root, "nice", self.config, **kwargs)

    def test_cli_rebuilds_all_four_frozen_benchmarks_from_reviewed_seeds(self):
        for dataset in ("nice", "mlm_tapt"):
            with self.subTest(dataset=dataset):
                review = eu.artifact_path(
                    Path("data/processed/seeds_review.csv"), dataset
                )
                (self.root / review).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO_ROOT / review, self.root / review)
                manifest_path = eu.artifact_path(
                    Path("outputs/benchmark_manifest.json"), dataset
                )
                (self.root / manifest_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO_ROOT / manifest_path, self.root / manifest_path)
                original_metadata = json.loads((self.root / manifest_path).read_text())[
                    "metadata"
                ]
                with redirect_stdout(io.StringIO()):
                    preparation.main(
                        [
                            "--stage",
                            "build",
                            "--dataset",
                            dataset,
                            "--root",
                            str(self.root),
                        ]
                    )
                for variant in ("must", "shall"):
                    relative = eu.artifact_path(
                        Path("data/processed/benchmark_items.csv"), dataset, variant
                    )
                    self.assertEqual(
                        eu.read_csv_rows(self.root / relative),
                        eu.read_csv_rows(REPO_ROOT / relative),
                    )
                self.assertEqual(
                    json.loads((self.root / manifest_path).read_text())["metadata"],
                    original_metadata,
                )
                verified = eu.verify_benchmark_manifest(
                    self.root / manifest_path, self.root
                )
                self.assertEqual(verified["missing"], [])

    def test_candidate_refresh_preserves_manual_review_for_both_sources(self):
        source_rows = [
            {
                "RequirementText": "The system shall export a report for each active project.",
                "source": "sample_corpus",
                "reqs": "The system shall export a report for each active project.",
            }
        ]
        eu.write_csv_rows(
            self.root / self.config["datasets"]["nice_local_path"], source_rows
        )
        for dataset in ("nice", "mlm_tapt"):
            with self.subTest(dataset=dataset):
                self.write_review(dataset)
                before = self.review_path(dataset).read_bytes()
                with (
                    mock.patch.object(
                        eu, "load_mlm_tapt_rows", return_value=source_rows
                    ),
                    mock.patch.object(
                        eu,
                        "download_file",
                        side_effect=AssertionError("unexpected download"),
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    destination = preparation.prepare_candidates(
                        self.root, dataset, self.config
                    )
                self.assertEqual(self.review_path(dataset).read_bytes(), before)
                self.assertEqual(
                    destination, eu.auto_candidates_path(self.review_path(dataset))
                )
                candidates = eu.read_csv_rows(destination)
                self.assertEqual(len(candidates), 1)
                self.assertEqual(candidates[0]["include"], "yes")
                self.assertNotEqual(
                    candidates[0]["capability_text_final"], "export reviewed report 0"
                )
                self.assertFalse((self.root / "outputs").exists())

    def test_incomplete_review_cannot_build(self):
        self.write_review()
        self.config["project"]["target_seed_count"] = 2
        with self.assertRaisesRegex(ValueError, "exactly 2 included seeds"):
            self.build()
        self.assertFalse((self.root / "outputs").exists())
        self.assertFalse((self.root / "data/processed/seeds_selected.csv").exists())

    def test_duplicate_seed_ids_cannot_build(self):
        rows = self.write_review(count=2)
        rows[1]["seed_id"] = rows[0]["seed_id"]
        eu.write_csv_rows(self.review_path(), rows)
        self.config["project"]["target_seed_count"] = 2
        with self.assertRaisesRegex(ValueError, "unique benchmark items"):
            self.build()
        self.assertFalse((self.root / "outputs").exists())

    def test_changed_benchmark_requires_explicit_acceptance(self):
        rows = self.write_review()
        self.build()
        benchmark_path = self.root / "data/processed/benchmark_items.csv"
        selected_path = self.root / "data/processed/seeds_selected.csv"
        manifest_path = self.root / "outputs/benchmark_manifest.json"
        originals = {
            p: p.read_bytes() for p in (benchmark_path, selected_path, manifest_path)
        }
        rows[0]["capability_text_final"] = "export the newly reviewed report"
        eu.write_csv_rows(self.review_path(), rows)
        with self.assertRaisesRegex(ValueError, "Frozen inputs were preserved"):
            self.build()
        for path, before in originals.items():
            self.assertEqual(path.read_bytes(), before)
        candidate = eu.read_csv_rows(eu.candidate_path(benchmark_path))
        self.assertEqual(
            candidate[0]["capability_text"], "export the newly reviewed report"
        )
        self.build(overwrite=True)
        self.assertEqual(eu.read_csv_rows(benchmark_path), candidate)
        self.assertEqual(
            eu.verify_benchmark_manifest(manifest_path, self.root)["missing"], []
        )

    def test_single_variant_does_not_certify_a_stale_sibling(self):
        rows = self.write_review()
        self.build()
        manifest_path = self.root / "outputs/benchmark_manifest.json"
        before = manifest_path.read_bytes()
        rows[0]["capability_text_final"] = "export the newly reviewed report"
        eu.write_csv_rows(self.review_path(), rows)
        with self.assertRaisesRegex(ValueError, "use --variant both"):
            self.build(variant="must", overwrite=True)
        self.assertEqual(manifest_path.read_bytes(), before)

    def test_can_build_only_the_requested_variant_in_a_fresh_root(self):
        self.write_review()
        manifest = self.build(variant="shall")
        self.assertFalse((self.root / "data/processed/benchmark_items.csv").exists())
        self.assertEqual(manifest["metadata"]["main_benchmark"], "")
        self.assertEqual(manifest["metadata"]["robustness_benchmark"], "SHALL")
        self.assertEqual(
            eu.read_csv_rows(self.root / "data/processed/benchmark_items_shall.csv")[0][
                "mandatory_keyword"
            ],
            "SHALL",
        )


if __name__ == "__main__":
    unittest.main()
