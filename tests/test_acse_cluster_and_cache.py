"""Regression tests for the ACSE clustering, projection, and cache-validity helpers.

Four contracts are pinned here, all offline:

* ``eval_utils`` exposes one clustering operation that normalizes once, builds the
  cosine-distance matrix once, fits once, and returns the diagnostics dict and the
  size-ranked labels together. The recorded golden values below were captured from
  the pre-refactor implementation, so any drift in cluster labels, tie handling, or
  score arithmetic fails loudly.
* Item centroids are averaged in the projected space rather than re-projected.
* Both embedding-diagnostic figures prepare their projection inputs identically,
  including RNG consumption.
* The ACSE artifact cache is discovered from provenance manifests (not directory
  names) and is invalidated by input fingerprints rather than a status string.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, ClassVar
from unittest import mock

import numpy as np

from scripts import (
    compute_acse_semantic_artifacts as semantic_cache,
    eval_utils as eu,
    plot_acse_embedding_visualizations as acse_viz,
    plot_embedding_diagnostic_figure as diagnostic_figure,
)

THRESHOLD = 0.35


def _boundary_matrix() -> np.ndarray:
    """Two unit vectors whose cosine distance is exactly ``THRESHOLD``."""
    cos = 1.0 - THRESHOLD
    return np.asarray([[1.0, 0.0], [cos, float(np.sqrt(1.0 - cos * cos))]])


def _cluster_fixtures() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260527)
    fixtures = {
        "empty": np.zeros((0, 4)),
        "singleton": np.asarray([[1.0, 2.0, 3.0]]),
        "all_zero": np.zeros((5, 4)),
        "identical": np.tile(np.asarray([[1.0, 0.0, 0.0]]), (6, 1)),
        "tied": np.asarray(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        "boundary": _boundary_matrix(),
        "zero_mixed": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    }
    for rows, columns in [(3, 2), (7, 5), (12, 3), (20, 8)]:
        fixtures[f"random_{rows}x{columns}"] = rng.normal(size=(rows, columns))
    return fixtures


# Captured from the pre-refactor implementation at distance_threshold=0.35.
# ``random_20x8`` is deliberately included: with 15 clusters the distribution is
# ordered by the *string* label, so "cluster_10" sorts between "cluster_1" and
# "cluster_2". That quirk decides dominant-cluster ties and must survive.
GOLDEN_CLUSTERS: dict[str, dict[str, Any]] = {
    "empty": {
        "labels": [],
        "semantic_cluster_count": 0,
        "semantic_cluster_distribution": "",
        "semantic_cluster_entropy": math.nan,
        "semantic_dominant_cluster_share": math.nan,
        "semantic_mean_pairwise_distance": math.nan,
        "semantic_uncertainty_score": math.nan,
    },
    "singleton": {
        "labels": ["cluster_0"],
        "semantic_cluster_count": 1,
        "semantic_cluster_distribution": '{"cluster_0":1.0}',
        "semantic_cluster_entropy": 0.0,
        "semantic_dominant_cluster_share": 1.0,
        "semantic_mean_pairwise_distance": 0.0,
        "semantic_uncertainty_score": 0.0,
    },
    "all_zero": {
        "labels": ["cluster_4", "cluster_3", "cluster_1", "cluster_2", "cluster_0"],
        "semantic_cluster_count": 5,
        "semantic_cluster_distribution": (
            '{"cluster_0":0.2,"cluster_1":0.2,"cluster_2":0.2,'
            '"cluster_3":0.2,"cluster_4":0.2}'
        ),
        "semantic_cluster_entropy": 1.0000000000000002,
        "semantic_dominant_cluster_share": 0.2,
        "semantic_mean_pairwise_distance": 1.0,
        "semantic_uncertainty_score": 1.0,
    },
    "identical": {
        "labels": ["cluster_0"] * 6,
        "semantic_cluster_count": 1,
        "semantic_cluster_distribution": '{"cluster_0":1.0}',
        "semantic_cluster_entropy": 0.0,
        "semantic_dominant_cluster_share": 1.0,
        "semantic_mean_pairwise_distance": 0.0,
        "semantic_uncertainty_score": 0.0,
    },
    "tied": {
        "labels": ["cluster_1", "cluster_1", "cluster_0", "cluster_0"],
        "semantic_cluster_count": 2,
        "semantic_cluster_distribution": '{"cluster_0":0.5,"cluster_1":0.5}',
        "semantic_cluster_entropy": 1.0,
        "semantic_dominant_cluster_share": 0.5,
        "semantic_mean_pairwise_distance": 0.6666666666666666,
        "semantic_uncertainty_score": 1.0,
    },
    "boundary": {
        "labels": ["cluster_1", "cluster_0"],
        "semantic_cluster_count": 2,
        "semantic_cluster_distribution": '{"cluster_0":0.5,"cluster_1":0.5}',
        "semantic_cluster_entropy": 1.0,
        "semantic_dominant_cluster_share": 0.5,
        "semantic_mean_pairwise_distance": 0.35,
        "semantic_uncertainty_score": 1.0,
    },
    "zero_mixed": {
        "labels": ["cluster_2", "cluster_1", "cluster_0"],
        "semantic_cluster_count": 3,
        "semantic_cluster_distribution": (
            '{"cluster_0":0.333333333333,"cluster_1":0.333333333333,'
            '"cluster_2":0.333333333333}'
        ),
        "semantic_cluster_entropy": 0.9999999999999998,
        "semantic_dominant_cluster_share": 0.3333333333333333,
        "semantic_mean_pairwise_distance": 1.0,
        "semantic_uncertainty_score": 0.9999999999999998,
    },
    "random_3x2": {
        "labels": ["cluster_2", "cluster_1", "cluster_0"],
        "semantic_cluster_count": 3,
        "semantic_cluster_distribution": (
            '{"cluster_0":0.333333333333,"cluster_1":0.333333333333,'
            '"cluster_2":0.333333333333}'
        ),
        "semantic_cluster_entropy": 0.9999999999999998,
        "semantic_dominant_cluster_share": 0.3333333333333333,
        "semantic_mean_pairwise_distance": 0.6697575529534356,
        "semantic_uncertainty_score": 0.9999999999999998,
    },
    "random_7x5": {
        "labels": [
            "cluster_0",
            "cluster_5",
            "cluster_2",
            "cluster_3",
            "cluster_4",
            "cluster_1",
            "cluster_0",
        ],
        "semantic_cluster_count": 6,
        "semantic_cluster_distribution": (
            '{"cluster_0":0.285714285714,"cluster_1":0.142857142857,'
            '"cluster_2":0.142857142857,"cluster_3":0.142857142857,'
            '"cluster_4":0.142857142857,"cluster_5":0.142857142857}'
        ),
        "semantic_cluster_entropy": 0.9755037590061084,
        "semantic_dominant_cluster_share": 0.2857142857142857,
        "semantic_mean_pairwise_distance": 0.7868728662785449,
        "semantic_uncertainty_score": 0.9804030072048868,
    },
    "random_12x3": {
        "labels": [
            "cluster_4",
            "cluster_3",
            "cluster_0",
            "cluster_1",
            "cluster_2",
            "cluster_0",
            "cluster_2",
            "cluster_0",
            "cluster_1",
            "cluster_5",
            "cluster_3",
            "cluster_0",
        ],
        "semantic_cluster_count": 6,
        "semantic_cluster_distribution": (
            '{"cluster_0":0.333333333333,"cluster_1":0.166666666667,'
            '"cluster_2":0.166666666667,"cluster_3":0.166666666667,'
            '"cluster_4":0.083333333333,"cluster_5":0.083333333333}'
        ),
        "semantic_cluster_entropy": 0.9355245321275765,
        "semantic_dominant_cluster_share": 0.3333333333333333,
        "semantic_mean_pairwise_distance": 0.7268441294711597,
        "semantic_uncertainty_score": 0.9484196257020612,
    },
    "random_20x8": {
        "labels": [
            "cluster_2",
            "cluster_11",
            "cluster_9",
            "cluster_0",
            "cluster_1",
            "cluster_13",
            "cluster_10",
            "cluster_14",
            "cluster_6",
            "cluster_0",
            "cluster_2",
            "cluster_5",
            "cluster_1",
            "cluster_3",
            "cluster_3",
            "cluster_8",
            "cluster_7",
            "cluster_4",
            "cluster_4",
            "cluster_12",
        ],
        "semantic_cluster_count": 15,
        "semantic_cluster_distribution": (
            '{"cluster_0":0.1,"cluster_1":0.1,"cluster_10":0.05,"cluster_11":0.05,'
            '"cluster_12":0.05,"cluster_13":0.05,"cluster_14":0.05,"cluster_2":0.1,'
            '"cluster_3":0.1,"cluster_4":0.1,"cluster_5":0.05,"cluster_6":0.05,'
            '"cluster_7":0.05,"cluster_8":0.05,"cluster_9":0.05}'
        ),
        "semantic_cluster_entropy": 0.9782531661325102,
        "semantic_dominant_cluster_share": 0.1,
        "semantic_mean_pairwise_distance": 0.8673256206677158,
        "semantic_uncertainty_score": 0.9826025329060082,
    },
}


class AcseClusterAnalysisTest(unittest.TestCase):
    def _assert_matches_golden(
        self, name: str, diagnostics: dict[str, Any], labels: list[str]
    ) -> None:
        expected = GOLDEN_CLUSTERS[name]
        self.assertEqual(labels, expected["labels"])
        for field, value in expected.items():
            if field == "labels":
                continue
            actual = diagnostics[field]
            if isinstance(value, float) and math.isnan(value):
                self.assertTrue(math.isnan(actual), f"{name}.{field} should be NaN")
            elif isinstance(value, float):
                self.assertAlmostEqual(actual, value, places=12, msg=f"{name}.{field}")
            else:
                self.assertEqual(actual, value, f"{name}.{field}")

    def test_canonical_analysis_reproduces_recorded_diagnostics_and_labels(
        self,
    ) -> None:
        for name, matrix in _cluster_fixtures().items():
            with self.subTest(fixture=name):
                analysis = eu.acse_semantic_cluster_analysis(
                    matrix, "fixture-backend", THRESHOLD
                )
                self._assert_matches_golden(
                    name, analysis.diagnostics, analysis.cluster_labels
                )
                self.assertEqual(
                    analysis.diagnostics["semantic_embedding_backend"],
                    "fixture-backend",
                )
                self.assertEqual(
                    analysis.diagnostics["semantic_distance_threshold"], THRESHOLD
                )

    def test_public_wrappers_return_the_same_result_as_the_single_analysis(
        self,
    ) -> None:
        for name, matrix in _cluster_fixtures().items():
            with self.subTest(fixture=name):
                analysis = eu.acse_semantic_cluster_analysis(
                    matrix, "fixture-backend", THRESHOLD
                )
                diagnostics = eu.acse_semantic_diagnostics_from_embeddings(
                    matrix, "fixture-backend", THRESHOLD
                )
                labels = eu.acse_cluster_labels_for_embeddings(matrix, THRESHOLD)
                self.assertEqual(labels, analysis.cluster_labels)
                self.assertEqual(
                    json.dumps(diagnostics, default=str),
                    json.dumps(analysis.diagnostics, default=str),
                )

    def test_analysis_fits_normalizes_and_measures_distance_exactly_once(self) -> None:
        matrix = _cluster_fixtures()["random_12x3"]
        fits: list[int] = []

        class CountingClusterer(eu.AgglomerativeClustering):  # type: ignore[misc]
            def fit_predict(self, X, y=None):
                fits.append(1)
                return super().fit_predict(X, y)

        with (
            mock.patch.object(eu, "AgglomerativeClustering", CountingClusterer),
            mock.patch.object(
                eu, "normalize_embedding_rows", wraps=eu.normalize_embedding_rows
            ) as normalize,
            mock.patch.object(
                eu, "cosine_similarity", wraps=eu.cosine_similarity
            ) as similarity,
        ):
            analysis = eu.acse_semantic_cluster_analysis(
                matrix, "fixture-backend", THRESHOLD
            )

        self.assertEqual(sum(fits), 1, "clustering must be fit exactly once")
        self.assertEqual(normalize.call_count, 1, "embeddings must be normalized once")
        self.assertEqual(
            similarity.call_count, 1, "the distance matrix must be built once"
        )
        self._assert_matches_golden(
            "random_12x3", analysis.diagnostics, analysis.cluster_labels
        )

    def test_analysis_rejects_a_non_matrix_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "2D embedding matrix"):
            eu.acse_semantic_cluster_analysis(
                np.zeros((2, 2, 2)), "fixture-backend", THRESHOLD
            )


class AcseVisualizationClusteringTest(unittest.TestCase):
    def test_visualization_script_uses_the_canonical_clustering_helper(self) -> None:
        """The figure script must not carry its own copy of the clustering code."""
        self.assertFalse(
            hasattr(acse_viz, "cluster_labels_for_embeddings"),
            "the duplicate clustering implementation should be gone",
        )
        source = Path(acse_viz.__file__).read_text(encoding="utf-8")
        self.assertNotIn("AgglomerativeClustering", source)
        self.assertIn("acse_cluster_labels_for_embeddings", source)


class AcseCentroidProjectionTest(unittest.TestCase):
    """Item centroids are the mean of the projected sample coordinates."""

    def _write_cache(self, cache_dir: Path, embeddings: np.ndarray) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Item A gets four samples, item B a single one (the singleton case).
        item_ids = ["A", "A", "A", "A", "B"]
        np.savez_compressed(
            cache_dir / "task2_acse_sample_embeddings.npz",
            embeddings=np.asarray(embeddings, dtype=np.float32),
        )
        eu.write_csv_rows(
            cache_dir / "task2_acse_samples.csv",
            [
                {
                    "embedding_index": index,
                    "item_id": item_id,
                    "seed_id": "S0001",
                    "sample_index": index,
                    "pred_modality": "optional",
                    "semantic_text": f"text {index}",
                    "requirement": f"The system MAY do {index}.",
                }
                for index, item_id in enumerate(item_ids)
            ],
        )
        eu.write_csv_rows(
            cache_dir / "task2_acse_items.csv",
            [
                {
                    "item_id": item_id,
                    "seed_id": "S0001",
                    "source_modality": "optional",
                    "strict_text_overcommit": False,
                    "text_overcommit": False,
                    "acse_uncertainty_score": 0.25,
                    "task2_requirement": "The system MAY do it.",
                }
                for item_id in ["A", "B"]
            ],
        )
        (cache_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": "full-1",
                    "model": "model-1",
                    "embedding_backend": "mlx:fixture/model",
                    "distance_threshold": THRESHOLD,
                }
            ),
            encoding="utf-8",
        )

    def _load(self, embeddings: np.ndarray, components: int) -> list[dict[str, Any]]:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            self._write_cache(cache_dir, embeddings)
            _, item_rows, _, _ = acse_viz.load_cached_projection_inputs(
                cache_dir, "full-1", "model-1", THRESHOLD, components
            )
        return item_rows

    @staticmethod
    def _as_cached(embeddings: np.ndarray) -> np.ndarray:
        """The cache stores float32, so the reference must round-trip too."""
        return np.asarray(np.asarray(embeddings, dtype=np.float32), dtype=float)

    def _assert_centroids_match_projected_mean(
        self, embeddings: np.ndarray, components: int
    ) -> None:
        item_rows = self._load(embeddings, components)
        projected, _ = acse_viz.projection_model(
            self._as_cached(embeddings), components
        )
        expected = {
            "A": np.mean(projected[[0, 1, 2, 3]], axis=0),
            "B": projected[4],
        }
        axes = ["x", "y", "z"]
        for row in item_rows:
            reference = expected[row["item_id"]]
            for axis_index, axis in enumerate(axes):
                wanted = (
                    float(reference[axis_index])
                    if components == 3 or axis != "z"
                    else 0.0
                )
                self.assertAlmostEqual(
                    row[axis],
                    wanted,
                    delta=1e-9,
                    msg=f"{row['item_id']}.{axis} (components={components})",
                )

    def test_centroids_equal_the_projected_mean_in_three_components(self) -> None:
        rng = np.random.default_rng(7)
        self._assert_centroids_match_projected_mean(rng.normal(size=(5, 6)), 3)

    def test_centroids_equal_the_projected_mean_in_two_components(self) -> None:
        rng = np.random.default_rng(11)
        self._assert_centroids_match_projected_mean(rng.normal(size=(5, 6)), 2)

    def test_two_dimensional_embeddings_pad_to_three_components(self) -> None:
        """A rank-2 projection padded to 3 columns must keep z at exactly 0."""
        rng = np.random.default_rng(13)
        embeddings = rng.normal(size=(5, 2))
        item_rows = self._load(embeddings, 3)
        self.assertEqual([row["z"] for row in item_rows], [0.0, 0.0])
        self._assert_centroids_match_projected_mean(embeddings, 3)

    def test_centroids_match_the_re_projected_centroid_they_replaced(self) -> None:
        """Equivalence with the old ``pca.transform(mean(embeddings))`` route."""
        rng = np.random.default_rng(17)
        embeddings = rng.normal(size=(5, 6))
        projected, pca = acse_viz.projection_model(embeddings, 3)
        for indices in ([0, 1, 2, 3], [4]):
            legacy = pca.transform(
                np.mean(embeddings[indices, :], axis=0, keepdims=True)
            )
            current = np.mean(projected[indices], axis=0)
            np.testing.assert_allclose(current, legacy[0], atol=1e-9)


class EmbeddingProjectionPrepTest(unittest.TestCase):
    """Both diagnostic figures must prepare identical inputs from one helper."""

    SOURCES: ClassVar[list[str]] = [
        "mandatory",
        "recommended",
        "optional",
        "nice_to_have",
    ]

    def _build_fixture(self, root: Path) -> tuple[Path, Path]:
        artifact_dir = root / "cache"
        artifact_dir.mkdir(parents=True)
        diagnostic_dir = root / "embedding_diagnostic"
        diagnostic_dir.mkdir(parents=True)

        samples: list[dict[str, Any]] = []
        items: list[dict[str, Any]] = []
        index = 0
        for source in self.SOURCES:
            for repeat in range(8):
                item_id = f"{source}-{repeat}"
                items.append(
                    {
                        "item_id": item_id,
                        "source_modality": source,
                        "strict_text_overcommit": repeat % 4 == 0,
                        "text_overcommit": repeat % 3 == 0,
                        "acse_uncertainty_score": 0.1 * repeat,
                    }
                )
                # Two rows per item, and every fourth item repeats an earlier
                # requirement so the de-duplication step actually drops rows.
                text = f"The system MUST do {source} {repeat % 6}."
                for sample_index in range(2):
                    samples.append(
                        {
                            "embedding_index": index,
                            "item_id": item_id,
                            "seed_id": f"S{repeat:04d}",
                            "sample_index": sample_index,
                            "pred_modality": "mandatory",
                            "semantic_text": text,
                            "requirement": text,
                        }
                    )
                    index += 1

        rng = np.random.default_rng(3)
        embeddings = rng.normal(size=(len(samples), 5)).astype(np.float32)
        np.savez_compressed(
            artifact_dir / "task2_acse_sample_embeddings.npz", embeddings=embeddings
        )
        eu.write_csv_rows(artifact_dir / "task2_acse_samples.csv", samples)
        eu.write_csv_rows(artifact_dir / "task2_acse_items.csv", items)
        np.savez_compressed(
            diagnostic_dir / "task2_reqonly_mlx_embeddings.npz", embeddings=embeddings
        )

        manifest_path = root / eu.ACSE_SEMANTIC_MANIFEST_FILENAME
        eu.write_csv_rows(
            manifest_path,
            [
                {
                    "dataset_id": "nice",
                    "benchmark_variant": "must",
                    "run_id": "full-1",
                    "model": "model-1",
                    "profile_id": "profile-1",
                    "embedding_backend": "mlx:fixture/model",
                    "artifact_dir": str(artifact_dir),
                }
            ],
        )
        return diagnostic_dir, manifest_path

    @staticmethod
    def _legacy_prepare(
        diagnostic_dir: Path, manifest_path: Path, per_modality: int, random_state: int
    ) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, np.random.Generator]:
        """The block both scripts inlined before the helper was extracted."""
        cache = np.load(
            diagnostic_dir / "task2_reqonly_mlx_embeddings.npz", allow_pickle=False
        )
        reqonly = cache["embeddings"].astype(np.float32, copy=False)
        rows = diagnostic_figure.manifest_rows(manifest_path, "mlx:")
        _, sample_rows = diagnostic_figure.load_embeddings_and_rows(rows)
        seen: set[str] = set()
        unique_global: list[int] = []
        for i, row in enumerate(sample_rows):
            text = str(row.get("requirement", ""))
            if text and text not in seen:
                seen.add(text)
                unique_global.append(i)
        unique_rows = [sample_rows[i] for i in unique_global]
        unique_global_arr = np.asarray(unique_global)
        rng = np.random.default_rng(random_state)
        keep_local = diagnostic_figure.balanced_subsample(
            unique_rows, per_modality, rng
        )
        global_idx = unique_global_arr[keep_local]
        sub_rows = [unique_rows[k] for k in keep_local]
        coords = diagnostic_figure.compute_projection(
            reqonly[global_idx], "pca", random_state
        )
        return coords, sub_rows, global_idx, rng

    def test_helper_reproduces_the_inlined_selection_and_rng_consumption(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_dir, manifest_path = self._build_fixture(Path(tmpdir))
            expected = self._legacy_prepare(diagnostic_dir, manifest_path, 3, 20260527)
            prepared = diagnostic_figure.prepare_projection_inputs(
                diagnostic_dir=diagnostic_dir,
                manifest_path=manifest_path,
                method="pca",
                per_modality=3,
                random_state=20260527,
            )

        expected_coords, expected_rows, expected_idx, expected_rng = expected
        np.testing.assert_array_equal(prepared.indices, expected_idx)
        self.assertEqual(
            [row["item_id"] for row in prepared.rows],
            [row["item_id"] for row in expected_rows],
        )
        self.assertEqual(
            [row["global_embedding_index"] for row in prepared.rows],
            [row["global_embedding_index"] for row in expected_rows],
        )
        np.testing.assert_allclose(prepared.coords, expected_coords, atol=1e-12)
        # The RNG must be handed back mid-stream: both figures keep drawing from
        # it for the jitter in panel (a).
        self.assertEqual(prepared.rng.random(), expected_rng.random())

    def test_subsampling_actually_consumes_the_rng(self) -> None:
        """Guards the test above: an untouched RNG would make it vacuous."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diagnostic_dir, manifest_path = self._build_fixture(Path(tmpdir))
            prepared = diagnostic_figure.prepare_projection_inputs(
                diagnostic_dir=diagnostic_dir,
                manifest_path=manifest_path,
                method="pca",
                per_modality=3,
                random_state=20260527,
            )
        self.assertLess(
            prepared.indices.size, 24, "subsampling should have dropped rows"
        )
        self.assertNotEqual(
            prepared.rng.random(), np.random.default_rng(20260527).random()
        )

    def test_supplementary_figure_calls_the_shared_helper(self) -> None:
        from scripts import plot_embedding_diagnostic_tsne_supp as supp

        source = Path(supp.__file__).read_text(encoding="utf-8")
        self.assertIn("prepare_projection_inputs", source)
        self.assertNotIn("balanced_subsample(", source)


def _seed_row() -> dict[str, Any]:
    return {
        "seed_id": "S0001",
        "source_dataset": "NICE",
        "original_requirement": "The system shall export reports.",
        "capability_text_final": "export reports",
    }


class AcseCacheValidityTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)
        self.output_root = self.root / "outputs"
        self.analysis_dir = self.output_root / "evaluation_nice_must_full_1"
        self.analysis_dir.mkdir(parents=True)
        self.items = [
            row
            for row in eu.build_benchmark_items([_seed_row()])
            if row["source_modality"] in {"optional", "recommended"}
        ]
        eu.write_csv_rows(self.root / "data/processed/benchmark_items.csv", self.items)
        self._write_raw_rows()
        self._write_scores()
        self.run = semantic_cache.CompletedRun(
            analysis_dir=self.analysis_dir,
            dataset_id="nice",
            variant="must",
            run_id="full-1",
            model="model-1",
            profile="profile-1",
            semantic_embedding_backend="tfidf",
        )

    @property
    def raw_path(self) -> Path:
        return eu.model_outputs_raw_path(self.root, "nice", "must")

    def _raw_row(
        self, item: dict[str, Any], sample_kind: str, sample_index: int, modality: str
    ) -> dict[str, Any]:
        modal = {"mandatory": "MUST", "recommended": "SHOULD", "optional": "MAY"}[
            modality
        ]
        return {
            "run_id": "full-1",
            "run_group_id": "group-1",
            "provider_id": "provider-1",
            "profile_id": "profile-1",
            "model": "model-1",
            "task": "task2",
            "item_id": item["item_id"],
            "seed_id": item["seed_id"],
            "source_modality": item["source_modality"],
            "sample_index": sample_index,
            "sample_kind": sample_kind,
            "parse_status": "ok",
            "parsed_json": {
                "requirement": f"The system {modal} export report {sample_index}.",
                "modality": modality,
                "confidence": 0.9,
            },
            "confidence_scale": eu.CONFIDENCE_SCALE_0_1,
        }

    def _write_raw_rows(self) -> None:
        for item in self.items:
            eu.append_jsonl(
                self.raw_path, self._raw_row(item, "deterministic", 0, "optional")
            )
            for sample_index, modality in enumerate(
                ["optional", "recommended", "mandatory"]
            ):
                eu.append_jsonl(
                    self.raw_path,
                    self._raw_row(item, "stochastic", sample_index, modality),
                )

    def _write_scores(self) -> None:
        eu.write_csv_rows(
            self.analysis_dir / "uq_scores.csv",
            [
                {
                    "run_id": "full-1",
                    "model": "model-1",
                    "task": "task2",
                    "uq_method": "verbalized_confidence",
                    "item_id": item["item_id"],
                    "confidence": 0.9,
                    **dict.fromkeys(semantic_cache.TEXT_MODALITY_FIELDS, ""),
                }
                for item in self.items
            ],
        )

    def _write_provenance(self, model_filter: str) -> None:
        (self.analysis_dir / "provenance_manifest.json").write_text(
            json.dumps(
                {
                    "run_id": "full-1",
                    "model_filter": model_filter,
                    "profile_filter": "profile-1",
                    "dataset_id": "nice",
                    "benchmark_variant": "must",
                    "semantic_embedding_backend": "tfidf",
                }
            ),
            encoding="utf-8",
        )

    def _compute(self, *, force: bool = False) -> dict[str, Any]:
        return semantic_cache.compute_run_backend(
            self.root,
            self.run,
            embedding_backend="tfidf",
            mlx_model_name=None,
            distance_threshold=THRESHOLD,
            embedding_batch_size=16,
            force=force,
        )

    def _manifest_path(self, manifest: dict[str, Any]) -> Path:
        return (
            eu.acse_semantic_cache_dir(
                self.analysis_dir, str(manifest["embedding_backend"])
            )
            / "manifest.json"
        )

    # -- discovery -------------------------------------------------------

    def test_discovery_accepts_a_manifest_with_an_empty_model_filter(self) -> None:
        self._write_provenance("")
        runs = semantic_cache.completed_runs_from_analysis_dirs(
            self.root, self.output_root
        )
        self.assertEqual([run.model for run in runs], ["model-1"])
        self.assertEqual(runs[0].run_id, "full-1")
        self.assertEqual(runs[0].analysis_dir, self.analysis_dir)
        self.assertEqual(runs[0].dataset_id, "nice")

    def test_discovery_expands_an_empty_model_filter_to_every_scored_model(
        self,
    ) -> None:
        self._write_provenance("")
        rows = eu.read_csv_rows(self.analysis_dir / "uq_scores.csv")
        extra = [dict(row, model="model-2") for row in rows]
        eu.write_csv_rows(self.analysis_dir / "uq_scores.csv", rows + extra)

        runs = semantic_cache.completed_runs_from_analysis_dirs(
            self.root, self.output_root
        )
        self.assertEqual(sorted(run.model for run in runs), ["model-1", "model-2"])

    def test_discovery_keeps_an_explicit_model_filter(self) -> None:
        self._write_provenance("model-1")
        runs = semantic_cache.completed_runs_from_analysis_dirs(
            self.root, self.output_root
        )
        self.assertEqual([run.model for run in runs], ["model-1"])

    def test_discovery_is_manifest_driven_not_directory_name_driven(self) -> None:
        renamed = self.output_root / "acse_reanalysis_nice_must"
        self.analysis_dir.rename(renamed)
        self.analysis_dir = renamed
        self._write_provenance("model-1")

        runs = semantic_cache.completed_runs_from_analysis_dirs(
            self.root, self.output_root
        )
        self.assertEqual([run.analysis_dir for run in runs], [renamed])

    # -- fingerprints ----------------------------------------------------

    def test_unchanged_inputs_reuse_the_cache(self) -> None:
        first = self._compute()
        second = self._compute()
        self.assertEqual(first["status"], "computed")
        self.assertEqual(second["status"], "reused")

    def test_manifest_records_input_fingerprints_with_manifest_relative_paths(
        self,
    ) -> None:
        manifest = self._compute()
        inputs = manifest["inputs"]
        self.assertEqual(
            sorted(inputs),
            [
                "benchmark_items",
                "embedding_backend",
                "embedding_model",
                "raw_rows",
                "uq_scores",
            ],
        )
        self.assertEqual(inputs["embedding_backend"], eu.ACSE_PROXY_EMBEDDING_BACKEND)
        manifest_dir = self._manifest_path(manifest).parent
        for key in ["raw_rows", "uq_scores", "benchmark_items"]:
            entry = inputs[key]
            recorded = Path(entry["path"])
            self.assertFalse(recorded.is_absolute(), f"{key} path must be relative")
            resolved = (manifest_dir / recorded).resolve()
            self.assertTrue(resolved.exists(), f"{key} -> {resolved}")
            self.assertEqual(entry["sha256"], eu.sha256_file(resolved))

    def test_changed_raw_rows_trigger_a_recompute(self) -> None:
        self._compute()
        with self.raw_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(self._raw_row(self.items[0], "stochastic", 9, "mandatory"))
                + "\n"
            )
        second = self._compute()
        self.assertEqual(second["status"], "computed")

    def test_changed_scores_trigger_a_recompute(self) -> None:
        first = self._compute()
        rows = eu.read_csv_rows(self.analysis_dir / "uq_scores.csv")
        rows[0]["confidence"] = "0.5"
        eu.write_csv_rows(self.analysis_dir / "uq_scores.csv", rows)
        second = self._compute()
        self.assertEqual(first["status"], "computed")
        self.assertEqual(second["status"], "computed")

    def test_changed_benchmark_items_trigger_a_recompute(self) -> None:
        self._compute()
        rows = eu.read_csv_rows(self.root / "data/processed/benchmark_items.csv")
        rows[0]["source_statement"] = rows[0]["source_statement"] + " Updated."
        eu.write_csv_rows(self.root / "data/processed/benchmark_items.csv", rows)
        self.assertEqual(self._compute()["status"], "computed")

    def test_a_manifest_without_fingerprints_is_readable_but_stale(self) -> None:
        manifest = self._compute()
        manifest_path = self._manifest_path(manifest)
        legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
        legacy.pop("inputs")
        manifest_path.write_text(json.dumps(legacy), encoding="utf-8")

        with self.assertLogs(eu.LOGGER_NAME, level="INFO") as logs:
            second = self._compute()
        self.assertEqual(second["status"], "computed")
        self.assertIn("inputs", second)
        self.assertTrue(
            any("fingerprint" in message for message in logs.output), logs.output
        )

    def test_a_changed_embedding_backend_is_a_different_cache(self) -> None:
        manifest = self._compute()
        self.assertEqual(
            manifest["inputs"]["embedding_backend"], eu.ACSE_PROXY_EMBEDDING_BACKEND
        )
        self.assertEqual(manifest["inputs"]["embedding_model"], "")

    # -- single clustering fit per item ----------------------------------

    def test_each_item_is_clustered_exactly_once(self) -> None:
        fits: list[int] = []

        class CountingClusterer(eu.AgglomerativeClustering):  # type: ignore[misc]
            def fit_predict(self, X, y=None):
                fits.append(1)
                return super().fit_predict(X, y)

        with mock.patch.object(eu, "AgglomerativeClustering", CountingClusterer):
            manifest = self._compute()

        self.assertEqual(manifest["item_rows"], len(self.items))
        self.assertEqual(sum(fits), len(self.items))


if __name__ == "__main__":
    unittest.main()
