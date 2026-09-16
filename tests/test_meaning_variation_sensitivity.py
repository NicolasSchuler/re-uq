"""Regressions for the offline meaning-variation sensitivity analysis.

Pinned here, all offline and synthetic:

* the optional ``dispersion_weight`` of ``acse_semantic_cluster_analysis``
  defaults to the module constant, so the reported score is unchanged, and its
  endpoints isolate the entropy and dispersion components;
* changing the threshold changes the clusters on a fixture built for it;
* the frozen-cohort join keeps readable deterministic rows with a scored group
  only, and its accounting is explicit about what was dropped;
* the paired cluster bootstrap draws exactly as ``eu.bootstrap_seed_metric``
  and evaluates every setting on the same draw;
* cached embeddings are bound to the sample texts they were computed from.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts import eval_utils as eu, meaning_variation_sensitivity as mvs

BACKEND = "mlx:test-encoder"


def _unit(angle_degrees: float) -> list[float]:
    angle = math.radians(angle_degrees)
    return [math.cos(angle), math.sin(angle)]


def _random_matrix(seed: int, rows: int = 5, columns: int = 16) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, columns))


def _group(
    model: str,
    item_id: str,
    seed_id: str,
    embeddings: np.ndarray,
    *,
    dataset: str = "nice",
    variant: str = "must",
    valid_n: int | None = None,
    total_n: int = 5,
) -> mvs.SampleGroup:
    count = embeddings.shape[0] if valid_n is None else valid_n
    return mvs.SampleGroup(
        model=model,
        dataset=dataset,
        variant=variant,
        dataset_id=eu.normalize_dataset_id(dataset),
        benchmark_variant=eu.normalize_benchmark_variant(variant),
        item_id=item_id,
        run_id=f"run-{model}",
        seed_id=seed_id,
        source_modality="mandatory",
        valid_n=count,
        total_n=total_n,
        sample_indices=[str(i) for i in range(count)],
        texts=[f"modality: mandatory requirement: text {i}" for i in range(count)],
        embeddings=embeddings,
    )


def _det_row(
    model: str,
    item_id: str,
    seed_id: str,
    *,
    readable: bool = True,
    strict: bool = False,
    dataset: str = "nice",
    variant: str = "must",
) -> dict:
    return {
        "model": model,
        "dataset_id": eu.normalize_dataset_id(dataset),
        "benchmark_variant": eu.normalize_benchmark_variant(variant),
        "item_id": item_id,
        "seed_id": seed_id,
        "source_modality": "mandatory",
        "text_modality_parse_status": "ok" if readable else "unparsed",
        "strict_text_overcommit": "True" if strict else "False",
    }


class DispersionWeightParameterTest(unittest.TestCase):
    def test_default_weight_reproduces_the_module_constant(self) -> None:
        for seed in range(6):
            matrix = _random_matrix(seed)
            default = eu.acse_semantic_cluster_analysis(matrix, BACKEND)
            explicit = eu.acse_semantic_cluster_analysis(
                matrix,
                BACKEND,
                distance_threshold=eu.ACSE_PROXY_DISTANCE_THRESHOLD,
                dispersion_weight=eu.ACSE_PROXY_INTERNAL_DISPERSION_WEIGHT,
            )
            self.assertEqual(default.diagnostics, explicit.diagnostics)
            self.assertEqual(default.cluster_labels, explicit.cluster_labels)
            self.assertEqual(
                eu.acse_semantic_diagnostics_from_embeddings(matrix, BACKEND),
                default.diagnostics,
            )

    def test_weight_endpoints_isolate_entropy_and_dispersion(self) -> None:
        for seed in range(6):
            matrix = _random_matrix(seed)
            for threshold in (0.10, 0.35, 0.70):
                entropy_only = eu.acse_semantic_diagnostics_from_embeddings(
                    matrix, BACKEND, distance_threshold=threshold, dispersion_weight=0.0
                )
                dispersion_only = eu.acse_semantic_diagnostics_from_embeddings(
                    matrix, BACKEND, distance_threshold=threshold, dispersion_weight=1.0
                )
                self.assertAlmostEqual(
                    entropy_only["semantic_uncertainty_score"],
                    entropy_only["semantic_cluster_entropy"],
                    places=12,
                )
                expected_dispersion = min(
                    1.0,
                    max(
                        dispersion_only["semantic_mean_pairwise_distance"],
                        dispersion_only["semantic_dominant_cluster_mean_distance"],
                    )
                    / threshold,
                )
                self.assertAlmostEqual(
                    dispersion_only["semantic_uncertainty_score"],
                    expected_dispersion,
                    places=12,
                )
                # Clustering does not depend on the weight.
                self.assertEqual(
                    entropy_only["semantic_cluster_count"],
                    dispersion_only["semantic_cluster_count"],
                )

    def test_weight_outside_the_unit_interval_is_rejected(self) -> None:
        matrix = _random_matrix(1)
        for weight in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                eu.acse_semantic_cluster_analysis(
                    matrix, BACKEND, dispersion_weight=weight
                )

    def test_threshold_changes_the_clusters(self) -> None:
        # Pairwise cosine distances: 0/30 deg 0.134, 30/75 deg 0.293,
        # 0/75 deg 0.741. Average linkage joins 0 and 30 first, then the pair
        # to 75 deg at (0.741 + 0.293) / 2 = 0.517.
        matrix = np.asarray([_unit(0.0), _unit(30.0), _unit(75.0)])
        counts = {
            threshold: eu.acse_semantic_diagnostics_from_embeddings(
                matrix, BACKEND, distance_threshold=threshold
            )["semantic_cluster_count"]
            for threshold in (0.10, 0.35, 0.70)
        }
        self.assertEqual(counts, {0.10: 3, 0.35: 2, 0.70: 1})


class ScoreGridTest(unittest.TestCase):
    def test_grid_reference_column_matches_the_default_scorer(self) -> None:
        groups = [
            _group("m", f"S{i}_mandatory", f"S{i}", _random_matrix(i)) for i in range(4)
        ]
        thresholds = (0.10, 0.35, 0.70)
        weights = (0.0, 0.2, 1.0)
        grid = mvs.score_grid(groups, thresholds, weights, backend=BACKEND)
        self.assertEqual(grid.scores.shape, (4, 3, 3))
        self.assertEqual(grid.cluster_counts.shape, (4, 3))
        t_ref, w_ref = grid.setting_index(0.35, 0.2)
        for g, group in enumerate(groups):
            default = eu.acse_semantic_diagnostics_from_embeddings(
                group.embeddings, BACKEND
            )
            self.assertEqual(
                grid.scores[g, t_ref, w_ref], default["semantic_uncertainty_score"]
            )
            self.assertEqual(
                grid.cluster_counts[g, t_ref], default["semantic_cluster_count"]
            )
            self.assertEqual(
                grid.mean_pairwise[g, t_ref], default["semantic_mean_pairwise_distance"]
            )
            # Entropy is threshold-dependent, mean pairwise distance is not.
            self.assertTrue(
                np.allclose(grid.mean_pairwise[g], grid.mean_pairwise[g, 0])
            )

    def test_grid_requires_embeddings(self) -> None:
        group = _group("m", "S1_mandatory", "S1", _random_matrix(1))
        group.embeddings = None
        with self.assertRaises(ValueError):
            mvs.score_grid([group], (0.35,), (0.2,), backend=BACKEND)


class CohortJoinTest(unittest.TestCase):
    def test_join_keeps_readable_rows_with_a_scored_group(self) -> None:
        groups = [
            _group("m", "S1_mandatory", "S1", _random_matrix(1)),
            _group("m", "S2_mandatory", "S2", _random_matrix(2), valid_n=3),
            _group("m", "S3_mandatory", "S3", _random_matrix(3)[:1], valid_n=1),
            _group("m", "S5_mandatory", "S5", _random_matrix(5)),
        ]
        det_rows = [
            _det_row("m", "S1_mandatory", "S1", strict=True),
            _det_row("m", "S2_mandatory", "S2"),
            _det_row("m", "S3_mandatory", "S3", readable=False),
            _det_row("m", "S4_mandatory", "S4"),  # readable, no group
            _det_row("m", "S5_mandatory", "S5", strict=True),
        ]
        cohort = mvs.join_cohort(
            [("nice", "must", det_rows, 6)], groups, {("m", "nice", "must"): 1}
        )
        self.assertEqual(
            [row["item_id"] for row in cohort.rows],
            ["S1_mandatory", "S2_mandatory", "S5_mandatory"],
        )
        self.assertEqual(cohort.labels.tolist(), [1, 0, 1])
        self.assertEqual(cohort.group_index.tolist(), [0, 1, 3])
        self.assertEqual(len(cohort.accounting), 1)
        tally = cohort.accounting[0]
        self.assertEqual(
            {
                key: tally[key]
                for key in (
                    "planned_items",
                    "deterministic_parsed",
                    "deterministic_missing",
                    "unreadable",
                    "readable",
                    "readable_without_group",
                    "evaluated",
                    "strict_positive",
                    "groups_scored",
                    "groups_without_valid_sample",
                    "groups_complete_five",
                    "groups_incomplete",
                    "groups_single_sample",
                    "missing_or_invalid_samples",
                )
            },
            {
                "planned_items": 6,
                "deterministic_parsed": 5,
                "deterministic_missing": 1,
                "unreadable": 1,
                "readable": 4,
                "readable_without_group": 1,
                "evaluated": 3,
                "strict_positive": 2,
                "groups_scored": 4,
                "groups_without_valid_sample": 1,
                "groups_complete_five": 2,
                "groups_incomplete": 2,
                "groups_single_sample": 1,
                "missing_or_invalid_samples": 2 + 4,
            },
        )
        self.assertTrue(cohort.mask(model="m").all())
        self.assertFalse(cohort.mask(cell=("nice", "shall")).any())

    def test_join_rejects_duplicate_group_keys(self) -> None:
        groups = [
            _group("m", "S1_mandatory", "S1", _random_matrix(1)),
            _group("m", "S1_mandatory", "S1", _random_matrix(2)),
        ]
        with self.assertRaises(ValueError):
            mvs.join_cohort([("nice", "must", [], 1)], groups, {})

    def test_join_key_separates_cells_that_reuse_item_ids(self) -> None:
        must = _group("m", "S1_mandatory", "S1", _random_matrix(1), variant="must")
        shall = _group("m", "S1_mandatory", "S1", _random_matrix(2), variant="shall")
        self.assertNotEqual(must.join_key, shall.join_key)
        self.assertEqual(
            must.join_key,
            mvs.paper.paper_join_key(
                _det_row("m", "S1_mandatory", "S1", variant="must")
            ),
        )


class PairedBootstrapTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260916)
        n = 240
        self.labels = (rng.random(n) < 0.3).astype(int)
        base = rng.random(n) + 0.4 * self.labels
        # Ties are deliberate: the AUROC has to handle them like scikit-learn.
        base[::7] = 0.0
        other = rng.random(n)
        self.scores = np.column_stack([base, base.copy(), other])
        self.seed_keys = [f"S{i // 4:03d}" for i in range(n)]

    def test_rank_auroc_matches_sklearn_with_ties(self) -> None:
        for column in range(3):
            expected = eu.auroc_score(
                self.labels.tolist(), self.scores[:, column].tolist()
            )
            self.assertAlmostEqual(
                mvs.rank_auroc(self.labels, self.scores[:, column]), expected, places=12
            )
        self.assertTrue(
            math.isnan(mvs.rank_auroc(np.ones(5, dtype=int), np.arange(5.0)))
        )

    def test_reference_interval_matches_the_exporter_bootstrap(self) -> None:
        result = mvs.paired_cluster_bootstrap(
            self.labels,
            self.scores,
            self.seed_keys,
            cluster_field="seed_id",
            iterations=150,
            seed=eu.ACSE_CALIBRATION_SEED,
        )
        rows = [
            {"seed_id": key, "label": int(label), "score": float(score)}
            for key, label, score in zip(
                self.seed_keys, self.labels, self.scores[:, 0], strict=True
            )
        ]
        point, low, high = eu.bootstrap_seed_metric(
            rows,
            lambda sample: eu.auroc_score(
                [row["label"] for row in sample], [row["score"] for row in sample]
            ),
            cluster_field="seed_id",
            iterations=150,
            seed=eu.ACSE_CALIBRATION_SEED,
        )
        self.assertAlmostEqual(result.point[0], point, places=12)
        self.assertAlmostEqual(result.interval(0)[0], low, places=12)
        self.assertAlmostEqual(result.interval(0)[1], high, places=12)
        self.assertEqual(result.n_clusters, 60)

    def test_identical_settings_have_a_zero_paired_difference(self) -> None:
        result = mvs.paired_cluster_bootstrap(
            self.labels,
            self.scores,
            self.seed_keys,
            cluster_field="seed_id",
            iterations=100,
            seed=1,
        )
        self.assertEqual(result.delta_interval(1, 0), (0.0, 0.0, 100))
        self.assertTrue(np.array_equal(result.draws[:, 0], result.draws[:, 1]))
        low, high, defined = result.delta_interval(2, 0)
        self.assertEqual(defined, 100)
        self.assertLess(low, high)
        # An independent resample of the other column would not line up draw
        # by draw with the reference column.
        alone = mvs.paired_cluster_bootstrap(
            self.labels,
            self.scores[:, [2]],
            self.seed_keys,
            cluster_field="seed_id",
            iterations=100,
            seed=1,
        )
        self.assertTrue(np.array_equal(alone.draws[:, 0], result.draws[:, 2]))

    def test_cluster_keys_follow_first_appearance(self) -> None:
        keys, rows = mvs.cluster_keys_in_order(["b", "a", "b", "c", "a"])
        self.assertEqual(keys.tolist(), ["b", "a", "c"])
        self.assertEqual(rows["b"].tolist(), [0, 2])
        self.assertEqual(rows["a"].tolist(), [1, 4])


class EmbeddingCacheTest(unittest.TestCase):
    def _write_cache(
        self, root: Path, texts: list[str], *, misalign: bool = False
    ) -> Path:
        artifact_dir = root / "cache"
        artifact_dir.mkdir()
        embeddings = _random_matrix(7, rows=len(texts), columns=4).astype(np.float32)
        item_ids = ["S1_mandatory"] * len(texts)
        np.savez_compressed(
            artifact_dir / "task2_acse_sample_embeddings.npz",
            embeddings=embeddings,
            item_ids=np.asarray(item_ids, dtype=str),
            sample_indices=np.asarray([str(i) for i in range(len(texts))], dtype=str),
        )
        rows = [
            {
                "embedding_index": (i + 1) % len(texts) if misalign else i,
                "run_id": "run-m",
                "model": "m",
                "item_id": "S1_mandatory",
                "sample_index": i,
                "semantic_text": text,
            }
            for i, text in enumerate(texts)
        ]
        eu.write_csv_rows(artifact_dir / "task2_acse_samples.csv", rows)
        (artifact_dir / "manifest.json").write_text(
            json.dumps({"embedding_backend": BACKEND, "run_id": "run-m", "model": "m"}),
            encoding="utf-8",
        )
        return artifact_dir

    def test_cache_binds_vectors_to_matching_texts_only(self) -> None:
        with TemporaryDirectory() as temp:
            texts = [f"modality: mandatory requirement: text {i}" for i in range(3)]
            artifact_dir = self._write_cache(Path(temp), texts)
            cache = mvs.load_embedding_cache(artifact_dir, BACKEND)
            group = _group("m", "S1_mandatory", "S1", None, valid_n=3)
            mvs.attach_cached_embeddings([group], cache)
            self.assertEqual(group.embeddings.shape, (3, 4))
            self.assertEqual(group.embeddings.dtype, np.float64)
            mvs.check_cache_fully_used(cache)

            changed = _group("m", "S1_mandatory", "S1", None, valid_n=3)
            changed.texts[1] = "modality: optional requirement: something else"
            with self.assertRaises(ValueError):
                mvs.attach_cached_embeddings(
                    [changed], mvs.load_embedding_cache(artifact_dir, BACKEND)
                )

            partial = _group("m", "S1_mandatory", "S1", None, valid_n=2)
            fresh = mvs.load_embedding_cache(artifact_dir, BACKEND)
            mvs.attach_cached_embeddings([partial], fresh)
            with self.assertRaises(ValueError):
                mvs.check_cache_fully_used(fresh)

    def test_cache_with_another_backend_or_misaligned_rows_is_rejected(self) -> None:
        with TemporaryDirectory() as temp:
            texts = [f"text {i}" for i in range(3)]
            artifact_dir = self._write_cache(Path(temp), texts)
            with self.assertRaises(ValueError):
                mvs.load_embedding_cache(artifact_dir, "mlx:another-encoder")
        with TemporaryDirectory() as temp:
            artifact_dir = self._write_cache(Path(temp), texts, misalign=True)
            with self.assertRaises(ValueError):
                mvs.load_embedding_cache(artifact_dir, BACKEND)


class SnapshotTest(unittest.TestCase):
    def test_snapshot_must_name_every_model_in_every_cell(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "prov.json"
            path.write_text(
                json.dumps(
                    {
                        "models_cohort": ["a", "b"],
                        "sampling_plan_source": "planned",
                        "cells": [
                            {
                                "dataset": "nice",
                                "variant": "must",
                                "run_ids": {"a": "r1"},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                mvs.load_snapshot(path)


class SettingsPathTest(unittest.TestCase):
    def test_settings_paths_are_recorded_relative_to_the_root(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.assertEqual(
                mvs._relative(root, root / "outputs" / "x.json"), "outputs/x.json"
            )
            self.assertEqual(
                mvs._relative(root, "/elsewhere/x.json"), "/elsewhere/x.json"
            )


if __name__ == "__main__":
    unittest.main()
