"""High-value contracts for semantic-UQ caches and diagnostic probes.

These tests stay offline: the real embedding backend is either TF-IDF or a small
boundary double, and every artifact is written below a temporary directory.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import (
    compute_acse_semantic_artifacts as semantic_cache,
    diagnose_embedding_separability as embedding_diagnostic,
    eval_utils as eu,
    plot_acse_global_embedding_projection as global_projection,
    probe_acse_embedding_separability as separability_probe,
)


def _benchmark_item() -> dict[str, object]:
    seed = {
        "seed_id": "S0001",
        "source_dataset": "NICE",
        "original_requirement": "The system shall export reports.",
        "capability_text_final": "export reports",
    }
    return next(
        row
        for row in eu.build_benchmark_items([seed])
        if row["source_modality"] == "optional"
    )


class SemanticCachePipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)
        self.analysis_dir = self.root / "outputs/evaluation_fixture"
        self.analysis_dir.mkdir(parents=True)
        self.item = _benchmark_item()
        eu.write_csv_rows(self.root / "data/processed/benchmark_items.csv", [self.item])
        self._write_raw_rows_and_scores()
        self.run = semantic_cache.CompletedRun(
            analysis_dir=self.analysis_dir,
            dataset_id="nice",
            variant="must",
            run_id="full-1",
            model="model-1",
            profile="profile-1",
            semantic_embedding_backend="tfidf",
        )

    def _raw_row(
        self,
        sample_kind: str,
        sample_index: int,
        *,
        modality: str = "optional",
        parse_status: str = "ok",
    ) -> dict[str, object]:
        modal = {
            "mandatory": "MUST",
            "recommended": "SHOULD",
            "optional": "MAY",
        }[modality]
        parsed_json = (
            {
                "requirement": f"The system {modal} export reports.",
                "modality": modality,
                "confidence": 0.9,
            }
            if parse_status == "ok"
            else None
        )
        return {
            "run_id": "full-1",
            "run_group_id": "group-1",
            "provider_id": "provider-1",
            "profile_id": "profile-1",
            "model": "model-1",
            "task": "task2",
            "item_id": self.item["item_id"],
            "seed_id": self.item["seed_id"],
            "source_modality": self.item["source_modality"],
            "sample_index": sample_index,
            "sample_kind": sample_kind,
            "parse_status": parse_status,
            "parsed_json": parsed_json,
            "confidence_scale": eu.CONFIDENCE_SCALE_0_1,
        }

    def _write_raw_rows_and_scores(self) -> None:
        # Scrambled sample order and one failed sample exercise ordering and counts.
        raw_rows = [
            self._raw_row("deterministic", 0),
            self._raw_row("stochastic", 2, modality="mandatory"),
            self._raw_row("stochastic", 0),
            self._raw_row("stochastic", 3, parse_status="invalid_json"),
            self._raw_row("stochastic", 1, modality="recommended"),
        ]
        raw_path = self.root / "data/processed/model_outputs_raw.jsonl"
        for row in raw_rows:
            eu.append_jsonl(raw_path, row)

        eu.write_csv_rows(
            self.analysis_dir / "uq_scores.csv",
            [
                {
                    "run_id": "full-1",
                    "model": "model-1",
                    "task": "task2",
                    "uq_method": "verbalized_confidence",
                    "item_id": self.item["item_id"],
                    "confidence": 0.9,
                    "text_modality": "optional",
                    "text_modality_basis": "explicit_modal",
                    "text_modality_parse_status": "ok",
                    "text_modality_correct": True,
                    "label_text_consistent": True,
                    "text_overcommit": False,
                    "text_undercommit": False,
                    "strict_text_overcommit": False,
                    "text_high_conf_overcommit_80": False,
                    "text_high_conf_overcommit_90": False,
                    "strict_text_high_conf_overcommit_80": False,
                    "strict_text_high_conf_overcommit_90": False,
                    "label_correct_text_overcommit_80": False,
                    "label_correct_text_overcommit_90": False,
                }
            ],
        )

    def _compute(self, distance_threshold: float = 0.2) -> dict[str, object]:
        return semantic_cache.compute_run_backend(
            self.root,
            self.run,
            embedding_backend="tfidf",
            mlx_model_name=None,
            distance_threshold=distance_threshold,
            embedding_batch_size=16,
            force=False,
        )

    def _cache_dir(self, manifest: dict[str, object]) -> Path:
        return eu.acse_semantic_cache_dir(
            self.analysis_dir, str(manifest["embedding_backend"])
        )

    def test_compute_run_backend_orders_samples_and_counts_failed_attempts(
        self,
    ) -> None:
        manifest = self._compute()
        cache_dir = self._cache_dir(manifest)

        self.assertEqual(manifest["status"], "computed")
        self.assertEqual(manifest["stochastic_sample_rows"], 3)
        self.assertEqual(manifest["item_rows"], 1)
        samples = eu.read_csv_rows(cache_dir / "task2_acse_samples.csv")
        self.assertEqual([row["sample_index"] for row in samples], ["0", "1", "2"])
        self.assertEqual(
            [row["pred_modality"] for row in samples],
            ["optional", "recommended", "mandatory"],
        )
        items = eu.read_csv_rows(cache_dir / "task2_acse_items.csv")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["valid_n"], "3")
        self.assertEqual(items[0]["total_n"], "4")
        self.assertEqual(items[0]["parse_failures"], "1")

    def test_compute_run_backend_carries_deterministic_text_diagnostics(self) -> None:
        manifest = self._compute()
        items = eu.read_csv_rows(self._cache_dir(manifest) / "task2_acse_items.csv")

        self.assertEqual(
            items[0]["task2_requirement"], "The system MAY export reports."
        )
        self.assertEqual(items[0]["text_modality"], "optional")
        self.assertEqual(items[0]["text_modality_parse_status"], "ok")

    def test_compute_run_backend_writes_float32_vectors_with_row_identities(
        self,
    ) -> None:
        manifest = self._compute()
        cache_dir = self._cache_dir(manifest)

        self.assertEqual(manifest["embedding_dtype"], "float32")
        expected_artifacts = {
            "task2_acse_sample_embeddings.npz",
            "task2_acse_samples.csv",
            "task2_acse_items.csv",
            "task2_acse_scores.csv",
            "task2_acse_normalized_scores.csv",
            "task2_acse_calibration.csv",
        }
        artifacts = [Path(path) for path in manifest["artifacts"]]
        self.assertEqual({path.name for path in artifacts}, expected_artifacts)
        self.assertEqual(len(artifacts), len(expected_artifacts))
        self.assertTrue(all(path.is_file() for path in artifacts))
        samples = eu.read_csv_rows(cache_dir / "task2_acse_samples.csv")
        self.assertTrue(
            all(row["cluster_label"] != "" for row in samples),
            "every cached vector must have the cluster label used by the ACSE score",
        )
        with np.load(
            cache_dir / "task2_acse_sample_embeddings.npz", allow_pickle=False
        ) as cached:
            embeddings = cached["embeddings"]
            self.assertEqual(embeddings.dtype, np.float32)
            self.assertEqual(list(embeddings.shape), manifest["embedding_shape"])
            self.assertEqual(cached["sample_indices"].tolist(), ["0", "1", "2"])
            self.assertEqual(cached["item_ids"].tolist(), [self.item["item_id"]] * 3)

    @unittest.expectedFailure
    def test_cache_is_recomputed_when_distance_threshold_changes(self) -> None:
        """Known bug: cache reuse currently ignores score-defining parameters."""
        first = self._compute(distance_threshold=0.2)
        cache_dir = self._cache_dir(first)
        first_samples = eu.read_csv_rows(cache_dir / "task2_acse_samples.csv")
        first_items = eu.read_csv_rows(cache_dir / "task2_acse_items.csv")
        first_labels = [row["cluster_label"] for row in first_samples]
        first_score = float(first_items[0]["acse_uncertainty_score"])

        second = self._compute(distance_threshold=0.5)

        self.assertEqual(first["distance_threshold"], 0.2)
        self.assertEqual(second["status"], "computed")
        self.assertEqual(
            second["distance_threshold"],
            0.5,
            "a cache computed at another threshold must be recomputed",
        )
        persisted_manifest = json.loads(
            (cache_dir / "manifest.json").read_text(encoding="utf-8")
        )
        second_samples = eu.read_csv_rows(cache_dir / "task2_acse_samples.csv")
        second_items = eu.read_csv_rows(cache_dir / "task2_acse_items.csv")
        self.assertEqual(persisted_manifest["distance_threshold"], 0.5)
        self.assertNotEqual(
            [row["cluster_label"] for row in second_samples], first_labels
        )
        self.assertNotAlmostEqual(
            float(second_items[0]["acse_uncertainty_score"]), first_score
        )

    def test_embedding_batches_preserve_order_and_backend_identity(self) -> None:
        calls: list[list[str]] = []
        call_parameters: list[tuple[str, str | None]] = []

        def fake_embedding_matrix(
            texts: list[str], *, embedding_backend: str, mlx_model_name: str | None
        ) -> tuple[np.ndarray, str]:
            calls.append(list(texts))
            call_parameters.append((embedding_backend, mlx_model_name))
            values = [[float(ord(text)), float(len(text))] for text in texts]
            return np.asarray(values), "mlx:fixture/model"

        with mock.patch.object(
            semantic_cache.eu,
            "semantic_embedding_matrix",
            side_effect=fake_embedding_matrix,
        ):
            matrix, label = semantic_cache.embedding_matrix_for_cache(
                ["a", "b", "c", "d", "e"],
                embedding_backend="mlx",
                mlx_model_name="fixture/model",
                batch_size=2,
            )

        self.assertTrue(all(0 < len(batch) <= 2 for batch in calls))
        self.assertEqual([text for batch in calls for text in batch], list("abcde"))
        self.assertEqual(call_parameters, [("mlx", "fixture/model")] * 3)
        self.assertEqual(label, "mlx:fixture/model")
        np.testing.assert_array_equal(
            matrix[:, 0], np.asarray([ord(char) for char in "abcde"], dtype=float)
        )

    def test_embedding_batches_reject_a_backend_change_mid_run(self) -> None:
        responses = [
            (np.ones((2, 2)), "mlx:fixture/model"),
            (np.ones((1, 2)), "mlx:different/model"),
        ]
        with (
            mock.patch.object(
                semantic_cache.eu,
                "semantic_embedding_matrix",
                side_effect=responses,
            ),
            self.assertRaisesRegex(
                RuntimeError, "Embedding backend changed within run"
            ),
        ):
            semantic_cache.embedding_matrix_for_cache(
                ["a", "b", "c"],
                embedding_backend="mlx",
                mlx_model_name="fixture/model",
                batch_size=2,
            )


class GlobalEmbeddingCacheIntegrityTest(unittest.TestCase):
    def _cache_fixture(
        self,
        root: Path,
        *,
        embedded_item_ids: list[str],
        csv_item_ids: list[str],
        embedded_sample_indices: list[str] | None = None,
        csv_sample_indices: list[str] | None = None,
    ) -> dict[str, str]:
        artifact_dir = root / "cache"
        artifact_dir.mkdir()
        if embedded_sample_indices is None:
            embedded_sample_indices = ["0"] * len(embedded_item_ids)
        if csv_sample_indices is None:
            csv_sample_indices = ["0"] * len(csv_item_ids)
        np.savez_compressed(
            artifact_dir / "task2_acse_sample_embeddings.npz",
            embeddings=np.arange(len(embedded_item_ids) * 2, dtype=np.float32).reshape(
                len(embedded_item_ids), 2
            ),
            item_ids=np.asarray(embedded_item_ids),
            sample_indices=np.asarray(embedded_sample_indices),
        )
        eu.write_csv_rows(
            artifact_dir / "task2_acse_samples.csv",
            [
                {
                    "embedding_index": index,
                    "item_id": item_id,
                    "seed_id": f"S{index:04d}",
                    "sample_index": csv_sample_indices[index],
                    "pred_modality": "optional",
                    "semantic_text": f"requirement: {item_id}",
                    "requirement": f"The system MAY {item_id}.",
                }
                for index, item_id in enumerate(csv_item_ids)
            ],
        )
        eu.write_csv_rows(
            artifact_dir / "task2_acse_items.csv",
            [
                {
                    "item_id": item_id,
                    "source_modality": "optional",
                    "strict_text_overcommit": False,
                    "text_overcommit": False,
                    "acse_uncertainty_score": 0.1,
                }
                for item_id in csv_item_ids
            ],
        )
        return {
            "artifact_dir": str(artifact_dir),
            "dataset_id": "nice",
            "benchmark_variant": "must",
            "run_id": "full-1",
            "model": "model-1",
            "profile_id": "profile-1",
            "embedding_backend": "mlx:fixture/model",
        }

    def test_loader_rejects_embedding_and_sample_row_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            row = self._cache_fixture(
                Path(tmpdir), embedded_item_ids=["A"], csv_item_ids=["A", "B"]
            )
            with self.assertRaisesRegex(ValueError, "Embedding/sample row mismatch"):
                global_projection.load_embeddings_and_rows([row])

    @unittest.expectedFailure
    def test_loader_rejects_same_length_cache_with_reordered_sample_ids(self) -> None:
        """Known bug: NPZ identity arrays are written but never validated on load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            row = self._cache_fixture(
                Path(tmpdir), embedded_item_ids=["A", "B"], csv_item_ids=["B", "A"]
            )
            with self.assertRaisesRegex(ValueError, "identity|order|item_id"):
                global_projection.load_embeddings_and_rows([row])

    @unittest.expectedFailure
    def test_loader_rejects_reordered_samples_within_one_item(self) -> None:
        """Known bug: cached sample indices are not checked against CSV order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            row = self._cache_fixture(
                Path(tmpdir),
                embedded_item_ids=["A", "A"],
                csv_item_ids=["A", "A"],
                embedded_sample_indices=["0", "1"],
                csv_sample_indices=["1", "0"],
            )
            with self.assertRaisesRegex(ValueError, "identity|order|sample_index"):
                global_projection.load_embeddings_and_rows([row])


class EmbeddingDiagnosticContractTest(unittest.TestCase):
    @staticmethod
    def _fake_embedding_matrix(
        calls: list[list[str]], texts: list[str]
    ) -> tuple[np.ndarray, str]:
        calls.append(list(texts))
        matrix = np.asarray(
            [[float(len(text)), float(ord(text[0]))] for text in texts],
            dtype=np.float32,
        )
        return matrix, "mlx:fixture/model"

    def test_requirement_embedding_cache_invalidates_same_length_changed_text(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "requirements.npz"
            calls: list[list[str]] = []
            call_parameters: list[dict[str, object]] = []

            def side_effect(
                texts: list[str], **kwargs: object
            ) -> tuple[np.ndarray, str]:
                call_parameters.append(kwargs)
                return self._fake_embedding_matrix(calls, texts)

            with mock.patch.object(
                embedding_diagnostic.eu,
                "semantic_embedding_matrix",
                side_effect=side_effect,
            ):
                first = embedding_diagnostic.embed_requirement_only(
                    ["beta", "alpha", "beta"],
                    batch_size=8,
                    cache_path=cache_path,
                    reuse_cache=True,
                )
                reused = embedding_diagnostic.embed_requirement_only(
                    ["beta", "alpha", "beta"],
                    batch_size=8,
                    cache_path=cache_path,
                    reuse_cache=True,
                )
                changed = embedding_diagnostic.embed_requirement_only(
                    ["beta", "gamma", "beta"],
                    batch_size=8,
                    cache_path=cache_path,
                    reuse_cache=True,
                )

        self.assertEqual(len(calls), 2)
        self.assertEqual(call_parameters, [{"embedding_backend": "mlx"}] * 2)
        self.assertEqual(set(calls[0]), {"alpha", "beta"})
        self.assertEqual(set(calls[1]), {"beta", "gamma"})
        np.testing.assert_array_equal(first, reused)
        self.assertFalse(np.array_equal(first[1], changed[1]))

    def test_requirement_embedding_deduplicates_but_restores_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            calls: list[list[str]] = []
            call_parameters: list[dict[str, object]] = []

            def side_effect(
                texts: list[str], **kwargs: object
            ) -> tuple[np.ndarray, str]:
                call_parameters.append(kwargs)
                return self._fake_embedding_matrix(calls, texts)

            with mock.patch.object(
                embedding_diagnostic.eu,
                "semantic_embedding_matrix",
                side_effect=side_effect,
            ):
                matrix = embedding_diagnostic.embed_requirement_only(
                    ["beta", "alpha", "beta"],
                    batch_size=8,
                    cache_path=Path(tmpdir) / "requirements.npz",
                    reuse_cache=False,
                )

        self.assertEqual(sum(len(batch) for batch in calls), 2)
        self.assertEqual(call_parameters, [{"embedding_backend": "mlx"}])
        self.assertEqual({text for batch in calls for text in batch}, {"alpha", "beta"})
        self.assertEqual(matrix.dtype, np.float32)
        np.testing.assert_array_equal(matrix[0], matrix[2])
        self.assertFalse(np.array_equal(matrix[0], matrix[1]))

    def test_grid_summary_reports_auprc_lift_over_prevalence(self) -> None:
        fold_rows = []
        for fold, (auprc, baseline) in enumerate([(0.4, 0.2), (0.6, 0.3)]):
            fold_rows.append(
                {
                    "feature_key": "tfidf::reqonly",
                    "group_mode": "item",
                    "scope": "global",
                    "target": "sample_strict_text_overcommit",
                    "model": "logreg",
                    "fold": fold,
                    "n_test": 10,
                    "accuracy": 0.7,
                    "balanced_accuracy": 0.65,
                    "macro_f1": 0.6,
                    "auroc_macro": 0.75,
                    "average_precision_macro": auprc,
                    "baseline_average_precision": baseline,
                    "positive_rate_test": baseline,
                }
            )

        summary = embedding_diagnostic.summarize_grid(fold_rows)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["feature_backend"], "tfidf")
        self.assertEqual(summary[0]["text_variant"], "reqonly")
        self.assertAlmostEqual(summary[0]["auprc_mean"], 0.5)
        self.assertAlmostEqual(summary[0]["baseline_auprc"], 0.25)
        self.assertEqual(summary[0]["auprc_lift_over_baseline"], 2.0)

    @unittest.expectedFailure
    def test_fold_metrics_skips_a_scope_with_only_one_unique_group(self) -> None:
        """Known bug: the split count becomes one and scikit-learn raises late."""
        rows = separability_probe.fold_metrics(
            X=np.asarray([[0.0], [1.0]]),
            y_raw=np.asarray([0, 1]),
            groups=np.asarray(["same", "same"]),
            target="sample_strict_text_overcommit",
            model_name="logreg",
            scope="global",
            n_splits=3,
            random_state=7,
        )
        self.assertEqual(rows, [], "a one-group scope cannot form a held-out fold")


if __name__ == "__main__":
    unittest.main()
