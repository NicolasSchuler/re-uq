"""Contracts for the embedding-separability probe's metrics and leak-free substrate.

Two things are pinned here:

1. **Probe metrics.** Two-class targets must reach scikit-learn's binary AUROC and
   an explicit two-column macro average precision instead of collapsing to a blank
   cell, while three or more classes keep the macro one-vs-rest path. Estimator
   column order is never assumed.
2. **Leak-free substrate.** The primary strengthening diagnostic runs on
   requirement-only text with PCA fitted inside each cross-validation fold, so a
   held-out fold never contributes to its own projection. The label-prefixed text
   survives only as an explicitly named positive control.
"""

from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from scripts import (
    plot_embedding_diagnostic_figure_v2 as figure_v2,
    probe_acse_embedding_separability as separability_probe,
)


def _grouped_binary_fixture(
    n_groups: int = 12, per_group: int = 6, n_features: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A grouped, class-balanced probe fixture with a unique marker per row.

    Column 0 is a unique row id, so a recorded PCA fit matrix can be traced back to
    the exact rows it saw; the remaining columns carry noise.
    """
    rng = np.random.default_rng(20260904)
    n_rows = n_groups * per_group
    X = np.empty((n_rows, n_features), dtype=float)
    X[:, 0] = np.arange(n_rows, dtype=float)
    X[:, 1:] = rng.normal(size=(n_rows, n_features - 1))
    y = np.asarray([index % 2 for index in range(n_rows)], dtype=int)
    groups = np.asarray(
        [f"g{index // per_group:02d}" for index in range(n_rows)], dtype=object
    )
    return X, y, groups


class ProbeMetricAdapterTest(unittest.TestCase):
    """`probe_auroc` / `probe_average_precision` dispatch on the class count."""

    def test_two_class_auroc_uses_the_binary_api(self) -> None:
        y_true = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray(
            [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], dtype=float
        )
        classes = np.asarray([0, 1])

        value = separability_probe.probe_auroc(y_true, probabilities, classes)

        # The multiclass wrapper used to raise here and report a blank cell.
        self.assertEqual(value, 1.0)
        self.assertEqual(value, roc_auc_score(y_true, probabilities[:, 1]))

    def test_two_class_average_precision_macro_averages_both_columns(self) -> None:
        y_true = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray(
            [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]], dtype=float
        )
        classes = np.asarray([0, 1])

        value = separability_probe.probe_average_precision(
            y_true, probabilities, classes
        )

        self.assertEqual(value, 1.0)
        one_hot = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]])
        self.assertAlmostEqual(
            value, average_precision_score(one_hot, probabilities, average="macro")
        )

    def test_two_class_metrics_are_not_swallowed_into_nan(self) -> None:
        """Every legitimate two-class fold reports a number, never a blank cell."""
        y_true = np.asarray([0, 1, 0, 1, 1])
        probabilities = np.asarray(
            [[0.7, 0.3], [0.4, 0.6], [0.55, 0.45], [0.3, 0.7], [0.45, 0.55]],
            dtype=float,
        )
        classes = np.asarray([0, 1])

        for metric in (
            separability_probe.probe_auroc,
            separability_probe.probe_average_precision,
        ):
            with self.subTest(metric=metric.__name__):
                self.assertFalse(math.isnan(metric(y_true, probabilities, classes)))

    def test_imbalanced_two_class_metrics_match_scikit_learn(self) -> None:
        y_true = np.asarray([0] * 18 + [1, 1])
        rng = np.random.default_rng(7)
        positive = np.concatenate(
            [rng.uniform(0.0, 0.4, size=18), rng.uniform(0.6, 1.0, size=2)]
        )
        probabilities = np.column_stack([1.0 - positive, positive])
        classes = np.asarray([0, 1])

        self.assertAlmostEqual(
            separability_probe.probe_auroc(y_true, probabilities, classes),
            roc_auc_score(y_true, positive),
        )
        one_hot = np.column_stack([1 - y_true, y_true])
        self.assertAlmostEqual(
            separability_probe.probe_average_precision(y_true, probabilities, classes),
            average_precision_score(one_hot, probabilities, average="macro"),
        )

    def test_three_class_metrics_use_the_macro_one_vs_rest_path(self) -> None:
        y_true = np.asarray([0, 1, 2, 0, 1, 2])
        probabilities = np.asarray(
            [
                [0.7, 0.2, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.3, 0.6, 0.1],
                [0.1, 0.3, 0.6],
            ],
            dtype=float,
        )
        classes = np.asarray([0, 1, 2])

        self.assertAlmostEqual(
            separability_probe.probe_auroc(y_true, probabilities, classes),
            roc_auc_score(
                y_true,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            ),
        )
        self.assertEqual(
            separability_probe.probe_average_precision(y_true, probabilities, classes),
            1.0,
        )

    def test_four_class_metrics_use_the_macro_one_vs_rest_path(self) -> None:
        y_true = np.asarray([0, 1, 2, 3, 0, 1, 2, 3])
        probabilities = np.full((8, 4), 0.1, dtype=float)
        probabilities[np.arange(8), y_true] = 0.7
        classes = np.asarray([0, 1, 2, 3])

        self.assertEqual(
            separability_probe.probe_auroc(y_true, probabilities, classes), 1.0
        )
        self.assertEqual(
            separability_probe.probe_average_precision(y_true, probabilities, classes),
            1.0,
        )

    def test_a_class_absent_from_the_held_out_fold_reports_no_metric(self) -> None:
        """Ranking metrics are undefined, not merely awkward, without every class."""
        y_true = np.asarray([0, 0, 0, 0])
        probabilities = np.asarray(
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]], dtype=float
        )
        classes = np.asarray([0, 1])

        self.assertTrue(
            math.isnan(separability_probe.probe_auroc(y_true, probabilities, classes))
        )
        self.assertTrue(
            math.isnan(
                separability_probe.probe_average_precision(
                    y_true, probabilities, classes
                )
            )
        )

    def test_a_three_class_target_missing_one_class_reports_no_metric(self) -> None:
        y_true = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray(
            [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.3, 0.6, 0.1]],
            dtype=float,
        )
        classes = np.asarray([0, 1, 2])

        self.assertTrue(
            math.isnan(separability_probe.probe_auroc(y_true, probabilities, classes))
        )
        self.assertTrue(
            math.isnan(
                separability_probe.probe_average_precision(
                    y_true, probabilities, classes
                )
            )
        )

    def test_a_single_class_target_reports_no_metric(self) -> None:
        y_true = np.asarray([0, 0, 0])
        probabilities = np.ones((3, 1), dtype=float)
        classes = np.asarray([0])

        self.assertTrue(
            math.isnan(separability_probe.probe_auroc(y_true, probabilities, classes))
        )
        self.assertTrue(
            math.isnan(
                separability_probe.probe_average_precision(
                    y_true, probabilities, classes
                )
            )
        )

    def test_constant_probabilities_score_chance_rather_than_failing(self) -> None:
        y_true = np.asarray([0, 0, 1, 1])
        probabilities = np.full((4, 2), 0.5, dtype=float)
        classes = np.asarray([0, 1])

        self.assertEqual(
            separability_probe.probe_auroc(y_true, probabilities, classes), 0.5
        )
        self.assertEqual(
            separability_probe.probe_average_precision(y_true, probabilities, classes),
            0.5,
        )

    def test_constant_probabilities_score_chance_for_three_classes(self) -> None:
        y_true = np.asarray([0, 1, 2, 0, 1, 2])
        probabilities = np.full((6, 3), 1.0 / 3.0, dtype=float)
        classes = np.asarray([0, 1, 2])

        self.assertAlmostEqual(
            separability_probe.probe_auroc(y_true, probabilities, classes), 0.5
        )
        self.assertAlmostEqual(
            separability_probe.probe_average_precision(y_true, probabilities, classes),
            1.0 / 3.0,
        )


class EstimatorColumnOrderTest(unittest.TestCase):
    """Probability columns are keyed by class label, never by position."""

    def test_reordered_estimator_classes_are_realigned(self) -> None:
        probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7]], dtype=float)

        ordered = separability_probe.ordered_class_probabilities(
            probabilities, np.asarray([1, 0]), np.asarray([0, 1])
        )

        np.testing.assert_allclose(ordered, np.asarray([[0.2, 0.8], [0.7, 0.3]]))

    def test_a_class_missing_from_training_leaves_a_zero_column(self) -> None:
        probabilities = np.asarray([[0.6, 0.4], [0.1, 0.9]], dtype=float)

        ordered = separability_probe.ordered_class_probabilities(
            probabilities, np.asarray([0, 2]), np.asarray([0, 1, 2])
        )

        np.testing.assert_allclose(
            ordered, np.asarray([[0.6, 0.0, 0.4], [0.1, 0.0, 0.9]])
        )

    def test_fold_metrics_scores_a_two_class_multiclass_target(self) -> None:
        """`dataset_variant` and friends can hold exactly two labels."""
        rows = [
            {
                "dataset_id": "d0",
                "seed_id": f"s{index // 4:02d}",
                "dataset_variant": "alpha" if index % 2 == 0 else "beta",
            }
            for index in range(48)
        ]
        y_raw = separability_probe.target_values(rows, "dataset_variant")
        separation = np.where(y_raw == "alpha", -1.0, 1.0)
        X = np.column_stack(
            [
                separation
                + np.random.default_rng(3).normal(scale=0.05, size=len(rows)),
                np.zeros(len(rows)),
            ]
        )

        fold_rows = separability_probe.fold_metrics(
            X=X,
            y_raw=y_raw,
            groups=separability_probe.group_values(rows, "seed"),
            target="dataset_variant",
            model_name="logreg",
            scope="global",
            n_splits=3,
            random_state=11,
        )

        self.assertTrue(fold_rows)
        for row in fold_rows:
            with self.subTest(fold=row["fold"]):
                # These cells were blank before the binary dispatch existed.
                self.assertNotEqual(row["auroc_macro"], "")
                self.assertNotEqual(row["average_precision_macro"], "")
                self.assertGreater(float(row["auroc_macro"]), 0.9)


class FoldLocalProjectionTest(unittest.TestCase):
    """PCA belongs inside the fold: a held-out row must not shape its own axes."""

    def test_pca_is_fitted_only_on_the_training_rows_of_each_fold(self) -> None:
        X, y, groups = _grouped_binary_fixture()
        fitted_on: list[np.ndarray] = []

        class RecordingPCA(PCA):
            def fit(self, X, y=None):
                fitted_on.append(np.array(X, copy=True))
                return super().fit(X, y)

            def fit_transform(self, X, y=None):
                fitted_on.append(np.array(X, copy=True))
                return super().fit_transform(X, y)

        with mock.patch.object(separability_probe, "PCA", RecordingPCA):
            fold_rows = separability_probe.fold_metrics(
                X=X,
                y_raw=y,
                groups=groups,
                target="sample_strict_text_overcommit",
                model_name="logreg",
                scope="global",
                n_splits=3,
                random_state=5,
                pca_components=3,
            )

        self.assertEqual(len(fold_rows), 3)
        self.assertEqual(len(fitted_on), len(fold_rows))
        all_ids = set(X[:, 0].tolist())
        for row, seen in zip(fold_rows, fitted_on, strict=True):
            with self.subTest(fold=row["fold"]):
                seen_ids = set(seen[:, 0].tolist())
                # Exactly the training rows: the ids are unique, the count matches
                # n_train, and the complement is the held-out fold, so no test row
                # entered the projection that later transforms it.
                self.assertEqual(len(seen_ids), row["n_train"])
                self.assertEqual(len(all_ids - seen_ids), row["n_test"])

    def test_fold_projection_differs_from_a_globally_fitted_projection(self) -> None:
        """Guards against a projection that is fold-local in name only."""
        X, y, groups = _grouped_binary_fixture()
        fold_components: list[np.ndarray] = []

        class CapturingPCA(PCA):
            def fit_transform(self, X, y=None):
                transformed = super().fit_transform(X, y)
                fold_components.append(np.array(self.components_, copy=True))
                return transformed

        with mock.patch.object(separability_probe, "PCA", CapturingPCA):
            separability_probe.fold_metrics(
                X=X,
                y_raw=y,
                groups=groups,
                target="sample_strict_text_overcommit",
                model_name="logreg",
                scope="global",
                n_splits=3,
                random_state=5,
                pca_components=3,
            )

        global_pca = PCA(n_components=3, svd_solver="randomized", random_state=5)
        global_pca.fit(X)
        self.assertTrue(fold_components)
        for index, components in enumerate(fold_components):
            with self.subTest(fold=index):
                self.assertFalse(np.allclose(components, global_pca.components_))

    def test_components_are_clamped_to_the_training_fold(self) -> None:
        """A fold smaller than the requested rank must not blow up the probe."""
        X, y, groups = _grouped_binary_fixture()

        fold_rows = separability_probe.fold_metrics(
            X=X,
            y_raw=y,
            groups=groups,
            target="sample_strict_text_overcommit",
            model_name="logreg",
            scope="global",
            n_splits=3,
            random_state=5,
            pca_components=4096,
        )

        self.assertEqual(len(fold_rows), 3)
        for row in fold_rows:
            with self.subTest(fold=row["fold"]):
                self.assertLessEqual(row["pca_components_fold"], X.shape[1])
                self.assertLessEqual(row["pca_components_fold"], row["n_train"])

    def test_estimators_are_unprojected_when_no_components_are_requested(self) -> None:
        """`diagnose_embedding_separability` passes already-reduced features."""
        self.assertNotIsInstance(separability_probe.make_estimator("hgb", 0), Pipeline)
        plain_logreg = separability_probe.make_estimator("logreg", 0)
        self.assertFalse(any(isinstance(step, PCA) for _, step in plain_logreg.steps))

        projected = separability_probe.make_estimator("hgb", 0, pca_components=4)
        self.assertIsInstance(projected.steps[0][1], PCA)
        self.assertEqual(projected.steps[0][1].n_components, 4)


class TextConditionTest(unittest.TestCase):
    """The prefixed substrate is only ever offered as a named leakage control."""

    def test_requirement_only_is_the_primary_condition(self) -> None:
        self.assertEqual(
            separability_probe.PRIMARY_TEXT_CONDITION,
            separability_probe.REQUIREMENT_ONLY_CONDITION,
        )
        self.assertEqual(
            separability_probe.TEXT_CONDITION_ROLES[
                separability_probe.REQUIREMENT_ONLY_CONDITION
            ],
            "primary",
        )

    def test_the_prefixed_condition_is_named_and_roled_as_a_leakage_control(
        self,
    ) -> None:
        name = separability_probe.PREFIXED_CONTROL_CONDITION
        self.assertIn("leakage_control", name)
        self.assertEqual(
            separability_probe.TEXT_CONDITION_ROLES[name],
            "positive_control_label_leakage",
        )

    def test_fold_rows_carry_the_condition_and_its_role(self) -> None:
        row = separability_probe.stamp_text_condition(
            {"scope": "global"}, separability_probe.PREFIXED_CONTROL_CONDITION
        )

        self.assertEqual(
            row["text_condition"], separability_probe.PREFIXED_CONTROL_CONDITION
        )
        self.assertEqual(row["text_condition_role"], "positive_control_label_leakage")
        self.assertEqual(row["scope"], "global")


class DiagnosticFigureConditionTest(unittest.TestCase):
    """The paper figure's target bars read the de-circularized substrate."""

    def test_target_bars_select_requirement_only_text(self) -> None:
        self.assertTrue(figure_v2.TARGET_BARS)
        for spec in figure_v2.TARGET_BARS:
            with self.subTest(bar=spec["label"]):
                self.assertEqual(spec["text"], "reqonly")

    def test_context_bars_stay_on_requirement_only_text(self) -> None:
        for spec in figure_v2.CONTEXT_BARS:
            with self.subTest(bar=spec["label"]):
                self.assertEqual(spec["text"], "reqonly")

    def test_any_prefixed_bar_is_labelled_a_leakage_control(self) -> None:
        shown = figure_v2.CONTEXT_BARS + figure_v2.TARGET_BARS + figure_v2.CONTROL_BARS
        prefixed = [spec for spec in shown if spec["text"] == "prefixed"]

        self.assertTrue(prefixed, "the positive control should stay visible")
        for spec in prefixed:
            with self.subTest(bar=spec["label"]):
                self.assertIn("leakage control", spec["label"].lower())


if __name__ == "__main__":
    unittest.main()
