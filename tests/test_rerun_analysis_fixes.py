"""Synthetic verification only; no experiment results or model/embedding APIs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from scripts import (
    compare_context_ablation as context,
    diagnose_embedding_separability as diagnostic,
    eval_utils as eu,
    plot_embedding_diagnostic_figure_v2 as figure,
    probe_acse_embedding_separability as probe,
)
from tests import test_paper_exports as paper_fixtures


def item(seed, modality, strengthened=False, **extra):
    return {
        "source_identity": seed,
        "seed_id": seed,
        "model": "fixture-model",
        "dataset_id": "pure",
        "benchmark_variant": "must",
        "gold_modality": modality,
        "pred_modality": modality,
        "text_modality_parse_status": "ok",
        "response_status": "ok",
        "strict_text_overcommit": strengthened,
        "text_overcommit": strengthened,
        **extra,
    }


class ExactPairingTest(unittest.TestCase):
    def compare(self, a, b):
        return {
            r["metric"]: r
            for r in context.delta_rows(
                "fixture-model", "all", a, b, bootstrap_samples=100
            )
        }

    def test_different_failed_variants_cannot_form_seed_pair(self):
        a = [
            item("s", "optional", True),
            item("s", "recommended", response_status="request_error"),
        ]
        b = [
            item("s", "optional", response_status="request_error"),
            item("s", "recommended", True),
        ]
        for row in self.compare(a, b).values():
            self.assertEqual(row["n_matched_items"], 0)
            self.assertEqual(row["n_excluded_ineligible_items"], 2)
            self.assertEqual(row["n_bare_failed_items"], 1)
            self.assertEqual(row["delta"], "")
            self.assertEqual(row["bare"], "")

    def test_metric_specific_cohorts_and_hand_computable_difference(self):
        a = [
            item("a", "optional"),
            item("b", "optional", True),
            item("c", "optional", True),
        ]
        b = [
            item("a", "optional", True),
            item("b", "optional", text_modality_parse_status="unknown"),
            item("c", "optional", True, pred_modality="mandatory"),
        ]
        rates = self.compare(a, b)
        text = rates["strict_text_strengthening"]
        self.assertEqual((text["bare"], text["document"], text["delta"]), (0.5, 1, 0.5))
        self.assertEqual(text["n_matched_items"], 2)
        self.assertEqual(text["n_document_unclassified_items"], 1)
        self.assertAlmostEqual(text["full_arm_bare_descriptive"], 2 / 3)
        label = rates["label_accuracy"]
        self.assertEqual(label["n_matched_items"], 3)
        self.assertAlmostEqual(label["delta"], -1 / 3)
        self.assertEqual(text["delta_cluster_field"], "seed_id")
        self.assertIn("missing request", text["cluster_note"])

    def test_unmatched_duplicates_missing_identity_and_variant_isolation(self):
        a = [
            item("s", "optional"),
            item("duplicate", "optional"),
            item("duplicate", "optional"),
            item("orphan", "optional"),
            item("", "optional"),
            item("variant", "optional"),
        ]
        b = [
            item("s", "optional", True),
            item("duplicate", "optional"),
            item("variant", "optional", benchmark_variant="shall"),
        ]
        row = self.compare(a, b)["label_accuracy"]
        self.assertEqual(row["n_matched_items"], 1)
        self.assertEqual(row["n_duplicate_identities"], 1)
        self.assertEqual(row["n_bare_duplicate_rows"], 2)
        self.assertEqual(row["n_missing_identity_rows"], 1)
        self.assertEqual(row["n_unmatched_items"], 3)
        self.assertEqual(row["delta"], 0)

    def test_crossing_request_partitions_preserve_all_dependence(self):
        a = [item(str(i), "optional", batch_id=f"a{i // 2}") for i in range(4)]
        b = [item(str(i), "optional", True, batch_id=f"b{i % 2}") for i in range(4)]
        row = self.compare(a, b)["strict_text_strengthening"]
        self.assertEqual(row["n_delta_clusters"], 1)
        self.assertEqual(row["n_matched_capabilities"], 4)
        self.assertEqual(
            (row["delta"], row["delta_ci_low"], row["delta_ci_high"]), (1, "", "")
        )

    def test_arm_local_ids_resolve_to_stable_original_requirement(self):
        helper = paper_fixtures.ContextAblationTableTest()
        benchmark = helper._pure_benchmark()
        a = helper._raw_rows(benchmark, "a", "fixture-model", strengthen_weak=False)
        b = helper._raw_rows(benchmark, "b", "fixture-model", strengthen_weak=True)
        b = [
            {
                **r,
                "source_statement": source["source_statement"],
                "source_modality": source["source_modality"],
                "item_id": "document-" + r["item_id"],
                "seed_id": "arm-specific",
            }
            for r, source in zip(b, benchmark, strict=True)
        ]
        sa = context.task2_scores(
            benchmark, a, sampling_plan=paper_fixtures.NO_STOCHASTIC_PLAN
        )
        sb = context.task2_scores(
            benchmark, b, sampling_plan=paper_fixtures.NO_STOCHASTIC_PLAN
        )
        row = self.compare(sa, sb)["strict_text_strengthening"]
        self.assertEqual(row["n_matched_items"], len(benchmark))
        self.assertEqual(row["delta"], 0.25)

    def test_reused_arm_ids_do_not_override_recorded_source_identity(self):
        helper = paper_fixtures.ContextAblationTableTest()
        benchmark = helper._pure_benchmark()
        raw = helper._raw_rows(
            benchmark, "document", "fixture-model", strengthen_weak=True
        )
        # Rotate IDs onto other real benchmark items, while retaining each
        # observation's actual recorded source. A mere ID-first join is wrong.
        shifted = [
            {
                **r,
                "item_id": benchmark[(i + 1) % len(benchmark)]["item_id"],
                "source_statement": source["source_statement"],
                "source_modality": source["source_modality"],
            }
            for i, (r, source) in enumerate(zip(raw, benchmark, strict=True))
        ]
        scores = context.task2_scores(
            benchmark, shifted, sampling_plan=paper_fixtures.NO_STOCHASTIC_PLAN
        )
        self.assertEqual(
            [r["gold_modality"] for r in scores],
            [r["source_modality"] for r in benchmark],
        )
        self.assertEqual(
            [r["source_statement"] for r in scores],
            [r["source_statement"] for r in benchmark],
        )


def observations():
    rows = []
    for capability in range(18):
        for target in (0, 1):
            for sample in range(2):
                rows.append(
                    {
                        "dataset_id": "synthetic",
                        "benchmark_variant": "must",
                        "seed_id": f"s{capability}",
                        "item_id": f"s{capability}-{target}",
                        "model": "fixture-generator",
                        "run_id": "fixture-only",
                        "sample_index": sample,
                        "source_modality": "optional",
                        "pred_modality": "optional",
                        "source_statement": f"The system may perform capability {capability}.",
                        "requirement": f"Sample {sample} class {target}",
                        "strict_text_overcommit": bool(target),
                        "deterministic_text_modality_parse_status": "ok",
                        "dataset_variant": "synthetic/must",
                    }
                )
    probe.add_probe_labels(rows)
    return rows


class PredictionExportTest(unittest.TestCase):
    def test_hgb_budget_validation_never_sees_outer_test_capabilities(self):
        groups = np.repeat(np.arange(12).astype(str), 8)
        y = np.tile([0, 1], 48)
        X = np.column_stack([y, np.arange(len(y)) % 3]).astype(float)
        observed = []
        real = probe.select_hgb_budget

        def select(X, y, groups, **kwargs):
            result = real(X, y, groups, **kwargs)
            observed.append((set(groups), result))
            return result

        with mock.patch.object(probe, "select_hgb_budget", side_effect=select):
            folds = probe.fold_metrics(
                X=X,
                y_raw=y,
                groups=groups,
                target="deterministic_strict_text_overcommit",
                model_name="hgb",
                scope="global",
                n_splits=3,
                random_state=7,
                pca_components=2,
                hgb_budgets=[2, 4],
            )
        self.assertEqual(len(observed), 3)
        for fold, (outer_train, (budget, diagnostics)) in zip(
            folds, observed, strict=True
        ):
            self.assertEqual(fold["status"], "ok")
            train = set(json.loads(diagnostics["budget_validation_train_groups"]))
            validation = set(
                json.loads(diagnostics["budget_validation_held_out_groups"])
            )
            self.assertFalse(train & validation)
            self.assertEqual(train | validation, outer_train)
            self.assertLess(len(outer_train), len(set(groups)))
            self.assertIn(budget, [2, 4])
            self.assertEqual(fold["max_iter"], budget)
            self.assertEqual(len(json.loads(fold["budget_validation_curves"])), 2)

    def test_invalid_hgb_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            probe.make_estimator("hgb", 7, hgb_max_iter=0)

    def fit(self, rows=None, **kwargs):
        rows = observations() if rows is None else rows
        y = probe.target_values(rows, "deterministic_strict_text_overcommit")
        X = np.column_stack([y, np.arange(len(y)) % 3]).astype(float)
        predictions = []
        folds = probe.fold_metrics(
            X=X,
            y_raw=y,
            groups=probe.group_values(rows, "seed"),
            target="deterministic_strict_text_overcommit",
            model_name="logreg",
            scope="global",
            n_splits=3,
            random_state=17,
            sample_rows=rows,
            prediction_rows=predictions,
            **kwargs,
        )
        return predictions, folds

    def test_recompute_saved_probabilities_baselines_intervals_and_group_isolation(
        self,
    ):
        predictions, folds = self.fit()
        by_capability = {}
        for row in predictions:
            by_capability.setdefault(row["capability_id"], set()).add(row["fold"])
            self.assertEqual(json.loads(row["class_order"]), ["0", "1"])
            self.assertAlmostEqual(sum(json.loads(row["probabilities"])), 1)
            self.assertEqual(row["source_model"], "fixture-generator")
        self.assertTrue(all(len(ids) == 1 for ids in by_capability.values()))
        self.assertEqual(len(predictions), len(observations()))
        summary = probe.prediction_summary(predictions, folds, iterations=100, seed=19)
        self.assertEqual(summary["auroc_mean"], 1)
        self.assertEqual(summary["auprc_mean"], 1)
        self.assertEqual(summary["baseline_auprc"], 0.5)
        self.assertEqual(summary["auroc_ci_low"], 1)
        self.assertEqual(summary["n_evaluated_capabilities"], 18)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in predictions))
            restored = eu.read_jsonl(path)
            self.assertEqual(
                summary,
                probe.prediction_summary(restored, folds, iterations=100, seed=19),
            )

    def test_repeated_samples_do_not_create_independent_bootstrap_units(self):
        predictions, folds = self.fit()
        rng = np.random.default_rng(23)
        for row in predictions:
            p = float(rng.uniform(0.05, 0.95))
            row["probabilities"] = json.dumps([1 - p, p])
        baseline = probe.prediction_summary(predictions, folds, iterations=70, seed=7)
        repeated = [dict(r) for r in predictions for _ in range(3)]
        result = probe.prediction_summary(repeated, folds, iterations=70, seed=7)
        self.assertEqual(
            result["n_evaluated_capabilities"], baseline["n_evaluated_capabilities"]
        )
        self.assertEqual(
            result["n_evaluated_samples"], baseline["n_evaluated_samples"] * 3
        )
        self.assertLess(baseline["auroc_ci_low"], baseline["auroc_ci_high"])
        for metric in ("auroc", "auprc"):
            for stat in ("mean", "ci_low", "ci_high"):
                self.assertAlmostEqual(
                    result[f"{metric}_{stat}"], baseline[f"{metric}_{stat}"]
                )

    def test_same_observations_and_folds_across_conditions(self):
        rows = observations()
        rows[0]["pred_modality"] = ""  # jointly excluded
        rows[1]["deterministic_text_modality_parse_status"] = "unknown"
        X = np.column_stack(
            [np.arange(len(rows)) % 2, np.arange(len(rows)) % 5]
        ).astype(float)
        features = {
            f"mlx::{condition}": {
                "X": X.copy(),
                "text_vectorizer": None,
                "backend": "mlx",
                "text": condition,
            }
            for condition in ("reqonly", "prefixed")
        }
        features["mlx::prefixed"]["X"][2, 0] = np.nan
        predictions = []
        with (
            mock.patch.object(
                diagnostic, "GLOBAL_TARGETS", ["deterministic_strict_text_overcommit"]
            ),
            mock.patch.object(diagnostic, "WITHIN_SCOPES", []),
        ):
            folds = diagnostic.run_grid(
                features,
                rows,
                models=["logreg"],
                n_splits=3,
                random_state=17,
                pca_components=2,
                prediction_rows=predictions,
            )
        groups = {}
        for r in predictions:
            key = (r["group_mode"], r["item_id"], r["sample_index"])
            groups.setdefault(key, set()).add((r["text_variant"], r["fold"]))
        self.assertTrue(
            all(
                len(v) == 2 and len({fold for _, fold in v}) == 1
                for v in groups.values()
            )
        )
        self.assertEqual({r["n_eligible"] for r in folds}, {len(rows) - 3})
        summary = diagnostic.summarize_grid(folds, predictions, iterations=30, seed=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe_grid_summary.csv"
            eu.write_csv_rows(path, summary)
            # Plot selects HGB, so a logreg-only artifact has no row for the
            # bar: that is a drifted grid, and the figure must refuse it.
            with self.assertRaises(ValueError):
                figure.resolve_bars(eu.read_csv_rows(path), [{**figure.TARGET_BARS[0]}])

    def test_diagnostic_cli_streams_recomputable_predictions_without_embedding_calls(
        self,
    ):
        rows = observations()
        X = np.asarray(
            [
                [int(r["deterministic_strict_text_overcommit"]), i % 3]
                for i, r in enumerate(rows)
            ],
            dtype=float,
        )
        features = {
            "mlx::reqonly": {
                "X": X,
                "backend": "mlx",
                "text": "reqonly",
                "text_vectorizer": None,
            }
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(diagnostic, "manifest_rows", return_value=[]),
            mock.patch.object(
                diagnostic, "load_embeddings_and_rows", return_value=(X, rows)
            ),
            mock.patch.object(diagnostic, "build_feature_sets", return_value=features),
            mock.patch.object(
                diagnostic, "GLOBAL_TARGETS", ["deterministic_strict_text_overcommit"]
            ),
            mock.patch.object(diagnostic, "WITHIN_SCOPES", []),
            mock.patch(
                "sys.argv",
                [
                    "diagnostic",
                    "--output-dir",
                    directory,
                    "--models",
                    "logreg",
                    "--bootstrap-samples",
                    "20",
                ],
            ),
        ):
            diagnostic.main()
            predictions = eu.read_jsonl(
                Path(directory) / "probe_grid_predictions.jsonl"
            )
            folds = eu.read_csv_rows(Path(directory) / "probe_grid_folds.csv")
            summary = eu.read_csv_rows(Path(directory) / "probe_grid_summary.csv")
            recomputed = diagnostic.summarize_grid(folds, predictions, iterations=20)
            for saved, fresh in zip(summary, recomputed, strict=True):
                self.assertAlmostEqual(float(saved["auroc_mean"]), fresh["auroc_mean"])
                self.assertAlmostEqual(
                    float(saved["auprc_ci_low"]), fresh["auprc_ci_low"]
                )
                self.assertEqual(
                    int(saved["n_evaluated_samples"]), fresh["n_evaluated_samples"]
                )

    def test_missing_classes_groups_and_fit_failure_are_explicit(self):
        rows = observations()
        one_class = [
            r for r in rows if r["deterministic_strict_text_overcommit"] == "1"
        ]
        predictions, folds = self.fit(one_class)
        self.assertFalse(predictions)
        self.assertIn("missing target classes", folds[0]["unavailable_reason"])
        one_group = [{**r, "seed_id": "same"} for r in rows]
        _, folds = self.fit(one_group)
        self.assertIn("insufficient capability", folds[0]["unavailable_reason"])
        with mock.patch.object(probe, "make_estimator") as factory:
            factory.return_value.fit.side_effect = ValueError(
                "synthetic fitting failure"
            )
            predictions, folds = self.fit()
        summary = probe.prediction_summary(predictions, folds, iterations=10)
        self.assertEqual(summary["auroc_mean"], "")
        self.assertEqual(summary["folds"], 0)
        self.assertIn("fit failed", summary["unavailable_reason"])

    def test_convergence_warning_and_training_diagnostics_are_retained(self):
        import warnings

        real = probe.make_estimator

        def estimator(*args, **kwargs):
            result = real(*args, **kwargs)
            original = result.fit

            def fit(X, y):
                warnings.warn(
                    "fixture iteration limit", ConvergenceWarning, stacklevel=2
                )
                return original(X, y)

            result.fit = fit
            return result

        with mock.patch.object(probe, "make_estimator", side_effect=estimator):
            predictions, folds = self.fit()
        self.assertTrue(all(r["convergence_warning"] for r in folds))
        self.assertTrue(all(r["max_iter"] == 1000 and r["n_iter"] > 0 for r in folds))
        self.assertTrue(
            probe.prediction_summary(predictions, folds, iterations=10)[
                "fit_review_required"
            ]
        )

    def test_multiclass_ap_baseline_and_missing_test_classes(self):
        predictions, folds = self.fit()
        # Two-label auxiliary AP is macro, hence its baseline is 1/K.
        for row in predictions:
            row["target"] = "dataset_variant"
        summary = probe.prediction_summary(predictions, folds, iterations=30)
        self.assertEqual(summary["baseline_auprc"], 0.5)
        self.assertEqual(summary["auprc_mean"], 1)
        # Missing classes in held-out folds are exported, never silently skipped.
        rows = observations()
        rows = [
            r
            for r in rows
            if (int(r["seed_id"][1:]) < 9)
            == (r["deterministic_strict_text_overcommit"] == "1")
        ]
        with mock.patch.object(
            probe.StratifiedGroupKFold,
            "split",
            return_value=iter([(np.arange(18, 36), np.arange(18))]),
        ):
            _, bad_folds = self.fit(rows)
        self.assertEqual(bad_folds[0]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
