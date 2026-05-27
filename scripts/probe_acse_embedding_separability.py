"""Probe whether cached ACSE embeddings separate dataset, modality, or drift labels."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu

try:
    from plot_acse_global_embedding_projection import load_embeddings_and_rows, manifest_rows
except ModuleNotFoundError:  # pragma: no cover
    from scripts.plot_acse_global_embedding_projection import load_embeddings_and_rows, manifest_rows


BINARY_TARGETS = {
    "deterministic_strict_text_overcommit",
    "deterministic_broad_text_overcommit",
    "sample_strict_text_overcommit",
    "sample_broad_text_overcommit",
}
MULTICLASS_TARGETS = {
    "dataset_id",
    "dataset_variant",
    "benchmark_variant",
    "source_modality",
    "pred_modality",
}
DEFAULT_TARGETS = [
    "dataset_id",
    "dataset_variant",
    "benchmark_variant",
    "source_modality",
    "deterministic_strict_text_overcommit",
    "deterministic_broad_text_overcommit",
    "sample_strict_text_overcommit",
    "sample_broad_text_overcommit",
]
DEFAULT_WITHIN_TARGETS = [
    "deterministic_strict_text_overcommit",
    "sample_strict_text_overcommit",
]


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def clean_label(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "unknown"


def sample_text_fields(row: dict[str, Any]) -> dict[str, Any]:
    return eu.text_modality_fields(
        row.get("requirement", ""),
        clean_label(row.get("source_modality", "")),
        clean_label(row.get("pred_modality", "")),
        confidence=1.0,
    )


def add_probe_labels(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        fields = sample_text_fields(row)
        row["deterministic_strict_text_overcommit"] = "1" if truthy(row.get("strict_text_overcommit", "")) else "0"
        row["deterministic_broad_text_overcommit"] = "1" if truthy(row.get("text_overcommit", "")) else "0"
        row["sample_strict_text_overcommit"] = "1" if truthy(fields.get("strict_text_overcommit", "")) else "0"
        row["sample_broad_text_overcommit"] = "1" if truthy(fields.get("text_overcommit", "")) else "0"
        row["sample_text_modality"] = fields.get("text_modality", "")
        row["sample_text_modality_basis"] = fields.get("text_modality_basis", "")


def target_values(rows: list[dict[str, Any]], target: str) -> np.ndarray:
    if target in BINARY_TARGETS:
        return np.asarray([1 if truthy(row.get(target, "")) else 0 for row in rows], dtype=int)
    if target in MULTICLASS_TARGETS:
        return np.asarray([clean_label(row.get(target, "")) for row in rows], dtype=object)
    raise ValueError(f"Unknown target: {target}")


def group_values(rows: list[dict[str, Any]], mode: str) -> np.ndarray:
    if mode == "seed":
        return np.asarray([f"{row.get('dataset_id')}::{row.get('seed_id')}" for row in rows], dtype=object)
    if mode == "item":
        return np.asarray([f"{row.get('dataset_id')}::{row.get('benchmark_variant')}::{row.get('item_id')}" for row in rows], dtype=object)
    raise ValueError(f"Unknown group mode: {mode}")


def make_estimator(model_name: str, random_state: int) -> Any:
    if model_name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_state,
            ),
        )
    if model_name == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=100,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=0.05,
            class_weight="balanced",
            random_state=random_state,
        )
    raise ValueError(f"Unknown model: {model_name}")


def finite_metric(value: float) -> float | str:
    return "" if math.isnan(value) else float(value)


def positive_rate(y: np.ndarray) -> float:
    if y.size == 0:
        return math.nan
    return float(np.mean(y.astype(int)))


def multiclass_auroc(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    if len(classes) < 2:
        return math.nan
    present_classes = np.unique(y_true)
    if len(present_classes) < len(classes):
        return math.nan
    try:
        return float(roc_auc_score(y_true, probabilities, labels=classes, multi_class="ovr", average="macro"))
    except ValueError:
        return math.nan


def multiclass_average_precision(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    if len(classes) < 2:
        return math.nan
    if len(np.unique(y_true)) < len(classes):
        return math.nan
    try:
        binary_true = label_binarize(y_true, classes=classes)
        return float(average_precision_score(binary_true, probabilities, average="macro"))
    except ValueError:
        return math.nan


def fold_metrics(
    *,
    X: np.ndarray,
    y_raw: np.ndarray,
    groups: np.ndarray,
    target: str,
    model_name: str,
    scope: str,
    n_splits: int,
    random_state: int,
) -> list[dict[str, Any]]:
    if len(np.unique(y_raw)) < 2:
        return []
    n_splits = min(n_splits, len(np.unique(groups)))
    if target in BINARY_TARGETS:
        y = y_raw.astype(int)
        class_labels = np.asarray([0, 1])
        y_for_split = y
    else:
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw)
        class_labels = np.arange(len(encoder.classes_))
        y_for_split = y
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows: list[dict[str, Any]] = []
    for fold_index, (train_index, test_index) in enumerate(splitter.split(X, y_for_split, groups=groups)):
        estimator = make_estimator(model_name, random_state + fold_index)
        estimator.fit(X[train_index], y[train_index])
        pred = estimator.predict(X[test_index])
        probabilities = estimator.predict_proba(X[test_index])
        y_test = y[test_index]
        if target in BINARY_TARGETS:
            positive_prob = probabilities[:, list(estimator.classes_).index(1)] if 1 in estimator.classes_ else np.zeros(len(test_index))
            auroc = float(roc_auc_score(y_test, positive_prob)) if len(np.unique(y_test)) == 2 else math.nan
            auprc = float(average_precision_score(y_test, positive_prob)) if len(np.unique(y_test)) == 2 else math.nan
            prevalence = positive_rate(y_test)
            macro_auroc = auroc
        else:
            probability_matrix = np.zeros((len(test_index), len(class_labels)), dtype=float)
            for local_col, class_id in enumerate(estimator.classes_):
                probability_matrix[:, int(class_id)] = probabilities[:, local_col]
            auprc = multiclass_average_precision(y_test, probability_matrix, class_labels)
            macro_auroc = multiclass_auroc(y_test, probability_matrix, class_labels)
            prevalence = math.nan
        rows.append(
            {
                "scope": scope,
                "target": target,
                "model": model_name,
                "fold": fold_index,
                "n_train": len(train_index),
                "n_test": len(test_index),
                "n_groups_train": len(np.unique(groups[train_index])),
                "n_groups_test": len(np.unique(groups[test_index])),
                "class_distribution_test": json.dumps(dict(Counter(str(value) for value in y_raw[test_index])), sort_keys=True),
                "positive_rate_test": finite_metric(prevalence),
                "accuracy": float(accuracy_score(y_test, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                "auroc_macro": finite_metric(macro_auroc),
                "average_precision_macro": finite_metric(auprc),
                "baseline_average_precision": finite_metric(prevalence if target in BINARY_TARGETS else math.nan),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auroc_macro",
        "average_precision_macro",
        "baseline_average_precision",
        "positive_rate_test",
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["scope"]), str(row["target"]), str(row["model"])), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (scope, target, model), group_rows in sorted(grouped.items()):
        output = {
            "scope": scope,
            "target": target,
            "model": model,
            "folds": len(group_rows),
            "mean_n_test": float(np.mean([float(row["n_test"]) for row in group_rows])),
        }
        for metric in metric_names:
            values = []
            for row in group_rows:
                value = row.get(metric, "")
                if value == "":
                    continue
                values.append(float(value))
            output[f"{metric}_mean"] = finite_metric(float(np.mean(values)) if values else math.nan)
            output[f"{metric}_std"] = finite_metric(float(np.std(values, ddof=1)) if len(values) > 1 else math.nan)
        summary_rows.append(output)
    return summary_rows


def filtered_scope(
    X: np.ndarray,
    rows: list[dict[str, Any]],
    scope: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if scope == "global":
        return X, rows
    key, value = scope.split("=", 1)
    mask = np.asarray([str(row.get(key, "")) == value for row in rows], dtype=bool)
    return X[mask], [row for row, keep in zip(rows, mask) if keep]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grouped supervised probes on cached ACSE embeddings.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/acse_semantic_artifact_manifest.csv"))
    parser.add_argument("--backend-prefix", default="mlx:")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/acse_embedding_separability_probe"))
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--within-targets", nargs="+", default=DEFAULT_WITHIN_TARGETS)
    parser.add_argument("--extra-scopes", nargs="*", default=[])
    parser.add_argument("--models", nargs="+", default=["logreg", "hgb"], choices=["logreg", "hgb"])
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--group-mode", choices=["seed", "item"], default="seed")
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--random-state", type=int, default=20260527)
    args = parser.parse_args()

    root = eu.project_root()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    rows = manifest_rows(manifest_path, args.backend_prefix)
    embeddings, sample_rows = load_embeddings_and_rows(rows)
    add_probe_labels(sample_rows)

    pca_components = min(args.pca_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=args.random_state)
    features = pca.fit_transform(embeddings)

    fold_rows: list[dict[str, Any]] = []
    scopes = ["global"]
    scopes.extend(f"dataset_variant={value}" for value in sorted({row["dataset_variant"] for row in sample_rows}))
    scopes.extend(args.extra_scopes)
    for scope in scopes:
        X_scope, rows_scope = filtered_scope(features, sample_rows, scope)
        target_names = args.targets if scope == "global" else args.within_targets
        if not rows_scope:
            continue
        groups = group_values(rows_scope, args.group_mode)
        for target in target_names:
            y = target_values(rows_scope, target)
            if len(np.unique(y)) < 2:
                continue
            for model_name in args.models:
                fold_rows.extend(
                    fold_metrics(
                        X=X_scope,
                        y_raw=y,
                        groups=groups,
                        target=target,
                        model_name=model_name,
                        scope=scope,
                        n_splits=args.n_splits,
                        random_state=args.random_state,
                    )
                )

    summary_rows = summarize(fold_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / "fold_metrics.csv"
    summary_path = output_dir / "summary.csv"
    eu.write_csv_rows(fold_path, fold_rows)
    eu.write_csv_rows(summary_path, summary_rows)
    (output_dir / "summary.md").write_text(eu.markdown_table(summary_rows, list(summary_rows[0])) + "\n", encoding="utf-8")
    manifest = {
        "backend_prefix": args.backend_prefix,
        "embedding_backend_rows": len(rows),
        "sample_rows": len(sample_rows),
        "embedding_shape": list(embeddings.shape),
        "feature_projection": "pca",
        "pca_components": int(pca_components),
        "pca_explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
        "models": args.models,
        "targets": args.targets,
        "within_targets": args.within_targets,
        "n_splits": args.n_splits,
        "group_mode": args.group_mode,
        "random_state": args.random_state,
        "artifacts": [str(fold_path), str(summary_path), str(output_dir / "summary.md")],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary_rows": len(summary_rows), **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
