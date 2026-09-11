"""Grouped prediction from sampled requirement text.

The primary target is strict strengthening in the separate single-pass output.
Adding the sampled declared label supplies additional input; it does not by
itself establish target leakage. Source modality and benchmark origin are
separate auxiliary targets. All learned preprocessing is fitted within folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu

try:
    from plot_acse_global_embedding_projection import (
        load_embeddings_and_rows,
        manifest_rows,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.plot_acse_global_embedding_projection import (
        load_embeddings_and_rows,
        manifest_rows,
    )


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

# Requirement text is primary; the sampled declared label is additional input.
REQUIREMENT_ONLY_CONDITION = "requirement_only"
PREFIXED_CONTROL_CONDITION = "requirement_with_declared_label"
PRIMARY_TEXT_CONDITION = REQUIREMENT_ONLY_CONDITION
TEXT_CONDITION_ROLES = {
    REQUIREMENT_ONLY_CONDITION: "primary",
    PREFIXED_CONTROL_CONDITION: "additional_input",
}
DEFAULT_TEXT_CONDITIONS = [REQUIREMENT_ONLY_CONDITION, PREFIXED_CONTROL_CONDITION]

# ``class_labels`` is ascending, so column 1 of a two-class probability matrix is
# the greater label -- the positive class under scikit-learn's binary convention.
POSITIVE_COLUMN = 1


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
        row["deterministic_strict_text_overcommit"] = (
            ("1" if eu.is_truthy_strict(row["strict_text_overcommit"]) else "0")
            if row.get("strict_text_overcommit", "") != ""
            else ""
        )
        row["deterministic_broad_text_overcommit"] = (
            ("1" if eu.is_truthy_strict(row["text_overcommit"]) else "0")
            if row.get("text_overcommit", "") != ""
            else ""
        )
        row["sample_strict_text_overcommit"] = (
            "1"
            if eu.is_truthy_strict(fields.get("strict_text_overcommit", ""))
            else "0"
        )
        row["sample_broad_text_overcommit"] = (
            "1" if eu.is_truthy_strict(fields.get("text_overcommit", "")) else "0"
        )


def stamp_text_condition(row: dict[str, Any], condition: str) -> dict[str, Any]:
    """Prefix a result row with the text condition it came from and that condition's role.

    Both columns are written on every artifact so a reader never has to know which
    condition happens to be the default to tell a diagnostic from a control.
    """
    return {
        "text_condition": condition,
        "text_condition_role": TEXT_CONDITION_ROLES[condition],
        **row,
    }


def requirements_digest(requirements: list[str]) -> str:
    """Bind a requirement-only embedding cache to the exact texts it was built from.

    A row count cannot tell a stale cache of equal length apart from the right
    one, so every reader of the cache compares this instead.
    """
    return hashlib.sha256(
        ("mlx\x00" + "\x00".join(requirements)).encode("utf-8")
    ).hexdigest()


def embed_requirement_only(
    requirements: list[str],
    *,
    batch_size: int,
    cache_path: Path,
    reuse_cache: bool,
) -> np.ndarray:
    """Embed each row's requirement text with MLX, deduplicating first."""
    digest = requirements_digest(requirements)
    if reuse_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        # --reqonly-cache can be pointed at any .npz, so every key this reads is
        # optional; a file written by something else falls through to re-embedding
        # instead of raising.
        cached_digest = (
            str(cached["requirements_digest"])
            if "requirements_digest" in cached.files
            else ""
        )
        cached_rows = (
            int(cached["n_rows"])
            if "n_rows" in cached.files
            else cached["embeddings"].shape[0]
            if "embeddings" in cached.files
            else -1
        )
        if cached_rows == len(requirements) and cached_digest == digest:
            print(
                f"[reqonly-mlx] reuse cache {cache_path} ({cached['embeddings'].shape})"
            )
            return cached["embeddings"].astype(np.float32, copy=False)

    unique_texts = sorted(set(requirements))
    index_of = {text: i for i, text in enumerate(unique_texts)}
    print(
        f"[reqonly-mlx] embedding {len(unique_texts)} unique requirement strings "
        f"(of {len(requirements)} rows) in batches of {batch_size}"
    )
    blocks: list[np.ndarray] = []
    for start in range(0, len(unique_texts), batch_size):
        batch = unique_texts[start : start + batch_size]
        matrix, _ = eu.semantic_embedding_matrix(batch, embedding_backend="mlx")
        blocks.append(np.asarray(matrix, dtype=np.float32))
        if (start // batch_size) % 5 == 0:
            print(
                f"  embedded {min(start + batch_size, len(unique_texts))}/{len(unique_texts)}"
            )
    unique_embeddings = np.vstack(blocks)
    row_embeddings = unique_embeddings[[index_of[text] for text in requirements]]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=row_embeddings.astype(np.float32),
        n_rows=np.asarray(len(requirements)),
        dim=np.asarray(row_embeddings.shape[1]),
        requirements_digest=np.asarray(digest),
    )
    print(f"[reqonly-mlx] cached -> {cache_path} ({row_embeddings.shape})")
    return row_embeddings.astype(np.float32, copy=False)


def build_text_conditions(
    sample_rows: list[dict[str, Any]],
    cached_prefixed: np.ndarray,
    *,
    conditions: list[str],
    mlx_batch: int,
    cache_path: Path,
    reuse_cache: bool,
) -> dict[str, np.ndarray]:
    """Return the raw (unprojected) feature matrix for each requested condition."""
    matrices: dict[str, np.ndarray] = {}
    for condition in conditions:
        if condition == PREFIXED_CONTROL_CONDITION:
            matrices[condition] = cached_prefixed
        elif condition == REQUIREMENT_ONLY_CONDITION:
            matrices[condition] = embed_requirement_only(
                [str(row.get("requirement", "")) for row in sample_rows],
                batch_size=mlx_batch,
                cache_path=cache_path,
                reuse_cache=reuse_cache,
            )
        else:
            raise ValueError(f"Unknown text condition: {condition}")
        print(f"[{condition}] {matrices[condition].shape}")
    return matrices


def target_values(rows: list[dict[str, Any]], target: str) -> np.ndarray:
    if target in BINARY_TARGETS:
        return np.asarray(
            [1 if eu.is_truthy_strict(row.get(target, "")) else 0 for row in rows],
            dtype=int,
        )
    if target in MULTICLASS_TARGETS:
        return np.asarray(
            [clean_label(row.get(target, "")) for row in rows], dtype=object
        )
    raise ValueError(f"Unknown target: {target}")


def group_values(rows: list[dict[str, Any]], mode: str) -> np.ndarray:
    if mode == "seed":
        return np.asarray(
            [f"{row.get('dataset_id')}::{row.get('seed_id')}" for row in rows],
            dtype=object,
        )
    if mode == "item":
        return np.asarray(
            [
                f"{row.get('dataset_id')}::{row.get('benchmark_variant')}::{row.get('item_id')}"
                for row in rows
            ],
            dtype=object,
        )
    raise ValueError(f"Unknown group mode: {mode}")


def model_steps(model_name: str, random_state: int) -> list[Any]:
    if model_name == "logreg":
        return [
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_state,
            ),
        ]
    if model_name == "hgb":
        return [
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=300,
                early_stopping=True,
                validation_fraction=None,
                n_iter_no_change=20,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                l2_regularization=0.05,
                class_weight="balanced",
                random_state=random_state,
            )
        ]
    raise ValueError(f"Unknown model: {model_name}")


class FoldSafeSVD(TruncatedSVD):
    """``TruncatedSVD`` that clamps its rank to what the training fold supports.

    When a vectorizer is fitted inside the pipeline, the fold's vocabulary size is
    unknown until ``fit`` runs, so the component count cannot be clamped by the
    caller the way a dense matrix allows.
    """

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        n_samples, n_features = X.shape
        self.n_components = max(1, min(self.n_components, n_samples, n_features - 1))
        return super().fit_transform(X, y)


def make_estimator(
    model_name: str,
    random_state: int,
    *,
    pca_components: int | None = None,
    text_vectorizer: Any | None = None,
    hgb_max_iter: int = 300,
    fixed_budget: bool = False,
) -> Any:
    """Build the probe estimator, with the vectorizer and PCA as pipeline steps.

    Putting both in the pipeline is what makes them fold-local: the estimator is
    fitted on the training rows of one fold, so neither the vocabulary and idf
    weights nor the axes the held-out rows are transformed into were chosen with
    them. Callers that pass features which are already reduced leave
    ``pca_components`` at ``None``.

    A text vectorizer produces a sparse matrix, which ``PCA`` cannot centre, so
    the reducer is the truncated SVD that ``TfidfVectorizer`` output calls for.
    """
    steps = model_steps(model_name, random_state)
    if model_name == "hgb":
        if hgb_max_iter < 1:
            raise ValueError("hgb_max_iter must be positive")
        steps[-1].set_params(max_iter=hgb_max_iter, early_stopping=not fixed_budget)
    if pca_components is not None:
        steps.insert(
            0,
            FoldSafeSVD(n_components=pca_components, random_state=random_state)
            if text_vectorizer is not None
            else PCA(
                n_components=pca_components,
                svd_solver="randomized",
                random_state=random_state,
            ),
        )
    if text_vectorizer is not None:
        # A fresh clone per fold: a fitted vectorizer would carry the previous
        # fold's vocabulary into this one.
        steps.insert(0, clone(text_vectorizer))
    return steps[0] if len(steps) == 1 else make_pipeline(*steps)


def finite_metric(value: float) -> float | str:
    return "" if math.isnan(value) else float(value)


def select_hgb_budget(
    X, y, groups, *, budgets, random_state, pca_components=None, text_vectorizer=None
):
    """Select a finite training budget without seeing the outer test fold.

    Preprocessing is refitted within the inner training split for every budget.
    The caller then refits the selected budget on the complete outer training
    fold. Curves are diagnostic evidence, not a claim of convergence.
    """
    budgets = sorted(set(budgets))
    if not budgets or budgets[0] < 1:
        raise ValueError("HGB budgets must be positive")
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("inner validation needs at least two capability groups")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    classes = np.unique(y)
    split = next(
        (
            (a, b)
            for a, b in splitter.split(X, y, groups)
            if len(np.unique(y[a])) == len(classes)
            and len(np.unique(y[b])) == len(classes)
        ),
        None,
    )
    if split is None:
        raise ValueError("no inner capability split contains all target classes")
    train, validation = split
    curves = []
    for budget in budgets:
        components = min(pca_components, len(train)) if pca_components else None
        estimator = make_estimator(
            "hgb",
            random_state,
            pca_components=components,
            text_vectorizer=text_vectorizer,
            hgb_max_iter=budget,
            fixed_budget=True,
        )
        estimator.fit(X[train], y[train])
        curves.append(
            {
                "iterations": budget,
                "training_log_loss": float(
                    log_loss(
                        y[train], estimator.predict_proba(X[train]), labels=classes
                    )
                ),
                "validation_log_loss": float(
                    log_loss(
                        y[validation],
                        estimator.predict_proba(X[validation]),
                        labels=classes,
                    )
                ),
            }
        )
    best = min(curves, key=lambda row: (row["validation_log_loss"], row["iterations"]))
    return best["iterations"], {
        "budget_selection": "inner capability validation log loss",
        "budget_validation_curves": json.dumps(curves),
        "budget_validation_train_groups": json.dumps(
            sorted(set(map(str, groups[train])))
        ),
        "budget_validation_held_out_groups": json.dumps(
            sorted(set(map(str, groups[validation])))
        ),
        "budget_selected_at_upper_limit": best["iterations"] == budgets[-1],
    }


def positive_rate(y: np.ndarray) -> float:
    if y.size == 0:
        return math.nan
    return float(np.mean(y.astype(int)))


def ordered_class_probabilities(
    probabilities: np.ndarray,
    estimator_classes: np.ndarray,
    class_labels: np.ndarray,
) -> np.ndarray:
    """Scatter an estimator's probability columns into ``class_labels`` order.

    ``estimator.classes_`` only covers the classes its training fold contained, in
    its own order. Keying the copy by class label rather than by position keeps the
    matrix aligned no matter how the estimator ordered or dropped classes; a class
    the estimator never saw keeps a zero column.
    """
    column_of = {int(label): column for column, label in enumerate(class_labels)}
    ordered = np.zeros((probabilities.shape[0], len(class_labels)), dtype=float)
    for local_column, class_id in enumerate(estimator_classes):
        ordered[:, column_of[int(class_id)]] = probabilities[:, local_column]
    return ordered


def ranking_metric_is_defined(y_true: np.ndarray, classes: np.ndarray) -> bool:
    """AUROC and average precision need at least two classes, all present in the fold.

    A held-out fold that is missing a class has no defined one-vs-rest score for it,
    so the probe reports an empty cell. That is a property of the fold, not a
    swallowed error.
    """
    return len(classes) >= 2 and len(np.unique(y_true)) == len(classes)


def probe_auroc(
    y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray
) -> float:
    """Held-out AUROC; ``probabilities`` must already be in ``classes`` order.

    Exactly two classes take scikit-learn's binary API on the positive column --
    the multiclass entry point rejects a two-column score matrix outright. Three or
    more classes keep macro-averaged one-vs-rest.
    """
    if not ranking_metric_is_defined(y_true, classes):
        return math.nan
    if len(classes) == 2:
        return float(roc_auc_score(y_true, probabilities[:, POSITIVE_COLUMN]))
    return float(
        roc_auc_score(
            y_true,
            probabilities,
            labels=classes,
            multi_class="ovr",
            average="macro",
        )
    )


def probe_average_precision(
    y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray
) -> float:
    """Macro-averaged held-out average precision, one column per class."""
    if not ranking_metric_is_defined(y_true, classes):
        return math.nan
    indicator = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        # label_binarize collapses two classes into a single column; macro-averaging
        # over both classes needs the negative class spelled out as its own column.
        indicator = np.hstack([1 - indicator, indicator])
    return float(average_precision_score(indicator, probabilities, average="macro"))


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
    pca_components: int | None = None,
    text_vectorizer: Any | None = None,
    sample_rows: list[dict[str, Any]] | None = None,
    prediction_rows: list[dict[str, Any]] | None = None,
    expected_classes: list[str] | None = None,
    hgb_max_iter: int = 300,
    hgb_budgets: list[int] | None = None,
) -> list[dict[str, Any]]:
    metadata = sample_rows if sample_rows is not None else [{} for _ in y_raw]
    names = (
        np.asarray(["0", "1"])
        if target in BINARY_TARGETS
        else np.asarray(sorted(expected_classes))
        if expected_classes
        else np.unique(y_raw.astype(str))
    )
    encoded = {str(value): i for i, value in enumerate(names)}
    y = np.asarray([encoded[str(value)] for value in y_raw], dtype=int)
    class_labels = np.arange(len(names))
    n_splits = min(n_splits, len(np.unique(groups)))
    base = {
        "scope": scope,
        "target": target,
        "model": model_name,
        "class_order": json.dumps(names.tolist()),
        "n_eligible": len(y),
        "n_splits_requested": n_splits,
        "class_distribution_eligible": json.dumps(
            dict(Counter(str(v) for v in y_raw)), sort_keys=True
        ),
    }
    reason = (
        "missing target classes"
        if len(np.unique(y)) < 2 or len(np.unique(y)) != len(names)
        else "insufficient capability groups"
        if n_splits < 2
        else ""
    )
    if reason:
        return [
            {
                **base,
                "fold": -1,
                "n_train": 0,
                "n_test": 0,
                "status": "unavailable",
                "unavailable_reason": reason,
            }
        ]
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    rows: list[dict[str, Any]] = []
    try:
        splits = list(splitter.split(X, y, groups=groups))
    except ValueError as error:
        return [
            {
                **base,
                "fold": -1,
                "n_train": 0,
                "n_test": 0,
                "status": "unavailable",
                "unavailable_reason": f"grouped split unavailable: {error}",
            }
        ]
    for fold_index, (train_index, test_index) in enumerate(splits):
        # PCA cannot exceed the rank the training fold can support. With a
        # vectorizer in the pipeline the feature count is only known at fit time,
        # so FoldSafeSVD finishes the clamping there.
        fold_components = (
            None
            if pca_components is None
            else max(
                1,
                min(pca_components, len(train_index))
                if text_vectorizer is not None
                else min(pca_components, len(train_index), X.shape[1]),
            )
        )
        estimator = make_estimator(
            model_name,
            random_state + fold_index,
            pca_components=fold_components,
            text_vectorizer=text_vectorizer,
            hgb_max_iter=hgb_max_iter,
        )
        fold_base = {
            **base,
            "fold": fold_index,
            "n_train": len(train_index),
            "n_test": len(test_index),
            "n_groups_train": len(np.unique(groups[train_index])),
            "n_groups_test": len(np.unique(groups[test_index])),
            "class_distribution_train": json.dumps(
                dict(Counter(str(v) for v in y_raw[train_index])), sort_keys=True
            ),
            "class_distribution_test": json.dumps(
                dict(Counter(str(v) for v in y_raw[test_index])), sort_keys=True
            ),
        }
        if len(np.unique(y[train_index])) != len(class_labels):
            rows.append(
                {
                    **fold_base,
                    "status": "unavailable",
                    "unavailable_reason": "training fold missing classes",
                }
            )
            continue
        started = time.monotonic()
        budget_diagnostics = {
            "budget_selection": "fixed maximum; no validation selection"
        }
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if model_name == "hgb" and hgb_budgets:
                    capability_groups = (
                        group_values(metadata, "seed")
                        if sample_rows is not None
                        else groups
                    )
                    selected_budget, budget_diagnostics = select_hgb_budget(
                        X[train_index],
                        y[train_index],
                        capability_groups[train_index],
                        budgets=hgb_budgets,
                        random_state=random_state + fold_index,
                        pca_components=fold_components,
                        text_vectorizer=text_vectorizer,
                    )
                    estimator = make_estimator(
                        model_name,
                        random_state + fold_index,
                        pca_components=fold_components,
                        text_vectorizer=text_vectorizer,
                        hgb_max_iter=selected_budget,
                        fixed_budget=True,
                    )
                estimator.fit(X[train_index], y[train_index])
            pred = estimator.predict(X[test_index])
            probability_matrix = ordered_class_probabilities(
                estimator.predict_proba(X[test_index]), estimator.classes_, class_labels
            )
        except (ValueError, RuntimeError, FloatingPointError) as error:
            rows.append(
                {
                    **fold_base,
                    "status": "unavailable",
                    "unavailable_reason": f"fit failed: {type(error).__name__}: {error}",
                }
            )
            print(f"[fit] {model_name} {scope} {target} fold={fold_index}: {error}")
            continue
        fitted = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
        n_iter = int(np.max(np.asarray(getattr(fitted, "n_iter_", 0))))
        limit = int(getattr(fitted, "max_iter", 0))
        fit_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        train_scores = np.asarray(getattr(fitted, "train_score_", []))
        no_improvement = bool(len(train_scores) > 1 and np.ptp(train_scores) < 1e-10)
        insufficient_split_samples = bool(
            model_name == "hgb" and len(train_index) < 2 * fitted.min_samples_leaf
        )
        training_probabilities = ordered_class_probabilities(
            estimator.predict_proba(X[train_index]), estimator.classes_, class_labels
        )
        diagnostics = {
            **budget_diagnostics,
            "stopping_score_source": "inner validation selected fixed budget"
            if hgb_budgets and model_name == "hgb"
            else "training loss (not validation)"
            if model_name == "hgb"
            else "optimizer tolerance",
            "fit_seconds": time.monotonic() - started,
            "estimator_parameters": json.dumps(
                fitted.get_params(), sort_keys=True, default=str
            ),
            "training_accuracy": float(
                accuracy_score(
                    y[train_index], np.argmax(training_probabilities, axis=1)
                )
            ),
            "training_log_loss": float(
                log_loss(y[train_index], training_probabilities, labels=class_labels)
            ),
            "no_training_loss_improvement": no_improvement,
            "insufficient_split_samples": insufficient_split_samples,
            "n_iter": n_iter,
            "max_iter": limit,
            "iteration_limit_reached": n_iter >= limit if limit else False,
            "convergence_warning": fit_warning,
            "fit_warnings": json.dumps([str(w.message) for w in caught]),
            "training_scores": json.dumps(
                np.asarray(getattr(fitted, "train_score_", [])).tolist()
            ),
            "validation_scores": json.dumps(
                np.asarray(getattr(fitted, "validation_score_", [])).tolist()
            ),
            "fit_assessment": "review fitting budget"
            if fit_warning
            or (limit and n_iter >= limit)
            or no_improvement
            or insufficient_split_samples
            else "no stopping warning; inspect training diagnostics",
        }
        print(
            f"[fit] {model_name} {scope} {target} fold={fold_index} train={len(train_index)} test={len(test_index)} iterations={n_iter}/{limit}: {diagnostics['fit_assessment']}"
        )
        if prediction_rows is not None:
            for index, probability in zip(test_index, probability_matrix, strict=True):
                source = metadata[index]
                prediction_rows.append(
                    {
                        **{
                            key: source.get(key, "")
                            for key in (
                                "dataset_id",
                                "benchmark_variant",
                                "run_id",
                                "model",
                                "profile_id",
                                "item_id",
                                "seed_id",
                                "source_modality",
                                "sample_index",
                                "global_embedding_index",
                                "pred_modality",
                                "source_artifact_dir",
                            )
                        },
                        "source_identity": json.dumps(
                            [
                                source.get(k, "")
                                for k in (
                                    "dataset_id",
                                    "benchmark_variant",
                                    "seed_id",
                                    "source_modality",
                                )
                            ]
                        ),
                        "source_model": source.get("model", ""),
                        "classifier": model_name,
                        "scope": scope,
                        "target": target,
                        "fold": fold_index,
                        "capability_id": str(group_values([source], "seed")[0])
                        if source
                        else str(groups[index]),
                        "split_group": str(groups[index]),
                        "class_order": json.dumps(names.tolist()),
                        "target_value": str(y_raw[index]),
                        "probabilities": json.dumps(probability.tolist()),
                    }
                )
        y_test = y[test_index]
        if target in BINARY_TARGETS:
            # Binary targets keep positive-class average precision (not the macro
            # form) because it is what `baseline_average_precision` is a baseline for.
            positive_prob = probability_matrix[:, POSITIVE_COLUMN]
            auroc = (
                float(roc_auc_score(y_test, positive_prob))
                if len(np.unique(y_test)) == 2
                else math.nan
            )
            auprc = (
                float(average_precision_score(y_test, positive_prob))
                if len(np.unique(y_test)) == 2
                else math.nan
            )
            prevalence = positive_rate(y_test)
            macro_auroc = auroc
        else:
            auprc = probe_average_precision(y_test, probability_matrix, class_labels)
            macro_auroc = probe_auroc(y_test, probability_matrix, class_labels)
            prevalence = math.nan
        fold_row: dict[str, Any] = {
            **fold_base,
            **diagnostics,
            "status": "ok"
            if ranking_metric_is_defined(y_test, class_labels)
            else "unavailable",
            "unavailable_reason": ""
            if ranking_metric_is_defined(y_test, class_labels)
            else "test fold missing classes",
            "scope": scope,
            "target": target,
            "model": model_name,
            "fold": fold_index,
            "n_train": len(train_index),
            "n_test": len(test_index),
            "n_groups_train": len(np.unique(groups[train_index])),
            "n_groups_test": len(np.unique(groups[test_index])),
            "class_distribution_test": json.dumps(
                dict(Counter(str(value) for value in y_raw[test_index])),
                sort_keys=True,
            ),
            "positive_rate_test": finite_metric(prevalence),
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "auroc_macro": finite_metric(macro_auroc),
            "average_precision_macro": finite_metric(auprc),
            "baseline_average_precision": finite_metric(
                prevalence if target in BINARY_TARGETS else 1.0 / len(class_labels)
            ),
        }
        if fold_components is not None:
            # Provenance for the fold-local projection; absent when the caller
            # supplied features that were already reduced.
            fold_row["pca_components_fold"] = int(fold_components)
        rows.append(fold_row)
    return rows


def eligible_observations(rows, target):
    """Common input cohort for text alone and text plus sampled declared label."""

    def eligible(row):
        if not str(row.get("requirement", "")).strip() or row.get(
            "pred_modality"
        ) not in ("mandatory", "recommended", "optional", "nice_to_have"):
            return False
        if not row.get("seed_id") or not row.get("dataset_id"):
            return False
        if target.startswith("deterministic_"):
            return row.get("deterministic_text_modality_parse_status") == "ok" and str(
                row.get(target, "")
            ) in {"0", "1"}
        if target.startswith("sample_"):
            return sample_text_fields(row).get("text_modality_parse_status") == "ok"
        return str(row.get(target, "")) not in {"", "unknown", "None"}

    return np.asarray([eligible(row) for row in rows], dtype=bool)


def prediction_summary(predictions, folds, *, iterations=1000, seed=20260527):
    """Equal mean over evaluable held-out folds; capability percentile bootstrap.

    Fixed OOF predictions/splits are resampled, not refitted. Each draw uses the
    same capability multiplicity across all samples, models and cells. A draw
    missing any class in an evaluated fold is undefined, counted, and omitted.
    This quantifies held-out cohort uncertainty conditional on training/splits;
    it excludes retraining, split-selection and model-generation uncertainty.
    """
    evaluated = {
        int(r["fold"])
        for r in folds
        if r.get("status", "ok") == "ok" and r.get("auroc_macro", "") != ""
    }
    rows = [r for r in predictions if int(r["fold"]) in evaluated]
    result = {
        "folds": len(evaluated),
        "folds_requested": max(
            (int(r.get("n_splits_requested", len(folds))) for r in folds), default=0
        ),
        "partial_evaluation": any(r.get("status", "ok") != "ok" for r in folds),
        "evaluated_fold_ids": json.dumps(sorted(evaluated)),
        "n_eligible": max((int(r.get("n_eligible", 0)) for r in folds), default=0),
        "class_counts_eligible": next(
            (r.get("class_distribution_eligible", "{}") for r in folds), "{}"
        ),
        "class_order": next((r.get("class_order", "[]") for r in folds), "[]"),
        "n_evaluated_samples": len(rows),
        "n_evaluated_capabilities": len({r["capability_id"] for r in rows}),
        "class_counts": json.dumps(
            dict(Counter(str(r["target_value"]) for r in rows)), sort_keys=True
        ),
        "averaging": "equal mean of evaluable folds; binary positive-class AP; multiclass macro OVR AUROC/AP",
        "resampling_unit": "capability across samples, models and keyword cells",
        "uncertainty_scope": "fixed held-out predictions and folds; no refitting or generation uncertainty",
        "bootstrap_samples": iterations,
        "bootstrap_seed": seed,
        "fit_review_required": any(
            str(r.get("iteration_limit_reached", "")).lower() == "true"
            or str(r.get("convergence_warning", "")).lower() == "true"
            or str(r.get("no_training_loss_improvement", "")).lower() == "true"
            or str(r.get("insufficient_split_samples", "")).lower() == "true"
            for r in folds
        ),
        "unavailable_reason": "; ".join(
            sorted({r.get("unavailable_reason", "") for r in folds} - {""})
        ),
    }
    metrics = ("auroc", "auprc")
    for metric in metrics:
        result[f"{metric}_mean"] = result[f"{metric}_ci_low"] = result[
            f"{metric}_ci_high"
        ] = ""
    result.update(baseline_auprc="", bootstrap_valid=0, ci_unavailable_reason="")
    if not rows:
        result["unavailable_reason"] = (
            result["unavailable_reason"] or "no evaluable held-out predictions"
        )
        result["ci_unavailable_reason"] = "no evaluable held-out predictions"
        return result
    names = json.loads(rows[0]["class_order"])
    classes = np.arange(len(names))
    y = np.asarray([names.index(str(r["target_value"])) for r in rows])
    probabilities = np.asarray([json.loads(r["probabilities"]) for r in rows])
    fold_ids = np.asarray([int(r["fold"]) for r in rows])
    capabilities, group_index = np.unique(
        [r["capability_id"] for r in rows], return_inverse=True
    )
    binary = rows[0]["target"] in BINARY_TARGETS

    def score(weights):
        values, baselines = [], []
        for fold in sorted(evaluated):
            index = np.repeat(
                np.flatnonzero(fold_ids == fold), weights[fold_ids == fold]
            )
            yt, pt = y[index], probabilities[index]
            if not ranking_metric_is_defined(yt, classes):
                return None
            values.append(
                (
                    probe_auroc(yt, pt, classes),
                    float(average_precision_score(yt, pt[:, 1]))
                    if binary
                    else probe_average_precision(yt, pt, classes),
                )
            )
            baselines.append(float(np.mean(yt)) if binary else 1 / len(classes))
        return np.mean(values, axis=0), float(np.mean(baselines))

    point, baseline = score(np.ones(len(rows), dtype=int))
    result.update(
        auroc_mean=float(point[0]),
        auprc_mean=float(point[1]),
        baseline_auprc=baseline,
        class_order=json.dumps(names),
    )
    if len(capabilities) < 2 or iterations <= 0:
        result["ci_unavailable_reason"] = (
            "insufficient capabilities"
            if len(capabilities) < 2
            else "bootstrap disabled"
        )
        return result
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        counts = rng.multinomial(
            len(capabilities), np.full(len(capabilities), 1 / len(capabilities))
        )
        value = score(counts[group_index])
        if value is not None:
            samples.append(value[0])
    result["bootstrap_valid"] = len(samples)
    if len(samples) < max(2, math.ceil(iterations * 0.9)):
        result["ci_unavailable_reason"] = (
            "fewer than 90% defined bootstrap draws (missing fold classes)"
        )
        return result
    bounds = np.quantile(samples, [0.025, 0.975], axis=0)
    for i, metric in enumerate(metrics):
        result[f"{metric}_ci_low"], result[f"{metric}_ci_high"] = map(
            float, bounds[:, i]
        )
    return result


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
        grouped.setdefault(
            (str(row["scope"]), str(row["target"]), str(row["model"])), []
        ).append(row)
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
            output[f"{metric}_mean"] = finite_metric(
                float(np.mean(values)) if values else math.nan
            )
            output[f"{metric}_std"] = finite_metric(
                float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            )
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
    return X[mask], [row for row, keep in zip(rows, mask, strict=True) if keep]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run grouped supervised probes on cached ACSE embeddings."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs") / eu.ACSE_SEMANTIC_MANIFEST_FILENAME,
    )
    parser.add_argument("--backend-prefix", default="mlx:")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/acse_embedding_separability_probe"),
    )
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--within-targets", nargs="+", default=DEFAULT_WITHIN_TARGETS)
    parser.add_argument("--extra-scopes", nargs="*", default=[])
    parser.add_argument(
        "--models", nargs="+", default=["logreg", "hgb"], choices=["logreg", "hgb"]
    )
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--hgb-max-iter", type=int, default=300)
    parser.add_argument(
        "--hgb-budgets",
        type=int,
        nargs="+",
        default=None,
        help="Optional finite budgets selected with inner capability validation, e.g. 300 600 1200",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--group-mode", choices=["seed", "item"], default="seed")
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument(
        "--text-conditions",
        nargs="+",
        default=DEFAULT_TEXT_CONDITIONS,
        choices=[REQUIREMENT_ONLY_CONDITION, PREFIXED_CONTROL_CONDITION],
        help=(
            f"{REQUIREMENT_ONLY_CONDITION} is the primary diagnostic; "
            f"{PREFIXED_CONTROL_CONDITION} adds the sampled declared modality"
        ),
    )
    parser.add_argument(
        "--reqonly-cache",
        type=Path,
        default=None,
        help=(
            "requirement-only embedding cache (default: "
            "<output-dir>/task2_reqonly_mlx_embeddings.npz); point it at the "
            "embedding-diagnostic cache to reuse an existing re-embedding"
        ),
    )
    parser.add_argument("--mlx-batch", type=int, default=256)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="force re-embedding requirement-only text",
    )
    parser.add_argument("--random-state", type=int, default=20260527)
    args = parser.parse_args()

    root = eu.project_root()
    manifest_path = (
        args.manifest if args.manifest.is_absolute() else root / args.manifest
    )
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    )
    reqonly_cache = (
        args.reqonly_cache or output_dir / "task2_reqonly_mlx_embeddings.npz"
    )
    if not reqonly_cache.is_absolute():
        reqonly_cache = root / reqonly_cache
    rows = manifest_rows(manifest_path, args.backend_prefix)
    cached_prefixed, sample_rows = load_embeddings_and_rows(rows)
    add_probe_labels(sample_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    condition_features = build_text_conditions(
        sample_rows,
        cached_prefixed,
        conditions=args.text_conditions,
        mlx_batch=args.mlx_batch,
        cache_path=reqonly_cache,
        reuse_cache=not args.no_cache,
    )

    scopes = ["global"]
    scopes.extend(
        f"dataset_variant={value}"
        for value in sorted({row["dataset_variant"] for row in sample_rows})
    )
    scopes.extend(args.extra_scopes)

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    with (output_dir / "held_out_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as prediction_file:
        for condition, features in condition_features.items():
            condition_folds: list[dict[str, Any]] = []
            for scope in scopes:
                X_scope, rows_scope = filtered_scope(features, sample_rows, scope)
                target_names = (
                    args.targets if scope == "global" else args.within_targets
                )
                if not rows_scope:
                    continue
                groups = group_values(rows_scope, args.group_mode)
                for target in target_names:
                    eligible = eligible_observations(rows_scope, target)
                    for matrix in condition_features.values():
                        matrix_scope, _ = filtered_scope(matrix, sample_rows, scope)
                        eligible &= np.isfinite(matrix_scope).all(axis=1)
                    target_rows = [
                        r for r, keep in zip(rows_scope, eligible, strict=True) if keep
                    ]
                    y = target_values(target_rows, target)
                    for model_name in args.models:
                        predictions = []
                        results = fold_metrics(
                            X=X_scope[eligible],
                            y_raw=y,
                            groups=groups[eligible],
                            target=target,
                            model_name=model_name,
                            hgb_max_iter=args.hgb_max_iter,
                            hgb_budgets=args.hgb_budgets,
                            scope=scope,
                            n_splits=args.n_splits,
                            random_state=args.random_state,
                            pca_components=args.pca_components,
                            sample_rows=target_rows,
                            prediction_rows=predictions,
                            expected_classes=sorted(
                                {
                                    str(r[target])
                                    for r in rows_scope
                                    if r.get(target, "") != ""
                                }
                            )
                            if target in MULTICLASS_TARGETS
                            else None,
                        )
                        condition_folds.extend(results)
                        for row in predictions:
                            prediction_file.write(
                                json.dumps(
                                    stamp_text_condition(row, condition), sort_keys=True
                                )
                                + "\n"
                            )
                        prediction_file.flush()
                        summary_rows.append(
                            stamp_text_condition(
                                {
                                    "scope": scope,
                                    "target": target,
                                    "model": model_name,
                                    **prediction_summary(
                                        predictions,
                                        results,
                                        iterations=args.bootstrap_samples,
                                        seed=args.random_state,
                                    ),
                                },
                                condition,
                            )
                        )
            fold_rows.extend(
                stamp_text_condition(row, condition) for row in condition_folds
            )

    fold_path = output_dir / "fold_metrics.csv"
    summary_path = output_dir / "summary.csv"
    eu.write_csv_rows(fold_path, fold_rows)
    eu.write_csv_rows(summary_path, summary_rows)
    (output_dir / "summary.md").write_text(
        eu.markdown_table(summary_rows, (list(summary_rows[0]) if summary_rows else []))
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "backend_prefix": args.backend_prefix,
        "embedding_backend_rows": len(rows),
        "sample_rows": len(sample_rows),
        "primary_text_condition": PRIMARY_TEXT_CONDITION,
        "text_conditions": {
            condition: {
                "role": TEXT_CONDITION_ROLES[condition],
                "feature_shape": list(features.shape),
            }
            for condition, features in condition_features.items()
        },
        "requirement_only_embedding_cache": str(reqonly_cache),
        "feature_projection": "pca",
        # Fitted per fold on training rows only, so a held-out fold never helps
        # choose the axes it is scored in.
        "pca_fit_scope": "train_rows_per_fold",
        "pca_components_requested": int(args.pca_components),
        "models": args.models,
        "targets": args.targets,
        "within_targets": args.within_targets,
        "n_splits": args.n_splits,
        "group_mode": args.group_mode,
        "random_state": args.random_state,
        "artifacts": [
            str(fold_path),
            str(summary_path),
            str(output_dir / "summary.md"),
        ],
    }
    eu.write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary_rows": len(summary_rows),
                **manifest,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
