"""Probe whether ACSE embeddings separate dataset, modality, or drift labels.

The probe runs one or more **text conditions**:

* ``requirement_only`` (the primary diagnostic) embeds the generated requirement
  wording alone. Nothing about the predicted modality is written into the string,
  so a strengthening probe here measures what the wording itself reveals.
* ``prefixed_leakage_control`` reuses the cached ACSE embeddings, whose text comes
  from ``eval_utils.semantic_response_text()`` and literally begins
  ``modality: <predicted label>``. Strengthening is derived partly from that same
  predicted label, so this condition is a **positive control**: it shows what the
  score looks like when the answer is inside the input. It is never the headline
  number, and every artifact names it as a control.

Dimensionality reduction is fitted **inside each cross-validation fold, on the
training rows only** (PCA is the first step of the estimator pipeline). Fitting it
once over all observations would let a held-out fold help choose the axes it is
later scored in.
"""

from __future__ import annotations

import argparse
import hashlib
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

# --- Text conditions ---------------------------------------------------------
# The predicted modality is an ingredient of the strengthening label, so a
# substrate that spells that modality out cannot answer "what does the wording
# reveal?". Requirement-only text is therefore the primary condition and the
# label-prefixed cache is demoted to a named positive control.
REQUIREMENT_ONLY_CONDITION = "requirement_only"
PREFIXED_CONTROL_CONDITION = "prefixed_leakage_control"
PRIMARY_TEXT_CONDITION = REQUIREMENT_ONLY_CONDITION
TEXT_CONDITION_ROLES = {
    REQUIREMENT_ONLY_CONDITION: "primary",
    PREFIXED_CONTROL_CONDITION: "positive_control_label_leakage",
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
            "1" if eu.is_truthy_strict(row.get("strict_text_overcommit", "")) else "0"
        )
        row["deterministic_broad_text_overcommit"] = (
            "1" if eu.is_truthy_strict(row.get("text_overcommit", "")) else "0"
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


def embed_requirement_only(
    requirements: list[str],
    *,
    batch_size: int,
    cache_path: Path,
    reuse_cache: bool,
) -> np.ndarray:
    """Embed each row's requirement text with MLX, deduplicating first."""
    # Bind the cache to the exact requirement texts (and backend), not just the row
    # count, so a stale cache from a different run of equal length is not reused.
    requirements_digest = hashlib.sha256(
        ("mlx\x00" + "\x00".join(requirements)).encode("utf-8")
    ).hexdigest()
    if reuse_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_digest = (
            str(cached["requirements_digest"])
            if "requirements_digest" in cached.files
            else ""
        )
        if (
            int(cached["n_rows"]) == len(requirements)
            and cached_digest == requirements_digest
        ):
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
        requirements_digest=np.asarray(requirements_digest),
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
                max_iter=100,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                l2_regularization=0.05,
                class_weight="balanced",
                random_state=random_state,
            )
        ]
    raise ValueError(f"Unknown model: {model_name}")


def make_estimator(
    model_name: str, random_state: int, *, pca_components: int | None = None
) -> Any:
    """Build the probe estimator, optionally with PCA as its first pipeline step.

    Putting PCA in the pipeline is what makes the projection fold-local: the
    estimator is fitted on the training rows of one fold, so the axes the held-out
    rows are transformed into were chosen without them. Callers that pass features
    which are already reduced leave ``pca_components`` at ``None``.
    """
    steps = model_steps(model_name, random_state)
    if pca_components is not None:
        steps.insert(
            0,
            PCA(
                n_components=pca_components,
                svd_solver="randomized",
                random_state=random_state,
            ),
        )
    return steps[0] if len(steps) == 1 else make_pipeline(*steps)


def finite_metric(value: float) -> float | str:
    return "" if math.isnan(value) else float(value)


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
) -> list[dict[str, Any]]:
    if len(np.unique(y_raw)) < 2:
        return []
    n_splits = min(n_splits, len(np.unique(groups)))
    if target in BINARY_TARGETS:
        y = y_raw.astype(int)
        class_labels = np.asarray([0, 1])
    else:
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw)
        class_labels = np.arange(len(encoder.classes_))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    rows: list[dict[str, Any]] = []
    for fold_index, (train_index, test_index) in enumerate(
        splitter.split(X, y, groups=groups)
    ):
        # PCA cannot exceed the rank the training fold can support.
        fold_components = (
            None
            if pca_components is None
            else max(1, min(pca_components, len(train_index), X.shape[1]))
        )
        estimator = make_estimator(
            model_name, random_state + fold_index, pca_components=fold_components
        )
        estimator.fit(X[train_index], y[train_index])
        pred = estimator.predict(X[test_index])
        probability_matrix = ordered_class_probabilities(
            estimator.predict_proba(X[test_index]), estimator.classes_, class_labels
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
                prevalence if target in BINARY_TARGETS else math.nan
            ),
        }
        if fold_components is not None:
            # Provenance for the fold-local projection; absent when the caller
            # supplied features that were already reduced.
            fold_row["pca_components_fold"] = int(fold_components)
        rows.append(fold_row)
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
    parser.add_argument("--group-mode", choices=["seed", "item"], default="seed")
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument(
        "--text-conditions",
        nargs="+",
        default=DEFAULT_TEXT_CONDITIONS,
        choices=[REQUIREMENT_ONLY_CONDITION, PREFIXED_CONTROL_CONDITION],
        help=(
            f"{REQUIREMENT_ONLY_CONDITION} is the primary diagnostic; "
            f"{PREFIXED_CONTROL_CONDITION} is a positive control whose text spells "
            "out the predicted modality and must not be read as a headline result"
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
    for condition, features in condition_features.items():
        condition_folds: list[dict[str, Any]] = []
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
                    condition_folds.extend(
                        fold_metrics(
                            X=X_scope,
                            y_raw=y,
                            groups=groups,
                            target=target,
                            model_name=model_name,
                            scope=scope,
                            n_splits=args.n_splits,
                            random_state=args.random_state,
                            pca_components=args.pca_components,
                        )
                    )
        fold_rows.extend(
            stamp_text_condition(row, condition) for row in condition_folds
        )
        summary_rows.extend(
            stamp_text_condition(row, condition) for row in summarize(condition_folds)
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
