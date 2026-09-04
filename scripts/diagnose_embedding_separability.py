"""Diagnostic probe: does embedding *geometry* encode modal-force drift, or only
the input commitment level and corpus identity it is confounded with?

This extends ``probe_acse_embedding_separability`` along three axes the paper
under-reports:

1. text variant -- ``prefixed`` (the cached ``modality: <label> requirement: ...``
   string, which leaks the predicted label) vs ``reqonly`` (the generated
   requirement text alone, the honest substrate for "what the wording reveals").
2. backend -- ``mlx`` (neural meaning embedding) vs ``tfidf`` (char n-gram, a
   near-oracle for the lexical modal keyword that *defines* strict overcommit).
3. grouping -- ``seed`` (an item's text straddles folds) vs ``item`` (a held-out
   item is genuinely unseen; the conservative test).

For every cell it reports AUROC *and* AUPRC against the prevalence baseline, so
the rare-positive collapse is visible as lift-over-chance rather than hidden
behind an AUROC near 0.5.

Nothing is fitted before the fold split. Dimensionality reduction and the TF-IDF
vocabulary are pipeline steps, so both are fitted on the training rows of each
fold; fitting either once over all observations would let a held-out fold help
choose the axes and the vocabulary it is later scored in, which inflates every
number in ``probe_grid_summary.csv`` -- the file the paper figure and
``export_paper_numbers.py`` read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu

try:
    from plot_acse_global_embedding_projection import (
        load_embeddings_and_rows,
        manifest_rows,
    )
    from probe_acse_embedding_separability import (
        add_probe_labels,
        embed_requirement_only,
        fold_metrics,
        group_values,
        summarize,
        target_values,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.plot_acse_global_embedding_projection import (
        load_embeddings_and_rows,
        manifest_rows,
    )
    from scripts.probe_acse_embedding_separability import (
        add_probe_labels,
        embed_requirement_only,
        fold_metrics,
        group_values,
        summarize,
        target_values,
    )


GLOBAL_TARGETS = [
    "source_modality",
    "dataset_variant",
    "deterministic_strict_text_overcommit",
    "sample_strict_text_overcommit",
]
WITHIN_TARGETS = [
    "deterministic_strict_text_overcommit",
    "sample_strict_text_overcommit",
]
WITHIN_SCOPES = ["recommended", "optional", "nice_to_have"]


def build_feature_sets(
    sample_rows: list[dict[str, Any]],
    cached_mlx_prefixed: np.ndarray,
    *,
    mlx_batch: int,
    cache_dir: Path,
    reuse_cache: bool,
) -> dict[str, dict[str, Any]]:
    """The four substrates, unreduced and unvectorized.

    Nothing is fitted here. Dimensionality reduction and the TF-IDF vocabulary
    are pipeline steps that ``run_grid`` hands to ``fold_metrics``, so they are
    fitted on the training rows of each fold; fitting them once over all
    observations would let a held-out fold help choose the axes and the
    vocabulary it is later scored in.
    """
    semantic_texts = [str(row.get("semantic_text", "")) for row in sample_rows]
    requirements = [str(row.get("requirement", "")) for row in sample_rows]
    features: dict[str, dict[str, Any]] = {}

    # --- MLX prefixed: cached embeddings (label leaks into the string) ---
    features["mlx::prefixed"] = {
        "X": np.asarray(cached_mlx_prefixed, dtype=np.float32),
        "text_vectorizer": None,
        "backend": "mlx",
        "text": "prefixed",
    }
    print(f"[mlx::prefixed] {features['mlx::prefixed']['X'].shape}")

    # --- MLX requirement-only: re-embed the generated wording alone ---
    reqonly = embed_requirement_only(
        requirements,
        batch_size=mlx_batch,
        cache_path=cache_dir / "task2_reqonly_mlx_embeddings.npz",
        reuse_cache=reuse_cache,
    )
    features["mlx::reqonly"] = {
        "X": np.asarray(reqonly, dtype=np.float32),
        "text_vectorizer": None,
        "backend": "mlx",
        "text": "reqonly",
    }
    print(f"[mlx::reqonly] {features['mlx::reqonly']['X'].shape}")

    # --- TF-IDF char n-gram: prefixed vs requirement-only ---
    for tag, corpus in (("prefixed", semantic_texts), ("reqonly", requirements)):
        features[f"tfidf::{tag}"] = {
            "X": np.asarray(
                [text if text else "<empty response>" for text in corpus], dtype=object
            ),
            "text_vectorizer": TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), lowercase=True
            ),
            "backend": "tfidf",
            "text": tag,
        }
        print(f"[tfidf::{tag}] {len(corpus)} texts, vocabulary fitted per fold")

    return features


def run_grid(
    features: dict[str, dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    *,
    models: list[str],
    n_splits: int,
    random_state: int,
    pca_components: int,
) -> list[dict[str, Any]]:
    """Score every feature set, with reduction and vectorization inside the fold."""
    source_modalities = np.asarray(
        [str(row.get("source_modality", "")) for row in sample_rows], dtype=object
    )
    fold_rows: list[dict[str, Any]] = []
    for feature_key, payload in features.items():
        X_full = payload["X"]
        for group_mode in ("seed", "item"):
            groups_full = group_values(sample_rows, group_mode)
            scopes: list[tuple[str, np.ndarray]] = [
                ("global", np.ones(len(sample_rows), dtype=bool))
            ]
            for modality in WITHIN_SCOPES:
                scopes.append(
                    (f"source_modality={modality}", source_modalities == modality)
                )
            for scope_name, mask in scopes:
                targets = GLOBAL_TARGETS if scope_name == "global" else WITHIN_TARGETS
                X_scope = X_full[mask]
                rows_scope = [
                    row for row, keep in zip(sample_rows, mask, strict=True) if keep
                ]
                groups_scope = groups_full[mask]
                for target in targets:
                    y = target_values(rows_scope, target)
                    if len(np.unique(y)) < 2:
                        continue
                    for model_name in models:
                        for row in fold_metrics(
                            X=X_scope,
                            y_raw=y,
                            groups=groups_scope,
                            target=target,
                            model_name=model_name,
                            scope=scope_name,
                            n_splits=n_splits,
                            random_state=random_state,
                            pca_components=pca_components,
                            text_vectorizer=payload["text_vectorizer"],
                        ):
                            row.update(
                                {
                                    "feature_backend": payload["backend"],
                                    "text_variant": payload["text"],
                                    "feature_key": feature_key,
                                    "group_mode": group_mode,
                                }
                            )
                            fold_rows.append(row)
            print(
                f"[grid] {feature_key} group={group_mode}: cumulative fold rows={len(fold_rows)}"
            )
    return fold_rows


def summarize_grid(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[tuple, list[dict[str, Any]]] = {}
    for row in fold_rows:
        key = (
            row["feature_key"],
            row["group_mode"],
            row["scope"],
            row["target"],
            row["model"],
        )
        by_cell.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for (feature_key, group_mode, scope, target, model), rows in sorted(
        by_cell.items()
    ):
        base = summarize(rows)[0]
        backend, text_variant = feature_key.split("::")
        auprc = base.get("average_precision_macro_mean", "")
        baseline = base.get("baseline_average_precision_mean", "")
        lift = ""
        if isinstance(auprc, float) and isinstance(baseline, float) and baseline > 0:
            lift = round(auprc / baseline, 3)
        summary.append(
            {
                "feature_backend": backend,
                "text_variant": text_variant,
                "group_mode": group_mode,
                "scope": scope,
                "target": target,
                "model": model,
                "folds": base["folds"],
                "auroc_mean": base.get("auroc_macro_mean", ""),
                "auroc_std": base.get("auroc_macro_std", ""),
                "auprc_mean": auprc,
                "baseline_auprc": baseline,
                "auprc_lift_over_baseline": lift,
                "balanced_accuracy_mean": base.get("balanced_accuracy_mean", ""),
                "positive_rate": base.get("positive_rate_test_mean", ""),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs") / eu.ACSE_SEMANTIC_MANIFEST_FILENAME,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/embedding_diagnostic")
    )
    parser.add_argument(
        "--models", nargs="+", default=["hgb", "logreg"], choices=["hgb", "logreg"]
    )
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--mlx-batch", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=20260527)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="force re-embedding requirement-only text",
    )
    args = parser.parse_args()

    root = eu.project_root()
    manifest_path = (
        args.manifest if args.manifest.is_absolute() else root / args.manifest
    )
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = manifest_rows(manifest_path, "mlx:")
    cached_mlx_prefixed, sample_rows = load_embeddings_and_rows(rows)
    add_probe_labels(sample_rows)
    print(
        f"loaded {len(sample_rows)} samples; cached mlx prefixed {cached_mlx_prefixed.shape}"
    )

    features = build_feature_sets(
        sample_rows,
        cached_mlx_prefixed,
        mlx_batch=args.mlx_batch,
        cache_dir=output_dir,
        reuse_cache=not args.no_cache,
    )
    del cached_mlx_prefixed

    fold_rows = run_grid(
        features,
        sample_rows,
        models=args.models,
        n_splits=args.n_splits,
        random_state=args.random_state,
        pca_components=args.pca_components,
    )
    summary = summarize_grid(fold_rows)

    eu.write_csv_rows(output_dir / "probe_grid_folds.csv", fold_rows)
    eu.write_csv_rows(output_dir / "probe_grid_summary.csv", summary)
    (output_dir / "probe_grid_summary.md").write_text(
        eu.markdown_table(summary, (list(summary[0]) if summary else [])) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "n_samples": len(sample_rows),
        "models": args.models,
        "pca_components": args.pca_components,
        "n_splits": args.n_splits,
        # Fitted per fold, so there is no single explained-variance figure to
        # report and no globally fitted vocabulary to size.
        "reduction": "fold-local",
        "text_vectorizer": "fold-local",
        "global_targets": GLOBAL_TARGETS,
        "within_targets": WITHIN_TARGETS,
        "within_scopes": WITHIN_SCOPES,
        "random_state": args.random_state,
    }
    eu.write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {"output_dir": str(output_dir), "summary_rows": len(summary), **manifest},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
