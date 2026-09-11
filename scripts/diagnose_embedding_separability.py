"""Evaluate matched input conditions on grouped held-out predictions.

Sampled text predicts strict strengthening in the corresponding separate
single-pass output. The sampled declared label is additional input. Auxiliary
source-modality and dataset-by-keyword targets are kept separate. Learned
preprocessing is fitted only within training folds.
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
        eligible_observations,
        embed_requirement_only,
        fold_metrics,
        group_values,
        prediction_summary,
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
        eligible_observations,
        embed_requirement_only,
        fold_metrics,
        group_values,
        prediction_summary,
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

    # --- Cached embeddings of text plus sampled declared label ---
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
    prediction_rows: list[dict[str, Any]] | None = None,
    cell_callback=None,
    hgb_max_iter: int = 300,
    hgb_budgets: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Score every feature set, with reduction and vectorization inside the fold."""
    source_modalities = np.asarray(
        [str(row.get("source_modality", "")) for row in sample_rows], dtype=object
    )
    fold_rows: list[dict[str, Any]] = []
    common_features = np.ones(len(sample_rows), dtype=bool)
    for payload in features.values():
        if np.asarray(payload["X"]).dtype != object:
            common_features &= np.isfinite(payload["X"]).all(axis=1)
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
                    eligible = (
                        eligible_observations(rows_scope, target)
                        & common_features[mask]
                    )
                    eligible_rows = [
                        r for r, keep in zip(rows_scope, eligible, strict=True) if keep
                    ]
                    y = target_values(eligible_rows, target)
                    for model_name in models:
                        predictions = []
                        results = fold_metrics(
                            X=X_scope[eligible],
                            y_raw=y,
                            groups=groups_scope[eligible],
                            target=target,
                            model_name=model_name,
                            hgb_max_iter=hgb_max_iter,
                            hgb_budgets=hgb_budgets,
                            scope=scope_name,
                            n_splits=n_splits,
                            random_state=random_state,
                            pca_components=pca_components,
                            text_vectorizer=payload["text_vectorizer"],
                            sample_rows=eligible_rows,
                            prediction_rows=predictions,
                            expected_classes=sorted(
                                {
                                    str(r[target])
                                    for r in rows_scope
                                    if r.get(target, "") != ""
                                }
                            )
                            if target in ("source_modality", "dataset_variant")
                            else None,
                        )
                        metadata = {
                            "feature_backend": payload["backend"],
                            "text_variant": payload["text"],
                            "feature_key": feature_key,
                            "group_mode": group_mode,
                        }
                        predictions = [{**r, **metadata} for r in predictions]
                        if prediction_rows is not None:
                            prediction_rows.extend(predictions)
                        for row in results:
                            row["n_excluded_samples"] = len(rows_scope) - len(
                                eligible_rows
                            )
                            row.update(
                                {
                                    "feature_backend": payload["backend"],
                                    "text_variant": payload["text"],
                                    "feature_key": feature_key,
                                    "group_mode": group_mode,
                                }
                            )
                            fold_rows.append(row)
                        if cell_callback is not None:
                            cell_callback(results, predictions)
            print(
                f"[grid] {feature_key} group={group_mode}: cumulative fold rows={len(fold_rows)}"
            )
    return fold_rows


def summarize_grid(
    fold_rows: list[dict[str, Any]],
    prediction_rows=None,
    *,
    iterations=1000,
    seed=20260527,
) -> list[dict[str, Any]]:
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
                # Filled below, once prediction_summary has had its say on
                # auprc_mean and baseline_auprc.
                "auprc_lift_over_baseline": "",
                "balanced_accuracy_mean": base.get("balanced_accuracy_mean", ""),
                "positive_rate": base.get("positive_rate_test_mean", ""),
                "n_excluded_samples": rows[0].get("n_excluded_samples", 0),
                **(
                    prediction_summary(
                        [
                            r
                            for r in prediction_rows
                            if (
                                r["feature_key"],
                                r["group_mode"],
                                r["scope"],
                                r["target"],
                                r["classifier"],
                            )
                            == (feature_key, group_mode, scope, target, model)
                        ],
                        rows,
                        iterations=iterations,
                        seed=seed,
                    )
                    if prediction_rows is not None
                    else {}
                ),
            }
        )
    for row in summary:
        ap, baseline = row.get("auprc_mean", ""), row.get("baseline_auprc", "")
        row["auprc_lift_over_baseline"] = (
            float(ap) / float(baseline)
            if ap != "" and baseline != "" and float(baseline) > 0
            else ""
        )
        if row["target"].endswith("text_overcommit"):
            row["positive_rate"] = baseline
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
    parser.add_argument("--hgb-max-iter", type=int, default=300)
    parser.add_argument(
        "--hgb-budgets",
        type=int,
        nargs="+",
        default=None,
        help="Optional finite budgets selected with inner capability validation, e.g. 300 600 1200",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
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

    summary = []
    # Keep only one grid cell's held-out predictions in memory. The full cohort
    # generates millions of records across representations, scopes and targets.
    with (output_dir / "probe_grid_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as prediction_file:

        def save_cell(folds, predictions):
            for row in predictions:
                prediction_file.write(json.dumps(row, sort_keys=True) + "\n")
            prediction_file.flush()
            summary.extend(
                summarize_grid(
                    folds,
                    predictions,
                    iterations=args.bootstrap_samples,
                    seed=args.random_state,
                )
            )
            print(
                f"[summary] {folds[0]['feature_key']} {folds[0]['scope']} {folds[0]['target']}: {len(predictions)} predictions exported",
                flush=True,
            )

        fold_rows = run_grid(
            features,
            sample_rows,
            models=args.models,
            n_splits=args.n_splits,
            random_state=args.random_state,
            pca_components=args.pca_components,
            cell_callback=save_cell,
            hgb_max_iter=args.hgb_max_iter,
            hgb_budgets=args.hgb_budgets,
        )

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
        "hgb_max_iter": args.hgb_max_iter,
        "hgb_budgets": args.hgb_budgets,
        "bootstrap_samples": args.bootstrap_samples,
        "held_out_predictions": "probe_grid_predictions.jsonl",
        "uncertainty_scope": "capability bootstrap of fixed predictions; conditional on fitted models and splits",
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
