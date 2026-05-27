"""Project cached ACSE sample embeddings from all completed runs into one space."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "re_uq_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
_XDG_CACHE_HOME = Path(tempfile.gettempdir()) / "re_uq_cache"
(_XDG_CACHE_HOME / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu


GROUP_COLORS = {
    "mlm_tapt/must": "#2563eb",
    "mlm_tapt/shall": "#0891b2",
    "nice/must": "#dc2626",
    "nice/shall": "#d97706",
}
SOURCE_COLORS = {
    "mandatory": "#111827",
    "recommended": "#2563eb",
    "optional": "#7c3aed",
    "nice_to_have": "#16a34a",
}
DRIFT_COLORS = {
    "strict_text_oc": "#b91c1c",
    "broad_text_oc": "#f97316",
    "clean": "#64748b",
}


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def drift_status(row: dict[str, str]) -> str:
    if truthy(row.get("strict_text_overcommit", "")):
        return "strict_text_oc"
    if truthy(row.get("text_overcommit", "")):
        return "broad_text_oc"
    return "clean"


def read_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def is_tfidf_backend(backend_label: str) -> bool:
    return backend_label in {"tfidf", eu.ACSE_PROXY_EMBEDDING_BACKEND}


def manifest_rows(path: Path, backend_prefix: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("embedding_backend", "")).startswith(backend_prefix)
        ]
    if not rows:
        raise ValueError(f"No manifest rows found for backend prefix {backend_prefix!r} in {path}.")
    return rows


def load_embeddings_and_rows(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    matrices: list[np.ndarray] = []
    sample_rows: list[dict[str, Any]] = []
    offset = 0
    needs_shared_tfidf_refit = False
    backend_labels: set[str] = set()
    for manifest_row in rows:
        artifact_dir = Path(manifest_row["artifact_dir"])
        embeddings_path = artifact_dir / "task2_acse_sample_embeddings.npz"
        samples_path = artifact_dir / "task2_acse_samples.csv"
        items_path = artifact_dir / "task2_acse_items.csv"
        missing = [path for path in [embeddings_path, samples_path, items_path] if not path.exists()]
        if missing:
            raise FileNotFoundError("Incomplete ACSE cache: " + ", ".join(str(path) for path in missing))

        embeddings = np.load(embeddings_path, allow_pickle=False)["embeddings"].astype(np.float32, copy=False)
        with samples_path.open(newline="", encoding="utf-8") as handle:
            samples = list(csv.DictReader(handle))
        if embeddings.shape[0] != len(samples):
            raise ValueError(f"Embedding/sample row mismatch in {artifact_dir}: {embeddings.shape[0]} != {len(samples)}")
        backend_label = str(manifest_row.get("embedding_backend", ""))
        backend_labels.add(backend_label)
        if is_tfidf_backend(backend_label):
            needs_shared_tfidf_refit = True
        else:
            matrices.append(embeddings)

        items = read_csv_by_key(items_path, "item_id")
        for local_index, sample in enumerate(samples):
            item = items.get(str(sample.get("item_id", "")), {})
            dataset_id = manifest_row["dataset_id"]
            variant = manifest_row["benchmark_variant"]
            source_modality = item.get("source_modality", "")
            status = drift_status(item)
            sample_rows.append(
                {
                    "global_embedding_index": offset + local_index,
                    "local_embedding_index": sample.get("embedding_index", local_index),
                    "dataset_id": dataset_id,
                    "benchmark_variant": variant,
                    "dataset_variant": f"{dataset_id}/{variant}",
                    "run_id": manifest_row["run_id"],
                    "model": manifest_row["model"],
                    "profile_id": manifest_row.get("profile_id", ""),
                    "item_id": sample.get("item_id", ""),
                    "seed_id": sample.get("seed_id", ""),
                    "sample_index": sample.get("sample_index", ""),
                    "pred_modality": sample.get("pred_modality", ""),
                    "source_modality": source_modality,
                    "drift_status": status,
                    "strict_text_overcommit": item.get("strict_text_overcommit", ""),
                    "text_overcommit": item.get("text_overcommit", ""),
                    "acse_uncertainty_score": item.get("acse_uncertainty_score", ""),
                    "semantic_text": sample.get("semantic_text", ""),
                    "requirement": sample.get("requirement", ""),
                }
            )
        offset += embeddings.shape[0]
    if needs_shared_tfidf_refit:
        non_tfidf = sorted(label for label in backend_labels if not is_tfidf_backend(label))
        if non_tfidf:
            raise ValueError(
                "Cannot mix per-run TF-IDF caches with fixed-width embedding caches in one global projection; "
                f"found non-TF-IDF backends: {', '.join(non_tfidf)}."
            )
        embeddings, _ = eu.semantic_embedding_matrix(
            [str(row.get("semantic_text", "")) for row in sample_rows],
            embedding_backend="tfidf",
        )
        for row in sample_rows:
            row["global_embedding_source"] = "shared_tfidf_refit"
        return embeddings.astype(np.float32, copy=False), sample_rows
    if not matrices:
        return np.zeros((0, 1), dtype=np.float32), sample_rows
    widths = {matrix.shape[1] for matrix in matrices}
    if len(widths) != 1:
        dimensions = ", ".join(str(matrix.shape[1]) for matrix in matrices)
        raise ValueError(
            "Cached embedding widths differ across selected runs "
            f"({dimensions}); use a fixed-width backend or recompute a shared feature space."
        )
    return np.vstack(matrices), sample_rows


def plot_grouped_3d(
    rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
    color_field: str,
    color_map: dict[str, str],
    alpha: float,
    point_size: float,
    axis_prefix: str,
) -> None:
    fig = plt.figure(figsize=(10.8, 8.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for label in sorted({str(row[color_field]) for row in rows}):
        subset = [row for row in rows if str(row[color_field]) == label]
        if not subset:
            continue
        ax.scatter(
            [row["pc1"] for row in subset],
            [row["pc2"] for row in subset],
            [row["pc3"] for row in subset],
            s=point_size,
            c=color_map.get(label, "#64748b"),
            alpha=alpha,
            linewidths=0,
            label=label.replace("_", " "),
            depthshade=False,
        )
    ax.set_title(title)
    ax.set_xlabel(f"{axis_prefix}1")
    ax.set_ylabel(f"{axis_prefix}2")
    ax.set_zlabel(f"{axis_prefix}3")
    ax.legend(frameon=False, loc="upper left", markerscale=4)
    ax.view_init(elev=22, azim=38)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a global 3D projection over cached ACSE sample embeddings.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/acse_semantic_artifact_manifest.csv"))
    parser.add_argument("--backend-prefix", default="mlx:")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--tsne-prepca-components", type=int, default=50)
    parser.add_argument("--tsne-perplexity", type=float, default=50.0)
    parser.add_argument("--tsne-max-iter", type=int, default=1000)
    parser.add_argument("--tsne-angle", type=float, default=0.5)
    parser.add_argument("--tsne-n-jobs", type=int, default=-1)
    parser.add_argument("--tsne-verbose", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.055)
    parser.add_argument("--point-size", type=float, default=2.2)
    args = parser.parse_args()

    root = eu.project_root()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    backend_tag = eu.safe_identifier(args.backend_prefix.rstrip(":") or args.backend_prefix or "selected")
    backend_title = args.backend_prefix.rstrip(":") or args.backend_prefix or "selected"
    default_output = Path(
        f"outputs/acse_global_{backend_tag}_projection"
        if args.method == "pca"
        else f"outputs/acse_global_{backend_tag}_tsne_projection"
    )
    requested_output = args.output_dir or default_output
    output_dir = requested_output if requested_output.is_absolute() else root / requested_output
    rows = manifest_rows(manifest_path, args.backend_prefix)
    embeddings, sample_rows = load_embeddings_and_rows(rows)
    projection_details: dict[str, Any]
    if args.method == "pca":
        pca = PCA(n_components=3, svd_solver="randomized", random_state=args.random_state)
        projected = pca.fit_transform(embeddings)
        projection_details = {
            "projection": "pca_3d",
            "axis_prefix": "PC",
            "pca_explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        }
    else:
        prepca_components = min(args.tsne_prepca_components, embeddings.shape[0], embeddings.shape[1])
        prepca = PCA(n_components=prepca_components, svd_solver="randomized", random_state=args.random_state)
        prepared = prepca.fit_transform(embeddings)
        tsne = TSNE(
            n_components=3,
            perplexity=args.tsne_perplexity,
            learning_rate="auto",
            max_iter=args.tsne_max_iter,
            init="pca",
            method="barnes_hut",
            angle=args.tsne_angle,
            n_jobs=args.tsne_n_jobs,
            random_state=args.random_state,
            verbose=args.tsne_verbose,
        )
        projected = tsne.fit_transform(prepared)
        projection_details = {
            "projection": "tsne_3d",
            "axis_prefix": "t-SNE",
            "random_state": args.random_state,
            "tsne_preprocessing": "pca",
            "tsne_prepca_components": int(prepca_components),
            "tsne_prepca_explained_variance_sum": float(np.sum(prepca.explained_variance_ratio_)),
            "tsne_perplexity": float(args.tsne_perplexity),
            "tsne_max_iter": int(args.tsne_max_iter),
            "tsne_angle": float(args.tsne_angle),
            "tsne_n_jobs": int(args.tsne_n_jobs),
            "tsne_kl_divergence": float(tsne.kl_divergence_),
        }
    for row, coords in zip(sample_rows, projected):
        row["projection_1"] = float(coords[0])
        row["projection_2"] = float(coords[1])
        row["projection_3"] = float(coords[2])
        row["pc1"] = float(coords[0])
        row["pc2"] = float(coords[1])
        row["pc3"] = float(coords[2])

    file_prefix = f"task2_acse_global_{backend_tag}_samples_3d_{args.method}"
    projection_csv = output_dir / f"{file_prefix}.csv"
    eu.write_csv_rows(projection_csv, sample_rows)
    plot_grouped_3d(
        sample_rows,
        output_dir / f"{file_prefix}_by_dataset_variant.png",
        f"Task 2 ACSE {backend_title} embeddings: all stochastic samples by dataset/variant ({args.method.upper()})",
        "dataset_variant",
        GROUP_COLORS,
        args.alpha,
        args.point_size,
        projection_details["axis_prefix"],
    )
    plot_grouped_3d(
        sample_rows,
        output_dir / f"{file_prefix}_by_source_modality.png",
        f"Task 2 ACSE {backend_title} embeddings: all stochastic samples by source modality ({args.method.upper()})",
        "source_modality",
        SOURCE_COLORS,
        args.alpha,
        args.point_size,
        projection_details["axis_prefix"],
    )
    plot_grouped_3d(
        sample_rows,
        output_dir / f"{file_prefix}_by_text_drift.png",
        f"Task 2 ACSE {backend_title} embeddings: all stochastic samples by deterministic text drift ({args.method.upper()})",
        "drift_status",
        DRIFT_COLORS,
        args.alpha,
        args.point_size,
        projection_details["axis_prefix"],
    )
    manifest = {
        "backend_prefix": args.backend_prefix,
        "embedding_backend_rows": len(rows),
        "sample_rows": len(sample_rows),
        "embedding_shape": list(embeddings.shape),
        "artifacts": [
            str(projection_csv),
            str(output_dir / f"{file_prefix}_by_dataset_variant.png"),
            str(output_dir / f"{file_prefix}_by_source_modality.png"),
            str(output_dir / f"{file_prefix}_by_text_drift.png"),
        ],
        **projection_details,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
