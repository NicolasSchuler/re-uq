"""Visualize ACSE semantic embeddings for repeated Task 2 samples.

The ACSE metric stores cluster diagnostics, not raw vectors. This script
recomputes the same embeddings from cached stochastic raw outputs, projects
them into a shared low-dimensional space, and writes static figures plus CSVs
for inspecting generated-text drift cases.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
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

try:
    import eval_utils as eu
    from plot_acse_global_embedding_projection import DRIFT_COLORS, drift_status
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu
    from scripts.plot_acse_global_embedding_projection import DRIFT_COLORS, drift_status


SOURCE_MARKERS = {
    "mandatory": "s",
    "recommended": "^",
    "optional": "D",
    "nice_to_have": "o",
}
CLUSTER_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#d97706",
    "#0f766e",
]


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def default_analysis_dir(
    root: Path, dataset_id: str, variant: str, run_id: str
) -> Path:
    return (
        root
        / "outputs"
        / f"evaluation_{dataset_id}_{variant}_{eu.safe_identifier(run_id)}"
    )


def projection_model(embeddings: np.ndarray, components: int) -> tuple[np.ndarray, PCA]:
    max_components = max(
        1, min(int(components), embeddings.shape[0], embeddings.shape[1])
    )
    pca = PCA(n_components=max_components, random_state=0)
    projected = pca.fit_transform(embeddings)
    if max_components < components:
        padding = np.zeros(
            (projected.shape[0], components - max_components), dtype=float
        )
        projected = np.hstack([projected, padding])
    return projected, pca


def load_task2_rows(
    root: Path,
    dataset_id: str,
    variant: str,
    run_id: str,
    model: str,
    profile: str | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    raw_rows = [
        row
        for row in eu.read_jsonl(eu.model_outputs_raw_path(root, dataset_id, variant))
        if str(row.get("run_id", "")) == run_id
        and str(row.get("model", "")) == model
        and str(row.get("task", "")) == "task2"
    ]
    if profile:
        raw_rows = [
            row for row in raw_rows if str(row.get("profile_id", "")) in {"", profile}
        ]
    stochastic = [
        row
        for row in raw_rows
        if str(row.get("sample_kind", "")) == "stochastic"
        and str(row.get("parse_status", "")) == "ok"
        and isinstance(row.get("parsed_json"), dict)
    ]
    benchmark_rows = {
        row["item_id"]: row
        for row in eu.read_csv_rows(
            eu.artifact_path(
                root / "data/processed/benchmark_items.csv", dataset_id, variant
            )
        )
    }
    return stochastic, benchmark_rows


def load_cached_projection_inputs(
    cache_dir: Path,
    run_id: str,
    model: str,
    distance_threshold: float,
    components: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, str]:
    manifest_path = cache_dir / "manifest.json"
    embeddings_path = cache_dir / "task2_acse_sample_embeddings.npz"
    samples_path = cache_dir / "task2_acse_samples.csv"
    items_path = cache_dir / "task2_acse_items.csv"
    missing = [
        path
        for path in [manifest_path, embeddings_path, samples_path, items_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Incomplete ACSE embedding cache: "
            + ", ".join(str(path) for path in missing)
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    backend_label = str(manifest.get("embedding_backend", ""))
    if (
        str(manifest.get("run_id", "")) != run_id
        or str(manifest.get("model", "")) != model
    ):
        raise ValueError(
            f"Cache {cache_dir} is for run/model "
            f"{manifest.get('run_id')!r}/{manifest.get('model')!r}, not {run_id!r}/{model!r}."
        )
    cached_threshold = finite_float(manifest.get("distance_threshold", math.nan))
    if (
        not math.isnan(cached_threshold)
        and abs(cached_threshold - float(distance_threshold)) > 1e-12
    ):
        raise ValueError(
            f"Cache {cache_dir} used distance_threshold={cached_threshold}; "
            f"requested {distance_threshold}. Recompute the cache for this threshold."
        )

    npz = np.load(embeddings_path, allow_pickle=False)
    embeddings = np.asarray(npz["embeddings"], dtype=float)
    sample_rows = eu.read_csv_rows(samples_path)
    item_rows = eu.read_csv_rows(items_path)
    if embeddings.shape[0] != len(sample_rows):
        raise ValueError(
            f"Cache row mismatch: {embeddings.shape[0]} embeddings for {len(sample_rows)} sample rows."
        )

    projected, _ = projection_model(embeddings, components)
    item_indices: dict[str, list[int]] = defaultdict(list)
    for fallback_index, row in enumerate(sample_rows):
        try:
            index = int(row.get("embedding_index", fallback_index))
        except (TypeError, ValueError):
            index = fallback_index
        item_id = str(row.get("item_id", ""))
        item_indices[item_id].append(index)
        row["x"] = float(projected[index, 0])
        row["y"] = float(projected[index, 1])
        row["z"] = float(projected[index, 2]) if components == 3 else 0.0

    for row in item_rows:
        item_id = str(row.get("item_id", ""))
        indices = item_indices.get(item_id, [])
        if indices:
            # The projection is affine, so the mean of the projected samples is
            # the projection of their mean -- without a second transform.
            centroid = np.mean(projected[indices], axis=0)
            row["x"] = float(centroid[0])
            row["y"] = float(centroid[1])
            row["z"] = float(centroid[2]) if components == 3 else 0.0
        else:
            row["x"] = 0.0
            row["y"] = 0.0
            row["z"] = 0.0
        row["status"] = drift_status(row)
        row["acse_score"] = finite_float(row.get("acse_uncertainty_score", ""))
        row.setdefault("task2_requirement", row.get("task2_requirement", ""))
    return sample_rows, item_rows, embeddings, backend_label


def selected_items(item_rows: list[dict[str, Any]], limit: int) -> list[str]:
    buckets = [
        ("strict_oc_high_acse", lambda row: row["status"] == "strict_text_oc", True),
        ("strict_oc_low_acse", lambda row: row["status"] == "strict_text_oc", False),
        ("clean_high_acse", lambda row: row["status"] == "clean", True),
        ("clean_low_acse", lambda row: row["status"] == "clean", False),
    ]
    chosen: list[str] = []
    for _, predicate, descending in buckets:
        candidates = [
            row for row in item_rows if predicate(row) and row["item_id"] not in chosen
        ]
        candidates = sorted(
            candidates,
            key=lambda row: finite_float(row.get("acse_score"), 0.0),
            reverse=descending,
        )
        if candidates:
            chosen.append(candidates[0]["item_id"])
        if len(chosen) >= limit:
            return chosen[:limit]
    for row in sorted(
        item_rows,
        key=lambda item: finite_float(item.get("acse_score"), 0.0),
        reverse=True,
    ):
        if row["item_id"] not in chosen:
            chosen.append(row["item_id"])
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def plot_item_centroids(
    item_rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
    components: int,
) -> None:
    if components == 3:
        fig = plt.figure(figsize=(13, 5.8), constrained_layout=True)
        axes = [
            fig.add_subplot(1, 2, 1, projection="3d"),
            fig.add_subplot(1, 2, 2, projection="3d"),
        ]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    for status, color in DRIFT_COLORS.items():
        subset = [row for row in item_rows if row["status"] == status]
        if not subset:
            continue
        scatter_kwargs = {
            "s": [
                18 + 420 * finite_float(row.get("acse_score"), 0.0) for row in subset
            ],
            "c": color,
            "alpha": 0.72,
            "edgecolors": "white",
            "linewidths": 0.35,
            "label": status.replace("_", " "),
        }
        if components == 3:
            axes[0].scatter(
                [row["x"] for row in subset],
                [row["y"] for row in subset],
                [row["z"] for row in subset],
                **scatter_kwargs,
            )
        else:
            axes[0].scatter(
                [row["x"] for row in subset],
                [row["y"] for row in subset],
                **scatter_kwargs,
            )
    axes[0].set_title("Text-drift status")
    axes[0].legend(frameon=False, loc="best")

    for source, marker in SOURCE_MARKERS.items():
        subset = [row for row in item_rows if row["source_modality"] == source]
        if not subset:
            continue
        scatter_kwargs = {
            "s": [
                18 + 420 * finite_float(row.get("acse_score"), 0.0) for row in subset
            ],
            "marker": marker,
            "alpha": 0.68,
            "edgecolors": "white",
            "linewidths": 0.35,
            "label": source,
        }
        if components == 3:
            axes[1].scatter(
                [row["x"] for row in subset],
                [row["y"] for row in subset],
                [row["z"] for row in subset],
                **scatter_kwargs,
            )
        else:
            axes[1].scatter(
                [row["x"] for row in subset],
                [row["y"] for row in subset],
                **scatter_kwargs,
            )
    axes[1].set_title("Source modality")
    axes[1].legend(frameon=False, loc="best")

    for ax in axes:
        if components == 2:
            ax.axhline(0, color="#e2e8f0", linewidth=0.8, zorder=0)
            ax.axvline(0, color="#e2e8f0", linewidth=0.8, zorder=0)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if components == 3:
            ax.set_zlabel("PC3")
        else:
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_selected_samples(
    sample_rows: list[dict[str, Any]],
    item_rows_by_id: dict[str, dict[str, Any]],
    selected_ids: list[str],
    output_path: Path,
    title: str,
    components: int,
) -> None:
    if not selected_ids:
        return
    cols = 2
    rows = math.ceil(len(selected_ids) / cols)
    if components == 3:
        fig = plt.figure(figsize=(12, max(4.2, rows * 4.1)), constrained_layout=True)
        axes_array = np.asarray(
            [
                fig.add_subplot(rows, cols, index + 1, projection="3d")
                for index in range(rows * cols)
            ]
        )
    else:
        fig, axes = plt.subplots(
            rows, cols, figsize=(12, max(4.2, rows * 4.1)), constrained_layout=True
        )
        axes_array = np.asarray(axes).reshape(-1)
    # The grid is rounded up to full rows, so trailing axes stay unused.
    for ax, item_id in zip(axes_array, selected_ids, strict=False):
        item = item_rows_by_id[item_id]
        subset = [row for row in sample_rows if row["item_id"] == item_id]
        for sample in subset:
            cluster = sample["cluster_label"]
            cluster_index = int(cluster.split("_", 1)[1]) if "_" in cluster else 0
            scatter_kwargs = {
                "s": 95,
                "c": CLUSTER_COLORS[cluster_index % len(CLUSTER_COLORS)],
                "edgecolors": "white",
                "linewidths": 0.7,
            }
            if components == 3:
                ax.scatter(sample["x"], sample["y"], sample["z"], **scatter_kwargs)
            else:
                ax.scatter(sample["x"], sample["y"], **scatter_kwargs)
            label = f"s{sample['sample_index']} {sample['pred_modality']}"
            if components == 3:
                ax.text(sample["x"], sample["y"], sample["z"], label, fontsize=8)
            else:
                ax.annotate(
                    label,
                    (sample["x"], sample["y"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.set_title(
            f"{item_id} | {item['source_modality']} | {item['status'].replace('_', ' ')} | ACSE {item['acse_score']:.3f}",
            fontsize=9,
        )
        if components == 2:
            ax.axhline(0, color="#e2e8f0", linewidth=0.8, zorder=0)
            ax.axvline(0, color="#e2e8f0", linewidth=0.8, zorder=0)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        if components == 3:
            ax.set_zlabel("PC3")
        else:
            ax.spines[["top", "right"]].set_visible(False)
    for ax in axes_array[len(selected_ids) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project and visualize ACSE Task 2 stochastic embeddings."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", default="must")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--recompute-embeddings", action="store_true")
    parser.add_argument(
        "--backend",
        default="tfidf",
        choices=["tfidf", eu.ACSE_PROXY_EMBEDDING_BACKEND, "mlx"],
    )
    parser.add_argument("--mlx-model")
    parser.add_argument("--components", type=int, default=2, choices=[2, 3])
    parser.add_argument("--selected-items", type=int, default=4)
    parser.add_argument(
        "--distance-threshold", type=float, default=eu.ACSE_PROXY_DISTANCE_THRESHOLD
    )
    args = parser.parse_args()

    root = eu.project_root()
    dataset_id = eu.normalize_dataset_id(args.dataset)
    variant = eu.normalize_benchmark_variant(args.variant)
    analysis_dir = args.analysis_dir or default_analysis_dir(
        root, dataset_id, variant, args.run_id
    )
    output_dir = args.output_dir or analysis_dir / "acse_embedding_visualizations"
    scores_path = analysis_dir / "uq_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing analysis scores: {scores_path}")

    requested_backend_label, _ = eu.semantic_embedding_backend_label(
        args.backend, args.mlx_model
    )
    cache_dir = args.cache_dir or eu.acse_semantic_cache_dir(
        analysis_dir, requested_backend_label
    )
    cache_used = False
    if cache_dir.exists() and not args.recompute_embeddings:
        sample_output_rows, item_output_rows, embeddings, backend_label = (
            load_cached_projection_inputs(
                cache_dir,
                args.run_id,
                args.model,
                args.distance_threshold,
                args.components,
            )
        )
        cache_used = True
    else:
        stochastic_rows, benchmark_by_item = load_task2_rows(
            root,
            dataset_id,
            variant,
            args.run_id,
            args.model,
            args.profile,
        )
        if not stochastic_rows:
            raise ValueError(
                "No valid stochastic Task 2 rows found for the requested run/model."
            )

        scores = eu.read_csv_rows(scores_path)
        det_scores = {
            row["item_id"]: row
            for row in scores
            if row.get("task") == "task2"
            and row.get("uq_method") == "verbalized_confidence"
            and row.get("model") == args.model
        }
        acse_scores = {
            row["item_id"]: row
            for row in scores
            if row.get("task") == "task2"
            and row.get("uq_method") == eu.ACSE_PROXY_METHOD
            and row.get("model") == args.model
        }
        rows_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
        deterministic_requirements = {
            str(row.get("item_id", "")): str(
                row.get("parsed_json", {}).get("requirement", "")
            )
            for row in eu.read_jsonl(
                eu.model_outputs_raw_path(root, dataset_id, variant)
            )
            if str(row.get("run_id", "")) == args.run_id
            and str(row.get("model", "")) == args.model
            and str(row.get("task", "")) == "task2"
            and str(row.get("sample_kind", "")) == "deterministic"
            and str(row.get("parse_status", "")) == "ok"
            and isinstance(row.get("parsed_json"), dict)
        }
        for row in stochastic_rows:
            if row.get("item_id") in det_scores:
                rows_by_item[str(row["item_id"])].append(row)
        if not rows_by_item:
            raise ValueError("No stochastic rows match scored Task 2 items.")

        ordered_samples: list[dict[str, Any]] = []
        texts: list[str] = []
        for item_id in sorted(rows_by_item):
            for row in sorted(
                rows_by_item[item_id],
                key=lambda sample: int(sample.get("sample_index", 0)),
            ):
                text = eu.semantic_response_text("task2", row["parsed_json"])
                ordered_samples.append(row)
                texts.append(text)

        embeddings, backend_label = eu.semantic_embedding_matrix(
            texts,
            embedding_backend=args.backend,
            mlx_model_name=args.mlx_model,
        )
        projected, _ = projection_model(embeddings, args.components)
        item_indices: dict[str, list[int]] = defaultdict(list)
        sample_output_rows = []
        for index, (row, text) in enumerate(zip(ordered_samples, texts, strict=True)):
            item_indices[str(row["item_id"])].append(index)
            parsed = row["parsed_json"]
            sample_output_rows.append(
                {
                    "item_id": row.get("item_id", ""),
                    "seed_id": row.get("seed_id", ""),
                    "sample_index": row.get("sample_index", ""),
                    "x": projected[index, 0],
                    "y": projected[index, 1],
                    "z": projected[index, 2] if args.components == 3 else 0.0,
                    "pred_modality": parsed.get("modality", ""),
                    "semantic_text": text,
                    "requirement": parsed.get("requirement", ""),
                    "cluster_label": "",
                }
            )

        item_output_rows = []
        sample_by_key = {
            (row["item_id"], str(row["sample_index"])): row
            for row in sample_output_rows
        }
        for item_id, indices in item_indices.items():
            item_texts = [texts[index] for index in indices]
            item_cluster_embeddings, _ = eu.semantic_embedding_matrix(
                item_texts,
                embedding_backend=args.backend,
                mlx_model_name=args.mlx_model,
            )
            cluster_labels = eu.acse_cluster_labels_for_embeddings(
                item_cluster_embeddings, args.distance_threshold
            )
            for sample_index, label in zip(indices, cluster_labels, strict=True):
                source = ordered_samples[sample_index]
                sample_by_key[
                    (str(source["item_id"]), str(source.get("sample_index", "")))
                ]["cluster_label"] = label

            # Affine projection: the mean of the projected samples already is
            # the projected centroid.
            centroid = np.mean(projected[indices], axis=0)
            det = det_scores[item_id]
            acse = acse_scores.get(item_id, {})
            benchmark = benchmark_by_item.get(item_id, {})
            item_output_rows.append(
                {
                    "item_id": item_id,
                    "seed_id": det.get("seed_id", ""),
                    "source_modality": det.get("source_modality", ""),
                    "status": drift_status(det),
                    "x": centroid[0],
                    "y": centroid[1],
                    "z": centroid[2] if args.components == 3 else 0.0,
                    "acse_score": finite_float(acse.get("uncertainty_score", "")),
                    "semantic_cluster_count": acse.get("semantic_cluster_count", ""),
                    "semantic_cluster_entropy": acse.get(
                        "semantic_cluster_entropy", ""
                    ),
                    "semantic_cluster_variation_ratio": acse.get(
                        "semantic_cluster_variation_ratio", ""
                    ),
                    "text_modality": det.get("text_modality", ""),
                    "text_modality_basis": det.get("text_modality_basis", ""),
                    "text_overcommit": det.get("text_overcommit", ""),
                    "strict_text_overcommit": det.get("strict_text_overcommit", ""),
                    "confidence": det.get("confidence", ""),
                    "source_statement": benchmark.get("source_statement", ""),
                    "task2_requirement": deterministic_requirements.get(item_id, ""),
                }
            )

    selected_ids = selected_items(item_output_rows, args.selected_items)
    item_rows_by_id = {row["item_id"]: row for row in item_output_rows}
    projection_suffix = f"{args.components}d"
    item_csv = output_dir / "task2_acse_projection_items.csv"
    sample_csv = output_dir / "task2_acse_projection_samples.csv"
    selected_csv = output_dir / "task2_acse_selected_items.csv"
    eu.write_csv_rows(item_csv, item_output_rows)
    eu.write_csv_rows(sample_csv, sample_output_rows)
    eu.write_csv_rows(
        selected_csv, [item_rows_by_id[item_id] for item_id in selected_ids]
    )

    title = (
        f"{dataset_id}/{variant} {args.model} Task 2 ACSE projection ({backend_label})"
    )
    centroid_png = output_dir / f"task2_acse_item_centroids_{projection_suffix}.png"
    samples_png = output_dir / f"task2_acse_selected_samples_{projection_suffix}.png"
    plot_item_centroids(
        item_output_rows,
        centroid_png,
        title,
        args.components,
    )
    plot_selected_samples(
        sample_output_rows,
        item_rows_by_id,
        selected_ids,
        samples_png,
        title,
        args.components,
    )

    manifest = {
        "dataset": dataset_id,
        "variant": variant,
        "run_id": args.run_id,
        "model": args.model,
        "profile": args.profile or "",
        "embedding_backend": backend_label,
        "embedding_cache_used": cache_used,
        "embedding_cache_dir": str(cache_dir) if cache_used else "",
        "distance_threshold": args.distance_threshold,
        "projection": "pca",
        "components": args.components,
        "stochastic_sample_rows": len(sample_output_rows),
        "item_rows": len(item_output_rows),
        "selected_item_ids": selected_ids,
        "artifacts": [
            str(item_csv),
            str(sample_csv),
            str(selected_csv),
            str(centroid_png),
            str(samples_png),
        ],
    }
    eu.write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote ACSE embedding visualization artifacts to {output_dir}")
    print(
        f"Items: {len(item_output_rows)}; stochastic samples: {len(sample_output_rows)}; backend: {backend_label}"
    )
    print("Selected items: " + ", ".join(selected_ids))


if __name__ == "__main__":
    main()
