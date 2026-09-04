"""Compute and cache ACSE semantic-UQ artifacts for completed Task 1/2 runs.

The normal evaluation export stores ACSE scalar diagnostics in ``uq_scores.csv``.
This script adds a backend-specific cache with sample embeddings, item-level
ACSE rows, and enough metadata to make 2D/3D projection plots later without
calling the embedding backend again.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu


TEXT_MODALITY_FIELDS = [
    "text_modality",
    "text_modality_basis",
    "text_modality_parse_status",
    "text_modality_correct",
    "label_text_consistent",
    "text_overcommit",
    "text_undercommit",
    "strict_text_overcommit",
    "text_high_conf_overcommit_80",
    "text_high_conf_overcommit_90",
    "strict_text_high_conf_overcommit_80",
    "strict_text_high_conf_overcommit_90",
    "label_correct_text_overcommit_80",
    "label_correct_text_overcommit_90",
]

ACSE_SCORE_FIELDS = [
    "run_id",
    "run_group_id",
    "provider_id",
    "profile_id",
    "model",
    "task",
    "uq_method",
    "item_id",
    "seed_id",
    "source_modality",
    "ordinal_strength",
    "numeric_strength",
    "valid_n",
    "total_n",
    "parse_failures",
    "y_true",
    "y_pred",
    "p_yes",
    "confidence",
    "uncertainty_score",
    "uncertainty_measure",
    "label_distribution",
    "semantic_embedding_backend",
    "semantic_distance_threshold",
    "semantic_cluster_count",
    "semantic_cluster_distribution",
    "semantic_cluster_entropy",
    "semantic_cluster_variation_ratio",
    "semantic_dominant_cluster_share",
    "semantic_mean_pairwise_distance",
    "semantic_dominant_cluster_mean_distance",
    "semantic_uncertainty_score",
    "gold_modality",
    "pred_modality",
    *TEXT_MODALITY_FIELDS,
]

ITEM_FIELDS = [
    "run_id",
    "model",
    "profile_id",
    "dataset_id",
    "benchmark_variant",
    "item_id",
    "seed_id",
    "source_modality",
    "gold_modality",
    "pred_modality",
    "deterministic_confidence",
    "label_distribution",
    "valid_n",
    "total_n",
    "parse_failures",
    "acse_uncertainty_score",
    "semantic_embedding_backend",
    "semantic_distance_threshold",
    "semantic_cluster_count",
    "semantic_cluster_distribution",
    "semantic_cluster_entropy",
    "semantic_cluster_variation_ratio",
    "semantic_dominant_cluster_share",
    "semantic_mean_pairwise_distance",
    "semantic_dominant_cluster_mean_distance",
    *TEXT_MODALITY_FIELDS,
    "source_statement",
    "task2_requirement",
]

SAMPLE_FIELDS = [
    "embedding_index",
    "run_id",
    "model",
    "profile_id",
    "dataset_id",
    "benchmark_variant",
    "item_id",
    "seed_id",
    "sample_index",
    "pred_modality",
    "cluster_label",
    "semantic_text",
    "requirement",
]


@dataclass(frozen=True)
class CompletedRun:
    analysis_dir: Path
    dataset_id: str
    variant: str
    run_id: str
    model: str
    profile: str
    semantic_embedding_backend: str


def scored_models(analysis_dir: Path, run_id: str) -> list[str]:
    """Models this analysis dir actually scored deterministic Task 2 items for."""
    return sorted(
        {
            str(row.get("model", "")).strip()
            for row in eu.read_csv_rows(analysis_dir / "uq_scores.csv")
            if str(row.get("run_id", "")) == run_id
            and str(row.get("task", "")) == "task2"
            and str(row.get("uq_method", "")) == "verbalized_confidence"
            and str(row.get("model", "")).strip()
        }
    )


def completed_runs_from_analysis_dirs(
    root: Path, output_root: Path
) -> list[CompletedRun]:
    """Discover cacheable runs from provenance manifests, not directory names.

    An analysis produced without ``--model`` records an empty ``model_filter``
    and covers every model in its scores; those runs used to be skipped
    entirely. The models are read back from ``uq_scores.csv`` so that a
    multi-model analysis dir caches exactly like the single-model dirs, and any
    directory carrying a provenance manifest is considered, whatever it is named.
    """
    del root  # discovery reads only the analysis dirs themselves
    runs: list[CompletedRun] = []
    for manifest_path in sorted(output_root.glob("*/provenance_manifest.json")):
        analysis_dir = manifest_path.parent
        if not (analysis_dir / "uq_scores.csv").exists():
            continue
        provenance = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = str(provenance.get("run_id", "")).strip()
        if not run_id:
            continue
        model_filter = str(provenance.get("model_filter", "")).strip()
        models = [model_filter] if model_filter else scored_models(analysis_dir, run_id)
        dataset_id = eu.normalize_dataset_id(provenance.get("dataset_id", ""))
        variant = eu.normalize_benchmark_variant(
            provenance.get("benchmark_variant", "must")
        )
        for model in models:
            runs.append(
                CompletedRun(
                    analysis_dir=analysis_dir,
                    dataset_id=dataset_id,
                    variant=variant,
                    run_id=run_id,
                    model=model,
                    profile=str(provenance.get("profile_filter", "")).strip(),
                    semantic_embedding_backend=str(
                        provenance.get("semantic_embedding_backend", "")
                    ).strip(),
                )
            )
    return runs


def registry_confirms_complete(root: Path, run: CompletedRun) -> bool:
    registry_path = eu.run_registry_path(root, run.dataset_id, run.variant)
    rows = eu.read_csv_rows(registry_path) if registry_path.exists() else []
    for row in rows:
        if (
            str(row.get("run_id", "")) == run.run_id
            and str(row.get("model", "")) == run.model
            and str(row.get("status", "")) == "complete"
        ):
            if run.profile and str(row.get("profile_id", "")) != run.profile:
                continue
            return True
    return False


def backend_values(raw_values: list[str] | None) -> list[str]:
    values = raw_values or ["tfidf"]
    expanded: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized == "all":
            expanded.extend(["tfidf", "mlx"])
        else:
            expanded.append(normalized)
    result: list[str] = []
    for value in expanded:
        if value not in result:
            result.append(value)
    return result


def backend_specs_for_run(
    run: CompletedRun,
    raw_values: list[str] | None,
    mlx_model_name: str | None,
) -> list[tuple[str, str | None]]:
    """Resolve cache backends, defaulting to the analysis run's provenance."""
    recorded_backend, recorded_model = eu.semantic_embedding_backend_args(
        run.semantic_embedding_backend
    )
    if raw_values is None:
        return [
            (
                recorded_backend or eu.ACSE_PROXY_EMBEDDING_BACKEND,
                recorded_model,
            )
        ]
    return [
        (
            backend,
            (mlx_model_name or recorded_model)
            if backend == eu.ACSE_MLX_EMBEDDING_BACKEND
            else None,
        )
        for backend in backend_values(raw_values)
    ]


def existing_backend_manifests(output_root: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for manifest_path in sorted(
        output_root.glob("evaluation_*/acse_semantic_*/manifest.json")
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        key = (
            str(manifest.get("dataset_id", "")),
            str(manifest.get("benchmark_variant", "")),
            str(manifest.get("run_id", "")),
            str(manifest.get("model", "")),
            str(manifest.get("embedding_backend", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        manifests.append(manifest)
    return manifests


def input_fingerprint(path: Path, manifest_dir: Path) -> dict[str, Any]:
    """Identify one consumed input by content, addressed relative to the manifest.

    Manifest-relative paths keep the cache readable after the tree is moved or
    copied, which absolute paths would not survive.
    """
    entry: dict[str, Any] = {
        "path": str(path.resolve().relative_to(manifest_dir.resolve(), walk_up=True)),
        "sha256": "",
        "bytes": 0,
    }
    if path.exists():
        entry["sha256"] = eu.sha256_file(path)
        entry["bytes"] = path.stat().st_size
    return entry


def cache_input_fingerprints(
    root: Path, run: CompletedRun, backend_label: str, manifest_dir: Path
) -> dict[str, Any]:
    """Everything the cached artifacts were derived from, as a comparable dict.

    A cache is only reusable when the raw outputs, the analysis scores, the
    benchmark items, and the embedding identity are all unchanged. Comparing
    these beats trusting the ``status`` field, which only says that some
    earlier invocation finished.
    """
    _, model_name = eu.semantic_embedding_backend_args(backend_label)
    return {
        "raw_rows": input_fingerprint(
            eu.model_outputs_raw_path(root, run.dataset_id, run.variant), manifest_dir
        ),
        "uq_scores": input_fingerprint(
            run.analysis_dir / "uq_scores.csv", manifest_dir
        ),
        "benchmark_items": input_fingerprint(
            eu.artifact_path(
                root / "data/processed/benchmark_items.csv",
                run.dataset_id,
                run.variant,
            ),
            manifest_dir,
        ),
        "embedding_backend": backend_label,
        "embedding_model": model_name or "",
    }


def cached_manifest_is_current(
    manifest: dict[str, Any], expected: dict[str, Any], manifest_path: Path
) -> bool:
    """True when the manifest's recorded inputs still match the files on disk."""
    recorded = manifest.get("inputs")
    if recorded is None:
        eu.logger.info(
            "%s: cache manifest predates input fingerprints; recomputing.",
            manifest_path,
        )
        return False
    if recorded != expected:
        eu.logger.info(
            "%s: input fingerprints changed since the cache was written; recomputing.",
            manifest_path,
        )
        return False
    return True


def sample_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    try:
        return int(row.get("sample_index", 0)), str(row.get("item_id", ""))
    except (TypeError, ValueError):
        return 0, str(row.get("item_id", ""))


def load_run_rows(
    root: Path, run: CompletedRun
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    raw_rows = [
        row
        for row in eu.read_jsonl(
            eu.model_outputs_raw_path(root, run.dataset_id, run.variant)
        )
        if str(row.get("run_id", "")) == run.run_id
        and str(row.get("model", "")) == run.model
        and str(row.get("task", "")) == "task2"
    ]
    if run.profile:
        raw_rows = [
            row
            for row in raw_rows
            if str(row.get("profile_id", "")) in {"", run.profile}
        ]
    benchmark = {
        row["item_id"]: row
        for row in eu.read_csv_rows(
            eu.artifact_path(
                root / "data/processed/benchmark_items.csv", run.dataset_id, run.variant
            )
        )
    }
    return raw_rows, benchmark


def load_deterministic_task2_scores(run: CompletedRun) -> dict[str, dict[str, str]]:
    scores = eu.read_csv_rows(run.analysis_dir / "uq_scores.csv")
    return {
        row["item_id"]: row
        for row in scores
        if row.get("run_id") == run.run_id
        and row.get("model") == run.model
        and row.get("task") == "task2"
        and row.get("uq_method") == "verbalized_confidence"
    }


def embedding_matrix_for_cache(
    texts: list[str],
    embedding_backend: str,
    mlx_model_name: str | None,
    batch_size: int,
) -> tuple[np.ndarray, str]:
    backend_label, model_name = eu.semantic_embedding_backend_label(
        embedding_backend, mlx_model_name
    )
    if model_name is None:
        return eu.semantic_embedding_matrix(
            texts, embedding_backend=embedding_backend, mlx_model_name=mlx_model_name
        )
    batches: list[np.ndarray] = []
    for start in range(0, len(texts), max(1, int(batch_size))):
        batch, label = eu.semantic_embedding_matrix(
            texts[start : start + max(1, int(batch_size))],
            embedding_backend=embedding_backend,
            mlx_model_name=mlx_model_name,
        )
        if label != backend_label:
            raise RuntimeError(
                f"Embedding backend changed within run: {backend_label!r} -> {label!r}"
            )
        batches.append(batch)
    if not batches:
        return np.zeros((0, 1), dtype=float), backend_label
    return np.vstack(batches), backend_label


def item_embeddings_for_scoring(
    backend_label: str,
    embedding_backend: str,
    mlx_model_name: str | None,
    texts: list[str],
    global_embeddings: np.ndarray,
    indices: list[int],
) -> np.ndarray:
    if backend_label.startswith(f"{eu.ACSE_MLX_EMBEDDING_BACKEND}:"):
        return global_embeddings[indices, :]
    item_embeddings, _ = eu.semantic_embedding_matrix(
        texts,
        embedding_backend=embedding_backend,
        mlx_model_name=mlx_model_name,
    )
    return item_embeddings


def deterministic_requirement(raw_rows: list[dict[str, Any]]) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for row in raw_rows:
        parsed = row.get("parsed_json")
        if (
            str(row.get("sample_kind", "")) == "deterministic"
            and str(row.get("parse_status", "")) == "ok"
            and isinstance(parsed, dict)
        ):
            requirements[str(row.get("item_id", ""))] = str(
                parsed.get("requirement", "")
            )
    return requirements


def compute_run_backend(
    root: Path,
    run: CompletedRun,
    embedding_backend: str,
    mlx_model_name: str | None,
    distance_threshold: float,
    embedding_batch_size: int,
    force: bool,
) -> dict[str, Any]:
    backend_label, _ = eu.semantic_embedding_backend_label(
        embedding_backend, mlx_model_name
    )
    output_dir = eu.acse_semantic_cache_dir(run.analysis_dir, backend_label)
    manifest_path = output_dir / "manifest.json"
    fingerprints = cache_input_fingerprints(root, run, backend_label, output_dir)
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if cached_manifest_is_current(manifest, fingerprints, manifest_path):
            manifest["status"] = "reused"
            return manifest

    raw_rows, benchmark_by_item = load_run_rows(root, run)
    det_scores = load_deterministic_task2_scores(run)
    if not raw_rows:
        raise ValueError(f"No Task 2 raw rows found for {run.run_id} / {run.model}.")
    if not det_scores:
        raise ValueError(
            f"No deterministic Task 2 scores found in {run.analysis_dir / 'uq_scores.csv'}."
        )

    stochastic_all_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    valid_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        item_id = str(row.get("item_id", ""))
        if item_id not in det_scores or str(row.get("sample_kind", "")) != "stochastic":
            continue
        stochastic_all_by_item[item_id].append(row)
        if str(row.get("parse_status", "")) == "ok" and isinstance(
            row.get("parsed_json"), dict
        ):
            valid_by_item[item_id].append(row)

    ordered_samples: list[dict[str, Any]] = []
    sample_texts: list[str] = []
    item_indices: dict[str, list[int]] = defaultdict(list)
    for item_id in sorted(valid_by_item):
        for row in sorted(valid_by_item[item_id], key=sample_sort_key):
            index = len(ordered_samples)
            ordered_samples.append(row)
            sample_texts.append(eu.semantic_response_text("task2", row["parsed_json"]))
            item_indices[item_id].append(index)
    if not ordered_samples:
        raise ValueError(
            f"No valid stochastic Task 2 samples found for {run.run_id} / {run.model}."
        )

    global_embeddings, backend_label = embedding_matrix_for_cache(
        sample_texts,
        embedding_backend,
        mlx_model_name,
        embedding_batch_size,
    )
    output_dir = eu.acse_semantic_cache_dir(run.analysis_dir, backend_label)
    output_dir.mkdir(parents=True, exist_ok=True)

    deterministic_requirements = deterministic_requirement(raw_rows)
    sample_rows: list[dict[str, Any]] = []
    for index, (row, text) in enumerate(
        zip(ordered_samples, sample_texts, strict=True)
    ):
        parsed = row["parsed_json"]
        sample_rows.append(
            {
                "embedding_index": index,
                "run_id": run.run_id,
                "model": run.model,
                "profile_id": row.get("profile_id", run.profile),
                "dataset_id": run.dataset_id,
                "benchmark_variant": run.variant,
                "item_id": row.get("item_id", ""),
                "seed_id": row.get("seed_id", ""),
                "sample_index": row.get("sample_index", ""),
                "pred_modality": parsed.get("modality", ""),
                "cluster_label": "",
                "semantic_text": text,
                "requirement": parsed.get("requirement", ""),
            }
        )

    score_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    for item_id, indices in item_indices.items():
        benchmark = benchmark_by_item.get(item_id)
        if not benchmark:
            continue
        valid_rows = valid_by_item[item_id]
        all_rows = stochastic_all_by_item[item_id]
        item_texts = [sample_texts[index] for index in indices]
        scoring_embeddings = item_embeddings_for_scoring(
            backend_label,
            embedding_backend,
            mlx_model_name,
            item_texts,
            global_embeddings,
            indices,
        )
        # One clustering pass per item: the cached per-sample labels are the
        # ones the persisted score was computed from, by construction.
        diagnostics, cluster_labels = eu.acse_semantic_cluster_analysis(
            scoring_embeddings,
            backend_label,
            distance_threshold=distance_threshold,
        )
        for index, cluster_label in zip(indices, cluster_labels, strict=True):
            sample_rows[index]["cluster_label"] = cluster_label

        distribution = eu.label_distribution_from_rows("task2", valid_rows)
        score_row = eu.score_from_distribution(
            valid_rows[0],
            benchmark,
            eu.ACSE_PROXY_METHOD,
            distribution,
            len(valid_rows),
            len(all_rows),
            eu.ACSE_PROXY_MEASURE,
            diagnostics["semantic_uncertainty_score"],
        )
        score_row.update(diagnostics)
        det_score = det_scores.get(item_id, {})
        for field in TEXT_MODALITY_FIELDS:
            score_row[field] = det_score.get(field, "")
        score_rows.append(score_row)

        item_row = {
            "run_id": run.run_id,
            "model": run.model,
            "profile_id": run.profile,
            "dataset_id": run.dataset_id,
            "benchmark_variant": run.variant,
            "item_id": item_id,
            "seed_id": benchmark.get("seed_id", ""),
            "source_modality": benchmark.get("source_modality", ""),
            "gold_modality": score_row.get("gold_modality", ""),
            "pred_modality": score_row.get("pred_modality", ""),
            "deterministic_confidence": det_score.get("confidence", ""),
            "label_distribution": score_row.get("label_distribution", ""),
            "valid_n": len(valid_rows),
            "total_n": len(all_rows),
            "parse_failures": len(all_rows) - len(valid_rows),
            "acse_uncertainty_score": diagnostics["semantic_uncertainty_score"],
            "semantic_embedding_backend": backend_label,
            "semantic_distance_threshold": diagnostics["semantic_distance_threshold"],
            "semantic_cluster_count": diagnostics["semantic_cluster_count"],
            "semantic_cluster_distribution": diagnostics[
                "semantic_cluster_distribution"
            ],
            "semantic_cluster_entropy": diagnostics["semantic_cluster_entropy"],
            "semantic_cluster_variation_ratio": diagnostics[
                "semantic_cluster_variation_ratio"
            ],
            "semantic_dominant_cluster_share": diagnostics[
                "semantic_dominant_cluster_share"
            ],
            "semantic_mean_pairwise_distance": diagnostics[
                "semantic_mean_pairwise_distance"
            ],
            "semantic_dominant_cluster_mean_distance": diagnostics[
                "semantic_dominant_cluster_mean_distance"
            ],
            "source_statement": benchmark.get("source_statement", ""),
            "task2_requirement": deterministic_requirements.get(item_id, ""),
        }
        for field in TEXT_MODALITY_FIELDS:
            item_row[field] = det_score.get(field, "")
        item_rows.append(item_row)

    normalized_rows = eu.acse_normalized_score_rows(score_rows)
    calibration_rows = eu.acse_calibration_diagnostic_rows(normalized_rows)
    embeddings_path = output_dir / "task2_acse_sample_embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        embeddings=np.asarray(global_embeddings, dtype=np.float32),
        item_ids=np.asarray([row["item_id"] for row in sample_rows], dtype=str),
        sample_indices=np.asarray(
            [row["sample_index"] for row in sample_rows], dtype=str
        ),
    )
    eu.write_csv_rows(
        output_dir / "task2_acse_samples.csv", sample_rows, fieldnames=SAMPLE_FIELDS
    )
    eu.write_csv_rows(
        output_dir / "task2_acse_items.csv", item_rows, fieldnames=ITEM_FIELDS
    )
    eu.write_csv_rows(
        output_dir / "task2_acse_scores.csv", score_rows, fieldnames=ACSE_SCORE_FIELDS
    )
    eu.write_csv_rows(
        output_dir / "task2_acse_normalized_scores.csv",
        normalized_rows,
        fieldnames=eu.ACSE_NORMALIZED_SCORE_FIELDS,
    )
    eu.write_csv_rows(
        output_dir / "task2_acse_calibration.csv",
        calibration_rows,
        fieldnames=eu.ACSE_CALIBRATION_FIELDS,
    )

    raw_scores = [
        float(row["semantic_uncertainty_score"])
        for row in score_rows
        if not math.isnan(float(row["semantic_uncertainty_score"]))
    ]
    manifest = {
        "status": "computed",
        "created_at_utc": eu.utc_now_iso(),
        "dataset_id": run.dataset_id,
        "benchmark_variant": run.variant,
        "run_id": run.run_id,
        "model": run.model,
        "profile_id": run.profile,
        "analysis_dir": str(run.analysis_dir),
        "embedding_backend": backend_label,
        "inputs": fingerprints,
        "distance_threshold": float(distance_threshold),
        "stochastic_sample_rows": len(sample_rows),
        "item_rows": len(item_rows),
        "embedding_shape": list(global_embeddings.shape),
        "embedding_dtype": "float32",
        "acse_score_mean": float(np.mean(raw_scores)) if raw_scores else math.nan,
        "acse_score_max": float(np.max(raw_scores)) if raw_scores else math.nan,
        "artifacts": [
            str(embeddings_path),
            str(output_dir / "task2_acse_samples.csv"),
            str(output_dir / "task2_acse_items.csv"),
            str(output_dir / "task2_acse_scores.csv"),
            str(output_dir / "task2_acse_normalized_scores.csv"),
            str(output_dir / "task2_acse_calibration.csv"),
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute backend-specific ACSE semantic caches for completed runs."
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        action="append",
        help="Limit to one or more evaluation output dirs.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=["tfidf", eu.ACSE_PROXY_EMBEDDING_BACKEND, "mlx", "all"],
    )
    parser.add_argument(
        "--mlx-model",
        default=None,
        help="MLX embedding model id; overrides the analysis run's persisted model. "
        "For legacy runs without embedding provenance, defaults to "
        f"RE_UQ_ACSE_MLX_MODEL or {eu.ACSE_MLX_DEFAULT_MODEL!r}.",
    )
    parser.add_argument(
        "--distance-threshold", type=float, default=eu.ACSE_PROXY_DISTANCE_THRESHOLD
    )
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--force", action="store_true", help="Recompute existing backend caches."
    )
    parser.add_argument("--allow-registry-mismatch", action="store_true")
    args = parser.parse_args()

    root = eu.project_root()
    output_root = (
        args.output_root if args.output_root.is_absolute() else root / args.output_root
    )
    if args.analysis_dir:
        all_runs = completed_runs_from_analysis_dirs(root, output_root)
        selected_dirs = {path.resolve() for path in args.analysis_dir}
        runs = [run for run in all_runs if run.analysis_dir.resolve() in selected_dirs]
    else:
        runs = completed_runs_from_analysis_dirs(root, output_root)
    if not runs:
        raise ValueError(
            "No completed evaluation output dirs with provenance_manifest.json and uq_scores.csv were found."
        )

    manifests: list[dict[str, Any]] = []
    for run in runs:
        if not args.allow_registry_mismatch and not registry_confirms_complete(
            root, run
        ):
            raise ValueError(
                f"Registry does not confirm complete run: {run.run_id} / {run.model}"
            )
        for backend, mlx_model_name in backend_specs_for_run(
            run, args.backend, args.mlx_model
        ):
            manifest = compute_run_backend(
                root,
                run,
                backend,
                mlx_model_name,
                args.distance_threshold,
                args.embedding_batch_size,
                args.force,
            )
            manifests.append(manifest)
            print(
                f"{manifest['status']}: {run.dataset_id}/{run.variant} {run.model} "
                f"{run.run_id} {manifest['embedding_backend']} "
                f"items={manifest['item_rows']} samples={manifest['stochastic_sample_rows']}"
            )

    all_manifests = existing_backend_manifests(output_root)
    manifest_rows = [
        {
            "dataset_id": row["dataset_id"],
            "benchmark_variant": row["benchmark_variant"],
            "run_id": row["run_id"],
            "model": row["model"],
            "profile_id": row["profile_id"],
            "embedding_backend": row["embedding_backend"],
            "status": row["status"],
            "item_rows": row["item_rows"],
            "stochastic_sample_rows": row["stochastic_sample_rows"],
            "embedding_shape": json.dumps(row["embedding_shape"]),
            "analysis_dir": row["analysis_dir"],
            "artifact_dir": str(
                eu.acse_semantic_cache_dir(
                    Path(row["analysis_dir"]), row["embedding_backend"]
                )
            ),
        }
        for row in all_manifests
    ]
    eu.write_csv_rows(output_root / eu.ACSE_SEMANTIC_MANIFEST_FILENAME, manifest_rows)
    (output_root / "acse_semantic_artifact_manifest.json").write_text(
        json.dumps(all_manifests, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
