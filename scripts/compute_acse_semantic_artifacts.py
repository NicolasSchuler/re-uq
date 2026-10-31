"""Compute and cache ACSE semantic-UQ artifacts for completed Task 1/2 runs.

The normal evaluation export stores ACSE scalar diagnostics in `uq_scores.csv`.
This script adds a backend-specific cache with sample embeddings, item-level
ACSE rows, and enough metadata to make 2D/3D projection plots later without
calling the embedding backend again.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
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
    """One completed analysis run and model whose semantic cache can be built.

    Attributes:
        analysis_dir: Evaluation output directory with the run's
            `provenance_manifest.json` and `uq_scores.csv`.
        dataset_id: Normalized dataset id from the provenance manifest.
        variant: Normalized benchmark variant from the provenance manifest.
        run_id: Run id from the provenance manifest.
        model: One model the analysis scored.
        profile: Profile filter of the analysis, empty when none was set.
        semantic_embedding_backend: Embedding backend label the analysis
            recorded, empty for runs without embedding provenance.
    """

    analysis_dir: Path
    dataset_id: str
    variant: str
    run_id: str
    model: str
    profile: str
    semantic_embedding_backend: str


def scored_models(analysis_dir: Path, run_id: str) -> list[str]:
    """List the models this analysis dir actually scored deterministic Task 2 items for.

    Args:
        analysis_dir: Evaluation output directory with `uq_scores.csv`.
        run_id: Run whose score rows are read.

    Returns:
        The sorted, non-empty model names with a `verbalized_confidence` Task 2
        row for the run.
    """
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

    An analysis produced without `--model` records an empty `model_filter`
    and covers every model in its scores; those runs used to be skipped
    entirely. The models are read back from `uq_scores.csv` so that a
    multi-model analysis dir caches exactly like the single-model dirs, and any
    directory carrying a provenance manifest is considered, whatever it is named.

    Args:
        root: Unused; discovery reads only the analysis dirs themselves.
        output_root: Directory whose immediate subdirectories are searched for
            `provenance_manifest.json`.

    Returns:
        One run per scored model of every analysis dir that has
        `uq_scores.csv` and a run id in its provenance.
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
    """Check the run registry for a `complete` entry of the run and model.

    Args:
        root: Checkout root that locates the registry of the run's dataset and
            variant.
        run: The run; when it has a profile, the registry entry must match it.

    Returns:
        True when the registry has such an entry, False otherwise, including
        when the registry file is missing.
    """
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
    """Expand and deduplicate the requested embedding backends.

    Args:
        raw_values: Values of `--backend`; `None` or empty means `tfidf`, and
            `all` expands to `tfidf` and `mlx`.

    Returns:
        The backends in first-request order, without repeats.
    """
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
    """Resolve cache backends, defaulting to the analysis run's provenance.

    Args:
        run: Run whose recorded backend and model are the defaults.
        raw_values: Requested backends; `None` uses the run's recorded backend
            and model, or the proxy backend when none was recorded.
        mlx_model_name: MLX model for `mlx` backends, overriding the recorded
            one.

    Returns:
        `(backend, model)` pairs; the model is `None` for requested non-MLX
        backends.
    """
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
    """Return every cache manifest under `output_root`, one per model and backend.

    The glob matches any analysis dir name, not just `evaluation_*`, because
    discovery reads `provenance_manifest.json` from any directory; a renamed
    analysis dir must stay visible here or its artifacts vanish downstream.

    Args:
        output_root: Directory whose analysis dirs hold
            `acse_semantic_*/manifest.json` caches.

    Returns:
        The parsed manifests, the first per dataset, variant, run, model and
        backend in path order, each with `artifact_dir` set to where it was
        found.
    """
    manifests: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for manifest_path in sorted(output_root.glob("*/acse_semantic_*/manifest.json")):
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
        # Where the manifest actually is, rather than a path re-derived from its
        # fields: the two disagree for any cache written under an older naming
        # scheme, and downstream readers follow this column.
        manifest["artifact_dir"] = str(manifest_path.parent)
        manifests.append(manifest)
    return manifests


def checkout_relative_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return manifest rows with `analysis_dir`/`artifact_dir` relative to the checkout.

    The selected-run manifest is tracked with the paper artifacts, so it must
    not carry one machine's absolute paths; `manifest_rows` in
    `plot_acse_global_embedding_projection` re-roots relative entries. Paths
    outside the checkout (tests, scratch runs) stay absolute.

    Args:
        rows: Manifest summary rows; they are copied, not modified.

    Returns:
        The copied rows with both paths rewritten.
    """
    root = eu.project_root().resolve()
    relative: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        for key in ("analysis_dir", "artifact_dir"):
            value = str(record.get(key, ""))
            if not value:
                continue
            with contextlib.suppress(ValueError):
                record[key] = Path(value).resolve().relative_to(root).as_posix()
        relative.append(record)
    return relative


def manifest_summary_rows(
    discovered: list[dict[str, Any]], computed: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build summary rows for the discovered caches, refusing a truncating rewrite.

    `computed` holds the manifests this invocation produced, whether it just
    wrote them or reused an existing cache. Discovery finding none of them means
    the summary CSV would be replaced by a header-less empty file and those
    artifacts would be invisible downstream, so that is an error rather than a
    silent truncation.

    Args:
        discovered: Cache manifests found on disk.
        computed: Manifests returned by this invocation, computed or reused.

    Returns:
        One summary row per discovered manifest.

    Raises:
        ValueError: If `computed` is non-empty but nothing was discovered.
    """
    if computed and not discovered:
        raise ValueError(
            f"{len(computed)} cache manifest(s) were just written, but discovery "
            f"found none; refusing to overwrite {eu.ACSE_SEMANTIC_MANIFEST_FILENAME} "
            "with an empty file."
        )
    return [
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
            "artifact_dir": row["artifact_dir"],
        }
        for row in discovered
    ]


def input_fingerprint(path: Path, manifest_dir: Path) -> dict[str, Any]:
    """Identify one consumed input by content, addressed relative to the manifest.

    Manifest-relative paths keep the cache readable after the tree is moved or
    copied, which absolute paths would not survive.

    Args:
        path: Consumed input file; a `.jsonl` path also covers its compacted
            Parquet sibling.
        manifest_dir: Directory the recorded path is relative to.

    Returns:
        The `path`, `sha256` and `bytes` of the input. With several backing
        files the hash is taken over their hashes and the sizes are summed;
        with none, the hash is empty and the size 0.
    """
    entry: dict[str, Any] = {
        "path": str(path.resolve().relative_to(manifest_dir.resolve(), walk_up=True)),
        "sha256": "",
        "bytes": 0,
    }
    # A raw JSONL path may be backed by its compacted Parquet sibling as well
    # (see `eu.compact_jsonl`); the fingerprint covers every backing file so a
    # compaction that changes nothing logical still reads as a changed input
    # only when bytes actually differ.
    backing = eu.raw_store_files(path) if path.suffix == ".jsonl" else [path]
    backing = [candidate for candidate in backing if candidate.exists()]
    if len(backing) == 1:
        entry["sha256"] = eu.sha256_file(backing[0])
        entry["bytes"] = backing[0].stat().st_size
    elif backing:
        digest = hashlib.sha256()
        for candidate in backing:
            digest.update(eu.sha256_file(candidate).encode())
        entry["sha256"] = digest.hexdigest()
        entry["bytes"] = sum(candidate.stat().st_size for candidate in backing)
    return entry


def cache_input_fingerprints(
    root: Path, run: CompletedRun, backend_label: str, manifest_dir: Path
) -> dict[str, Any]:
    """Return everything the cached artifacts were derived from, as a comparable dict.

    A cache is only reusable when the raw outputs, the analysis scores, the
    benchmark items, the embedding identity, and the scored model are all
    unchanged. Comparing these beats trusting the `status` field, which only
    says that some earlier invocation finished.

    Args:
        root: Checkout root locating the raw outputs and benchmark items.
        run: Run and model the cache is for.
        backend_label: Embedding backend label; its model part is also
            recorded on its own.
        manifest_dir: Cache directory that recorded paths are relative to.

    Returns:
        The fingerprints of the raw rows, analysis scores and benchmark items,
        and the embedding backend, embedding model and scored model.
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
        "model": run.model,
    }


def cached_manifest_is_current(
    manifest: dict[str, Any], expected: dict[str, Any], manifest_path: Path
) -> bool:
    """Return True when the manifest's recorded inputs still match the files on disk.

    Args:
        manifest: The cached manifest.
        expected: Fresh fingerprints from `cache_input_fingerprints`.
        manifest_path: Location of the manifest, named in the log message.

    Returns:
        True when the recorded inputs equal `expected`. False, with an info log,
        when the manifest records no inputs or they differ.
    """
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
    """Return the key that orders an item's samples by sample index.

    Args:
        row: Raw sample row.

    Returns:
        `(sample_index, item_id)`, with index 0 when it does not parse as an
        integer.
    """
    try:
        return int(row.get("sample_index", 0)), str(row.get("item_id", ""))
    except (TypeError, ValueError):
        return 0, str(row.get("item_id", ""))


def load_run_rows(
    root: Path, run: CompletedRun
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Load the run's Task 2 raw rows and its benchmark items.

    Args:
        root: Checkout root locating the raw outputs and benchmark items.
        run: Run and model to load. With a profile, rows of other profiles are
            dropped and rows without a profile kept.

    Returns:
        The raw rows and the benchmark rows keyed by item id.
    """
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
    """Load the run's deterministic Task 2 score rows from its analysis dir.

    Args:
        run: Run and model whose `verbalized_confidence` Task 2 rows are read
            from `uq_scores.csv`.

    Returns:
        The score rows keyed by item id.
    """
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
    """Embed the sample texts, in batches when the backend has a model.

    A backend without a model, such as the TF-IDF proxy, embeds all texts in
    one call.

    Args:
        texts: Sample texts in cache order.
        embedding_backend: Backend name, such as `tfidf` or `mlx`.
        mlx_model_name: MLX model, `None` for the default.
        batch_size: Texts per embedding call for a model backend; values below
            1 count as 1.

    Returns:
        The embedding matrix and the backend label. A model backend given no
        texts returns an empty `(0, 1)` matrix.

    Raises:
        RuntimeError: If the backend label changes between batches.
    """
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
    """Return the embeddings an item's score is computed from.

    MLX vectors are sliced from the run-wide matrix; any other backend
    re-embeds the item's own texts.

    Args:
        backend_label: Label returned by the run-wide embedding; an `mlx:`
            prefix selects slicing.
        embedding_backend: Backend name used to re-embed.
        mlx_model_name: MLX model used to re-embed.
        texts: The item's sample texts.
        global_embeddings: Run-wide embedding matrix.
        indices: Rows of the item's samples in `global_embeddings`.

    Returns:
        One embedding row per sample of the item.
    """
    if backend_label.startswith(f"{eu.ACSE_MLX_EMBEDDING_BACKEND}:"):
        return global_embeddings[indices, :]
    item_embeddings, _ = eu.semantic_embedding_matrix(
        texts,
        embedding_backend=embedding_backend,
        mlx_model_name=mlx_model_name,
    )
    return item_embeddings


def deterministic_requirement(raw_rows: list[dict[str, Any]]) -> dict[str, str]:
    """Map each item to the requirement text of its parsed deterministic output.

    Args:
        raw_rows: Task 2 raw rows of one run and model.

    Returns:
        The `requirement` of each item's parsed deterministic row, keyed by
        item id; a later row for the same item overwrites an earlier one.
    """
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
    """Compute, or reuse, one run's semantic cache for one embedding backend.

    Embeds every parsed stochastic Task 2 sample of the items with a
    deterministic score, clusters each item's samples once, and writes the
    sample embeddings, the sample, item, score, normalized-score and
    calibration tables, and a manifest with input fingerprints to the cache
    directory. A cache whose recorded inputs still match is returned as is.

    Args:
        root: Checkout root locating the raw outputs and benchmark items.
        run: Run and model to cache.
        embedding_backend: Backend name, such as `tfidf` or `mlx`.
        mlx_model_name: MLX model for the `mlx` backend.
        distance_threshold: Cosine-distance threshold of the clustering.
        embedding_batch_size: Texts per embedding call for a model backend.
        force: Recompute even when the cached inputs are current.

    Returns:
        The cache manifest, with status `computed` or `reused`.

    Raises:
        ValueError: If the run has no Task 2 raw rows, no deterministic Task 2
            scores, or no parsed stochastic samples.
    """
    backend_label, _ = eu.semantic_embedding_backend_label(
        embedding_backend, mlx_model_name
    )
    output_dir = eu.acse_semantic_cache_dir(run.analysis_dir, backend_label, run.model)
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
    output_dir = eu.acse_semantic_cache_dir(run.analysis_dir, backend_label, run.model)
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
    eu.write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    """Compute the semantic caches of completed runs and write the cache inventory.

    Writes the inventory CSV and `acse_semantic_artifact_manifest.json` under
    `--output-root`, and with `--selected-manifest` an inventory of only this
    invocation's caches with checkout-relative paths.

    Raises:
        ValueError: If no completed run is found, or the run registry does not
            confirm a run as complete and `--allow-registry-mismatch` is unset.
    """
    parser = argparse.ArgumentParser(
        description="Compute backend-specific ACSE semantic caches for completed runs."
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--selected-manifest",
        type=Path,
        help="Also write an inventory containing only this invocation's caches.",
    )
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
    manifest_rows = manifest_summary_rows(all_manifests, manifests)
    eu.write_csv_rows(output_root / eu.ACSE_SEMANTIC_MANIFEST_FILENAME, manifest_rows)
    eu.write_json(output_root / "acse_semantic_artifact_manifest.json", all_manifests)
    if args.selected_manifest:
        selected = {
            (str(row["run_id"]), str(row["model"]), str(row["embedding_backend"]))
            for row in manifests
        }
        scoped = [
            row
            for row in all_manifests
            if (str(row["run_id"]), str(row["model"]), str(row["embedding_backend"]))
            in selected
        ]
        eu.write_csv_rows(
            args.selected_manifest,
            checkout_relative_rows(manifest_summary_rows(scoped, manifests)),
        )


if __name__ == "__main__":
    main()
