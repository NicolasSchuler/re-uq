#!/usr/bin/env python3
"""Offline sensitivity of the meaning-variation score to its two settings.

The reported score clusters the five sampled Task 2 answers of an item at a
cosine-distance threshold and mixes normalized cluster entropy with a
dispersion term (`(1 - w) * H + w * D`). This script re-scores every item of
the frozen main study over a grid of thresholds and dispersion weights and
reports how the strict-strengthening AUROC moves, with a paired
capability-clustered bootstrap for the differences from the reported setting.

Everything is read-only with respect to the data: the frozen snapshot
(`outputs/paper_snapshot_provenance.json`) names the runs, the cached
`mlx:Qwen3-Embedding-0.6B-8bit` sample embeddings are reused after their text
and identity are verified row by row, and the cohort (readable deterministic
extractions joined to their sampled group) is built once and held fixed across
every setting. No embeddings are regenerated and no provider is called.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

try:
    from scripts import eval_utils as eu, export_paper_tables as paper
except ImportError:  # pragma: no cover - direct invocation from scripts/
    import eval_utils as eu
    import export_paper_tables as paper

logger = logging.getLogger(__name__)

THRESHOLDS: tuple[float, ...] = (0.10, 0.20, 0.35, 0.50, 0.70)
WEIGHTS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
REFERENCE_THRESHOLD = eu.ACSE_PROXY_DISTANCE_THRESHOLD
REFERENCE_WEIGHT = eu.ACSE_PROXY_INTERNAL_DISPERSION_WEIGHT
EXPECTED_BACKEND = f"{eu.ACSE_MLX_EMBEDDING_BACKEND}:{eu.ACSE_MLX_DEFAULT_MODEL}"
MAX_CLUSTER_COUNT = 5
DEFAULT_PROVENANCE = Path("outputs/paper_snapshot_provenance.json")
DEFAULT_MANIFEST = Path("outputs/rerun/acse_selected_manifest.csv")
DEFAULT_RQ_TABLE = Path("outputs/paper_per_model_rq_table.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/meaning_variation_sensitivity")
CAPABILITY_FIELD = eu.BOOTSTRAP_CLUSTER_FALLBACK_FIELD  # seed_id
ITEM_FIELD = eu.BOOTSTRAP_ITEM_CLUSTER_FIELD  # item_id


def setting_label(threshold: float, weight: float) -> str:
    """Return the label of one grid setting, such as `t0.35_w0.2`.

    Args:
        threshold: Cosine-distance threshold of the setting.
        weight: Dispersion weight of the setting.

    Returns:
        `t<threshold>_w<weight>` with two and one decimals, the suffix of the
        per-item score columns.
    """
    return f"t{threshold:.2f}_w{weight:.1f}"


# ---------------------------------------------------------------------------
# Frozen snapshot and raw rows
# ---------------------------------------------------------------------------


def _path(root: Path, value: str) -> Path:
    """Resolve a frozen-snapshot path (recorded absolute) inside `root`.

    Args:
        root: Checkout root the recorded path is relocated into.
        value: Path as recorded in the snapshot. A relative path is joined to
            `root`, an absolute path with a `data` component is re-rooted from
            that component on, and any other absolute path is kept.

    Returns:
        The resolved path.
    """
    path = Path(value)
    if path.is_absolute():
        if "data" in path.parts:
            return root.joinpath(*path.parts[path.parts.index("data") :])
        return path
    return root / path


def _relative(root: Path, path: Path | str) -> str:
    """Return a path as recorded in `settings.json`: checkout-relative in `root`.

    Args:
        root: Checkout root.
        path: Path to record.

    Returns:
        The POSIX path relative to `root` when `path` resolves inside it,
        otherwise `path` unchanged as a string.
    """
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_snapshot(provenance_path: Path) -> dict[str, Any]:
    """Load the frozen snapshot and check its model cohort and cell selection.

    Args:
        provenance_path: The frozen snapshot provenance JSON.

    Returns:
        The parsed snapshot.

    Raises:
        ValueError: If the model cohort is empty, repeats a model or contains
            `all`, the sampling plan is not the planned one, a
            `(dataset, variant)` cell appears twice, or a cell's runs do not
            cover exactly the cohort models with distinct run ids.
    """
    snapshot = json.loads(provenance_path.read_text(encoding="utf-8"))
    models = snapshot["models_cohort"]
    if not models or len(set(models)) != len(models) or "all" in models:
        raise ValueError("Frozen model cohort must be nonempty and unique")
    if snapshot.get("sampling_plan_source") != eu.SAMPLING_PLAN_SOURCE_PLANNED:
        raise ValueError("Frozen sampling plan must be planned")
    seen: set[tuple[str, str]] = set()
    for cell in snapshot["cells"]:
        key = (cell["dataset"], cell["variant"])
        if key in seen:
            raise ValueError(f"Duplicate frozen cell {key}")
        seen.add(key)
        runs = cell["run_ids"]
        if set(runs) != set(models) or len(set(runs.values())) != len(runs):
            raise ValueError(f"Unexpected model/run selection in frozen cell {key}")
    return snapshot


def load_cell_raw_rows(root: Path, cell: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the raw rows of the cell's frozen runs, read through the raw store.

    Args:
        root: Checkout root used to resolve the cell's recorded `raw_path`.
        cell: One snapshot cell with its `raw_path` and per-model `run_ids`.

    Returns:
        The raw rows whose `run_id` is one of the cell's frozen runs. A
        non-empty JSONL file is read directly, otherwise the Parquet store.

    Raises:
        FileNotFoundError: If no raw store exists at the cell's `raw_path`.
    """
    raw_path = _path(root, cell["raw_path"])
    if not eu.raw_store_exists(raw_path):
        raise FileNotFoundError(f"raw store missing for {raw_path}")
    run_ids = {str(run_id) for run_id in cell["run_ids"].values()}
    if raw_path.exists() and raw_path.stat().st_size > 0:
        # A non-empty JSONL tail holds rows the Parquet half does not.
        return [
            row for row in eu.read_jsonl(raw_path) if str(row.get("run_id")) in run_ids
        ]
    return eu.read_raw_store(raw_path, run_ids=run_ids)


# ---------------------------------------------------------------------------
# Cohort: deterministic rows and stochastic groups
# ---------------------------------------------------------------------------


@dataclass
class SampleGroup:
    """One item's stochastic Task 2 samples under one frozen run.

    Attributes:
        model: Model that produced the samples.
        dataset: Dataset as named in the snapshot cell.
        variant: Benchmark variant as named in the snapshot cell.
        dataset_id: Normalized dataset id, part of the join key.
        benchmark_variant: Normalized benchmark variant, part of the join key.
        item_id: Benchmark item the samples answer.
        run_id: Frozen run the samples belong to.
        seed_id: Seed requirement of the item, the capability cluster of the
            bootstrap.
        source_modality: Modality of the item's source requirement.
        valid_n: Number of parsed samples, the ones that are scored.
        total_n: Sample count the group is measured against: the planned
            stochastic samples, or more when more rows were written.
        sample_indices: `sample_index` of each parsed sample, aligned with
            `texts`.
        texts: Semantic text of each parsed sample, the text that is embedded.
        embeddings: Cached vector of each parsed sample, one row per text, or
            `None` until `attach_cached_embeddings` binds them.
    """

    model: str
    dataset: str
    variant: str
    dataset_id: str
    benchmark_variant: str
    item_id: str
    run_id: str
    seed_id: str
    source_modality: str
    valid_n: int
    total_n: int
    sample_indices: list[str]
    texts: list[str]
    embeddings: np.ndarray | None = None

    @property
    def join_key(self) -> paper.PaperJoinKey:
        """Return the exporter's join key of the group's model, cell and item.

        Returns:
            The `paper.paper_join_key` of the group, which matches the key of
            the item's deterministic row.
        """
        return paper.paper_join_key(
            {
                "model": self.model,
                "dataset_id": self.dataset_id,
                "benchmark_variant": self.benchmark_variant,
                "item_id": self.item_id,
            }
        )


def deterministic_task2_rows(
    dataset: str,
    variant: str,
    scored_benchmark: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    plan: eu.SamplingPlan,
) -> list[dict[str, Any]]:
    """Return the exporter's deterministic Task 2 score rows, without live embedding.

    `build_uq_scores` embeds every stochastic group it sees, so it is handed
    the deterministic rows only. Dedupe keys include the sample kind, so the
    deterministic subset dedupes exactly as it does inside the full set.

    Args:
        dataset: Dataset of the cell, stamped on the score rows.
        variant: Benchmark variant of the cell, stamped on the score rows.
        scored_benchmark: Benchmark rows that have current raw outputs.
        raw_rows: Raw rows of the cell; only the deterministic ones are scored.
        plan: Sampling plan passed to `build_uq_scores`.

    Returns:
        The deterministic Task 2 rows as `paper.task2_deterministic_rows`
        selects them.
    """
    det_raw = [row for row in raw_rows if row.get("sample_kind") == "deterministic"]
    scores = eu.build_uq_scores(scored_benchmark, det_raw, sampling_plan=plan)
    paper.stamp_cell_identity(scores, dataset, variant)
    return paper.task2_deterministic_rows(scores)


def stochastic_groups(
    dataset: str,
    variant: str,
    scored_benchmark: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    plan: eu.SamplingPlan,
) -> tuple[list[SampleGroup], Counter[str]]:
    """Return the stochastic Task 2 groups exactly as `build_uq_scores` forms them.

    Args:
        dataset: Dataset of the cell, recorded on each group and normalized
            into its join-key fields.
        variant: Benchmark variant of the cell, handled like `dataset`.
        scored_benchmark: Benchmark rows of the cell; groups whose item is not
            among them are dropped.
        raw_rows: Raw rows of the cell, deduplicated and filtered to the
            current benchmark before grouping.
        plan: Sampling plan that sets each group's expected sample count.

    Returns:
        The groups with at least one parsed sample (the ones that receive a
        score) and, per model, the number of groups with none (no score row).
    """
    benchmark_by_item = {str(row["item_id"]): row for row in scored_benchmark}
    rows = eu.filter_raw_rows_to_current_benchmark(
        scored_benchmark, eu.dedupe_raw_rows(raw_rows)
    )
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("sample_kind") != "stochastic" or row.get("task") != "task2":
            continue
        key = (
            str(row.get("model", "")),
            str(row.get("task", "")),
            str(row.get("item_id", "")),
            str(row.get("run_id", "")),
        )
        grouped.setdefault(key, []).append(row)
    dataset_id = eu.normalize_dataset_id(dataset)
    benchmark_variant = eu.normalize_benchmark_variant(variant)
    groups: list[SampleGroup] = []
    without_valid: Counter[str] = Counter()
    for (model, _task, item_id, run_id), group in grouped.items():
        item = benchmark_by_item.get(item_id)
        if not item:
            continue
        valid = [
            row
            for row in group
            if row.get("parse_status") == "ok"
            and isinstance(row.get("parsed_json"), dict)
        ]
        if not valid:
            without_valid[model] += 1
            continue
        groups.append(
            SampleGroup(
                model=model,
                dataset=dataset,
                variant=variant,
                dataset_id=dataset_id,
                benchmark_variant=benchmark_variant,
                item_id=item_id,
                run_id=run_id,
                seed_id=str(item.get("seed_id", "")),
                source_modality=str(item.get("source_modality", "")),
                valid_n=len(valid),
                total_n=plan.total_samples(len(group)),
                sample_indices=[str(row.get("sample_index", "")) for row in valid],
                texts=eu.semantic_texts_from_rows("task2", valid),
            )
        )
    return groups, without_valid


# ---------------------------------------------------------------------------
# Cached embeddings
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingCache:
    """One run's cached Task 2 sample embeddings, their identity and use counts.

    Attributes:
        artifact_dir: Directory with the run's `manifest.json`,
            `task2_acse_samples.csv` and `task2_acse_sample_embeddings.npz`.
        run_id: Run the cache belongs to, from its manifest.
        model: Model the cache belongs to, from its manifest.
        backend: Embedding backend recorded in the manifest.
        embeddings: Cached vectors, one row per sample row.
        index: Row in `embeddings` of each `(item_id, sample_index)`.
        texts: Semantic text of each cached sample, compared with the text of
            the sample it is bound to.
        used: How often each cached vector was bound to a sample; zeros after
            construction, checked by `check_cache_fully_used`.
    """

    artifact_dir: Path
    run_id: str
    model: str
    backend: str
    embeddings: np.ndarray
    index: dict[tuple[str, str], int]
    texts: list[str]
    used: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.used = np.zeros(self.embeddings.shape[0], dtype=int)


def load_embedding_cache(artifact_dir: Path, expected_backend: str) -> EmbeddingCache:
    """Load one run's cached sample embeddings and check their internal identity.

    Args:
        artifact_dir: The run's cached-embedding directory.
        expected_backend: Backend the manifest must record, as
            `backend:model`.

    Returns:
        The cache, indexed by `(item_id, sample_index)`.

    Raises:
        ValueError: If the manifest records another backend, the embedding
            count differs from the sample rows, a sample row's
            `embedding_index` or item does not match its position, a row has
            another `run_id` than the manifest, or a sample repeats.
    """
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    backend = str(manifest.get("embedding_backend", ""))
    if backend != expected_backend:
        raise ValueError(
            f"{artifact_dir}: cached backend {backend!r} is not {expected_backend!r}"
        )
    samples = eu.read_csv_rows(artifact_dir / "task2_acse_samples.csv")
    store = np.load(
        artifact_dir / "task2_acse_sample_embeddings.npz", allow_pickle=False
    )
    embeddings = np.asarray(store["embeddings"])
    if embeddings.shape[0] != len(samples):
        raise ValueError(
            f"{artifact_dir}: {embeddings.shape[0]} embeddings for {len(samples)} samples"
        )
    cached_items = [str(value) for value in store["item_ids"]]
    index: dict[tuple[str, str], int] = {}
    texts: list[str] = []
    for position, row in enumerate(samples):
        if (
            int(row["embedding_index"]) != position
            or cached_items[position] != row["item_id"]
        ):
            raise ValueError(f"{artifact_dir}: sample row {position} is misaligned")
        if str(row.get("run_id", "")) != str(manifest.get("run_id", "")):
            raise ValueError(
                f"{artifact_dir}: sample row {position} has another run_id"
            )
        key = (str(row["item_id"]), str(row["sample_index"]))
        if key in index:
            raise ValueError(f"{artifact_dir}: duplicate cached sample {key}")
        index[key] = position
        texts.append(str(row["semantic_text"]))
    return EmbeddingCache(
        artifact_dir=artifact_dir,
        run_id=str(manifest.get("run_id", "")),
        model=str(manifest.get("model", "")),
        backend=backend,
        embeddings=embeddings,
        index=index,
        texts=texts,
    )


def attach_cached_embeddings(groups: list[SampleGroup], cache: EmbeddingCache) -> None:
    """Bind every group's samples to cached vectors, failing on any mismatch.

    Args:
        groups: Groups of the cache's run; each gets its `embeddings` set in
            place.
        cache: The run's cache; its `used` counts are incremented.

    Raises:
        ValueError: If a group belongs to another model or run than the cache,
            a sample has no cached vector, or the cached text differs from the
            sample text.
    """
    for group in groups:
        if group.run_id != cache.run_id or group.model != cache.model:
            raise ValueError(
                f"cache {cache.artifact_dir} is for {cache.model}/{cache.run_id}, "
                f"group is {group.model}/{group.run_id}"
            )
        positions: list[int] = []
        for sample_index, text in zip(group.sample_indices, group.texts, strict=True):
            key = (group.item_id, sample_index)
            position = cache.index.get(key)
            if position is None:
                raise ValueError(f"cache {cache.artifact_dir}: no vector for {key}")
            if cache.texts[position] != text:
                raise ValueError(
                    f"cache {cache.artifact_dir}: text differs for {key}: "
                    f"{cache.texts[position]!r} vs {text!r}"
                )
            positions.append(position)
        cache.used[positions] += 1
        group.embeddings = np.asarray(cache.embeddings[positions, :], dtype=float)


def check_cache_fully_used(cache: EmbeddingCache) -> None:
    """Check that every cached vector was bound to exactly one sample.

    Args:
        cache: A cache after `attach_cached_embeddings` has run for all groups
            of its run.

    Raises:
        ValueError: If a cached sample was never used or used more than once.
    """
    unused = int(np.sum(cache.used == 0))
    reused = int(np.sum(cache.used > 1))
    if unused or reused:
        raise ValueError(
            f"cache {cache.artifact_dir}: {unused} cached samples unused, "
            f"{reused} used more than once"
        )


# ---------------------------------------------------------------------------
# Scoring over the grid
# ---------------------------------------------------------------------------


@dataclass
class GridScores:
    """Meaning-variation scores and cluster diagnostics of every group on the grid.

    Attributes:
        thresholds: Cosine-distance thresholds of the grid, in axis order.
        weights: Dispersion weights of the grid, in axis order.
        scores: `[group, threshold, weight]` uncertainty scores.
        cluster_counts: `[group, threshold]` cluster counts.
        mean_pairwise: `[group, threshold]` mean pairwise cosine distance,
            which does not depend on the threshold.
        dominant_mean: `[group, threshold]` dominant-cluster mean distance.
    """

    thresholds: tuple[float, ...]
    weights: tuple[float, ...]
    #: ``[group, threshold, weight]`` uncertainty scores.
    scores: np.ndarray
    #: ``[group, threshold]`` cluster counts.
    cluster_counts: np.ndarray
    #: ``[group, threshold]`` mean pairwise cosine distance (threshold-free).
    mean_pairwise: np.ndarray
    #: ``[group, threshold]`` dominant-cluster mean distance.
    dominant_mean: np.ndarray

    def setting_index(self, threshold: float, weight: float) -> tuple[int, int]:
        """Return the grid position of one setting.

        Args:
            threshold: A threshold on the grid.
            weight: A dispersion weight on the grid.

        Returns:
            The threshold index and the weight index.
        """
        return self.thresholds.index(threshold), self.weights.index(weight)


def score_grid(
    groups: list[SampleGroup],
    thresholds: tuple[float, ...],
    weights: tuple[float, ...],
    backend: str = EXPECTED_BACKEND,
) -> GridScores:
    """Score every group at every (threshold, weight) with the shared scorer.

    Args:
        groups: Groups with their embeddings bound.
        thresholds: Cosine-distance thresholds to score at.
        weights: Dispersion weights to score at.
        backend: Embedding backend label handed to
            `eu.acse_semantic_diagnostics_from_embeddings`.

    Returns:
        The scores and per-threshold cluster diagnostics, in group order.

    Raises:
        ValueError: If a group has no embeddings bound.
    """
    n = len(groups)
    scores = np.full((n, len(thresholds), len(weights)), np.nan)
    counts = np.zeros((n, len(thresholds)), dtype=int)
    pairwise = np.full((n, len(thresholds)), np.nan)
    dominant = np.full((n, len(thresholds)), np.nan)
    for g, group in enumerate(groups):
        if group.embeddings is None:
            raise ValueError(f"group {group.join_key} has no embeddings")
        for t, threshold in enumerate(thresholds):
            for w, weight in enumerate(weights):
                diagnostics = eu.acse_semantic_diagnostics_from_embeddings(
                    group.embeddings,
                    backend,
                    distance_threshold=threshold,
                    dispersion_weight=weight,
                )
                scores[g, t, w] = diagnostics["semantic_uncertainty_score"]
            counts[g, t] = int(diagnostics["semantic_cluster_count"])
            pairwise[g, t] = float(diagnostics["semantic_mean_pairwise_distance"])
            dominant[g, t] = float(
                diagnostics["semantic_dominant_cluster_mean_distance"]
            )
    return GridScores(thresholds, weights, scores, counts, pairwise, dominant)


# ---------------------------------------------------------------------------
# Frozen cohort join
# ---------------------------------------------------------------------------


@dataclass
class Cohort:
    """The evaluated rows: readable deterministic extractions with a scored group.

    Attributes:
        rows: One record per evaluated row with its model, cell, item, seed,
            source modality, strict label and join key, in the exporter's
            pooled order.
        group_index: Position of each row's group in the scored groups.
        labels: Strict-strengthening label of each row, 1 when
            `strict_text_overcommit` is true.
        accounting: Per model and cell, the counts from planned items down to
            evaluated rows, and the completeness of the scored groups.
    """

    rows: list[dict[str, Any]]
    group_index: np.ndarray
    labels: np.ndarray
    accounting: list[dict[str, Any]]

    def mask(
        self, *, model: str | None = None, cell: tuple[str, str] | None = None
    ) -> np.ndarray:
        """Return a row mask for one model, one cell, or both.

        Args:
            model: Keep only rows of this model; `None` keeps every model.
            cell: Keep only rows of this `(dataset, variant)`; `None` keeps
                every cell.

        Returns:
            A boolean array over `rows`.
        """
        keep = np.ones(len(self.rows), dtype=bool)
        for i, row in enumerate(self.rows):
            if model is not None and row["model"] != model:
                keep[i] = False
            if cell is not None and (row["dataset"], row["variant"]) != cell:
                keep[i] = False
        return keep


def join_cohort(
    cells: list[tuple[str, str, list[dict[str, Any]], int]],
    groups: list[SampleGroup],
    without_valid: Mapping[tuple[str, str, str], int],
) -> Cohort:
    """Join readable deterministic rows to their group, in the exporter's order.

    Rows keep the order the exporter's pooled slice has (cells in order,
    raw-row order within a cell), which is what makes the bootstrap's cluster
    ordering reproducible.

    Args:
        cells: `(dataset, variant, deterministic rows, planned items)` in
            snapshot order.
        groups: All scored stochastic groups; each readable row is joined to
            the group with its join key.
        without_valid: Number of groups without a parsed sample per
            `(model, dataset, variant)`, copied into the accounting.

    Returns:
        The cohort with its per-model and per-cell accounting.

    Raises:
        ValueError: If two groups share a join key or a readable row lacks its
            `seed_id` or `item_id`.
    """
    group_by_key = {group.join_key: g for g, group in enumerate(groups)}
    if len(group_by_key) != len(groups):
        raise ValueError("duplicate stochastic group join keys")
    rows: list[dict[str, Any]] = []
    group_index: list[int] = []
    labels: list[int] = []
    counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for dataset, variant, det_rows, planned in cells:
        for row in det_rows:
            model = str(row.get("model", ""))
            tally = counts[(model, dataset, variant)]
            tally["planned_items"] = planned
            tally["deterministic_parsed"] += 1
            if str(row.get("text_modality_parse_status", "")) != "ok":
                tally["unreadable"] += 1
                continue
            tally["readable"] += 1
            g = group_by_key.get(paper.paper_join_key(row))
            if g is None:
                tally["readable_without_group"] += 1
                continue
            label = 1 if eu.is_truthy_strict(row.get("strict_text_overcommit")) else 0
            tally["evaluated"] += 1
            tally["strict_positive"] += label
            seed_id = str(row.get(CAPABILITY_FIELD, "") or "")
            item_id = str(row.get(ITEM_FIELD, "") or "")
            if not seed_id or not item_id:
                raise ValueError("deterministic row without seed_id/item_id")
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "variant": variant,
                    "item_id": item_id,
                    "seed_id": seed_id,
                    "source_modality": str(row.get("source_modality", "")),
                    "strict_text_overcommit": label,
                    "join_key": paper.paper_join_key(row),
                }
            )
            group_index.append(g)
            labels.append(label)
    for group in groups:
        tally = counts[(group.model, group.dataset, group.variant)]
        tally["groups_scored"] += 1
        if group.valid_n == group.total_n:
            tally["groups_complete"] += 1
        else:
            tally["groups_incomplete"] += 1
            tally["missing_or_invalid_samples"] += group.total_n - group.valid_n
        if group.valid_n == 1:
            tally["groups_single_sample"] += 1
    accounting = []
    for (model, dataset, variant), tally in sorted(counts.items()):
        accounting.append(
            {
                "model": model,
                "dataset": dataset,
                "variant": variant,
                "planned_items": tally["planned_items"],
                "deterministic_parsed": tally["deterministic_parsed"],
                "deterministic_missing": tally["planned_items"]
                - tally["deterministic_parsed"],
                "unreadable": tally["unreadable"],
                "readable": tally["readable"],
                "readable_without_group": tally["readable_without_group"],
                "evaluated": tally["evaluated"],
                "strict_positive": tally["strict_positive"],
                "groups_scored": tally["groups_scored"],
                "groups_without_valid_sample": without_valid.get(
                    (model, dataset, variant), 0
                ),
                "groups_complete_five": tally["groups_complete"],
                "groups_incomplete": tally["groups_incomplete"],
                "groups_single_sample": tally["groups_single_sample"],
                "missing_or_invalid_samples": tally["missing_or_invalid_samples"],
            }
        )
    return Cohort(
        rows,
        np.asarray(group_index, dtype=int),
        np.asarray(labels, dtype=int),
        accounting,
    )


# ---------------------------------------------------------------------------
# AUROC and the paired cluster bootstrap
# ---------------------------------------------------------------------------


def rank_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC by the Mann-Whitney statistic with average ranks for ties.

    Equal to `sklearn.metrics.roc_auc_score` (trapezoid over tie-grouped
    thresholds) up to floating-point rounding, and cheap enough to evaluate
    thousands of times per bootstrap.

    Args:
        labels: Binary labels, 1 for a positive row.
        scores: Score of each row; higher scores should rank positives first.

    Returns:
        The AUROC, NaN when one class is absent.
    """
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(scores)
    return float(
        (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


@dataclass
class PairedBootstrap:
    """Point estimates and draw-wise AUROCs for every setting on one slice.

    Attributes:
        cluster_field: Row field whose values were resampled as clusters.
        n_clusters: Number of distinct clusters in the slice.
        point: AUROC of each setting on the full slice.
        draws: AUROC of each setting on each draw, NaN where a draw holds one
            class only.
    """

    cluster_field: str
    n_clusters: int
    point: np.ndarray  # [settings]
    draws: np.ndarray  # [iterations, settings], NaN where undefined

    def interval(self, setting: int) -> tuple[float, float]:
        """Return the 95% percentile interval of one setting's AUROC.

        Args:
            setting: Column of the setting in `draws`.

        Returns:
            The 2.5% and 97.5% quantiles over the defined draws, or two NaNs
            when no draw is defined.
        """
        values = self.draws[:, setting]
        values = values[~np.isnan(values)]
        if values.size == 0:
            return math.nan, math.nan
        low, high = np.quantile(values, [0.025, 0.975])
        return float(low), float(high)

    def delta_interval(self, setting: int, reference: int) -> tuple[float, float, int]:
        """Return the 95% interval of the paired AUROC difference to a reference.

        Args:
            setting: Column of the compared setting in `draws`.
            reference: Column of the reference setting in `draws`.

        Returns:
            The 2.5% and 97.5% quantiles of the draw-wise differences and the
            number of draws where both are defined, or two NaNs and 0 when no
            draw is.
        """
        deltas = self.draws[:, setting] - self.draws[:, reference]
        deltas = deltas[~np.isnan(deltas)]
        if deltas.size == 0:
            return math.nan, math.nan, 0
        low, high = np.quantile(deltas, [0.025, 0.975])
        return float(low), float(high), int(deltas.size)


def cluster_keys_in_order(keys: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return cluster keys in first-appearance order and their row indices.

    Mirrors `pandas.groupby(sort=False)` inside `eu.bootstrap_seed_metric`.

    Args:
        keys: Cluster key of each row.

    Returns:
        The distinct keys as an object array in first-appearance order, and
        the row indices of each key.
    """
    rows_by_key: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        rows_by_key.setdefault(str(key), []).append(i)
    key_array = np.asarray(list(rows_by_key), dtype=object)
    return key_array, {
        key: np.asarray(rows, dtype=int) for key, rows in rows_by_key.items()
    }


def paired_cluster_bootstrap(
    labels: np.ndarray,
    score_matrix: np.ndarray,
    cluster_keys: list[str],
    *,
    cluster_field: str,
    iterations: int,
    seed: int,
) -> PairedBootstrap:
    """Cluster bootstrap of AUROC for every score column on the same draws.

    Draws clusters exactly as `eu.bootstrap_seed_metric` does (same RNG seed,
    same `rng.choice` over the first-appearance-ordered keys), so the column
    for the reported setting reproduces the exporter's interval, and every
    other column is evaluated on the identical resample, which is what makes
    the differences paired.

    Args:
        labels: Binary label of each row of the slice.
        score_matrix: `[row, setting]` scores; every column is evaluated on
            the same draws.
        cluster_keys: Cluster key of each row; whole clusters are resampled.
        cluster_field: Name of the field the keys come from, recorded in the
            result.
        iterations: Number of bootstrap draws.
        seed: RNG seed; the snapshot's seed reproduces the exporter's draws.

    Returns:
        The point AUROCs and the per-draw AUROCs of every column.

    Raises:
        ValueError: If the score matrix and the labels differ in row count.
    """
    if score_matrix.shape[0] != labels.size:
        raise ValueError("score matrix and labels disagree on the row count")
    key_array, rows_by_key = cluster_keys_in_order(cluster_keys)
    n_settings = score_matrix.shape[1]
    point = np.asarray(
        [rank_auroc(labels, score_matrix[:, s]) for s in range(n_settings)]
    )
    draws = np.full((iterations, n_settings), np.nan)
    rng = np.random.default_rng(seed)
    for i in range(iterations):
        chosen = rng.choice(key_array, size=key_array.size, replace=True)
        idx = np.concatenate([rows_by_key[str(key)] for key in chosen])
        y = labels[idx]
        if y.min() == y.max():
            continue
        for s in range(n_settings):
            draws[i, s] = rank_auroc(y, score_matrix[idx, s])
    return PairedBootstrap(cluster_field, int(key_array.size), point, draws)


# ---------------------------------------------------------------------------
# Cluster diagnostics
# ---------------------------------------------------------------------------


def cluster_diagnostics_row(
    slice_name: str,
    threshold: float,
    grid: GridScores,
    group_index: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Cluster and component diagnostics over the evaluated rows at one threshold.

    Args:
        slice_name: Label written to the `slice` column.
        threshold: Threshold on the grid to diagnose.
        grid: Grid scores of all groups; its weights must include 0.0, 1.0 and
            `REFERENCE_WEIGHT`.
        group_index: Group of each evaluated row in the slice.
        labels: Strict label of each evaluated row in the slice.

    Returns:
        One row with cluster-count fractions, the entropy and dispersion
        components, pairwise-distance summaries, the AUROCs of the two rankings
        the score collapses to when every item is one cluster, and zero-score
        and one-cluster fractions per class.
    """
    t = grid.thresholds.index(threshold)
    counts = grid.cluster_counts[group_index, t]
    entropy = grid.scores[group_index, t, grid.weights.index(0.0)]
    dispersion = grid.scores[group_index, t, grid.weights.index(1.0)]
    reference = grid.scores[group_index, t, grid.weights.index(REFERENCE_WEIGHT)]
    pairwise = grid.mean_pairwise[group_index, t]
    n = int(group_index.size)
    row: dict[str, Any] = {
        "slice": slice_name,
        "threshold": threshold,
        "n": n,
        "n_positive": int(labels.sum()),
        "frac_one_cluster": float(np.mean(counts == 1)),
    }
    for k in range(1, MAX_CLUSTER_COUNT + 1):
        row[f"frac_clusters_{k}"] = float(np.mean(counts == k))
    row.update(
        {
            "mean_entropy": float(np.mean(entropy)),
            "frac_entropy_zero": float(np.mean(entropy == 0.0)),
            "mean_dispersion": float(np.mean(dispersion)),
            "frac_dispersion_clipped": float(np.mean(dispersion >= 1.0)),
            "mean_pairwise_distance_min": float(np.min(pairwise)),
            "mean_pairwise_distance_median": float(np.median(pairwise)),
            "mean_pairwise_distance_max": float(np.max(pairwise)),
            "frac_pairwise_above_threshold": float(np.mean(pairwise > threshold)),
            # The two rankings the score collapses to when every item is one
            # cluster: the raw mean pairwise distance and the tie structure
            # alone (five identical texts or not).
            "auroc_mean_pairwise_distance": rank_auroc(labels, pairwise),
            "auroc_any_variation": rank_auroc(labels, (pairwise > 0.0).astype(float)),
            "distinct_scores_reference_weight": int(np.unique(reference).size),
            "frac_score_zero_reference_weight": float(np.mean(reference == 0.0)),
            "frac_score_zero_positives": float(np.mean(reference[labels == 1] == 0.0))
            if labels.sum()
            else math.nan,
            "frac_score_zero_negatives": float(np.mean(reference[labels == 0] == 0.0))
            if (labels == 0).sum()
            else math.nan,
            "frac_one_cluster_positives": float(np.mean(counts[labels == 1] == 1))
            if labels.sum()
            else math.nan,
            "frac_one_cluster_negatives": float(np.mean(counts[labels == 0] == 1))
            if (labels == 0).sum()
            else math.nan,
        }
    )
    return row


# ---------------------------------------------------------------------------
# Live reference scores (the exporter's per-group embedding path)
# ---------------------------------------------------------------------------


def load_live_reference_scores(
    manifest_rows: list[dict[str, Any]],
) -> dict[paper.PaperJoinKey, float]:
    """Load the reported setting's per-item scores from each run's analysis dir.

    `uq_scores.csv` in the run's analysis dir holds the ACSE row the headless
    analysis computed by embedding each five-sample group on its own, which is
    the path the paper exporter also takes. The cache embeds all of a run's
    texts in large padded batches, so its vectors differ from the live ones at
    the fourth decimal. These rows are what the reported AUROCs are made of and
    let the reproduction check separate the cohort from the embeddings.

    Args:
        manifest_rows: Selected-run manifest rows with `analysis_dir`,
            `run_id`, `model`, `dataset_id` and `benchmark_variant`.

    Returns:
        The live Task 2 meaning-variation score per join key, NaN where the
        stored score does not parse.

    Raises:
        FileNotFoundError: If a run's `uq_scores.csv` is missing.
        ValueError: If a join key has two live scores.
    """
    scores: dict[paper.PaperJoinKey, float] = {}
    for record in manifest_rows:
        path = Path(record["analysis_dir"]) / "uq_scores.csv"
        if not path.exists():
            raise FileNotFoundError(f"live reference scores missing: {path}")
        for row in eu.read_csv_rows(path):
            if (
                row.get("uq_method") != eu.ACSE_PROXY_METHOD
                or row.get("task") != "task2"
                or row.get("run_id") != record["run_id"]
                or row.get("model") != record["model"]
            ):
                continue
            key = paper.paper_join_key(
                {
                    "model": record["model"],
                    "dataset_id": record["dataset_id"],
                    "benchmark_variant": record["benchmark_variant"],
                    "item_id": row["item_id"],
                }
            )
            if key in scores:
                raise ValueError(f"duplicate live reference score for {key}")
            scores[key] = _float_or_nan(row.get("uncertainty_score"))
    return scores


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def reported_rq_rows(rq_table: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Index the reported per-model research-question table.

    Args:
        rq_table: The exporter's per-model research-question table CSV.

    Returns:
        Each table row keyed by `(model, dataset, variant)`, where `all` marks
        a pooled level.
    """
    return {
        (row["model"], row["dataset"], row["variant"]): row
        for row in eu.read_csv_rows(rq_table)
    }


def build_cohort(
    root: Path,
    snapshot: Mapping[str, Any],
    manifest_rows: list[dict[str, Any]],
) -> tuple[Cohort, list[SampleGroup], list[dict[str, Any]]]:
    """Load the frozen cells, rebuild the cohort and bind cached embeddings.

    Args:
        root: Checkout root that snapshot paths are resolved against.
        snapshot: The validated frozen snapshot.
        manifest_rows: Selected-run manifest rows with resolved
            `artifact_dir` paths.

    Returns:
        The cohort, all scored groups with their embeddings bound, and one
        record per cache with its identity and SHA-256 file hashes.

    Raises:
        ValueError: If a benchmark's size differs from the snapshot or a frozen
            run has no cached embeddings in the manifest.
    """
    artifact_by_run = {
        (
            row["dataset_id"],
            row["benchmark_variant"],
            row["model"],
            row["run_id"],
        ): Path(row["artifact_dir"])
        for row in manifest_rows
    }
    cells: list[tuple[str, str, list[dict[str, Any]], int]] = []
    groups: list[SampleGroup] = []
    without_valid: dict[tuple[str, str, str], int] = {}
    cache_records: list[dict[str, Any]] = []
    for cell in snapshot["cells"]:
        dataset, variant = cell["dataset"], cell["variant"]
        plan = eu.SamplingPlan(
            stochastic_samples=int(cell["expected_stochastic_samples"])
        )
        started = time.time()
        benchmark = eu.read_csv_rows(_path(root, cell["benchmark_path"]))
        if len(benchmark) != int(cell["n_benchmark_items"]):
            raise ValueError(f"benchmark size changed for {dataset}/{variant}")
        raw_rows = load_cell_raw_rows(root, cell)
        scored_benchmark = eu.benchmark_rows_with_current_raw_outputs(
            benchmark, raw_rows
        )
        det_rows = deterministic_task2_rows(
            dataset, variant, scored_benchmark, raw_rows, plan
        )
        cell_groups, none_valid = stochastic_groups(
            dataset, variant, scored_benchmark, raw_rows, plan
        )
        for model, count in none_valid.items():
            without_valid[(model, dataset, variant)] = count
        by_run: dict[tuple[str, str], list[SampleGroup]] = defaultdict(list)
        for group in cell_groups:
            by_run[(group.model, group.run_id)].append(group)
        for model, run_id in sorted(cell["run_ids"].items()):
            artifact_dir = artifact_by_run.get((dataset, variant, model, run_id))
            if artifact_dir is None:
                raise ValueError(f"no cached embeddings listed for {model}/{run_id}")
            cache = load_embedding_cache(artifact_dir, EXPECTED_BACKEND)
            attach_cached_embeddings(by_run.get((model, run_id), []), cache)
            check_cache_fully_used(cache)
            cache_records.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "model": model,
                    "run_id": run_id,
                    "artifact_dir": _relative(root, artifact_dir),
                    "cached_samples": int(cache.embeddings.shape[0]),
                    "embedding_dtype": str(cache.embeddings.dtype),
                    "samples_sha256": eu.sha256_file(
                        artifact_dir / "task2_acse_samples.csv"
                    ),
                    "embeddings_sha256": eu.sha256_file(
                        artifact_dir / "task2_acse_sample_embeddings.npz"
                    ),
                }
            )
        cells.append((dataset, variant, det_rows, len(scored_benchmark)))
        groups.extend(cell_groups)
        logger.info(
            "cell %s/%s: %d raw rows, %d deterministic rows, %d groups (%.1fs)",
            dataset,
            variant,
            len(raw_rows),
            len(det_rows),
            len(cell_groups),
            time.time() - started,
        )
    cohort = join_cohort(cells, groups, without_valid)
    return cohort, groups, cache_records


def slices_for(
    cohort: Cohort, snapshot: Mapping[str, Any]
) -> list[tuple[str, str, np.ndarray]]:
    """Return `(kind, name, mask)` for the grand, per-model and per-cell slices.

    Args:
        cohort: The evaluated rows to slice.
        snapshot: Frozen snapshot that sets the model and cell order.

    Returns:
        The `pooled` slice named `all`, one `model` slice per cohort model and
        one `cell` slice per `dataset/variant`, in snapshot order.
    """
    slices: list[tuple[str, str, np.ndarray]] = [("pooled", "all", cohort.mask())]
    for model in snapshot["models_cohort"]:
        slices.append(("model", model, cohort.mask(model=model)))
    for cell in snapshot["cells"]:
        slices.append(
            (
                "cell",
                f"{cell['dataset']}/{cell['variant']}",
                cohort.mask(cell=(cell["dataset"], cell["variant"])),
            )
        )
    return slices


def reported_row_value(
    reported: Mapping[tuple[str, str, str], Mapping[str, Any]], kind: str, name: str
) -> Any:
    """Look up the reported meaning-variation AUROC of one slice.

    Args:
        reported: Reported table rows from `reported_rq_rows`.
        kind: Slice kind: `pooled`, `model` or `cell`.
        name: Slice name as built by `slices_for`.

    Returns:
        The `task2_meaning_variation_auroc` value as read from the CSV, or
        `None` when the slice has no reported row.
    """
    return reported.get(rq_key(kind, name), {}).get("task2_meaning_variation_auroc")


def rq_key(kind: str, name: str) -> tuple[str, str, str]:
    """Map a slice to its key in the reported table.

    Args:
        kind: Slice kind: `pooled`, `model` or `cell`.
        name: Model for a model slice, `dataset/variant` for a cell slice;
            ignored for the pooled slice.

    Returns:
        The `(model, dataset, variant)` key, with `all` for pooled levels.
    """
    if kind == "pooled":
        return ("all", "all", "all")
    if kind == "model":
        return (name, "all", "all")
    dataset, variant = name.split("/")
    return ("all", dataset, variant)


def run(
    root: Path,
    provenance_path: Path,
    manifest_path: Path,
    rq_table_path: Path,
    output_dir: Path,
    *,
    thresholds: tuple[float, ...],
    weights: tuple[float, ...],
    iterations: int | None,
    reproduce_only: bool,
) -> dict[str, Any]:
    """Re-score the frozen cohort over the grid and write the sensitivity outputs.

    Builds the cohort once, compares the cached-embedding scores with the live
    reference scores and the reported table, and bootstraps every setting on
    the same capability-clustered draws per slice. Writes
    `cohort_accounting.csv`, `reproduction_check.csv`, `grid_auroc.csv`,
    `cluster_diagnostics.csv`, `per_item_scores.csv` and `settings.json`.

    Args:
        root: Checkout root that recorded paths are resolved against.
        provenance_path: Frozen snapshot provenance JSON.
        manifest_path: Selected-run manifest with each run's analysis and
            cached-embedding directories.
        rq_table_path: Reported per-model research-question table; when it is
            missing, the reported columns stay empty.
        output_dir: Directory for the outputs, created when absent.
        thresholds: Cosine-distance thresholds of the grid.
        weights: Dispersion weights of the grid; 0.0 and 1.0 are added when
            absent because they double as the entropy and dispersion
            components.
        iterations: Bootstrap draws per slice; `None` uses the snapshot's
            `bootstrap_samples`.
        reproduce_only: Score the reported setting only, in place of
            `thresholds` and `weights`.

    Returns:
        The reproduction, grid and diagnostics rows and the cohort.

    Raises:
        ValueError: If an evaluated row has no live reference score or the grid
            lacks the reported setting.
    """
    snapshot = load_snapshot(provenance_path)
    seed = int(snapshot["bootstrap_seed"])
    iterations = (
        int(snapshot["bootstrap_samples"]) if iterations is None else iterations
    )
    # The tracked selected-run manifest records checkout-relative directories.
    manifest_rows = [
        {
            **row,
            **{
                key: str(_path(root, str(row[key])))
                for key in ("analysis_dir", "artifact_dir")
                if row.get(key)
            },
        }
        for row in eu.read_csv_rows(manifest_path)
    ]
    reported = reported_rq_rows(rq_table_path) if rq_table_path.exists() else {}
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort, groups, cache_records = build_cohort(root, snapshot, manifest_rows)
    logger.info("cohort: %d evaluated rows, %d groups", len(cohort.rows), len(groups))
    eu.write_csv_rows(output_dir / "cohort_accounting.csv", cohort.accounting)
    live_by_key = load_live_reference_scores(manifest_rows)
    live = np.asarray(
        [live_by_key.get(row["join_key"], math.nan) for row in cohort.rows], dtype=float
    )
    if np.isnan(live).any():
        raise ValueError(
            f"{int(np.isnan(live).sum())} evaluated rows have no live reference score"
        )

    if reproduce_only:
        thresholds = (REFERENCE_THRESHOLD,)
        weights = (REFERENCE_WEIGHT,)
    if REFERENCE_THRESHOLD not in thresholds or REFERENCE_WEIGHT not in weights:
        raise ValueError("the grid must contain the reported setting")
    if 0.0 not in weights or 1.0 not in weights:
        # The endpoints double as the entropy and dispersion components.
        weights = tuple(sorted(set(weights) | {0.0, 1.0}))
    started = time.time()
    grid = score_grid(groups, thresholds, weights)
    logger.info(
        "scored %d groups x %d settings in %.1fs",
        len(groups),
        grid.scores[0].size,
        time.time() - started,
    )
    t_ref, w_ref = grid.setting_index(REFERENCE_THRESHOLD, REFERENCE_WEIGHT)

    settings = [(t, w) for t in range(len(thresholds)) for w in range(len(weights))]
    reference_setting = settings.index((t_ref, w_ref))
    row_scores = grid.scores[cohort.group_index]  # [rows, thresholds, weights]
    matrix = row_scores.reshape(len(cohort.rows), -1)

    reproduction_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    for kind, name, mask in slices_for(cohort, snapshot):
        labels = cohort.labels[mask]
        slice_matrix = matrix[mask]
        rows = [row for row, keep in zip(cohort.rows, mask, strict=True) if keep]
        seed_keys = [row["seed_id"] for row in rows]
        item_keys = [row["item_id"] for row in rows]
        started = time.time()
        capability = paired_cluster_bootstrap(
            labels,
            slice_matrix,
            seed_keys,
            cluster_field=CAPABILITY_FIELD,
            iterations=iterations,
            seed=seed,
        )
        # The exporter's primary interval resamples the item; reproduce it for
        # the reported setting only.
        item = paired_cluster_bootstrap(
            labels,
            slice_matrix[:, [reference_setting]],
            item_keys,
            cluster_field=ITEM_FIELD,
            iterations=iterations,
            seed=seed,
        )
        live_scores = live[mask][:, None]
        live_capability = paired_cluster_bootstrap(
            labels,
            live_scores,
            seed_keys,
            cluster_field=CAPABILITY_FIELD,
            iterations=iterations,
            seed=seed,
        )
        live_item = paired_cluster_bootstrap(
            labels,
            live_scores,
            item_keys,
            cluster_field=ITEM_FIELD,
            iterations=iterations,
            seed=seed,
        )
        logger.info(
            "slice %s/%s: %d rows, bootstrap %.1fs",
            kind,
            name,
            labels.size,
            time.time() - started,
        )
        reference_auroc = capability.point[reference_setting]
        live_auroc = live_capability.point[0]
        live_cap_low, live_cap_high = live_capability.interval(0)
        live_item_low, live_item_high = live_item.interval(0)
        reported_auroc = _float_or_nan(reported_row_value(reported, kind, name))
        sklearn_auroc = eu.auroc_score(
            labels.tolist(), slice_matrix[:, reference_setting].tolist()
        )
        reported_row = reported.get(rq_key(kind, name), {})
        cap_low, cap_high = capability.interval(reference_setting)
        item_low, item_high = item.interval(0)
        reproduction_rows.append(
            {
                "slice_kind": kind,
                "slice": name,
                "n": int(labels.size),
                "n_positive": int(labels.sum()),
                "reported_n": _float_or_nan(
                    reported_row.get("task2_meaning_variation_auroc_n")
                ),
                "reported_n_positive": _float_or_nan(
                    reported_row.get("task2_meaning_variation_auroc_n_positive")
                ),
                "reported_auroc": reported_auroc,
                "live_auroc": live_auroc,
                "abs_diff_live_vs_reported": abs(live_auroc - reported_auroc),
                "live_capability_ci_low": live_cap_low,
                "live_capability_ci_high": live_cap_high,
                "live_item_ci_low": live_item_low,
                "live_item_ci_high": live_item_high,
                "auroc": reference_auroc,
                "auroc_sklearn": sklearn_auroc,
                "abs_diff_auroc": abs(reference_auroc - reported_auroc),
                "abs_diff_cache_vs_live_auroc": abs(reference_auroc - live_auroc),
                "max_abs_score_diff_cache_vs_live": float(
                    np.max(np.abs(slice_matrix[:, reference_setting] - live[mask]))
                ),
                "capability_ci_low": cap_low,
                "capability_ci_high": cap_high,
                "reported_capability_ci_low": _float_or_nan(
                    reported_row.get("task2_meaning_variation_auroc_seed_ci_low")
                ),
                "reported_capability_ci_high": _float_or_nan(
                    reported_row.get("task2_meaning_variation_auroc_seed_ci_high")
                ),
                "item_ci_low": item_low,
                "item_ci_high": item_high,
                "reported_item_ci_low": _float_or_nan(
                    reported_row.get("task2_meaning_variation_auroc_ci_low")
                ),
                "reported_item_ci_high": _float_or_nan(
                    reported_row.get("task2_meaning_variation_auroc_ci_high")
                ),
                "reported_ci_cluster_field": reported_row.get(
                    "task2_ci_cluster_field", ""
                ),
                "n_capability_clusters": capability.n_clusters,
                "n_item_clusters": item.n_clusters,
            }
        )
        for s, (t, w) in enumerate(settings):
            low, high = capability.interval(s)
            d_low, d_high, n_defined = capability.delta_interval(s, reference_setting)
            auroc = capability.point[s]
            grid_rows.append(
                {
                    "slice_kind": kind,
                    "slice": name,
                    "threshold": thresholds[t],
                    "dispersion_weight": weights[w],
                    "entropy_weight": 1.0 - weights[w],
                    "is_reference": s == reference_setting,
                    "n": int(labels.size),
                    "n_positive": int(labels.sum()),
                    "auroc": auroc,
                    "auroc_undefined": bool(math.isnan(auroc)),
                    "ci_low": low,
                    "ci_high": high,
                    "delta_vs_reference": auroc - reference_auroc,
                    "delta_ci_low": d_low,
                    "delta_ci_high": d_high,
                    "n_defined_draws": n_defined,
                    "ci_cluster_field": CAPABILITY_FIELD,
                    "n_clusters": capability.n_clusters,
                }
            )
        for threshold in thresholds:
            diagnostics_rows.append(
                cluster_diagnostics_row(
                    f"{kind}:{name}", threshold, grid, cohort.group_index[mask], labels
                )
            )

    eu.write_csv_rows(output_dir / "reproduction_check.csv", reproduction_rows)
    eu.write_csv_rows(output_dir / "grid_auroc.csv", grid_rows)
    eu.write_csv_rows(output_dir / "cluster_diagnostics.csv", diagnostics_rows)
    write_per_item_scores(output_dir / "per_item_scores.csv", cohort, grid, live)
    settings_record = {
        "generated_at_utc": eu.utc_now_iso(),
        "script": "scripts/meaning_variation_sensitivity.py",
        "provenance": _relative(root, provenance_path),
        "provenance_generated_at_utc": snapshot.get("generated_at_utc", ""),
        "selected_manifest": _relative(root, manifest_path),
        "reported_table": _relative(root, rq_table_path),
        "embedding_backend": EXPECTED_BACKEND,
        "thresholds": list(thresholds),
        "dispersion_weights": list(weights),
        "reference": {
            "threshold": REFERENCE_THRESHOLD,
            "dispersion_weight": REFERENCE_WEIGHT,
        },
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "bootstrap_cluster_field": CAPABILITY_FIELD,
        "reproduction_cluster_fields": [CAPABILITY_FIELD, ITEM_FIELD],
        "reproduce_only": reproduce_only,
        "models": snapshot["models_cohort"],
        "cells": [
            {"dataset": c["dataset"], "variant": c["variant"], "run_ids": c["run_ids"]}
            for c in snapshot["cells"]
        ],
        "n_evaluated_rows": len(cohort.rows),
        "n_groups_scored": len(groups),
        "caches": cache_records,
    }
    eu.write_json(output_dir / "settings.json", settings_record)
    return {
        "reproduction": reproduction_rows,
        "grid": grid_rows,
        "diagnostics": diagnostics_rows,
        "cohort": cohort,
    }


def write_per_item_scores(
    path: Path, cohort: Cohort, grid: GridScores, live: np.ndarray
) -> None:
    """Write each evaluated row's scores and cluster diagnostics at every setting.

    Args:
        path: Output CSV path.
        cohort: The evaluated rows; their join key is left out.
        grid: Grid scores of all groups.
        live: Live reference score of each evaluated row, in cohort order.
    """
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(cohort.rows):
        g = int(cohort.group_index[i])
        record = {key: value for key, value in row.items() if key != "join_key"}
        record["score_reference_live"] = float(live[i])
        for t, threshold in enumerate(grid.thresholds):
            record[f"clusters_t{threshold:.2f}"] = int(grid.cluster_counts[g, t])
            record[f"mean_pairwise_t{threshold:.2f}"] = float(grid.mean_pairwise[g, t])
            record[f"dominant_mean_t{threshold:.2f}"] = float(grid.dominant_mean[g, t])
            for w, weight in enumerate(grid.weights):
                record[f"score_{setting_label(threshold, weight)}"] = float(
                    grid.scores[g, t, w]
                )
        rows.append(record)
    eu.write_csv_rows(path, rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line options.

    Args:
        argv: Arguments to parse; `None` reads `sys.argv`.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rq-table", type=Path, default=DEFAULT_RQ_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--iterations", type=int, default=None, help="default: the snapshot's"
    )
    parser.add_argument(
        "--reproduce-only",
        action="store_true",
        help="score the reported setting only and write the reproduction check",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the sensitivity analysis and print the reproduction check per slice.

    Relative paths among the options are resolved under `--root`.

    Args:
        argv: Command-line arguments; `None` reads `sys.argv`.

    Returns:
        The process exit code, 0 on success.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    root = args.root.resolve()

    def under_root(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    result = run(
        root,
        under_root(args.provenance),
        under_root(args.manifest),
        under_root(args.rq_table),
        under_root(args.output_dir),
        thresholds=THRESHOLDS,
        weights=WEIGHTS,
        iterations=args.iterations,
        reproduce_only=args.reproduce_only,
    )
    for row in result["reproduction"]:
        print(
            f"{row['slice_kind']:6s} {row['slice']:18s} n={row['n']:6d} "
            f"pos={row['n_positive']:5d} reported={row['reported_auroc']:.9f} "
            f"live={row['live_auroc']:.9f} |d|={row['abs_diff_live_vs_reported']:.1e} "
            f"cache={row['auroc']:.6f} |d|={row['abs_diff_auroc']:.1e} "
            f"live cap CI [{row['live_capability_ci_low']:.6f}, "
            f"{row['live_capability_ci_high']:.6f}] reported "
            f"[{row['reported_capability_ci_low']:.6f}, "
            f"{row['reported_capability_ci_high']:.6f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
