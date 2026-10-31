"""Compose the figure for the embedding modal-force-drift diagnostic.

Three panels:
  (a) 2D projection of requirement-only MLX embeddings, colored by *source
      modality* -- the structure the geometry does carry (input commitment level).
  (b) same coordinates, colored by *text drift* (strengthened vs clean) -- the
      target, which is sprinkled within every region rather than separated.
  (c) held-out probe AUROC on requirement-only text (item-grouped by default):
      source modality and dataset from the MLX embeddings, then strict
      strengthening, averaged over the recommended, optional and nice-to-have
      input levels, from the MLX embeddings and from char n-gram TF-IDF
      features (the "keyword search" bar).

Reads the requirement-only embedding cache and probe grid written by
`diagnose_embedding_separability.py`. The paper figure is now drawn by
`plot_embedding_diagnostic_figure_v2.py`. This script stays because
`plot_embedding_diagnostic_tsne_supp.py` reuses its projection helpers.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "re_uq_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import eval_utils as eu
    from plot_acse_global_embedding_projection import (
        drift_status,
        load_embeddings_and_rows,
        manifest_rows,
    )
    from probe_acse_embedding_separability import requirements_digest
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu
    from scripts.plot_acse_global_embedding_projection import (
        drift_status,
        load_embeddings_and_rows,
        manifest_rows,
    )
    from scripts.probe_acse_embedding_separability import requirements_digest

SOURCE_ORDER = ["mandatory", "recommended", "optional", "nice_to_have"]
SOURCE_LABEL = {
    "mandatory": "mandatory",
    "recommended": "recommended",
    "optional": "optional",
    "nice_to_have": "weak intent",
}


def set_style() -> None:
    """Set the serif manuscript style for fonts, spines and resolution."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 12.0,
            "axes.titlesize": 12.5,
            "axes.labelsize": 12.0,
            "legend.fontsize": 10.5,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "figure.dpi": 200,
        }
    )


def read_summary(path: Path) -> list[dict[str, str]]:
    """Read the probe grid summary CSV.

    Args:
        path: `probe_grid_summary.csv` written by
            `diagnose_embedding_separability.py`.

    Returns:
        One dict of strings per row.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float:
    """Parse a value as a float, NaN when it does not parse.

    Args:
        value: Value of a CSV cell.

    Returns:
        The float or NaN.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def balanced_subsample(
    rows: list[dict[str, Any]],
    per_modality: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an equal number of points per source modality.

    Points are sampled uniformly so drift appears at its true within-level
    prevalence (preserved stays the visible majority).

    Args:
        rows: Candidate rows with `source_modality`.
        per_modality: Most points per source modality; smaller levels are kept
            whole.
        rng: Random generator, advanced by the draws.

    Returns:
        The sorted indices of the chosen rows. Rows of other modalities than
        the four in `SOURCE_ORDER` are never chosen.
    """
    chosen: list[int] = []
    for modality in SOURCE_ORDER:
        idx = np.asarray(
            [
                i
                for i, row in enumerate(rows)
                if str(row.get("source_modality", "")) == modality
            ]
        )
        if idx.size == 0:
            continue
        if idx.size > per_modality:
            idx = rng.choice(idx, size=per_modality, replace=False)
        chosen.extend(idx.tolist())
    return np.asarray(sorted(set(chosen)))


class ProjectionInputs(NamedTuple):
    """Everything both embedding-diagnostic figures need to draw a projection.

    `rng` is handed back mid-stream on purpose: `balanced_subsample` has
    already drawn from it, and the draw-order shuffles of the projection panels
    continue from there. Two figures sharing this helper therefore also share
    one RNG consumption order.

    Attributes:
        coords: 2D projection of each subsampled point.
        rows: Sample rows of the subsampled points, aligned with `coords`.
        indices: Positions of the subsampled points in the full sample rows
            and in the embedding cache.
        rng: The random generator after the subsampling draws.
    """

    coords: np.ndarray
    rows: list[dict[str, Any]]
    indices: np.ndarray
    rng: np.random.Generator


def prepare_projection_inputs(
    diagnostic_dir: Path,
    manifest_path: Path,
    method: str,
    per_modality: int,
    random_state: int,
) -> ProjectionInputs:
    """Load the requirement-only cache, subsample it, and project it.

    Each unique generated requirement appears ~17x across seeds, so one point
    per distinct string is plotted; stacking identical points would otherwise
    make the maps unreadable.

    Args:
        diagnostic_dir: Directory with `task2_reqonly_mlx_embeddings.npz`.
        manifest_path: Cache inventory whose `mlx:` rows give the sample rows
            aligned with the cache.
        method: Projection method, `pca` or `tsne` (see `compute_projection`).
        per_modality: Most distinct requirements per source modality.
        random_state: Seed of the subsample and the projection.

    Returns:
        The projection, its rows and indices, and the advanced generator.

    Raises:
        ValueError: If the cache's row count or requirements digest does not
            match the sample rows.
    """
    cache_path = diagnostic_dir / "task2_reqonly_mlx_embeddings.npz"
    cache = np.load(cache_path, allow_pickle=False)
    reqonly = cache["embeddings"].astype(np.float32, copy=False)
    rows = manifest_rows(manifest_path, "mlx:")
    _, sample_rows = load_embeddings_and_rows(rows)
    if len(sample_rows) != reqonly.shape[0]:
        raise ValueError(
            f"row/embedding mismatch: {len(sample_rows)} vs {reqonly.shape[0]}"
        )
    # A row count alone cannot tell a stale cache of equal length apart from the
    # right one, which is why the probe writes the digest; the probe's
    # --reqonly-cache flag can also overwrite this file from another run.
    requirements = [str(row.get("requirement", "")) for row in sample_rows]
    expected_digest = requirements_digest(requirements)
    cached_digest = (
        str(cache["requirements_digest"])
        if "requirements_digest" in cache.files
        else ""
    )
    if cached_digest != expected_digest:
        raise ValueError(
            f"{cache_path} was written for different requirement text "
            f"({cached_digest or 'no digest recorded'} != {expected_digest}); "
            "re-run diagnose_embedding_separability.py to rebuild it."
        )

    seen: set[str] = set()
    unique_global: list[int] = []
    for i, row in enumerate(sample_rows):
        text = str(row.get("requirement", ""))
        if text and text not in seen:
            seen.add(text)
            unique_global.append(i)
    unique_rows = [sample_rows[i] for i in unique_global]
    unique_global_arr = np.asarray(unique_global)
    print(f"unique generated requirements: {len(unique_rows)}")

    rng = np.random.default_rng(random_state)
    keep_local = balanced_subsample(unique_rows, per_modality, rng)
    global_idx = unique_global_arr[keep_local]
    sub_rows = [unique_rows[k] for k in keep_local]
    print(
        f"projection subsample: {global_idx.size} points "
        f"(strict positives kept: {sum(drift_status(r) == 'strict_text_oc' for r in sub_rows)})"
    )
    coords = compute_projection(reqonly[global_idx], method, random_state)
    return ProjectionInputs(coords, sub_rows, global_idx, rng)


def compute_projection(
    embeddings: np.ndarray, method: str, random_state: int
) -> np.ndarray:
    """Project embeddings to two dimensions with PCA or t-SNE.

    Args:
        embeddings: Matrix with one embedding per row.
        method: `pca` for a direct projection; any other value runs t-SNE
            (perplexity 40) on a PCA reduction to at most 50 components.
        random_state: Seed of PCA and t-SNE.

    Returns:
        The 2D coordinates, one row per embedding.
    """
    if method == "pca":
        pca = PCA(n_components=2, svd_solver="randomized", random_state=random_state)
        return np.asarray(pca.fit_transform(embeddings), dtype=float)
    prepca = PCA(
        n_components=min(50, embeddings.shape[1]),
        svd_solver="randomized",
        random_state=random_state,
    )
    prepared = prepca.fit_transform(embeddings)
    tsne = TSNE(
        n_components=2,
        perplexity=40.0,
        learning_rate="auto",
        init="pca",
        max_iter=1000,
        random_state=random_state,
        verbose=1,
    )
    return np.asarray(tsne.fit_transform(prepared), dtype=float)


# Okabe-Ito colour-blind-safe, maximally distinct across the four input levels
SRC_PALETTE = {
    "mandatory": "#0072B2",  # blue
    "recommended": "#009E73",  # bluish green
    "optional": "#E69F00",  # orange
    "nice_to_have": "#CC79A7",  # reddish purple
}
DRIFT_PALETTE = {
    "clean": "#9aa3af",
    "broad_text_oc": "#E69F00",
    "strict_text_oc": "#C1121F",
}
DRIFT_LABEL = {
    "clean": "Not flagged",
    "broad_text_oc": "Strengthened: broad check only",
    "strict_text_oc": "Strengthened: strict check",
}
SOURCE_LEGEND_ORDER = ["mandatory", "recommended", "optional", "nice_to_have"]


def panel_projection(
    ax: Axes,
    coords: np.ndarray,
    rows: list[dict[str, Any]],
    color_field: str,
    title: str,
    rng: np.random.Generator | None = None,
) -> None:
    """Draw one projection panel, coloured by source modality or by drift.

    Points are drawn in one shuffled scatter so that no group systematically
    hides another.

    Args:
        ax: Axes to draw into.
        coords: 2D coordinates of the points.
        rows: Sample rows aligned with `coords`.
        color_field: `source_modality` colours by input level; any other value
            colours by drift status.
        title: Panel title.
        rng: Generator for the shuffle. `None` leaves the source panel
            unshuffled and shuffles the drift panel with a fixed seed of 0.
    """
    if color_field == "source_modality":
        keys = np.asarray(
            [str(row.get("source_modality", "")) for row in rows], dtype=object
        )
        # single scatter in shuffled order so no input level systematically occludes another
        colors = np.asarray([SRC_PALETTE.get(k, "#94a3b8") for k in keys], dtype=object)
        order = np.arange(len(rows))
        if rng is not None:
            rng.shuffle(order)
        ax.scatter(
            coords[order, 0],
            coords[order, 1],
            s=9.5,
            c=list(colors[order]),
            alpha=0.72,
            linewidths=0.0,
            zorder=2,
            rasterized=True,
        )
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=4.5,
                markerfacecolor=SRC_PALETTE[name],
                markeredgecolor="none",
                label=SOURCE_LABEL[name],
            )
            for name in SOURCE_LEGEND_ORDER
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            handletextpad=0.2,
            borderpad=0.25,
            labelspacing=0.25,
            framealpha=0.85,
        )
    else:
        keys = np.asarray([drift_status(row) for row in rows], dtype=object)
        # One scatter in shuffled order, as in the source panel: strengthened
        # outputs are about a quarter of the points and sit next to their
        # capability's other variants, so drawing them last would hide the rest.
        colors = np.asarray(
            [DRIFT_PALETTE.get(k, "#94a3b8") for k in keys], dtype=object
        )
        order = np.arange(len(rows))
        (rng if rng is not None else np.random.default_rng(0)).shuffle(order)
        ax.scatter(
            coords[order, 0],
            coords[order, 1],
            s=9.5,
            c=list(colors[order]),
            alpha=0.72,
            linewidths=0.0,
            zorder=2,
            rasterized=True,
        )
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=4.5,
                markerfacecolor=DRIFT_PALETTE[name],
                markeredgecolor="none",
                label=DRIFT_LABEL[name],
            )
            for name in DRIFT_LABEL
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            handletextpad=0.2,
            borderpad=0.25,
            labelspacing=0.25,
            framealpha=0.85,
        )
    ax.set_title(title, fontsize=12.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.02)
    for (
        spine
    ) in ax.spines.values():  # light frame so (a) and (b) read as separate panels
        spine.set_visible(True)
        spine.set_edgecolor("#cbd5e1")
        spine.set_linewidth(0.8)


def auroc_cell(
    summary: list[dict[str, Any]],
    *,
    backend: str,
    text: str,
    group: str,
    scope: str,
    target: str,
    model: str = "hgb",
) -> float:
    """Look up the mean held-out AUROC of one probe grid cell.

    Args:
        summary: Rows of `probe_grid_summary.csv`.
        backend: Feature backend, `mlx` or `tfidf`.
        text: Text variant, `reqonly` or `prefixed`.
        group: Grouping mode, `seed` or `item`.
        scope: Probe scope, such as `global` or `source_modality=optional`.
        target: Probe target.
        model: Classifier.

    Returns:
        The cell's `auroc_mean`, NaN when the cell is missing or empty.
    """
    for row in summary:
        if (
            row["feature_backend"] == backend
            and row["text_variant"] == text
            and row["group_mode"] == group
            and row["scope"] == scope
            and row["target"] == target
            and row["model"] == model
        ):
            return as_float(row["auroc_mean"])
    return float("nan")


def style_detection_axis(ax: Axes) -> None:
    """Style the shared 'detection score' x-axis: 0.5 = coin flip, 1.0 = perfect.

    Args:
        ax: Axes of a detection bar chart.
    """
    ax.set_xlim(0.5, 1.02)
    ax.axvline(0.5, ls="--", lw=0.9, color="#0f172a", zorder=0)
    ax.set_xticks([0.5, 0.75, 1.0])
    ax.set_xticklabels(["0.5\ncoin flip", "0.75", "1.0\nperfect"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def hbar_detection(
    ax: Axes, labels: list[str], values: list[float], colors: list[str]
) -> np.ndarray:
    """Draw horizontal detection bars that start at the 0.5 chance line.

    Args:
        ax: Axes to draw into.
        labels: Bar labels, drawn top to bottom.
        values: AUROC per bar; NaN bars are skipped.
        colors: Colour per bar.

    Returns:
        The y position of each bar, in label order.
    """
    y = np.arange(len(labels))[::-1]
    for yi, val, color in zip(y, values, colors, strict=True):
        if np.isnan(val):
            continue
        ax.barh(yi, val - 0.5, left=0.5, color=color, height=0.52, zorder=2)
        ax.text(
            val + 0.012,
            yi,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=11.5,
            color="#0f172a",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11.0)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    return y


def panel_readout(
    ax: Axes, summary: list[dict[str, Any]], group_mode: str, model: str
) -> None:
    """Draw panel (c): what gradient-boosted trees can read off the generated text.

    The bars show the held-out AUROC for source modality and dataset, and the
    mean within-level AUROC for strict strengthening with the neural embedding
    and with TF-IDF features, all on requirement-only text.

    Args:
        ax: Axes to draw into.
        summary: Rows of `probe_grid_summary.csv`.
        group_mode: Grouping mode of the cells read, `seed` or `item`.
        model: Classifier of the cells read.
    """

    def within_mean(backend: str) -> float:
        return float(
            np.nanmean(
                [
                    auroc_cell(
                        summary,
                        backend=backend,
                        text="reqonly",
                        group=group_mode,
                        scope=f"source_modality={modality}",
                        target="deterministic_strict_text_overcommit",
                        model=model,
                    )
                    for modality in ("nice_to_have", "optional", "recommended")
                ]
            )
        )

    source_modality_auroc = auroc_cell(
        summary,
        backend="mlx",
        text="reqonly",
        group=group_mode,
        scope="global",
        target="source_modality",
        model=model,
    )
    dataset_variant_auroc = auroc_cell(
        summary,
        backend="mlx",
        text="reqonly",
        group=group_mode,
        scope="global",
        target="dataset_variant",
        model=model,
    )
    strengthened_embedding_auroc = within_mean("mlx")
    strengthened_tfidf_auroc = within_mean("tfidf")

    labels = [
        "Original input strength",
        "Source dataset",
        "Strengthened?\n(neural embedding)",
        "Strengthened?\n(keyword search)",
    ]
    values = [
        source_modality_auroc,
        dataset_variant_auroc,
        strengthened_embedding_auroc,
        strengthened_tfidf_auroc,
    ]
    colors = ["#009E73", "#009E73", "#D55E00", "#D55E00"]
    y = hbar_detection(ax, labels, values, colors)
    mid = (y[2] + y[3]) / 2.0
    ax.annotate(
        "",
        xy=(0.5 + 0.001, y[3] - 0.33),
        xytext=(0.5 + 0.001, y[2] + 0.33),
        arrowprops={"arrowstyle": "-", "lw": 0.0},
    )
    ax.text(
        max(strengthened_embedding_auroc, strengthened_tfidf_auroc) + 0.05,
        mid,
        "≈ coin flip\n(neither works)",
        va="center",
        ha="left",
        fontsize=10.5,
        color="#D55E00",
        fontstyle="italic",
    )
    style_detection_axis(ax)
    ax.set_title(
        "(c) What can be read from a generated requirement  (HistGradientBoosting)",
        fontsize=12.0,
    )


def main() -> None:
    """Compose the embedding diagnostic figure and save it as PDF and PNG.

    Reads `probe_grid_summary.csv` and the requirement-only embedding cache
    from `--diagnostic-dir` and draws the two projection panels and the
    read-out panel.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs") / eu.ACSE_SEMANTIC_MANIFEST_FILENAME,
    )
    parser.add_argument(
        "--diagnostic-dir", type=Path, default=Path("outputs/embedding_diagnostic")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/figures/embedding_diagnostic.pdf")
    )
    parser.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    parser.add_argument("--per-modality", type=int, default=850)
    parser.add_argument("--group-mode", default="item")
    parser.add_argument("--model", default="hgb")
    parser.add_argument("--random-state", type=int, default=20260527)
    args = parser.parse_args()

    root = eu.project_root()
    diagnostic_dir = (
        args.diagnostic_dir
        if args.diagnostic_dir.is_absolute()
        else root / args.diagnostic_dir
    )
    manifest_path = (
        args.manifest if args.manifest.is_absolute() else root / args.manifest
    )
    output_path = args.output if args.output.is_absolute() else root / args.output

    summary = read_summary(diagnostic_dir / "probe_grid_summary.csv")
    coords, sub_rows, _, rng = prepare_projection_inputs(
        diagnostic_dir=diagnostic_dir,
        manifest_path=manifest_path,
        method=args.method,
        per_modality=args.per_modality,
        random_state=args.random_state,
    )

    set_style()
    fig = plt.figure(figsize=(8.0, 5.7), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.015, h_pad=0.015, wspace=0.05, hspace=0.05)
    gs = fig.add_gridspec(2, 12, height_ratios=[1.62, 0.68])
    ax_a = fig.add_subplot(gs[0, 0:6])
    ax_b = fig.add_subplot(gs[0, 6:12])
    ax_c = fig.add_subplot(gs[1, 1:11])

    panel_projection(
        ax_a,
        coords,
        sub_rows,
        "source_modality",
        "(a) Colored by input strength",
        rng=rng,
    )
    panel_projection(
        ax_b, coords, sub_rows, "drift", "(b) Colored by strength increase", rng=rng
    )
    panel_readout(ax_c, summary, args.group_mode, args.model)
    fig.suptitle("Generated requirements", fontsize=14.0, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {output_path} and {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
