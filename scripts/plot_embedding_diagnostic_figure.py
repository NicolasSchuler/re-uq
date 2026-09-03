"""Compose the paper figure for the embedding modal-force-drift diagnostic.

Four panels:
  (a) 2D projection of requirement-only MLX embeddings, colored by *source
      modality* -- the structure the geometry does carry (input commitment level).
  (b) same coordinates, colored by *text drift* (strengthened vs clean) -- the
      target, which is sprinkled within every region rather than separated.
  (c) held-out probe AUROC (item-grouped, requirement-only MLX): the confounds
      (source modality, dataset) are recoverable; drift detection collapses to
      chance once the input commitment level is held fixed.
  (d) drift-detection AUPRC as lift over the prevalence baseline, within input
      level, MLX vs TF-IDF -- the char n-gram keyword detector keeps some signal
      that the neural meaning embedding washes out.

Reads the requirement-only embedding cache and probe grid written by
``diagnose_embedding_separability.py``.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Any

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "re_uq_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu
    from scripts.plot_acse_global_embedding_projection import (
        drift_status,
        load_embeddings_and_rows,
        manifest_rows,
    )

SOURCE_ORDER = ["mandatory", "recommended", "optional", "nice_to_have"]
SOURCE_LABEL = {
    "mandatory": "mandatory",
    "recommended": "recommended",
    "optional": "optional",
    "nice_to_have": "weak intent",
}
def set_style() -> None:
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
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def balanced_subsample(
    rows: list[dict[str, Any]],
    per_modality: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Equal points per source modality, sampled uniformly so drift appears at its
    true within-level prevalence (preserved stays the visible majority)."""
    chosen: list[int] = []
    for modality in SOURCE_ORDER:
        idx = np.asarray([i for i, row in enumerate(rows) if str(row.get("source_modality", "")) == modality])
        if idx.size == 0:
            continue
        if idx.size > per_modality:
            idx = rng.choice(idx, size=per_modality, replace=False)
        chosen.extend(idx.tolist())
    return np.asarray(sorted(set(chosen)))


def compute_projection(embeddings: np.ndarray, method: str, random_state: int) -> np.ndarray:
    if method == "pca":
        pca = PCA(n_components=2, svd_solver="randomized", random_state=random_state)
        return np.asarray(pca.fit_transform(embeddings), dtype=float)
    prepca = PCA(n_components=min(50, embeddings.shape[1]), svd_solver="randomized", random_state=random_state)
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
    "mandatory": "#0072B2",     # blue
    "recommended": "#009E73",   # bluish green
    "optional": "#E69F00",      # orange
    "nice_to_have": "#CC79A7",  # reddish purple
}
DRIFT_PALETTE = {"clean": "#9aa3af", "broad_text_oc": "#E69F00", "strict_text_oc": "#C1121F"}
# per-class (size, alpha, edge width, zorder) for the drift panel
DRIFT_STYLE = {
    "clean": (9.0, 0.70, 0.0, 1),
    "broad_text_oc": (10.0, 0.92, 0.2, 3),
    "strict_text_oc": (11.0, 0.92, 0.3, 4),
}
SOURCE_LEGEND_ORDER = ["mandatory", "recommended", "optional", "nice_to_have"]


def panel_projection(ax, coords, rows, color_field: str, title: str, rng: np.random.Generator | None = None) -> None:
    if color_field == "source_modality":
        keys = np.asarray([str(row.get("source_modality", "")) for row in rows], dtype=object)
        # single scatter in shuffled order so no input level systematically occludes another
        colors = np.asarray([SRC_PALETTE.get(k, "#94a3b8") for k in keys], dtype=object)
        order = np.arange(len(rows))
        if rng is not None:
            rng.shuffle(order)
        ax.scatter(coords[order, 0], coords[order, 1], s=9.5, c=list(colors[order]),
                   alpha=0.72, linewidths=0.0, zorder=2, rasterized=True)
        handles = [
            Line2D([], [], marker="o", linestyle="none", markersize=4.5,
                   markerfacecolor=SRC_PALETTE[name], markeredgecolor="none", label=SOURCE_LABEL[name])
            for name in SOURCE_LEGEND_ORDER
        ]
        ax.legend(handles=handles, loc="upper right", handletextpad=0.2, borderpad=0.25,
                  labelspacing=0.25, framealpha=0.85)
    else:
        keys = np.asarray([drift_status(row) for row in rows], dtype=object)
        label_map = {"clean": "preserved", "broad_text_oc": "strengthened (implied)", "strict_text_oc": "strengthened (explicit word)"}
        for label in ["clean", "broad_text_oc", "strict_text_oc"]:  # rare classes last (on top)
            mask = keys == label
            if not mask.any():
                continue
            size, alpha, edge_w, zorder = DRIFT_STYLE[label]
            ax.scatter(
                coords[mask, 0], coords[mask, 1], s=size, c=DRIFT_PALETTE[label], alpha=alpha,
                linewidths=edge_w, edgecolors="#0f172a" if edge_w > 0 else "none",
                zorder=zorder, label=label_map[label], rasterized=True,
            )
        leg = ax.legend(loc="upper right", handletextpad=0.2, borderpad=0.25, labelspacing=0.25, framealpha=0.85)
        for handle in leg.legend_handles:
            handle.set_alpha(1.0)
            handle.set_sizes([16])
    ax.set_title(title, fontsize=12.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.02)
    for spine in ax.spines.values():  # light frame so (a) and (b) read as separate panels
        spine.set_visible(True)
        spine.set_edgecolor("#cbd5e1")
        spine.set_linewidth(0.8)


def auroc_cell(summary, *, backend, text, group, scope, target, model="hgb") -> float:
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


def style_detection_axis(ax) -> None:
    """Shared 'detection score' x-axis: 0.5 = coin flip, 1.0 = perfect."""
    ax.set_xlim(0.5, 1.02)
    ax.axvline(0.5, ls="--", lw=0.9, color="#0f172a", zorder=0)
    ax.set_xticks([0.5, 0.75, 1.0])
    ax.set_xticklabels(["0.5\ncoin flip", "0.75", "1.0\nperfect"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def hbar_detection(ax, labels, values, colors) -> np.ndarray:
    y = np.arange(len(labels))[::-1]
    for yi, val, color in zip(y, values, colors, strict=True):
        if np.isnan(val):
            continue
        ax.barh(yi, val - 0.5, left=0.5, color=color, height=0.52, zorder=2)
        ax.text(val + 0.012, yi, f"{val:.2f}", va="center", ha="left", fontsize=11.5, color="#0f172a")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11.0)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    return y


def panel_readout(ax, summary, group_mode: str, model: str) -> None:
    """(c) What can be read off the generated text, by gradient-boosted trees."""
    def within_mean(backend: str) -> float:
        return float(np.nanmean([
            auroc_cell(summary, backend=backend, text="reqonly", group=group_mode,
                       scope=f"source_modality={lvl}", target="deterministic_strict_text_overcommit", model=model)
            for lvl in ("nice_to_have", "optional", "recommended")
        ]))

    src = auroc_cell(summary, backend="mlx", text="reqonly", group=group_mode,
                     scope="global", target="source_modality", model=model)
    dataset = auroc_cell(summary, backend="mlx", text="reqonly", group=group_mode,
                         scope="global", target="dataset_variant", model=model)
    strengthen_emb = within_mean("mlx")
    strengthen_kw = within_mean("tfidf")

    labels = [
        "Original input strength",
        "Source dataset",
        "Strengthened?\n(neural embedding)",
        "Strengthened?\n(keyword search)",
    ]
    values = [src, dataset, strengthen_emb, strengthen_kw]
    colors = ["#009E73", "#009E73", "#D55E00", "#D55E00"]
    y = hbar_detection(ax, labels, values, colors)
    mid = (y[2] + y[3]) / 2.0
    ax.annotate("", xy=(0.5 + 0.001, y[3] - 0.33), xytext=(0.5 + 0.001, y[2] + 0.33),
                arrowprops={"arrowstyle": "-", "lw": 0.0})
    ax.text(max(strengthen_emb, strengthen_kw) + 0.05, mid, "≈ coin flip\n(neither works)",
            va="center", ha="left", fontsize=10.5, color="#D55E00", fontstyle="italic")
    style_detection_axis(ax)
    ax.set_title("(c) What can be read from a generated requirement  (HistGradientBoosting)",
                 fontsize=12.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs") / eu.ACSE_SEMANTIC_MANIFEST_FILENAME)
    parser.add_argument("--diagnostic-dir", type=Path, default=Path("outputs/embedding_diagnostic"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures/embedding_diagnostic.pdf"))
    parser.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    parser.add_argument("--per-modality", type=int, default=850)
    parser.add_argument("--group-mode", default="item")
    parser.add_argument("--model", default="hgb")
    parser.add_argument("--random-state", type=int, default=20260527)
    args = parser.parse_args()

    root = eu.project_root()
    diagnostic_dir = args.diagnostic_dir if args.diagnostic_dir.is_absolute() else root / args.diagnostic_dir
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_path = args.output if args.output.is_absolute() else root / args.output

    cache = np.load(diagnostic_dir / "task2_reqonly_mlx_embeddings.npz", allow_pickle=False)
    reqonly = cache["embeddings"].astype(np.float32, copy=False)
    rows = manifest_rows(manifest_path, "mlx:")
    _, sample_rows = load_embeddings_and_rows(rows)
    if len(sample_rows) != reqonly.shape[0]:
        raise ValueError(f"row/embedding mismatch: {len(sample_rows)} vs {reqonly.shape[0]}")
    summary = read_summary(diagnostic_dir / "probe_grid_summary.csv")

    # Project distinct generated requirements (each unique text appears ~17x across
    # seeds); plotting one point per unique string avoids stacking identical points.
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

    rng = np.random.default_rng(args.random_state)
    keep_local = balanced_subsample(unique_rows, args.per_modality, rng)
    global_idx = unique_global_arr[keep_local]
    sub_rows = [unique_rows[k] for k in keep_local]
    print(f"projection subsample: {global_idx.size} points "
          f"(strict positives kept: {sum(drift_status(r) == 'strict_text_oc' for r in sub_rows)})")
    coords = compute_projection(reqonly[global_idx], args.method, args.random_state)

    set_style()
    fig = plt.figure(figsize=(8.0, 5.7), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.015, h_pad=0.015, wspace=0.05, hspace=0.05)
    gs = fig.add_gridspec(2, 12, height_ratios=[1.62, 0.68])
    ax_a = fig.add_subplot(gs[0, 0:6])
    ax_b = fig.add_subplot(gs[0, 6:12])
    ax_c = fig.add_subplot(gs[1, 1:11])

    panel_projection(ax_a, coords, sub_rows, "source_modality",
                     "(a) Colored by input strength", rng=rng)
    panel_projection(ax_b, coords, sub_rows, "drift",
                     "(b) Colored by strength increase")
    panel_readout(ax_c, summary, args.group_mode, args.model)
    fig.suptitle("Generated requirements", fontsize=14.0, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {output_path} and {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
