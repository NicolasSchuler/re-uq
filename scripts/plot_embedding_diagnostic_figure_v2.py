"""Paper figure (v2) for the embedding modal-force-drift diagnostic.

A SINGLE horizontal bar chart, replacing the earlier three-panel (t-SNE a/b +
bar c) figure. Every bar is a held-out **AUROC** (0.5 = chance, 1.0 = perfect),
the one metric named on the x-axis. Multi-class probes (input level, dataset)
use macro-averaged one-vs-rest AUROC; the strengthening probes are binary AUROC.

Two groups:
  Context (not the target) -- what the embedding trivially encodes:
    * Input commitment level   (source_modality, macro-OvR)
    * Source dataset           (dataset_variant, macro-OvR)
  Target: was the text strengthened? -- what we actually want to detect:
    * Global classifier            (all inputs pooled)   == paper Table 1
    * Within recommended-only inputs                     == paper Table 1
    * Within optional-only inputs                        == paper Table 1

Reading the plot: input commitment level and source dataset are easy to recover;
strengthening is close to chance within a single input level, so the global
strengthening signal partly reflects the input condition (input-level leakage).

Numbers are pulled from ``outputs/embedding_diagnostic/probe_grid_summary.csv``
(written by ``diagnose_embedding_separability.py``) -- nothing is hardcoded. The
three strengthening bars are asserted to match the paper's Table 1 exactly.

The two t-SNE projections are no longer part of the paper figure; they are
exported as a separate supplementary figure by
``plot_embedding_diagnostic_tsne_supp.py`` for the replication package.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "re_uq_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import eval_utils as eu
    from plot_embedding_diagnostic_figure import as_float, auroc_cell, read_summary
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu
    from scripts.plot_embedding_diagnostic_figure import (
        auroc_cell,
        read_summary,
    )

# --- Bar specification -------------------------------------------------------
# Each bar names exactly one row of probe_grid_summary.csv, so provenance is
# auditable and nothing is hardcoded. ``decimals`` controls the value label
# (3 for the Table-1 numbers so they read as exact matches).
#
# Context bars use the de-circularized requirement-text-only, item-grouped
# substrate (the label is NOT baked into the embedded string). The prefixed
# source_modality probe scores 0.96 only because the modality word is literally
# prepended to the text, so we deliberately report the honest 0.84/0.73 instead.
CONTEXT_BARS = [
    dict(label="Input commitment level", decimals=2,
         backend="mlx", text="reqonly", group="item", scope="global",
         target="source_modality"),
    dict(label="Source dataset", decimals=2,
         backend="mlx", text="reqonly", group="item", scope="global",
         target="dataset_variant"),
]

# Strengthening bars come from the exact configuration the paper's Table 1
# reports: neural embedding, label-prefixed string, seed-grouped CV, one-shot
# (deterministic) strengthening target. These must match Table 1 to the digit.
TARGET_BARS = [
    dict(label="Global classifier\n(all inputs pooled)", decimals=3, expect=0.822,
         backend="mlx", text="prefixed", group="seed", scope="global",
         target="deterministic_strict_text_overcommit"),
    dict(label="Within recommended-only inputs", decimals=3, expect=0.519,
         backend="mlx", text="prefixed", group="seed",
         scope="source_modality=recommended",
         target="deterministic_strict_text_overcommit"),
    dict(label="Within optional-only inputs", decimals=3, expect=0.582,
         backend="mlx", text="prefixed", group="seed",
         scope="source_modality=optional",
         target="deterministic_strict_text_overcommit"),
    # Weak-intent-only is a looser test (there "strengthened" nearly coincides
    # with "contains any modal word"); shown so the within-level check is not
    # cherry-picked to two strata. Mandatory-only is undefined (no stronger level).
    dict(label="Within weak-intent inputs", decimals=3, expect=0.600,
         backend="mlx", text="prefixed", group="seed",
         scope="source_modality=nice_to_have",
         target="deterministic_strict_text_overcommit"),
]

# Colour-blind-safe: neutral slate-blue for context, one orange accent for the
# target group. Blue vs orange is the safest deutan/protan-distinguishable pair.
CONTEXT_COLOR = "#5B7FA6"   # muted slate blue
TARGET_COLOR = "#E69F00"    # Okabe-Ito orange
INK = "#1f2937"             # dark slate for text
MUTED = "#475569"           # secondary text

CHANCE = 0.5


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "figure.dpi": 200,
        }
    )


def resolve_bars(summary, specs) -> list[dict]:
    out = []
    for spec in specs:
        value = auroc_cell(
            summary,
            backend=spec["backend"], text=spec["text"], group=spec["group"],
            scope=spec["scope"], target=spec["target"], model="hgb",
        )
        if np.isnan(value):
            raise ValueError(f"no probe_grid row for {spec['label']!r}: {spec}")
        expect = spec.get("expect")
        if expect is not None and round(value, 3) != round(expect, 3):
            raise ValueError(
                f"Table-1 mismatch for {spec['label']!r}: got {value:.4f}, "
                f"expected {expect:.3f}. The figure must not drift from Table 1."
            )
        out.append({**spec, "value": float(value)})
    return out


def draw(context, target, output_path: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 2.95))

    # y positions: context group on top, a gap, then the target group.
    gap = 0.9
    y_ctx = [len(context) + len(target) + gap - i for i in range(len(context))]
    y_tgt = [len(target) - i for i in range(len(target))]
    all_bars = [
        *zip(context, y_ctx, [CONTEXT_COLOR] * len(context), strict=True),
        *zip(target, y_tgt, [TARGET_COLOR] * len(target), strict=True),
    ]

    for bar, y, color in all_bars:
        val = bar["value"]
        ax.barh(y, val - CHANCE, left=CHANCE, height=0.62, color=color,
                edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(val + 0.008, y, f"{val:.{bar['decimals']}f}", va="center",
                ha="left", fontsize=8.5, color=INK, zorder=4)

    labels = [b["label"] for b, _, _ in all_bars]
    ys = [y for _, y, _ in all_bars]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)

    # Dashed "chance" reference line at 0.5, doubling as the left boundary; the
    # 0.5 tick carries its "chance" label so nothing collides with the axis text.
    ax.axvline(CHANCE, ls=(0, (4, 3)), lw=1.0, color=INK, zorder=2)
    ax.set_xlim(CHANCE, 1.0)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 1.05)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["0.5\nchance", "0.6", "0.7", "0.8", "0.9", "1.0"])
    ax.set_xlabel("AUROC  (0.5 = chance,  1.0 = perfect)", labelpad=4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3)

    # Group headers above each group.
    ax.text(CHANCE, max(y_ctx) + 0.62, "Context (not the target)",
            fontsize=9.0, fontweight="bold", color=MUTED, va="bottom", ha="left")
    ax.text(CHANCE, max(y_tgt) + 0.62, "Target: was the text strengthened?",
            fontsize=9.0, fontweight="bold", color="#B26F00", va="bottom", ha="left")

    # Annotation on the global-classifier bar: it looks usable only because it
    # can read the input level off the representation. Placed just right of the
    # value label (no connector) so nothing overlaps the "0.822".
    g_bar, g_y, _ = next(t for t in all_bars if t[0] is target[0])
    ax.text(g_bar["value"] + 0.065, g_y, "inflated by\ninput-level leakage",
            fontsize=7.8, color="#B26F00", fontstyle="italic", va="center",
            ha="left", linespacing=1.1, clip_on=False)

    # Bracket spanning every within-level bar: "~ chance within one input level".
    y_hi, y_lo = y_tgt[1], y_tgt[-1]
    xb = 0.655
    ax.plot([xb, xb], [y_lo, y_hi], color=MUTED, lw=0.9, zorder=4)
    for yy in (y_lo, y_hi):
        ax.plot([xb - 0.012, xb], [yy, yy], color=MUTED, lw=0.9, zorder=4)
    ax.text(xb + 0.02, (y_hi + y_lo) / 2.0, "≈ chance within\none input level",
            fontsize=7.8, color=MUTED, fontstyle="italic", va="center", ha="left")

    fig.tight_layout(pad=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path,
                        default=Path("outputs/embedding_diagnostic"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/figures/embedding_diagnostic.pdf"))
    args = parser.parse_args()

    root = eu.project_root()
    diagnostic_dir = args.diagnostic_dir if args.diagnostic_dir.is_absolute() else root / args.diagnostic_dir
    output_path = args.output if args.output.is_absolute() else root / args.output

    summary = read_summary(diagnostic_dir / "probe_grid_summary.csv")
    context = resolve_bars(summary, CONTEXT_BARS)
    target = resolve_bars(summary, TARGET_BARS)

    print("Bars pulled from probe_grid_summary.csv (metric = AUROC):")
    for b in context + target:
        note = f"  == Table 1 ({b['expect']:.3f})" if b.get("expect") else "  (diagnostic substrate)"
        flat = b["label"].replace("\n", " ")
        print(f"  {flat:<34s} {b['value']:.3f}   [{b['backend']}/{b['text']}/{b['group']}"
              f"/{b['scope']}/{b['target']}]{note}")

    draw(context, target, output_path)
    print(f"wrote {output_path} and {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
