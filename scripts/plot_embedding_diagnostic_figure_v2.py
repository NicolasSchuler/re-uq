"""Paper figure (v2) for the embedding modal-force-drift diagnostic.

A SINGLE horizontal bar chart, replacing the earlier three-panel (t-SNE a/b +
bar c) figure. Every bar is a held-out **AUROC** (0.5 = chance, 1.0 = perfect),
the one metric named on the x-axis. Multi-class probes (input level, dataset)
use macro-averaged one-vs-rest AUROC; the strengthening probes are binary AUROC.

Three groups:
  Context (not the target) -- what the embedding trivially encodes:
    * Input commitment level   (source_modality, macro-OvR)
    * Source dataset           (dataset_variant, macro-OvR)
  Target: was the text strengthened? -- what we actually want to detect:
    * Global classifier            (all inputs pooled)
    * Within recommended-only inputs
    * Within optional-only inputs
    * Within weak-intent inputs
  Positive control -- the same probe with the answer inside the input.

Every bar in the first two groups reads the **requirement-only** substrate: the
embedded string is the generated requirement alone. The alternative ``prefixed``
substrate begins ``modality: <predicted label>``, and the strengthening label is
derived partly from that predicted label, so a probe on it is measuring its own
input. That configuration is kept as one explicitly labelled **leakage control**
rather than as a result.

Reading the plot: input commitment level and source dataset are easy to recover;
strengthening is weak within a single input level, so the pooled strengthening
number partly reflects the input condition (input-level leakage). The control bar
shows how much higher the same probe scores once the predicted label is written
into the text.

Numbers are pulled from ``outputs/embedding_diagnostic/probe_grid_summary.csv``
(written by ``diagnose_embedding_separability.py``) -- nothing is hardcoded. Every
bar is pinned to the value in that file, so a regenerated grid cannot silently
move the figure.

The two t-SNE projections are no longer part of the paper figure; they are
exported as a separate supplementary figure by
``plot_embedding_diagnostic_tsne_supp.py`` for the replication package.
"""

from __future__ import annotations

import argparse
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

try:
    import eval_utils as eu
    from plot_embedding_diagnostic_figure import auroc_cell, read_summary
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
# The ``expect`` values below date from the grid that fitted PCA and the TF-IDF
# vocabulary over all rows at once; that leak is fixed in
# diagnose_embedding_separability.py, so they must be re-pinned from the rerun's
# probe_grid_summary.csv and resolve_bars() fails loudly until they are. The
# shift is small: measured on the archived prefixed embeddings (86,305 rows,
# 1024 dims, 128 components, 3 folds), the control bar moves 0.822 -> 0.827,
# because axes chosen from two thirds of that many rows barely differ from axes
# chosen from all of them. The fix is about the claim the probe makes, not about
# a number that was badly wrong.
#
# Context bars use the de-circularized requirement-text-only, item-grouped
# substrate (the label is NOT baked into the embedded string). The prefixed
# source_modality probe scores 0.96 only because the modality word is literally
# prepended to the text, so we deliberately report the honest 0.84/0.73 instead.
CONTEXT_BARS = [
    {
        "label": "Input commitment level",
        "decimals": 2,
        "backend": "mlx",
        "text": "reqonly",
        "group": "item",
        "scope": "global",
        "target": "source_modality",
    },
    {
        "label": "Source dataset",
        "decimals": 2,
        "backend": "mlx",
        "text": "reqonly",
        "group": "item",
        "scope": "global",
        "target": "dataset_variant",
    },
]

# Strengthening bars: neural embedding, requirement-only string, seed-grouped CV,
# one-shot (deterministic) strengthening target. Requirement-only is the primary
# substrate because the label-prefixed string encodes an ingredient of the target.
# ``expect`` pins each bar to its row in probe_grid_summary.csv.
TARGET_BARS = [
    {
        "label": "Global classifier\n(all inputs pooled)",
        "decimals": 3,
        "expect": 0.707,
        "backend": "mlx",
        "text": "reqonly",
        "group": "seed",
        "scope": "global",
        "target": "deterministic_strict_text_overcommit",
    },
    {
        "label": "Within recommended-only inputs",
        "decimals": 3,
        "expect": 0.613,
        "backend": "mlx",
        "text": "reqonly",
        "group": "seed",
        "scope": "source_modality=recommended",
        "target": "deterministic_strict_text_overcommit",
    },
    {
        "label": "Within optional-only inputs",
        "decimals": 3,
        "expect": 0.612,
        "backend": "mlx",
        "text": "reqonly",
        "group": "seed",
        "scope": "source_modality=optional",
        "target": "deterministic_strict_text_overcommit",
    },
    # Weak-intent-only is a looser test (there "strengthened" nearly coincides
    # with "contains any modal word"); shown so the within-level check is not
    # cherry-picked to two strata. Mandatory-only is undefined (no stronger level).
    {
        "label": "Within weak-intent inputs",
        "decimals": 3,
        "expect": 0.620,
        "backend": "mlx",
        "text": "reqonly",
        "group": "seed",
        "scope": "source_modality=nice_to_have",
        "target": "deterministic_strict_text_overcommit",
    },
]

# The one place the prefixed substrate appears. Its text literally starts
# "modality: <predicted label>", and strengthening is derived partly from that
# label, so this bar measures label leakage, not what the wording reveals. It is
# drawn apart from the target group and says so in its own tick label.
CONTROL_BARS = [
    {
        "label": "Same probe on label-prefixed text\n(leakage control)",
        "decimals": 3,
        "expect": 0.822,
        "backend": "mlx",
        "text": "prefixed",
        "group": "seed",
        "scope": "global",
        "target": "deterministic_strict_text_overcommit",
    },
]

# Colour-blind-safe: neutral slate-blue for context, one orange accent for the
# target group. Blue vs orange is the safest deutan/protan-distinguishable pair.
# The control is deliberately drab and hatched so it never reads as a result.
CONTEXT_COLOR = "#5B7FA6"  # muted slate blue
TARGET_COLOR = "#E69F00"  # Okabe-Ito orange
CONTROL_COLOR = "#B8BEC7"  # desaturated grey
INK = "#1f2937"  # dark slate for text
MUTED = "#475569"  # secondary text

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


def resolve_bars(
    summary: list[dict[str, Any]], specs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    out = []
    for spec in specs:
        value = auroc_cell(
            summary,
            backend=spec["backend"],
            text=spec["text"],
            group=spec["group"],
            scope=spec["scope"],
            target=spec["target"],
            model="hgb",
        )
        if np.isnan(value):
            raise ValueError(f"no probe_grid row for {spec['label']!r}: {spec}")
        expect = spec.get("expect")
        if expect is not None and round(value, 3) != round(expect, 3):
            raise ValueError(
                f"pinned-value mismatch for {spec['label']!r}: got {value:.4f}, "
                f"expected {expect:.3f}. Update the pin (and the reported table) "
                "deliberately; the figure must not drift on its own."
            )
        out.append({**spec, "value": float(value)})
    return out


def stack_positions(group_sizes: list[int], gap: float) -> list[list[float]]:
    """Top-down y positions per group, stacked bottom-up with a gap between groups."""
    positions: list[list[float]] = []
    top = 0.0
    for size in reversed(group_sizes):
        positions.insert(0, [top + size - index for index in range(size)])
        top += size + gap
    return positions


def draw(
    context: list[dict[str, Any]],
    target: list[dict[str, Any]],
    control: list[dict[str, Any]],
    output_path: Path,
) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 3.45))

    # y positions: context on top, then the target group, then the leakage control.
    gap = 0.9
    y_ctx, y_tgt, y_ctl = stack_positions(
        [len(context), len(target), len(control)], gap
    )
    all_bars = [
        *zip(context, y_ctx, [CONTEXT_COLOR] * len(context), strict=True),
        *zip(target, y_tgt, [TARGET_COLOR] * len(target), strict=True),
        *zip(control, y_ctl, [CONTROL_COLOR] * len(control), strict=True),
    ]

    for bar, y, color in all_bars:
        val = bar["value"]
        ax.barh(
            y,
            val - CHANCE,
            left=CHANCE,
            height=0.62,
            color=color,
            edgecolor="white",
            hatch="//" if color == CONTROL_COLOR else None,
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            val + 0.008,
            y,
            f"{val:.{bar['decimals']}f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK,
            zorder=4,
        )

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
    ax.text(
        CHANCE,
        max(y_ctx) + 0.62,
        "Context (not the target)",
        fontsize=9.0,
        fontweight="bold",
        color=MUTED,
        va="bottom",
        ha="left",
    )
    ax.text(
        CHANCE,
        max(y_tgt) + 0.62,
        "Target: was the text strengthened?  (requirement text only)",
        fontsize=9.0,
        fontweight="bold",
        color="#B26F00",
        va="bottom",
        ha="left",
    )
    if control:
        ax.text(
            CHANCE,
            max(y_ctl) + 0.62,
            "Positive control: predicted label prepended to the text",
            fontsize=9.0,
            fontweight="bold",
            color=MUTED,
            va="bottom",
            ha="left",
        )

    # Annotation on the global-classifier bar: the pooled number is higher than any
    # single-input-level number, so part of it is the input condition rather than
    # strengthening. Placed right of the value label (no connector) so nothing
    # overlaps the number.
    g_bar, g_y, _ = next(t for t in all_bars if t[0] is target[0])
    ax.text(
        g_bar["value"] + 0.065,
        g_y,
        "part input-level\nleakage",
        fontsize=7.8,
        color="#B26F00",
        fontstyle="italic",
        va="center",
        ha="left",
        linespacing=1.1,
        clip_on=False,
    )

    # Bracket spanning every within-level bar. Its x is derived from the bars it
    # spans (plus room for their value labels) so it cannot collide with a number
    # after the probe is regenerated.
    y_hi, y_lo = y_tgt[1], y_tgt[-1]
    xb = max(bar["value"] for bar in target[1:]) + 0.075
    ax.plot([xb, xb], [y_lo, y_hi], color=MUTED, lw=0.9, zorder=4)
    for yy in (y_lo, y_hi):
        ax.plot([xb - 0.012, xb], [yy, yy], color=MUTED, lw=0.9, zorder=4)
    ax.text(
        xb + 0.02,
        (y_hi + y_lo) / 2.0,
        "weak within a\nsingle input level",
        fontsize=7.8,
        color=MUTED,
        fontstyle="italic",
        va="center",
        ha="left",
    )

    fig.tight_layout(pad=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-dir", type=Path, default=Path("outputs/embedding_diagnostic")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/figures/embedding_diagnostic.pdf")
    )
    args = parser.parse_args()

    root = eu.project_root()
    diagnostic_dir = (
        args.diagnostic_dir
        if args.diagnostic_dir.is_absolute()
        else root / args.diagnostic_dir
    )
    output_path = args.output if args.output.is_absolute() else root / args.output

    summary = read_summary(diagnostic_dir / "probe_grid_summary.csv")
    context = resolve_bars(summary, CONTEXT_BARS)
    target = resolve_bars(summary, TARGET_BARS)
    control = resolve_bars(summary, CONTROL_BARS)

    print("Bars pulled from probe_grid_summary.csv (metric = AUROC):")
    for b in context + target + control:
        note = f"  == pinned ({b['expect']:.3f})" if b.get("expect") else ""
        flat = b["label"].replace("\n", " ")
        print(
            f"  {flat:<44s} {b['value']:.3f}   [{b['backend']}/{b['text']}/{b['group']}"
            f"/{b['scope']}/{b['target']}]{note}"
        )

    draw(context, target, control, output_path)
    print(f"wrote {output_path} and {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
