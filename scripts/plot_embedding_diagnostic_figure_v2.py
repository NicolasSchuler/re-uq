"""Separate prediction tasks using generated-requirement embeddings.

The primary input is sampled requirement text; the primary target is strict
strengthening of the separate single-pass output for the same source.
Requirement text with the sampled declared modality prepended is an additional
input comparison. It does not, by itself, establish target leakage.

Auxiliary probes predict source modality and dataset-by-keyword origin. All
specifications below use capability grouping; different prediction targets
remain separate diagnostics. Scores come from the current summary, never from
historical constants. The manuscript figure stays pending until held-out
predictions and 95% intervals are exported and incorporated (see
``docs/figures/embedding_diagnostic.md``).

The exploratory t-SNE projections are retained separately in the replication
package by ``plot_embedding_diagnostic_tsne_supp.py``.
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
    from plot_embedding_diagnostic_figure import read_summary
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu
    from scripts.plot_embedding_diagnostic_figure import (
        read_summary,
    )

# Each specification selects one prediction target and input representation.
# Capability grouping prevents variants of a held-out capability from entering
# training. Comparable representations use the same target and grouping.
CONTEXT_BARS = [
    {
        "label": "Source modality",
        "decimals": 2,
        "backend": "mlx",
        "text": "reqonly",
        "group": "seed",
        "scope": "global",
        "target": "source_modality",
    },
    {
        "label": "Source dataset × keyword variant",
        "decimals": 2,
        "backend": "mlx",
        "text": "reqonly",
        "group": "seed",
        "scope": "global",
        "target": "dataset_variant",
    },
]

# Strengthening bars: neural embedding, requirement-only string, seed-grouped CV,
# one-shot (deterministic) strengthening target. Requirement-only is the primary
# input, with the additional declared label evaluated separately.
TARGET_BARS = [
    {
        "label": "Global classifier\n(all sources)",
        "decimals": 3,
        "backend": "mlx",
        "text": "reqonly",
        "group": "seed",
        "scope": "global",
        "target": "deterministic_strict_text_overcommit",
    },
    {
        "label": "Within recommended-only inputs",
        "decimals": 3,
        "backend": "mlx",
        "text": "reqonly",
        "group": "seed",
        "scope": "source_modality=recommended",
        "target": "deterministic_strict_text_overcommit",
    },
    {
        "label": "Within optional-only inputs",
        "decimals": 3,
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
        "backend": "mlx",
        "text": "reqonly",
        "group": "seed",
        "scope": "source_modality=nice_to_have",
        "target": "deterministic_strict_text_overcommit",
    },
]

# Additional input comparison: the declared modality accompanies the wording.
CONTROL_BARS = [
    {
        "label": "Requirement text with declared modality\n(additional input)",
        "decimals": 3,
        "backend": "mlx",
        "text": "prefixed",
        "group": "seed",
        "scope": "global",
        "target": "deterministic_strict_text_overcommit",
    },
]

# Colour-blind-safe: neutral slate-blue for context, one orange accent for the
# target group. Blue vs orange is the safest deutan/protan-distinguishable pair.
# Hatching distinguishes the additional-input comparison from text alone.
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
        matches = [
            row
            for row in summary
            if all(
                str(row.get(field, "")) == spec[key]
                for field, key in (
                    ("feature_backend", "backend"),
                    ("text_variant", "text"),
                    ("group_mode", "group"),
                    ("scope", "scope"),
                    ("target", "target"),
                )
            )
            and row.get("model", "hgb") == "hgb"
        ]
        if len(matches) > 1:
            raise ValueError(f"ambiguous probe summary for {spec['label']}")
        if not matches:
            # A row the grid never produced is a drifted key (renamed backend,
            # text variant, scope or target), not an unavailable score: the
            # figure must not render a silent gap for a quantity the paper
            # reports from the same artifact.
            raise ValueError(
                f"no probe_grid_summary.csv row for {spec['label']!r} "
                f"({spec['backend']}/{spec['text']}/{spec['group']}"
                f"/{spec['scope']}/{spec['target']}, model=hgb)"
            )
        row = matches[0]

        def number(key, selected=row):
            value = selected.get(key, "")
            return float(value) if value not in ("", None) else float("nan")

        out.append(
            {
                **spec,
                "value": number("auroc_mean"),
                "ci_low": number("auroc_ci_low"),
                "ci_high": number("auroc_ci_high"),
                "unavailable_reason": row.get("unavailable_reason", ""),
                "ci_unavailable_reason": row.get(
                    "ci_unavailable_reason", "interval not exported"
                ),
                "fit_review_required": str(row.get("fit_review_required", "")).lower()
                == "true",
            }
        )
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
    fig, axes = plt.subplots(
        3, 1, figsize=(7.0, 4.5), sharex=True, gridspec_kw={"height_ratios": [2, 3, 2]}
    )
    fig.subplots_adjust(left=0.29, right=0.98, top=0.93, bottom=0.18, hspace=0.66)

    panels = [
        (
            axes[0],
            [target[0], *control],
            ["Text only", "Text + declared modality"],
            "A  Does adding the label help detect strengthening?",
            TARGET_COLOR,
        ),
        (
            axes[1],
            target[1:],
            ["Recommended only", "Optional only", "Weak intent only"],
            "B  Can text alone detect strengthening in each source group?",
            TARGET_COLOR,
        ),
        (
            axes[2],
            context,
            ["Commitment level", "Dataset and keyword"],
            "C  What does text reveal about the source?",
            CONTEXT_COLOR,
        ),
    ]
    for panel_ax, bars, labels, heading, color in panels:
        panel_ax.set_title(heading, loc="left", fontsize=9.5, fontweight="bold", pad=6)
        for y, bar in enumerate(reversed(bars)):
            val = bar["value"]
            low, high = bar.get("ci_low", np.nan), bar.get("ci_high", np.nan)
            if not np.isfinite(val):
                panel_ax.text(0.3, y, "Unavailable", va="center", fontsize=10)
                continue
            panel_ax.plot(val, y, "o", color=color, ms=5, zorder=3)
            if np.isfinite(low) and np.isfinite(high):
                panel_ax.plot([low, high], [y, y], color=color, lw=1.5)
                panel_ax.plot([low, high], [y, y], "|", color=color, ms=7)
            panel_ax.text(
                1.02, y, f"{val:.{bar['decimals']}f}", va="center", fontsize=10
            )
        panel_ax.set_yticks(range(len(bars)))
        panel_ax.set_yticklabels(list(reversed(labels)), fontsize=10)
        panel_ax.set_ylim(-0.55, len(bars) - 0.45)
        panel_ax.axvline(CHANCE, ls=(0, (4, 3)), lw=0.9, color=MUTED)
        panel_ax.set_xlim(0.0, 1.16)
        panel_ax.spines["bottom"].set_bounds(0.0, 1.0)
        panel_ax.set_xticks([0.0, 0.5, 0.75, 1.0])
        panel_ax.spines["left"].set_visible(False)
        panel_ax.tick_params(axis="y", length=0)
    axes[-1].set_xticklabels(
        ["0.0", "0.5\nchance", "0.75", "1.0\nperfect ranking"], fontsize=10
    )
    fig.text(0.59, 0.025, "AUROC (higher is better)", ha="center", fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(
        output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.06
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
        flat = b["label"].replace("\n", " ")
        print(
            f"  {flat:<44s} {b['value']:.3f}   [{b['backend']}/{b['text']}/{b['group']}"
            f"/{b['scope']}/{b['target']}]"
        )

    draw(context, target, control, output_path)
    print(f"wrote {output_path} and {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
