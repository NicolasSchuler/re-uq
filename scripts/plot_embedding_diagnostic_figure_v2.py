"""Horizontal comparison of predictions from generated-requirement embeddings.

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
    fig, ax = plt.subplots(figsize=(9.0, 4.7))

    # y positions: context on top, then the target group, then the additional-input comparison.
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
        if not np.isfinite(val):
            ax.text(
                0.03,
                y,
                "×  unavailable: " + bar["unavailable_reason"],
                va="center",
                fontsize=8,
                color=MUTED,
            )
            continue
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
        low, high = bar.get("ci_low", np.nan), bar.get("ci_high", np.nan)
        if np.isfinite(low) and np.isfinite(high):
            ax.plot([low, high], [y, y], color=INK, lw=1.2, zorder=5)
            ax.plot([low, high], [y, y], "|", color=INK, ms=7, zorder=5)
        suffix = " †" if bar.get("fit_review_required") else ""
        if not np.isfinite(low):
            suffix += " (CI unavailable)"
        ax.text(
            max(val, high if np.isfinite(high) else val) + 0.015,
            y,
            f"{val:.{bar['decimals']}f}{suffix}",
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
    ax.set_xlim(0.0, 1.3)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 1.05)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0.0", "0.25", "0.5\nchance", "0.75", "1.0"])
    ax.set_xlabel(
        "AUROC with 95% capability-bootstrap intervals (fixed held-out predictions)",
        labelpad=4,
    )
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3)

    # Group headers above each group.
    ax.text(
        CHANCE,
        max(y_ctx) + 0.62,
        "Auxiliary prediction targets",
        fontsize=9.0,
        fontweight="bold",
        color=MUTED,
        va="bottom",
        ha="left",
    )
    ax.text(
        CHANCE,
        max(y_tgt) + 0.62,
        "Strict strengthening in separate single-pass output",
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
            "Additional input: declared modality",
            fontsize=9.0,
            fontweight="bold",
            color=MUTED,
            va="bottom",
            ha="left",
        )

    if any(b.get("fit_review_required") for b, _, _ in all_bars):
        fig.text(
            0.5,
            0.005,
            "† Fitting budget needs review; weak scores are not evidence against the method.",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout(pad=0.6, rect=(0, 0.04, 1, 1))
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
        flat = b["label"].replace("\n", " ")
        print(
            f"  {flat:<44s} {b['value']:.3f}   [{b['backend']}/{b['text']}/{b['group']}"
            f"/{b['scope']}/{b['target']}]"
        )

    draw(context, target, control, output_path)
    print(f"wrote {output_path} and {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
