"""Separate prediction tasks using generated-requirement embeddings.

The primary input is sampled requirement text; the primary target is strict
strengthening of the separate single-pass output for the same source.
Requirement text with the sampled declared modality prepended is an additional
input comparison. It does not, by itself, establish target leakage.

Auxiliary probes predict source modality and dataset-by-keyword origin. All
specifications below use capability grouping; different prediction targets
remain separate diagnostics. Scores come from the current summary, never from
historical constants. Points summarize held-out folds and intervals describe
capability-bootstrap uncertainty conditional on the fitted classifiers and
splits (see ``docs/figures/embedding_diagnostic.md``).

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
        "label": "Source commitment level",
        "decimals": 3,
        "backend": "mlx",
        "text": "reqonly",
        "group": "seed",
        "scope": "global",
        "target": "source_modality",
    },
    {
        "label": "Source dataset × keyword variant",
        "decimals": 3,
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

# Colour-blind-safe hues, darkened for readable intervals in grayscale.
CONTEXT_COLOR = "#5B7FA6"  # muted slate blue
TARGET_COLOR = "#A66B00"  # dark amber
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
                "folds": number("folds"),
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


def training_budget_note(diagnostic_dir: Path, bars: list[dict[str, Any]]) -> str:
    """Only claim a common limit when all selected folds document reaching it."""
    path = diagnostic_dir / "probe_grid_folds.csv"
    fallback = "Fitting diagnostics require review. See the training-budget qualification in the text."
    if not path.exists():
        return fallback if any(bar["fit_review_required"] for bar in bars) else ""
    folds = read_summary(path)
    selected = []
    for bar in bars:
        matches = [
            row
            for row in folds
            if row.get("model") == "hgb"
            and all(
                row.get(field) == bar[key]
                for field, key in (
                    ("feature_backend", "backend"),
                    ("text_variant", "text"),
                    ("group_mode", "group"),
                    ("scope", "scope"),
                    ("target", "target"),
                )
            )
        ]
        if not matches or len(matches) != bar["folds"]:
            return fallback if any(bar["fit_review_required"] for bar in bars) else ""
        selected.extend(matches)
    limits = {row.get("max_iter", "") for row in selected}
    if len(limits) == 1 and all(
        row.get("iteration_limit_reached", "").lower() == "true"
        and row.get("n_iter") == row.get("max_iter")
        for row in selected
    ):
        limit = next(iter(limits))
        if limit.isdigit() and int(limit) > 0:
            return f"All displayed fits reached the {limit}-step training limit."
    return fallback if any(bar["fit_review_required"] for bar in bars) else ""


def draw(
    context: list[dict[str, Any]],
    target: list[dict[str, Any]],
    control: list[dict[str, Any]],
    output_path: Path,
) -> None:
    set_style()
    fig = plt.figure(figsize=(7.0, 3.9))
    # Keep the graphic focused on the three findings.
    axes = [
        fig.add_axes((0.32, 0.71, 0.66, 0.17)),
        fig.add_axes((0.32, 0.375, 0.66, 0.23)),
        fig.add_axes((0.32, 0.085, 0.66, 0.15)),
    ]
    panels = [
        (
            axes[0],
            [target[0], *control],
            ["Text only", "Text + declared label"],
            "A  Stronger detection across mixed source modalities",
            0.98,
            TARGET_COLOR,
        ),
        (
            axes[1],
            target[1:],
            ["Recommended sources", "Optional sources", "Weak-intent sources"],
            "B  Weaker detection within source modalities",
            0.65,
            TARGET_COLOR,
        ),
        (
            axes[2],
            context,
            ["Source modality", "Dataset × keyword"],
            "C  Source information is also recoverable",
            0.305,
            CONTEXT_COLOR,
        ),
    ]
    missing_intervals = False
    collapsed_intervals = False
    for (
        panel_ax,
        bars,
        labels,
        heading,
        heading_y,
        color,
    ) in panels:
        fig.text(0.015, heading_y, heading, va="top", fontsize=10, fontweight="bold")
        for y, bar in enumerate(reversed(bars)):
            val = bar["value"]
            low, high = bar.get("ci_low", np.nan), bar.get("ci_high", np.nan)
            if not np.isfinite(val):
                panel_ax.text(0.3, y, "Unavailable", va="center", fontsize=9)
                continue
            suffix = ""
            if np.isfinite(low) and np.isfinite(high):
                panel_ax.plot([low, high], [y, y], color=color, lw=1.0)
                panel_ax.plot([low, high], [y, y], "|", color=color, ms=5)
                if low == high:
                    suffix = "‡"
                    collapsed_intervals = True
            else:
                suffix = "†"
                missing_intervals = True
            panel_ax.plot(val, y, "o", color=color, mfc="none", ms=3, mew=1, zorder=3)
            panel_ax.text(1.035, y, f"{val:.3f}{suffix}", va="center", fontsize=9)
        panel_ax.set_yticks(range(len(bars)))
        panel_ax.set_yticklabels(list(reversed(labels)), fontsize=9)
        panel_ax.set_ylim(-0.55, len(bars) - 0.45)
        panel_ax.axvline(CHANCE, ls=(0, (4, 3)), lw=0.8, color=MUTED)
        panel_ax.set_xlim(0.0, 1.16)
        panel_ax.spines["bottom"].set_bounds(0.0, 1.0)
        panel_ax.set_xticks([0.0, 0.5, 0.75, 1.0])
        panel_ax.set_xticklabels(["0.0", "0.5", "0.75", "1.0"], fontsize=8.5)
        panel_ax.spines["left"].set_visible(False)
        panel_ax.tick_params(axis="y", length=0, pad=7)
    fig.text(0.645, 0.005, "AUROC · 95% CI · chance = 0.5", ha="center", fontsize=9)
    notes = []
    if missing_intervals:
        notes.append("† CI unavailable")
    if collapsed_intervals:
        notes.append("‡ Collapsed CI")
    if notes:
        fig.text(0.015, 0.005, " · ".join(notes), fontsize=8.5, color=INK)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # No CreationDate: the tracked PDF must not change on a byte-identical rerun.
    fig.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0.06,
        metadata={"CreationDate": None},
    )
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
        "--output",
        type=Path,
        default=Path("outputs/rerun/figures/embedding_diagnostic.pdf"),
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

    print(training_budget_note(diagnostic_dir, context + target + control))
    for bar in context + target + control:
        if bar["ci_unavailable_reason"]:
            print(f"  {bar['label']}: {bar['ci_unavailable_reason']}")
    draw(context, target, control, output_path)
    print(f"wrote {output_path} and {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
