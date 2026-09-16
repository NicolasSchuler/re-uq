"""Paired changes in strict text strengthening under the three ablations.

One figure, three stacked panels, all in percentage points of strict text
strengthening with 95% intervals and a zero-change reference:

* Request composition: each batched arm minus the single-item baseline on the
  weak-intent stratum (every model strengthens almost every weak-intent item
  under the single-item protocol, so that stratum is where a change is
  readable).
* Document context: document arm minus bare arm on rebuilt PURE capabilities,
  for all sources, weak-intent sources and recommended sources.
* Phrasing: each alternative weak-intent template minus the anchor template on
  the deterministic pass.

Values come from the comparison exports only (``compare_batching_ablation.py``,
``compare_context_ablation.py``, ``run_weak_modality_probe.py``); nothing is
typed in. Exact matched pair counts remain available in the comparison exports.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent

# Paper order: hosted cohort first, then local, as in the results tables.
MODEL_LABELS = {
    "glm-5.3": "GLM-5.3",
    "glm-5.3-flash": "GLM-5.3-Flash",
    "qwen3.6-27b": "Qwen3.6-27B",
    "qwen3.8-27b": "Qwen3.8-27B",
    "qwen3.5-9b": "Qwen3.5-9B",
    "gemma4-31b-it": "Gemma-4-31B",
    "gemma4-12b-it": "Gemma-4-12B",
    "muse-glimmer-30b": "Muse-Glimmer-30B",
    "gpt-oss-20b": "gpt-oss-20B",
}
MODEL_ORDER = list(MODEL_LABELS)

METRIC = "strict_text_strengthening"

BATCH_ARMS = [
    # arm id, label, colour key, filled marker
    ("grouped_4", "4 items, variants together", "grouped", False),
    ("grouped", "16 items, variants together", "grouped", True),
    ("shuffled_4", "4 items, variants separated", "separated", False),
    ("shuffled", "16 items, variants separated", "separated", True),
]
CONTEXT_STRATA = [
    ("all", "All sources", "all"),
    ("weak_intent", "Weak-intent sources", "weak"),
    ("modality_recommended", "Recommended sources", "recommended"),
]
PHRASING_TEMPLATES = [
    ("nice_if", "“nice if”"),
    ("low_priority_enhancement", "“low-priority enhancement”"),
    ("future_enhancement", "“possible future enhancement”"),
]

# Same palette as the embedding figure: slate blue and Okabe-Ito orange are the
# safest pair for deutan/protan vision; grey is the neutral third series.
COLORS = {
    "grouped": "#E69F00",
    "separated": "#5B7FA6",
    "all": "#475569",
    "weak": "#E69F00",
    "recommended": "#5B7FA6",
}
INK = "#1f2937"
MUTED = "#475569"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            # The figure is reduced to the manuscript's text width. Inspect
            # the compiled page as well as the standalone export for legibility.
            "font.size": 11.5,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "figure.dpi": 200,
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pp(value: str) -> float:
    return float(value) * 100.0


class Point:
    def __init__(self, row: dict[str, str], low: str, high: str, pairs: str):
        self.delta = pp(row["delta"])
        self.low = pp(row[low])
        self.high = pp(row[high])
        self.pairs = int(float(row[pairs]))


def batching_points(rows: list[dict[str, str]]) -> dict[str, dict[str, Point]]:
    out: dict[str, dict[str, Point]] = {}
    for row in rows:
        if row["metric"] != METRIC or row["stratum"] != "weak_intent":
            continue
        if row.get("baseline_arm", "single") != "single":
            raise SystemExit(f"unexpected batching baseline {row['baseline_arm']}")
        if row.get("unavailable_reason"):
            continue
        out.setdefault(row["model"], {})[row["arm"]] = Point(
            row, "delta_ci_low", "delta_ci_high", "n_complete_pairs"
        )
    return out


def context_points(rows: list[dict[str, str]]) -> dict[str, dict[str, Point]]:
    out: dict[str, dict[str, Point]] = {}
    wanted = {stratum for stratum, _, _ in CONTEXT_STRATA}
    for row in rows:
        if row["metric"] != METRIC or row["stratum"] not in wanted:
            continue
        if row.get("unavailable_reason"):
            continue
        out.setdefault(row["model"], {})[row["stratum"]] = Point(
            row, "delta_ci_low", "delta_ci_high", "n_complete_pairs"
        )
    return out


def phrasing_points(run_dirs: list[Path]) -> dict[str, dict[str, Point]]:
    out: dict[str, dict[str, Point]] = {}
    for run_dir in run_dirs:
        for row in read_rows(run_dir / "weak_modality_text_deltas.csv"):
            if row["metric"] != "strict" or row["baseline"] != "useful_if":
                continue
            out.setdefault(row["model"], {})[row["template_id"]] = Point(
                row, "ci_low", "ci_high", "n_complete_pairs"
            )
    return out


def ordered_models(points: dict[str, Any]) -> list[str]:
    known = [m for m in MODEL_ORDER if m in points]
    unknown = sorted(set(points) - set(known))
    return known + unknown


def draw_panel(
    ax: plt.Axes,
    points: dict[str, dict[str, Point]],
    series: list[tuple[str, str, str, bool]],
    title: str,
    legend_loc: str = "center right",
) -> None:
    models = ordered_models(points)
    n_series = len(series)
    step = 0.8 / max(n_series, 1)
    labels = []
    for i, model in enumerate(models):
        y0 = len(models) - 1 - i
        labels.append(MODEL_LABELS.get(model, model))
        for j, (key, _label, color_key, filled) in enumerate(series):
            point = points[model].get(key)
            if point is None:
                continue
            y = y0 + 0.4 - step * (j + 0.5)
            color = COLORS[color_key]
            ax.plot([point.low, point.high], [y, y], color=color, lw=1.3, zorder=3)
            ax.plot(
                point.delta,
                y,
                marker="o",
                ms=5.0,
                mfc=color if filled else "white",
                mec=color,
                mew=1.1,
                ls="none",
                zorder=4,
            )
    ax.axvline(0, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.set_yticks([len(models) - 1 - i for i in range(len(models))])
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.5, max(len(models), 1) - 0.5)
    ax.set_xlim(-105, 105)
    ax.set_xticks([-100, -75, -50, -25, 0, 25, 50, 75, 100])
    ax.set_xticklabels(["−100", "−75", "−50", "−25", "0", "+25", "+50", "+75", "+100"])
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", fontsize=12.0, fontweight="bold", color=MUTED)
    for y in range(1, len(models)):
        ax.axhline(y - 0.5, color="#E5E7EB", lw=0.6, zorder=1)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            ms=5.0,
            mfc=COLORS[color_key] if filled else "white",
            mec=COLORS[color_key],
            mew=1.1,
            color=COLORS[color_key],
            lw=1.3,
            label=label,
        )
        for _key, label, color_key, filled in series
    ]
    ax.legend(
        handles=handles,
        loc=legend_loc,
        fontsize=10.0,
        frameon=False,
        ncol=1,
        handlelength=1.6,
        borderaxespad=0.2,
    )


def build_figure(
    batching: dict[str, dict[str, Point]],
    context: dict[str, dict[str, Point]],
    phrasing: dict[str, dict[str, Point]],
) -> plt.Figure:
    heights = [
        max(len(points), 1) * len(series)
        for points, series in (
            (batching, BATCH_ARMS),
            (context, CONTEXT_STRATA),
            (phrasing, PHRASING_TEMPLATES),
        )
    ]
    fig, axes = plt.subplots(
        3,
        1,
        # 0.08 in per model-series row keeps the nine-model phrasing panel on one
        # manuscript page (0.105 overflowed the float by 74 pt).
        figsize=(7.0, 0.08 * sum(heights) + 1.3),
        gridspec_kw={
            "height_ratios": heights,
            "left": 0.24,
            "right": 0.99,
            "bottom": 0.07,
            "top": 0.95,
            "hspace": 0.18,
        },
    )
    draw_panel(
        axes[0],
        batching,
        [(arm, label, key, filled) for arm, label, key, filled in BATCH_ARMS],
        "Batching versus one item per request\n(weak-intent sources)",
        legend_loc="center right",
    )
    draw_panel(
        axes[1],
        context,
        [(stratum, label, key, True) for stratum, label, key in CONTEXT_STRATA],
        "With versus without document context",
        legend_loc="upper right",
    )
    draw_panel(
        axes[2],
        phrasing,
        [
            (template, label, key, True)
            for (template, label), key in zip(
                PHRASING_TEMPLATES, ("all", "weak", "recommended"), strict=True
            )
        ],
        "Alternative versus original weak-intent phrasing",
        legend_loc="center right",
    )
    # All panels use the same effect scale. Show it once to leave room for
    # panel headings without compressing the comparison rows.
    for ax in axes[:2]:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    axes[2].set_xlabel("Change in strengthening (percentage points)")
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batching",
        type=Path,
        default=REPO / "outputs/batching_ablation_summary_deltas.csv",
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=REPO / "outputs/context_ablation_summary_deltas.csv",
    )
    parser.add_argument(
        "--weak-probe-dir",
        type=Path,
        action="append",
        help="weak probe run directory holding weak_modality_text_deltas.csv (repeatable)",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=REPO / "outputs/rerun/manuscript-final/state.json",
        help="rerun state file used to locate the weak probe runs when --weak-probe-dir is absent",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "outputs/rerun/figures/ablation_deltas.pdf",
    )
    args = parser.parse_args(argv)

    probe_dirs = args.weak_probe_dir
    if not probe_dirs:
        state = json.loads(args.state.read_text(encoding="utf-8"))
        probe_dirs = [
            REPO
            / "outputs/weak_modality_probe"
            / cell["run_id"]
            .replace("weak-probe-must-", "weak_probe_must_")
            .replace("-", "_")
            for name, cell in state["cells"].items()
            if name.startswith("weak_phrasing:") and cell.get("status") == "complete"
        ]
    missing = [
        p for p in probe_dirs if not (p / "weak_modality_text_deltas.csv").exists()
    ]
    if missing:
        raise SystemExit("missing weak probe deltas: " + ", ".join(map(str, missing)))

    set_style()
    fig = build_figure(
        batching_points(read_rows(args.batching)),
        context_points(read_rows(args.context)),
        phrasing_points(probe_dirs),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # No CreationDate: the tracked PDF must not change on a byte-identical rerun.
    fig.savefig(
        args.output,
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={"CreationDate": None},
    )
    fig.savefig(args.output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
