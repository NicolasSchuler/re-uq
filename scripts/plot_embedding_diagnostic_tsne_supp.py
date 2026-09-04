"""Supplementary figure: the two t-SNE projections dropped from the paper figure.

The main paper figure (``plot_embedding_diagnostic_figure_v2.py``) is now a
single AUROC bar chart. The two 2-D t-SNE maps of the requirement-only MLX
embeddings -- colored by input strength (a) and by strength increase (b) -- are
kept here for the replication package only, not the paper.

Reuses the projection helpers from the original three-panel script so the maps
are pixel-identical to the earlier panels (a)/(b).
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

try:
    import eval_utils as eu
    from plot_embedding_diagnostic_figure import (
        panel_projection,
        prepare_projection_inputs,
        set_style,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu
    from scripts.plot_embedding_diagnostic_figure import (
        panel_projection,
        prepare_projection_inputs,
        set_style,
    )


def main() -> None:
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
        "--output",
        type=Path,
        default=Path("docs/figures/embedding_diagnostic_tsne_supp.pdf"),
    )
    parser.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    parser.add_argument("--per-modality", type=int, default=850)
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

    coords, sub_rows, _, rng = prepare_projection_inputs(
        diagnostic_dir=diagnostic_dir,
        manifest_path=manifest_path,
        method=args.method,
        per_modality=args.per_modality,
        random_state=args.random_state,
    )

    set_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.6), layout="constrained")
    panel_projection(
        ax_a,
        coords,
        sub_rows,
        "source_modality",
        "(a) Colored by input strength",
        rng=rng,
    )
    panel_projection(
        ax_b, coords, sub_rows, "drift", "(b) Colored by strength increase"
    )
    fig.suptitle(
        "Generated requirements (t-SNE of requirement-only embeddings)",
        fontsize=12.5,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"wrote {output_path} and {output_path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
