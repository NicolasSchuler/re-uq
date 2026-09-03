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
import numpy as np

try:
    import eval_utils as eu
    from plot_acse_global_embedding_projection import (
        drift_status,
        load_embeddings_and_rows,
        manifest_rows,
    )
    from plot_embedding_diagnostic_figure import (
        balanced_subsample,
        compute_projection,
        panel_projection,
        set_style,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu
    from scripts.plot_acse_global_embedding_projection import (
        drift_status,
        load_embeddings_and_rows,
        manifest_rows,
    )
    from scripts.plot_embedding_diagnostic_figure import (
        balanced_subsample,
        compute_projection,
        panel_projection,
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

    cache = np.load(
        diagnostic_dir / "task2_reqonly_mlx_embeddings.npz", allow_pickle=False
    )
    reqonly = cache["embeddings"].astype(np.float32, copy=False)
    rows = manifest_rows(manifest_path, "mlx:")
    _, sample_rows = load_embeddings_and_rows(rows)
    if len(sample_rows) != reqonly.shape[0]:
        raise ValueError(
            f"row/embedding mismatch: {len(sample_rows)} vs {reqonly.shape[0]}"
        )

    # One point per distinct generated requirement (each text recurs ~17x).
    seen: set[str] = set()
    unique_global: list[int] = []
    for i, row in enumerate(sample_rows):
        text = str(row.get("requirement", ""))
        if text and text not in seen:
            seen.add(text)
            unique_global.append(i)
    unique_rows = [sample_rows[i] for i in unique_global]
    unique_global_arr = np.asarray(unique_global)

    rng = np.random.default_rng(args.random_state)
    keep_local = balanced_subsample(unique_rows, args.per_modality, rng)
    global_idx = unique_global_arr[keep_local]
    sub_rows = [unique_rows[k] for k in keep_local]
    print(
        f"projection subsample: {global_idx.size} points "
        f"(strict positives: {sum(drift_status(r) == 'strict_text_oc' for r in sub_rows)})"
    )
    coords = compute_projection(reqonly[global_idx], args.method, args.random_state)

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
