# Embedding diagnostic figure (RQ3)

> **Status: this page describes the committed PDF, which is one revision behind the
> script.** `plot_embedding_diagnostic_figure_v2.py` now selects the
> **requirement-only** substrate for the target bars (`mlx / reqonly / seed`:
> 0.707 global, 0.613 / 0.612 / 0.620 within level) and shows the label-prefixed
> 0.822 as a separate, hatched **leakage control**. The prefixed string begins
> `modality: <predicted label>` and strengthening is derived partly from that
> label, so it cannot serve as the headline substrate. The numbers, PDF, LaTeX
> caption and the paper's Table 1 below still show the previous prefixed
> configuration; rerun the commands under **Reproduce** to bring them in line.

Paper figure showing that black-box text embeddings encode the **input commitment
level** and **source dataset** of a generated requirement, but not whether the
model **strengthened** it — the strengthening signal that a global classifier
appears to pick up collapses to chance once the input level is held fixed.

- `docs/figures/embedding_diagnostic.pdf` — the paper figure (single AUROC bar chart)
- `docs/figures/embedding_diagnostic.png` — raster preview
- `docs/figures/embedding_diagnostic_tsne_supp.{pdf,png}` — **supplementary only**
  (the two t-SNE maps dropped from the paper figure), for the replication package
- Data: `outputs/embedding_diagnostic/probe_grid_summary.{csv,md}` (full factorial),
  `outputs/embedding_diagnostic/task2_reqonly_mlx_embeddings.npz` (requirement-only re-embedding cache)

## What the figure shows (v2)

A single horizontal bar chart. Every bar is a held-out **AUROC** (the one metric,
named on the x-axis; 0.5 = chance, 1.0 = perfect). The multi-class probes use
macro-averaged one-vs-rest AUROC; the strengthening probes are binary AUROC. Two
visually separated groups:

**Context (not the target)** — what the embedding trivially encodes:

| Bar | AUROC | probe_grid row |
|---|---|---|
| Input commitment level | **0.84** | `mlx / reqonly / item / global / source_modality` |
| Source dataset | **0.73** | `mlx / reqonly / item / global / dataset_variant` |

**Target: was the text strengthened?** — what we actually want to detect
(**these three match the paper's Table 1 exactly**):

| Bar | AUROC | probe_grid row |
|---|---|---|
| Global classifier (all inputs pooled) | **0.822** | `mlx / prefixed / seed / global / deterministic_strict_text_overcommit` |
| Within recommended-only inputs | **0.519** | `mlx / prefixed / seed / source_modality=recommended / deterministic_strict_text_overcommit` |
| Within optional-only inputs | **0.582** | `mlx / prefixed / seed / source_modality=optional / deterministic_strict_text_overcommit` |
| Within weak-intent inputs | **0.600** | `mlx / prefixed / seed / source_modality=nice_to_have / deterministic_strict_text_overcommit` |

All three within-level strata are shown so the check is not cherry-picked to two
of them. The **mandatory** stratum is omitted because it is *undefined*: mandatory
is the top of the ordinal scale (`ORDINAL_STRENGTH` in `eval_utils.py`), so no
generated text can be *stronger* — there are zero positives and no AUROC. For
**weak-intent** inputs, "strengthened" nearly coincides with "contains any
explicit modal word," so its 0.600 is a looser test than the recommended/optional
strata (which require *must*-vs-*should* / *should*-vs-*may* distinctions); it is
still near chance, which is why showing it strengthens rather than weakens the
claim. These four bars match the paper's Table 1 exactly.

The global-vs-within comparison (0.822 → 0.519 / 0.582 / 0.600) is on a **single substrate**
(mlx / prefixed / seed / deterministic-strict), so the collapse is a clean,
same-metric, same-target contrast: holding the input level fixed removes the
signal, i.e. the global 0.822 is inflated by **input-level leakage**.

### Metric / substrate notes

- **Metric consistency** (the critical fix): all five bars are AUROC. The context
  0.84/0.73 were *already* AUROC (macro-OvR), not accuracy — `roc_auc_score(...,
  multi_class="ovr", average="macro")` in `diagnose_embedding_separability.py` —
  so nothing had to be recomputed; no metrics are mixed.
- **Context substrate** is the de-circularized *requirement-text-only, item-grouped*
  probe. We deliberately do **not** use the prefixed source-modality probe (AUROC
  0.96), because there the modality word is literally prepended to the embedded
  string, so 0.96 just reads the label off the prefix. The honest 0.84/0.73 make
  the "input level is recoverable" point without that circularity. (Pinning
  dataset to 0.73 specifically requires item-grouping; seed-grouping gives ~0.82.)
- **Target substrate** is `prefixed / seed`, exactly the configuration the paper's
  Table 1 reports, so bars 3–5 match Table 1 to the digit. The global bar is the
  one-shot (`deterministic_strict`) variant, 0.822; the sampled variant is 0.828.
- The figure script **asserts** bars 3–5 equal 0.822 / 0.519 / 0.582 and errors
  out if a regenerated `probe_grid_summary.csv` ever drifts from Table 1.

## LaTeX include

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/embedding_diagnostic.pdf}
  \caption{\textbf{Black-box text embeddings reveal what kind of input a generated
  requirement came from, but not whether the model strengthened it.} Held-out AUROC
  of a probe on \texttt{Qwen3-Embedding-0.6B} embeddings of Task~2 generated
  requirements (0.5~=~chance, 1.0~=~perfect). The input's commitment level ($0.84$)
  and its source dataset ($0.73$) are easily recovered. A global classifier appears
  to detect strengthening ($0.822$), but within a single input level the signal
  collapses to chance (recommended $0.519$, optional $0.582$, weak intent $0.600$;
  mandatory is undefined): the global number partly reflects the input condition
  rather than a genuine strengthening signal.}
  \label{fig:embedding-diagnostic}
\end{figure}
```

The two t-SNE maps (previously panels a/b) are exported separately as
`embedding_diagnostic_tsne_supp.pdf` for the replication package; include them in
the supplement if desired, not in the main paper.

## Reproduce

```bash
# optional MLX dep (the Qwen3 model is already cached under .cache/huggingface)
VIRTUAL_ENV=.venv uv pip install mlx mlx-embeddings

export HF_HOME=$PWD/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# factorial probe (MLX/TF-IDF x prefixed/reqonly x seed/item) + AUPRC
.venv/bin/python scripts/diagnose_embedding_separability.py --models hgb

# the paper figure (single AUROC bar chart) -> docs/figures/embedding_diagnostic.pdf
.venv/bin/python scripts/plot_embedding_diagnostic_figure_v2.py

# supplementary t-SNE maps -> docs/figures/embedding_diagnostic_tsne_supp.pdf
.venv/bin/python scripts/plot_embedding_diagnostic_tsne_supp.py
```

Scripts:
- `scripts/diagnose_embedding_separability.py` — builds the four feature sets,
  reduces each to 128 dims, runs grouped CV, writes `outputs/embedding_diagnostic/`.
- `scripts/plot_embedding_diagnostic_figure_v2.py` — the single-bar paper figure
  (pulls the five bars from `probe_grid_summary.csv`; asserts Table-1 agreement).
- `scripts/plot_embedding_diagnostic_tsne_supp.py` — the supplementary t-SNE maps.
- `scripts/plot_embedding_diagnostic_figure.py` — the previous three-panel figure
  (superseded; kept for reference).
