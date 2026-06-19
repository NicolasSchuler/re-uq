# Embedding diagnostic figure (RQ3)

Artifacts for the new paper figure showing that black-box text embeddings encode
the **input commitment level**, not the **modal-force drift**.

- `docs/figures/embedding_diagnostic.pdf` — vector figure for LaTeX
- `docs/figures/embedding_diagnostic.png` — raster preview
- Data: `outputs/embedding_diagnostic/probe_grid_summary.{csv,md}` (full factorial),
  `outputs/embedding_diagnostic/task2_reqonly_mlx_embeddings.npz` (requirement-only re-embedding cache)

## LaTeX include

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/embedding_diagnostic.pdf}
  \caption{\textbf{Black-box text embeddings reveal what kind of input a generated
  requirement came from, but not whether the model strengthened it.} Task~2 generated
  requirements are embedded with \texttt{Qwen3-Embedding-0.6B} using the
  \emph{requirement text only}. \textbf{(a)}~A 2-D t-SNE of distinct generated
  requirements colored by input strength and \textbf{(b)}~the same map colored by
  whether the requirement's strength was increased show no separation by either
  property. \textbf{(c)}~The \emph{detection score} is the held-out ability of a
  histogram gradient-boosted tree ensemble (\texttt{HistGradientBoosting} on PCA-128
  features; 3-fold \textsc{StratifiedGroupKFold} over \emph{unseen requirements}) to tell
  two cases apart---$0.5$ is a coin-flip guess, $1.0$ is perfect (equivalently, AUROC). From the embedding a probe reads the original input strength
  ($0.84$) and the source dataset ($0.73$), but detecting whether the model strengthened
  the requirement, comparing inputs of equal strength, is a coin flip---and a plain
  keyword search over the same text does no better ($0.55$ vs.\ $0.58$; averaged over
  input levels, $0.48$ for recommended-only inputs). Strengthened points in (b) are
  enlarged for visibility.}
  \label{fig:embedding-diagnostic}
\end{figure}
```

## How this sharpens RQ3 vs. the current Table 1

The current Table 1 reports the text-embedding classifier with **seed-grouped** CV on
the **label-prefixed** string (`modality: <label>  requirement: <text>`). Three things
make those numbers optimistic; the figure uses the conservative version and the
conclusion gets *stronger*, not weaker.

| Issue in current pipeline | Fix | Effect on the numbers |
|---|---|---|
| Embedded string carries the predicted label, so "embeddings encode commitment level" is partly circular | Re-embed **requirement text only** | `source_modality` AUROC $0.99\!\to\!0.84$ (still high — wording carries modality — but not circular) |
| CV groups by **seed**, so an item's text straddles train/test (memorise per-item drift) | Group by **item** (held-out capability) | within-level drift AUROC drops further, e.g. recommended $0.519\!\to\!0.48$ |
| AUROC hides a 2–13% positive rate | Report **AUPRC vs. prevalence** | within-level lift $\approx 1.0$ — i.e. no better than guessing at the base rate |
| Only the neural backend was probed | Add **TF-IDF char $n$-gram** | the within-level signal is shown to be lexical (keyword), which the neural embedding largely discards |

### Suggested replacement numbers (held-out, requirement-only, item-grouped, HGB)

"Detection score" in the figure is AUROC (0.5 = chance, 1.0 = perfect).

| Quantity | Detection score (AUROC) | Note |
|---|---|---|
| Input strength (`source_modality`) | **0.84** | confound; 0.99 if the label is left in the embedded string |
| Source dataset | **0.73** | corpus-identity confound |
| Strengthening, global | **0.69** | looks usable, but rides on the input-strength confound |
| Strengthening, within input strength (weak / optional / recommended) | **0.60 / 0.55 / 0.48** | collapses to (and below) chance |
| Strengthening, within input strength — embedding vs keyword search | **0.55 vs 0.58** | a plain keyword baseline does no better (Fig. panel c) |
| cosine(weak-intent, its `shall` rewrite) | **0.97** | the embedding barely moves when modal force flips |

Supplementary (rare-positive view): within-level strengthening AUPRC lift over
prevalence is **≈ 1.0** (no usable precision at the base rate). At the *sample* level
("is an explicit stronger modal word literally present in this output?"), keyword search
recovers it far better than the neural embedding (AUROC 0.97 vs 0.73; AUPRC lift 33× vs
5× for recommended-only inputs) — the meaning embedding blurs the modal keyword. Full grid:
`outputs/embedding_diagnostic/probe_grid_summary.md`.

Drop-in sentence for §4 (RQ3): *"On the honest substrate (requirement text only,
held out over unseen requirements), a trained probe still recovers the input commitment
level (0.84) and corpus identity (0.73) from the embedding geometry, but detecting whether
the model strengthened the requirement, comparing inputs of equal strength, is a coin flip
(0.48–0.60) — and a plain keyword baseline does no better (0.58). Unsupervised clusters and
2-D embedding plots (Fig.~X a–b) therefore neither expose the strengthening nor reflect the
structure a probe can exploit."*

## Reproduce

```bash
# optional MLX dep (the Qwen3 model is already cached under .cache/huggingface)
VIRTUAL_ENV=.venv uv pip install mlx mlx-embeddings

export HF_HOME=$PWD/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# factorial probe (MLX/TF-IDF x prefixed/reqonly x seed/item) + AUPRC
.venv/bin/python scripts/diagnose_embedding_separability.py --models hgb

# the figure
.venv/bin/python scripts/plot_embedding_diagnostic_figure.py --per-modality 850 --method tsne
```

Scripts:
- `scripts/diagnose_embedding_separability.py` — builds the four feature sets, reduces
  each to 128 dims, runs grouped CV, writes `outputs/embedding_diagnostic/`.
- `scripts/plot_embedding_diagnostic_figure.py` — composes the 4-panel figure.
