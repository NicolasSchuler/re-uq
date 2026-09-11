# Embedding diagnostic figure

Status: the manuscript contains an explicit rerun placeholder. The existing
PDFs are archived diagnostics and must not be presented as revised results.

Figure 2 retains a horizontal comparison with three separate groups:

- Requirement-only predictions of strict strengthening in the corresponding
  single-pass output, overall and within each nonmandatory source modality.
- The same prediction with the sampled declared modality added to the text.
- Auxiliary predictions of source modality and dataset-by-keyword origin.

The classifier reads sampled requirement text. The primary strengthening target
belongs to the separate single-pass output, not to the text being embedded.
Adding a declared label is an input comparison; a performance change alone does
not demonstrate target leakage. Auxiliary targets also answer different questions.

The plotting selectors and manuscript macros use capability grouping. Compare
input representations on the same target, eligible observations, and capability
splits. All preprocessing is fitted within the training split.

Analysis/export support is implemented; experimental evidence is pending.
`diagnose_embedding_separability.py` writes:

- `probe_grid_predictions.jsonl`: identity of each held-out sampled text, original capability
  and source-condition identity, item, source model/run, sample index, input
  condition/backend, target and observed target value, classifier, ordered class
  labels, probabilities, and fold. The source artifact directory and item/sample indices resolve the sampled text
  and corresponding single-pass requirement in the ACSE cache. Only successful
  fits have predictions. Grid predictions are streamed per analysis cell.
- `probe_grid_folds.csv`: training/test sample and group counts, class counts,
  status and unavailable reasons, iteration count and limit, convergence warnings,
  fitting duration, estimator settings, training accuracy/log loss, and available
  training/validation score histories.
- `probe_grid_summary.csv` / `.md`: recomputable AUROC/AP, AP baseline, ordered
  classes and class counts for eligible and evaluated samples, evaluated fold IDs,
  excluded samples, capability counts, fitting-review flags, and 95% intervals.

For each target/scope the eligible observations are the intersection of usable
inputs across representations, recognized sampled labels, identified capabilities,
valid targets, and (for strengthening) classifiable corresponding output wording.
Unclassified or failed deterministic outputs are not negative target examples.
The same observations, labels, ordering and seeded grouped splitter give comparable
representations identical folds. Capability grouping (`seed`) holds all samples,
modalities, models and keyword renderings of a capability together. Item grouping
remains a separate sensitivity analysis; only capability grouping feeds Figure 2
and manuscript macros. PCA, SVD, scaling and text vectorization remain fold-local.

The reported score is the **equal mean of evaluable held-out folds**, not a pooled
OOF score and not a sample-size-weighted fold mean. Binary AP uses the strengthened
class and its baseline is the mean positive prevalence in those same folds.
Auxiliary multiclass AUROC/AP are macro one-vs-rest; macro AP's baseline is `1/K`
for `K` classes. Fold standard deviations remain descriptive columns only.

Intervals use a deterministic percentile bootstrap of **capabilities**, retaining
all their repeated samples (and associated model/keyword observations), with
fixed predictions and fixed folds. Every draw must retain all target classes in
every evaluated fold; the number of defined draws is recorded. Fewer than 90%
defined draws (or fewer than two defined draws) makes the interval unavailable.
This describes held-out cohort uncertainty conditional on the fitted classifiers
and splits. It does not quantify retraining, alternative split selection, or
new model-generation uncertainty. The CLI accepts `--bootstrap-samples`; the
one-command driver passes its analysis bootstrap setting.

Missing target classes, insufficient groups, failed fitting, and test folds missing
classes are exported explicitly. Class-deficient test-fold predictions are retained
but excluded from ranking summaries. Zero evaluable folds yields unavailable
estimates. The HGB fit uses up to 300 boosting iterations and records its loss
history with training-loss stopping (20 rounds without improvement, no separate
internal validation subset). Logistic regression retains its 1,000-iteration
limit. Limit hits, convergence warnings, no training-loss improvement, or too few
training samples for a tree split flag the summary and plot for fitting review. Training-loss stopping is not evidence of generalization or convergence;
inspect the curves and limits before interpreting weak performance. No fitting
budget extension or experimental comparison was run as part of implementation.

The horizontal figure reads estimates and interval bounds from the current summary,
separates prediction targets, displays unavailable scores/intervals explicitly, and
marks fitting-review flags. `export_paper_numbers.py` reads the same summary and
exports intervals, baseline, sample/capability/fold counts as additional macros.
`numEmbAddedLabel` names the added-input comparison; `numEmbLeakControl` is retained
only as a legacy macro alias, without a leakage claim. No historical score constants
are used. The standalone `probe_acse_embedding_separability.py` also exports held-out
predictions and the same summary convention.

Before publication, run the configured cohort, inspect actual fitting diagnostics
and exclusions, check intervals and class balance, and regenerate the figure.
The manuscript's result placeholders and stale archived figures remain unchanged.

Figure 3 is reserved for paired ablation changes with intervals, matched counts,
and a zero-change reference. Keep document-context results separate from the
main benchmark. The t-SNE files remain exploratory supplementary material;
visible overlap is not evidence that embeddings lack relevant information.
