# Embedding diagnostic figure

Status: the manuscript includes the current held-out diagnostic results from
`outputs/embedding_diagnostic/probe_grid_summary.csv`. The tracked figure is
`outputs/rerun/figures/embedding_diagnostic.pdf`, with a PNG preview beside it;
the manuscript includes a copy under `manuscript/figures/`.

Figure 2 presents three dot-and-interval panels:

- A compares strengthening detection across all source modalities, using sampled
  requirement text alone or with the sampled answer's declared modality label.
- B shows text-only strengthening detection within recommended, optional, and
  weak-intent sources. Mandatory sources cannot be strengthened further on the
  ordered commitment scale and therefore have no within-modality binary AUROC.
- C separately predicts source commitment level and the joint dataset × keyword
  variant. Its macro one-vs-rest AUROC answers different questions from the
  binary strengthening AUROC in A/B. Recoverable source information may help
  explain the overall/within-modality difference, but does not establish which
  information drove strengthening predictions.

The classifier reads sampled requirement text. The primary strengthening target
belongs to the separate single-pass output, not to the text being embedded.
Adding a declared label is an input comparison; a performance change alone does
not demonstrate target leakage. Auxiliary targets also answer different questions.

The plotting selectors and manuscript macros use capability grouping. Compare
input representations on the same target, eligible observations, and capability
splits. All preprocessing is fitted within the training split.

The current diagnostic exports held-out predictions, folds, and summary intervals.
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
limit. Full classifier parameters (`model_steps` in
`scripts/probe_acse_embedding_separability.py`): fold-local PCA to 128
components (randomized SVD; a fold-safe truncated SVD for the TF-IDF baseline),
then `HistGradientBoostingClassifier(learning_rate=0.08, max_iter=300,
early_stopping=True, validation_fraction=None, n_iter_no_change=20,
max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=0.05,
class_weight="balanced")`; the logistic-regression alternative is
`StandardScaler` + `LogisticRegression(max_iter=1000, class_weight="balanced",
solver="lbfgs")`. Splits are three-fold `GroupKFold`-style by capability with
the seeded grouped splitter; the manuscript reports the HGB run. Limit hits, convergence warnings, no training-loss improvement, or too few
training samples for a tree split flag the summary and plot for fitting review. Training-loss stopping is not evidence of generalization or convergence;
inspect the curves and limits before interpreting weak performance. No fitting
budget extension or experimental comparison was run as part of implementation.

The figure reads estimates and interval bounds from the current summary without
changing their values. All seven scores use three decimal places. Six selected
intervals have finite, unequal bounds. Smaller outlined markers and interval
caps distinguish narrow intervals from missing ones. A dagger marks an
unavailable interval, while a double dagger identifies identical exported bounds
if a future result has a collapsed interval.

The optional-source AUROC is available, but its interval is not: only 859 of
1,000 capability-bootstrap draws retain all target classes in every evaluated
fold, below the required 90%. The estimate remains visible with a dagger and the brief note "CI unavailable".
The detailed reason is reported here and in the plotting command output. Its evaluated cohort contains 30,600 negative and 60 positive
samples. No interval is invented, widened, or omitted for visual convenience.

All 21 folds for the seven displayed conditions reached the 300-step limit.
The plotting command reports this qualification from the selected fold exports
and otherwise reports a generic fitting-review note when the summary flags a concern.
Training-budget and interval-method details stay outside the compact graphic. The
manuscript retains its training-budget qualification. These results do not
establish convergence, and the intervals do not describe budget sensitivity.

`export_paper_numbers.py` reads the same summary and exports intervals, baseline,
sample/capability/fold counts as additional macros. `numEmbAddedLabel` names the
added-input comparison. `numEmbLeakControl` remains a legacy alias without a
leakage claim. No historical score constants are used.

To regenerate only this figure from the existing results, without fitting or
calling a model, run from the repository root:

```bash
.venv/bin/python scripts/plot_embedding_diagnostic_figure_v2.py --diagnostic-dir outputs/embedding_diagnostic --output outputs/rerun/figures/embedding_diagnostic.pdf
cp outputs/rerun/figures/embedding_diagnostic.pdf manuscript/figures/embedding_diagnostic.pdf
cp outputs/rerun/figures/embedding_diagnostic.png manuscript/figures/embedding_diagnostic.png
```

Inspect the standalone figure and its compiled manuscript page before treating
an updated rendering as ready. Historical numbered figures and documentation
previews are not the manuscript's included artifact.

Figure 3 is reserved for paired ablation changes with intervals, matched counts,
and a zero-change reference. Keep document-context results separate from the
main benchmark. The t-SNE files remain exploratory supplementary material;
visible overlap is not evidence that embeddings lack relevant information.
