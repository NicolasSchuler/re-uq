# Aggregation Formula

This page states exactly how a raw model completion becomes a headline number.
It covers the scoring unit, how deterministic and stochastic rows enter each
metric, what happens to parse failures and to text whose modality cannot be
read, pooled versus macro-of-cells aggregation, how models are pooled, and the
bootstrap procedure.

Line numbers are anchors into the current tree and may drift; the function names
are canonical.

## 1. Scoring unit

The scoring unit is **one model answer for one benchmark item**.

- A benchmark item is a `(seed_id, source_modality)` pair: one reviewed
  capability rendered in one of the four source conditions
  (`mandatory`, `recommended`, `optional`, `nice_to_have`). Each cell of the
  design (dataset x benchmark variant) holds 180 seeds x 4 = 720 items.
- A cell is a `(dataset, variant)` pair: `mlm_tapt/must`, `mlm_tapt/shall`,
  `nice/must`, `nice/shall`. The `pure/must` cell of the document-context
  ablation ([`context_ablation.md`](context_ablation.md)) is scored the same
  way but is reported on its own and never enters any pooled or macro figure.
- Score rows are produced by `build_uq_scores`
  (`scripts/eval_utils.py:5606`). Every row carries `run_id`, `model`, `task`,
  `uq_method`, `item_id`, `seed_id`, `batch_id`, `sample_batch_ids`,
  `source_modality`, `valid_n`, `total_n`, `parse_failures`,
  `stochastic_complete`, and `sampling_plan_source`, built by
  `score_base` (`scripts/eval_utils.py`, Section 9).
- `batch_id` is the provider **request** the answer was decided in, forwarded
  from the raw record (`<run_id>:<model>:<task>:<sample_kind>:<sample_index>:<min>-<max>`).
  Every archived run sent 16 items per request — four whole seeds x four source
  conditions — so a request contains whole seeds and clustering by request
  nests clustering by seed. It is the default bootstrap cluster, see Section 6.
  `sample_batch_ids` lists every contributing request, semicolon-joined; it
  equals `batch_id` for a deterministic row and names all five sample requests
  for a collapsed stochastic row (whose `batch_id` is its representative
  sample's request). Both are blank for rows that were never batched: legacy
  runs, single-item requests, synthesised completions, and the rule baseline.
- `build_uq_scores` requires a `SamplingPlan`: the number of stochastic samples
  the run *planned* per item, threaded from the run config by every caller.
  `total_n` is `max(observed, planned)`, so three written samples of a planned
  five read as an incomplete `3/5` rather than a complete `3/3`. Reconstructing
  the denominator from the rows that happen to exist is exploratory only, needs
  `infer_plan_from_observations=True`, and stamps every row it produces with
  `sampling_plan_source = "inferred"`.
- Each run records the plan it was executed under in the registry's
  `expected_stochastic_samples` column, so a later comparison does not have to
  guess it. `compare_run_matrix.py` scores each completed run against its own
  recorded plan and refuses to compare a run whose recorded count disagrees
  with the comparison config: judging a three-sample run against a five-sample
  config would mark every one of its stochastic groups incomplete and report
  blank agreement for a run that was in fact complete. Rows written before the
  column existed leave it blank, fall back to the comparison config, and are
  named once in a warning.
- Raw rows are **deduplicated before scoring**. If one
  `(run_id, model, task, item_id, sample_kind, sample_index)` key carries
  several raw rows — possible after a resume that changed the job config — the
  latest parsed (`ok`) row wins, falling back to the latest row when none
  parsed. The cohort data contained no such duplicates; the rule exists so a
  future resume cannot double-count an item.

All Task 2 headline metrics are computed over rows with
`task == "task2"` and `uq_method == "verbalized_confidence"`: the single
deterministic (temperature 0) answer per item and model.

## 2. Deterministic vs stochastic rows

| Row kind | `uq_method` | `total_n` | Enters |
| --- | --- | --- | --- |
| deterministic answer | `verbalized_confidence` | 1 | label accuracy, over-commitment, text-drift, confidence shares, length metrics |
| repeated samples (5 at temperature 0.7) | `modality_consistency`, `predictive_entropy`, `variation_ratio`, `acse_semantic_entropy` | 5 | stability / agreement / dispersion metrics only |
| ensemble view | `model_ensemble_disagreement` | k models | diagnostics only |
| rule baseline | `rule_based_modality` | 1 | reference row only |

A stochastic group collapses to one row per `uq_method` per item, holding the
sampled label distribution. Those rows never enter a strengthening numerator or
denominator: the strengthening construct is defined on the deterministic answer
text.

Repeated-sample **agreement and unanimity** are computed only over items where
every requested sample parsed, i.e. `stochastic_complete == True`
(`valid_n == total_n`, set in `score_base`). `total_n` is the **expected**
sample count (5), not the number of rows that happen to exist, so an item whose
fifth sample was never written counts as incomplete instead of passing as a
complete group of four. See
`repeated_sample_agreement_metrics` (`scripts/eval_utils.py:7272`), which
returns `repeated_sample_unanimity`, `mean_repeated_sample_agreement`,
`agreement_n_complete`, and `agreement_n_incomplete_excluded`. Items with a
partially failed sample set are excluded from the numerator *and* the
denominator and are visible only through the excluded count. The paper exporter
builds this population from every strict-strengthened deterministic item, so a
group with zero valid stochastic samples is also counted as incomplete even
though `build_uq_scores` cannot emit a distribution row for it. Pooled agreement
weights each cell by `agreement_n_complete`, never by the larger strict-
strengthened population.
`build_uq_scores(..., min_valid_samples=k)` additionally drops stochastic groups
with fewer than `k` parsed samples before any of this.

## 3. Parse failures and unknown text

Two different failure modes are kept apart.

1. **Response parse failure.** The completion did not yield a valid JSON object
   with a valid label (`parse_task_response`, `scripts/eval_utils.py`, Section
   6). Such a record produces **no score row**, so it is absent from every
   numerator and denominator. It is reported separately as
   `parse_failure_rate = sum(parse_failures) / sum(total_n)` in
   `metric_summary_by_model_task_method` (`scripts/eval_utils.py:7951`) and as
   `n_parse_failures` in the per-model tables. In the per-cell paper snapshot
   that quantity was structurally 0: a failed record never becomes a score row,
   and every deterministic score row carries `valid_n == total_n == 1`, so
   nothing could reach the numerator. `scripts/export_paper_tables.py` now
   derives the snapshot's `parse_failure_rate` from the raw Task 2
   deterministic rows, where the failures actually are.
2. **Unreadable text modality.** The answer parsed, but the generated
   requirement text carries no usable modal signal, or carries a *negated*
   modal (`must not`, `cannot`, `shouldn't`, ...). The detector
   `requirement_text_modality_diagnostic` (`scripts/eval_utils.py:3162`) then
   returns `unknown` / `negated`, `text_modality_fields`
   (`scripts/eval_utils.py:3240`) sets
   `text_modality_parse_status = "unknown"`, and the row is **excluded from the
   text-strengthening denominator**. A negated modal never counts as
   strengthening and never enters the strict evidence basis. Negation only wins
   when the text carries no positive modal cue: "The system must ensure that
   users cannot delete records." resolves to mandatory and is flagged
   `text_modality_multi_modal`, not dropped as `negated`. No cohort row
   contained a negated cue — the negated rate is 0.0 in all four cells — so
   this precedence changes no published number.

Because that exclusion is a coverage assumption, every text-strengthening rate
is published together with its accounting, from `text_modality_summary_metrics`
(`scripts/eval_utils.py:7178`) via `_coverage_adjusted_bounds`
(`scripts/eval_utils.py:7155`):

```
rate            = numerator / denominator                       # published value
*_n_numerator   = strengthened rows
*_n_denominator = rows with a readable text modality
*_n_unknown_excluded = rows excluded for unreadable/negated text
*_lower_bound   = numerator / (denominator + unknown)           # unknowns never strengthen
*_upper_bound   = (numerator + unknown) / (denominator + unknown)  # unknowns always strengthen
```

`heuristic_text_modality_rate` is reported next to the broad rate in the same
table: it is the share of the denominator whose modality came only from the
generic `heuristic_system_verb` fallback rather than an explicit modal or weak
phrase. The strict metric excludes exactly that basis
(`STRICT_TEXT_MODALITY_BASES`).

## 4. Pooling models inside a cell

Within one cell the cohort models are pooled by **concatenating their score
rows**: every deterministic Task 2 answer is one unit, so a model contributes in
proportion to the items it answered. There is no per-model macro step, and no
model weighting. With a complete cohort run each of the 6 models contributes 720
items, giving `n = 4320` answers and a coverage-adjusted denominator of
`n_denominator <= 4320`.

The cohort is fixed: `glm-4.5-air`, `glm-4.7`, `glm-5`, `glm-5-turbo`,
`glm-5.1`, `kit.gemma4-31b-it`. Private `azure.*` rows are excluded
(`--exclude-model-prefix azure.`), as are smoke runs and incompatible registry
rows. The default exporter pins `run_group_id=provider-matrix-2026-05` and
requires exact Task 1+2 membership, the full benchmark, five stochastic
samples, complete deterministic/stochastic coverage, batch size 16, and grouped
batching. Historical paper rows have a blank `batch_order` field; that legacy
blank is accepted only inside the pinned group with the other constraints.
When a model has several compatible runs in a cell, the most recent wins; the
resolved run ids and compatibility settings are written to
`outputs/paper_snapshot_provenance.json` by `export_tables`
(`scripts/export_paper_tables.py:548`, selection in `select_cell_runs`,
`scripts/export_paper_tables.py:180`). A registry row marked `complete` whose
raw rows were later removed is not a usable run: the exporter now fails on a
selected run that has no raw rows instead of exporting an empty cell. Local
registries still hold such rows from 2026-05-21 ([`TODO.md`](../TODO.md),
section E).

Per-model disaggregation of the same rows is exported to
`outputs/paper_per_model_modality_table.csv` (model x dataset x variant x
source modality) and `outputs/paper_per_model_headline.csv` (model, pooled over
cells) by `per_model_row` (`scripts/export_paper_tables.py:410`).

## 5. Pooled vs macro-of-cells

`scripts/aggregate_paper_headline_metrics.py` reports both for every headline:

- **pooled (item-weighted)**: `sum(per-cell numerators) / sum(per-cell
  coverage-adjusted denominators)`. For strict text strengthening that is
  `1412 / 16448 = 0.0858`; for broad, `2268 / 16448 = 0.1379`.
- **macro-of-cells (unweighted)**: the arithmetic mean of the four per-cell
  rates. Strict `0.0860`, broad `0.1386`.

The README quotes 8.6% strict (pooled and macro agree at that rounding) and
13.8% broad (pooled, the conservative convention; the macro is 13.9%). The
weak-intent headline (29.8%) is a single named cell, `mlm_tapt/must`, not an
aggregate. It is strict text strengthening at `confidence >= 0.90` over the
weak-intent (`nice_to_have`) source rows of that cell whose generated text had a
readable modality: `304 / 1020 = 0.298`. Over all 1080 weak rows of the cell,
including those whose text modality was unreadable, it is 28.1%. The exporter
writes it as `weak_strict_text_strengthening_90` — alongside `weak_n` (1080),
`weak_n_readable` (1020) and `weak_strict_text_strengthening_90_all_weak`
(0.281) — into `outputs/paper_task2_text_drift_metrics.csv`, and
`scripts/aggregate_paper_headline_metrics.py` reads the headline from that
column, falling back to `outputs/blind_task3_analysis_summary.csv` only when the
column is absent.

The high-confidence share (98.4%) is the unweighted macro over cells, with the
strict-strengthened counts as per-cell n. Repeated-sample agreement is unanimous
in every cell.

## 6. Bootstrap procedure

All confidence intervals are **clustered nonparametric bootstraps**
(`bootstrap_seed_metric`, `scripts/eval_utils.py`, Section 7):

1. Group the score rows by the cluster field. The default is the provider
   **request** (`batch_id`, `DEFAULT_BOOTSTRAP_CLUSTER_FIELD`), not the seed:
   every archived run sent 16 items per request, so one request carries four
   whole seeds x four source conditions. The item is not independent of its
   seed (the four source-modality variants share a capability) and the seed is
   not independent of its request — strict text strengthening is all-or-none
   within each `(request, source condition)` group, and unreadable text
   modality is all-or-none per request. Because a request contains whole
   seeds, request clustering *nests* seed clustering and is the conservative
   choice.
2. Draw `n_clusters` clusters with replacement and concatenate their rows.
3. Recompute the metric on the resampled rows; repeat `iterations` times
   (default 1000).
4. Report the 2.5% and 97.5% percentiles of the resampled values as
   `*_ci_low` / `*_ci_high`; the point estimate is the metric on the observed
   rows.

The seed-clustered interval is reported alongside the primary one as
`*_seed_ci_low` / `*_seed_ci_high`, and `bootstrap_ci_cluster_field` records
which field the primary interval actually used.

Read the seed-clustered column with one caveat: `seed_id` is only unique within
a dataset, not across benchmark variants, so the `must` and `shall` renderings
of one capability share an id and land in the same cluster. Pooling the four
paper cells therefore yields 360 seed clusters (2 datasets x 180 seeds) against
1080 request clusters. `batch_id` embeds the run id and is globally unique, so
the primary interval is unaffected.

Rows fall back to clustering on `seed_id` unless **every** row carries a
non-blank `batch_id` (`resolve_bootstrap_cluster_field`). A partially populated
column — legacy runs, single-item requests, synthesised completions, the rule
baseline — would otherwise collapse all unbatched rows into one meaningless
cluster. When the fallback engages the two intervals coincide, only one
bootstrap runs, and `bootstrap_ci_cluster_field` reads `seed_id`.

The RNG seed is fixed at `20260518` (`BOOTSTRAP_SEED` in
`scripts/export_paper_tables.py` and `scripts/compare_run_matrix.py`), so the
intervals are reproducible.

Entry points, and the columns each writes:

- `cluster_ci_fields` (`scripts/eval_utils.py`, Section 10) — the shared shape:
  `{metric}_ci_low` / `{metric}_ci_high`, `{metric}_seed_ci_low` /
  `{metric}_seed_ci_high`, and one `bootstrap_ci_cluster_field` per row.
- `text_over_commitment_ci_fields` (`scripts/eval_utils.py`, Section 10) — broad
  and strict text strengthening with counts and both intervals.
- `headline_risk_ci_fields` (`scripts/eval_utils.py`, Section 10) — Task 1/2
  high-confidence risks used by `scripts/generate_evaluation_analysis.py`.
- `annotate_text_drift_cis` (`scripts/compare_run_matrix.py`) — per run and
  model in the run-matrix table, `--bootstrap-samples` controls the resamples.
- `scripts/export_paper_tables.py` — `broad_strengthening_ci_*` /
  `strict_strengthening_ci_*` plus their `*_seed_ci_*` pairs and
  `strengthening_ci_cluster_field` in the per-model and headline tables, and
  `ci_low` / `ci_high` / `seed_ci_low` / `seed_ci_high` / `ci_cluster_field` in
  `paper_headline_bootstrap_ci.csv`.
- `scripts/aggregate_paper_headline_metrics.py --regenerate-snapshots` — pools
  the deterministic rows across all requested cells, bootstraps over requests,
  and appends `value_ci_low` / `value_ci_high` / `value_seed_ci_low` /
  `value_seed_ci_high` / `value_ci_cluster_field` / `bootstrap_samples` to the
  headline table (`attach_bootstrap_cis`).
- `scripts/compare_context_ablation.py` — arm rows carry
  `*_ci_*` / `*_seed_ci_*` / `bootstrap_ci_cluster_field`; delta rows carry
  `delta_ci_*`, `delta_seed_ci_*`, `delta_cluster_field` and
  `n_delta_clusters`. The paired delta **pairs by seed and resamples by
  request**: the two arms are separate runs whose request ids differ, but the
  partition of seeds into requests is the same, and a seed's four source
  conditions sit in one request, so a request carries whole pairs.

The snapshots committed under `outputs/` predate this change: their
`*_ci_low` / `*_ci_high` columns are seed-clustered and they carry no
`*_seed_ci_*` or cluster-field column. Regenerating them (Section 8) produces
request-clustered primaries and the seed-clustered pair alongside.

## 7. Answer length and bloat

`answer_length_fields` (`scripts/eval_utils.py:5308`) adds
`requirement_word_count`, `source_word_count`,
`length_ratio = requirement_word_count / source_word_count`, and
`completion_tokens` to every score row. For a multi-item request, provider token
usage describes the whole batch and `completion_tokens` is therefore left blank
rather than copied onto every answer; the raw batch-level usage remains in
`batch_usage_completion_tokens`. Runner-recorded
`requirement_word_count` / `usage_completion_tokens` are used when present;
legacy rows fall back to counting words in `parsed_json.requirement`.

`length_bloat_metrics` (`scripts/eval_utils.py:7350`) buckets the rows into
three equal-count terciles of `length_ratio` and reports broad and strict
strengthening per bucket (`strengthening_rate_by_length_tercile`,
`length_tercile_{1,2,3}_{text,strict_text}_over_commitment`), plus mean word
counts per source modality
(`mean_requirement_word_count_by_source_modality`). These appear in the
per-model summary and in the regenerated text-drift snapshot, so a
bloat/strengthening interaction is reportable rather than pooled away.

## 8. Regenerating the tables

```bash
.venv/bin/python scripts/export_paper_tables.py \
  --cell mlm_tapt/must --cell mlm_tapt/shall --cell nice/must --cell nice/shall \
  --bootstrap-samples 1000

.venv/bin/python scripts/aggregate_paper_headline_metrics.py --regenerate-snapshots
```

By default the exporter writes the regenerated per-cell snapshots under the
`*_regenerated.csv` names so the shipped diagnostic snapshots are left intact;
pass `--overwrite-snapshots` to write the canonical names.
