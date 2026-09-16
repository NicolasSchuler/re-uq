# Results Mapping

This page maps each element of the paper to the artifact behind it, the script
that wrote it, and the inputs it was computed from, so a reviewer can follow
any number or figure back to hashed inputs. All artifacts below are the
`manuscript-final` campaign; run ids and input hashes are in
`outputs/paper_snapshot_provenance.json` and `outputs/rerun/manuscript-final/state.json`.

## Tracked cross-cell artifacts (the paper's numbers)

| Paper element | Artifact | Written by | Computed from |
| --- | --- | --- | --- |
| Every number the text prints (`\num...` macros) | `outputs/paper_numbers.tex` | `scripts/export_paper_numbers.py --strict` | the tables below; the file header lists their SHA-256 |
| RQ1 to RQ3 tables, per model and pooled | `outputs/paper_per_model_rq_table.csv` / `.md` | `scripts/export_paper_tables.py` | deterministic and sampled Task 2 rows, Task 1 rows and blind Task 3 rows of the selected run ids |
| Model x source-modality table | `outputs/paper_per_model_modality_pooled.csv` / `.md` (per cell: `paper_per_model_modality_table.csv` / `.md`) | `scripts/export_paper_tables.py` | same |
| Per-model strict and broad strengthening with intervals | `outputs/paper_per_model_headline.csv` / `.md` | `scripts/export_paper_tables.py` | same |
| Pooled headline rates and intervals | `outputs/paper_headline_metrics.csv`, `outputs/paper_headline_bootstrap_ci.csv` | `scripts/aggregate_paper_headline_metrics.py`, `scripts/export_paper_tables.py` | the per-cell snapshots `paper_task2_text_drift_metrics.csv` and `paper_text_drift_confidence_and_stability.csv` |
| Commitment-transition figure (RQ1) | `outputs/rerun/figures/commitment_transitions.tex`; counts in `outputs/commitment_transition_counts.csv`, denominators in `commitment_transition_accounting.csv` | `scripts/export_commitment_transitions.py` | deterministic Task 2 rows of the frozen cells |
| Request-composition ablation | `outputs/batching_ablation_summary.csv` / `.md`, deltas with intervals in `_deltas.csv`, provenance in `_provenance.json` | `scripts/compare_batching_ablation.py` | the arms' run ids recorded in the campaign state |
| Document-context ablation | `outputs/context_ablation_summary.csv` / `.md`, `_deltas.csv`, `_provenance.json` | `scripts/compare_context_ablation.py` | same |
| Phrasing ablation | `outputs/weak_modality_probe/<run>/weak_modality_probe_summary.csv` / `.md`, `weak_modality_text_deltas.csv` (nine runs) | `scripts/run_weak_modality_probe.py` | the probe runs' raw rows |
| Ablation figure | `outputs/rerun/figures/ablation_deltas.pdf` / `.png` | `scripts/plot_ablation_deltas.py` | the two `_deltas.csv` files and the nine `weak_modality_text_deltas.csv` |
| Embedding classifier (RQ3) and its figure | `outputs/embedding_diagnostic/probe_grid_summary.csv` / `.md`, folds in `probe_grid_folds.csv`; `outputs/rerun/figures/embedding_diagnostic.pdf`, supplement `embedding_diagnostic_tsne_supp.pdf` | `scripts/diagnose_embedding_separability.py`, `scripts/plot_embedding_diagnostic_figure_v2.py`, `scripts/plot_embedding_diagnostic_tsne_supp.py` | the embedding caches named in `outputs/rerun/acse_selected_manifest.csv` |
| Meaning-variation sensitivity appendix | `outputs/meaning_variation_sensitivity/grid_auroc.csv`, `cluster_diagnostics.csv`, `reproduction_check.csv`, `cohort_accounting.csv`, `settings.json` | `scripts/meaning_variation_sensitivity.py` | the same embedding caches and `paper_per_model_rq_table.csv` |
| Run selection and input hashes | `outputs/paper_snapshot_provenance.json` | `scripts/export_paper_tables.py` | registries, raw store, benchmark CSVs |
| Modality template inventory | `outputs/modality_template_inventory.csv` / `.md` | `eval_utils.write_main_modality_template_inventory` | static; described in [`docs/experimental_setup.md`](experimental_setup.md) |

## Per-cell analysis artifacts (regenerable; in the Zenodo dataset record)

`scripts/generate_evaluation_analysis.py` writes one directory per cell and
run, `outputs/evaluation_<dataset>_<variant>_<run_id>/`. These directories are
not tracked; `scripts/rerun_all.py --config conf/rerun/final.yaml --only analysis`
regenerates them from the raw store.

| Artifact | Content |
| --- | --- |
| `uq_scores.csv` | One row per model, task, item and uncertainty method: the rows every table above is computed from |
| `metrics_summary.csv` / `.md`, `paper_results_table.md` | Per model x task x method metrics of that cell |
| `bootstrap_seed_ci.csv` / `.md` | Clustered bootstrap intervals of that cell |
| `acse_semantic_calibration.csv` / `.md`, `acse_semantic_normalized_scores.csv` | Meaning-variation scores and calibration |
| `task1_p_yes_by_modality.svg` | Task 1 acceptance by source modality |
| `qualitative_overcommitment_examples.csv` / `.md` | Illustrative high-confidence strengthening cases |
| `uq_method_inventory.csv` / `.md` | The uncertainty methods and their input signals |
| `provenance_manifest.json` | SHA-256 of every input the cell analysis read |
| `acse_semantic_mlx_*/` | The embedding cache the classifier and the sensitivity grid read |

## Robustness and diagnostics

| Element | Artifact | Notes |
| --- | --- | --- |
| Blind Task 3 preservation check | rows tagged `task=task3` and `task3_audit_mode=blind` in `uq_scores.csv`; pooled in `paper_per_model_rq_table.csv` | Diagnostic; never replaces Task 2 outputs. Declared-modality rows are anchored ablations. |
| `SHALL` wording variant | the `*_shall` cells of every table | Reported alongside `MUST` and pooled into the headline cells. |
| Prompt-wording sensitivity (May pilot) | `outputs/archive/2026-05-grouped-cohort/prompt_sensitivity_summary.csv`, `task2_prompt_sensitivity_summary.csv` | Archived campaign; not cited by the revised manuscript. |
| External chat-service probe (May) | `outputs/archive/2026-05-grouped-cohort/external_ai_service_probe/` | Archived campaign; not cited by the revised manuscript. |

## Construct validity gate

Weak-intent paper claims (`nice_to_have` results) are gated on a completed
construct-validity table:

- Input: `docs/weak_modality_construct_review.csv`
- Pass condition: every weak template is marked weaker than `SHOULD/recommended` by both reviewer slots.
- The analysis script refuses to write paper-facing artifacts if this gate is incomplete unless `--skip-construct-review-check` is set (for diagnostic local runs only).

Human validation is complete, as confirmed by the author on 2026-09-04, and
will be repeated before submission. The original LLM-assisted judgments remain
separately identified. See [validation review](validation_review.md) for the
scope and the wording checks' limitations; no independent two-human agreement
is claimed.

## Confidence-scale contract

Every paper-facing claim depends on the v2 confidence contract:
`confidence ∈ [0.0, 1.0]`, prompt version in `{v2-conf01, v2-instructor-conf01}`,
raw records tagged `confidence_scale=0_1`. The analysis script fails closed if
it encounters mixed-scale rows.

## Tracing a single number back to its source

Given a macro in `outputs/paper_numbers.tex`:

1. Its comment names the scope; the README's Source column names the CSV.
   Open the row for the model and cell (`model=all`, `dataset=all` is the
   pooled row) in `outputs/paper_per_model_rq_table.csv` or the CSV named.
2. `outputs/paper_snapshot_provenance.json` lists, for that model and cell,
   the `run_id` and the SHA-256 of the registry, raw store and benchmark CSV
   that were read.
3. In the Zenodo bundle, `data/processed/model_outputs_raw*.parquet` holds the
   raw answers of that `run_id`, and `data/processed/logs/<run_id>.transcript.jsonl`
   the exact request and response bodies.
4. The benchmark item (capability, source condition, prompt) is the row in
   `data/processed/benchmark_items*.csv` with the same `item_id`; its prompt
   hash and seed list are in `outputs/benchmark_manifest*.json`.

Every paper-facing number therefore has a short, verifiable chain back to a
hashed input.
