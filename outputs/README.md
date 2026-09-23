# outputs/

The tracked results of the replication package, plus the benchmark reviews
that document how the inputs were built. Every file that carries a number is
written by the script named below; nothing here is edited by hand.

## What is tracked and where it comes from

| Group | Files | Written by |
| --- | --- | --- |
| Paper tables (`manuscript-final`) | `paper_task2_text_drift_metrics.csv`, `paper_text_drift_confidence_and_stability.csv`, `paper_per_model_modality_table.csv` / `.md`, `paper_per_model_modality_pooled.csv` / `.md`, `paper_per_model_headline.csv` / `.md`, `paper_per_model_rq_table.csv` / `.md`, `paper_headline_bootstrap_ci.csv`, `paper_snapshot_provenance.json`, `task3_audit_coverage.csv` | `scripts/export_paper_tables.py` |
| Headline aggregates | `paper_headline_metrics.csv` (pooled and macro-of-cells values for the README figures) | `scripts/aggregate_paper_headline_metrics.py` |
| Manuscript macros | `paper_numbers.tex` (every number the paper prints; its header hashes the tables it read) | `scripts/export_paper_numbers.py --strict` |
| Ablation summaries | `batching_ablation_summary.csv`, `_deltas.csv`, `.md`, `_provenance.json`; the same four for `context_ablation_summary` | `scripts/compare_batching_ablation.py`, `scripts/compare_context_ablation.py` |
| Commitment transitions | `commitment_transition_counts.csv`, `commitment_transition_accounting.csv`, `rerun/figures/commitment_transitions.tex` | `scripts/export_commitment_transitions.py` |
| Embedding classifier | `embedding_diagnostic/probe_grid_summary.csv` / `.md`, `probe_grid_folds.csv`, `manifest.json` | `scripts/diagnose_embedding_separability.py` |
| Meaning-variation sensitivity | `meaning_variation_sensitivity/cohort_accounting.csv`, `reproduction_check.csv`, `grid_auroc.csv`, `cluster_diagnostics.csv`, `settings.json` | `scripts/meaning_variation_sensitivity.py` |
| Weak-phrasing probe | `weak_modality_probe/<run>/weak_modality_probe_summary.csv` / `.md`, `weak_modality_text_deltas.csv`, for the nine runs named in `rerun/manuscript-final/state.json` | `scripts/run_weak_modality_probe.py` |
| Figures | `rerun/figures/ablation_deltas`, `embedding_diagnostic`, `embedding_diagnostic_tsne_supp` (`.pdf` and `.png`) | `scripts/plot_ablation_deltas.py`, `scripts/plot_embedding_diagnostic_figure_v2.py`, `scripts/plot_embedding_diagnostic_tsne_supp.py` |
| Campaign record | `rerun/manuscript-final/state.json` (every cell with its run id and the resolved configuration), `rerun/acse_selected_manifest.csv` (the embedding caches the analysis used) | `scripts/rerun_all.py`, `scripts/compute_acse_semantic_artifacts.py` |
| Benchmark construction | `benchmark_manifest*.json` (SHA-256 of every seed file, prompt and benchmark CSV), `benchmark_statements_review*`, `benchmark_grammar_check.*` (LanguageTool check of every rendered statement), `included_capabilities_review*`, `modality_template_inventory.*`, `weak_modality_template_sanity_check.*` | notebooks 01 and 02b via `scripts/populate_notebooks.py`; `eval_utils.write_main_modality_template_inventory`; `scripts/check_benchmark_grammar.py` |
| Archived | `archive/2026-05-grouped-cohort/` | see its README |

`paper_numbers.tex` reads only the paper tables, the provenance JSON and the
probe-grid summary; `tests/test_readme_numbers.py` regenerates it from them
and compares, and checks the README table against it.

## Local only, shipped in the Zenodo dataset record

Archived at [doi:10.5281/zenodo.22802294](https://doi.org/10.5281/zenodo.22802294).

Per-cell analysis directories `evaluation_<dataset>_<variant>_<run_id>/`
(`uq_scores.csv`, `metrics_summary.csv`, `bootstrap_seed_ci.csv`,
`provenance_manifest.json`, qualitative examples, the ACSE embedding caches),
`task3_audit_review.csv`, the `*_regenerated.csv` siblings,
`meaning_variation_sensitivity/per_item_scores.csv`,
`embedding_diagnostic/probe_grid_predictions.jsonl` and the requirement-only
embedding cache, `rerun/logs/`, and the `smoke/` tree. All of it regenerates
from the raw store with
`scripts/rerun_all.py --only analysis --refresh-analysis --state outputs/rerun/manuscript-final/state.json`
(README, tier 2).

## Provenance redactions in the tracked copies

Two substitutions distinguish the tracked provenance files from the ones the
campaign wrote on the author's machine; neither changes a number.

- `rerun/manuscript-final/state.json`: the local llama.cpp profile's
  `base_url` is the published placeholder (`http://localhost:9292/v1`). The
  run registries in the Zenodo bundle carry the address each run used.
- `paper_snapshot_provenance.json`, `meaning_variation_sensitivity/settings.json`
  and `rerun/acse_selected_manifest.csv`: paths are checkout-relative, which is
  what the exporters write since version 2.0.0.
