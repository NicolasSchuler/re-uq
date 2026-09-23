# Changelog

## v2.0.1 (2026-09-23): replication-package fixes before resubmission

Every reported number is unchanged.

- Exporter: per-model table cells use `\estcell` (estimate, interval, n/N),
  and the Z.AI group is labelled "Commercial"; `outputs/paper_numbers.tex`
  regenerated.
- Benchmark grammar check with LanguageTool 6.8
  (`scripts/check_benchmark_grammar.py`, `outputs/benchmark_grammar_check.*`).
- Tier 1 works on a fresh clone: the fake smoke run falls back to the example
  run config, and the number exporter no longer warns about checkout mtimes.
- Tier 2: working refresh command and unpacking instructions for the Zenodo
  archives, and `scripts/verify_paper_numbers.py` to check the result tables
  without MLX.
- `seeds_review_mlm_tapt.csv` keeps the source text only for the 180 selected
  requirements.
- DOIs in the README, `CITATION.cff` and `.zenodo.json`.
- Docs aligned with the final campaign: capability-clustered intervals,
  final answer-length figures, seeds and served models, current cohort and
  configuration, the authors' review of the benchmark; the PURE revision
  decisions are labelled `accepted`.
- `docs/experimental_setup.md` lists the served model file of every local
  model and how to add your own model; `docs/experiment_runbook.md` gives the
  requests, tokens, wall clock and hardware of the reported campaign.
- The t-SNE figure's wording-check panel draws its points in shuffled order.

## v2.0.0 (2026-09-16): revision for the Journal of Systems and Software

- Final campaign `manuscript-final` (2026-09-11 to 2026-09-16): nine models
  from five families, one item per request, one deterministic and five sampled
  answers per item with recorded seeds, JSON-schema decoding for local models,
  served model recorded per response.
- Three ablations (request composition at 4 and 16 items per request against
  a single-item reference, document context on 180 PURE capabilities,
  weak-intent phrasing over all NICE capabilities) and a meaning-variation
  sensitivity grid.
- The paper tables, the macro file, provenance, ablation summaries, probe
  summaries and figures are tracked under `outputs/`; raw outputs, registries,
  transcripts and embedding caches are archived in a Zenodo dataset record.
- `scripts/rerun_all.py` runs generation, audits, ablations and every analysis
  step from one configuration file.
- Repository hygiene: engineering records under `docs/internal/`, the May 2026
  snapshots archived, machine-specific details removed, link and README-number
  tests in CI.

## v1.0.0 (2026-06-19): original submission

- Six models (five GLM models on the Z.AI endpoint and `kit.gemma4-31b-it` on
  the KIT endpoint), sixteen benchmark items per request, run group
  `provider-matrix-2026-05`. Its summaries are kept under
  `outputs/archive/2026-05-grouped-cohort/` and
  `data/processed/archive/2026-05-grouped-cohort/`.
