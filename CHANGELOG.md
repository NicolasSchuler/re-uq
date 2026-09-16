# Changelog

## v2.0.0 (unreleased, September 2026): revision for the Journal of Systems and Software

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
