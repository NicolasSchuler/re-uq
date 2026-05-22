# data/

This directory holds dataset inputs and curated/derived data products.

## Layout

| Path | Tracked in Git? | Purpose |
| --- | --- | --- |
| `data/raw/` | No (except `.gitkeep`) | Local-only raw seed inputs (e.g. NICE/PROMISE CSV downloads). |
| `data/processed/` | Selectively | Curated benchmark CSVs, reviewed seed tables, compact metric summaries. Raw run outputs (`model_outputs_raw*.jsonl`), run registries, progress files, and UQ score CSVs are gitignored and remain local. |

## What is tracked under `data/processed/`

- `benchmark_items*.csv` — frozen benchmark inputs for each dataset/variant.
- `seeds_review*.csv`, `seeds_selected*.csv` — reviewed seed tables.
- `metrics_summary*.csv`, `bootstrap_seed_ci.csv` — compact paper-facing metric snapshots.
- `weak_modality_probe_items.csv` — robustness probe inputs.

Variant suffixes are documented in `docs/repository_layout.md`.

## What is local-only

Raw model outputs and run-execution bookkeeping stay on the machine that ran the experiment:

- `model_outputs_raw*.jsonl`
- `run_registry*.csv`, `run_progress*.csv`, `run_events*.jsonl`
- `uq_scores*.csv`

These are reproducibility evidence, not paper-shipping artifacts. They are promoted to Git only with a deliberate commit and a justification (see `docs/repository_hygiene.md`).
