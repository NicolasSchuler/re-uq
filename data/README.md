# data/

This directory holds dataset inputs and curated/derived data products.

## Layout

| Path | Tracked in Git? | Purpose |
| --- | --- | --- |
| `data/raw/` | Selectively | Raw seed inputs needed to rebuild curated benchmark tables. Additional local downloads remain untracked. |
| `data/processed/` | Selectively | Curated benchmark CSVs, reviewed seed tables, and compact diagnostic metric snapshots. Raw run outputs (`model_outputs_raw*.jsonl`), run registries, progress files, and UQ score CSVs are gitignored and remain local. |

## What is tracked under `data/processed/`

- `benchmark_items*.csv` — frozen benchmark inputs for each dataset/variant.
- `seeds_review*.csv`, `seeds_selected*.csv` — reviewed seed tables.
- `metrics_summary*.csv`, `bootstrap_seed_ci.csv` — compact metric snapshots; treat them as diagnostic/stale unless regenerated from a complete current run.
- `weak_modality_probe_items.csv` — robustness probe inputs (the four weak-intent phrasing templates).

Variant suffixes are documented in `docs/repository_layout.md`.

## What is local-only

Raw model outputs and run-execution bookkeeping stay on the machine that ran the experiment:

- `model_outputs_raw*.jsonl`
- `run_registry*.csv`, `run_progress*.csv`, `run_events*.jsonl`
- `logs/<run_id>.log` — per-run execution logs
- `uq_scores*.csv`

These are reproducibility evidence, not paper-shipping artifacts. They are promoted to Git only with a deliberate commit and a justification (see `docs/repository_hygiene.md`).

## Smoke Runs

Fake-completion smoke runs write into a separate `data/processed/smoke/` tree, using the same filenames as the real tree (registries, progress files, raw JSONL). Smoke analysis output goes to `outputs/smoke/evaluation_<dataset>_<variant>_<run_id>/`. Set `RE_UQ_SMOKE_TREE=1` to force the smoke tree. Smoke artifacts never mix with real-run artifacts and are always local-only. See `docs/reproduction_smoke.md`.

## Run Registry Columns

Beyond the run identity and request parameters, `run_registry*.csv` records per-run quality counters: `batch_order`, `parse_status_histogram`, `retry_total`, `truncated_records`, `latency_p50_s`, `latency_p95_s`, and `usage_completion_tokens`. A `truncated` response counts as a parse failure, so `parse_success_rate` and `truncated_records` should be read together: a low success rate with nonzero truncations is a token-budget problem.

## What Raw Records Contain

`model_outputs_raw*.jsonl` rows carry run/model identity, the request parameters, the raw response text, the parsed JSON, and a parse status. Newer runs additionally record `finish_reason`, token `usage_*`, `served_model`, `system_fingerprint`, `request_seed`, `request_payload_sha`, `system_prompt` (always empty: only a user message is sent), `batch_variant_mix`, `response_chars`, and `requirement_word_count`, and can report a `truncated` parse status. The archived runs predate those fields; the field-by-field before/after table is in `docs/experimental_setup.md`.
