# data/

## Sources and licences

| Dataset | Source and licence | In this repository |
| --- | --- | --- |
| NICE / PROMISE relabelled | [Zenodo record 14590935](https://zenodo.org/records/14590935), CC BY 4.0 (Rejithkumar and Anish) | `data/raw/PROMISE-relabeled-NICE.csv`, redistributed with attribution |
| `limsc/mlm-tapt-requirements` | [Hugging Face](https://huggingface.co/datasets/limsc/mlm-tapt-requirements); no licence declared | not redistributed. `seeds_review_mlm_tapt.csv` records the screening decision for all 39,139 source rows but keeps the source text only for the 180 selected requirements, which the benchmark needs. Notebook 00 fetches the dataset with `datasets` to rebuild the full table locally; its full-text candidate file is ignored by Git |
| PURE requirements XML | [Zenodo record 7118517](https://zenodo.org/records/7118517), CC BY 4.0 (Ferrari, Spagnolo and Gnesi) | `scripts/build_pure_benchmark.py` downloads the zip into `data/raw/` (ignored); the reviewed capabilities and their revisions are tracked |

## Layout

| Path | Tracked | Purpose |
| --- | --- | --- |
| `data/raw/` | the NICE CSV only | Raw inputs needed to rebuild the seed tables. Other downloads stay local. |
| `data/processed/` | `benchmark_items*.csv`, `seeds_review*.csv`, `seeds_selected*.csv`, `weak_modality_probe_items.csv` | Frozen benchmark inputs per dataset and wording variant, the reviewed seed tables (the `_pure` tables carry the `context_*` columns of the document-context ablation), and the weak-phrasing probe items. Variant suffixes are explained in `docs/repository_layout.md`. |
| `data/processed/archive/2026-05-grouped-cohort/` | yes | Metric snapshots of the original submission's campaign; see its README. |

## Local only, shipped in the Zenodo dataset record

Archived at [doi:10.5281/zenodo.22802293](https://doi.org/10.5281/zenodo.22802293).

Raw model outputs and run bookkeeping stay on the machine that ran the
campaign and are archived with the release:

- `model_outputs_raw*.jsonl` and their compacted `model_outputs_raw*.parquet`
  siblings (every campaign; `run_group_id` selects `manuscript-final`)
- `run_registry*.csv`, `run_progress*.csv`, `run_events*.jsonl` / `.parquet`
- `logs/<run_id>.log`, `<run_id>.transcript.jsonl` (complete request and
  response bodies) and `<run_id>.resolved.yaml`
- `task3_verification_items/` and `uq_scores*.csv`

Notebook execution also writes `metrics_summary*.csv` and
`bootstrap_seed_ci.csv` here; those are ignored, the archived May copies are
the tracked ones.

## Smoke runs

Fake-completion smoke runs write into a separate `data/processed/smoke/` tree,
using the same filenames as the real tree (registries, progress files, raw
JSONL). Smoke analysis output goes to `outputs/smoke/evaluation_<dataset>_<variant>_<run_id>/`.
Set `RE_UQ_SMOKE_TREE=1` to force the smoke tree. Smoke artifacts never mix
with real-run artifacts and are always local-only. See `docs/reproduction_smoke.md`.

## Run registry columns

Beyond the run identity and request parameters, `run_registry*.csv` records
per-run quality counters: `batch_order`, `item_context`,
`parse_status_histogram`, `retry_total`, `truncated_records`, `latency_p50_s`,
`latency_p95_s`, and `usage_completion_tokens`. A `truncated` response counts
as a parse failure, so `parse_success_rate` and `truncated_records` should be
read together: a low success rate with nonzero truncations is a token-budget
problem.

## What raw records contain

Raw files are append-only JSONL while a run is in flight;
`scripts/compact_raw_store.py` moves finished rows into a zstd Parquet sibling
(`<stem>.parquet`, roughly 40x smaller) and leaves an empty JSONL tail for later
appends. Every reader goes through `eval_utils.read_jsonl`, which returns the
Parquet rows followed by the tail, so the two halves are one logical file.

`model_outputs_raw*.jsonl` rows carry run and model identity, the request
parameters, the raw response text, the parsed JSON, and a parse status. Runs
of the final campaign additionally record `finish_reason`, token `usage_*`,
`served_model`, `system_fingerprint`, `request_seed`, `request_payload_sha`,
`system_prompt` (always empty: only a user message is sent),
`batch_variant_mix`, `response_chars`, and `requirement_word_count`, and can
report a `truncated` parse status. The archived May runs predate those fields;
the field-by-field table is in `docs/experimental_setup.md`.
