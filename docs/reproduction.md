# Reproduction Guide

This is the command-first path for reproducing the publication artifacts. The notebooks are useful for inspection, but these scripts are the canonical interface for provider runs and final analysis.

For the prepared campaign, follow the [experiment rerun runbook](experiment_runbook.md)
for ordered setup, credentials, configuration, smoke checks, launch, monitoring,
resume and result review. This page is the detailed command reference.

The final campaign is explicitly `--config conf/rerun/final.yaml`: size 1,
five stochastic samples plus one deterministic answer, and the corrected
AI-reviewed PURE inputs. Older commands and snapshots on this page document
historical campaigns; do not resume them under changed profiles or treat their
grouped outputs as single-item evidence.

## The Whole Rerun, One Command

Everything below is still the canonical, fine-grained interface. For a full
rerun -- cohort Task 1+2, the blind Task 3 audits, the ablations, then every
table, macro and figure -- there is one driver that walks them in order:

```bash
export ZAI_API_KEY=...            # 1. credentials
export LLAMA_API_KEY=...
$EDITOR conf/profile/local_llama_cpp.yaml   # 2. local base_url + models
$EDITOR conf/profile/zai.yaml               #    hosted models
.venv/bin/python scripts/rerun_all.py --config conf/rerun/final.yaml   # 3. run
```

`conf/rerun/final.yaml` (the reported campaign; `default.yaml` is the driver's
built-in default) says which profiles are the official cohort, which
are reported separately as local, and which models carry the ablations.
Everything else about a run stays in `conf/profile/<id>.yaml`.

| Flag | Effect |
| --- | --- |
| `--dry-run` | Print every command the driver would run, change nothing. |
| `--fake-completion` | Verify the whole chain with synthesized answers, in the smoke tree, for free. The MLX-backed steps and the macro file are skipped (they need embeddings a fake run does not produce). |
| `--only <stage>` | `preflight`, `cohort`, `task3`, `ablations`, `analysis`. Repeatable. |

It is resumable: `outputs/rerun/<run_group_id>/state.json` records each cell's
status, run ID, and resolved configuration. Re-invoking the same command skips
complete cells and resumes unfinished ones, including the phrasing probe.
Failed transport or parsing attempts can be retried; completed item records
are reused. A changed configuration requires a new `run_group_id` (or a new
`--state` path), so old pre-fix runs cannot silently satisfy the new plan.

All generation stages finish for one model before the next model starts:
Task 1/2 → blind audits of those exact outputs → that model's ablations.
The local endpoint must route/load by the requested model name. Both profiles
contribute to pooled estimates and retain separate per-model table groups.
Each ablation uses the **first model in each profile's list** unless explicit
`models` are supplied in `conf/rerun/default.yaml`; reorder the lists to choose
representatives. See [ablation proposals](internal/ablation_proposals.md).

The driver forces request transcripts, events, and progress files on:

| Artifact | Location |
| --- | --- |
| Stage stdout/stderr, command and exit status (appended) | `outputs/rerun/logs/*.log` |
| Per-run log | `data/processed/logs/<safe_run_id>.log` |
| Complete request payloads, response bodies, retries and HTTP error bodies | `data/processed/logs/<safe_run_id>.transcript.jsonl` |
| Resolved configuration (key environment-variable names only) | `data/processed/logs/<safe_run_id>.resolved.yaml` |
| Parsed item rows and error records | `data/processed/model_outputs_raw*.jsonl` |
| Events, progress, registry | `data/processed/run_events*.jsonl`, `run_progress_live*.csv`, `run_registry*.csv` |
| Exported CSVs and generated LaTeX macros | `outputs/paper_*.csv`, `outputs/paper_numbers.tex` |
| Candidate regenerated figures | `outputs/rerun/figures/` |

The driver uses exact recorded run IDs for audits, exports, embedding caches,
and ablation comparisons. The manuscript's RQ tables consume generated row
macros, but its numbers file is not overwritten automatically: integrate the
fresh export when reviewing the rerun results. A zero denominator or undefined AUROC remains missing, never
converted into a success rate or a chance score.

Run `--fake-completion` to check execution without credentials. Its state,
generated config, analysis, and stage logs live under `outputs/smoke/`; raw
records use the smoke tree and per-run logs remain named by smoke run IDs.
This checks execution and table exports, not real providers or neural
embedding results. `--dry-run` needs no API keys and writes no state.

## Task → Command Cheat-sheet

| Goal | Command |
| --- | --- |
| One-time env setup | `uv sync --group dev --locked` |
| Full rerun (everything) | `.venv/bin/python scripts/rerun_all.py --config conf/rerun/final.yaml` (see above) |
| Re-derive every table from the archived raw outputs | unpack both archives of the Zenodo dataset record into the repository (commands in the README, tier 2), then `.venv/bin/python scripts/rerun_all.py --only analysis --refresh-analysis --state outputs/rerun/manuscript-final/state.json` (Apple Silicon with MLX; several hours) |
| Check the result tables without MLX (any OS) | after unpacking the raw archive, `.venv/bin/python scripts/verify_paper_numbers.py --raw-check` |
| Sanity-check pipeline without API access | `bash scripts/reproduce.sh smoke-fake` (uses `--fake-completion`) |
| Fake-completion Task 3 smoke | `bash scripts/reproduce.sh smoke-fake-task3` |
| Fake-completion analysis smoke | `bash scripts/reproduce.sh smoke-fake-analysis` |
| Smoke test a provider/model cell | `bash scripts/reproduce.sh smoke` |
| Run Task 1 + Task 2 (main) | `bash scripts/reproduce.sh full` |
| Run Task 3 diagnostic | `bash scripts/reproduce.sh task3 --source-run-id RUN_ID` |
| Generate paper-facing analysis | `bash scripts/reproduce.sh analysis --run-id RUN_ID --task3-run-id TASK3_RUN_ID` |
| Monitor a live run | `.venv/bin/python scripts/show_run_progress.py --dataset mlm_tapt --run-id RUN_ID --watch 30` |
| Compare completed cells | `.venv/bin/python scripts/compare_run_matrix.py --config run_configs/current_run.json --dataset mlm_tapt` |
| Document-context ablation table | `.venv/bin/python scripts/compare_context_ablation.py` (after `+experiment=context_ablation`; see [`context_ablation.md`](context_ablation.md)) |
| Batching ablation table | `.venv/bin/python scripts/compare_batching_ablation.py` (after the three `+experiment=batching_ablation` arms) |
| Weak-phrasing probe | `.venv/bin/python scripts/run_weak_modality_probe.py --config run_configs/current_run.json --profile zai --model glm-5.3 --dataset nice --mode full` |
| Resolve Task 3 source runs | `.venv/bin/python scripts/task3_sources.py --config run_configs/current_run.json` |
| Export cross-cell paper tables | `.venv/bin/python scripts/export_paper_tables.py` (the driver passes `--run-group-id manuscript-final` and the nine models; the archived May runs are `--run-group-id provider-matrix-2026-05` with the six models in `export_paper_tables.ARCHIVED_COHORT`) |
| Recompute request- and seed-clustered CIs without touching `outputs/` | `.venv/bin/python scripts/export_paper_tables.py --output-dir /tmp/reuq-cluster-ci` (see §6) |
| Regenerate the manuscript's number macros | `.venv/bin/python scripts/export_paper_numbers.py --strict` (add `--output manuscript/numbers.tex` to write the paper; the default writes `outputs/paper_numbers.tex` and prints a macro-level diff). `--strict` refuses to write when a macro would carry a non-finite value or a consistency check disagrees, so the manuscript never receives a `nan` |
| Print the Task 3 queue without running it | `TASK3_DRY_RUN=1 bash scripts/enqueue_task3_runs.sh` |
| Regenerate the modality template inventory | see [`docs/experimental_setup.md`](experimental_setup.md) §2.2 |

`scripts/reproduce.sh` is a thin convenience wrapper around the canonical CLIs documented below; use the raw commands directly when you need fine-grained control.

## Canonical Example Cell

Every command example on this page and in [`docs/evaluation.md`](evaluation.md) uses **one** cell, so the two pages cannot drift:

| Field | Value |
| --- | --- |
| Profile | `zai` |
| Model | `glm-5.3` |
| Dataset | `mlm_tapt` |
| Variant | `must` |
| Task 3 audit mode | `blind` |

These are also the defaults built into `scripts/reproduce.sh`. **The default cell is a paid provider endpoint.** Override it with environment variables rather than editing the script:

| Variable | Default | Meaning |
| --- | --- | --- |
| `RE_UQ_CONFIG` | `run_configs/current_run.json` | Run config path. |
| `RE_UQ_PROFILE` | `zai` | Provider profile id. |
| `RE_UQ_MODEL` | `glm-5.3` | Model id. |
| `RE_UQ_DATASET` | `mlm_tapt` | `nice` or `mlm_tapt`. |
| `RE_UQ_VARIANT` | `must` | `must` or `shall`. |

```bash
RE_UQ_PROFILE=local_llama_cpp RE_UQ_MODEL=qwen3.5-9b RE_UQ_DATASET=nice \
  bash scripts/reproduce.sh smoke
```

Substitute your own profile/model in the raw commands below; nothing in the pipeline depends on the example values. Runs that will end up in the paper must use a cohort model (see [`docs/experimental_setup.md`](experimental_setup.md)).

## 1. Environment

```bash
uv sync --group dev --locked
.venv/bin/python -m unittest discover -s tests -v
```

If the lock file needs to be refreshed locally, use `uv sync --group dev` and commit the resulting `uv.lock` only when the dependency change is intentional.

For a no-credentials sanity check before configuring a provider, see [`docs/reproduction_smoke.md`](reproduction_smoke.md).

## 2. Configure A Provider Matrix

```bash
cp run_configs/full_matrix.example.json run_configs/current_run.json
```

Edit `run_configs/current_run.json` for the provider, model, endpoint, concurrency, and structured-output mode. This file is ignored by Git because it is machine- and credential-specific. Real runs need your own model access: an API key for a hosted provider or your own OpenAI-compatible server. The authors' keys and infrastructure are not part of the package.

The tracked example configs define Task 1 and Task 2 as the primary benchmark tasks. Task 3 is run separately after a complete Task 2 run.

**The reported campaign versus the archived runs.** The reported campaign
(`manuscript-final`, September 2026) is fully described by
`conf/rerun/final.yaml` and the two profiles it names, and every run id is in
`outputs/rerun/manuscript-final/state.json`. The archived May runs (run group
`provider-matrix-2026-05`, prompt version `v1`, sixteen items per request)
were produced from configs the first release did not ship faithfully; their
raw rows remain in the raw store for the record. The reported campaign sends
reproducible, distinct seeds for stochastic repetitions, persists the
embedding configuration, and records complete requests and responses.
The deterministic request uses the configured seed; stochastic repetition
indices 0–4 use seed+1 through seed+5. Provider support for seeds remains
provider-dependent. Details in
[`docs/experimental_setup.md`](experimental_setup.md) §5.3.

## 3. Task 1 And Task 2 Runs

Smoke test one provider/model/dataset cell:

```bash
.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile zai \
  --model glm-5.3 \
  --dataset mlm_tapt \
  --task both \
  --mode smoke
```

Run the full Task 1 + Task 2 cell after smoke checks pass:

```bash
.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile zai \
  --model glm-5.3 \
  --dataset mlm_tapt \
  --task both \
  --mode full
```

Run one task at a time when isolating failures:

```bash
.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile zai \
  --model glm-5.3 \
  --dataset mlm_tapt \
  --task task1 \
  --mode full
```

```bash
.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile zai \
  --model glm-5.3 \
  --dataset mlm_tapt \
  --task task2 \
  --mode full
```

Resume a partial run by reusing the same run ID and the same selection it was
started with:

```bash
.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile zai \
  --model glm-5.3 \
  --dataset mlm_tapt \
  --task task2 \
  --mode resume \
  --run-id RUN_ID
```

`--mode resume` refuses to start unless the run ID already exists in the run
registry for that dataset/variant with the same provider, profile, model,
dataset, variant, and task selection, so a mistyped ID fails instead of quietly
creating a second run that only looks resumed. A resumed run keeps the
`started_at_utc` and provenance `notes` of its first attempt; each resume is
appended to `data/processed/logs/RUN_ID.resume.json` instead. If the run is
aborted (a crash, or Ctrl-C), its registry row is reconciled to `failed` or
`interrupted` rather than being left at `running`.

While a cell is running, the runner holds an advisory lease at
`data/processed/logs/RUN_ID.<cell>.lease.json`. A second runner that finds a
live lease (owning process still alive, recent heartbeat) refuses the cell
instead of duplicating paid requests; a lease left by a dead process is taken
over with a warning.

Resume reuses a cached raw row only when the row's `job_config_sha` matches the
sha the resumed job would produce, so a config change silently re-requests the
affected items instead of mixing two request shapes into one run id. That hash
is now at **version 3**: on top of the request parameters it already covered it
now also covers `batch_size`, `batch_order`, `fallback_batch_size`, and a hash
of the batch wrapper text. Rows written under version 2 therefore do not match
and are re-requested on the next resume — expected, and the reason a resume of
an old run can look like a full rerun.

## 4. Monitor And Compare Runs

Monitor a live run:

```bash
.venv/bin/python scripts/show_run_progress.py \
  --dataset mlm_tapt \
  --run-id RUN_ID \
  --model glm-5.3 \
  --profile zai \
  --watch 30
```

Compare complete runs in the configured matrix:

```bash
.venv/bin/python scripts/compare_run_matrix.py \
  --config run_configs/current_run.json \
  --dataset mlm_tapt \
  --exclude-model-prefix azure.
```

The official paper cohort excludes private Azure-hosted `azure.*` rows. Those raw
registry entries can remain useful diagnostics, but do not count them in
paper-facing model aggregates.

## 5. Task 3 Self-Audit Diagnostic

Task 3 audits deterministic Task 2 extracted text with a source-grounded blind prompt. It is not independent verification and does not repair or overwrite Task 2 outputs. Declared-modality Task 3 runs are anchoring ablations, not official Task 3 results.

Smoke test Task 3 against a complete Task 2 source run:

```bash
.venv/bin/python scripts/run_task3_verification_from_config.py \
  --config run_configs/current_run.json \
  --profile zai \
  --model glm-5.3 \
  --dataset mlm_tapt \
  --source-run-id RUN_ID \
  --audit-mode blind \
  --mode smoke
```

Run the full Task 3 diagnostic:

```bash
.venv/bin/python scripts/run_task3_verification_from_config.py \
  --config run_configs/current_run.json \
  --profile zai \
  --model glm-5.3 \
  --dataset mlm_tapt \
  --source-run-id RUN_ID \
  --audit-mode blind \
  --mode full
```

Add `--allow-partial-source` when the source Task 2 run was a `--mode smoke`
run: source completeness is checked against the full 720-item benchmark, so a
smoke source is rejected without it. A paper-facing Task 3 run never needs the
flag.

Task 3 writes local-only raw outputs to `data/processed/model_outputs_raw_task3_verification*.jsonl`, run-specific item CSVs under `data/processed/task3_verification_items/`, plus Task 3 registry/progress files. Legacy flat `data/processed/task3_verification_items*.csv` files from earlier runs are also local-only. Existing Task 3 rows without `task3_audit_mode=blind` are legacy anchored diagnostics.

## 6. Generate Final Analysis

Complete `docs/weak_modality_construct_review.csv` before generating paper-facing weak-intent claims. The analysis command fails by default if the construct-review gate is incomplete, run registries are not complete, confidence values violate the v2 `0.0-1.0` contract, or stale prompt rows are detected.

```bash
.venv/bin/python scripts/generate_evaluation_analysis.py \
  --dataset mlm_tapt \
  --variant must \
  --run-id RUN_ID \
  --task3-run-id TASK3_RUN_ID \
  --task3-audit-mode blind
```

The command writes a local analysis directory under `outputs/evaluation_<dataset>_<variant>_<run_id>/` unless `--output-dir` is provided. Expected artifacts include:

- `uq_scores.csv`
- `acse_semantic_normalized_scores.csv`
- `acse_semantic_calibration.csv`
- `acse_semantic_calibration.md`
- `metrics_summary.csv`
- `metrics_summary.md`
- `bootstrap_seed_ci.csv`
- `bootstrap_seed_ci.md`
- `paper_results_table.md`
- `task1_p_yes_by_modality.svg`
- `qualitative_overcommitment_examples.csv`
- `qualitative_overcommitment_examples.md`
- `uq_method_inventory.csv`
- `uq_method_inventory.md`
- `provenance_manifest.json`
- `result_notes_template.md`

See `docs/results_mapping.md` for how each artifact backs a specific paper figure, table, or claim.

`uq_scores.csv` includes the diagnostic `acse_semantic_entropy` method when stochastic samples are available. This row clusters the five generated answer texts and should be interpreted as a semantic-diversity ranking signal unless a held-out calibration split is used to set an accept/abstain threshold.
The analysis also writes `acse_semantic_normalized_scores.csv` and `acse_semantic_calibration.*`, which min-max normalize ACSE scores within each run/model/task/backend group and select empirical accept thresholds on a deterministic seed-level calibration split. Treat those thresholds as post hoc triage diagnostics unless the split and target risk level are declared before running the final analysis.
The Task 1/2 runner records the resolved embedding backend on every raw row.
`generate_evaluation_analysis.py` consumes that value and writes it to its
provenance manifest; `compute_acse_semantic_artifacts.py` then uses the manifest
value by default. Hydra runs select Qwen3-Embedding-0.6B unless `embedding=` is
overridden. Legacy JSON configs without embedding fields use the dependency-free
TF-IDF backend unless the environment selects MLX at run launch. For example:

```bash
.venv/bin/python scripts/run.py \
  embedding=qwen3_4b \
  profile=local_llama_cpp \
  model=qwen/qwen3.5-9b \
  dataset=mlm_tapt \
  variant=must \
  mode=full

.venv/bin/python scripts/generate_evaluation_analysis.py \
  --dataset mlm_tapt \
  --variant must \
  --run-id RUN_ID \
  --task3-run-id TASK3_RUN_ID \
  --task3-audit-mode blind
```

Install `mlx-embeddings` before analyzing an MLX-configured run. Pass
`--backend` and `--mlx-model` to `compute_acse_semantic_artifacts.py` only when
intentionally computing an additional cache ablation instead of the run's
persisted selection.

Use diagnostic flags such as `--allow-partial`, `--skip-registry-check`, `--skip-construct-review-check`, `--skip-manifest-check`, `--max-parse-failure-rate`, `--bootstrap-iterations`, or `--expected-stochastic-samples` only for local investigation, not for paper-ready results. (`--skip-manifest-check` bypasses the SHA-256 integrity check against `outputs/benchmark_manifest*.json`.)

### Recomputing Both Bootstrap Intervals

The manuscript reports capability-clustered intervals; the tables also carry a
second interval whose cluster `bootstrap_ci_cluster_field` names
([`aggregation.md`](aggregation.md) §6).
To recompute both from the local raw rows without touching the committed
snapshots, point the exporter at a scratch directory:

```bash
.venv/bin/python scripts/export_paper_tables.py \
  --output-dir /tmp/reuq-cluster-ci \
  --bootstrap-samples 1000
column -s, -t < /tmp/reuq-cluster-ci/paper_headline_bootstrap_ci.csv
```

`paper_headline_bootstrap_ci.csv` carries `ci_low` / `ci_high` (clustered on
`ci_cluster_field`, the item under single-item requests), `seed_ci_low` /
`seed_ci_high` (clustered on the capability, the intervals the manuscript
reports), and `n_numerator` / `n_denominator`; `paper_per_model_headline.csv`
carries the same pair per model. On the reported cohort (9 models x 4 cells,
24,828 readable Task 2 answers, RNG seed `20260518`, 1000 resamples) the
tracked file reads:

| Metric | Point | Item-clustered 95% CI | Capability-clustered 95% CI |
| --- | --- | --- | --- |
| Strict text strengthening | 26.01% (6459/24828) | [23.74%, 28.27%] | [25.88%, 26.17%] |
| Broad text strengthening | 27.88% (6923/24828) | [25.65%, 30.05%] | [27.64%, 28.17%] |

The archived May cohort (6 models x 4 cells, 16,448 readable rows, 1080
request clusters) reproduced as strict 8.58% (1412/16448) and broad 13.79%
(2268/16448) with request-clustered intervals [7.75%, 9.52%] and
[12.43%, 15.21%]; see `outputs/archive/2026-05-grouped-cohort/`.

Nothing is written under `outputs/`, so this is safe to run against a clean
checkout. Drop `--output-dir` only when you intend to regenerate the shipped
snapshots.

## 7. Common CLI Flags

The runner and analysis CLIs share a few operational flags:

| Flag | Available on | Effect |
| --- | --- | --- |
| `--log-level` | `run_experiment_from_config.py`, `run_task3_verification_from_config.py` | Console verbosity. Per-run logs are always written to `data/processed/logs/<run_id>.log`. |
| `--dry-run` | `run_experiment_from_config.py` | Plan the requests, print what would be sent, exit without calling the provider. Use it before every paid full run. |
| `--all-models` | `run_experiment_from_config.py` | Iterate every model in the selected profile instead of requiring `--model`. |
| `--fake-completion` | both runners | Synthesize deterministic local responses; no HTTP. |
| `--no-request-transcripts` | both runners | Skip the per-request transcript sidecar `data/processed/logs/<run_id>.transcript.jsonl` (on by default; see [`configuration.md`](configuration.md) §6). |

Environment variables recognised by the runners and the wrapper:

| Variable | Effect |
| --- | --- |
| `RE_UQ_MODE` | Run mode for the `task3` wrapper subcommand (default `full`). |
| `RE_UQ_RUN_ID` | Explicit run id for `task3` / `analysis` source lookups. |
| `RE_UQ_SMOKE_TREE=1` | Force reads and writes into the `data/processed/smoke/` tree. |
| `TASK3_SKIP_PROFILES` | Comma-separated profile ids to skip in `scripts/enqueue_task3_runs.sh`. |
| `TASK3_DRY_RUN` | Print the Task 3 queue without executing it. |

## 8. Run Config Must Cover The Whole Cohort

`scripts/enqueue_task3_runs.sh` derives its Task 3 matrix from the run config rather than from a hard-coded list. `run_configs/current_run.json` must therefore list **every** cohort profile, model, and benchmark variant you intend to audit; a model missing from the config is silently absent from the Task 3 queue. Check the config against the cohort table in [`docs/experimental_setup.md`](experimental_setup.md) before enqueuing, and use `TASK3_DRY_RUN=1` to print the resolved matrix first.

### Per-Profile And Run-Level Knobs

| Knob | Scope | Meaning |
| --- | --- | --- |
| `seed` | profile and run | Request seed for reproducibility. |
| `send_seed` | profile | Whether the seed is actually put on the wire. Set `false` for providers whose OpenAI-compatible layer ignores it. |
| `max_retries` | profile | Request retry budget. |
| `batch_order` | profile and run | `grouped` (consecutive request indices; the archived policy and one ablation arm) or `shuffled` — a constrained shuffle that never places two source variants of one seed in the same batch, derived deterministically from the run seed and stable across resume (the request-composition ablation). |
| `batch_size` | profile | Benchmark items per request. `1` in the reported campaign; 4 and 16 in the request-composition ablation; 16 in the archived runs. |
| `item_context` | run | `bare` (every reported run) or `document` — Task 2 items are shown with their document, section, author marker and neighbouring requirements. Only the `pure` dataset carries that context; see [`context_ablation.md`](context_ablation.md). |

### Example Provider Profiles

| Profile | JSON mode | Structured output | Seed sent |
| --- | --- | --- | --- |
| `zai`, `kit_toolbox` | yes | `json_object` | yes |
| `institutional_llm` | yes | `json_schema` | yes |
| `local_llama_cpp` | no | `json_schema` | yes |
| `ollama_local` | no | `none` | n.a. |
| `openai`, `mistral` | yes | `json_object` | yes |
| `google_gemini` | yes | `json_object` | `send_seed: false` — same reason. |

Every profile targets an OpenAI-compatible chat-completions endpoint; that is the only provider integration the pipeline supports, and new families are added as profile files (see `docs/configuration.md`). `ollama_local` gives a fully offline replication path. Running further families is not part of the reported campaign.

### Registry Columns

The run registry gained per-run quality columns: `batch_order`, `parse_status_histogram`, `parse_repairs`, `retry_total`, `truncated_records`, `latency_p50_s`, `latency_p95_s`, `usage_completion_tokens`. A `truncated` response counts as a **parse failure**, so a run with a low `parse_success_rate` and a nonzero `truncated_records` is a token-budget problem, not a prompt problem. `parse_repairs` counts responses the tolerant parser had to repair before they validated; those are `ok` in `parse_status_histogram`, so this is the only column that shows them.

A raw JSONL file records physical **attempts**, and a resume re-requests a
failed cell, so one planned observation can leave two rows. The registry
reports both readings and says which is which: `observed_records`,
`parse_success_rate`, and the two coverage columns count **logical
observations** (a failed attempt superseded by a successful retry is one
record that parsed), while `observed_attempts`, `observed_api_calls`, and the
quality columns count **every attempt**. `scripts/show_run_progress.py` labels
its sections the same way.

## 9. Embedding And Figure Scripts

These are diagnostic/figure scripts run **after** a complete analysis. They read cached raw outputs and the ACSE embedding cache; they do not change any metric.

| Script | Stage | Current? | Purpose |
| --- | --- | --- | --- |
| `compute_acse_semantic_artifacts.py` | 1. cache | current | Recomputes and caches sample embeddings, item-level ACSE rows, and metadata for completed Task 1/2 runs, so later scripts need no embedding backend. Writes `outputs/acse_semantic_artifact_manifest.*`. |
| `probe_acse_embedding_separability.py` | 2. probe | current | Baseline probe: do embeddings separate dataset, source modality, or drift labels? Primary condition is `requirement_only` (requirement text alone, PCA fitted inside each CV fold); the cached label-prefixed embeddings run only as the `prefixed_leakage_control` positive control. |
| `diagnose_embedding_separability.py` | 2. probe | current | Extended probe grid over text variant (`prefixed` vs `reqonly`), backend (`mlx` vs `tfidf`), and grouping (`seed` vs `item`); reports AUROC and AUPRC against the prevalence baseline. Feeds the paper figure. |
| `plot_embedding_diagnostic_figure_v2.py` | 3. figure | **current — the paper figure** | Three dot-and-interval panels: overall and within-source-modality strengthening detection, followed by auxiliary source-attribute prediction. |
| `plot_embedding_diagnostic_figure.py` | 3. figure | **superseded** | The earlier four-panel version (two t-SNE maps + AUROC bars + AUPRC lift). Kept for provenance; do not use in the paper. |
| `plot_embedding_diagnostic_tsne_supp.py` | 3. figure | current (supplement) | The two t-SNE projections dropped from the paper figure, kept for the replication package. Reuses the projection helpers of the superseded script so the maps stay pixel-identical. |
| `plot_acse_embedding_visualizations.py` | 3. figure | current (inspection) | Per-run projections of repeated Task 2 sample embeddings plus CSVs for inspecting individual drift cases. |
| `plot_acse_global_embedding_projection.py` | 3. figure | current (inspection) | One shared projection space across all completed runs. |

Order: cache -> probe -> plot. Every plotting script fails fast if the cache from step 1 is missing.

Embedding backend selection (model, override env var, TF-IDF fallback) is documented in [`docs/experimental_setup.md`](experimental_setup.md) §10.

## 10. Publication Verification Gate

Before treating the repository as publication-ready:

```bash
git status --short
git diff --check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m coverage run --branch --source=scripts -m unittest discover -s tests -v
.venv/bin/python -m coverage report -m
```

CI runs the same suite and prints the coverage report without enforcing a threshold; treat a coverage drop as a review signal, not as a gate.

Also inspect generated figures and tables manually. A figure or table is not paper-ready until its labels, clipping, row counts, and claim support have been checked.

## 11. Script Index

Every file under `scripts/`, in one line each; the docstring of each file says
more.

| Script | Role |
| --- | --- |
| `rerun_all.py` | The one-command driver: cohort generation, blind audits, ablations, then every analysis step, from a `conf/rerun/*.yaml` plan. |
| `run.py` | Hydra entry point for the Task 1/2/3 runners (`conf/` composition). |
| `run_experiment_from_config.py` | Task 1 and Task 2 provider-matrix runs from a JSON run config. |
| `run_task3_verification_from_config.py` | The blind Task 3 preservation check over a completed Task 2 run. |
| `run_weak_modality_probe.py` | The weak-phrasing probe: four weak templates over the benchmark capabilities. |
| `runner_args.py`, `runner_lifecycle.py` | The argument contract and the execute-cell lifecycle the two runners share. |
| `run_provenance.py` | Provenance helpers shared by the JSON and Hydra entry points. |
| `run_transcripts.py` | Per-request transcripts: what was sent and what came back. |
| `hydra_bridge.py` | Bridge between the Hydra `conf/` composition and the JSON run-config dictionary. |
| `task3_sources.py` | Which Task 2 run each Task 3 audit should read. |
| `enqueue_task3_runs.sh` | Queue blind Task 3 runs for completed Task 1/2 source runs. |
| `show_run_progress.py` | Read-only progress reporter for a live run. |
| `compact_raw_store.py` | Compact append-only raw JSONL into its Parquet sibling. |
| `structured_outputs.py` | Pydantic response models for the strict structured-output paths. |
| `build_pure_benchmark.py` | Build the `pure` document-context ablation dataset. |
| `export_benchmark_ground_truth.py` | Write `docs/benchmark_ground_truth.md` from the template code. |
| `populate_notebooks.py` | Generate the stripped companion notebooks. |
| `generate_evaluation_analysis.py` | Per-cell analysis: UQ scores, metrics, intervals, examples, provenance manifest. |
| `compute_acse_semantic_artifacts.py` | Compute and cache the sample embeddings and item-level meaning-variation rows per run. |
| `export_paper_tables.py` | The cross-cell paper tables and the snapshot provenance. |
| `aggregate_paper_headline_metrics.py` | The pooled headline rates the README quotes. |
| `export_paper_numbers.py` | The manuscript's `numbers.tex` macro file from the paper tables. |
| `export_commitment_transitions.py`, `commitment_transition_figure.py` | Commitment-transition counts and the TikZ figure that renders them. |
| `compare_batching_ablation.py`, `compare_context_ablation.py` | The request-composition and document-context ablation tables with paired intervals. |
| `compare_run_matrix.py` | Summarise completed runs of a run group (operational overview). |
| `meaning_variation_sensitivity.py` | The threshold x weight sensitivity grid of the meaning-variation score (appendix). |
| `diagnose_embedding_separability.py`, `probe_acse_embedding_separability.py` | The held-out embedding classifier grid and its baseline probe. |
| `plot_embedding_diagnostic_figure_v2.py`, `plot_embedding_diagnostic_tsne_supp.py` | The embedding-diagnostic figure and its t-SNE supplement; `plot_embedding_diagnostic_figure.py` is the superseded first version. |
| `plot_ablation_deltas.py` | The ablation-delta figure (batching, context, phrasing side by side). |
| `plot_acse_embedding_visualizations.py`, `plot_acse_global_embedding_projection.py` | Inspection projections of the cached embeddings (not in the paper). |
| `evaluate_external_ai_probe.py`, `export_external_ai_probe.py` | The archived May 2026 external chat-service probe: input bundle and scoring. |
| `eval_utils.py` | The shared module behind all of the above (configuration, paths, prompts, parsing, scoring, exports). |
| `reproduce.sh` | Thin wrapper with the canonical subcommands (`smoke-fake-all`, `full`, `task3`, `analysis`, `verify`). |
