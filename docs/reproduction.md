# Reproduction Guide

This is the command-first path for reproducing the publication artifacts. The notebooks are useful for inspection, but these scripts are the canonical interface for provider runs and final analysis.

## Task → Command Cheat-sheet

| Goal | Command |
| --- | --- |
| One-time env setup | `uv sync --group dev --locked` |
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
| Export cross-cell paper tables | `.venv/bin/python scripts/export_paper_tables.py` |
| Regenerate the manuscript's number macros | `.venv/bin/python scripts/export_paper_numbers.py` (add `--output manuscript/numbers.tex` to write the paper; the default writes `outputs/paper_numbers.tex` and prints a macro-level diff) |
| Print the Task 3 queue without running it | `TASK3_DRY_RUN=1 bash scripts/enqueue_task3_runs.sh` |
| Regenerate the modality template inventory | see [`docs/experimental_setup.md`](experimental_setup.md) §2.2 |

`scripts/reproduce.sh` is a thin convenience wrapper around the canonical CLIs documented below; use the raw commands directly when you need fine-grained control.

## Canonical Example Cell

Every command example on this page and in [`docs/evaluation.md`](evaluation.md) uses **one** cell, so the two pages cannot drift:

| Field | Value |
| --- | --- |
| Profile | `zai` |
| Model | `glm-5.1` |
| Dataset | `mlm_tapt` |
| Variant | `must` |
| Task 3 audit mode | `blind` |

These are also the defaults built into `scripts/reproduce.sh`. **The default cell is a paid provider endpoint.** Override it with environment variables rather than editing the script:

| Variable | Default | Meaning |
| --- | --- | --- |
| `RE_UQ_CONFIG` | `run_configs/current_run.json` | Run config path. |
| `RE_UQ_PROFILE` | `zai` | Provider profile id. |
| `RE_UQ_MODEL` | `glm-5.1` | Model id. |
| `RE_UQ_DATASET` | `mlm_tapt` | `nice` or `mlm_tapt`. |
| `RE_UQ_VARIANT` | `must` | `must` or `shall`. |

```bash
RE_UQ_PROFILE=local_llama_cpp RE_UQ_MODEL=qwen/qwen3.5-9b RE_UQ_DATASET=nice \
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

Edit `run_configs/current_run.json` for the provider, model, endpoint, concurrency, and structured-output mode. This file is ignored by Git because it is machine- and credential-specific.

The tracked example configs define Task 1 and Task 2 as the primary benchmark tasks. Task 3 is run separately after a complete Task 2 run.

**A rerun differs from the archived runs in labels, not in prompts.** The
configs shipped with the first release could not have produced the cohort
(`kit.gemma4-31b-it` ran under a `kit_toolbox` profile, not `institutional_llm`;
the `zai` profile listed 3 of the 5 GLM models; every profile said
`batch_size: 8` where the runs used 16). That is fixed, but the run labels still
differ: the archived raw rows carry `prompt_version "v1"` and `run_group_id
provider-matrix-2026-05`, while the current configs write `v2-conf01` and
`provider-matrix-v2-2026-05`. The prompt text is unchanged — the batch prompt
hashes in the archived rows match the current builder — so treat the difference
as bookkeeping. Details in
[`docs/experimental_setup.md`](experimental_setup.md) §5.3.

## 3. Task 1 And Task 2 Runs

Smoke test one provider/model/dataset cell:

```bash
.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile zai \
  --model glm-5.1 \
  --dataset mlm_tapt \
  --task both \
  --mode smoke
```

Run the full Task 1 + Task 2 cell after smoke checks pass:

```bash
.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile zai \
  --model glm-5.1 \
  --dataset mlm_tapt \
  --task both \
  --mode full
```

Run one task at a time when isolating failures:

```bash
.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile zai \
  --model glm-5.1 \
  --dataset mlm_tapt \
  --task task1 \
  --mode full
```

```bash
.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile zai \
  --model glm-5.1 \
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
  --model glm-5.1 \
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
  --model glm-5.1 \
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
  --model glm-5.1 \
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
  --model glm-5.1 \
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

## 7. Common CLI Flags

The runner and analysis CLIs share a few operational flags:

| Flag | Available on | Effect |
| --- | --- | --- |
| `--log-level` | `run_experiment_from_config.py`, `run_task3_verification_from_config.py` | Console verbosity. Per-run logs are always written to `data/processed/logs/<run_id>.log`. |
| `--dry-run` | `run_experiment_from_config.py` | Plan the requests, print what would be sent, exit without calling the provider. Use it before every paid full run. |
| `--all-models` | `run_experiment_from_config.py` | Iterate every model in the selected profile instead of requiring `--model`. |
| `--fake-completion` | both runners | Synthesize deterministic local responses; no HTTP. |

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
| `batch_order` | profile and run | `grouped` (consecutive request indices, the policy of every reported run) or `shuffled` — a constrained shuffle that never places two source variants of one seed in the same batch, derived deterministically from the run seed and stable across resume (the ablation; see [`TODO.md`](../TODO.md) section A). |
| `batch_size` | profile | Benchmark items per request. 16 in every official-cohort run; `1` gives true single-item delivery. |
| `item_context` | run | `bare` (every reported run) or `document` — Task 2 items are shown with their document, section, author marker and neighbouring requirements. Only the `pure` dataset carries that context; see [`context_ablation.md`](context_ablation.md). |

### Example Provider Profiles

| Profile | JSON mode | Structured output | Seed sent |
| --- | --- | --- | --- |
| `zai`, `kit_toolbox` | yes | `json_object` | yes |
| `institutional_llm` | yes | `json_schema` | yes |
| `local_llama_cpp`, `ollama_local` | no | `none` | yes / n.a. |
| `openai`, `mistral` | yes | `json_object` | yes |
| `google_gemini` | yes | `json_object` | `send_seed: false` — same reason. |

Every profile targets an OpenAI-compatible chat-completions endpoint; that is the only provider integration the pipeline supports, and new families are added as profile files (see `docs/configuration.md`). `ollama_local` gives a fully offline replication path. Running the new families is [`TODO.md`](../TODO.md) section C.

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
| `plot_embedding_diagnostic_figure_v2.py` | 3. figure | **current — the paper figure** | Single horizontal AUROC bar chart: context probes (input commitment level, dataset) vs strengthening-detection probes. |
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
