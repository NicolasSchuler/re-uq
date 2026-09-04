# Smoke Reproduction Without Provider Credentials

This is a quick sanity-check path for reviewers or new collaborators who do not have a provider API key set up. It exercises the full request planner, runner, parser, and registry plumbing using a built-in `--fake-completion` mode that synthesizes deterministic responses locally instead of calling an LLM.

The fake-completion path is **not** a substitute for a real benchmark run. It confirms that the pipeline is wired correctly end-to-end. Paper-facing claims still require a real provider run.

Smoke runs write into a separate tree, so they never append to or overwrite the artifacts of a real run:

| Artifact | Smoke location |
| --- | --- |
| Registries, progress files, raw JSONL | `data/processed/smoke/` (same filenames as the real tree) |
| Task 3 verification items CSV | `data/processed/task3_verification_items/smoke/` |
| Analysis output | `outputs/smoke/evaluation_<dataset>_<variant>_<run_id>/` |
| Per-run log | `data/processed/logs/<run_id>.log` |
| Per-request transcript | `data/processed/logs/<run_id>.transcript.jsonl` |

Set `RE_UQ_SMOKE_TREE=1` to force the smoke tree explicitly; the `smoke-fake*` subcommands set it for you.

## End-To-End Chain

The three stages are chained: Task 3 needs a Task 2 run, and the analysis needs both.

```bash
# 0. Environment.
uv sync --group dev --locked
.venv/bin/python -m unittest discover -s tests -v

# 1. Copy the example config. Credentials are not required for the
#    fake-completion path, but the file structure must exist.
cp run_configs/full_matrix.example.json run_configs/current_run.json

# 2-4. All three stages in one command.
bash scripts/reproduce.sh smoke-fake-all

# ... or one stage at a time:
bash scripts/reproduce.sh smoke-fake            # Task 1 + Task 2
bash scripts/reproduce.sh smoke-fake-task3      # blind Task 3 over the smoke Task 2 run
bash scripts/reproduce.sh smoke-fake-analysis   # analysis over both
```

Every subcommand prints a `resolved: ...` line naming the config, profile, model, dataset, variant, and run id it is about to use. Read it before assuming which cell you just ran.

The equivalent raw commands, for when you need to change a flag:

```bash
.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile zai --model glm-5.1 --dataset mlm_tapt \
  --task both --mode smoke --fake-completion

.venv/bin/python scripts/run_task3_verification_from_config.py \
  --config run_configs/current_run.json \
  --profile zai --model glm-5.1 --dataset mlm_tapt \
  --source-run-id SMOKE_RUN_ID --audit-mode blind \
  --mode smoke --fake-completion --allow-partial-source

RE_UQ_SMOKE_TREE=1 .venv/bin/python scripts/generate_evaluation_analysis.py \
  --dataset mlm_tapt --variant must \
  --run-id SMOKE_RUN_ID --task3-run-id SMOKE_TASK3_RUN_ID \
  --task3-audit-mode blind \
  --output-dir outputs/smoke/evaluation_mlm_tapt_must_SMOKE_RUN_ID \
  --allow-partial --skip-registry-check --skip-construct-review-check
```

`--allow-partial-source` is required whenever the Task 2 source is a smoke run:
source completeness is checked against the full 720-item benchmark, and a smoke
run only covers `smoke_items` per cell. `scripts/reproduce.sh smoke-fake-task3`
passes the flag for you.

`zai` / `glm-5.1` / `mlm_tapt` is only the default cell; no provider is contacted on this path. Override with `RE_UQ_PROFILE`, `RE_UQ_MODEL`, `RE_UQ_DATASET` (see [`docs/reproduction.md`](reproduction.md)).

## Smoke Runs Stay In The Smoke Namespace

Two guards keep the smoke chain from contaminating real results:

1. **Run-id prefixes.** A smoke Task 1+2 run id is prefixed `smoke...`; a smoke Task 3 run id ends in `-smoke`. Full runs use `full...` / `task3...`.
2. **Task 3 accepts a `smoke-` source run only in `--mode smoke`.** In `--mode full`, Task 3 resolves its source run against the full-run prefix and refuses a `smoke-` Task 2 run id, so a fake-completion Task 2 run can never become the source of a paper-facing Task 3 diagnostic.

The same applies downstream: analysis over smoke runs writes into the smoke tree and must not be promoted.

## What You Should See

- A `resolved: ...` line for each stage.
- A new smoke run ID registered in `data/processed/smoke/run_registry_mlm_tapt.csv`.
- Raw synthesized rows appended to `data/processed/smoke/model_outputs_raw_mlm_tapt.jsonl`.
- A Task 3 smoke run id derived from the Task 2 smoke run.
- An analysis directory under `outputs/smoke/` containing `uq_scores.csv`, `metrics_summary.csv`, and `provenance_manifest.json`.
- A log file at `data/processed/logs/<run_id>.log`.
- A short progress summary on the terminal.
- No HTTP calls.

If any of the above fail, the pipeline plumbing has regressed. If they succeed, the repository is correctly wired and a real provider run can be configured by editing `run_configs/current_run.json` to point at your endpoint and credentials.

## What This Does **Not** Verify

- Real provider response parsing under network conditions, rate limits, or structured-output enforcement.
- Batching behaviour against a real model. Fake completions always parse, so the batch-fallback path (re-sending a failed batch as single items) is not exercised end-to-end.
- The construct-validity gate (`docs/weak_modality_construct_review.csv`) — that still needs a two-reviewer pass before paper-facing weak-intent claims, and the tracked file is currently LLM-assisted and pending human confirmation.
- The analysis gates in `scripts/generate_evaluation_analysis.py` in their strict form (registry completeness, confidence-scale contract, prompt-row freshness) — the smoke analysis relaxes them.
- Any numeric result. Fake completions carry no information about model behaviour.

For the real path, see [`docs/reproduction.md`](reproduction.md); for what the real runs actually sent, see [`docs/experimental_setup.md`](experimental_setup.md).
