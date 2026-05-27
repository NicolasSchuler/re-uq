# Reproduction Guide

This is the command-first path for reproducing the publication artifacts. The notebooks are useful for inspection, but these scripts are the canonical interface for provider runs and final analysis.

## Task → Command Cheat-sheet

| Goal | Command |
| --- | --- |
| One-time env setup | `uv sync --group dev --locked` |
| Sanity-check pipeline without API access | `bash scripts/reproduce.sh smoke-fake` (uses `--fake-completion`) |
| Smoke test a provider/model cell | `bash scripts/reproduce.sh smoke` |
| Run Task 1 + Task 2 (main) | `bash scripts/reproduce.sh full` |
| Run Task 3 diagnostic | `bash scripts/reproduce.sh task3 --source-run-id RUN_ID` |
| Generate paper-facing analysis | `bash scripts/reproduce.sh analysis --run-id RUN_ID --task3-run-id TASK3_RUN_ID` |
| Monitor a live run | `.venv/bin/python scripts/show_run_progress.py --dataset mlm_tapt --run-id RUN_ID --watch 30` |
| Compare completed cells | `.venv/bin/python scripts/compare_run_matrix.py --config run_configs/current_run.json --dataset mlm_tapt` |

`scripts/reproduce.sh` is a thin convenience wrapper around the canonical CLIs documented below; use the raw commands directly when you need fine-grained control.

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

Resume a partial run by reusing the same run ID:

```bash
.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile zai \
  --model glm-5.1 \
  --dataset mlm_tapt \
  --mode resume \
  --run-id RUN_ID
```

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
By default, this diagnostic uses the dependency-free TF-IDF fallback. To use a local MLX embedding model on Apple Silicon, install `mlx-embeddings` in the environment and run with:

```bash
RE_UQ_ACSE_EMBEDDING_BACKEND=mlx \
RE_UQ_ACSE_MLX_MODEL=mlx-community/Qwen3-Embedding-0.6B-8bit \
.venv/bin/python scripts/generate_evaluation_analysis.py \
  --dataset mlm_tapt \
  --variant must \
  --run-id RUN_ID \
  --task3-run-id TASK3_RUN_ID \
  --task3-audit-mode blind
```

Use diagnostic flags such as `--allow-partial`, `--skip-registry-check`, `--skip-construct-review-check`, `--max-parse-failure-rate`, `--bootstrap-iterations`, or `--expected-stochastic-samples` only for local investigation, not for paper-ready results.

## 7. Publication Verification Gate

Before treating the repository as publication-ready:

```bash
git status --short
git diff --check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m coverage run --branch --source=scripts -m unittest discover -s tests -v
.venv/bin/python -m coverage report -m
```

Also inspect generated figures and tables manually. A figure or table is not paper-ready until its labels, clipping, row counts, and claim support have been checked.
