# Phased Evaluation Implementation Plan

This plan implements the modality-conditioned uncertainty evaluation described in `docs/evaluation.md`.

## Environment

- Use the existing `uv`-managed virtual environment at `.venv/`.
- Run Python commands with `.venv/bin/python` or activate the environment with `source .venv/bin/activate`.
- Dependencies are declared in `pyproject.toml` and locked in `uv.lock`; refresh with `uv sync --group dev`.
- Use the declared scientific stack (`pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `openai`, `requests`, and `nbformat`) instead of custom replacements for common CSV, metric, plotting, HTTP, and notebook-JSON tasks.

## Phase 1: Prepare Requirement Seeds

- Run `notebooks/00_prepare_data.ipynb`.
- Load or download the NICE/PROMISE-derived CSV into `data/raw/`.
- Generate `data/processed/seeds_review.csv`.
- Manually review included seeds until exactly 120 accepted capabilities remain.

## Phase 2: Build Modality Benchmark

- Run `notebooks/01_build_modality_benchmark.ipynb`.
- Generate four controlled variants for each accepted seed: mandatory, recommended, optional, and nice-to-have.
- Write `data/processed/benchmark_items.csv`.

## Phase 3: Pilot Local LLM Calls

- Run `notebooks/02_pilot_local_llms.ipynb`.
- Configure the OpenAI-compatible local endpoint with `HOST` and `MODEL`.
- Set `RUN_PILOT=true` only after the endpoint is available.
- Continue only if JSON parse success is at least 95%.

## Phase 4: Run Full Experiments

- Run `notebooks/03_run_experiments.ipynb`.
- Configure `MODELS` for the locally provided model IDs.
- Set `RUN_FULL_EXPERIMENT=true` after the pilot passes.
- Cache every raw response in `data/processed/model_outputs_raw.jsonl`.

## Phase 5: Compute UQ and Metrics

- Run `notebooks/04_compute_uq_and_metrics.ipynb`.
- Analyze one full experiment run at a time. The notebook selects the latest `full-*` run by default; set `RUN_ID` or `ANALYSIS_RUN_ID` to reproduce an earlier run.
- Compute verbalized confidence, label self-consistency, and modality consistency.
- Export UQ scores, metric summaries, bootstrap confidence intervals, and the recommended-strength sensitivity check.

## Phase 6: Export Paper Artifacts

- Run `notebooks/05_analyze_and_export_results.ipynb`.
- Export the compact paper-facing result table and modality figure under `outputs/`.
- Inspect the figure and result notes before using them in the IST manuscript.

## Verification

- Use `.venv/bin/python -m unittest discover -s tests -v` for utility tests.
- Validate notebook JSON after changes.
- Treat model results as paper-ready only after inspecting benchmark items, parse failures, metric summaries, and generated plots.
