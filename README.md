# Modality-Conditioned UQ Evaluation

This repository contains a notebook-based evaluation pipeline for an IST short communication on uncertainty quantification in LLM-assisted requirements engineering.

## Environment

The project uses a `uv`-managed virtual environment at `.venv/`.

Install or refresh dependencies from `pyproject.toml` and `uv.lock`:

```bash
uv sync --group dev
```

Use the environment for local commands:

```bash
source .venv/bin/activate
.venv/bin/python -m unittest discover -s tests -v
```

The evaluation utilities intentionally use established libraries from this environment: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `openai`, `requests`, and `nbformat`.

## Pipeline

Run the notebooks in order:

1. `notebooks/00_prepare_data.ipynb`
2. `notebooks/01_build_modality_benchmark.ipynb`
3. `notebooks/02_pilot_local_llms.ipynb`
4. `notebooks/03_run_experiments.ipynb`
5. `notebooks/03b_run_modality_verification.ipynb`
6. `notebooks/04_compute_uq_and_metrics.ipynb`
7. `notebooks/05_analyze_and_export_results.ipynb`

The local LLM endpoint is configured through `HOST` and `MODEL`/`MODELS` variables in the notebooks. By default, the pipeline expects an OpenAI-compatible endpoint at `http://localhost:8000/v1`.

For persistent local settings, copy `config.example.json` to `config.json` and edit the model list or endpoint there.

Generated raw model outputs are cached as JSONL under `data/processed/`. The NICE dataset is downloaded or placed manually under `data/raw/`.

`notebooks/04_compute_uq_and_metrics.ipynb` analyzes one full experiment run at a time. By default it selects the latest `full-*` run in `data/processed/model_outputs_raw.jsonl`; set `RUN_ID` or `ANALYSIS_RUN_ID` to reproduce metrics for an earlier run.

`notebooks/03b_run_modality_verification.ipynb` adds the Task 3 self-verification diagnostic after a complete full run. It verifies deterministic Task 2 extractions with the same model and caches raw verifier outputs in `data/processed/model_outputs_raw_task3_verification.jsonl`.
