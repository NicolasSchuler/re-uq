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

The default seed dataset is NICE/PROMISE. To build and run the Hugging Face `limsc/mlm-tapt-requirements` co-primary dataset, set:

```bash
DATASET_ID=mlm_tapt
```

The `mlm_tapt` path loads the Hugging Face dataset through `datasets`, excludes `_PURE` sources, applies stricter candidate filtering, and writes suffixed artifacts such as `data/processed/seeds_review_mlm_tapt.csv` and `data/processed/benchmark_items_mlm_tapt.csv`. NICE/PROMISE filenames remain unchanged.

The final reviewed seed capability documents are committed under `docs/final_seed_documents/` for both NICE/PROMISE and `mlm_tapt`.

For reproducible multi-provider runs, copy `run_configs/full_matrix.example.json` to the ignored `run_configs/current_run.json`, edit the provider/model list, and start with a smoke run:

```bash
.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile local_llama_cpp \
  --model qwen/qwen3.5-9b \
  --dataset nice \
  --mode smoke
```

Then run `--mode full` after the provider preflight and smoke parse checks pass. Provider profiles can set `batch_size` to pack multiple same-task benchmark items into one API request; the runner still writes one canonical raw JSONL row per item/sample so existing analysis remains unchanged. Raw outputs still append to the canonical `data/processed/model_outputs_raw*.jsonl` files; `data/processed/run_registry*.csv` records which provider/model/run IDs are complete. Compare a run group with:

```bash
.venv/bin/python scripts/compare_run_matrix.py --config run_configs/current_run.json
```

`notebooks/04_compute_uq_and_metrics.ipynb` analyzes one full experiment run at a time. By default it selects the latest `full-*` run in `data/processed/model_outputs_raw.jsonl`; set `RUN_ID` or `ANALYSIS_RUN_ID` to reproduce metrics for an earlier run.

`notebooks/03b_run_modality_verification.ipynb` adds the Task 3 self-verification diagnostic after a complete full run. It verifies deterministic Task 2 extractions with the same model and caches raw verifier outputs in `data/processed/model_outputs_raw_task3_verification.jsonl`.
