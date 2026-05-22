# Modality-Conditioned UQ Evaluation

This repository supports an IST short communication on a specific requirements-engineering risk: LLMs may preserve the functional content of a requirement while confidently strengthening weak stakeholder intent into firmer requirement language.

The study uses controlled requirement variants that hold capability constant and vary only modality. Task 1 is a mandatory-entailment control, Task 2 is the main modality-preserving extraction task, and Task 3 is a downstream self-audit diagnostic over Task 2 outputs.

For the command-first reproduction path, see `docs/reproduction.md`.

## Repository Map

| Path | Purpose |
| --- | --- |
| `data/processed/` | Curated benchmark inputs and compact metric snapshots. Raw run outputs are local-only by default. |
| `docs/` | Evaluation specification, paper framing, reproduction guide, hygiene policy, and final reviewed seed documents. |
| `notebooks/` | Stripped, generated companion notebooks for narrative inspection. |
| `outputs/` | Curated summaries and review artifacts. Final regenerated analysis directories are local until promoted. |
| `prompts/` | Task prompt contracts used by runs and provenance manifests. |
| `run_configs/` | Tracked example provider matrices; local `current_run*.json` files are ignored. |
| `scripts/` | Canonical command-line entry points and shared evaluation utilities. |
| `tests/` | Unit, CLI, parsing, and notebook-boundary regression checks. |

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
.venv/bin/python -m coverage run --branch --source=scripts -m unittest discover -s tests -v
.venv/bin/python -m coverage report -m
```

The evaluation utilities intentionally use established libraries from this environment: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `openai`, `requests`, and `nbformat`.

Repository workflow and artifact tracking expectations are documented in `docs/repository_hygiene.md`.

## Canonical Workflow

Prepare a local run config:

```bash
cp run_configs/full_matrix.example.json run_configs/current_run.json
```

Run Task 1 and Task 2 from the provider-aware CLI:

```bash
.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile zai \
  --model glm-4.5-air \
  --dataset mlm_tapt \
  --task both \
  --mode smoke
```

Then run the complete matrix cell after the smoke run passes provider preflight and parse checks:

```bash
.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile zai \
  --model glm-4.5-air \
  --dataset mlm_tapt \
  --task both \
  --mode full
```

Run Task 3 only after the selected full Task 2 run is complete:

```bash
.venv/bin/python scripts/run_task3_verification_from_config.py \
  --config run_configs/current_run.json \
  --profile zai \
  --model glm-4.5-air \
  --dataset mlm_tapt \
  --source-run-id RUN_ID \
  --mode full
```

Generate paper-facing analysis artifacts headlessly:

```bash
.venv/bin/python scripts/generate_evaluation_analysis.py \
  --dataset mlm_tapt \
  --variant must \
  --run-id RUN_ID \
  --task3-run-id TASK3_RUN_ID
```

The notebooks remain available for inspection or local exploratory work, but the scripts above are the publication reproduction interface.

## Data And Providers

The default seed dataset is NICE/PROMISE. The Hugging Face `limsc/mlm-tapt-requirements` path is the co-primary dataset and writes suffixed artifacts such as `data/processed/seeds_review_mlm_tapt.csv` and `data/processed/benchmark_items_mlm_tapt.csv`.

For persistent local notebook settings, copy `config.example.json` to `config.json`. Provider-matrix runs should use ignored files under `run_configs/current_run*.json`.

Provider profiles can set `batch_size` to pack multiple same-task benchmark items into one API request. The runner still writes one canonical raw JSONL row per item/sample so existing analysis remains unchanged.

Profiles can set `structured_output` to `json_object`, `json_schema`, or `instructor`. Current v2 prompts use `confidence` as a `0.0`-to-`1.0` selected-label probability; new v2 raw records include `confidence_scale=0_1`. The Instructor path validates Pydantic response models and retries partial or invalid batches unbatched.

The example Z.ai profile uses the GLM Coding Plan endpoint (`https://api.z.ai/api/coding/paas/v4`). Switch it to the general endpoint only when using a standard paid API balance.

## Monitoring

Raw outputs append to the canonical `data/processed/model_outputs_raw*.jsonl` files. `data/processed/run_registry*.csv` records which provider/model/run IDs are complete. During long runs, the CLI periodically refreshes `data/processed/run_progress_live*.csv` and appends structured events to `data/processed/run_events*.jsonl`.

Monitor a run from another terminal with:

```bash
.venv/bin/python scripts/show_run_progress.py --dataset nice --run-id RUN_ID --watch 30
```

Compare a completed run group with:

```bash
.venv/bin/python scripts/compare_run_matrix.py --config run_configs/current_run.json
```

## Notebooks

The generated notebooks provide a readable staged version of the workflow:

1. `notebooks/00_prepare_data.ipynb`
2. `notebooks/01_build_modality_benchmark.ipynb`
3. `notebooks/02_pilot_local_llms.ipynb`
4. `notebooks/02b_weak_modality_robustness_probe.ipynb`
5. `notebooks/03_run_experiments.ipynb`
6. `notebooks/03b_run_modality_verification.ipynb`
7. `notebooks/04_compute_uq_and_metrics.ipynb`
8. `notebooks/05_analyze_and_export_results.ipynb`

The current checked-in metric snapshots are preliminary. Paper-facing claims require a clean post-fix full run, recomputed metrics, and inspected exported figures/tables.
