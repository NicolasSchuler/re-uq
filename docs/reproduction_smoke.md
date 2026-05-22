# Smoke Reproduction Without Provider Credentials

This is a quick sanity-check path for reviewers or new collaborators who do not have a provider API key set up. It exercises the full request planner, runner, parser, and registry plumbing using a built-in `--fake-completion` mode that synthesizes deterministic responses locally instead of calling an LLM.

The fake-completion path is **not** a substitute for a real benchmark run. It confirms that the pipeline is wired correctly end-to-end. Paper-facing claims still require a real provider run.

## Steps

```bash
# 1. Refresh the environment and confirm tests pass.
uv sync --group dev --locked
.venv/bin/python -m unittest discover -s tests -v

# 2. Copy the example config (provider credentials are not required for the
#    fake-completion path, but the file structure must exist).
cp run_configs/full_matrix.example.json run_configs/current_run.json

# 3. Run the smoke path with synthesized completions.
bash scripts/reproduce.sh smoke-fake
#   or, equivalently, the raw command:
# .venv/bin/python scripts/run_experiment_from_config.py \
#   --config run_configs/current_run.json \
#   --profile zai \
#   --model glm-4.5-air \
#   --dataset mlm_tapt \
#   --task both \
#   --mode smoke \
#   --fake-completion
```

## What you should see

- A new run ID is registered in `data/processed/run_registry_mlm_tapt.csv`.
- Raw synthesized rows are appended to `data/processed/model_outputs_raw_mlm_tapt.jsonl`.
- A short progress summary prints to the terminal.
- No HTTP calls are made.

If any of the above fail, the pipeline plumbing has regressed. If they succeed, the repository is correctly wired and a real provider run can be configured by editing `run_configs/current_run.json` to point at your endpoint and credentials.

## What this does **not** verify

- Real provider response parsing under network conditions, rate limits, or structured-output enforcement.
- The construct-validity gate (`docs/weak_modality_construct_review.csv`) — that still requires the two-reviewer manual pass before paper-facing weak-intent claims.
- The analysis gates in `scripts/generate_evaluation_analysis.py` (registry completeness, confidence-scale contract, prompt-row freshness).

For those, see the full reproduction path in `docs/reproduction.md`.
