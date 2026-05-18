# AGENTS.md

## Repository Purpose

This repository supports an Information and Software Technology short communication paper on uncertainty quantification for LLM-assisted requirements engineering.

The core research question is whether LLM uncertainty estimates are sensitive to linguistic modality in requirements, especially cases where weak stakeholder intent such as "may", "should", or "nice to have" is transformed into mandatory language such as "must" or "shall".

The intended contribution is a focused empirical evaluation, not a broad survey.

## Research Framing

Treat this project as a research-engineering artifact.

Distinguish clearly between:

- Observation: grounded in datasets, model outputs, metrics, logs, or cited sources.
- Hypothesis: plausible interpretation that still needs evidence.
- Recommendation: the next action justified by the current evidence.
- Open question: uncertainty that could affect the paper's claims.

Do not overclaim. The goal is credible, compact evidence suitable for a 2,500-word short communication.

## Expected Evaluation Scope

The planned evaluation should focus on:

- controlled modality variants derived from real requirements seeds;
- mandatory-requirement entailment;
- modality-preserving extraction;
- lightweight black-box uncertainty quantification;
- over-commitment, calibration, and monotonicity metrics.

The primary failure mode of interest is a modality upgrade, such as optional or recommended functionality being treated as mandatory with high confidence.

## Engineering Principles

Keep exploratory work and durable code separate.

Prefer:

- small, inspectable notebooks or scripts;
- reproducible data generation;
- cached raw model outputs;
- exact model identifiers and run metadata;
- simple metrics that directly support the paper's claims;
- concise figures and tables suitable for the IST word limit.

Avoid:

- broad benchmark sprawl;
- unnecessary abstractions;
- hidden manual edits to generated data;
- undocumented prompt or model changes;
- mixing temporary diagnostics into stable analysis code.

## Environment

This repository uses a `uv`-managed virtual environment at `.venv/`.

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. Run `uv sync --group dev` to refresh the environment.

Prefer running Python and notebook-related commands through that environment, for example:

- `.venv/bin/python -m unittest discover -s tests -v`
- `.venv/bin/python scripts/populate_notebooks.py`
- `source .venv/bin/activate` before launching Jupyter locally

Use established scientific/notebook libraries already declared for the project (`pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `openai`, `requests`, and `nbformat`) rather than reimplementing common CSV, metric, plotting, HTTP, or notebook-JSON functionality.

## Verification Expectations

Before treating results as paper-ready:

- inspect generated benchmark items;
- validate gold labels for modality variants;
- check JSON parsing and failed model responses;
- recompute metrics from cached raw outputs;
- visually inspect plots and tables;
- record important caveats and threats to validity.

The final artifact should make it easy to trace each reported result back to prompts, model outputs, and analysis code.
