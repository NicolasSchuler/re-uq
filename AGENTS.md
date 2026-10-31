# AGENTS.md

> This file configures AI coding assistants (e.g. Copilot CLI, Codex) that operate on this repository. It is tooling, not part of the published research artifact. Reviewers can safely ignore it. The reader-facing entry points are `README.md`, `docs/repository_layout.md`, and `docs/reproduction.md`.

## Repository Purpose

This repository supports a Journal of Systems and Software short paper (New Ideas and Trends) on uncertainty quantification for LLM-assisted requirements engineering.

The core research question is whether LLM uncertainty estimates are sensitive to linguistic modality in requirements, especially cases where weak stakeholder intent such as "may", "should", or "nice to have" is transformed into mandatory language such as "must" or "shall".

The intended contribution is a focused empirical evaluation, not a broad survey.

## Research Framing

Treat this project as a research-engineering artifact.

Distinguish clearly between:

- Observation: grounded in datasets, model outputs, metrics, logs, or cited sources.
- Hypothesis: plausible interpretation that still needs evidence.
- Recommendation: the next action justified by the current evidence.
- Open question: uncertainty that could affect the paper's claims.

Do not overclaim. The goal is credible, compact evidence suitable for a short paper of about 3,500 words.

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

- small, inspectable scripts;
- reproducible data generation;
- cached raw model outputs;
- exact model identifiers and run metadata;
- simple metrics that directly support the paper's claims;
- concise figures and tables suitable for the JSS short-paper word limit.

Avoid:

- broad benchmark sprawl;
- unnecessary abstractions;
- hidden manual edits to generated data;
- undocumented prompt or model changes;
- mixing temporary diagnostics into stable analysis code.

## Docstring Convention

Docstrings follow the Google style, enforced by ruff (`D` rules with
`convention = "google"` in `pyproject.toml`). Run
`.venv/bin/ruff check scripts tests manuscript` before committing.

Shape:

- One summary line in the imperative mood ("Return ...", "Build ..."), ending
  with a period, on the same line as the opening quotes. A docstring that is
  only a summary keeps its closing quotes on that line; otherwise the closing
  quotes stand on their own line.
- A blank line, then free prose that explains intent, rationale and caveats.
  This prose is the valuable part: keep it, do not replace it with a restated
  signature.
- Then the sections, in this order and only when they apply: `Args:`,
  `Returns:` or `Yields:`, `Raises:`. Section bodies are indented four spaces,
  entries are `name: description.` with continuation lines indented a further
  four spaces.
- `Args:` lists every parameter except `self` and `cls` whenever the function
  has any, including keyword-only parameters and `*args` / `**kwargs`. Say what
  the parameter means and which values matter, not its type: the annotations
  carry the types.
- `Returns:` describes the value whenever the function returns something other
  than `None`. `Raises:` lists only exceptions the function raises itself.
- Classes document their purpose in prose and list fields of dataclasses,
  `NamedTuple`s and `TypedDict`s under `Attributes:`. Constructor parameters
  of ordinary classes go under `Args:` in the class docstring; `__init__` has
  no docstring.
- Public modules, classes, functions and methods need a docstring. Private
  helpers (leading underscore), magic methods and test cases do not, but any
  docstring they have follows the same shape.
- Wrap docstring lines at 88 columns, matching the formatter's line length.

Markup:

- Code, identifiers, literals, CLI flags, file names and column names are set
  in single backticks (`eval_utils.read_jsonl`, `--dry-run`, `batch_size = 1`).
  No double backticks and no Sphinx roles such as `:func:`.
- Follow the naming rules above inside docstrings as well: plain-language
  descriptions, no version labels or opaque abbreviations.
- The first line of a module docstring doubles as the CLI description for
  scripts that pass `__doc__` to `argparse`, so keep it self-contained.

## Environment

This repository uses a `uv`-managed virtual environment at `.venv/`.

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. Run `uv sync --group dev` to refresh the environment.

Prefer running Python commands through that environment, for example:

- `.venv/bin/python -m unittest discover -s tests -v`
- `.venv/bin/python scripts/prepare_benchmark.py --help`

Use established scientific libraries already declared for the project (`pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `openai`, and `requests`) rather than reimplementing common CSV, metric, plotting, or HTTP functionality.

## Verification Expectations

Before treating results as paper-ready:

- inspect generated benchmark items;
- validate gold labels for modality variants;
- check JSON parsing and failed model responses;
- recompute metrics from cached raw outputs;
- visually inspect plots and tables;
- record important caveats and threats to validity.

The final artifact should make it easy to trace each reported result back to prompts, model outputs, and analysis code.
