# Modality-Conditioned Uncertainty in LLM-Assisted Requirements Engineering

> **Why this matters.** LLMs increasingly help extract requirements from raw stakeholder text. They handle the *what* (the functional content) reasonably well, but they routinely upgrade the *how strongly the stakeholder meant it* — turning "it would be nice if …" into something a downstream specification will read as a hard obligation. If the model is confident while doing this, no lightweight uncertainty signal will catch it. This repository is the evaluation artifact for a short empirical study of that specific failure mode.

This repository supports an IST short communication on **modality-conditioned uncertainty** in LLM-assisted RE. The study holds the *functional capability* constant and varies only the *source modality* (`mandatory` / `recommended` / `optional` / `nice_to_have`), and asks two questions:

1. Do LLMs preserve stakeholder commitment when extracting requirements?
2. Do lightweight black-box UQ signals reveal it when they do not?

The headline observable is **high-confidence over-commitment of weak stakeholder intent**: a model can be stable and confident while systematically wrong about modality.

## Reading Order

| You want to … | Go to |
| --- | --- |
| Understand the study and its framing | [`docs/evaluation.md`](docs/evaluation.md), [`docs/paper_framing.md`](docs/paper_framing.md) |
| Understand what is where in the repo | [`docs/repository_layout.md`](docs/repository_layout.md) |
| Understand the pipeline architecture | [`docs/architecture.md`](docs/architecture.md) |
| Reproduce the pipeline (command-first) | [`docs/reproduction.md`](docs/reproduction.md) |
| Reproduce without API credentials | [`docs/reproduction_smoke.md`](docs/reproduction_smoke.md) |
| Trace a paper result back to inputs | [`docs/results_mapping.md`](docs/results_mapping.md) |
| Understand Git / artifact policy | [`docs/repository_hygiene.md`](docs/repository_hygiene.md) |
| Understand the double-blind anonymization plan | [`docs/anonymization.md`](docs/anonymization.md) |
| Common reviewer questions | [`docs/faq.md`](docs/faq.md) |

## Quick Start

```bash
# 1. Refresh the environment.
uv sync --group dev --locked

# 2. Run the unit suite.
.venv/bin/python -m unittest discover -s tests -v

# 3. (Optional) Smoke-test the pipeline without any API credentials.
cp run_configs/full_matrix.example.json run_configs/current_run.json
bash scripts/reproduce.sh smoke-fake
```

For real provider runs, see the Task → Command cheat-sheet at the top of [`docs/reproduction.md`](docs/reproduction.md).

## Repository Map (Teaser)

For the full map, conventions, and the variant-suffix table, see [`docs/repository_layout.md`](docs/repository_layout.md).

- `scripts/` — canonical CLI entry points; the publication reproduction interface.
- `prompts/` — frozen task prompts (Task 1 entailment, Task 2 extraction/text-drift, Task 3 blind audit plus anchoring ablation).
- `data/processed/` — curated benchmark inputs and compact metric snapshots. Raw run outputs are local-only.
- `outputs/` — curated paper-facing summaries and review artifacts.
- `notebooks/` — generated companion notebooks for narrative inspection (scripts remain canonical).
- `docs/` — reader-facing documentation.
- `tests/` — unit, CLI, parsing, and notebook-boundary regression checks.

## Environment

The project uses a `uv`-managed virtual environment at `.venv/`. Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. The intentional scientific stack is `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `openai`, `requests`, `instructor`, `pydantic`, and `nbformat`.

## Threats To Validity

Two scope notes carry over into the paper and should travel with this artifact:

- Paper-facing weak-intent claims are gated on `docs/weak_modality_construct_review.csv`. Until both reviewer slots have judged every weak template as weaker than `SHOULD/recommended`, weak-class results are diagnostic, not headline.
- The metric snapshots checked in at the time of writing are diagnostic/stale until a clean post-fix full run regenerates paper-facing tables and figures from raw outputs. The analysis script refuses stale prompts, mixed confidence scales, or incomplete registries.

See `docs/paper_framing.md` for the full framing, `docs/results_mapping.md` for the artifact-to-claim trail, and `outputs/README.md` for artifact status.
