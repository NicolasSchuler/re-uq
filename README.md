# When Weak Intent Becomes a Requirement

> **Why this matters.** LLMs increasingly help extract requirements from raw stakeholder text. Requirements carry both *what* is requested and *how strongly* the stakeholder commits to it. If "it would be nice if ..." becomes "the system should ..." or "the system must ...", downstream design, validation, effort estimation, and stakeholder expectations can shift even when the functional capability is preserved.

This repository supports the IST short communication **"When Weak Intent Becomes a Requirement: Limits of Uncertainty Signals for Modal-Force Strengthening in LLM-Assisted Requirements Engineering."** The study holds the *functional capability* constant and varies only the *source modal force* (`mandatory` / `recommended` / `optional` / `nice_to_have`), then asks:

1. To what extent do LLMs preserve source modal force in labels and generated text?
2. Do lightweight black-box uncertainty signals expose text-level modal-force strengthening?
3. Can audits detect extractions that change source modal force?

The headline observable is **high-confidence over-commitment of weak stakeholder intent**: a model can return the correct modality label while generating requirement text that strengthens the source.

## Paper Motivation And Findings

Requirement modal force is treated as an ordinal commitment signal:
`mandatory > recommended > optional > weak stakeholder intent`. The benchmark
uses 360 reviewed capability seeds from NICE/PROMISE and
`limsc/mlm-tapt-requirements`, rewritten into four controlled source variants,
for 1440 benchmark items. The main mandatory wording is `MUST`; `SHALL` is a
supplementary robustness variant.

The paper evaluates three tasks:

- **Task 1:** mandatory-requirement entailment, a control for unsupported upgrades to mandatory language.
- **Task 2:** modality-preserving extraction, the primary LLM-assisted RE workflow.
- **Task 3:** blind source-grounded audit, a diagnostic check over deterministic Task 2 outputs.

Key reported observations from the manuscript-level macro summary:

- Declared modality labels are preserved, but generated requirement text still strengthens source modal force in 8.6% of cases under strict evidence and 13.9% under broad evidence.
- For weak stakeholder-intent sources, strict text strengthening reaches 29.8%.
- Strengthened outputs are typically high-confidence and sample-stable: 98.4% high-confidence rate and 100.0% repeated-sample agreement in the reported macro summary.
- Semantic dispersion and supervised embedding probes provide partial diagnostic signal, but source-controlled probes and blind audits are not reliable certification mechanisms.

The practical takeaway is that UQ for LLM-assisted RE should test whether
generated requirement text preserves stakeholder commitment, not only whether a
declared label is correct or stable.

## Artifact Status

This checkout is organized as a reviewer-facing artifact. The tracked prompts,
reviewed seeds, benchmark CSVs, manifests, and documentation are the durable
inputs. Checked metric snapshots and legacy external-probe reports are
diagnostic or stale unless regenerated from a complete post-fix run and marked
paper-ready in [`outputs/README.md`](outputs/README.md). Raw model outputs and
run registries are local reproducibility evidence by default; see
[`docs/repository_hygiene.md`](docs/repository_hygiene.md) for promotion rules.

## Reading Order

| You want to … | Go to |
| --- | --- |
| Understand the study and evaluation design | [`docs/evaluation.md`](docs/evaluation.md) |
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

## Reproduce The Main Results

The commands below are templates for a real provider/model cell. Replace the
provider placeholders in `run_configs/current_run.json`, then set `PROFILE` and
`MODEL` to the profile and model IDs from that config. The canonical task order
is: Task 1+2 first, blind Task 3 second, analysis last.

```bash
uv sync --group dev --locked
.venv/bin/python -m unittest discover -s tests -v

cp run_configs/full_matrix.example.json run_configs/current_run.json
# Edit run_configs/current_run.json for your endpoint, API-key env var, profile, and model.

PROFILE="your_profile_id"
MODEL="your_model_id"
DATASET="mlm_tapt"      # nice | mlm_tapt
VARIANT="must"          # must | shall

# Task 1 control + Task 2 extraction/text-drift run.
.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile "${PROFILE}" \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --variant "${VARIANT}" \
  --task both \
  --mode full
```

Take the completed Task 1+2 `RUN_ID` from the matching
`data/processed/run_registry*.csv` file, then run official blind Task 3:

```bash
SOURCE_RUN_ID="full-..."

.venv/bin/python scripts/run_task3_verification_from_config.py \
  --config run_configs/current_run.json \
  --profile "${PROFILE}" \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --variant "${VARIANT}" \
  --source-run-id "${SOURCE_RUN_ID}" \
  --audit-mode blind \
  --mode full
```

Take the completed Task 3 `TASK3_RUN_ID` from
`data/processed/run_registry_task3_verification*.csv`, then generate the
paper-facing analysis:

```bash
TASK3_RUN_ID="task3-..."

.venv/bin/python scripts/generate_evaluation_analysis.py \
  --dataset "${DATASET}" \
  --variant "${VARIANT}" \
  --run-id "${SOURCE_RUN_ID}" \
  --task3-run-id "${TASK3_RUN_ID}" \
  --task3-audit-mode blind \
  --profile "${PROFILE}" \
  --model "${MODEL}"
```

Repeat the template for both datasets (`nice`, `mlm_tapt`) and both benchmark
variants (`must`, `shall`). `must` is the primary mandatory-wording condition;
`shall` is the robustness variant. The official model cohort excludes private
Azure-hosted `azure.*` runs; those rows may remain in local registries as
diagnostic attempts, but should not be counted in paper-facing results.
Declared-modality Task 3 modes
(`declared_text`, `declared_source`) are anchoring ablations, not official blind
Task 3 results. If `docs/weak_modality_construct_review.csv` is incomplete, the
analysis should be treated as diagnostic; use `--skip-construct-review-check`
only for local inspection, not paper-ready results.

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

## License, Citation, And Maintainers

The code and documentation are distributed under the MIT license in
[`LICENSE`](LICENSE). During double-blind review, authors and maintainers are
listed as **Anonymous Authors**; the deanonymized release should add final
citation metadata after submission identity constraints are lifted. See
[`docs/anonymization.md`](docs/anonymization.md) for the submission sanitization
plan.

## Threats To Validity

Two scope notes carry over into the paper and should travel with this artifact:

- Paper-facing weak-intent claims are gated on `docs/weak_modality_construct_review.csv`. Until both reviewer slots have judged every weak template as weaker than `SHOULD/recommended`, weak-class results are diagnostic, not headline.
- The metric snapshots checked in at the time of writing are diagnostic/stale until a clean post-fix full run regenerates paper-facing tables and figures from raw outputs. The analysis script refuses stale prompts, mixed confidence scales, or incomplete registries.

See `docs/evaluation.md` for the study design, `docs/results_mapping.md` for the artifact-to-claim trail, and `outputs/README.md` for artifact status.
