# When Weak Intent Becomes a Requirement

This repository is the evaluation artifact for the IST short communication
**"When Weak Intent Becomes a Requirement: Limits of Uncertainty Signals for
Modal-Force Strengthening in LLM-Assisted Requirements Engineering."**

## In One Minute

LLMs can extract the right feature from a stakeholder statement but change how
strongly the stakeholder meant it.

Example:

- Source: "It would be useful if the system could export reports."
- Risky extraction: "The system should export reports."

The feature is the same. The commitment is not. A weak wish has become a
stronger requirement.

This project tests that failure mode in LLM-assisted requirements engineering
(RE). It asks whether common uncertainty signals notice when this happens.

## Main Takeaway

Label accuracy is not enough.

In the paper, models often preserve the declared modality label, but the
generated requirement text can still strengthen the source. These strengthened
outputs are usually high-confidence and stable across repeated samples.

That means RE uncertainty checks should ask:

- Did the generated text preserve the stakeholder's commitment?
- Did the model strengthen weak or optional language?
- Would a reviewer notice the change from uncertainty signals alone?

## Study At A Glance

| Piece | What it means |
| --- | --- |
| Core construct | Requirement modal force: how strong the stakeholder commitment is. |
| Modal scale | `mandatory > recommended > optional > weak stakeholder intent` |
| Datasets | NICE/PROMISE and `limsc/mlm-tapt-requirements` |
| Benchmark size | 360 reviewed seeds x 4 source variants = 1440 items |
| Main wording | `MUST`; `SHALL` is a robustness variant |
| Main risk | High-confidence strengthening of weak intent |

The study uses controlled minimal pairs. For each reviewed capability, only the
source modal force changes:

- `MUST`: mandatory
- `SHOULD`: recommended
- `MAY`: optional
- `It would be useful if ...`: weak stakeholder intent

## Tasks

| Task | Purpose |
| --- | --- |
| Task 1: mandatory entailment | Checks whether a mandatory requirement is supported by the source. |
| Task 2: modality-preserving extraction | Main task: extracts one requirement and a modality label. |
| Task 3: blind audit | Diagnostic task: checks whether Task 2 text preserves, strengthens, or weakens the source. |

Task 3 is not a final verifier. It is a stress test for whether an audit prompt
can catch text-level modal-force changes.

## Prompts Per Task

The findings here are not the product of one carefully tuned prompt. Every prompt
is frozen as a plain text file, content-addressed by SHA-256 in
[`outputs/benchmark_manifest*.json`](outputs/), and verified by the analysis gate.
Each task also ships **sensitivity/ablation variants** that exist specifically to
show the effect is not an artifact of a single phrasing (see
[`prompts/README.md`](prompts/README.md) for the full index). The verbatim main
prompts are reproduced below. The `{{...}}` braces are literal in the frozen file;
`{source_statement}` and similar are the only substituted fields.

### Task 1 — Mandatory Entailment (control)

Does the source statement entail the candidate *mandatory* requirement?
Source: [`prompts/mandatory_entailment.txt`](prompts/mandatory_entailment.txt).

```text
You are reviewing software requirements.

Source statement:
"{source_statement}"

Candidate requirement:
"{candidate_requirement}"

Question:
Is the candidate a faithful mandatory requirement entailed by the source?

Return JSON only:
{{
  "decision": "yes" | "no",
  "confidence": 0.0-1.0,
  "brief_reason": "<max 12 words>"
}}

Use confidence as a decimal probability for confidence in the selected decision.
Do not return percentages such as 95 or strings such as "95%".
```

Robustness variant: [`prompts/mandatory_entailment_strict.txt`](prompts/mandatory_entailment_strict.txt)
(stricter wording, run on the pilot subset).

### Task 2 — Modality-Preserving Extraction (main task)

Extract one requirement and label its modality, preserving the source modal force.
Source: [`prompts/modality_extraction.txt`](prompts/modality_extraction.txt).

```text
Extract exactly one requirement from the source statement.
Preserve the modality of the source.

Source:
"{source_statement}"

Return JSON only:
{{
  "requirement": "...",
  "modality": "mandatory" | "recommended" | "optional" | "nice_to_have",
  "confidence": 0.0-1.0
}}

Use confidence as a decimal probability for confidence in the selected modality.
Do not return percentages such as 95 or strings such as "95%".
```

Robustness variant: [`prompts/modality_extraction_labels_only.txt`](prompts/modality_extraction_labels_only.txt)
(states the allowed labels without mapping rules or examples).

### Task 3 — Blind Text Audit (diagnostic)

Given the source and the Task 2 output, judge whether the requirement preserves,
strengthens, weakens, or changes the content — **without** seeing the declared
Task 2 modality. Source: [`prompts/modality_verification.txt`](prompts/modality_verification.txt).

```text
You are auditing whether an extracted software requirement preserves the source statement.

Source statement:
"{source_statement}"

Extracted requirement:
"{extracted_requirement}"

Question:
Compared with the source statement, does the extracted requirement preserve, strengthen, weaken, or change the functional content?

Use these labels:
- preserves: same functional content and same modality strength
- strengthens: same functional content but stronger modality or stakeholder commitment
- weakens: same functional content but weaker modality or stakeholder commitment
- content_changed: functional content was added, removed, or changed

Return JSON only:
{{
  "relation": "preserves" | "strengthens" | "weakens" | "content_changed",
  "confidence": 0.0-1.0,
  "evidence_phrase": "<exact phrase from the source that supports the relation>",
  "brief_reason": "<max 16 words>"
}}

Use confidence as a decimal probability for confidence in the selected relation.
Do not return percentages such as 95 or strings such as "95%".
```

Ablation variant: [`prompts/modality_verification_declared.txt`](prompts/modality_verification_declared.txt)
(reveals the declared Task 2 modality to measure anchoring).

The allowed labels and the `0.0-1.0` confidence contract are not free text: they
are enforced as Pydantic models in
[`scripts/structured_outputs.py`](scripts/structured_outputs.py) and covered by
[`tests/test_structured_outputs.py`](tests/test_structured_outputs.py).

## Reported Paper Findings

From the manuscript-level macro summary:

- Declared modality labels were preserved.
- Generated text still strengthened source modal force in 8.6% of cases under
  strict evidence and 13.9% under broad evidence.
- For weak stakeholder-intent sources, strict strengthening reached 29.8%.
- Strengthened outputs were usually high-confidence: 98.4%.
- Strengthened outputs were sample-stable in the reported summary: 100.0%
  repeated-sample agreement.
- Semantic dispersion, embedding probes, and blind audits helped only partly.

Interpret these numbers through the artifact status below. Checked metric
snapshots in this repository are diagnostic/stale unless regenerated from a
complete current run.

## Where To Go

| You want to ... | Read |
| --- | --- |
| Understand the study design | [`docs/evaluation.md`](docs/evaluation.md) |
| Reproduce the pipeline | [`docs/reproduction.md`](docs/reproduction.md) |
| Run without API credentials | [`docs/reproduction_smoke.md`](docs/reproduction_smoke.md) |
| Trace a result to inputs | [`docs/results_mapping.md`](docs/results_mapping.md) |
| Understand the code/data layout | [`docs/repository_layout.md`](docs/repository_layout.md) |
| Understand the pipeline architecture | [`docs/architecture.md`](docs/architecture.md) |
| Understand artifact policy | [`docs/repository_hygiene.md`](docs/repository_hygiene.md) |
| See common reviewer questions | [`docs/faq.md`](docs/faq.md) |

## Quick Check

Use this path if you only want to confirm that the repository is wired
correctly.

```bash
uv sync --group dev --locked
.venv/bin/python -m unittest discover -s tests -v

cp run_configs/full_matrix.example.json run_configs/current_run.json
bash scripts/reproduce.sh smoke-fake
```

The `smoke-fake` path uses local fake completions. It does not call a provider
and does not reproduce paper results.

## Full Provider Run

For real results, configure a provider in `run_configs/current_run.json`, then
run Task 1+2, Task 3, and analysis in that order.

```bash
PROFILE="your_profile_id"
MODEL="your_model_id"
DATASET="mlm_tapt"      # nice | mlm_tapt
VARIANT="must"          # must | shall

.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile "${PROFILE}" \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --variant "${VARIANT}" \
  --task both \
  --mode full
```

Then take the completed Task 1+2 `RUN_ID` from
`data/processed/run_registry*.csv` and run blind Task 3:

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

Then take `TASK3_RUN_ID` from
`data/processed/run_registry_task3_verification*.csv` and generate analysis:

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

For the complete command matrix and diagnostic flags, use
[`docs/reproduction.md`](docs/reproduction.md).

## Paper-Ready Rules

Do not treat a run as paper-ready unless:

- Task 1+2 completed for the intended provider/model/dataset/variant cell.
- Task 3 used `--audit-mode blind`.
- `docs/weak_modality_construct_review.csv` is complete.
- Confidence values follow the current `0.0-1.0` contract.
- The analysis script ran without diagnostic opt-out flags.
- Generated tables and figures were manually inspected.

Private Azure-hosted `azure.*` rows may remain in local registries as
diagnostics, but they are excluded from the official paper cohort.

Declared-modality Task 3 modes (`declared_text`, `declared_source`) are
anchoring ablations, not official blind Task 3 results.

## Repository Map

| Path | Purpose |
| --- | --- |
| `scripts/` | Canonical CLI entry points for reproduction. |
| `prompts/` | Frozen prompts for Tasks 1, 2, and 3. |
| `data/processed/` | Curated benchmark inputs and compact metric snapshots. |
| `outputs/` | Curated summaries, manifests, and review artifacts. |
| `notebooks/` | Generated companion notebooks for inspection. |
| `docs/` | Reader-facing documentation. |
| `tests/` | Unit, CLI, parsing, and notebook-boundary checks. |

Scripts are canonical. Notebooks are for reading and inspection.

## Artifact Status

Tracked durable inputs:

- prompts
- reviewed seeds
- benchmark CSVs
- manifests
- documentation

Local-only by default:

- raw model outputs
- run registries
- run progress files
- run-specific analysis directories

Checked metric snapshots and legacy external-probe reports are diagnostic or
stale unless regenerated from a complete current run and marked paper-ready in
[`outputs/README.md`](outputs/README.md). See
[`docs/repository_hygiene.md`](docs/repository_hygiene.md) for artifact policy.

## Environment

The project uses `uv` and a local `.venv/`.

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`.

Core libraries include `pandas`, `numpy`, `scipy`, `scikit-learn`,
`matplotlib`, `openai`, `requests`, `instructor`, `pydantic`, and `nbformat`.

## Authors, License, And Citation

This artifact is developed and maintained by:

- **Nicolas Schuler** — Karlsruhe Institute of Technology (KIT)
- **Vincenzo Scotti** — Karlsruhe Institute of Technology (KIT)
- **Raffaela Mirandola** — Karlsruhe Institute of Technology (KIT)

The code and documentation use the MIT license in [`LICENSE`](LICENSE).

To cite this repository, use the metadata in [`CITATION.cff`](CITATION.cff)
(GitHub renders a "Cite this repository" button from it). Manuscript citation
details will be added on acceptance.
