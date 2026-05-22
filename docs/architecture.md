# Architecture

This document gives a one-page mental model of the evaluation pipeline. It complements `docs/repository_layout.md` (what is where) and `docs/reproduction.md` (which command to run).

## Goal

Measure whether an LLM, given a stakeholder requirement, preserves the **modality** (`mandatory` / `recommended` / `optional` / `nice_to_have`) of that requirement when extracting it. The headline risk is **high-confidence over-commitment**: confidently strengthening weak intent into firmer requirement language.

## Top-Level Data Flow

```
                     +-------------------------+
seed datasets   ---> | seed candidates (raw)   |
(NICE, mlm_tapt)     +-----------+-------------+
                                 |  manual review gate
                                 v
                     +-------------------------+
                     | reviewed seed table     |  data/processed/seeds_review*.csv
                     +-----------+-------------+
                                 |
                                 v
                     +-------------------------+   variant generator
                     | benchmark items         |   (mandatory, recommended,
                     |   capability x variant  |    optional, nice_to_have)
                     +-----+-------------+-----+
                           |             |
            +--------------+             +----------------+
            v                                             v
   Task 1: mandatory entailment             Task 2: modality-preserving extraction
   prompts/mandatory_entailment*.txt        prompts/modality_extraction*.txt
            |                                             |
            +------------------+--------------------------+
                               |
                               v
                +---------------------------------+
                | provider runner                 |  scripts/run_experiment_from_config.py
                |   (OpenAI-compatible / Instructor) |
                +-----+----------------------+----+
                      |                      |
                      v                      v
       raw JSONL (per item, per sample)   run registry / progress
       data/processed/model_outputs_raw*.jsonl
                      |
                      v
                +---------------------------------+
                | Task 3 diagnostic (optional)    |  scripts/run_task3_verification_from_config.py
                |   prompts/modality_verification |   (consumes deterministic Task 2 outputs)
                +-----+---------------------------+
                      |
                      v
                +---------------------------------+
                | analysis & exports              |  scripts/generate_evaluation_analysis.py
                |   UQ scores, metrics, figures,  |
                |   bootstrap CIs, qualitative    |
                |   examples, provenance manifest |
                +---------------------------------+
                      |
                      v
                paper-facing artifacts in outputs/evaluation_<dataset>_<variant>_<run_id>/
```

## Tasks

| Task | Role | Prompt | Output contract |
| --- | --- | --- | --- |
| **Task 1** | Capability/control: does the source statement entail a mandatory requirement? | `prompts/mandatory_entailment.txt` (+ `_strict` sensitivity variant) | `{decision: yes|no, confidence: 0.0-1.0, brief_reason}` |
| **Task 2** | Main: extract modality-preserving requirement label from the source. | `prompts/modality_extraction.txt` (+ `_labels_only` sensitivity variant) | `{modality: mandatory|recommended|optional|nice_to_have, confidence: 0.0-1.0, evidence_phrase}` |
| **Task 3** | Diagnostic: does the deterministic Task 2 extraction preserve, strengthen, weaken, or change the source? | `prompts/modality_verification.txt` | `{relation: preserves|strengthens|weakens|content_changed, confidence: 0.0-1.0, evidence_phrase}` |

Task 3 is **strictly downstream** of a complete deterministic Task 2 run and never repairs Task 2 outputs.

## Confidence Contract

All current v2 prompts (`v2-conf01`, `v2-instructor-conf01`) return `confidence` as a `0.0`-to-`1.0` probability of the selected label. Raw records carry `confidence_scale=0_1`. The analysis gate fails closed if it encounters records on the old `0-100` scale mixed with the v2 path.

## UQ Methods (lightweight, black-box)

- `verbalized_confidence` — the model's own self-reported probability.
- `label_self_consistency`, `modality_consistency` — agreement across stochastic samples.
- `predictive_entropy`, `variation_ratio` — distributional UQ over the same stochastic samples.
- `model_ensemble_disagreement` — only when two deterministic model outputs exist for the same item.
- `token_logprob_confidence` — optional, gated on a provider logprob capability probe.

Mechanistic interpretability is out of scope.

## Code Module Map (`scripts/eval_utils.py`)

The shared utility module is organized into ten labelled sections, each marked with a `# === Section N: …` banner:

1. Configuration loading, validation, and run-config normalization
2. Artifact paths, manifests, and JSON/JSONL IO
3. Text normalization and seed-candidate construction
4. Capability review and modality-benchmark construction
5. Prompts, request planning, and JSON-schema response formats
6. Response parsing, confidence handling, and modality normalization
7. Provider completion drivers (OpenAI-compatible, Instructor, logprobs)
8. Metrics and UQ scoring
9. Run status, registries, progress, and provider preflight
10. Paper-facing exports, summaries, and figures

CLI scripts (`run_experiment_from_config.py`, `run_task3_verification_from_config.py`, `generate_evaluation_analysis.py`, `compare_run_matrix.py`, `show_run_progress.py`, …) are thin wrappers over these sections.

## Failure Modes the Pipeline Is Designed to Surface

- Confidence-scale drift (v1 `0-100` records appearing in a v2 run).
- Stale prompt rows from a previous prompt-text version.
- Incomplete provider runs (registry not complete) being silently analyzed.
- Weak-template construct-validity gate not completed before paper-facing weak-intent claims.
- Source-modality strengthening at high confidence (`p ≥ 0.80` and `p ≥ 0.90`).

The analysis command (`scripts/generate_evaluation_analysis.py`) fails closed on each of these unless an explicit diagnostic flag is set.
