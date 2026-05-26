# Evaluation Specification

This repository supports a short empirical study of modality-conditioned uncertainty in LLM-assisted requirements engineering. The evaluation asks whether a model preserves stakeholder commitment when the functional capability is held constant and only the source modality changes.

## Research Focus

Primary claim:

> LLMs may preserve the functional content of a requirement while confidently strengthening weak stakeholder intent into firmer requirement categories.

The study is not a broad UQ survey and does not claim that all uncertainty methods fail. It evaluates a compact, RE-specific failure mode: high-confidence over-commitment.

Paper-facing terminology:

- `mandatory`: obligation, expressed with `MUST` in the main benchmark.
- `recommended`: recommendation, expressed with `SHOULD`.
- `optional`: optionality or permission, expressed with `MAY`.
- `nice_to_have`: an operational weak stakeholder-intent class, not an RFC-standard keyword level.

`SHALL` is a robustness benchmark variant only. The main result should use the `MUST` benchmark unless explicitly stated otherwise.

## Benchmark Design

Use reviewed requirement seeds from two datasets:

- NICE/PROMISE-derived requirements, the default dataset.
- `limsc/mlm-tapt-requirements`, the co-primary dataset after the same manual review gate.

For each accepted seed capability, generate four controlled source variants:

| Source condition | Example |
| --- | --- |
| `mandatory` | `The system MUST export reports.` |
| `recommended` | `The system SHOULD export reports.` |
| `optional` | `The system MAY export reports.` |
| `nice_to_have` | `It would be useful if the system could export reports.` |

The benchmark holds functional content constant and varies only requirement strength. This makes modality preservation and over-commitment measurable without relying on naturally occurring modality labels.

Before treating weak-intent findings as paper-ready, complete `docs/weak_modality_construct_review.csv` with two reviewer judgments. Every weak template should be marked weaker than `SHOULD/recommended` on the study scale.

## Tasks

Task 1 is a control task: mandatory-requirement entailment.

Given a source statement and a mandatory candidate requirement, the model answers whether the mandatory candidate is faithfully entailed by the source. The expected risk metric is unsupported mandatory acceptance among non-mandatory sources.

Task 2 is the main empirical task: modality-preserving extraction plus generated-text modality drift.

Given a source statement, the model extracts one requirement and labels its modality. The expected failure mode is strengthening in either the declared label or the generated requirement text, especially when a weak stakeholder-intent source is rewritten into `optional`, `recommended`, or `mandatory` language.

Task 3 is a post-run blind text-audit diagnostic.

After a complete Task 2 run, the same model judges whether its extracted requirement text `preserves`, `strengthens`, `weakens`, or changes the source content. The official prompt does not reveal the Task 2 declared modality. Declared-modality Task 3 prompts are anchoring ablations only. Task 3 is not independent verification and must not be framed as correcting the benchmark outputs.

## Confidence Contract

Current v2 prompts use probability-scale confidence:

```json
{
  "confidence": 0.95
}
```

For `v2-conf01`, `v2-instructor-conf01`, and external v2 probe prompts:

- confidence is a numeric decimal from `0.0` to `1.0`;
- confidence is for the selected label or decision;
- percentages such as `95` and strings such as `"95%"` are invalid.

Legacy unmarked raw rows may still contain `0-100` confidence values. New v2 raw records must include `confidence_scale=0_1` so scoring does not divide probability-scale confidence by 100.

## UQ Signals

Use lightweight black-box uncertainty signals:

- `verbalized_confidence`: model-reported confidence for the selected label.
- `label_self_consistency`: Task 1 stochastic label distribution.
- `modality_consistency`: Task 2 stochastic label distribution.
- `relation_consistency`: Task 3 stochastic relation distribution.
- `predictive_entropy`: normalized entropy of the same stochastic label distribution.
- `variation_ratio`: `1 - p_majority` over the same stochastic label distribution.
- `acse_semantic_entropy`: ACSE-inspired semantic-cluster dispersion over the stochastic answer texts. With the default five stochastic samples, use this as a ranking or triage signal, not as a calibrated accept/abstain rule.
- `model_ensemble_disagreement`: available only when multiple deterministic model runs cover the same item.
- `token_logprob_confidence`: optional and non-headline until endpoint support and scoring are stable.

Entropy, variation ratio, and consistency are related summaries of the same stochastic distribution. Do not present them as independent prediction methods.
The ACSE-inspired score adds a separate text-clustering view of the same five samples. The current implementation uses local TF-IDF character n-gram embeddings plus average-linkage HAC as a lightweight proxy for sentence-encoder clustering; formal ACSE-style threshold guarantees would require a held-out calibration set and a stronger embedding backend.
For local Apple Silicon runs, set `RE_UQ_ACSE_EMBEDDING_BACKEND=mlx` after installing `mlx-embeddings`; the default MLX model is `mlx-community/Qwen3-Embedding-0.6B-8bit`, with `RE_UQ_ACSE_MLX_MODEL` available for alternatives such as the 4-bit DWQ variant.
The headless analysis exports normalized ACSE scores and an empirical seed-split calibration table. Use these outputs to discuss triage behavior and accepted-error/coverage trade-offs, not as a conformal guarantee unless a predeclared calibration protocol is added.

## Metrics

Core correctness and calibration:

- accuracy and macro-F1;
- Brier score and ECE;
- error-detection AUROC using uncertainty scores;
- selective error after deferring the most uncertain 10% and 20% of predictions.

Task 1 headline risk:

- `unsupported_mandatory_acceptance@tau`: among non-mandatory sources, the fraction assigned `p_yes >= tau` for the mandatory candidate.

Task 2 headline risks:

- `HC-OC_all@tau`: strengthening with confidence at least `tau`, denominator all valid Task 2 outputs.
- `HC-OC_overcommittable@tau`: same numerator, excluding mandatory source rows from the denominator.
- `weak_strengthening@tau`: among weak stakeholder-intent sources, prediction as any stronger modality with confidence at least `tau`.
- `label_correct_text_overcommit@tau`: among label-correct Task 2 outputs, generated requirement text still strengthens the source at confidence at least `tau`.
- `strict_text_over_commitment`: generated-text strengthening using explicit modal or weak-phrase evidence, excluding the generic system-verb fallback.

Task 3 diagnostic metrics:

- relation macro-F1;
- blind strengthening recall;
- blind false-preserve rate;
- evidence-phrase-in-source rate.

Report bootstrap confidence intervals over seeds for accuracy, Brier score, unsupported mandatory acceptance, `HC-OC_overcommittable`, `weak_strengthening`, and `label_correct_text_overcommit`.

## Run Protocol

Use the provider-aware CLI for reproducible Task 1 and Task 2 provider/model matrices:

```bash
.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile local_llama_cpp \
  --model qwen/qwen3.5-9b \
  --dataset nice \
  --mode smoke
```

Then run `--mode full` only after provider preflight and smoke parse checks pass. Use `--task task1`, `--task task2`, or `--task both` to isolate failures or run the primary matrix.

Task 3 should run only after a complete Task 2 full run:

```bash
.venv/bin/python scripts/run_task3_verification_from_config.py \
  --config run_configs/current_run.json \
  --profile local_llama_cpp \
  --model qwen/qwen3.5-9b \
  --dataset nice \
  --source-run-id RUN_ID \
  --audit-mode blind \
  --mode full
```

Generate paper-facing analysis artifacts from a complete run with:

```bash
.venv/bin/python scripts/generate_evaluation_analysis.py \
  --dataset nice \
  --variant must \
  --run-id RUN_ID \
  --task3-run-id TASK3_RUN_ID \
  --task3-audit-mode blind
```

The notebooks remain useful for inspection, but `docs/reproduction.md` is the canonical command-first reproduction guide.

## External Probe

The external AI service probe is a small blind Task 2 check for larger web models. It uses 20 pilot seeds, the four main benchmark conditions, and three additional weak-intent phrasings.

Curated external reports are paper-ready only when:

- the raw output was generated with the current `0.0-1.0` confidence prompt;
- the evaluator reports zero invalid confidence values;
- no IDs are missing, duplicated, or extra;
- the report includes prompt version, confidence scale, raw-output SHA-256, gold-key SHA-256, and prompt SHA-256.

Old reports generated with `0-100` confidence are legacy diagnostics and should not be cited as current paper results.

## Paper-Readiness Checklist

Before using results in the IST manuscript:

- inspect the reviewed seed files and benchmark item CSVs;
- verify the weak-intent construct review;
- run smoke checks for every provider/profile;
- run a clean full matrix after the confidence-scale fix;
- confirm parse success, missing-batch handling, and run registry completeness;
- recompute metrics from the current raw outputs;
- inspect bootstrap CIs, qualitative examples, and generated figures;
- treat the current 468-row metric tables as stale until replaced by a post-fix full run.

## Verification

Use these local checks before committing or treating the implementation as stable:

```bash
git diff --check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m coverage run --branch --source=scripts -m unittest discover -s tests -v
.venv/bin/python -m coverage report -m
```
