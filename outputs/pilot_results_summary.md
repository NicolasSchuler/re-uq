# Pilot Results Summary

Generated for Oracle review of whether the modality-UQ evaluation can proceed safely.

## Run Context
- Latest pilot run id: `pilot-20260518-200442-95d70b36`
- Raw pilot rows in latest run: `960`
- Parse status counts: `{'ok': 960}`
- Mean latency seconds: `10.77`
- Pilot size: 20 seeds x 4 modalities = 80 benchmark items.
- Each task used deterministic decoding plus 5 stochastic samples.

## Deterministic Task 1
- Accuracy: `1.000` over `80` rows.
- Crosstab:
| source_modality   |   no |   yes |
|:------------------|-----:|------:|
| mandatory         |    0 |    20 |
| nice_to_have      |   20 |     0 |
| optional          |   20 |     0 |
| recommended       |   20 |     0 |

## Deterministic Task 2
- Accuracy: `0.750` over `80` rows.
- Crosstab:
| source_modality   |   mandatory |   optional |   recommended |
|:------------------|------------:|-----------:|--------------:|
| mandatory         |          20 |          0 |             0 |
| nice_to_have      |           0 |          0 |            20 |
| optional          |           0 |         20 |             0 |
| recommended       |           0 |          0 |            20 |
- Key observation: all 20 deterministic `nice_to_have` items were predicted as `recommended` with confidence 95.

## Stochastic Task 2 Counts
| source_modality   |   mandatory |   optional |   recommended |
|:------------------|------------:|-----------:|--------------:|
| mandatory         |         100 |          0 |             0 |
| nice_to_have      |           5 |          4 |            91 |
| optional          |           5 |         94 |             1 |
| recommended       |           4 |          0 |            96 |

## Metric Summary
| model           | task   | uq_method              |   n |   accuracy |   f1_or_macro_f1 |   over_commitment |   high_conf_overcommit_80 |   high_conf_overcommit_90 |   brier |    ece |   error_detection_auroc |   monotonicity_violations |
|:----------------|:-------|:-----------------------|----:|-----------:|-----------------:|------------------:|--------------------------:|--------------------------:|--------:|-------:|------------------------:|--------------------------:|
| qwen/qwen3.5-9b | task1  | label_self_consistency |  80 |       1    |           1      |                   |                    0      |                      0    |  0.0035 | 0.0125 |                nan      |                      0.2  |
| qwen/qwen3.5-9b | task1  | predictive_entropy     |  80 |       1    |           1      |                   |                    0      |                      0    |  0.0035 | 0.0125 |                nan      |                      0.2  |
| qwen/qwen3.5-9b | task1  | variation_ratio        |  80 |       1    |           1      |                   |                    0      |                      0    |  0.0035 | 0.0125 |                nan      |                      0.2  |
| qwen/qwen3.5-9b | task1  | verbalized_confidence  |  80 |       1    |           1      |                   |                    0      |                      0    |  0.0011 | 0.0219 |                nan      |                      0.95 |
| qwen/qwen3.5-9b | task2  | modality_consistency   |  80 |       0.75 |           0.6667 |              0.25 |                    0.2375 |                      0.15 |  0.2155 | 0.2025 |                  0.6208 |                           |
| qwen/qwen3.5-9b | task2  | predictive_entropy     |  80 |       0.75 |           0.6667 |              0.25 |                    0.2375 |                      0.15 |  0.2155 | 0.2025 |                  0.6208 |                           |
| qwen/qwen3.5-9b | task2  | variation_ratio        |  80 |       0.75 |           0.6667 |              0.25 |                    0.2375 |                      0.15 |  0.2155 | 0.2025 |                  0.6208 |                           |
| qwen/qwen3.5-9b | task2  | verbalized_confidence  |  80 |       0.75 |           0.6667 |              0.25 |                    0.25   |                      0.25 |  0.2258 | 0.2344 |                  0.9583 |                           |

## outputs/prompt_sensitivity_summary.csv
```
model,prompt_run_id,n,accuracy,weak_source_high_p_yes_80,weak_source_high_p_yes_90,mean_weak_p_yes
qwen/qwen3.5-9b:default,prompt-sensitivity-20260518-204754-e546b407-default,80,1.0,0.0,0.0,0.02916666666666669
qwen/qwen3.5-9b:strict,prompt-sensitivity-20260518-204754-e546b407-strict,80,1.0,0.0,0.0,0.0
```

## outputs/task2_prompt_sensitivity_summary.csv
```
model,prompt_run_id,n,valid_n,parse_success_rate,accuracy,nice_to_have_n,nice_to_have_accuracy,nice_to_have_to_recommended_rate,over_commitment,high_conf_overcommit_80,high_conf_overcommit_90
qwen/qwen3.5-9b:task2_default,task2-prompt-sensitivity-20260518-205559-4a662202-default,20,20,1.0,0.0,20,0.0,1.0,1.0,1.0,1.0
qwen/qwen3.5-9b:task2_labels_only,task2-prompt-sensitivity-20260518-205559-4a662202-labels_only,20,20,1.0,0.0,20,0.0,1.0,1.0,1.0,1.0
```

## Weak-Modality Robustness Probe

Run id: `weak-modality-probe-20260518-220538-6a39dfa8`

Purpose: check whether the deterministic Task 2 `nice_to_have` failure was tied to the single source wording `It would be useful if...`.

Pre-model sanity check: the four weak templates were reviewed and marked by R1 as weaker than `SHOULD` / `recommended`.

| template_id | n | parse_success | predicted_nice_to_have | predicted_recommended | predicted_optional | mean_confidence |
|:--|--:|--:|--:|--:|--:|--:|
| `useful_if` | 20 | 20/20 | 0/20 | 20/20 | 0/20 | 0.95 |
| `nice_if` | 20 | 20/20 | 0/20 | 18/20 | 2/20 | 0.95 |
| `low_priority_enhancement` | 20 | 20/20 | 0/20 | 3/20 | 17/20 | 0.95 |
| `future_enhancement` | 20 | 20/20 | 0/20 | 0/20 | 20/20 | 0.95 |

Observation: across all four weak stakeholder-intent templates, the model never preserved the gold label `nice_to_have`.

Observation: the upgraded class is lexical-template sensitive. Desiderative wordings (`useful_if`, `nice_if`) mostly became `recommended`, whereas enhancement/future-oriented wordings mostly became `optional`.

Observation: all weak-template predictions were high-confidence over-commitments relative to the study's ordinal scale, because both `optional` and `recommended` are stronger than `nice_to_have`.

Hypothesis: the model does not represent `nice_to_have` as a stable output class under the current Task 2 extraction prompt; it normalizes weak stakeholder intent into the nearest familiar requirement category.

Caveat: this robustness probe used one model and deterministic decoding on the 20 pilot seeds. It supports proceeding to full runs, but should be reported as formative construct-validity evidence rather than a headline result.

## Proposed Manuscript Statement

> A weak-modality robustness probe showed that the model never preserved weak stakeholder intent across four weak-intent phrasings; however, the upgraded class varied lexically between `optional` and `recommended`. This suggests that the extraction failure is not specific to the phrase "it would be useful if", but that the exact stronger modality assigned to weak stakeholder intent is wording-sensitive.

Shorter variant:

> In a pilot robustness probe, weak stakeholder intent was never preserved across four weak-intent phrasings; the model instead assigned either `optional` or `recommended` with high confidence.

## Strengthened Paper Interpretation

Primary observable result:

> high-confidence over-commitment of weak stakeholder intent

Discussion shorthand:

> commitment normalization

Use `commitment normalization` as an interpretation of the observed behavior, not as a claim about the model's internal mechanism. The safer paper claim is that the model preserves functional content while assigning a stronger requirement modality or priority class to weak stakeholder intent.

Paper-facing class wording:

- Keep `nice_to_have` as an implementation label.
- In manuscript prose, call this class `weak stakeholder intent` or `weak desiderative intent`.
- State explicitly that this class is not a standardized RFC 2119 modality equivalent to `MUST`, `SHOULD`, or `MAY`.
- Ground the class as an operational weak-intent condition below specification-level optionality or recommendation.

Recommended full-run framing:

> Requirements engineering is not only about identifying functional content; it is also about preserving stakeholder commitment. A statement such as "it would be useful if the system could export reports" and a statement such as "the system should export reports" may refer to the same capability, but they convey different levels of priority, obligation, and delivery expectation. We study whether LLM-assisted requirements extraction preserves this distinction, and whether lightweight uncertainty estimates expose or obscure failures to do so.

Recommended results wording:

> In the pilot, the model correctly rejected unsupported mandatory interpretations in Task 1, but failed to preserve weak stakeholder intent in Task 2. Across four weak-intent templates, it never predicted the weak `nice_to_have` label; instead, it normalized the source into `recommended` or `optional` with high confidence. This suggests that the salient failure is not loss of functional content, but high-confidence over-commitment of stakeholder intent.

Recommended implementation-facing note:

> Before full runs, prefer balancing the weak-intent condition across the four already sanity-checked templates. This keeps the 120 x 4 benchmark size unchanged while reducing the phrase-artifact critique. The exact upgraded class should still be analyzed by weak-template wording, because the pilot showed lexical sensitivity.

Safe caveat:

> We do not claim that `nice_to_have` is a standardized normative keyword equivalent to RFC 2119 terms. Instead, we use it as an operational weak-intent class, grounded in requirements-prioritization practice and validated through pre-run template checks. The central result is therefore not that models violate RFC semantics for `nice_to_have`, but that they may fail to preserve weak stakeholder commitment when translating elicitation-like statements into requirement labels.

## outputs/logprob_probe.json
```
{
  "endpoint": "http://127.0.0.1:1234/v1/responses",
  "error": "",
  "host": "http://127.0.0.1:1234/v1",
  "include": [
    "message.output_text.logprobs"
  ],
  "latency_s": 1.7806055000110064,
  "model": "qwen/qwen3.5-9b",
  "raw_text": "{\"decision\":\"yes\",\"confidence\":100,\"brief_reason\":\"probe\"}",
  "supported": true,
  "token_count": 16,
  "top_logprobs": 5
}
```

## Current Local Interpretation To Challenge
- Observation: Task 1 mandatory-entailment is clean in the pilot.
- Observation: Task 2 extraction systematically upgrades `nice_to_have` to `recommended`.
- Observation: the labels-only Task 2 prompt, which lists labels without examples or deterministic mapping rules, produced the same upgrade pattern.
- Hypothesis: the model collapses weak desirability language into recommendation language during extraction.
- Open question: whether the current benchmark has sufficient objectivity and construct validity to proceed to full execution, or whether the weakest modality should receive additional independent justification or robustness variants first.
