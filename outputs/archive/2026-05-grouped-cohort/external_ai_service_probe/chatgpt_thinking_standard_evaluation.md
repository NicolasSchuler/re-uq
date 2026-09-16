# External Probe Evaluation: ChatGPT thinking standard

Legacy/non-paper-ready status: this report does not satisfy the current external-probe contract.
Blockers: invalid_confidence.

## Validation

- Output rows: 140
- Gold rows: 140
- Parse errors: 0
- Duplicate IDs: 0
- Missing IDs: 0
- Extra IDs: 0
- Invalid labels: 0
- Invalid confidence values: 140
- Confidence scale: 0_1
- Prompt version: external-task2-v2-conf01
- Evaluated at UTC: 2026-05-22T10:19:44Z
- Raw output SHA-256: 0576772179ab8b26550e250e82a51b2186c20b33aeaff77be5586e7e3c3019d1
- Gold key SHA-256: dec62b6af9d4112863e1111e9f8bb988c608bfa8ceea5c2b63f5452d20945a2d
- Prompt SHA-256: 1511e86c16ba68d1a8baf3888ebcfbad3093a31b67042cd97c43267c0bc0fef3
- Paper-ready under current contract: no
- Paper-ready blockers: invalid_confidence

## Overall

- Accuracy: 0.857
- Over-commitment rate: 0.143
- Under-commitment rate: 0.000
- High-confidence over-commitment >= 0.80: 0.143
- High-confidence over-commitment >= 0.90: 0.143
- Mean confidence: 95.143
- Weak-modality accuracy: 0.750
- Weak-modality over-commitment rate: 0.250
- Text-modality accuracy: 0.143
- Label-text consistency: 0.286
- Text-level over-commitment rate: 0.857
- Text-level high-confidence over-commitment >= 0.80: 0.857
- Text-level high-confidence over-commitment >= 0.90: 0.857
- Weak text-modality accuracy: 0.000

## By Source Condition

| source_condition              |   n |   accuracy |   overcommit_rate |   undercommit_rate |   high_conf_overcommit_80 |   high_conf_overcommit_90 |   text_modality_accuracy |   label_text_consistency |   text_overcommit_rate |   text_undercommit_rate |   text_high_conf_overcommit_80 |   text_high_conf_overcommit_90 |   mean_confidence |
|:------------------------------|----:|-----------:|------------------:|-------------------:|--------------------------:|--------------------------:|-------------------------:|-------------------------:|-----------------------:|------------------------:|-------------------------------:|-------------------------------:|------------------:|
| mandatory                     |  20 |      1.000 |             0.000 |              0.000 |                     0.000 |                     0.000 |                    1.000 |                    1.000 |                  0.000 |                   0.000 |                          0.000 |                          0.000 |            98.000 |
| nice_to_have                  |  20 |      0.000 |             1.000 |              0.000 |                     1.000 |                     1.000 |                    0.000 |                    1.000 |                  1.000 |                   0.000 |                          1.000 |                          1.000 |            90.000 |
| optional                      |  20 |      1.000 |             0.000 |              0.000 |                     0.000 |                     0.000 |                    0.000 |                    0.000 |                  1.000 |                   0.000 |                          1.000 |                          1.000 |            98.000 |
| recommended                   |  20 |      1.000 |             0.000 |              0.000 |                     0.000 |                     0.000 |                    0.000 |                    0.000 |                  1.000 |                   0.000 |                          1.000 |                          1.000 |            98.000 |
| weak_future_enhancement       |  20 |      1.000 |             0.000 |              0.000 |                     0.000 |                     0.000 |                    0.000 |                    0.000 |                  1.000 |                   0.000 |                          1.000 |                          1.000 |            94.000 |
| weak_low_priority_enhancement |  20 |      1.000 |             0.000 |              0.000 |                     0.000 |                     0.000 |                    0.000 |                    0.000 |                  1.000 |                   0.000 |                          1.000 |                          1.000 |            94.000 |
| weak_nice_if                  |  20 |      1.000 |             0.000 |              0.000 |                     0.000 |                     0.000 |                    0.000 |                    0.000 |                  1.000 |                   0.000 |                          1.000 |                          1.000 |            94.000 |

## Confusion Matrix

| task2_gold_modality   |   mandatory |   nice_to_have |   optional |   recommended |
|:----------------------|------------:|---------------:|-----------:|--------------:|
| mandatory             |          20 |              0 |          0 |             0 |
| nice_to_have          |           0 |             60 |          0 |            20 |
| optional              |           0 |              0 |         20 |             0 |
| recommended           |           0 |              0 |          0 |            20 |

## Text-Modality Confusion Matrix

| task2_gold_modality   |   mandatory |   optional |   recommended |
|:----------------------|------------:|-----------:|--------------:|
| mandatory             |          20 |          0 |             0 |
| nice_to_have          |           0 |         40 |            40 |
| optional              |          20 |          0 |             0 |
| recommended           |          20 |          0 |             0 |
