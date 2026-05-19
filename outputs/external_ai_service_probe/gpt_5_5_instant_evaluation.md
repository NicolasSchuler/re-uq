# External Probe Evaluation: GPT-5.5 instant

## Validation

- Output rows: 140
- Gold rows: 140
- Parse errors: 0
- Duplicate IDs: 0
- Missing IDs: 0
- Extra IDs: 0
- Invalid labels: 0
- Invalid confidence values: 0

## Overall

- Accuracy: 0.857
- Over-commitment rate: 0.143
- Under-commitment rate: 0.000
- High-confidence over-commitment >= 0.80: 0.143
- High-confidence over-commitment >= 0.90: 0.000
- Mean confidence: 84.9
- Weak-modality accuracy: 0.750
- Weak-modality over-commitment rate: 0.250
- Text-modality accuracy: 0.714
- Label-text consistency: 0.571
- Text-level over-commitment rate: 0.286
- Text-level high-confidence over-commitment >= 0.80: 0.000
- Text-level high-confidence over-commitment >= 0.90: 0.000
- Weak text-modality accuracy: 0.500

## By Source Condition

| source_condition              |   n |   accuracy |   overcommit_rate |   undercommit_rate |   high_conf_overcommit_80 |   high_conf_overcommit_90 |   text_modality_accuracy |   label_text_consistency |   text_overcommit_rate |   text_undercommit_rate |   text_high_conf_overcommit_80 |   text_high_conf_overcommit_90 |   mean_confidence |
|:------------------------------|----:|-----------:|------------------:|-------------------:|--------------------------:|--------------------------:|-------------------------:|-------------------------:|-----------------------:|------------------------:|-------------------------------:|-------------------------------:|------------------:|
| mandatory                     |  20 |      1.000 |             0.000 |              0.000 |                     0.000 |                     0.000 |                    1.000 |                    1.000 |                  0.000 |                   0.000 |                          0.000 |                          0.000 |            98.000 |
| nice_to_have                  |  20 |      0.000 |             1.000 |              0.000 |                     1.000 |                     0.000 |                    1.000 |                    0.000 |                  0.000 |                   0.000 |                          0.000 |                          0.000 |            88.000 |
| optional                      |  20 |      1.000 |             0.000 |              0.000 |                     0.000 |                     0.000 |                    1.000 |                    1.000 |                  0.000 |                   0.000 |                          0.000 |                          0.000 |            88.000 |
| recommended                   |  20 |      1.000 |             0.000 |              0.000 |                     0.000 |                     0.000 |                    1.000 |                    1.000 |                  0.000 |                   0.000 |                          0.000 |                          0.000 |            95.000 |
| weak_future_enhancement       |  20 |      1.000 |             0.000 |              0.000 |                     0.000 |                     0.000 |                    1.000 |                    1.000 |                  0.000 |                   0.000 |                          0.000 |                          0.000 |            75.000 |
| weak_low_priority_enhancement |  20 |      1.000 |             0.000 |              0.000 |                     0.000 |                     0.000 |                    0.000 |                    0.000 |                  1.000 |                   0.000 |                          0.000 |                          0.000 |            75.000 |
| weak_nice_if                  |  20 |      1.000 |             0.000 |              0.000 |                     0.000 |                     0.000 |                    0.000 |                    0.000 |                  1.000 |                   0.000 |                          0.000 |                          0.000 |            75.000 |

## Confusion Matrix

| task2_gold_modality   |   mandatory |   nice_to_have |   optional |   recommended |
|:----------------------|------------:|---------------:|-----------:|--------------:|
| mandatory             |          20 |              0 |          0 |             0 |
| nice_to_have          |           0 |             60 |         20 |             0 |
| optional              |           0 |              0 |         20 |             0 |
| recommended           |           0 |              0 |          0 |            20 |

## Text-Modality Confusion Matrix

| task2_gold_modality   |   mandatory |   nice_to_have |   optional |   recommended |
|:----------------------|------------:|---------------:|-----------:|--------------:|
| mandatory             |          20 |              0 |          0 |             0 |
| nice_to_have          |           0 |             40 |         40 |             0 |
| optional              |           0 |              0 |         20 |             0 |
| recommended           |           0 |              0 |          0 |            20 |
