# Ablation proposals for the resubmission

These are the intended studies confirmed by the author. The implementation
is available; the new measurements and manuscript interpretation remain
pending. `conf/rerun/final.yaml` selects one hosted and seven local models for
batching/context, and one hosted plus two local models for weak phrasing.
The final main campaign uses one item per request and five stochastic samples
in addition to the deterministic answer.

| Study | Controlled comparison | Evidence to report |
| --- | --- | --- |
| Batching | MLM-TAPT/MUST; sizes 1, 4, 16, with grouped and sibling-separated arms at sizes 4 and 16; deterministic Task 2 | Strict/broad text strengthening, label accuracy, unreadable and failed outputs; paired changes against single-item delivery, counts and conditional 95% intervals. This does not establish invariance of other tasks or stochastic UQ. |
| Document context | 180 AI-reviewed corrected PURE capabilities; bare versus original heading, marker and neighbors plus the context instruction; deterministic Task 2, single-item requests | Outcomes and paired document-minus-bare changes, including strata where synthetic wording agrees/conflicts with original M/O markers. No claim of renewed human validation or isolated information-only effects. |
| Weak phrasing | All 180 NICE capabilities × four weak templates; identical sampling and delivery settings within each model | Label and generated-text strengthening by template; readable and unresolved counts; paired deterministic text-strengthening changes against `useful_if`. |

The detailed context proposal is in [context_ablation.md](context_ablation.md).
The batching rationale and remaining extensions are in [TODO.md](../TODO.md).
Marker flipping and additional context factors are proposed extensions,
not part of the default two-arm context run. Commitment-specific fine-tuning
remains a future research question, not an experiment this command runs.

Sampling repetitions use distinct reproducible seeds; every delivered
request and returned response is retained. Estimates use each metric's
readable-output denominator and separately count missing/unreadable cases.
Batching deltas pair the same item and resample capabilities because request
membership changes. Context and phrasing deltas preserve matched capabilities
and use request clusters when available. Seed-clustered sensitivity intervals
accompany the per-arm rates. These intervals address dependence within the
chosen unit; they are not population-level guarantees across model families.

TODO after rerunning: inspect failures and repeat human output validation;
check that differences survive exclusions and the alternative wording rule;
then integrate the results without attributing causality to response length
or inferring unstated stakeholder intent from synthetic transformations.

The proposed ablation figure should show paired changes with a zero-change
reference, confidence intervals, and counts, grouped by model and comparison.
Keep context results separate from the main benchmark's pooled rates.
Figure 2's embedding analysis is separate: explicitly distinguish predicting
the deterministic answer's label from judging the embedded sampled answer.
