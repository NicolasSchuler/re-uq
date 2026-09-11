# Ablation proposals for the resubmission

These are the intended studies confirmed by the author. The implementation
is available; the new measurements and manuscript interpretation remain
pending. `scripts/rerun_all.py` executes them using the first configured ZAI
model and the first configured local llama model. Choose representatives from
different families if family diversity is part of the argument.

| Study | Controlled comparison | Evidence to report |
| --- | --- | --- |
| Batching | MLM-TAPT/MUST; grouped 16-item, shuffled 16-item, and single-item requests; deterministic Task 2 | Strict/broad text strengthening, label accuracy, unreadable and failed outputs; paired changes against grouped delivery, counts and 95% intervals. |
| Document context | 180 human-validated PURE capabilities; bare versus original document heading, marker and neighbors; deterministic Task 2, grouped batches of 16 | Outcomes and paired document-minus-bare changes, including strata where synthetic wording agrees/conflicts with original M/O markers. |
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
