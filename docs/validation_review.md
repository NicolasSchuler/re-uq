# Validation status and consistency review

## Current readiness (2026-09-11)

The earlier author confirmation below is historical, not a sign-off on the
current revision. The latest source-based inspection found lost subjects and
conditions in PURE and a residual obligation in S0544. Current PURE context
results must not be used as final evidence. The builder now rejects mechanical
red flags before writing outputs; it does not establish semantic equivalence.

The author delegated the renewed PURE review to the assistant and an
independent subagent. All 180 proposals were checked: 129 accepted and 51
revised, followed by main-agent reconciliation. The accepted corrections are
applied to the seed review and all 720 items have been rebuilt and checked.
See [AI-assisted row-level review](internal/pure_capability_ai_review.md) for decisions,
source evidence and remaining underspecification. This is **not renewed human
validation**, expert agreement, or validation of model outputs. Earlier inputs
and exports are retained under `outputs/pure_before_capability_review/`.
Both context arms need fresh single-item generations; old responses do not
answer the corrected prompts.

The author confirmed on 2026-09-04 that the human validation had been
completed and would be repeated before submission. This covers the reviewed
capabilities, benchmark transformations, weak-template judgments, and
generated-text judgments discussed in the manuscript review. This records
the author's confirmation; the date is the reporting date. No second human
rater, inter-rater agreement statistic, or new experimental result is implied.

The original two LLM-assisted weak-template reviews remain identified as
such in `weak_modality_construct_review.csv`. Separate `AUTHOR` rows record
the human confirmation. The completed human review includes the 180 PURE
capability clauses used in the context ablation.

## Assistant review, 2026-09-04

**Correct: construction and operational template mapping.** I checked all
3,600 generated items: NICE and MLM-TAPT, each with 180 capabilities × four
modalities × two mandatory-keyword variants, plus 720 PURE items. Every
source statement, mandatory candidate, Task 1 gold label, and Task 2 gold
modality matches the current builder. MUST and SHALL cells use the matching
keyword in their candidate. This is a consistency check, distinct from the
author's semantic validation of the capabilities.

I also reviewed the four weak phrasings. The two conditional wishes express
desirability, and the low-priority and possible-future enhancements express
tentative scope. All four are consistent with the study's operational
weak-intent category and weaker than its explicit SHOULD recommendation.
Priority and temporal scope alone would not determine obligation in a real
document; surrounding requirements can change the interpretation.

**Qualified: generated-text wording checks.** The six worked examples agree
with the implementation after replacing “preserved” with “not flagged” where
appropriate. The rules are reproducible diagnostics, not a validated general
semantic judge. For example, “It would be useful if the system could export
reports, but it must retain records” was assigned weak intent in the earlier
implementation. The 2026-09-11 correction gives positive obligations and
recommendations precedence over weak phrases and flags mixed wording. This is
still a lexical diagnostic, not a resolution of scope or negation semantics.
Unknown wording is excluded from the readable-text
denominator. Strict and broad results therefore are sensitivity analyses;
they do not establish lower and upper bounds on semantic error.

TODO after the rerun: repeat the author's output review, sampling flagged,
unflagged, mixed-cue, and unreadable cases by model and source modality.
Save judgments and report the reviewed counts and disagreements. Do not
infer parser precision, recall, or agreement from this consistency review.
