# Validation status and consistency review

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
reports, but it must retain records” is assigned weak intent because the weak
phrase takes precedence, although a second mandatory clause is present.
The implementation records multiple modal cues, but neither strengthening
rule flags this case. Unknown wording is excluded from the readable-text
denominator. Strict and broad results therefore are sensitivity analyses;
they do not establish lower and upper bounds on semantic error.

TODO after the rerun: repeat the author's output review, sampling flagged,
unflagged, mixed-cue, and unreadable cases by model and source modality.
Save judgments and report the reviewed counts and disagreements. Do not
infer parser precision, recall, or agreement from this consistency review.
