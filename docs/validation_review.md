# Validation status and consistency review

## Current status

The authors reviewed the benchmark before resubmission: the reviewed
capabilities and benchmark transformations, the weak-template judgments, all
180 PURE capability clauses used in the context ablation (129 kept, 51
revised; `docs/pure_capability_revisions.csv`), and the wording checks'
decisions on generated outputs. The PURE builder rejects mechanical red flags
before writing items. No second rater or inter-rater agreement statistic is
claimed.

The original two LLM-assisted weak-template reviews remain identified as
such in `weak_modality_construct_review.csv`. Separate `AUTHOR` rows record
the human confirmation.

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

The authors reviewed the wording checks' decisions on generated outputs of
the final campaign before resubmission (see Current status). Parser precision,
recall or agreement are not inferred from this consistency review.
