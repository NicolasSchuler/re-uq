# Benchmark and wording-check validation

The authors reviewed the benchmark before resubmission: the main capabilities
and controlled transformations, the weak-template judgments, all 180 PURE
capability clauses used in the context ablation, and the wording checks'
decisions on 100 generated outputs covering every model and source condition.
No independent second-rater assessment or inter-rater agreement statistic is
claimed.

## Reviewed inputs and construction checks

- The reviewed seed tables retain inclusion decisions and final capability
  clauses. [Benchmark ground truth](benchmark_ground_truth.md) traces the main
  datasets through the templates and gold labels.
- The PURE review kept 129 clauses and revised 51. The changes are recorded in
  [`pure_capability_revisions.csv`](pure_capability_revisions.csv), and its 720
  items use the corrected clauses. The builder checks mechanical red flags
  before writing items.
- [`weak_modality_construct_review.csv`](weak_modality_construct_review.csv)
  retains the original `R1` and `R2` LLM-assisted reviews, identified by role.
  Separate `AUTHOR` rows record human confirmation. The four weak phrasings
  belong to the study's operational weak-intent class; priority or temporal
  scope alone does not establish obligation in arbitrary documents.
- A separate construction consistency check covered all 3,600 generated items:
  180 capabilities × four source modalities × two keyword variants in each of
  the two main datasets, plus 720 PURE items. Source statements, mandatory
  candidates, Task 1 gold labels and Task 2 gold modalities matched the builder.
  This checks the transformations, not the semantic validity of the capabilities.

## Limits of the wording checks

The strict and broad rules are reproducible lexical diagnostics, not a validated
general semantic judge. A result unflagged by the strict rule does not establish
preservation of functional content or commitment. Broad-only classifications
depend on the convention that bare system-verb wording expresses obligation;
unclassified wording is separately excluded from the readable-text denominator.
Consequently, the two rules are sensitivity analyses, not lower and upper bounds
on semantic error.

Positive obligations and recommendations take precedence over weak phrases in
mixed wording. For example, “It would be useful if the system could export
reports, but it must retain records” is flagged as mixed and assigned the
positive obligation. This precedence does not resolve the scope of each clause
or general negation semantics.

The authors' 100-output review supports inspection of these decisions, but no
parser precision, recall or agreement estimate is reported from that sample.
The exact rules and handling of unclassified outputs are documented in
[evaluation](evaluation.md) and [aggregation](aggregation.md).
