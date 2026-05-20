# Paper Framing Notes

These notes capture the current paper-facing interpretation before the full experiment runs.

## Decision Memo

Proceed with the full NICE benchmark and add `limsc/mlm-tapt-requirements` as a second reviewed seed source after the same manual review gate. Complete the two-reviewer weak-modality template check in `docs/weak_modality_construct_review.csv` before treating weak-class claims as paper-ready.

Use **high-confidence over-commitment of weak stakeholder intent** as the primary empirical framing. Keep **commitment normalization** as a discussion-level interpretation, because the current evidence demonstrates output behavior rather than an internal model mechanism.

The prompts, modality labels, and gold labels remain unchanged across seed datasets.

## Paper-Facing Claim

Primary claim:

> LLMs may preserve the functional content of a requirement while high-confidently strengthening weak stakeholder intent into firmer requirement categories.

Safer short claim:

> The main RE risk is high-confidence over-commitment at the elicitation-specification boundary.

Discussion-only interpretation:

> This pattern is consistent with commitment normalization: weak stakeholder language is converted into more specification-like commitment language.

Avoid claiming that all UQ fails. The current evidence supports the narrower claim that **consistency-based UQ can miss systematic, stable errors**, while verbalized confidence and other signals may behave differently.

## Title Variants

- When LLMs Over-Commit: High-Confidence Strengthening of Weak Stakeholder Intent in Requirements Extraction
- Consistency Is Not Correctness in Requirements Extraction: High-Confidence Over-Commitment Under Weak Stakeholder Intent
- Uncertainty Quantification for Modality Preservation in LLM-Assisted Requirements Engineering

## Research Questions

- RQ1: When functional content is held constant and only modality changes, how reliably do local LLMs preserve requirement strength across mandatory, recommended, optional, and weak stakeholder-intent phrasings?
- RQ2: Do lightweight UQ signals reveal high-confidence over-commitment errors, or can models be stable and confident while wrong?
- RQ3: Is weak-intent collapse robust to prompt simplification and lexical variation in weak stakeholder-intent wording?
- RQ4 diagnostic: Can a model detect, in a source-grounded verifier prompt, that its own extracted requirement strengthened or weakened the source modality?

## Contribution Bullets

- A compact controlled RE benchmark that isolates requirement-strength preservation by holding functional content constant and varying modality.
- Pilot evidence of high-confidence over-commitment: the model correctly handles mandatory entailment but fails to preserve the weakest stakeholder-intent class during extraction.
- A UQ-focused diagnostic showing that consistency-based uncertainty is not a sufficient correctness proxy for this RE extraction setting.
- A source-grounded self-verification diagnostic that tests whether models recognize modality strengthening after extraction.
- A construct-validity probe showing that the weak-intent failure is not tied to a single wording, while the exact upgraded class is wording-sensitive.

## Evidence Table

| Evidence | Observation | Interpretation | Status |
|---|---|---|---|
| Task 1 deterministic pilot | 80/80 mandatory-entailment decisions correct. | Treat as a capability/control task: the model can reject unsupported mandatory entailment in the pilot. | Formative pilot evidence |
| Task 2 deterministic pilot | Overall accuracy 75%; all 20 `nice_to_have` items predicted `recommended` with confidence 95. | Main signal: weak stakeholder intent is strengthened during modality-preserving extraction. | Formative pilot evidence |
| Task 2 stochastic pilot | For `nice_to_have`, samples were 91 `recommended`, 4 `optional`, 5 `mandatory`, and 0 `nice_to_have`. | The error is stable under sampling rather than a one-off deterministic artifact. | Formative pilot evidence |
| Task 2 labels-only prompt check | Default and labels-only prompts both produced 20/20 `nice_to_have` to `recommended` on the pilot subset. | The pattern is unlikely to be caused only by examples or deterministic mapping rules in the prompt. | Prompt-validity check |
| Weak-modality robustness probe | Across four weak templates, 0/80 predictions preserved `nice_to_have`; predictions shifted between `optional` and `recommended`. | The weakest class is not preserved across wordings, but the upgraded class is lexically sensitive. | Construct-validity support |
| Logprob probe | LM Studio `/v1/responses` returned token logprobs for the local model. | Token-level UQ is technically possible, but should remain optional and non-headline until integrated. | Capability probe |

## Claim-Evidence Audit

Observation:

- The local pilot model handled explicit mandatory-entailment correctly.
- The same model systematically failed to preserve `nice_to_have` during Task 2 extraction.
- The weak-intent collapse persisted under labels-only prompting and across four weak wording templates.
- Consistency-style UQ signals can be low-uncertainty when the model is systematically wrong.

Hypothesis:

- The model lacks a stable operational category for weak stakeholder-intent language and maps it to more familiar requirement categories.
- The behavior is consistent with commitment normalization, but the experiment does not identify the model's internal mechanism.
- Part of the effect may be weak-phrase or requirements-smell laundering rather than pure modality reasoning.

Recommendation:

- Proceed to full local-model runs with the current benchmark.
- Frame Task 1 as a control and Task 2 as the main empirical result.
- Complete `docs/weak_modality_construct_review.csv` before treating weak-class claims as paper-ready.
- Report `nice_to_have` as an operationalized weak stakeholder-intent class, not as an RFC-standard modality.
- Use denominator-specific risk names in the paper: `unsupported mandatory acceptance@tau` for Task 1, and `HC-OC_all@tau`, `HC-OC_overcommittable@tau`, and `weak strengthening@tau` for Task 2.
- Keep Pearson/Spearman modality-confidence correlations and text-modality parser diagnostics out of the headline table unless needed as diagnostics.
- Describe verbalized confidence calibration as behavioral calibration of elicited confidence, not internal model probability.

Open question:

- Whether the same high-confidence over-commitment pattern persists across all planned local models.
- Whether independent RE-informed reviewers agree that all weak templates sit below `SHOULD/recommended` on the study's ordinal scale.
- Whether verbalized confidence remains informative in full runs, given that consistency-style UQ is expected to struggle with systematic errors.
- Whether Task 3 self-verification detects over-commitment or instead repeats the same modality normalization error.
- Whether the `mlm_tapt` co-primary seed split reproduces the NICE pattern after manual review.

## Construct-Validity Gate

`docs/weak_modality_construct_review.csv` is the durable review template for the weakest modality construct.

Completion rule:

- Two reviewer slots are provided for each weak template.
- Paper-ready claims require every row to mark `weaker_than_should=yes`.
- `ordinal_rank` should use the study scale: `0=weak/nice-to-have`, `1=optional/MAY`, `2=recommended/SHOULD`, `3=mandatory/MUST`.
- This is a paper-readiness gate, not a notebook execution gate.

## Draft Manuscript Wording

Introduction:

> Requirements engineering depends on preserving not only what stakeholders want, but also how strongly they commit to it. In LLM-assisted requirements extraction, a model that rewrites weak stakeholder intent into firmer requirement language may produce outputs that look cleaner while changing prioritization, optionality, and stakeholder commitment.

Method:

> We construct controlled modality variants from reviewed requirement seeds, holding functional content constant while varying only the source modality. The benchmark covers mandatory, recommended, optional, and an operational weak stakeholder-intent class.

Results/discussion:

> In the pilot, the model correctly rejected unsupported mandatory interpretations, but did not preserve weak stakeholder intent during extraction. Across prompt and wording checks, `nice_to_have` was never preserved; the model instead assigned stronger labels such as `optional` or `recommended`, often with high confidence. This suggests that consistency is not a sufficient correctness proxy for modality preservation in RE tasks.

Caveat:

> Unlike `MUST`, `SHOULD`, and `MAY`, the weak stakeholder-intent class is not an RFC-standard keyword level. We therefore treat it as an operationalized family of weak desiderative and low-priority phrasings, supported by a separate construct-validity check.

## Literature Anchors

- Nuseibeh and Easterbrook, "Requirements Engineering: A Roadmap" - RE identifies stakeholders and needs, then documents them for analysis, communication, agreement, and implementation. https://researchr.org/publication/NuseibehE00
- ISO/IEC/IEEE 29148:2018 - requirements engineering standard covering stakeholder needs, requirements processes, and traceability. https://www.iso.org/standard/72089.html
- IEEE Std 830-1998 - SRS quality properties include unambiguous, ranked for importance/stability, verifiable, and traceable requirements. https://people.eecs.ku.edu/~hossein/Teaching/Stds/0830.pdf
- RFC 2119 and RFC 8174 - normative grounding for `MUST/SHALL`, `SHOULD/RECOMMENDED`, and `MAY/OPTIONAL`. https://www.rfc-editor.org/rfc/rfc2119 and https://www.rfc-editor.org/rfc/rfc8174
- DSDM MoSCoW prioritisation - supports a weaker `Could Have` category as wanted or desirable but less important than `Should Have`. https://www.agilebusiness.org/page/ProjectFramework_10_MoSCoWPrioritisation
- Hemmat et al. 2025, "Research directions for using LLM in software requirement engineering: a systematic review" - LLM4RE review highlighting RE use cases, risks, and evaluation gaps. https://doi.org/10.3389/fcomp.2025.1519437
- Shorinwa et al., "A Survey on Uncertainty Quantification of Large Language Models" - UQ taxonomy and motivation around plausible incorrect outputs with high confidence. https://doi.org/10.1145/3744238 and https://arxiv.org/abs/2412.05563
- Tian et al., "Just Ask for Calibration" - useful counterpoint that verbalized confidence can be informative for some LLMs. https://arxiv.org/abs/2305.14975
- Manakul et al., SelfCheckGPT - canonical black-box self-consistency style hallucination detection reference. https://doi.org/10.18653/v1/2023.emnlp-main.557
- Zhang et al., SAC3 - relevant to the risk that self-consistency can be insufficient when models are consistently wrong. https://arxiv.org/abs/2311.01740
- Femmer et al., "Rapid requirements checks with requirements smells" - requirements-smell framing for weak, vague, or hard-to-verify phrasing. https://doi.org/10.1016/j.jss.2016.02.047
