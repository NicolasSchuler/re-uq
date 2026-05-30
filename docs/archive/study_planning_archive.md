# Study Planning Archive

This archive compresses the former `docs/paper_framing.md` and
`docs/internal/implementation_plan.md` notes into one internal traceability
document. It is not required for reproduction. The reader-facing entry points
are `README.md`, `docs/evaluation.md`, `docs/reproduction.md`,
`docs/results_mapping.md`, and `docs/repository_layout.md`.

## Purpose

The study evaluates whether LLM-assisted requirements extraction preserves
stakeholder commitment when functional content is held constant and only source
modality varies. The controlled modalities are mandatory, recommended,
optional, and an operational weak stakeholder-intent class implemented as
`nice_to_have`.

The central observable is high-confidence over-commitment: a model may preserve
functional content while strengthening weak stakeholder intent into firmer
requirement language. This should be described as output behavior. Commitment
normalization is a discussion-level interpretation, not a demonstrated internal
mechanism.

## Claim Boundaries

- Primary claim: LLMs may preserve functional content while high-confidently
  strengthening weak stakeholder intent.
- Safer short claim: the RE risk is high-confidence over-commitment at the
  elicitation-specification boundary.
- Avoid claiming that all UQ fails. The supported narrower claim is that
  consistency-based UQ can miss systematic, stable modality errors.
- Treat Task 1 as a control, Task 2 as the main empirical task, and Task 3 as a
  blind source-grounded diagnostic over deterministic Task 2 outputs.
- Treat declared-modality Task 3 prompts as anchoring ablations, not official
  blind Task 3 evidence.
- Treat ACSE-style semantic entropy as a five-sample triage/ranking signal
  unless thresholds are calibrated on a declared held-out split.

## Research Questions

1. When functional content is held constant and only modality changes, do models
   preserve requirement strength across mandatory, recommended, optional, and
   weak stakeholder-intent phrasings?
2. Do lightweight black-box UQ signals reveal high-confidence over-commitment,
   or can models be stable and confident while wrong?
3. Is weak-intent collapse robust to prompt simplification and lexical variation
   in weak stakeholder-intent wording?
4. Diagnostic: can a model detect in a blind source-grounded audit that its own
   extracted requirement strengthened or weakened the source modality?

## Durable Design Decisions

- Use NICE/PROMISE-derived requirements as the default dataset and
  `limsc/mlm-tapt-requirements` as a co-primary reviewed seed source.
- Exclude `_PURE` sources from the `mlm_tapt` path because document structure is
  too heterogeneous for the short-communication scope.
- Avoid Dalpiaz user-story datasets for primary results unless licensing/IP
  uncertainty is resolved.
- Generate controlled minimal pairs for each reviewed seed and keep prompt,
  modality-label, and gold-label contracts stable across datasets.
- Keep `MUST` as the main mandatory wording and `SHALL` as a robustness variant.
- Preserve raw model outputs locally or in an external archive; keep Git focused
  on curated inputs, prompts, stripped notebooks, tests, manifests, and compact
  summaries.
- Use command-first reproduction through `scripts/*.py` and
  `scripts/reproduce.sh`; notebooks remain companion inspection artifacts.

## Evaluation Phases

1. Prepare reviewed requirement seeds:
   `notebooks/00_prepare_data.ipynb` and the corresponding seed review CSVs.
2. Build modality benchmarks:
   `notebooks/01_build_modality_benchmark.ipynb`,
   `data/processed/benchmark_items*.csv`, and
   `outputs/benchmark_manifest*.json`.
3. Run pilot and probe checks:
   Task 1/2 smoke runs, prompt sensitivity, weak-modality robustness, and
   optional token-logprob capability probing.
4. Run full provider/model cells:
   `scripts/run_experiment_from_config.py` with smoke-first execution, provider
   preflight, structured-output validation, raw JSONL caching, and run registry
   updates.
5. Run blind Task 3 diagnostics:
   `scripts/run_task3_verification_from_config.py` over complete Task 2 source
   runs, writing run-specific Task 3 items and raw verifier outputs.
6. Generate paper-facing analysis:
   `scripts/generate_evaluation_analysis.py`, producing `uq_scores.csv`,
   metric summaries, bootstrap intervals, qualitative examples, ACSE
   calibration diagnostics, provenance manifests, and the Task 1 modality
   figure.

## UQ And Metrics

The planned UQ surface is lightweight and black-box:

- verbalized confidence;
- label/modality/relation self-consistency;
- predictive entropy;
- variation ratio;
- ACSE-inspired semantic entropy over stochastic answer texts;
- model-ensemble disagreement when multiple complete deterministic model runs
  are available;
- optional token-logprob confidence only after endpoint capability checks.

High-confidence risks should use denominator-specific names:

- Task 1 `unsupported_mandatory_acceptance@tau`;
- Task 2 `HC-OC_all@tau`;
- Task 2 `HC-OC_overcommittable@tau`;
- Task 2 `weak_strengthening@tau`;
- Task 2 `label-correct text overcommitment@tau`.

## Construct-Validity Gate

`docs/weak_modality_construct_review.csv` is the durable two-reviewer gate for
the weakest modality construct.

Paper-ready weak-intent claims require:

- two reviewer slots for every weak template;
- every row marked `weaker_than_should=yes`;
- the ordinal scale `0=weak/nice-to-have`, `1=optional/MAY`,
  `2=recommended/SHOULD`, `3=mandatory/MUST`;
- no unresolved reviewer disagreement.

Full experimental runs may proceed before this gate is complete, but weak-class
paper claims remain diagnostic until it passes.

## Pilot Evidence Preserved For Context

The initial pilot suggested:

- Task 1 mandatory-entailment decisions were correct on the pilot subset.
- Task 2 preserved functional content but systematically strengthened
  `nice_to_have` outputs.
- Stochastic samples made the weak-intent error look stable rather than a single
  deterministic accident.
- Labels-only and weak-template probes suggested the effect was not only caused
  by prompt examples or one exact weak phrase.

These are formative observations. Current paper-facing claims should come from
the regenerated `outputs/evaluation_*` artifacts, their provenance manifests,
and `docs/results_mapping.md`.

## Open Questions

- Whether all final model/dataset/variant cells support the same
  high-confidence over-commitment pattern.
- Whether independent RE-informed reviewers complete and agree on the weak
  construct-validity gate.
- Whether verbalized confidence remains informative in final full runs.
- Whether blind Task 3 detects generated-text strengthening without declared
  modality anchoring.
- Which final raw outputs and run-level artifacts should be deposited in the
  external archive.

## Literature Anchors

- Requirements engineering roadmap and standards: Nuseibeh and Easterbrook;
  ISO/IEC/IEEE 29148; IEEE 830.
- Normative modality vocabulary: RFC 2119 and RFC 8174.
- Weaker desiderative/prioritization language: DSDM MoSCoW `Could Have`.
- LLM4RE and requirements-smell context: recent LLM4RE surveys and Femmer et
  al. on requirements smells.
- UQ context: broad LLM UQ surveys, verbalized calibration work, SelfCheckGPT,
  and SAC3-style self-consistency limitations.
