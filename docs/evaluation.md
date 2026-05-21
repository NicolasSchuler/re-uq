
You:
I had a student write a short seminar paper about uncertainty quantification in LLMs and why it is important for Software Engineering especially something like Requirements Engineering (see attachment).
We wanted to create a survey paper for ACM computing surveys journal. However, we saw that there was already a published paper about LLM uncertainty and methods for quantifying it.

Therefore, our next best solution (also in-terms of time required to write the paper) was to write a short communication paper for the IST journal (https://www.sciencedirect.com/journal/information-and-software-technology/publish/guide-for-authors)

For short communication, the maximum length is 2,500 words. Please note that References and appendixes are part of the submission and count against the total number of words and that figures and tables count 200 words each.

Now, even for a short journal paper like this we would need some form of short evaluation. I would assume nothing too big (just a couple of jupyter notebook files are probably fine). But we still need them. Can you help me define an experimentation plan for this?

In particular, our initial idea was to utilize existing requirement datasets, analyze them for Linguistic Modality, and then evaluate multiple LLMs and multiple uncertainty quantification methods on that and more importantly, compare the intended modality (e.g., according to the modality survey mentioned in the paper "should means 80%" or something like that) and then present our results.

That was my general idea. However it is just a rough draft and not finished, a lot of details are missing for actually having a concrete experimentation plan.

Please carefully and based on the paper and the general idea do the following:
1. Suggest further improvements of the overall methodology before diving deeper into the concrete details of the experimentation plan.
2. Carefully evaluate the feasibility, risks, and time requirements (in work hours) for conducting and implementing the experiments.
3. Investigate and fill in the details of the experimentation e.g., which dataset to use, which uncertainty quantification etc.
4. Re-iterate at least 10 times over your suggested plan based on the mentioned steps above and then cohesively present your analysis, and proposed experimentation plan.

ChatGPT:
## Recommended pivot 🎯

The strongest paper angle is not “a smaller UQ survey.” The strongest angle is:

> A short empirical communication on whether LLM uncertainty estimates are sensitive to linguistic modality in requirements engineering, especially dangerous modality upgrades such as turning “may” or “nice to have” into “must.”

That fits the IST short-communication format much better than a broad survey. IST’s current guide says short communications are limited to 2,500 words, with references and appendices included, and each figure/table counting as 200 words. ([ScienceDirect](https://www.sciencedirect.com/journal/information-and-software-technology/publish/guide-for-authors?utm_source=chatgpt.com)) The motivation is also defensible because a 2025 ACM Computing Surveys paper already covers UQ for LLMs broadly, so your novelty should come from the RE-specific modality/UQ interaction, not from another taxonomy. ([ACM Digital Library](https://dl.acm.org/doi/10.1145/3744238?utm_source=chatgpt.com))

The attached seminar paper already points exactly there: it asks whether model uncertainty aligns with intended linguistic modality, emphasizes RE risk from words such as “must,” “should,” and “can,” and structures black-box UQ around verbalized confidence, surrogate models, and sample consistency.   Full uploaded paper:

---

# 1. Methodology improvements before defining the experiment 🧪

## Improvement 1 - Do not rely only on naturally occurring modality

Your initial idea-take existing requirements datasets, detect modal verbs, compare LLM UQ to “intended modality”-is good as a starting point, but weak as a main evaluation because natural datasets usually do not contain gold labels for intended certainty, obligation, or stakeholder commitment.

A better design is:

> Use existing RE datasets as realistic seed material, then construct controlled minimal-pair modality variants.

Example seed capability:

> “export results as CSV”

Controlled source variants:

| Variant      | Source statement                                                |
| ------------ | --------------------------------------------------------------- |
| Mandatory    | “The system MUST export results as CSV.”                        |
| Recommended  | “The system SHOULD export results as CSV.”                      |
| Optional     | “The system MAY export results as CSV.”                         |
| Nice-to-have | “It would be useful if the system could export results as CSV.” |

This gives you a clean causal factor: only the modality changes.

---

## Improvement 2 - Separate deontic modality from epistemic probability

This is the most important conceptual correction.

In requirements engineering, “must,” “should,” and “may” are mostly deontic: they express obligation, recommendation, permission, or optionality. They are not simply probability words. So the paper should avoid saying “should means 80%” as if this were a universal numeric truth.

For RE, use commitment/obligation strength, not probability, as the main target. RFC 2119 gives a widely used interpretation: MUST/REQUIRED/SHALL indicate absolute requirements, SHOULD/RECOMMENDED indicate strong recommendations with possible exceptions, and MAY/OPTIONAL indicate optionality. ([IETF Datatracker](https://datatracker.ietf.org/doc/html/rfc2119?utm_source=chatgpt.com)) RFC 8174 also clarifies that these special meanings strictly apply when the keywords are uppercase. ([IETF Datatracker](https://datatracker.ietf.org/doc/rfc8174/?utm_source=chatgpt.com))

Probability-word surveys such as Fagen-Ulmschneider are still useful, but mainly for epistemic expressions such as “likely,” “possibly,” “certainly,” or “probably.” The Fagen-Ulmschneider dataset is an open survey of perceived probability words with 123 respondents. ([WAF](https://waf.cs.illinois.edu/visualizations/Perception-of-Probability-Words/?utm_source=chatgpt.com)) Sileo and Moens also build on this line of work for probing neural language models’ understanding of words of estimative probability. ([ACL Anthology](https://aclanthology.org/2023.starsem-1.41.pdf?utm_source=chatgpt.com))

Recommended paper framing:\
Use RFC-style modality for the main RE experiment, and optionally mention probability-word mappings only as a secondary sensitivity analysis.

---

## Improvement 3 - Evaluate a concrete RE failure mode

The most compelling failure mode is:

> The source says something weak or optional, but the LLM produces or validates a mandatory requirement with high confidence.

That is much more relevant to RE than generic hallucination detection.

Example:

Source:

> “It would be nice if the system supported JSON export.”

Candidate requirement:

> “The system MUST support JSON export.”

Question:

> Does the LLM correctly reject this as a faithful mandatory requirement, and is its uncertainty calibrated?

This gives you a crisp binary task and an RE-specific risk metric: over-commitment.

---

## Improvement 4 - Evaluate UQ as calibration, not just correlation

A simple correlation between modal strength and confidence is not enough. You should evaluate:

1. Correctness: Does the model preserve or judge modality correctly?
2. Calibration: When it says 80% confidence, is it correct about 80% of the time?
3. Over-commitment risk: Does it turn weak modality into stronger modality?
4. Selective reliability: If you reject low-confidence outputs, does error decrease?

This makes the paper more convincing even with a small experiment.

---

# 2. Ten-pass refinement summary 🔁

| Pass | Refinement                                                                                                                                                      |
| ---: | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|    1 | Shifted from “mini survey” to short empirical communication because a broad CSUR survey already exists.                                                         |
|    2 | Replaced passive corpus analysis with controlled minimal-pair modality manipulation.                                                                            |
|    3 | Separated deontic RE modality from epistemic probability words.                                                                                                 |
|    4 | Defined the main safety-relevant task as mandatory-requirement entailment.                                                                                      |
|    5 | Added a second task for modality-preserving requirement extraction.                                                                                             |
|    6 | Chose existing RE datasets only as seed sources, not as gold-label modality datasets.                                                                           |
|    7 | Selected black-box UQ methods that align with the seminar paper: verbalized confidence and sample consistency, with surrogate scoring as an optional extension. |
|    8 | Added metrics for over-commitment, not only accuracy.                                                                                                           |
|    9 | Reduced scope to a realistic short-communication experiment: 80-150 seeds, 3 models, 2-3 UQ methods.                                                            |
|   10 | Designed the result presentation around one table and one compact figure to survive IST’s 2,500-word limit.                                                     |

---

# 3. Proposed concrete experimentation plan ✅

## Candidate paper title

“Modality-Conditioned Uncertainty in LLM-Assisted Requirements Engineering: A Short Empirical Study”

Alternative title:

“When ‘May’ Becomes ‘Must’: Uncertainty Quantification for Linguistic Modality in LLM-Assisted Requirements Engineering”

---

## Research questions

RQ1 - Modality preservation:\
Do LLMs preserve deontic modality when extracting requirements from stakeholder-like statements?

RQ2 - Modality-conditioned uncertainty:\
Does an LLM’s confidence decrease when a candidate mandatory requirement is only weakly supported by the source modality?

RQ3 - UQ method comparison:\
Which lightweight black-box UQ method best detects modality mismatch and over-commitment?

---

## Datasets

### Co-primary dataset 1: NICE / PROMISE-derived requirements

Use the recent NICE dataset as the main seed source. It is based on the original PROMISE dataset and provides requirement text plus functional/non-functional and NFR subclass labels in CSV form. ([Zenodo](https://zenodo.org/records/14590935)) This is attractive because it is small, structured, and easy to use in a short project.

Use it only for realistic capability content, not for gold modality.

Example seed extraction:

Original:

> “The response time shall be no more than 2 seconds…”

Normalized seed:

> “respond within 2 seconds”

Then generate controlled variants.

### Co-primary dataset 2: `limsc/mlm-tapt-requirements`

Use the Hugging Face `limsc/mlm-tapt-requirements` corpus as a second reviewed seed source. It is much easier to ingest than raw PURE because it already exposes requirement-like rows with `source` and `reqs` columns. However, it still contains headings, fragments, list material, and very technical clauses, so it should be treated as a candidate pool rather than a ready benchmark. The pipeline excludes `_PURE` sources, filters aggressively, then samples a source-diverse 180-row review table before generating controlled modality variants. ([Hugging Face](https://huggingface.co/datasets/limsc/mlm-tapt-requirements))

### Avoid as benchmark source: raw PURE

PURE contains 79 public natural-language requirements documents and 34,268 sentences, with XML files for a subset. It remains useful background for RE NLP experiments, but its raw structure is too heterogeneous for this compact controlled benchmark without substantial document-specific cleaning. ([Zenodo](https://zenodo.org/records/1414117))

### Avoid as primary: Dalpiaz user-story datasets

The 22 user-story datasets are tempting because they are RE-relevant and contain 50+ requirements each, but the Zenodo description explicitly notes uncertainty about license agreements and intellectual-property rights. ([Zenodo](https://zenodo.org/records/13880060)) I would not make them the main dataset unless you only use them for private robustness checks or carefully handle licensing.

---

## Benchmark construction

### Step A - Select seed capabilities

Target size:

| Version     | Seed capabilities | Modal variants | Items |
| ----------- | ----------------: | -------------: | ----: |
| Minimal     |                80 |              4 |   320 |
| Recommended |               180 |              4 |   720 |
| Extended    |               240 |              4 |   960 |

For IST, 180 seeds × 4 variants = 720 items per seed dataset is enough while keeping manual review manageable.

Filter requirements to keep only relatively clean statements:

- one main capability;
- 5-35 words;
- no complex multi-sentence dependencies;
- no heavy domain-specific formulas;
- no negation in the first version;
- no already-ambiguous phrasing.

Manually inspect the final 180 seeds per dataset. That is worth the small time investment.

---

## Step B - Generate controlled modality variants

Use four modality classes:

| Class        | Example wording                            | Commitment score |
| ------------ | ------------------------------------------ | ---------------: |
| Mandatory    | “The system MUST …” / “The system SHALL …” |             1.00 |
| Recommended  | “The system SHOULD …”                      |     0.67 or 0.75 |
| Optional     | “The system MAY …”                         |             0.33 |
| Nice-to-have | “It would be useful if the system could …” |     0.00 or 0.10 |

Do not overclaim that these numeric scores are universal. Treat them as an ordinal reference scale and include a sensitivity check: report whether results change if SHOULD is scored as 0.60, 0.75, or 0.80.

---

# Main task 1: Mandatory-requirement entailment

For every source variant, pair it with a mandatory candidate requirement.

Example:

Source

> “The system SHOULD export results as CSV.”

Candidate

> “The system MUST export results as CSV.”

Prompt

```text
You are reviewing software requirements.

Source statement:
"The system SHOULD export results as CSV."

Candidate requirement:
"The system MUST export results as CSV."

Question:
Is the candidate a faithful mandatory requirement entailed by the source?

Return JSON only:
{
  "decision": "yes" | "no",
  "confidence": 0-100,
  "brief_reason": "<max 12 words>"
}
```

### Gold label

| Source modality |    Candidate “MUST …” faithful? |
| --------------- | ------------------------------: |
| MUST / SHALL    |                             Yes |
| SHOULD          | No, or “not strictly mandatory” |
| MAY             |                              No |
| Nice-to-have    |                              No |

This is the cleanest evaluation because it avoids arguing whether SHOULD is “80%.” The question is stricter and more RE-relevant:

> Is the statement strong enough to justify a mandatory requirement?

---

# Main task 2: Modality-preserving extraction

Ask the model to extract the requirement while preserving modality.

Prompt

```text
Extract exactly one requirement from the source statement.
Preserve the modality of the source.

Source:
"It would be useful if the system could export results as CSV."

Return JSON only:
{
  "requirement": "...",
  "modality": "mandatory" | "recommended" | "optional" | "nice_to_have",
  "confidence": 0-100
}
```

### Metrics

- modality classification accuracy;
- macro-F1 over modality classes;
- over-commitment rate: predicted modality stronger than source modality;
- under-commitment rate: predicted modality weaker than source modality;
- confidence calibration by modality class.

This task is more realistic than Task 1 but slightly noisier.

---

# UQ methods to compare

Use three methods for the main paper. They are enough for a short communication and map well to the taxonomy in the seminar paper.

## UQ1 - Verbalized confidence

Use the model’s JSON confidence field.

For Task 1:

- if decision = yes, use `p_yes = confidence / 100`;
- if decision = no, use `p_yes = 1 - confidence / 100`.

This gives a probability that the candidate mandatory requirement is faithful.

Verbalized confidence is easy to implement and directly connected to prior work on eliciting calibrated confidence from RLHF-tuned LLMs. Tian et al. found that verbalized confidences can be better calibrated than conditional token probabilities for RLHF models on several QA benchmarks. ([arXiv](https://arxiv.org/abs/2305.14975?utm_source=chatgpt.com))

## UQ2 - Label self-consistency

Sample the same prompt K = 5 or K = 10 times at temperature 0.7.

For Task 1:

```text
p_yes = number_of_yes_outputs / K
uncertainty = 1 - max(p_yes, 1 - p_yes)
```

For Task 2:

```text
p_modality = frequency of most common modality label / K
uncertainty = 1 - p_modality
```

This is simple, black-box, and faithful to the SelfCheckGPT intuition that stochastic samples diverge more when the model is uncertain. ([ACL Anthology](https://aclanthology.org/2023.emnlp-main.557/?utm_source=chatgpt.com))

## UQ3 - Semantic or modality-class consistency

For Task 2, generate K extracted requirements and compare whether they preserve the same modality class. This avoids needing a full NLI pipeline.

Optional richer variant:

- group sampled requirements by semantic equivalence using an NLI model;
- compute entropy over semantic/modality clusters.

Semantic entropy was proposed to address the fact that different sentences can mean the same thing, which is a central issue in uncertainty estimation for natural-language generation. ([arXiv](https://arxiv.org/abs/2302.09664?utm_source=chatgpt.com))

## Optional UQ4 - Surrogate model scoring

Use a local open-weight model as a surrogate confidence estimator, e.g., Qwen2.5-7B-Instruct. Qwen2.5 models are available in multiple sizes, including 7B, and the model card provides Transformers/vLLM usage. ([Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct?utm_source=chatgpt.com))

Surrogate scoring is relevant because prior work found that probabilities from a different, often weaker, model can outperform linguistic confidence for estimating confidence in black-box LLM outputs. ([arXiv](https://arxiv.org/abs/2311.08877?utm_source=chatgpt.com))

For the short paper, I would keep this as an extension unless you already have local GPU access.

---

# LLMs to evaluate

Use three models:

| Role | Model type                                  | Reason                                                              |
| ---- | ------------------------------------------- | ------------------------------------------------------------------- |
| M1   | Strong closed-source API model              | Represents current high-performing LLM use in RE tools              |
| M2   | Low-cost/fast closed-source API model       | Represents realistic engineering usage                              |
| M3   | Open-weight model, e.g. Qwen2.5-7B-Instruct | Enables reproducibility and optional log-prob/surrogate experiments |

Record the exact model IDs and dates. Do not use aliases such as “latest” in the paper. Gemini’s model docs explicitly distinguish stable, preview, latest, and experimental model versions, so exact model strings matter for reproducibility. ([Google AI for Developers](https://ai.google.dev/gemini-api/docs/models?utm_source=chatgpt.com)) Anthropic similarly exposes model lists through its API documentation. ([Claude API Docs](https://docs.anthropic.com/en/api/models-list?utm_source=chatgpt.com))

---

# Evaluation metrics

## Task 1: Mandatory entailment

Use:

| Metric                              | Purpose                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------ |
| Accuracy / F1                       | Basic correctness                                                                          |
| Brier score                         | Quality of probabilistic confidence                                                        |
| Expected Calibration Error, 10 bins | Calibration                                                                                |
| AUROC                               | Ability to separate faithful vs unfaithful mandatory candidates                            |
| Spearman ρ                          | Correlation between modality strength and `p_yes`                                          |
| Monotonicity violations             | Whether `p_yes(MUST) ≥ p_yes(SHOULD) ≥ p_yes(MAY) ≥ p_yes(nice-to-have)` for the same seed |

The monotonicity violation rate is especially nice for a short paper because it is intuitive:

> For the same requirement content, confidence in a mandatory interpretation should not increase when the source modality becomes weaker.

## Task 2: Modality-preserving extraction

Use:

| Metric                        | Purpose                                  |
| ----------------------------- | ---------------------------------------- |
| Modality accuracy             | Does the model preserve modality?        |
| Macro-F1                      | Handles class imbalance                  |
| Over-commitment rate          | Dangerous upgrades, e.g. MAY → MUST      |
| Mean over-commitment severity | How severe the upgrades are              |
| Calibration by modality       | Whether confidence is reliable per class |

Define severity with the ordinal scale:

```text
severity = max(0, strength(predicted_modality) - strength(source_modality))
```

---

# Statistical treatment

Keep it lightweight:

- Report 95% bootstrap confidence intervals over seed capabilities.
- Use paired bootstrap comparisons between UQ methods.
- Avoid heavy significance machinery unless needed.

Because each seed appears in all modality variants, use the seed as the resampling unit, not individual prompt instances.

---

# Recommended experiment size

## Minimal viable version

| Component                             |          Value |
| ------------------------------------- | -------------: |
| Seed capabilities                     |             80 |
| Modal variants                        |              4 |
| Items                                 |            320 |
| Models                                |              3 |
| Samples per item for self-consistency |              5 |
| Total calls for Task 1                |          4,800 |
| Task 2 direct extraction only         | 960 more calls |

This is already publishable as a short communication if the framing is careful.

## Recommended version

| Component          |                                    Value |
| ------------------ | ---------------------------------------: |
| Seed capabilities  |                                      180 |
| Modal variants     |                                        4 |
| Items              |                                      720 |
| Models             |                                        3 |
| Samples per item   |                                        5 |
| Total Task 1 calls |                                    7,200 |
| Task 2 calls       | 1,440-7,200 depending on whether sampled |

This is still manageable.

---

# Suggested notebook structure 📓

You can implement this with 4-5 notebooks:

| Notebook                            | Purpose                                                                        |
| ----------------------------------- | ------------------------------------------------------------------------------ |
| `00_prepare_data.ipynb`             | Load NICE/PROMISE or `mlm_tapt`; filter and review seed requirements          |
| `01_build_modality_benchmark.ipynb` | Create controlled modality variants and gold labels                            |
| `02_run_llms.ipynb`                 | Run prompts against selected models; cache raw JSON outputs                    |
| `03_compute_uq.ipynb`               | Compute verbalized confidence, self-consistency, semantic/modality consistency |
| `04_analyze_results.ipynb`          | Metrics, calibration plots, over-commitment table, bootstrap CIs               |

Also create:

```text
data/
  seeds.csv
  benchmark_items.csv
  model_outputs_raw.jsonl
  uq_scores.csv
prompts/
  mandatory_entailment.txt
  modality_extraction.txt
config.yml
```

For the implemented provider/model matrix workflow, prefer the CLI runner over manually editing notebook environment variables:

```bash
cp run_configs/full_matrix.example.json run_configs/current_run.json
.venv/bin/python scripts/run_experiment_from_config.py --config run_configs/current_run.json --profile local_llama_cpp --model qwen/qwen3.5-9b --dataset nice --mode smoke
```

The CLI preserves the same canonical raw-output files used by the notebooks and adds `run_registry*.csv` files so complete `RUN_ID`s can be audited and compared across providers. Set `batch_size` in a provider profile to evaluate multiple compatible benchmark items in one API request while keeping one raw JSONL row per item/sample. Set `structured_output=json_schema` for providers that accept OpenAI-style strict structured outputs; otherwise use `json_object` or `none`. During long runs, `run_progress_live*.csv` and `run_events*.jsonl` provide live progress and warning history; use `scripts/show_run_progress.py --dataset nice --run-id RUN_ID --watch 30` to inspect a run without mutating outputs. For Z.ai GLM Coding Plan keys, use `https://api.z.ai/api/coding/paas/v4`; the general endpoint is for standard API balance.

---

# 4. Feasibility, risks, and time requirements ⏱️

## Feasibility summary

This is feasible as a short communication if you keep the experiment narrow. The main danger is trying to evaluate too many models, too many UQ methods, and too many datasets.

The best scope is:

> 2 co-primary seed datasets, 180 seed capabilities each, 4 modalities, 3 LLMs, 2 tasks, 3 UQ methods.

That is enough to support a concise empirical claim without becoming a full benchmark paper.

---

## Estimated work hours

| Activity                                     | MVP hours | Extended hours |
| -------------------------------------------- | --------: | -------------: |
| Final protocol and prompt design             |       4-6 |            6-8 |
| Dataset loading and filtering                |      6-10 |          10-16 |
| Manual seed inspection                       |       4-6 |           6-10 |
| Benchmark generation                         |       4-6 |            6-8 |
| LLM API/model wrappers                       |      6-10 |          10-16 |
| Verbalized confidence implementation         |       2-3 |            3-4 |
| Self-consistency implementation              |       3-5 |            5-7 |
| Semantic/modality consistency implementation |       5-8 |           8-12 |
| Optional surrogate/logprob method            |         - |          10-20 |
| Running experiments                          |       4-8 |           8-16 |
| Metrics and bootstrap analysis               |      8-12 |          12-18 |
| Result tables/figures                        |       4-6 |            6-8 |
| Experiment write-up for IST                  |      8-12 |          12-18 |

Total experiment-only MVP: roughly 45-70 hours.\
With paper integration: roughly 60-90 hours.\
With surrogate/logprob extension: roughly 75-115 hours.

For a small team, the MVP is realistic in about 1-2 focused weeks.

---

## Main risks and mitigations

| Risk                                        |      Severity | Mitigation                                                                      |
| ------------------------------------------- | ------------: | ------------------------------------------------------------------------------- |
| “Should = 80%” is conceptually questionable |          High | Frame as deontic commitment, not probability; use RFC-style ordinal mapping     |
| Synthetic variants may look artificial      |        Medium | Use real RE datasets as seed content; manually inspect seeds                    |
| Task may be too easy                        |        Medium | Include both mandatory-entailment and extraction tasks                          |
| LLM confidence may be badly calibrated      | Not a problem | This is part of the empirical contribution                                      |
| Dataset provenance/licensing issues         |        Medium | Use NICE/PROMISE and reviewed `mlm_tapt` seeds; avoid raw PURE and user-story datasets as primary |
| API model changes                           |        Medium | Record exact model IDs, dates, temperatures, prompts                            |
| Too many calls                              |    Low-Medium | Use K=5 first; increase to K=10 only if needed                                  |
| Word limit too tight                        |          High | One table, one small figure, 8-12 references max, no appendix                   |

---

# Final recommended MVP 🧭

I would implement exactly this:

1. Dataset: NICE/PROMISE and reviewed `mlm_tapt` as co-primary seed sources.
2. Seeds: 180 manually checked requirement capabilities per dataset, 360 total.
3. Modalities: MUST, SHOULD, MAY, nice-to-have.
4. Task 1: mandatory-requirement entailment.
5. Task 2: modality-preserving extraction.
6. Models: 3 models: strong API, cheap API, open-weight Qwen2.5-7B-Instruct or equivalent.
7. UQ methods: verbalized confidence, label self-consistency, modality/semantic consistency.
8. Metrics: accuracy, ECE, Brier, Spearman correlation, monotonicity violations, over-commitment rate.
9. Paper claim: preliminary evidence on whether LLM uncertainty and confidence are sensitive to RE modality.
10. IST format: one concise table, one compact figure, code/data repository instead of appendix.

That gives you a focused, defensible short communication with a clear contribution:

> LLM-assisted requirements engineering needs uncertainty estimates that respect linguistic modality; otherwise, models may confidently transform optional stakeholder wishes into mandatory requirements.


You:
Thank you, given your suggested plan. Can you help me define the general outline of the paper and the abstract? Importantly, we need to consider the 2500 words limit.

ChatGPT:
Absolutely - for 2,500 words, the paper needs to be framed as a focused empirical short communication, not as a broad conceptual or survey paper. The attached seminar paper already gives you the core motivation: LLMs may mishandle linguistic modality in RE, where words like “must,” “should,” and “can/may” encode different degrees of obligation, and the paper’s motivating figure shows exactly the dangerous case where an optional JSON feature is upgraded into a mandatory requirement.  The same paper also identifies black-box UQ families - verbalized confidence, surrogate models, and sample consistency - which can be reused as the empirical lens.

IST’s current guidance is strict: 2,500 words, with references and appendices included, and figures/tables counting as 200 words each. ([ScienceDirect](https://www.sciencedirect.com/journal/information-and-software-technology/publish/guide-for-authors?utm_source=chatgpt.com)) The short-communication page also says short communications should have no more than 10 references, so the paper must be aggressively compressed. ([ScienceDirect](https://www.sciencedirect.com/journal/information-and-software-technology/about/short-communications/information-and-software-technology-now-publishing-short-com?utm_source=chatgpt.com))

---

# Recommended paper framing

## Working title

When “May” Becomes “Must”: Modality-Conditioned Uncertainty in LLM-Assisted Requirements Engineering

More conservative alternative:

Modality-Conditioned Uncertainty Quantification for LLM-Assisted Requirements Engineering

I prefer the first one. It is memorable, directly tied to the empirical failure mode, and suitable for a short communication.

---

# Core thesis

> LLM-assisted requirements engineering needs uncertainty estimates that are sensitive not only to factual correctness, but also to linguistic modality, because confidently upgrading weak stakeholder intentions into mandatory requirements is a distinct RE-specific failure mode.

This should be the spine of the whole paper.

---

# Proposed 2,500-word structure

Assume the abstract counts unless IST or the production system explicitly says otherwise. I would target ≈2,350-2,400 words to leave safety margin.

| Part                           | Target words | Purpose                                     |
| ------------------------------ | -----------: | ------------------------------------------- |
| Abstract                       |      150-170 | Problem, method, key result, implication    |
| 1. Introduction                |      280-320 | Motivation, gap, contributions              |
| 2. Background and Related Work |      220-260 | Modality + black-box UQ, highly compressed  |
| 3. Study Design                |      550-650 | Dataset, tasks, models, UQ methods, metrics |
| 4. Results                     |      400-500 | RQ-oriented empirical findings              |
| 5. Discussion and Threats      |      280-350 | Meaning, limitations, implications          |
| 6. Conclusion                  |        60-90 | One takeaway paragraph                      |
| References                     |      250-350 | Max. 8-10 references                        |
| One display item               |          200 | Prefer one compact results table            |
| Total                          | ≈2,350-2,500 | With small buffer                           |

Important: I would use one table only and no appendix. Put prompts, full results, notebooks, and dataset construction scripts in a replication package.

---

# Detailed outline

## Abstract - 150-170 words

The abstract should be empirical and result-driven. Since the experiments are not yet run, the final sentence should be filled in after results. See draft below.

---

## 1. Introduction - 280-320 words

Goal: Establish the RE-specific risk and state the contribution quickly.

Suggested structure:

1. Opening problem: LLMs are increasingly used for RE tasks such as requirement generation, summarization, and review.
2. RE-specific issue: Requirements are not only about content; they encode obligation through modality.
3. Motivating example: A stakeholder says JSON export “would be nice,” but the LLM outputs “The system must support JSON export.”
4. Gap: Existing UQ work studies confidence, hallucination, or calibration, but not whether UQ tracks modality-sensitive RE errors.
5. Contribution paragraph: Present the short communication as a controlled empirical study.

Suggested contribution wording:

> This paper makes three contributions:\
> C1: a controlled modality benchmark built from real requirement seeds;\
> C2: an empirical evaluation of LLMs on mandatory-entailment and modality-preserving extraction;\
> C3: a comparison of lightweight black-box UQ methods using calibration, over-commitment, and monotonicity metrics.

Keep the contributions in prose or very short bullets. Do not use a long related-work-style introduction.

---

## 2. Background and Related Work - 220-260 words

Goal: Give only the concepts needed to understand the experiment.

Include:

- Deontic modality in RE: must/shall = mandatory; should = recommended; may/can/nice-to-have = weaker commitment.
- Why not “should = 80%”: Say that numeric modality scores are used only as an ordinal analysis device, not as universal semantics.
- UQ methods: Briefly define verbalized confidence and sample/self-consistency. Mention surrogate models only if used; otherwise leave them out.
- Gap: Prior UQ surveys and studies do not directly test modality-conditioned uncertainty in RE.

Do not include the full taxonomy from the seminar paper. It is too expensive for 2,500 words. One sentence is enough:

> Following prior black-box UQ taxonomies, we focus on methods that require only generated text: verbalized confidence and consistency across repeated samples.

---

## 3. Study Design - 550-650 words

This should be the most detailed section because it makes the paper credible.

### 3.1 Dataset construction

Describe the controlled benchmark:

> We sample \(N\) requirement seeds from an existing RE dataset and rewrite each seed into four minimal-pair variants: MUST, SHOULD, MAY, and nice-to-have. The content is held constant while only the modality changes.

Example:

| Modality     | Source                                                          |
| ------------ | --------------------------------------------------------------- |
| Mandatory    | “The system MUST export reports as CSV.”                        |
| Recommended  | “The system SHOULD export reports as CSV.”                      |
| Optional     | “The system MAY export reports as CSV.”                         |
| Nice-to-have | “It would be useful if the system could export reports as CSV.” |

Mention manual filtering briefly:

> We exclude seeds with multiple capabilities, negation, or domain-specific ambiguity.

### 3.2 Tasks

Use two tasks.

Task 1: Mandatory-requirement entailment

Input: source statement + candidate mandatory requirement.\
Output: yes/no + confidence.

Gold label:

- MUST source → yes.
- SHOULD/MAY/nice-to-have source → no, because the source does not entail a mandatory requirement.

Task 2: Modality-preserving extraction

Input: source statement.\
Output: extracted requirement + modality label + confidence.

Gold label: the source modality class.

This gives both a strict safety test and a more realistic extraction test.

### 3.3 Models

Keep this generic until the experiments are fixed:

> We evaluate three instruction-tuned LLMs: one strong proprietary model, one lower-cost proprietary model, and one open-weight model. Exact model identifiers, dates, decoding parameters, and prompts are reported in the replication package.

### 3.4 UQ methods

Use only two or three:

1. Verbalized confidence: confidence score returned by the model.
2. Label self-consistency: repeated sampling; uncertainty from disagreement over yes/no or modality labels.
3. Modality consistency: repeated extraction; uncertainty from disagreement over modality class.

Do not include surrogate models unless you actually implement them. For this paper size, they are likely too much.

### 3.5 Metrics

Use a compact metric set:

- accuracy / macro-F1;
- over-commitment rate;
- Brier score;
- Expected Calibration Error;
- monotonicity violations.

Define over-commitment in one sentence:

> Over-commitment occurs when the model assigns a stronger modality than the source, e.g., MAY → MUST.

Define monotonicity in one sentence:

> For the same requirement seed, confidence in a mandatory interpretation should not increase as the source modality weakens from MUST to SHOULD to MAY to nice-to-have.

---

## 4. Results - 400-500 words

Structure this directly around the RQs.

Recommended RQs:

RQ1: Do LLMs preserve requirement modality?\
Report modality accuracy and over-commitment rate.

RQ2: Are LLM uncertainty estimates modality-sensitive?\
Report whether confidence decreases from MUST to SHOULD to MAY to nice-to-have.

RQ3: Which black-box UQ method best detects unsupported mandatory requirements?\
Compare verbalized confidence and self-consistency using Brier/ECE/AUROC.

Use one compact table.

Suggested table:

| Model | UQ method | Accuracy | Over-commitment ↓ | Brier ↓ | ECE ↓ | AUROC ↑ | Monotonicity violations ↓ |
| ----- | --------- | -------: | ----------------: | ------: | ----: | ------: | ------------------------: |

This one table can carry most of the empirical payload.

---

## 5. Discussion and Threats to Validity - 280-350 words

Combine discussion and threats to save space.

Suggested paragraph structure:

1. Interpretation: What does the main result mean for LLM-assisted RE?
2. Design implication: RE tools should flag modality upgrades and expose uncertainty.
3. Method implication: Accuracy alone is insufficient; over-commitment and monotonicity are better RE-specific diagnostics.
4. Threats: controlled variants are synthetic; modality classes are simplified; prompts and model versions may affect results; only English requirements are studied; confidence scores are not guaranteed to reflect internal uncertainty.

This section should be honest but not defensive.

---

## 6. Conclusion - 60-90 words

One short paragraph:

> This short communication introduced a modality-conditioned evaluation of uncertainty in LLM-assisted requirements engineering. By testing whether models confidently transform weak stakeholder intentions into mandatory requirements, we expose a failure mode that is not captured by generic UQ benchmarks. The results suggest that future RE assistants should evaluate uncertainty with respect to modality preservation, not only factual correctness.

---

# Abstract draft with result placeholders

Use this as the final abstract template after inserting actual values.

> Large language models are increasingly used to support requirements engineering tasks, yet requirements often depend on linguistic modality: “must” expresses obligation, whereas “should,” “may,” and “nice to have” encode weaker commitment. A model that confidently rewrites optional stakeholder wishes as mandatory requirements can introduce costly downstream errors. This short communication studies whether black-box uncertainty estimates are sensitive to modality in LLM-assisted requirements engineering. We construct a controlled benchmark from \[N] real requirement seeds by generating minimal-pair variants with four modality levels. We evaluate \[M] instruction-tuned LLMs on two tasks: mandatory-requirement entailment and modality-preserving extraction. We compare verbalized confidence, label self-consistency, and modality consistency using accuracy, over-commitment, calibration, and monotonicity metrics. Our results show that \[main empirical finding, e.g., models often remain highly confident when weak modalities are upgraded to mandatory requirements], while \[best UQ method] best identifies unsupported mandatory interpretations. These findings suggest that uncertainty quantification for LLM-assisted requirements engineering should explicitly account for modality preservation, not only factual correctness.

Approx. 165 words.

---

# Even shorter abstract version

This is safer if the final manuscript becomes too tight.

> Large language models are increasingly used in requirements engineering, where small changes in linguistic modality can change stakeholder intent. In particular, transforming “may” or “nice to have” into “must” can turn optional functionality into an apparent obligation. We study whether black-box uncertainty estimates are sensitive to this modality-sensitive failure mode. We construct a controlled benchmark from \[N] real requirement seeds by generating minimal-pair variants with four modality levels: mandatory, recommended, optional, and nice-to-have. We evaluate \[M] LLMs on mandatory-requirement entailment and modality-preserving extraction, comparing verbalized confidence, label self-consistency, and modality consistency. We assess accuracy, over-commitment, calibration, and monotonicity of confidence across modality levels. The results show \[main finding]. Our findings indicate that LLM-assisted requirements engineering requires uncertainty estimates that preserve modality, not only estimates of generic answer correctness.

Approx. 130 words.

---

# Recommended reference strategy

Use 8-10 references maximum. I would include only:

1. one RE importance / requirements reference;
2. one RFC-style modality/requirement-level reference;
3. one LLMs-for-RE reference;
4. one calibration reference;
5. one UQ survey;
6. one verbalized-confidence paper;
7. one self-consistency or SelfCheckGPT paper;
8. one semantic uncertainty or black-box UQ paper;
9. one dataset reference;
10. optionally one linguistic modality/probability-word reference.

Avoid citing three surveys. Avoid citing many model cards. Exact model IDs can go in the replication package instead of references.

---

# Practical manuscript rule

For this submission, I would write the paper as:

> one motivating example, three RQs, one benchmark construction paragraph, one compact results table, one discussion paragraph, and no appendix.

That gives you the best chance of staying under 2,500 words while still looking like a real empirical contribution rather than a position paper.
