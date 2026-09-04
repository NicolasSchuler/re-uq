# Experimental Setup

This page states the complete experimental setup. The setup is part of the
result: the numbers in the paper hold for these items, these prompts, this
batching policy, and this model cohort. Read it before citing any figure.

Related pages: [`docs/evaluation.md`](evaluation.md) (metric definitions),
[`docs/aggregation.md`](aggregation.md) (how per-cell numbers are pooled into
headline numbers), [`docs/reproduction.md`](reproduction.md) (commands),
[`TODO.md`](../TODO.md) (known gaps and planned work).

## 1. Overview

| Dimension | Value |
| --- | --- |
| Unit of analysis | One controlled single-sentence source statement. |
| Construct | Requirement modal force (`mandatory > recommended > optional > nice_to_have`). |
| Design | Minimal pairs: capability held constant, only source modal force varies. |
| Datasets | NICE/PROMISE-derived requirements; `limsc/mlm-tapt-requirements`. |
| Seeds | 180 reviewed seeds per dataset (360 total). |
| Conditions | 4 source conditions per seed → 720 items per dataset, 1440 total. |
| Variants | `MUST` (main), `SHALL` (robustness). |
| Tasks | Task 1 entailment control, Task 2 extraction (main), Task 3 blind audit (diagnostic). |
| Sampling | 1 deterministic sample at temperature 0.0; 5 stochastic samples at temperature 0.7. |
| Delivery | Batched prompts, 16 benchmark items per request (see §4). |

## 2. Benchmark Construction

### 2.1 Seed extraction

1. Start from raw requirement sentences of each source dataset.
2. Extract a bare *capability clause* with `auto_capability_text` in
   `scripts/eval_utils.py`: strip list markers, leading requirement boilerplate,
   the subject (`the system`, `the software`, …), the modal (`shall`, `must`,
   `should`, `may`, `will`, `can`, `could`), and `be able to`.
3. Apply the automatic filter (`automatic_filter`, plus `mlm_tapt_filter` for
   the second dataset). A candidate is dropped if any of these fire:

| Filter reason | Rule |
| --- | --- |
| `too_short` / `too_long` | Source requirement shorter than 5 or longer than 35 words. |
| `multi_sentence` | Sentence-final punctuation inside the requirement or capability. |
| `negation` | Negation cue in the source requirement. |
| `formula_or_symbol` | Formula/symbol pattern in the requirement. |
| `possibly_multiple_capabilities` | More than one of ` and `, ` or `, `;`. |
| `empty_or_too_short_capability` | Extracted capability shorter than 2 words. |
| `residual_modal_in_capability` | A modal survived extraction (would double the modal in the template). |
| `stranded_preposition` | Capability ends in a stranded preposition. |
| `no_requirement_cue`, `table_or_figure_reference`, `colon_structure`, `list_or_heading_marker`, `note_text`, `symbol_heavy`, `excluded_source` | Additional `mlm_tapt` corpus-hygiene filters. |

4. Review the surviving candidates manually. Grammatical coherence of the
   generated statements was checked by manual review and additionally with AI
   grammar tools. Review tables are tracked in
   `outputs/included_capabilities_review*.csv` and
   `outputs/benchmark_statements_review*.csv`.
5. Keep 180 reviewed seeds per dataset.

Filtering is deliberately conservative. It removes exactly the cases where the
template would produce an ungrammatical or double-modal sentence. This buys
construct control at the cost of naturalness (see §11).

### 2.2 Template inventory

Every source statement is one of the templates below, with the reviewed
capability clause substituted for `{capability}`. The four main conditions and
the `SHALL` swap are produced by `source_statement()` in `scripts/eval_utils.py`;
the weak probe templates are `WEAK_MODALITY_PROBE_TEMPLATES`.

| template_id | Condition | Variant | Template |
| --- | --- | --- | --- |
| `main_mandatory_must` | mandatory | must | `The system MUST {capability}.` |
| `main_recommended_should` | recommended | must | `The system SHOULD {capability}.` |
| `main_optional_may` | optional | must | `The system MAY {capability}.` |
| `main_nice_to_have_useful_if` | nice_to_have | must | `It would be useful if the system could {capability}.` |
| `shall_mandatory_shall` | mandatory | shall | `The system SHALL {capability}.` |
| `probe_useful_if` | nice_to_have | weak probe | `It would be useful if the system could {capability}.` |
| `probe_nice_if` | nice_to_have | weak probe | `It would be nice if the system could {capability}.` |
| `probe_low_priority_enhancement` | nice_to_have | weak probe | `As a low-priority enhancement, the system could {capability}.` |
| `probe_future_enhancement` | nice_to_have | weak probe | `Stakeholders mentioned that the system could {capability} as a possible future enhancement.` |

The `SHALL` variant swaps `MUST` for `SHALL` in the mandatory condition only.
The other three conditions are identical across the `must` and `shall` variants.
`probe_useful_if` is identical to the main weak condition and acts as the anchor
of the phrasing probe.

This table is generated, not hand-maintained. Regenerate it with:

```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'scripts'); import eval_utils as eu; \
print(eu.write_main_modality_template_inventory('outputs/modality_template_inventory.csv'))"
```

which writes `outputs/modality_template_inventory.csv` and
`outputs/modality_template_inventory.md` (`template_id`, `condition`, `variant`,
template string, example realisation with the sample capability `export
reports`, intended gold modality, note).

### 2.3 Weak-intent phrasing probe

The four `probe_*` templates were run as a separate probe to check that the
weak-intent finding is not an artifact of the single `It would be useful if …`
wording. Scope: `qwen/qwen3.5-9b` only, 20 pilot seeds, run through notebook
`02b`. Results are in `outputs/weak_modality_probe_summary.csv`: all four
templates show 100% strengthening (over-commitment rate 1.0, deterministic
samples). The probe is a diagnostic, not a headline result, and it was not
repeated on the official cohort.

### 2.4 Construct-validity review

`docs/weak_modality_construct_review.csv` records two reviewer slots (`R1`,
`R2`) judging whether each weak template is weaker than `SHOULD/recommended`.
Both slots currently contain an **author-delegated LLM-assisted review**, which
is declared in the `reviewer_role` column. The gate passes, but the judgments
are **pending human confirmation** before submission. See
[`TODO.md`](../TODO.md).

## 3. Prompts

All prompts are frozen plain-text files under `prompts/`, content-addressed by
SHA-256 in `outputs/benchmark_manifest*.json`, and verified by the analysis
gate. `prompts/README.md` is the index, and README.md reproduces the
single-item prompt bodies.

There are two layers:

| Layer | Where | Used by |
| --- | --- | --- |
| Single-item prompt | `prompts/*.txt`, rendered by `prompt_for_benchmark_task` | The frozen task contract; the wording the batched wrapper restates. |
| Batched wrapper | `batch_prompt_for_completion_jobs` in `scripts/eval_utils.py` | **Every real run in this repository.** |

The single-item files define the task contract, the label set, and the
confidence contract. The batched wrapper is what was actually sent. The two
agree on task, labels, and confidence scale; they differ in surface form,
because the wrapper carries several items at once and asks for an array of
results keyed by `request_index`.

### 3.1 Batched prompt, Task 2 (verbatim)

Reproduced from `batch_prompt_for_completion_jobs`, with a two-item batch:

```text
Extract exactly one requirement from each source statement.
Preserve the modality of each source. Evaluate each item independently.

Use one of: "mandatory", "recommended", "optional", "nice_to_have".
Use confidence as a decimal from 0.0 to 1.0 for confidence in the selected modality.
Do not return percentages such as 95 or strings such as "95%".
Return JSON only as this object:
{"results":[{"request_index":0,"requirement":"...","modality":"mandatory","confidence":0.95}]}

Items:
[
  {
    "request_index": 0,
    "source_statement": "The system MUST export reports."
  },
  {
    "request_index": 1,
    "source_statement": "It would be useful if the system could export reports."
  }
]
```

### 3.2 Batched prompt, Task 1 (verbatim)

```text
You are reviewing software requirements.
Evaluate each item independently. Do not infer an answer for one item from another item.

Use "yes" or "no" for decision.
Use confidence as a decimal from 0.0 to 1.0 for confidence in the selected decision.
Do not return percentages such as 95 or strings such as "95%".
Return JSON only as this object:
{"results":[{"request_index":0,"decision":"yes","confidence":0.95,"brief_reason":"<max 12 words>"}]}

Items:
[
  {
    "request_index": 0,
    "source_statement": "The system MAY export reports.",
    "candidate_requirement": "The system must export reports."
  }
]
```

### 3.3 Batched prompt, Task 3 blind (verbatim)

```text
Audit whether each extracted software requirement preserves the source statement.
Evaluate each item independently and do not repair the extracted requirement.

Use one of: "preserves", "strengthens", "weakens", "content_changed".
Use confidence as a decimal from 0.0 to 1.0 for confidence in the selected relation.
Do not return percentages such as 95 or strings such as "95%".
Return JSON only as this object:
{"results":[{"request_index":0,"relation":"preserves","confidence":0.95,"evidence_phrase":"...","brief_reason":"<max 12 words>"}]}

Items:
[
  {
    "request_index": 0,
    "source_statement": "The system MAY export reports.",
    "extracted_requirement": "The system must export reports."
  }
]
```

In the declared-modality Task 3 ablations, each item additionally carries
`declared_extracted_modality` or `declared_source_modality`.

No system message is sent. The batch prompt is the entire user message.

## 4. Batching Policy And Its Confound

| Property | Value in the reported runs |
| --- | --- |
| Items per request | 16 (Task 1, Task 2, and Task 3) in every official-cohort run. A handful of early `glm-5.1` registry rows used 8; they are not part of the reported cells. |
| Batch membership | Consecutive `request_index` values, no shuffling. |
| Benchmark row order | seed × variant: the four conditions of one seed are adjacent. |
| Consequence | Every Task 2 batch of 16 contains all four modality variants (`MUST`, `SHOULD`, `MAY`, `It would be useful if …`) of the same four seeds, side by side. |
| Max tokens | 256 per item, multiplied by the batch size for the request. |
| Batch id | `run_id:model:task:sample_kind:sample_index:<min>-<max> request index`. |
| Fallback | **None on the path the reported runs used** (see below). Single-item re-sends now exist on every path. |

**The batch fallback did not exist for the reported runs.** Only the Instructor
path re-sent the items of an unparsable batch as single-item requests. The raw
`response_format: {"type": "json_object"}` path — the one every official run
took — wrote a `missing_batch_result` row per affected item and never re-sent
it. Across the four cells that is 1, 1, 0 and 16 Task 1 deterministic rows and
48, 65, 60 and 51 Task 1 stochastic rows, plus 95 Task 2 stochastic rows in
`nice/shall`. **Task 2 deterministic rows had none**, so no strengthening,
label-accuracy, or confidence headline is affected; the Task 1 control and the
stochastic stability metrics lose the listed items, which are excluded rather
than counted (§7). The raw path now falls back to single-item requests as well,
so from this change onward the fallback applies to every path.

**This is a known confound.** The prompt instructs the model to evaluate each
item independently, but the model can see the minimal-pair contrast inside its
own context window. Contrastive context plausibly makes modality preservation
*easier* (the four strengths are visible next to each other), so the reported
strengthening rates are, if anything, conservative — but the direction is an
assumption, not a measurement. Nothing in this repository isolates the
per-item behaviour of a model that sees one statement at a time.

`batch_order` is now a knob, settable per profile or run-wide, with values
`grouped` (what every reported run used) and `shuffled`. Combined with
`batch_size: 1`, that lets grouped, shuffled, and single-item delivery be
compared on the same items (see [`TODO.md`](../TODO.md), section A). The
resolved value is recorded in the run registry column `batch_order`. Until
those ablations are run, every number in the paper must be read as "under
grouped 16-item batching".

`shuffled` is a **constrained** shuffle: it never places two source variants of
one seed in the same batch, and it is derived deterministically from the
recorded run seed, so the arm is reproducible. The first implementation only
permuted job order, which at batch size 16 still left two variants of one seed
together in roughly half of the 45 Task 2 batches (22 to 24 depending on the derived RNG seed) — it would have weakened the confound, not
removed it. Resume no longer re-shuffles the pending subset, so a resumed
shuffled run keeps the batch membership of the original run. The
`batching_ablation` preset pins `batch_size: 16` for both batch-order arms, so
grouped and shuffled differ only in membership, not in request size.

## 5. Model Cohort And Request Parameters

### 5.1 Official cohort

| Model | Endpoint | Notes |
| --- | --- | --- |
| `glm-4.5-air` | z.ai (`https://api.z.ai/api/...`) | |
| `glm-4.7` | z.ai | |
| `glm-5` | z.ai | |
| `glm-5-turbo` | z.ai | |
| `glm-5.1` | z.ai | Main model for smoke/first-pass documentation examples. |
| `kit.gemma4-31b-it` | KIT institutional endpoint (`kit_toolbox` profile, `https://ki-toolbox.scc.kit.edu/api/v1`) | The only non-GLM model in the cohort, and the highest strict-strengthening rate at 16.9%. |

`azure.*` rows may exist in local registries. They are private-endpoint
diagnostics and are excluded from every paper-facing aggregate
(`--exclude-model-prefix azure.` in `scripts/compare_run_matrix.py`).

Five of the six official models are from one family (GLM). See §11.

### 5.2 Request parameters

| Parameter | Deterministic pass | Stochastic pass |
| --- | --- | --- |
| `temperature` | 0.0 | 0.7 |
| `top_p` | 1.0 | 1.0 |
| Samples per item | 1 | 5 |
| `max_tokens` | 256 per item, scaled by batch size | same |
| JSON mode | `response_format: {"type": "json_object"}` | same |
| System prompt | none | none |
| `seed` | **not sent** in the reported runs | **not sent** |
| Retries | the OpenAI SDK client's built-in default of 2 silent internal retries — up to 3 HTTP attempts per batch, none of them recorded | same |

The reported runs had **no** application-level retry layer. The OpenAI SDK
client was constructed with its default `max_retries=2`, which silently retries
408/409/429/5xx responses and connection errors, so a batch could be sent up to
three times without any record of it. Going forward the SDK's internal retries
are disabled (`max_retries=0`) and `call_with_retries` is the only retry layer:
3 attempts, retrying 408/429/5xx, timeouts and connection errors, and failing
fast on 400/401/403/404/422 and every other 4xx. `retry_count` and
`retry_total` are now recorded for batched rows as well (see §6).

Going forward these are explicit profile knobs: `seed`, `send_seed`,
`max_retries`, and `batch_order`; `seed` and `batch_order` can also be set
run-wide. `send_seed` exists because some OpenAI-compatible layers accept and
silently ignore `seed` — the `google_gemini` example profile sets
`send_seed: false` for exactly that reason, so the raw records do not claim a
seed that never took effect. The `ollama_local` profile runs with
`json_mode: false` and `structured_output: none`. Provider selection is
restricted to OpenAI-compatible chat-completions endpoints: every profile is
a `base_url` + API-key pair against that one client, and providers without
such an endpoint are out of scope.

The z.ai profile additionally sends
`extra_body: {"thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}}`.
The `local_llama_cpp` example profile runs without JSON mode
(`structured_output: none`); the institutional profile uses `json_schema`.

Temperature 0.0 is treated as deterministic. It is not guaranteed to be
deterministic on a hosted endpoint, and without a request seed and a recorded
served-model version this cannot be checked after the fact for the existing
runs.

### 5.3 Configuration provenance

The configs committed with the first release **could not have produced the
cohort**. This is now fixed, and the discrepancies are recorded here so nobody
reads the old files as the provenance record:

| What the committed configs said | What the runs actually used |
| --- | --- |
| `kit.gemma4-31b-it` under `institutional_llm` | registry profile `kit_toolbox`, base URL `https://ki-toolbox.scc.kit.edu/api/v1`, `json_object`, batch 16 |
| `zai` listing 3 GLM models | all five GLM models of the cohort |
| `batch_size: 8` on every profile | 16 in every official-cohort run (§4) |
| `paper_cohort` preset sweeping 3 models, `must` only | all five GLM models and both benchmark variants |

`run_configs/full_matrix.example.json` and `conf/profile/` now carry a
`kit_toolbox` profile, the full `zai` model list, and `batch_size: 16`; the
`paper_cohort` preset sweeps all five GLM models and both variants.

**Labels differ from the archived runs; prompts do not.** The official raw rows
carry `prompt_version "v1"` (under the output contract `prompt_v2_confidence_0_1`)
and `run_group_id provider-matrix-2026-05`, whereas the current configs label
runs `v2-conf01` and `provider-matrix-v2-2026-05`. The prompt text itself is
unchanged: the batch prompt hashes recorded in the raw rows match the current
`batch_prompt_for_completion_jobs` output for 45/45 `glm-5.1` Task 2
deterministic batches in every one of the four cells. A rerun from the current
configs therefore differs from the archive in run labels, not in what is sent.

Local registries also contain `complete` rows dated 2026-05-21 whose raw rows
were later removed. They are stale bookkeeping, not results; the exporter now
refuses a chosen run that has no raw rows, and the rows should be pruned
([`TODO.md`](../TODO.md), section E).

## 6. What Is Recorded Per Response

Raw records are written to `data/processed/model_outputs_raw*.jsonl`
(local-only; see [`docs/repository_hygiene.md`](repository_hygiene.md)).

| Field group | In the reported runs | Added after this change |
| --- | --- | --- |
| Identity | `run_id`, `run_group_id`, `model`, `profile_id`, `provider_id`, `host`, `task`, `item_id`, `sample_kind`, `sample_index`, `request_index` | — |
| Request | `prompt_version`, prompt hash, `temperature`, `top_p`, `max_tokens`, `json_mode`, `structured_output` | `request_seed`, `request_payload_sha`, `system_prompt` (always empty: only a user message is sent), `batch_variant_mix`, `max_retries`, `retry_count`, `item_context` (`bare` unless the document-context ablation) |
| Response | `raw_text`, `parsed_json`, `parse_status`, latency | `finish_reason`, `usage_*` (prompt/completion/total tokens), `served_model`, `system_fingerprint`, `response_chars`, `requirement_word_count` |
| Parse status values | `ok`, `invalid_json`, `invalid_confidence`, `invalid_label`, `missing_fields` | `truncated` |

A `truncated` response counts as a **parse failure**, not as a separate
category, so `parse_success_rate` now reflects token-budget losses instead of
hiding them. Run-level quality is summarised in the registry columns
`batch_order`, `parse_status_histogram`, `parse_repairs`, `retry_total`,
`truncated_records`, `latency_p50_s`, `latency_p95_s`, and
`usage_completion_tokens`. `observed_records` counts logical observations
and `observed_attempts` the physical raw rows behind them; the two differ
only when a resume re-requested a failed cell. Per-run logs
are written to `data/processed/logs/<run_id>.log`.

**Batched-path caveat on the added fields.** The new provenance fields were at
first written correctly only on the single-item path. On the batched path —
which is every official run and every rerun at the default batch size — the
writer dropped the driver's `retry_count`, `request_seed` and
`request_payload_sha`, so `retry_count` read 0 on every batched row whatever
had actually happened, and the `request_payload_sha` that was recorded came
from the *single-item* prompt rather than from the batch payload that was sent.
Both are fixed: batched rows now carry the driver's retry count and request
seed, and `request_payload_sha` hashes the batch payload. As with everything
else in this column, the fix applies to runs made from this change onward;
already-written raw rows keep the old values.

The gap that matters for reproducibility: the reported runs recorded the
**requested** model string, not the **served** model version, and sent no
request seed. Both are fixed going forward, but the existing raw outputs cannot
be re-derived. Any rerun should therefore be treated as a new run, not as a
verification of the old one (see [`TODO.md`](../TODO.md), section F).

## 7. Sampling Design

Per benchmark item and task:

- one deterministic sample (temperature 0.0) — the row used for label accuracy,
  text-strengthening detection, calibration, and as the Task 3 source text;
- five stochastic samples (temperature 0.7) — the distribution used for
  `modality_consistency`, `predictive_entropy`, `variation_ratio`, and the
  ACSE-inspired semantic-dispersion score.

Repeated-sample agreement and unanimity are only computed over items whose
stochastic group is complete (all five samples parsed). Incomplete groups are
excluded rather than counted as agreeing; this is what keeps the reported
100% agreement figure from being an artifact of dropped samples.

`model_ensemble_disagreement` needs several deterministic runs over the same
items and is therefore available only where the run matrix provides them.

## 8. Text-Strengthening Detector

Label accuracy alone misses the failure mode, so the generated requirement text
is classified independently by `requirement_text_modality_diagnostic` in
`scripts/eval_utils.py`.

| Basis | Rule | Counted as strengthening evidence |
| --- | --- | --- |
| `weak_phrase` | `would be nice/useful if`, `low-priority enhancement`, `future enhancement`, `nice-to-have`, `wishlist`. | strict and broad |
| `explicit_modal` | Positive modal cue: `must`/`shall`/`required to` → mandatory, `should`/`recommended` → recommended, `may`/`optional`/`could`/`can` → optional. | strict and broad |
| `negated_modal` | A modal cue negated by a contraction, by `not`/`never` within 3 preceding tokens, or by a following `not`/`n't`, **and no positive modal cue anywhere in the text**. Resolves to `negated`, never to a positive strength. | neither |
| `heuristic_system_verb` | No modal at all, but the text matches `^(the )?system <verb>`; defaulted to mandatory. | broad only |
| `unknown` | Nothing matched. | neither |

- **Strict strengthening** requires explicit modal or weak-phrase evidence.
  This is the conservative measure.
- **Broad strengthening** additionally accepts the `heuristic_system_verb`
  default. It rests on the RE convention that a bare `The system X.` reads as an
  obligation; that convention is an assumption, not an observation.
- The gap is not small: **11.5% of successful Task 2 outputs contain no modal at
  all** (1,992 of 17,280 over all four cells), so the strict/broad spread is
  driven by a large, genuinely ambiguous slice. Report both. The share is very
  uneven across variants — 17.6% (1,520/8,640) in the two `MUST` cells against
  5.5% (472/8,640) in the two `SHALL` cells — so always name the scope.
- When several distinct modal categories co-occur the record is flagged
  (`text_modality_multi_modal`) and the strongest positive category wins
  (`mandatory > recommended > optional`). Negation loses this contest: a
  negated cue resolves to `negated` only when no positive cue is present, so
  "The system must ensure that users cannot delete records." is read as
  mandatory and flagged multi-modal rather than dropped as negated. No cohort
  row contained a negated cue (negated rate 0.0 in every cell), so no published
  number changes.

Answer length is recorded alongside (`requirement_word_count`,
`source_word_count`, `response_chars`). Weak-intent outputs average **18.65
words** against **15.57 words** for the other three conditions over all four
cells (18.31 vs 15.48 in the two `MUST` cells alone): the model does
not just re-label a weak wish, it writes more when hedging it. Treat this as an
answer-bloat signal worth reporting next to the strengthening rate.

## 9. Metrics And Aggregation

Metric definitions live in [`docs/evaluation.md`](evaluation.md). How per-cell
numbers become the headline numbers — the four dataset × variant cells, pooled
vs macro-of-cells, denominators, and confidence intervals — is specified in
[`docs/aggregation.md`](aggregation.md). Do not restate an aggregate without
naming its scope.

## 10. Embedding Model

| Property | Value |
| --- | --- |
| Model | `mlx-community/Qwen3-Embedding-0.6B-8bit` |
| Backend | MLX (requires `mlx-embeddings`) |
| Override | Hydra group `embedding=` for a run; `--backend` / `--mlx-model` for an explicit post-analysis cache ablation |
| Ablation options | `embedding=qwen3_4b`, `embedding=multilingual_e5_large`, `embedding=bge_m3`, `embedding=embeddinggemma_300m`, `embedding=tfidf_proxy` |
| Provenance | Resolved label on raw run rows, the analysis manifest, and the ACSE artifact manifest. |
| Legacy default | Dependency-free TF-IDF character n-gram proxy when a JSON run has no embedding selection. |

Rationale: the analysis had to run locally on Apple Silicon without shipping
text to a third party, so the choice was restricted to MLX-executable
embedders. Within that set, Qwen3-Embedding-0.6B is a commonly used,
general-purpose, multilingual embedding model — a current-generation family
with strong MTEB retrieval and STS scores at 0.6B parameters — which keeps
the semantic-dispersion signal meaningful without a GPU budget; the 8-bit
quantization was chosen for memory headroom. The backend is a first-class
configuration choice (`embedding=` in Hydra) that is persisted with the raw
run and consumed by later analysis, so a reviewer-requested ablation is a run
flag, not a code change:
same-family scale-up (`qwen3_4b`), independent families (`multilingual_e5_large`,
`bge_m3`, `embeddinggemma_300m`), and the non-neural TF-IDF proxy are all
ready in `conf/embedding/`.

Caveats: the quantization was **not ablated** against the 4-bit or full-precision
variants, and no other embedding family was benchmarked on this data. The
TF-IDF proxy remains available and is a useful contrast, because a character
n-gram model is close to an oracle for the surface modal keyword that defines
strict strengthening — see `scripts/diagnose_embedding_separability.py`.

## 11. Limitations

1. **Single-sentence controlled items.** Each item is one synthetic sentence
   built from a fixed template. Real requirements arrive in documents.
2. **No surrounding context.** Section headings, document status (draft vs
   approved), stakeholder role, priority fields, rationale, neighbouring
   requirements, and elicitation transcripts are not modelled. A model that
   strengthens a bare sentence might behave differently with a "Wishlist"
   heading above it. This is the largest external-validity gap. A minimal
   two-arm ablation now exists (`item_context: bare|document` on the `pure`
   cell built from two PURE documents with author-assigned M/O markers; see
   [`context_ablation.md`](context_ablation.md)); its numbers are reported
   separately and never pooled into the headline cells. The fuller extension
   remains [`TODO.md`](../TODO.md), section B.
3. **Grouped batching.** All reported numbers are under 16-item grouped batches
   that contain the minimal-pair contrast (§4). The ablation that would bound
   this effect has not been run.
4. **Model-family concentration.** Five of the six official models are GLM
   variants from one provider. `kit.gemma4-31b-it` is the only outside model.
   Cross-family generalisation is not established.
5. **Provenance gaps in the existing runs.** No request seed was sent and no
   served model version was recorded, so an exact rerun cannot be verified.
6. **Construct review pending human sign-off.** Both reviewer slots in
   `docs/weak_modality_construct_review.csv` are LLM-assisted.
7. **Broad strengthening rests on a convention.** The `heuristic_system_verb`
   default is a modelling choice, and it covers an 11.5% slice over all four
   cells (17.6% in the `MUST` cells, 5.5% in the `SHALL` cells).
8. **Fine-tuning is not in scope.** We do not fine-tune. This study measures
   off-the-shelf behaviour of hosted instruction-tuned models under a frozen
   prompt contract. Whether fine-tuning removes the failure mode is an open
   question and is deliberately *not* a to-do item here.
9. **Task 3 is not verification.** It is the same model auditing its own output;
   it is a stress test of an audit prompt, not ground truth.
10. **ACSE-inspired scores are a proxy.** Five samples and no held-out
    calibration protocol; use them for ranking and triage, not as a guarantee.
