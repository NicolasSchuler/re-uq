# Experimental Setup

This page states the complete experimental setup. The setup is part of the
result: the numbers in the paper hold for these items, these prompts, this
request protocol, and this model cohort. Read it before citing any figure.

**Which campaign this page describes.** The reported results are the
`manuscript-final` campaign of 11 to 16 September 2026: nine models, one item
per request, one deterministic and five sampled answers per item, request
seeds sent and the served model recorded, task-specific JSON-schema decoding
for the local models, and the authors' review of the PURE capabilities.
Reference-free extractions are audited but excluded from automatic audit
correctness scores. The original submission's campaign (May 2026, six models,
sixteen items per request) is referred to below as the *archived* campaign;
its summaries are under `outputs/archive/2026-05-grouped-cohort/` and are not
relabelled. The [generation qualification](internal/generation_qualification_2026-09-11.md)
documents the formatting comparison and failure accounting behind the local
profile; the launch gates are in [final-run readiness](internal/final_run_readiness.md).

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
| Delivery | One benchmark item per request; 4 and 16 items per request are the request-composition ablation (see §4). |
| Provenance per response | Request seed sent; served model identifier, request and response bodies, and resolved settings recorded (§6). |

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
   generated statements was checked by manual review and additionally with
   LanguageTool 6.8 at its default rule level
   (`uv run --with language-tool-python python scripts/check_benchmark_grammar.py`;
   results in `outputs/benchmark_grammar_check.{csv,md}`). Of 3,060 distinct
   sentences (540 capability clauses, including PURE and the phrasing probe), it
   flags grammar in 7 clauses and never in the template wording. All 7 keep
   wording copied verbatim from the source requirement: two are false positives
   ("transfer to Shunting", "on/off key"), four are hyphenation or article
   conventions, and one is a repeated word ("vehicle vehicle", S0240). Review
   tables are tracked in
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

The four `probe_*` templates are run as a separate probe to check that the
weak-intent finding is not an artifact of the single `It would be useful if …`
wording. `scripts/run_weak_modality_probe.py` runs every benchmark capability
of one cell through all four templates (a smoke run keeps the 20 pilot seeds)
and writes one directory per run under `outputs/weak_modality_probe/<run_id>/`:
the per-template summary and the paired text-strengthening deltas against the
`useful_if` baseline. The rerun driver runs it on the representative model of
each endpoint. The probe is a diagnostic, not a headline result.

### 2.4 Construct-validity review

`docs/weak_modality_construct_review.csv` retains the two original
LLM-assisted reviews and adds separate author-confirmation rows. The author
confirmed completed human validation on 2026-09-04 and will repeat it before
submission. The assistant also checked the benchmark construction and four
weak-template mappings. See [validation review](validation_review.md) for the
scope and the wording parser's mixed-clause limitation. Two independent human
raters or agreement statistics are not claimed.

## 3. Prompts

All prompts are frozen plain-text files under `prompts/`, content-addressed by
SHA-256 in `outputs/benchmark_manifest*.json`, and verified by the analysis
gate. `prompts/README.md` is the index, and README.md reproduces the
single-item prompt bodies.

There are two layers:

| Layer | Where | Used by |
| --- | --- | --- |
| Single-item prompt | `prompts/*.txt`, rendered by `prompt_for_benchmark_task` | **Every request of the reported campaign**, and the single-item reference arm of the request-composition ablation. |
| Batched wrapper | `batch_prompt_for_completion_jobs` in `scripts/eval_utils.py` | The 4- and 16-item arms of the request-composition ablation, and the archived May campaign. |

The single-item files define the task contract, the label set, and the
confidence contract, and they are the request body of the reported runs. The
batched wrapper restates them for several items at once and asks for an array
of results keyed by `request_index`; the two agree on task, labels, and
confidence scale and differ in surface form. The wrapper bodies below are
reproduced because the ablation and the archived campaign sent them.

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

No system message is sent on either path; the prompt is the entire user message.

## 4. Request Composition: The Reported Protocol And The Ablation

Every request of the reported campaign carries **one** benchmark item. The
request-composition ablation repeats the deterministic Task 2 pass on the
MLM-TAPT/MUST cell at 4 and 16 items per request, either keeping the four
source conditions of a capability in one request (grouped) or spreading them
across requests (sibling-separated), against a fresh single-item reference,
for all nine models (GLM-5.3-Flash added on 2026-09-24 with the same protocol,
`conf/rerun/final_flash_ablations.yaml`). The comparison table is
`outputs/batching_ablation_summary.md`; the paper reports that several items
per request lower the weak-intent strengthening rate by tens of percentage
points for most models, so single-item results do not transfer to batched use.

The rest of this section records the batching policy of the **archived** May
2026 campaign, whose numbers are not the paper's.

| Property | Value in the archived runs |
| --- | --- |
| Items per request | 16 (Task 1, Task 2, and Task 3) in every archived-cohort run. A handful of early registry rows used 8; they are not part of any reported cell. |
| Batch membership | Consecutive `request_index` values, no shuffling. |
| Benchmark row order | seed × variant: the four conditions of one seed are adjacent. |
| Consequence | Every Task 2 batch of 16 contains all four modality variants (`MUST`, `SHOULD`, `MAY`, `It would be useful if …`) of the same four seeds, side by side. |
| Max tokens | 256 per item, multiplied by the batch size for the request. |
| Batch id | `run_id:model:task:sample_kind:sample_index:<min>-<max> request index`. |
| Fallback | **None on the path the reported runs used** (see below). Single-item re-sends now exist on every path. |

**The batch fallback did not exist for the archived runs.** Only the Instructor
path re-sent the items of an unparsable batch as single-item requests. The raw
`response_format: {"type": "json_object"}` path — the one every archived run
took — wrote a `missing_batch_result` row per affected item and never re-sent
it. Across the four cells that is 1, 1, 0 and 16 Task 1 deterministic rows and
48, 65, 60 and 51 Task 1 stochastic rows, plus 95 Task 2 stochastic rows in
`nice/shall`. **Task 2 deterministic rows had none**, so no strengthening,
label-accuracy, or confidence headline is affected; the Task 1 control and the
stochastic stability metrics lose the listed items, which are excluded rather
than counted (§7). The raw path now falls back to single-item requests as well,
so from this change onward the fallback applies to every path.

**Why it mattered.** The prompt instructs the model to evaluate each item
independently, but a batched model sees the minimal-pair contrast inside its
own context window. The reported campaign removes the confound by sending one
item per request; the ablation measures its size.

`batch_order` is a knob, settable per profile or run-wide, with values
`grouped` (the archived policy and one ablation arm) and `shuffled`; together
with `batch_size` it defines the ablation arms. The resolved value is recorded
in the run registry column `batch_order`.

`shuffled` is a **constrained** shuffle: it never places two source variants of
one seed in the same batch, and it is derived deterministically from the
recorded run seed, so the arm is reproducible. The first implementation only
permuted job order, which at batch size 16 still left two variants of one seed
together in roughly half of the 45 Task 2 batches (22 to 24 depending on the derived RNG seed) — it would have weakened the confound, not
removed it. Resume no longer re-shuffles the pending subset, so a resumed
shuffled run keeps the batch membership of the original run. The
historical comparison used size 16 for both batch-order arms. The final
campaign instead uses single-item primary requests, with size-4 and size-16
grouped/sibling-separated ablations. Historical numbers retain their original
protocol; see [final-run readiness](internal/final_run_readiness.md).

## 5. Model Cohort And Request Parameters

### 5.1 Cohort

Nine models from five independently developed families: two hosted GLM models
on the Z.AI coding endpoint (`conf/profile/zai.yaml`) and seven open-weight
models served by llama.cpp (`conf/profile/local_llama_cpp.yaml`; build b10900,
Unsloth GGUF repositories, UD-Q4_K_XL quantisation, one NVIDIA RTX 6000 Pro).

| Model | Request model id | Repository | Thinking |
| --- | --- | --- | --- |
| GLM-5.3 | `glm-5.3` | hosted | disabled |
| GLM-5.3-Flash | `glm-5.3-flash` | hosted | disabled |
| Qwen3.8-27B | `qwen3.8-27b` | `Qwen3.8-27B-GGUF` | off |
| Qwen3.6-27B | `qwen3.6-27b` | `Qwen3.6-27B-GGUF` | off |
| Qwen3.5-9B | `qwen3.5-9b` | `Qwen3.5-9B-GGUF` | off |
| Gemma-4-31B | `gemma4-31b-it` | `gemma-4-31B-it-GGUF` | off |
| Gemma-4-12B | `gemma4-12b-it` | `gemma-4-12B-it-qat-GGUF` | off |
| Muse-Glimmer-30B | `muse-glimmer-30b` | `Muse-Glimmer-30B-GGUF` | low effort |
| gpt-oss-20B | `gpt-oss-20b` | `gpt-oss-20b-GGUF` | low effort |

The two hosted models share a developer, so the main results are reported per
model and conclusions are restricted to the evaluated models and conditions. The
Z.AI coding endpoint answers requests for older GLM ids with these two served
models (recorded in `served_model`), which is why only the two served ids are
configured.

What is recorded to identify the model and the request, as asked for in
review: the served model identifier and `system_fingerprint` per response,
the request seed, the full request and response bodies in the transcripts,
the resolved profile settings, and the fact that no system prompt, tools,
memory, or conversation history are used (§6). Recorded seeds do not make
local outputs byte-identical under concurrent serving (§5.2.1).

`azure.*` rows may exist in local registries. They are private-endpoint
diagnostics and are excluded from every paper-facing aggregate
(`--exclude-model-prefix azure.` in `scripts/compare_run_matrix.py`).

### 5.2 Request parameters

| Parameter | Deterministic pass | Stochastic pass |
| --- | --- | --- |
| `temperature` | 0.0 | 0.7 |
| `top_p` | 1.0 | 1.0 |
| Samples per item | 1 | 5 |
| `max_tokens` | 256 per item hosted, 1024 per item local | same |
| Output format | hosted: `response_format: {"type": "json_object"}`; local: the task's JSON schema (`structured_output: json_schema`) | same |
| System prompt | none | none |
| `seed` | 20260518, sent (`send_seed: true`) | seed+1 to seed+5 for the five repetitions, sent |
| Retries | `call_with_retries`: up to 3 attempts on 408/429/5xx, timeouts and connection errors, each recorded (`retry_count`); SDK-internal retries disabled | same |

The archived runs had **no** application-level retry layer. The OpenAI SDK
client was constructed with its default `max_retries=2`, which silently retries
408/409/429/5xx responses and connection errors, so a batch could be sent up to
three times without any record of it. In the reported campaign the SDK's internal retries
are disabled (`max_retries=0`) and `call_with_retries` is the only retry layer:
3 attempts, retrying transient 408/429/5xx, timeouts and connection errors, and failing
fast on 400/401/403/404/422 and every other 4xx. Native output-format HTTP 500
rejections are generated failures, not transient errors: they remain failed
samples and are not replaced during ordinary resume. `retry_count` and
`retry_total` are now recorded for batched rows as well (see §6).

These are explicit profile knobs: `seed`, `send_seed`,
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
The `local_llama_cpp` profile uses `structured_output: json_schema` without the
legacy `json_mode` flag, as the reported local runs did; the earlier
unconstrained local smoke runs keep their recorded settings.

### 5.2.1 Local llama.cpp server (September 2026 cohort)

The seven local models run on one llama.cpp server (build b10900) behind
llama-swap v255 on a single RTX PRO 6000 (96 GB), one model resident at a
time, all layers on the GPU, flash attention on, continuous batching on,
unquantised KV cache, context 131,072 shared across the request slots. The
harness sends `temperature`, `top_p`, `seed`, `max_tokens` and the thinking
switch (`chat_template_kwargs.enable_thinking=false` for Qwen and Gemma,
`reasoning_effort: low` for Muse Glimmer and gpt-oss, whose reasoning the
server separates into `reasoning_content`). Parameters the harness does not
send keep the GGUF metadata defaults, which differ per model family: `top_k`
20 (Qwen3.6/3.8), 40 (Qwen3.5-9B, gpt-oss, Muse Glimmer), 64 (Gemma 4);
`min_p` 0.05; repetition penalty off. The server honours every parameter
sent: a repeated request with the same seed at temperature 0.7 returns a
byte-identical answer, checked 2026-09-10.

Concurrency is a throughput knob with one reproducibility caveat. The
headline cohort ran with two harness workers against four server slots
(`--parallel 4`). After that cohort the server was raised to eight slots and
the profile to eight workers, because decode on one GPU is bandwidth-bound and
extra streams cost almost nothing per step. Request content, batch membership
and request seeds are planned before dispatch and do not depend on the worker
count; only the order in which rows land in the raw file does, and every
downstream step keys on identifiers rather than file position. Whether the
seed-repeat identity survives concurrency was checked on 2026-09-11 by
replaying recorded Qwen3.5-9B 16-item batches with their original seeds. It
does not, and the loss is not specific to the slot change: on the original
4-slot server, 8 stochastic batches replayed byte-identically 7/8 times at one
worker, 6/8 at two and 4/8 at four; after the move to 8 slots the same batches
matched 3/8 at one worker and 1/8 at eight. Temperature-0 batches matched 6/8
at either worker count. The differences are wording changes inside the
generated requirement text (a few dozen of 128 items per replay), never the
modality label. Co-scheduled load and floating-point execution are plausible
contributors, but these observations do not isolate the mechanism or establish
that matching the load guarantees byte identity. The replay counts above are
recorded observations; the underlying replay artifacts were not located in the
2026-09-11 worktree review. Because wording is the evaluated outcome, unchanged
labels do not establish unchanged strengthening or semantic-variation scores.
The reproducibility claim for the local models is therefore at the level
of the sampling configuration and the served model file, not byte identity of
individual answers; the 2026-09-10 byte-identical repeats were obtained on an
otherwise idle server.

The model files, as recorded in the served-model identifier of every local
response of the final campaign (61,200 per model; all Unsloth dynamic 4-bit
quantisations, `UD-Q4_K_XL`):

| Model id | Served model (Hugging Face repository : quantisation) |
| --- | --- |
| `qwen3.8-27b` | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL` |
| `qwen3.6-27b` | `unsloth/Qwen3.6-27B-GGUF:UD-Q4_K_XL:Q4_K_XL` (recorded with the quantisation type appended) |
| `qwen3.5-9b` | `unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL` |
| `gemma4-31b-it` | `unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL` |
| `gemma4-12b-it` | `unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL` (quantisation-aware-trained base) |
| `muse-glimmer-30b` | `unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL` |
| `gpt-oss-20b` | `unsloth/gpt-oss-20b-GGUF:UD-Q4_K_XL` |

The exact files, read from the server's Hugging Face cache on 2026-09-24:
the repository revision (commit) of the cached snapshot, the file name, and
the SHA-256 of the file content (`sha256sum`, which equals the Git LFS object
id on Hugging Face, so a download can be checked against the published
revision). Every file was written to the cache before the final campaign
started (2026-09-11 21:52 CEST), and none has changed since. For
`qwen3.6-27b`, whose served identifier carries the appended quantisation type,
the server's `/props` endpoint confirms the loaded file.

| Model id | Revision | File | SHA-256 |
| --- | --- | --- | --- |
| `qwen3.8-27b` | `4ca720788d1e01f1bff70c033e0d0028fd02e502` | `Qwen3.8-27B-UD-Q4_K_XL.gguf` | `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e` |
| `qwen3.6-27b` | `82d411acf4a06cfb8d9b073a5211bf410bfc29bf` | `Qwen3.6-27B-UD-Q4_K_XL.gguf` | `ff6941ded525b34eb159496762c29dd0ec6e71dc31b74d57e75d871a03eec259` |
| `qwen3.5-9b` | `3885219b6810b007914f3a7950a8d1b469d598a5` | `Qwen3.5-9B-UD-Q4_K_XL.gguf` | `6f5d30666c2d8ae16a306e616d95341dcf3cc46810df84d7e6f5a7d1e4c1b293` |
| `gemma4-31b-it` | `c1ac76e99d5513b141e8adde7288b85c3f9c32ec` | `gemma-4-31B-it-UD-Q4_K_XL.gguf` | `9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575` |
| `gemma4-12b-it` | `980b060c40a8539ac159e0501a3e0f66a6365af3` | `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` | `90fd44e29e0d7cffeb0fd00dc73cfdab9ed0b0e95306ecf7821ea634c940c370` |
| `muse-glimmer-30b` | `faa5b025c584459c13febfa5c59883516710ae39` | `Muse-Glimmer-30B-UD-Q4_K_XL.gguf` | `82bece304887a313ece08400bc030f6066c7bff5b906b0cd40308ec8a409fd38` |
| `gpt-oss-20b` | `d449b42d93e1c2c7bda5312f5c25c8fb91dfa9b4` | `gpt-oss-20b-UD-Q4_K_XL.gguf` | `10fe673de12c20b74b8d670a9fdf0fd36b43b0a86ffc04daeb175c0a2b98c4f9` |

The snapshots also hold multimodal projector files (`mmproj-*.gguf`), which
llama.cpp loads alongside the model (the server reports vision support).
Every request is text only, so the projector processes no input.

To serve them yourself, load these files in any OpenAI-compatible llama.cpp
server (llama-swap routes by the request's `model` field), name them with the
ids above, and point `conf/profile/local_llama_cpp.yaml` at it via
`RE_UQ_LOCAL_LLAMA_CPP_BASE_URL`. The server configuration of the authors'
machine is not part of the package; the settings above are the ones that
matter for the outputs.

Intervals from the saved generations are conditional on those outputs; they
do not measure variability across repeated server executions.

Temperature 0.0 is treated as deterministic. It is not guaranteed to be
deterministic on a hosted endpoint, and without a request seed and a recorded
served-model version this cannot be checked after the fact for the existing
runs.

### 5.2.2 Adding your own model

Any OpenAI-compatible chat-completions endpoint works. Add a profile under
`conf/profile/` (copy `local_llama_cpp.yaml` or `zai.yaml`: endpoint, the name
of the environment variable that holds your key, model ids, concurrency,
output format, thinking switch), then copy `conf/rerun/final.yaml` with its own
`run_group_id` and your profile and models, and start
`scripts/rerun_all.py --config <your copy>`. Nothing downstream is specific to
the nine models: the analysis, the tables and
`scripts/verify_paper_numbers.py` take the model ids from the state file. A new
`run_group_id` keeps your runs out of the reported campaign.

### 5.3 Configuration provenance

The reported campaign is fully described by `conf/rerun/final.yaml` and the
two profiles it names, as tracked, except that the local profile's endpoint
address is a placeholder (`base_url`; see `outputs/README.md`).
`outputs/rerun/manuscript-final/state.json` records the resolved configuration
and every run id; `outputs/paper_snapshot_provenance.json` names the run each
table pooled per model and cell.

The archived May campaign (run group `provider-matrix-2026-05`, prompt
version `v1` under the output contract `prompt_v2_confidence_0_1`) predates
that discipline: the configs of the first release could not have produced it,
its registries still hold `complete` rows whose raw rows were removed later,
and its prompt text was verified only by hash against the batched builder. Its
summaries are archived and not cited by the revised manuscript.

## 6. What Is Recorded Per Response

Raw records are written to `data/processed/model_outputs_raw*.jsonl`
(local-only, archived in the Zenodo dataset record; see [`docs/repository_hygiene.md`](repository_hygiene.md)).

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
the archived runs and the 4- and 16-item ablation arms — the
writer dropped the driver's `retry_count`, `request_seed` and
`request_payload_sha`, so `retry_count` read 0 on every batched row whatever
had actually happened, and the `request_payload_sha` that was recorded came
from the *single-item* prompt rather than from the batch payload that was sent.
Both are fixed: batched rows now carry the driver's retry count and request
seed, and `request_payload_sha` hashes the batch payload. As with everything
else in this column, the fix applies to runs made from this change onward;
already-written raw rows keep the old values.

The gap that mattered for the archived campaign: those runs recorded the
**requested** model string, not the **served** model version, and sent no
request seed. Both are recorded in the reported campaign, but the archived raw outputs cannot
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
3. **Request composition.** The reported numbers are single-item. The
   ablation (§4) shows that 4 and 16 items per request lower weak-intent
   strengthening by tens of percentage points for most models, so they do not
   transfer to batched use; batched deployments need their own measurement.
4. **Model families.** Nine models from five families, two of them hosted by
   one developer; the main results are reported per model and conclusions are
   restricted to the evaluated models and conditions.
5. **Provenance of local generation.** Request seeds are sent and the served
   model is recorded, but recorded seeds do not make local outputs
   byte-identical under concurrent serving (§5.2.1); an exact replay is not
   guaranteed.
6. **Construct review completed by the author.** The original LLM-assisted
   judgments remain labelled separately; see `docs/validation_review.md`. The
   operational ordering still does not establish intent in arbitrary contexts.
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
