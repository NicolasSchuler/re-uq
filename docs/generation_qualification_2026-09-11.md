# Generation qualification — 11 September 2026

This follow-up tests the failure modes found in the first live smoke. It is
operational qualification, not new manuscript effect estimates. No full
campaign, model training, container rebuild, remote configuration edit, or
publication was performed.

Verdict: the revised local profile passed the bounded operational qualification.
Across 88 stages, 4,090 item-level records correspond to 3,712 provider calls.
The final schema condition contains 2,581/2,581 valid records in 2,203 calls,
with no retries, returned length terminations, or parser repairs. This includes:

| Schema-enabled check | Valid item records |
| --- | ---: |
| Fresh single-item Task 1/2, seven models | 1,344/1,344 |
| Blind audit, seven models | 672/672 |
| PURE document context and weak phrases | 112/112 |
| Grouped/shuffled size-4/16 checks, 70 calls | 448/448 |
| Previously failing GPT-OSS request replay | 5/5 |

The unconstrained diagnostic arms retain four native GPT-OSS failures and nine
invalid Qwen3.5 audit responses. Their failure records were not rewritten.
Hosted qualification and full-campaign authorization remain outstanding.

## Implemented changes

- The local profile and its JSON example now request `json_schema`, using the
  existing task-specific JSON contracts. Primary request size remains one item,
  with one deterministic answer and five stochastic samples, concurrency 8,
  1,024 output tokens per item, and the existing low/disabled reasoning settings.
- Native output-format rejections have `error_kind=model_output_format` and
  `parse_status=model_output_error`, distinct from transient transport errors.
  They are not replaced by transport retry, batch fallback, Instructor's own
  retry loop, or same-configuration resume. Their failure remains in the
  planned-sample denominator. A changed configuration needs a new condition/run.
- Every nonempty, structurally valid deterministic Task 2 extraction is eligible
  for a blind audit, including unknown/negated extracted wording. These cases
  have an explicit unavailable reference and no automatic correctness label.
  Run analyses and paper exports write `task3_audit_review.csv` and
  `task3_audit_coverage.csv`. Missing audits remain visible in coverage.
- Production Task 3 scoring uses the declared stochastic sample count, not
  merely the number of rows returned. Companion notebook sources match.

The independent code review found that ordinary resume could still replace a
native failure. That path was corrected and regression-tested; the review
reported no remaining correctness finding in the scoped implementation.

## GPT-OSS mechanism and remedy

The unchanged container serves `unsloth/gpt-oss-20b-GGUF:UD-Q4_K_XL` with
llama.cpp build `b10900-50182a53f`, eight slots. The buffered upstream logs
contain complete JSON answers preceded by a final-channel JSON-format marker.
The corresponding unconstrained GPT-OSS parser branch expects the final channel
to be immediately followed by the message delimiter. Its structured-response
branch accepts the optional format marker. This explains the captured rejection
mechanism; it is not evidence of a generic network fault or insufficient output
tokens. See the [exact build's GPT-OSS parser](https://github.com/ggml-org/llama.cpp/blob/50182a53f/common/parsers/gpt-oss.cpp).

The format separator belongs to the model's structured conversation format.
The remedy preserves that format and reasoning/final separation; it does not
strip reasoning into the scored answer or bypass the native parser. This follows
the [OpenAI Harmony guidance](https://developers.openai.com/cookbook/articles/openai-harmony).

Five previously rejected logical requests passed once each on replay with both
formats. That small replay alone did not establish a fix: a fresh 192-request
test without a schema produced four native failures, all in Task 1 stochastic
samples (4/80 in that stratum). The matched schema-enabled test passed 192/192
with no retry and no returned length termination. No server restart was needed.

The raw `/props` values are server defaults, not all effective chat-request
settings: the request overrides temperature, top-p, budget and reasoning effort.
GPT-OSS's inspected defaults also include top-k 40 and min-p approximately 0.05;
top-p 1 alone must not be described as unrestricted sampling. The chat adapter
[maps the requested reasoning effort into template arguments](https://github.com/ggml-org/llama.cpp/blob/50182a53f/tools/server/server-common.cpp).

## Scope and scientific boundaries

The completed audit comparison used 96 responses per arm/model:

| Model | Unconstrained valid | JSON-schema valid |
| --- | ---: | ---: |
| GPT-OSS-20B | 96 | 96 |
| Qwen3.6-27B | 96 | 96 |
| Qwen3.8-27B | 96 | 96 |
| Qwen3.5-9B | 87 | 96 |
| Gemma4-31B | 96 | 96 |
| Gemma4-12B | 96 | 96 |
| Muse Glimmer-30B | 96 | 96 |

The fresh primary panel has eight NICE/SHALL and eight PURE/MUST items per model:
two fresh capabilities per dataset, one near median statement length and one
longest available, with all four modality variants. Each item receives Task 1
and Task 2, deterministic plus five stochastic responses (192 calls/model).
Additional deterministic checks use document context and all four weak phrases.
The selected bare request prompts have 71–129 words; document prompts have
172–178 words. This exercises longer benchmark cases, not long-context capacity.
Small grouped/shuffled size-4/16 checks qualify structured batched responses;
they do not estimate a batch-size effect.

The audit format comparison holds source extractions, prompts, sample indices,
and seeds fixed, with 16 source outputs × six answers per arm and model. It runs
the unconstrained arm before the schema arm, so it is not a counterbalanced
causal estimate of format effects. For Qwen3.5, valid audit responses improved
from 87/96 to 96/96; Qwen3.6 passed both arms this time. In the first four models,
jointly valid relation judgments changed in 1/96 GPT-OSS, 0/96 Qwen3.6, 2/87
Qwen3.5 and 5/96 Muse pairs. Thus format support is not semantic invariance.
Identical requested seeds also did not guarantee bitwise replay under this
continuously batched GPU server: the five historical failing requests passed
when replayed. Preserve request order/concurrency and raw outputs; do not infer
repeat-server uncertainty from a single saved-generation bootstrap.

All four previously excluded bare extractions (three GPT-OSS, one Muse) are now
audited in both arms. Their judgments are review material, not reference-based
accuracy evidence. This qualification does not settle natural-language modality
ambiguity, reviewer-requested construct validity, model calibration, or final
effect sizes. Historical unconstrained/grouped results remain separate.

A compact methods sentence for the new campaign is: “Local models generated
task-specific, schema-constrained JSON using single-item requests; uncertainty
was estimated from one deterministic answer and five stochastic samples, with
generation failures retained in the planned-sample denominator.” State the
hosted format separately and report batch composition as an ablation.

## Evidence and launch status

The full repository suite passed 725 tests, including 14 expected failures.
Ruff, whitespace checks, and the complete final-plan dry-run passed. No new
failing tests were waived. The cached-artifact verifier rechecks logical sample
counts, actual request hashes/seeds, parser results, and batched item-to-request
and response-slice correspondence without making provider calls or loading
embedding models.

Raw responses, exact request transcripts, selected inputs and per-stage summaries
are isolated under `outputs/final_run_readiness/qualification-*`; the local
diagnostic entry point is `outputs/final_run_readiness/qualify_generation.py`.
Serving evidence is under `outputs/final_run_readiness/gpt-serving-inspection/`.
`observed-parser-log-excerpts.md` preserves exact warnings from the first log
read's tool output; the original JSON capture was accidentally overwritten by
a later empty-buffer diagnostic read. The later empty capture is retained and
the script now writes timestamped log captures. The complete provider response
and request transcripts were unaffected.
These diagnostic artifacts are ignored by Git and must be included explicitly
if used in a frozen evidence archive. Original smoke outputs are unchanged.
The final machine-readable verification report is
`outputs/final_run_readiness/qualification-verification.json`; its 88 stage
entries identify every raw-response and transcript file. All SSH tunnels opened
by these diagnostics were closed on completion.

Hosted qualification is pending: the approval guard blocked sending selected
NICE/PURE prompts to `api.z.ai`. No hosted requests were made in this follow-up.
The user supplied the remote `.zshenv` location for `ZAI_API_KEY`; credentials
are read only in memory and are never included in transcripts or saved profiles.
Explicit payload/destination approval remains necessary. The cap is 192 logical
requests per hosted model. The full manuscript campaign has not been launched.
