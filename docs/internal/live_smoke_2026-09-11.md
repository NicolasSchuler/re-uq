# Live GPU smoke: 11 September 2026

This records the original unconstrained-output smoke. The later
[generation qualification](generation_qualification_2026-09-11.md) diagnoses its
failure modes and documents the new schema-enabled profile and wider audit
coverage; it does not relabel or replace these historical results.

## Verdict

The local single-item execution path works across all seven configured models,
with five stochastic samples plus one deterministic answer per item/task.
This is a **qualified operational pass**, not unconditional final-campaign or
manuscript readiness. Before launch, resolve or explicitly accept the GPT-OSS
server-format retries and stochastic audit formatting failures, and retain
the audit exclusions in every relevant denominator.

No durable provider profile, server configuration, prompt, parser, or model
output was changed to obtain these results. No full campaign was launched.

## Findings and decisions before launch

1. **GPT-OSS server output-format errors.** Five stochastic primary requests
   received HTTP 500 with: `The model produced output that does not match the
   expected peg-native format`. All recovered through the existing retry
   policy; 192 logical answers required 197 attempts. This is a server-side
   output-format rejection, not a credential failure or observed token-limit
   truncation. The smoke does not establish whether the root cause is model
   emission, the chat template, or the server parser. Inspect compatibility
   inside the actual container before freezing the server build; repeat a
   focused stochastic check after any change. Do not erase or hide these
   attempts when reporting reliability or sampling behavior.
2. **Stochastic audit JSON failures.** Qwen3.6 returned 2 invalid JSON answers
   among 96 audit responses; Qwen3.5 returned 8 among 96. All ten were
   stochastic: 2/80 and 8/80, respectively. Missing closing quotes or literal
   newlines inside quoted strings caused the failures. These were normal
   `stop` responses, not truncation, and the audit prompt's example has its
   closing quote. Deterministic audit answers all parsed. Keep these samples
   incomplete in UQ analysis. If lower format-failure rates are required,
   test an explicit structured-output condition separately before choosing
   and freezing it; do not silently repair or replace failed generations.
3. **Audit coverage is conditional on text-modality recognition.** All primary
   Task 2 answers were structurally parseable, but the conservative text
   scorer could not classify one Muse and three GPT-OSS deterministic outputs.
   The existing Task 3 builder consequently excluded them. Muse audited 15/16
   source outputs and GPT-OSS 13/16, completing 90/90 and 78/78 eligible
   responses. This is intended filtering, not missing transport jobs. Report
   the excluded source outputs, including unknown wording, rather than
   describing the audit as covering every valid extraction. Review unknown
   cases before interpreting strengthening or audit-detection rates.

The excluded items are:

| Model | Source item | Generated wording |
| --- | --- | --- |
| Muse Glimmer | `S0456_mandatory` | `provide communication services for the S-NPP, JPSS-1, and JPSS-2 missions` |
| GPT-OSS | `S0391_mandatory` | `provide space for storage of mobile equipment and electrical equipment` |
| GPT-OSS | `S0391_optional` | `provide space for storage of mobile equipment and electrical equipment` |
| GPT-OSS | `S0456_mandatory` | `provide communication services for the S-NPP, JPSS-1, and JPSS-2 missions` |

These bare verb phrases have no explicit modality and do not match the
scorer's subject-plus-verb heuristic. An unknown outcome is not evidence of
modality preservation. In particular, GPT-OSS used identical wording for one
mandatory and optional source while reporting different modality labels.

## Coverage and results

Every model received:

- Primary MLM-TAPT/MUST: the first 16 items (four capabilities, all four
  modalities), Task 1 and Task 2, single-item requests, one deterministic
  answer and five stochastic samples: 192 item responses.
- Blind Task 3: eligible deterministic Task 2 outputs, with the same
  deterministic/stochastic sampling plan.
- Deterministic Task 2 batching: 64 MLM-TAPT/MUST items (16 capabilities),
  grouped and sibling-separated requests at sizes 4 and 16: four arms,
  256 item responses in 40 requests. The size-1 primary smoke uses only
  16 items; this is not a matched-rate ablation analysis or a separate
  64-item single-arm repeat.
- PURE context: the same 16 rebuilt items in bare and document-context arms,
  deterministic Task 2 with single-item requests: 32 responses.

| Model | Primary parsed | Audit parsed / eligible responses | Audited source outputs | Primary elapsed seconds |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6-27B | 192/192 | 94/96 | 16/16 | 62.48 |
| Qwen3.8-27B | 192/192 | 96/96 | 16/16 | 64.29 |
| Qwen3.5-9B | 192/192 | 88/96 | 16/16 | 30.94 |
| Gemma4-31B | 192/192 | 96/96 | 16/16 | 61.41 |
| Gemma4-12B | 192/192 | 96/96 | 16/16 | 30.50 |
| Muse Glimmer-30B | 192/192 | 90/90 | 15/16 | 95.87 |
| GPT-OSS-20B | 192/192 | 78/78 | 13/16 | 34.32 |

All 1,792 batched item responses and all 224 context responses parsed without
fallback requests, transport retries, or truncation. Overall there are 4,008
saved item responses: 3,998 parseable and ten invalid JSON audit responses.
The transcripts contain 2,501 workload attempts, including the five failed
GPT-OSS attempts; 56 additional successful preflight calls are outside those
transcripts. There were no unresolved request errors and no observed
`finish_reason=length` responses.

Parseable does not mean strict JSON-only compliance: both Gemma models wrapped
all their primary, audit and context responses in Markdown fences (640
records); four GPT-OSS responses also required wrapper removal. The existing
tolerant parser records these 644 `prose_wrapper` repairs. Batch-wrapper
formatting is not counted as a per-item repair here. No other per-item repairs
were observed.

The primary elapsed times include subprocess startup, preflight, possible
model loading and generation. They are smoke timings, not steady-state
throughput measurements or final-campaign duration forecasts. Maximum primary
completion-token usage was 264 for Muse and 167 for GPT-OSS, within the 1,024
per-item limit. Longer inputs and a larger population remain untested.

## Runtime and verification

The endpoint is served by Docker container `llama-swap`, image
`lab-llama-swap:local`, exposed from container port 8080 to host port 9292.
The in-container `llama-server --version` reports:

```text
version: 0.4.0-dev (build 10900, commit 50182a53f)
built with GNU 13.3.0 for Linux x86_64
```

The image ID observed during this smoke was
`sha256:69c71f0259846917573b039fa1ae32583a8805d585c4f15551eff1a42a68cd0d`.
The running Qwen3.6 and Muse workers both showed `--parallel 8` and
`--ctx-size 131072`. The GPU is an NVIDIA RTX PRO 6000 Blackwell Workstation
Edition with 97,887 MiB reported memory. Muse used about 20,216 MiB during
the observed active request workload. No server/container restart or
configuration modification was performed.

Requests used an SSH tunnel to GPU host loopback port 9292, not the profile's
unencrypted network address. The remote key was loaded only into process
memory, without sourcing or displaying the host's shell environment file; it
was not saved into
configuration or diagnostic files. The temporary tunnels were closed. Their
ephemeral local addresses remain in smoke provenance and are not reusable
final-run endpoints. Use an appropriate encrypted route for a remote client
when launching the final campaign.

Cached verification checked all completed runs for duplicate logical answers,
the declared six-response primary/audit sample grids, requested model names,
temperatures, seeds, per-item-scaled token budgets, thinking options, user-only
messages, current benchmark matching, PURE context content, and blind audits.
Successful stored parses reproduced from raw text. All separated 4-item
batches contained four distinct capabilities; separated 16-item batches
contained sixteen. Grouped batches contained one and four capabilities,
respectively. The only excess workload attempts were the five recorded
GPT-OSS retries; no batching fallback occurred.

The final verifier derives audit eligibility from the actual source outputs.
The initial harness summaries conservatively expected all 16 source outputs
to be eligible, so their `healthy=false` flag for Muse/GPT-OSS audits denotes
that coverage reduction, not a failed eligible job. GPT-OSS's initial primary
`healthy=false` flag records recovered transport errors. Its remaining arms
were run separately against the same saved primary outputs after inspection.

Deterministic scores and lightweight label-distribution UQ were recomputed
from the primary raw outputs: 224 complete five-sample item/task groups and
896 score rows across the seven models. Probabilities normalized correctly
and label entropies were finite. Semantic embeddings, classifier fits,
calibration figures and paper exports were not recomputed in this smoke.
Whole-second transcript start times are insufficient to infer exact
concurrency from timestamp overlap; the configuration and observed worker
flags, not such a reconstruction, establish the slot setting.

Exact served-model names, including quantization and the Gemma4-12B QAT
variant, are retained in the transcripts and verification artifact. Failed
GPT-OSS attempts have no served-model name; successful responses match the
requested model families and sizes.

## Evidence and remaining scope

Detailed, ignored local artifacts:

- `outputs/final_run_readiness/live-smoke-20260911-170915/`: Qwen3.6.
- `outputs/final_run_readiness/live-smoke-20260911-171346/`: the other six
  primary runs and all non-GPT-OSS secondary arms.
- `outputs/final_run_readiness/live-smoke-20260911-173125/`: GPT-OSS audit,
  batching and context continuation, reusing its earlier primary run.
- `outputs/final_run_readiness/live_smoke_verification.json`: per-run
  protocol checks, current-source eligibility, repairs, and transcript paths.
- `outputs/final_run_readiness/live_smoke_lightweight_scores.json`: cached
  deterministic and label-UQ recomputation, without semantic-model loading.

Raw responses and registries are isolated under `data/processed/smoke/`;
transmitted requests and full provider responses are under
`data/processed/logs/` with the explicit smoke run IDs. Historical and
paper-facing outputs were not overwritten.

This smoke did not cover the hosted Z.AI models, NICE/SHALL generation cells,
the weak-phrasing study, long-duration server stability, full-dataset failure
rates, embedding/classifier convergence, refreshed manuscript figures, or
manuscript compilation. It cannot establish modality-error rates, batching
invariance, audit accuracy, or naturalistic validity. The broader gates in
`docs/final_run_readiness.md` still apply. Freeze failure/exclusion handling
and the server/profile combination before authorizing the full workload.
