# Final-run readiness

Implementation started from commit `66aeaac` on `manuscript/final-run-readiness`.
This is a readiness record, not authorization to launch or a claim that all
findings have been resolved.

## Implemented

- Positive obligations/recommendations are no longer hidden by weak phrases;
  mixed wording remains visible to the output review.
- Batching deltas use jointly eligible exact source items. Full-arm rates,
  failures, unreadable text, duplicate identities and exclusions are separate.
  Capability-bootstrap intervals are conditional on the saved generations;
  they do not model cross-capability request dependence or repeated-server
  variability. An interval crossing zero does not demonstrate invariance.
- The final primary protocol is one item per request, with one deterministic
  answer and five stochastic samples per item/task. All shipped provider
  defaults use size 1. Final preflight rejects drift from these settings.
  Batch-size identities stay distinct in selection, deltas use a named
  baseline, and paper exports read the recorded request protocol.
- PURE automatic suggestions retain non-system subjects and conditions for
  manual rewriting. Source selection remains the existing deterministic
  sampling procedure, not semantic approval. The builder rejects residual
  modalities, suspicious subject deletion, duplicate clauses and source-copy
  conflicts before publishing outputs.
- `--refresh-analysis` reruns analysis in dependency order from the explicitly
  selected state's recorded configuration, without reading live profiles or
  allowing generation. Downstream completion flags are invalidated before
  refresh, so a failure cannot leave stale figures marked complete.
- Classifier CLIs accept `--hgb-max-iter` and optional finite `--hgb-budgets`.
  Budget selection uses inner capability-separated validation, including
  inner-fold preprocessing, then refits on the outer training fold. Curves,
  selected limits and group membership are recorded. Defaults remain the
  earlier fixed maximum of 300; no research fit has been extended merely by
  changing the code.

## Evidence and remaining work

The corrected batching exports are under
`outputs/final_run_readiness/batching_ablation_summary*`. They contain 25
selected arm runs, 50 descriptive rows and 96 paired metric comparisons from
the existing run group. Raw model outputs were not changed.

The additional `outputs/final_run_readiness/exploratory_batching_single_reference*`
exports compare those same historical arms against `single`: 25 runs, 50
descriptive rows and 102 comparison rows, six explicitly unavailable because
GLM-5.3-Flash has no single-item reference in that group. These are exploratory
size-1/16 results, not the new campaign and not evidence for size 4.

PURE source review is complete as **AI-assisted review**, authorized by the
author, not human or expert validation. The independent subagent reviewed all
180 proposals: 129 accepted, 51 revised. The main agent checked the revisions
against source/context and refined the equipment-design clause. The decisions
and limitations are in `docs/pure_capability_ai_review.md`; accepted clauses
are in `docs/pure_capability_revisions.csv` with explicit AI-review decisions.
No original marker, source text, selection or context field was changed.

The reviewed clauses were applied and all 720 PURE items rebuilt. The builder's
source-copy, unique-identity and existing-benchmark checks pass. Old benchmark
inputs, selected seeds, review exports and manifest are preserved under
`outputs/pure_before_capability_review/` with their original relative paths.
Raw model responses were not changed. Reproduce the application/build with:

```bash
.venv/bin/python scripts/build_pure_benchmark.py --stage apply-revisions
.venv/bin/python scripts/build_pure_benchmark.py --stage benchmark --overwrite
```

The earlier `outputs/pure_capability_revision_review*.csv` files remain
pre-review exports, not final approval records. Some source underspecification
remains (special indications, additional interfaces, national-value settings).
Generic system phrasing abstracts capabilities rather than preserving the
original allocation of every operator/design obligation. Repeat **both**
context arms with the corrected items and matching single-item settings.
Old PURE outputs are historical and cannot be scored against revised prompts.

Refresh a complete historical campaign without regenerating answers:

```bash
.venv/bin/python scripts/rerun_all.py --only analysis --refresh-analysis \
  --state outputs/rerun/modality-rerun-2026-09/headline_state.json --dry-run
```

Removing `--dry-run` runs local analysis and classifier fits, not generation.
It can still be computationally substantial. The completeness gate remains
active, including Task 3. If a historical state is incomplete or split across
campaigns, reconcile the exact run selection first; do not fabricate completion
entries or bypass configuration checks. For generation/resume, live settings
must still match the stored configuration.

After freezing the scoring rules, refresh per-run scores, semantic artifacts,
classifier inputs/fits, figures, paired summaries and paper numbers. Determine
newly eligible Task 3 items and unresolved failures; only those missing
judgments need new calls. Do not treat the current older derived artifacts as
already refreshed by these code changes.

For the classifier follow-up, use a capped inner-validation budget such as
`--hgb-budgets 300 600 1200`. Prefer the requirement-only, capability-held-out
primary diagnostic first. Inspect training/validation loss, upper-limit hits,
class counts and AP baselines before expanding to the remaining conditions.
If validation performance has not stabilized, retain a fixed-budget limitation.
Do not select budgets from outer test AUROC. No such research follow-up has
been run as part of the initial implementation.

## Decisions before final generation

1. **Revised author decision on 2026-09-11:** use single-item requests as
   primary, with five stochastic samples plus one deterministic answer.
   This supersedes the earlier grouped-primary choice. The planned batching
   sweep uses sizes 1, 4 and 16; the two larger sizes each have grouped and
   sibling-separated arms. Its single-item arm is a fresh deterministic
   repeat, so timing/repeated-generation variation still limits causal claims.
   The ablation tests deterministic Task 2, not all tasks or stochastic UQ.
2. Freeze profiles and server settings. Current profiles contain nine models,
   local concurrency 8 and 1024 output tokens per item; historical headline
   runs used concurrency 2 and Qwen3.6 used 512 tokens per item. Treat these as
   different execution settings, not identical reruns.
3. List each selected historical run as reusable, supplementary or replaced.
   A shared run-group name does not merge driver state. The expanded ablation
   configuration is an extension, not an all-model standalone campaign.
4. PURE's delegated source review and rebuild are complete. Confirm the
   planned workload and permission for new provider requests before launching.
   Keep prior outputs and state files; AI review must not be called human review.

`conf/rerun/final.yaml` is a standalone fresh-run plan with the agreed primary
protocol, eight batching/context models and three weak-phrasing models. It
uses new main/context groups and deliberately does not merge exploratory
states. With the current profiles it schedules all 36 main runs (311,040
Task 1/2 item answers and 311,040 single-item requests before audits, ablations,
probes and retries). Local concurrency remains 8 and hosted concurrency 2;
GPU parallelism does not establish hosted throughput or cost. This is a
proposed fresh-run scope, **not** authorization or a requirement to discard
reusable pilot results. Any reuse alternative must explicitly select compatible
run IDs and record historical settings before the final workload is approved.

Inspect the complete fresh-run plan without provider calls:

```bash
.venv/bin/python scripts/rerun_all.py --config conf/rerun/final.yaml --dry-run
```

The final-plan preflight checks single-item profiles and five stochastic
samples. PURE problems are printed as planning warnings in a dry-run and reject
real execution before provider calls. Historical analysis refresh reads the
old recorded profiles, including batch size 16; it does not relabel those runs.

## Manuscript and submission gates

- Reconcile coverage, failed responses, unreadable wording, incomplete samples
  and unavailable audits with every table's denominator.
- Emphasize per-model effects. Pooled meaning-variation ranking and blind-check
  recall can conceal strongly different within-model behavior. Confidence in
  a modality label is not a direct confidence judgment about wording fidelity.
- Report context results by original marker and transformed modality. Adding
  context plus preservation instructions is not a test of latent stakeholder
  intent. Weak-phrasing comparisons use their own matched probe baseline.
- Repeat author output review across flagged, unflagged, mixed and unreadable
  cases. State that weak intent is operational, not a universal modal ordering.
- Complete refreshed figures, result prose, embedding rationale, verbosity
  analysis, reviewer responses and exact execution metadata. Fine-tuning and
  naturalistic validation remain future work unless explicitly added.
- Compile and visually check the final short manuscript and its word budget.
  Verify the replication package locally. Publishing requires separate approval.

## Implementation verification

Local regression tests, lint and whitespace checks passed. The full final-plan
dry-run selected 128 Task 1/2, audit, batching and context generation commands
without writing state or calling providers. Its paper export requires size 1;
the historical headline refresh still requires its recorded size 16 and plans
no generation. The independent code review found no further concrete defects
in the changed protocol, pairing, export or PURE-application paths.

The draft manuscript compiles and the edited methods pages were visually
checked. It still contains pending-result notes and is not a final submission.
No live smoke, new provider generation, research classifier budget follow-up,
or complete historical analysis refresh was run in this implementation pass.

## Subsequent live smoke: 11 September 2026

The subsequently authorized GPU smoke is documented in
[`live_smoke_2026-09-11.md`](live_smoke_2026-09-11.md). All seven local models
completed single-item Task 1/2 sampling, eligible blind audits, grouped and
sibling-separated size-4/16 checks, and rebuilt PURE context arms. All 1,344
primary responses parsed, with five recovered GPT-OSS native-format HTTP 500
errors. Ten stochastic Qwen audit responses were invalid JSON. One Muse and
three GPT-OSS source outputs were excluded from auditing because their text
modality was unknown. These failures and exclusions remain visible in the
saved artifacts; they were not repaired away.

This is a qualified local operational pass, not an unconditional launch gate.
Resolve or explicitly accept the server-format and audit-format reliability
issues and retain the conditional audit denominators before freezing the
campaign. Hosted models, other generation cells, full statistical analysis,
classifier convergence and final manuscript artifacts remain outside this
smoke's verification scope. No server/profile changes or final campaign were
performed.

## Follow-up generation qualification

The later [generation qualification](generation_qualification_2026-09-11.md)
supersedes the operational caveats immediately above for new runs. It diagnosed
GPT-OSS's optional Harmony JSON-format header rejection, enabled the existing
JSON-schema output contract in the local profile, retained native failures as
terminal observations, and expanded blind audit coverage to reference-free
extractions without inventing correctness labels. Historical outputs and the
original smoke report remain unchanged.

This is still not a full-campaign launch authorization or a final manuscript.
Hosted qualification needs explicit approval to send the selected benchmark
prompts to Z.AI; the statistical, author-review, manuscript and publication gates
above remain in force. See the qualification report for the exact local checks
and their limits.
