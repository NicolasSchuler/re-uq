# TODO

Open work on this artifact, ordered by how much it affects the paper's claims.
Setup facts referenced here are documented in
[`docs/experimental_setup.md`](docs/experimental_setup.md).

## A. Batching impact assessment

**Why.** Every reported run sent 16 benchmark items per request, in consecutive
`request_index` order, over rows ordered seed × variant. Every Task 2 batch
therefore contained all four modality variants of the same four seeds side by
side. The prompt says "evaluate each item independently", but the minimal-pair
contrast was inside the context window. No number in the paper is currently
free of this confound.

**Do.**

1. Run the `batch_order: shuffled` ablation (per-profile knob) on the same
   items, same seeds, same parameters.
2. Run a `batch_size: 1` ablation (true single-item delivery, the frozen
   `prompts/*.txt` wording).
3. Minimum cohort: `glm-5.1` and `kit.gemma4-31b-it`, dataset `mlm_tapt`,
   variant `must`, Task 2, deterministic pass.
4. Report deltas against the grouped baseline for: declared-label accuracy,
   strict text strengthening, broad text strengthening — overall and within the
   weak-intent condition.

**Report as.** One table, three rows (grouped / shuffled / size-1), with
seed-level bootstrap CIs on each delta. If the deltas are within CI, say so and
keep the grouped runs as the headline. If they are not, the grouped numbers
become an upper or lower bound and the paper must say which.

**Prerequisites — done.**

- `batch_order: shuffled` is a *constrained* shuffle: it never places two source
  variants of one seed in the same batch, and the permutation is derived
  deterministically from the recorded run seed. The first implementation only
  permuted job order and, at batch size 16, still left two variants of one seed
  together in roughly half of the 45 Task 2 batches (22 to 24 depending on the derived RNG seed) — it would not have removed the confound
  it was meant to ablate.
- Resume no longer re-shuffles the pending subset, so a resumed shuffled arm
  keeps the batch membership of the original run.
- `+experiment=batching_ablation` pins `batch_size: 16` for both batch-order
  arms, so grouped and shuffled differ only in batch membership, not in request
  size.
- The per-profile `seed` knob exists.

**Still open.** The three ablation runs themselves and the comparison table.

## B. Contextual requirements extension

**Why.** The current benchmark is a single synthetic sentence per item. That is
what buys the causal control, and it is also the largest external-validity gap.
Requirements in practice are read inside a document, with a heading, a status,
an owner, and neighbours.

**Done (minimal two-arm ablation).** See
[`docs/context_ablation.md`](docs/context_ablation.md).

- Dataset `pure`: 180 seeds from the two PURE documents that attach an
  author-assigned mandatory/optional marker to every requirement (EIRENE FRS 7,
  ERTMS FRS 5.0; CC BY 4.0). Every eligible optional-marked requirement (25) is
  included, the rest is mandatory-marked. Items keep the minimal-pair spine and
  carry the document, section path, marker and neighbouring requirements as
  separate columns (`scripts/build_pure_benchmark.py`).
- Knob `item_context: bare|document` (run-level, mirrors `batch_order` end to
  end: run config, job fingerprint, raw row, registry column). The `document`
  arm shows each Task 2 item with its real context; the `bare` arm is
  byte-identical to the paper condition (pinned).
- Preset `+experiment=context_ablation` (both arms, `pure`/`must`, Task 2,
  deterministic, grouped 16-item batches, own run group) and
  `scripts/compare_context_ablation.py` (per model × arm × stratum table plus
  `document − bare` deltas with a paired seed-clustered bootstrap CI,
  `eu.bootstrap_seed_metric_delta`).

**Still open.**

1. The runs themselves (`glm-5.1`, `kit.gemma4-31b-it`) and the table; then
   the write-up in `docs/context_ablation.md` and the manuscript's robustness
   paragraph.
2. A human pass over `capability_text_final` for the 180 `pure` seeds (the
   automatic extraction is used as is).
3. A **marker-flipped** third arm (same context, M ↔ O swapped) to isolate the
   marker from heading and neighbours.
4. The remaining envelope factors of the original sketch, as a fractional
   design with pre-registered cells:

   | Context factor | Levels (sketch) |
   | --- | --- |
   | Document status | none / `Draft` / `Approved` |
   | Stakeholder role | none / product owner / end user / support engineer |
   | Priority field | none / `High` / `Low` |
   | Rationale sentence | absent / present |
   | Neighbouring requirements | none / two same-strength / two mixed-strength |
   | Elicitation transcript | absent / short interview excerpt containing the wish |

5. Second track: **naturally occurring stakeholder statements** (issue
   trackers, interview transcripts, feature requests) with annotated intended
   strength, reported separately from the controlled benchmark. Best first
   candidate: the Public Jira Dataset's Apache "Wish" issues, whose declared
   type and priority are contextual cues orthogonal to the phrasing; survey in
   [`docs/external_validity_datasets.md`](docs/external_validity_datasets.md).

## C. More diverse model families

**Why.** Five of the six official models are GLM. One outside model is not a
cross-family claim.

**Do.**

1. Run the new example provider profiles: `openai`, `mistral`,
   `google_gemini`, `ollama_local`. Scope rule: OpenAI-compatible
   chat-completions endpoints only — a provider is in scope as soon as it
   exposes one; providers without such an endpoint are out of scope.
2. Same cells as the official cohort (`nice` and `mlm_tapt`, `must`; `shall` if
   budget allows).
3. Export per-model tables with `scripts/export_paper_tables.py` →
   `outputs/paper_per_model_modality_table.csv` and
   `outputs/paper_per_model_headline.csv`.
4. State which models are hosted, which are local, and which had JSON mode.

**Note.** `ollama_local` also gives a fully offline replication path for
reviewers without paid API access.

## D. Human confirmation of the construct review

`docs/weak_modality_construct_review.csv` is complete and the analysis gate
passes, but both reviewer slots (`R1`, `R2`) hold an author-delegated
LLM-assisted review, declared in the `reviewer_role` column.

**Do.** Two human reviewers independently re-judge the four weak templates
against `SHOULD/recommended`, record agreement, and replace the `reviewer_role`
values. Until then, weak-intent claims carry a construct-validity caveat and
`outputs/README.md`, `docs/results_mapping.md`, and
`docs/experimental_setup.md` must keep saying so.

## E. Regenerate paper snapshots

**Do.** Regenerate every paper-facing table with
`scripts/export_paper_tables.py`, including per-model breakdowns and seed-level
bootstrap confidence intervals on all headline rates. Retire the ad-hoc
root-level `paper_*` snapshots that predate the exporter, and record the
generating run ids in the provenance JSON next to each table.

**Check.** Headline numbers in `README.md` and the manuscript must be re-read
off the regenerated tables, not carried over. Aggregation scope (pooled vs
macro-of-cells) must match [`docs/aggregation.md`](docs/aggregation.md).

**Also.** Prune the `complete` rows dated 2026-05-21 from the local run
registries: their raw rows were removed later, so the registry claims runs that
no longer have data. The exporter now refuses a selected run with no raw rows,
but the stale rows still show up in run selection and in the run-matrix
comparison.

## F. Rerun with seed sent and served model recorded

**Why.** The reported runs sent no `seed` and recorded only the requested model
string. A hosted `glm-5.1` in May and a hosted `glm-5.1` today are not
guaranteed to be the same weights, and nothing in the raw records would show
the difference.

**Do.** Rerun at least the headline cell per model with `send_seed` enabled and
a fixed per-profile `seed`, and with `served_model` / `system_fingerprint`
captured per response. Compare the deterministic pass against the archived run
item-by-item and report the disagreement rate. Treat the rerun as a new run;
it verifies stability, not the old numbers.

## G. Out of scope: fine-tuning

We do not fine-tune, and this is **not** a to-do item. The study measures
off-the-shelf behaviour of hosted instruction-tuned models under a frozen
prompt contract. Whether supervised fine-tuning or preference tuning removes
modal-force strengthening is a real question and a different paper. It is
recorded as limitation 8 in
[`docs/experimental_setup.md`](docs/experimental_setup.md#11-limitations), not
as planned work.
