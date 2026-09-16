# Document-Context Ablation (`pure` cell, `item_context: bare|document`)

A minimal two-arm ablation for the reviewer's question whether *contextual*
cues around a requirement (heading, the author's own priority marker,
neighbouring requirements) change how a model treats the *lexical* cue inside
it (the modal verb or the weak phrase). It is the small, runnable version of
[`TODO.md`](../TODO.md) section B. Its numbers are reported on their own and
are **never pooled** into the four headline cells of
[`docs/aggregation.md`](aggregation.md).

## 1. Question

Every headline item in this artifact is one bare synthetic sentence. In
practice a requirement is read inside a document, under a heading, next to its
neighbours, and often with an explicit priority marker. Does showing that
context change (a) the declared modality label and (b) the modal force of the
generated requirement text, compared with the same sentence shown bare?

## 2. Data: the `pure` cell

**Corpus.** PURE (Ferrari, Spagnolo & Gnesi, RE 2017;
[Zenodo 7118517](https://zenodo.org/records/7118517), CC BY 4.0), the 19
documents in the common XML format. Only two of them attach an
author-assigned commitment marker to every requirement, and those two are
used:

| Document | Marker encoding | Requirements | M | O | I / none |
| --- | --- | --- | --- | --- | --- |
| EIRENE Functional Requirements Specification, version 7 (`2007-eirene_fun_7-2.xml`) | inline `(M)`, `(O)`, `(I)` at the end of the text body; the body starts with the requirement id | 583 | 378 | 82 | 102 / 21 |
| ERTMS/ETCS Functional Requirements Specification, version 5.0 (`2007-ertms.xml`) | `<modifier>M|O</modifier>` child of each `<req>` | 199 | 196 | 3 | 0 |

Both documents define the legend in their introduction: `(M)` mandatory,
`(O)` optional, and for EIRENE `(I)` informative. The marker is therefore a
**document-level commitment cue that is independent of the modal verb**; in
the natural text it correlates with the verb (O goes with "should"), which is
exactly why the verb is manipulated by the minimal-pair templates below.

**Seeds.** `scripts/build_pure_benchmark.py --stage candidates` parses both
documents (`eu.load_pure_requirement_rows`), keeps for each requirement its
id, marker, section title path, and its previous and next requirement in
document order, screens the sentence with the same filter the `mlm_tapt`
seeds went through plus marker checks (`eu.pure_filter`: informative
marker, no marker, one marker per list item, impersonal "it shall be possible"
constructions), and then **includes every eligible optional-marked
requirement** and fills to 180 with mandatory-marked ones sampled
deterministically per document (`eu.make_pure_seed_candidates`, seed
20260518). The optional stratum is small by nature (25 of 180 seeds); the
table says so rather than hiding it.

| Selected seeds | EIRENE | ERTMS | total |
| --- | --- | --- | --- |
| marker M | 88 | 67 | 155 |
| marker O | 24 | 1 | 25 |

The seed review table is `data/processed/seeds_review_pure.csv`
(`include`, `capability_text_final` are the reviewer's columns, as for the
other datasets). On 2026-09-11 the author delegated a renewed AI-assisted
source review after defects in the old extraction were found. All 180
capabilities were reviewed and corrected, then 720 items regenerated. This is
not renewed human validation. See [row-level decisions](internal/pure_capability_ai_review.md)
and [validation review](validation_review.md). Earlier benchmark inputs are
archived under `outputs/pure_before_capability_review/`.

**Items.** `--stage benchmark` renders the 180 seeds through the unchanged
four modality templates (`eu.source_statement`: MUST / SHOULD / MAY / "It
would be useful if the system could ...") into
`data/processed/benchmark_items_pure.csv` (720 items, MUST cell only). Every
item additionally carries its `context_*` columns: `context_document`,
`context_requirement_id`, `context_marker`, `context_section`,
`context_before`, `context_after`, `context_legend`. The `source_statement`
column stays the bare sentence, so the text detector, the length metrics and
every existing scorer read exactly what they read on the other cells. The
manifest `outputs/benchmark_manifest_pure.json` hashes the seed tables, the
benchmark and both Task 2 prompt files.

The `mlm_tapt` seeds excluded every `_PURE`-tagged source, so the `pure`
seeds are disjoint from the reported benchmark.

## 3. Arms

One run-level knob, `item_context`, recorded on every raw row, in the run
registry column of the same name, in the dry-run log line and in the
resolved-config digest:

| Arm | What the model sees | Prompt |
| --- | --- | --- |
| `bare` | one Task 2 source statement per request, using the final primary delivery protocol | `prompts/modality_extraction.txt` |
| `document` | the same single item plus document, section path, original marker and neighboring requirements, with a context/preservation instruction | `prompts/modality_extraction_context.txt` |

Everything else is held fixed: dataset, items, batch size 1, `batch_order:
grouped`, deterministic sampling, request seed. Marker M vs O is a reported
stratum, not a manipulated factor; the context always shows the author's
real marker. The knob is refused outside `task=task2`, and the context prompt
is only loaded for the `document` arm, so bare runs keep exactly the two
frozen prompt inputs of the paper.

The old exploratory arms used size 16 and earlier capability clauses. They
are not interchangeable with the corrected single-item context experiment.
The ablation changes both document information and its accompanying
instruction, so it does not isolate a pure information-only effect.

### 3.1 Archived batched prompt, Task 2, `document` arm (two real items)

```text
Extract exactly one requirement from each source statement.
Preserve the modality of each source. Evaluate each item independently.
Each item's context shows where its source statement appears: the document, section, the author's requirement marker, and the neighbouring requirements. Extract from the source statement only.

Use one of: "mandatory", "recommended", "optional", "nice_to_have".
Use confidence as a decimal from 0.0 to 1.0 for confidence in the selected modality.
Do not return percentages such as 95 or strings such as "95%".
Return JSON only as this object:
{"results":[{"request_index":0,"requirement":"...","modality":"mandatory","confidence":0.95}]}

Items:
[
  {
    "request_index": 0,
    "source_statement": "The system MUST support the receipt of mobile-originated text messages by the ground.",
    "context": "Document: EIRENE Functional Requirements Specification, version 7 (markers: (M) mandatory, (O) optional, (I) informative)\nSection: 2 Network requirements > 2.3 Data services\nPreceding requirement 2.3.2 (O): The network should support the transmission of point-to-point and point-to- multipoint text messages from the ground to mobile users.\nThis requirement: 2.3.3, marker (O)\nFollowing requirement 2.3.4 (M): If the text message facility is implemented, it shall not interfere with the ability of users to make or receive high priority voice or data calls."
  },
  {
    "request_index": 1,
    "source_statement": "It would be useful if the system could support the receipt of mobile-originated text messages by the ground.",
    "context": "Document: EIRENE Functional Requirements Specification, version 7 (markers: (M) mandatory, (O) optional, (I) informative)\nSection: 2 Network requirements > 2.3 Data services\nPreceding requirement 2.3.2 (O): The network should support the transmission of point-to-point and point-to- multipoint text messages from the ground to mobile users.\nThis requirement: 2.3.3, marker (O)\nFollowing requirement 2.3.4 (M): If the text message facility is implemented, it shall not interfere with the ability of users to make or receive high priority voice or data calls."
  }
]
```

The `context` string is rendered by one helper, `eu.document_context_text`,
for both the batched wrapper and the single-item template, so the two cannot
drift. The per-item fallback after an unparsable batch re-sends the
context-rendered single-item prompt, so the `document` arm never silently
degrades to bare. The first line of the bare wrapper is unchanged; the only
differences are the third instruction line and the `context` values.

## 4. Running it

```bash
# both arms, hosted profile (720 requests per arm per model)
.venv/bin/python scripts/run.py --multirun +experiment=context_ablation
# both arms, one local model
.venv/bin/python scripts/run.py --multirun +experiment=context_ablation \
  profile=local_llama_cpp model=qwen3.5-9b
# offline dry run of the wiring
.venv/bin/python scripts/run.py --multirun +experiment=context_ablation \
  mode=smoke fake_completion=true smoke_items=4
# the table
.venv/bin/python scripts/compare_context_ablation.py            # real runs
.venv/bin/python scripts/compare_context_ablation.py --include-smoke
```

The preset pins dataset `pure`, variant `must`, Task 2, deterministic
sampling, batch size 16, grouped batching, and its own run group
`context-ablation-2026-09`, so the arms can never be selected into the paper
tables (`scripts/export_paper_tables.py` gates on the run group). Each
Hydra run writes its resolved config next to the logs and its digest into the
registry `notes` column.

## 5. The table

Results for the eight models of the reported campaign are in
`outputs/context_ablation_summary.md` (deltas with intervals in
`outputs/context_ablation_summary_deltas.csv`).

`scripts/compare_context_ablation.py` selects the latest complete, fully covered
MUST run per `(model, item_context)` in the PURE context run group. The rerun
driver also restricts selection to its recorded run IDs. It scores deterministic
Task 2 observations only; comparison does not compute stochastic embeddings.

The original document/corpus and requirement identifier establish the stable
capability identity. The current benchmark resolves raw sources with different
arm-local IDs by their exact source statement and modality; ambiguous or stale
source mappings fail explicitly. Pairing keys include model, dataset, keyword
variant, original requirement identity, and transformed modality. Raw retry
attempts are first deduplicated by the existing completion-key rule. Multiple
remaining observations with one pairing identity are excluded from both arms,
never resolved by choosing an arbitrary row.

Eligibility is metric-specific: label accuracy requires a valid parsed response
with a recognized declared modality in both arms. Strict and broad wording rates
also require classifiable text in both arms. Thus different surviving conditions
of one capability cannot form a pair. Each metric's `bare`, `document`, `delta`,
and intervals use exactly that matched cohort. Empty cohorts produce blank
estimates and an explicit reason.

Outputs:

- `context_ablation_summary.csv` / `.md`: **full-arm descriptive** estimates,
  explicitly stamped `estimate_cohort=full_arm_descriptive`. `n` counts observed
  deterministic answers, including failed answers; `n_failed_items` counts
  failed answers and `n_unclassified_items` counts valid answers whose wording
  is unclassified. `n_text_readable` is the wording denominator.
- `context_ablation_summary_deltas.csv`: matched estimates and differences;
  `full_arm_bare_descriptive` / `full_arm_document_descriptive` are separate
  descriptions of all eligible arm observations. `n_bare` / `n_document` count
  observed answers. `n_matched_items` counts exact source-condition pairs;
  `n_matched_capabilities` counts distinct original requirements represented.
  `n_unmatched_items` counts identities observed in only one arm;
  `n_excluded_ineligible_items` counts unique identities present in both arms
  but failing either arm's metric eligibility. Duplicate identities and their
  observation-row counts, missing-identity rows, per-arm failed/unclassified
  answers and eligible answers are separate columns. Legacy `n_complete_pairs`
  and `n_excluded_single_arm` now explicitly count **items**, not capabilities.
  Duplicate, unmatched, and ineligible identity exclusions are disjoint.
- `context_ablation_summary_provenance.json`: selected runs and their registry
  metadata, including resolved configuration references in `notes`.

Bootstrap draws always retain both members of each exact pair. The primary
request clusters are connected components of the request memberships in **both**
arms. Identical partitions reduce to ordinary paired request resampling; crossing
partitions merge requests until every dependent pair is kept together. They are
not assumed to match merely because nominal batch sizes match. If any matched
answer lacks a request ID, the entire primary interval falls back to capability
clustering, with the reason recorded in `cluster_note`. The companion
`delta_seed_ci_*` always resamples capabilities. `delta_cluster_field` and
`n_delta_clusters` identify the actual primary resampling unit and count.
Intervals with fewer than two clusters are unavailable. These percentile
intervals describe the observed matched cohort, not failures excluded from it.

Breakdowns include `all`, `weak_intent`, each transformed modality, each original
marker (`M`, `O`), and marker × transformed-modality intersections, separately
for every model. Context results remain separate from the headline benchmark.

Synthetic regression fixtures verify these contracts; existing experimental
outputs are stale and were not regenerated. Rerun measurements and interpretation
remain pending.

## 6. Reading it

- The **primary contrast** is the weak-intent delta on strict text
  strengthening: is the same weak wish strengthened less often when the
  model can see it sits under an `(O)` marker and next to "should"
  neighbours? Report the delta with its CI; if the CI covers zero, say so.
- The `marker_O` stratum answers the reviewer's literal question (context
  says optional, sentence says MUST): compare its label accuracy and
  strengthening across arms. With 25 seeds (100 items) its CIs are wide by
  construction.
- The `marker_M` stratum is the control: for a mandatory-marked requirement
  the context agrees with the strongest template and disagrees with the
  three weaker ones.
- `label_accuracy` deltas show whether the declared label follows the
  context; the text-strengthening deltas show whether the generated
  requirement does. The two can move in opposite directions.

## 7. What this ablation does not show

- The marker is never manipulated. A marker-flipped third arm (same context,
  M ↔ O swapped) would isolate the marker from the heading and neighbours;
  it is left in [`TODO.md`](../TODO.md) section B.
- The other envelope factors of the TODO sketch (document status,
  stakeholder role, priority field, rationale sentence, elicitation
  transcript) are not modelled.
- One domain (railway signalling and radio), two documents, one variant
  (MUST), Task 2 only, deterministic pass only.
- The 180 corrected capability texts have an author-delegated AI-assisted
  review, not renewed human/expert validation. Generic system phrasing is a
  capability abstraction; original operator/design responsibilities are not
  an additional evaluated construct. Stakeholder intent remains unmeasured.
- Naturally occurring stakeholder statements (for example Apache Jira "Wish"
  issues with a declared priority) are a separate track; candidates are in
  [`docs/external_validity_datasets.md`](internal/external_validity_datasets.md).

## 8. Provenance and what did not change

- Adding the knob changed no existing fingerprint: `item_context` enters the
  job-config SHA only when it is not `bare`, and the bare batch wrapper is
  byte-identical (both pinned by `tests/test_eval_utils.py`). Archived runs
  therefore resume without re-requests.
- `completion_batch_key` was deliberately left untouched, because that tuple
  seeds the shuffled-arm RNG of the batching ablation; two arms of this
  ablation can never share a batch anyway because `run_id` is already in
  the key.
- The new `item_context` key in `conf/config.yaml` changes the
  `resolved_config_sha` of every Hydra run from now on; that is provenance,
  not behaviour.
- `docs/benchmark_ground_truth.md`, the four headline benchmark CSVs and
  their manifests are untouched.
