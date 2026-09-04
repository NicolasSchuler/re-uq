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
other datasets); the capability texts were taken from the automatic
extraction and have not yet had a separate human pass, which the ablation
write-up must state.

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
| `bare` | today's Task 2 request, byte-identical to the paper condition (the batched wrapper's SHA and the job-config fingerprint are pinned in the tests) | `prompts/modality_extraction.txt`; batched wrapper of `docs/experimental_setup.md` §3.1 |
| `document` | the same items in the same 16-item grouped batches, each with one extra `context` value (document, section path, the author's real marker, preceding and following requirement) and one neutral instruction sentence | `prompts/modality_extraction_context.txt`; batched wrapper below |

Everything else is held fixed: dataset, items, batch size 16, `batch_order:
grouped`, deterministic sampling, request seed. Marker M vs O is a reported
stratum, not a manipulated factor; the context always shows the author's
real marker. The knob is refused outside `task=task2`, and the context prompt
is only loaded for the `document` arm, so bare runs keep exactly the two
frozen prompt inputs of the paper.

### 3.1 Batched prompt, Task 2, `document` arm (verbatim, two real items)

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
# both arms, GLM half of the cohort (45 requests per arm per model)
.venv/bin/python scripts/run.py --multirun +experiment=context_ablation
# both arms, non-GLM half
.venv/bin/python scripts/run.py --multirun +experiment=context_ablation \
  profile=kit_toolbox model=kit.gemma4-31b-it
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

`scripts/compare_context_ablation.py` selects the latest complete, fully
covered run per `(model, item_context)` (a blank registry column reads as
`bare`, exactly like `batch_order`), scores the deterministic Task 2 rows
with the same calls as `scripts/compare_run_matrix.py`
(`eu.benchmark_rows_with_current_raw_outputs` → `eu.build_uq_scores`), and
joins the author marker back on `item_id`. It writes:

- `outputs/context_ablation_summary.csv` / `.md`: one row per model × arm ×
  stratum with `n`, the number of rows whose generated text yielded a
  modality, declared-label accuracy, strict and broad text strengthening
  with request-clustered bootstrap CIs (`eu.text_over_commitment_ci_fields`,
  [`aggregation.md`](aggregation.md) §6) and the seed-clustered pair alongside
  as `*_seed_ci_low` / `*_seed_ci_high`,
  and in the weak-intent stratum the README-style p ≥ 0.90 strict rate.
- `outputs/context_ablation_summary_deltas.csv` and the second table of the
  Markdown: `document − bare` per model × stratum × metric with a **paired**
  cluster bootstrap CI (`eu.bootstrap_seed_metric_delta`): each iteration
  draws one resample and evaluates both arms on it before differencing.
  Resampling the arms independently would treat paired observations as
  unrelated and overstate the interval. Pairing and resampling are different
  units: rows are **paired by seed**, because that is the same capability seen
  bare and in its document, while the **resampling unit is the request**. A
  seed's four source conditions sit inside one request, so a request contains
  whole pairs and is only the coarser unit. The two arms are separate runs and
  their request ids differ, but the partition of seeds into requests is a
  property of the batching and is the same in both, so each paired seed is
  assigned to the request observed for it in the bare arm.
  `delta_ci_low` / `delta_ci_high` are request-clustered,
  `delta_seed_ci_low` / `delta_seed_ci_high` the seed-clustered pair, and
  `delta_cluster_field` / `n_delta_clusters` record the unit used. The cohort is
  the seeds *both* arms answered, built once before any resampling: a seed only
  one arm answered is excluded up front and counted, so the number of pairs
  cannot vary from replicate to replicate. `n_complete_pairs` and
  `n_excluded_single_arm` report that cohort on every delta row.
- `outputs/context_ablation_summary_provenance.json`: the run ids, start
  times, batching settings and `resolved_config_sha` behind every row.

Strata: `all`; `weak_intent` (source modality `nice_to_have`, the paper's
headline condition); `marker_M` and `marker_O` (the author's marker on the
seed requirement, 155 and 25 seeds).

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
- The capability texts of the 180 seeds come from the automatic extraction
  without a separate human pass yet, the same caveat the weak-intent
  construct review carries.
- Naturally occurring stakeholder statements (for example Apache Jira "Wish"
  issues with a declared priority) are a separate track; candidates are in
  [`docs/external_validity_datasets.md`](external_validity_datasets.md).

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
