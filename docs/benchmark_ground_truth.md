# Benchmark Ground Truth

This document derives every benchmark item step by step. It is *generated* by
`scripts/export_benchmark_ground_truth.py` from the same template code that
builds `data/processed/benchmark_items*.csv`, so what you read here is what the
models received — not a hand-written paraphrase of it. Regenerate with:

```bash
.venv/bin/python scripts/export_benchmark_ground_truth.py
```

Each cell of the design (dataset x mandatory keyword) holds 180 reviewed seeds
x 4 modality conditions = 720 items; there are four cells (`nice`/`mlm_tapt`
datasets x `MUST`/`SHALL` keywords).

## Dataset `nice` (NICE/PROMISE seeds, `MUST` + `SHALL`)

Selected seeds: **180** (screened from 622 corpus rows in `seeds_review.csv`; the same review table records the inclusion/exclusion decision for every candidate seed). Items per `MUST` cell: **720** (4 conditions per seed).

### Step 1 — Source requirement and capability (S0001)

The seed row in `data/processed/seeds_selected*.csv` records the original corpus requirement and the reviewed capability clause:

| Field | Value |
| --- | --- |
| `seed_id` | `S0001` |
| `source_dataset` | `NICE` |
| `original_requirement` | `The system shall refresh the display every 60 seconds.` |
| `capability_text_final` | `refresh the display every 60 seconds` |

The capability clause is the *content* that must survive every transformation unchanged; only its modal force varies.

### Step 2 — The four controlled variants

Every seed is rendered through the four fixed templates below (verbatim from `eval_utils.source_statement`); nothing else in the sentence changes:

| template_id                    | condition    | variant    | source_statement_template                                                                   | example_source_statement                                                                                            | intended_gold_modality   | note                                                                     |
|:-------------------------------|:-------------|:-----------|:--------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|:-------------------------|:-------------------------------------------------------------------------|
| main_mandatory_must            | mandatory    | must       | The system MUST {capability}.                                                               | The system MUST refresh the display every 60 seconds.                                                               | mandatory                | Main benchmark mandatory condition.                                      |
| main_recommended_should        | recommended  | must       | The system SHOULD {capability}.                                                             | The system SHOULD refresh the display every 60 seconds.                                                             | recommended              | Main benchmark recommended condition.                                    |
| main_optional_may              | optional     | must       | The system MAY {capability}.                                                                | The system MAY refresh the display every 60 seconds.                                                                | optional                 | Main benchmark optional condition.                                       |
| main_nice_to_have_useful_if    | nice_to_have | must       | It would be useful if the system could {capability}.                                        | It would be useful if the system could refresh the display every 60 seconds.                                        | nice_to_have             | Main benchmark weak stakeholder-intent condition.                        |
| shall_mandatory_shall          | mandatory    | shall      | The system SHALL {capability}.                                                              | The system SHALL refresh the display every 60 seconds.                                                              | mandatory                | SHALL robustness variant; swaps MUST in the mandatory condition only.    |
| probe_useful_if                | nice_to_have | weak_probe | It would be useful if the system could {capability}.                                        | It would be useful if the system could refresh the display every 60 seconds.                                        | nice_to_have             | Weak-intent phrasing probe; identical to the main nice_to_have template. |
| probe_nice_if                  | nice_to_have | weak_probe | It would be nice if the system could {capability}.                                          | It would be nice if the system could refresh the display every 60 seconds.                                          | nice_to_have             | Weak-intent phrasing probe.                                              |
| probe_low_priority_enhancement | nice_to_have | weak_probe | As a low-priority enhancement, the system could {capability}.                               | As a low-priority enhancement, the system could refresh the display every 60 seconds.                               | nice_to_have             | Weak-intent phrasing probe.                                              |
| probe_future_enhancement       | nice_to_have | weak_probe | Stakeholders mentioned that the system could {capability} as a possible future enhancement. | Stakeholders mentioned that the system could refresh the display every 60 seconds as a possible future enhancement. | nice_to_have             | Weak-intent phrasing probe.                                              |

### Step 3 — The resulting items and their gold labels

| `item_id` | Source statement (what the model sees) | Task 1 gold | Task 2 gold | Ordinal |
| --- | --- | --- | --- | --- |
| `S0001_nice_to_have` | `It would be useful if the system could refresh the display every 60 seconds.` | `no` (yes=0) | `nice_to_have` | 0 |
| `S0001_optional` | `The system MAY refresh the display every 60 seconds.` | `no` (yes=0) | `optional` | 1 |
| `S0001_recommended` | `The system SHOULD refresh the display every 60 seconds.` | `no` (yes=0) | `recommended` | 2 |
| `S0001_mandatory` | `The system MUST refresh the display every 60 seconds.` | `yes` (yes=1) | `mandatory` | 3 |
| `S0001_mandatory (SHALL cell)` | `The system SHALL refresh the display every 60 seconds.` | `yes` (yes=1) | `mandatory` | 3 |

Gold labels are **structural**, not judged per item: the gold modality is the condition the template encodes, Task 1's gold entailment decision is `yes` iff the condition is `mandatory`, and the ordinal strength is fixed per condition (`mandatory`=3, `recommended`=2, `optional`=1, `nice_to_have`=0). The `SHALL` row above is the robustness variant: identical to the mandatory condition except the keyword `MUST` is swapped for `SHALL`.

### Step 4 — Weak-intent phrasing probes (same seed)

The benchmark's weak condition uses the `useful_if` template. To check that results are not tied to that single surface form, the same capability is also rendered through three alternative weak phrasings (`eval_utils.WEAK_MODALITY_PROBE_TEMPLATES`); all keep the gold label `nice_to_have`:

| template_id                    | source statement                                                                                                    | gold modality   |
|:-------------------------------|:--------------------------------------------------------------------------------------------------------------------|:----------------|
| probe_useful_if                | It would be useful if the system could refresh the display every 60 seconds.                                        | nice_to_have    |
| probe_nice_if                  | It would be nice if the system could refresh the display every 60 seconds.                                          | nice_to_have    |
| probe_low_priority_enhancement | As a low-priority enhancement, the system could refresh the display every 60 seconds.                               | nice_to_have    |
| probe_future_enhancement       | Stakeholders mentioned that the system could refresh the display every 60 seconds as a possible future enhancement. | nice_to_have    |

## Dataset `mlm_tapt` (`limsc/mlm-tapt-requirements` seeds, `MUST` + `SHALL`)

Selected seeds: **180** (screened from 39139 corpus rows in `seeds_review_mlm_tapt.csv`; the same review table records the inclusion/exclusion decision for every candidate seed). Items per `MUST` cell: **720** (4 conditions per seed).

### Step 1 — Source requirement and capability (S0110)

The seed row in `data/processed/seeds_selected*.csv` records the original corpus requirement and the reviewed capability clause:

| Field | Value |
| --- | --- |
| `seed_id` | `S0110` |
| `source_dataset` | `mlm_tapt` |
| `original_requirement` | `More than 10 (TBC) of the planets observed in the Chemical Census tier shall be observed in the Rosetta Stone tier.` |
| `capability_text_final` | `observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier` |

The capability clause is the *content* that must survive every transformation unchanged; only its modal force varies.

### Step 2 — The four controlled variants

Every seed is rendered through the four fixed templates below (verbatim from `eval_utils.source_statement`); nothing else in the sentence changes:

| template_id                    | condition    | variant    | source_statement_template                                                                   | example_source_statement                                                                                                                                                          | intended_gold_modality   | note                                                                     |
|:-------------------------------|:-------------|:-----------|:--------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------|:-------------------------------------------------------------------------|
| main_mandatory_must            | mandatory    | must       | The system MUST {capability}.                                                               | The system MUST observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier.                                                               | mandatory                | Main benchmark mandatory condition.                                      |
| main_recommended_should        | recommended  | must       | The system SHOULD {capability}.                                                             | The system SHOULD observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier.                                                             | recommended              | Main benchmark recommended condition.                                    |
| main_optional_may              | optional     | must       | The system MAY {capability}.                                                                | The system MAY observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier.                                                                | optional                 | Main benchmark optional condition.                                       |
| main_nice_to_have_useful_if    | nice_to_have | must       | It would be useful if the system could {capability}.                                        | It would be useful if the system could observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier.                                        | nice_to_have             | Main benchmark weak stakeholder-intent condition.                        |
| shall_mandatory_shall          | mandatory    | shall      | The system SHALL {capability}.                                                              | The system SHALL observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier.                                                              | mandatory                | SHALL robustness variant; swaps MUST in the mandatory condition only.    |
| probe_useful_if                | nice_to_have | weak_probe | It would be useful if the system could {capability}.                                        | It would be useful if the system could observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier.                                        | nice_to_have             | Weak-intent phrasing probe; identical to the main nice_to_have template. |
| probe_nice_if                  | nice_to_have | weak_probe | It would be nice if the system could {capability}.                                          | It would be nice if the system could observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier.                                          | nice_to_have             | Weak-intent phrasing probe.                                              |
| probe_low_priority_enhancement | nice_to_have | weak_probe | As a low-priority enhancement, the system could {capability}.                               | As a low-priority enhancement, the system could observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier.                               | nice_to_have             | Weak-intent phrasing probe.                                              |
| probe_future_enhancement       | nice_to_have | weak_probe | Stakeholders mentioned that the system could {capability} as a possible future enhancement. | Stakeholders mentioned that the system could observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier as a possible future enhancement. | nice_to_have             | Weak-intent phrasing probe.                                              |

### Step 3 — The resulting items and their gold labels

| `item_id` | Source statement (what the model sees) | Task 1 gold | Task 2 gold | Ordinal |
| --- | --- | --- | --- | --- |
| `S0110_nice_to_have` | `It would be useful if the system could observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier.` | `no` (yes=0) | `nice_to_have` | 0 |
| `S0110_optional` | `The system MAY observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier.` | `no` (yes=0) | `optional` | 1 |
| `S0110_recommended` | `The system SHOULD observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier.` | `no` (yes=0) | `recommended` | 2 |
| `S0110_mandatory` | `The system MUST observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier.` | `yes` (yes=1) | `mandatory` | 3 |
| `S0110_mandatory (SHALL cell)` | `The system SHALL observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier.` | `yes` (yes=1) | `mandatory` | 3 |

Gold labels are **structural**, not judged per item: the gold modality is the condition the template encodes, Task 1's gold entailment decision is `yes` iff the condition is `mandatory`, and the ordinal strength is fixed per condition (`mandatory`=3, `recommended`=2, `optional`=1, `nice_to_have`=0). The `SHALL` row above is the robustness variant: identical to the mandatory condition except the keyword `MUST` is swapped for `SHALL`.

### Step 4 — Weak-intent phrasing probes (same seed)

The benchmark's weak condition uses the `useful_if` template. To check that results are not tied to that single surface form, the same capability is also rendered through three alternative weak phrasings (`eval_utils.WEAK_MODALITY_PROBE_TEMPLATES`); all keep the gold label `nice_to_have`:

| template_id                    | source statement                                                                                                                                                                  | gold modality   |
|:-------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------|
| probe_useful_if                | It would be useful if the system could observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier.                                        | nice_to_have    |
| probe_nice_if                  | It would be nice if the system could observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier.                                          | nice_to_have    |
| probe_low_priority_enhancement | As a low-priority enhancement, the system could observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier.                               | nice_to_have    |
| probe_future_enhancement       | Stakeholders mentioned that the system could observe more than 10 of the planets observed in the Chemical Census tier in the Rosetta Stone tier as a possible future enhancement. | nice_to_have    |

## Validation trail

Every derivation step above is backed by a tracked review record:

1. **Seed review.** Both source datasets have per-candidate inclusion/exclusion decisions and final capability clauses:

| Dataset | Review record | Candidate rows |
| --- | --- | ---: |
| `nice` | [`data/processed/seeds_review.csv`](../data/processed/seeds_review.csv) | 622 |
| `mlm_tapt` | [`data/processed/seeds_review_mlm_tapt.csv`](../data/processed/seeds_review_mlm_tapt.csv) | 39139 |

2. **Statement review.** Every dataset/mandatory-keyword cell has its own review table:

| Dataset | Mandatory keyword | Review record | Reviewed statement rows |
| --- | --- | --- | ---: |
| `nice` | `MUST` | [`outputs/benchmark_statements_review.csv`](../outputs/benchmark_statements_review.csv) | 180 |
| `nice` | `SHALL` | [`outputs/benchmark_statements_review_shall.csv`](../outputs/benchmark_statements_review_shall.csv) | 180 |
| `mlm_tapt` | `MUST` | [`outputs/benchmark_statements_review_mlm_tapt.csv`](../outputs/benchmark_statements_review_mlm_tapt.csv) | 180 |
| `mlm_tapt` | `SHALL` | [`outputs/benchmark_statements_review_mlm_tapt_shall.csv`](../outputs/benchmark_statements_review_mlm_tapt_shall.csv) | 180 |

3. **Weak-template construct review.** `docs/weak_modality_construct_review.csv` — 8 rows over the four weak templates; current reviewer roles: `llm-assisted review (author-delegated); PENDING HUMAN CONFIRMATION`. Until the pending human sign-off ([`TODO.md`](../TODO.md) section D), weak-intent claims carry that caveat.

4. **File integrity.** Each dataset manifest records sha256 digests and row counts for its seed and benchmark tables:

| Dataset | Integrity manifest | Artifact entries |
| --- | --- | ---: |
| `nice` | [`outputs/benchmark_manifest.json`](../outputs/benchmark_manifest.json) | 10 |
| `mlm_tapt` | [`outputs/benchmark_manifest_mlm_tapt.json`](../outputs/benchmark_manifest_mlm_tapt.json) | 10 |

The benchmark construction itself is a pure function of the reviewed seed table: `eval_utils.build_benchmark_items` renders each seed through `eval_utils.source_statement`; no manual edits happen between the two.
