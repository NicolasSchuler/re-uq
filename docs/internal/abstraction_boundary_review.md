# Abstraction-Boundary Review

## Outcome

No Critical findings were identified. The review found seven High-severity and three Medium-severity material boundary weaknesses. The main theme is the absence of first-class experiment identities and contracts: profile, dataset, variant, sampling plan, and prompt/schema versions are repeatedly reconstructed from loose dictionaries and partial keys.

## Findings

### 1. High — Distinct experiment observations collapse onto incomplete identities

- **Severity:** High
- **Boundary:** Runner completion cache, Task 2-to-Task 3 handoff, and paper-export joins.
- **Problem:** There is no canonical run-cell or observation identity. Different modules define identity with different subsets of fields.
- **Evidence:**
  - `completion_record_key()` uses only `(model, task, item_id, sample_kind, sample_index)` in `scripts/eval_utils.py:7358-7367`, omitting provider, profile, dataset, variant, and run group.
  - `source_rows_for_model()` falls back from exact-profile rows to all rows for the model in `scripts/run_task3_verification_from_config.py:136-143`.
  - `stochastic_rows_by_method()` indexes paper-facing UQ rows by `(model, item_id)` in `scripts/export_paper_tables.py:321-333`.
  - All 720 IDs overlap between `must` and `shall` in both benchmark families. A focused reproduction pooling two cells retained one row instead of two. Another reproduction showed that the same job under another provider/profile was treated as already complete.
- **Consequence:** Resume can reuse another provider's result; Task 3 can audit a different profile's Task 2 output while recording the requested profile; pooled paper agreement can attach the wrong stochastic row to a deterministic observation.
- **Improvement:** Introduce immutable `RunCellIdentity` and `ObservationIdentity` values. Include run group, run ID, provider, profile, model, dataset, variant, task, item, sample kind, and sample index as appropriate, and use them uniformly for cache keys, handoffs, registries, joins, and manifests.
- **Scope:** Architectural migration, with compatibility handling for legacy raw rows.
- **Confidence:** High. The key omissions and benchmark collisions were directly reproduced. Current raw files contain no same-run/model multi-profile group, but the paper-export collision is reachable with the current four-cell matrix.

### 2. High — The stochastic sampling plan is optional at scoring boundaries

- **Severity:** High
- **Boundary:** Raw stochastic responses to UQ scores, analysis, and paper exports.
- **Problem:** Completeness is inferred from observed samples when callers omit the intended sample count.
- **Evidence:**
  - `build_uq_scores()` only detects missing planned samples when given `expected_stochastic_samples` (`scripts/eval_utils.py:6851-6874,6958-6969`).
  - `scripts/generate_evaluation_analysis.py:340,474,584`, preliminary snapshots in `scripts/eval_utils.py:8084-8099`, and `scripts/compare_run_matrix.py:274` omit the value.
  - `scripts/export_paper_tables.py:727-731` threads it correctly, and `tests/test_paper_exports.py:1300-1337` protects only that path.
  - A direct check found that three observed samples were reported as complete with `3/3`; supplying the intended plan of five correctly produced incomplete `3/5`.
- **Consequence:** Partially executed runs can silently enter calibration, agreement, and comparison outputs as complete observations.
- **Improvement:** Make a validated `SamplingPlan` required for paper-facing scoring. Do not infer the denominator from observations except in an explicitly named exploratory mode.
- **Scope:** Small signature and caller refactor, but it establishes an architectural contract.
- **Confidence:** High.

### 3. High — The primary embedding diagnostic encodes target ingredients and fits preprocessing outside cross-validation

- **Severity:** High
- **Boundary:** Parsed Task 2 response to semantic representation and held-out separability result.
- **Problem:** The reported strengthening probes use text prefixed with the model's predicted modality, while strengthening is derived partly from that modality. PCA is also fitted on all observations before folds are created.
- **Evidence:**
  - `semantic_response_text()` prepends `modality: <predicted label>` in `scripts/eval_utils.py:6005-6023`.
  - The paper figure explicitly selects `text="prefixed"` for all target bars in `scripts/plot_embedding_diagnostic_figure_v2.py:88-143`.
  - The same file deliberately uses requirement-only text for context bars because prefixed text is recognized as circular (`scripts/plot_embedding_diagnostic_figure_v2.py:62-67`).
  - `scripts/probe_acse_embedding_separability.py:391-397` calls `PCA.fit_transform(embeddings)` globally before fold evaluation.
- **Consequence:** The reported held-out AUROC can reflect the explicit predicted label and transductive preprocessing rather than semantic evidence recoverable from requirement text.
- **Improvement:** Make requirement-only text with fold-local PCA the primary diagnostic. Retain prefixed text only as a clearly labelled positive-control or leakage condition, then regenerate the affected table and figure.
- **Scope:** Local probe and figure changes followed by artifact regeneration.
- **Confidence:** High regarding leakage; the numerical effect requires rerunning the probe.

### 4. High — Credential-shaped provider data can cross into durable artifacts

- **Severity:** High
- **Boundary:** Provider configuration to normalized configuration and raw/provenance exports.
- **Problem:** The configuration contract says secrets remain outside configuration, but arbitrary request bodies and malformed `api_key_env` values can be persisted unredacted.
- **Evidence:**
  - `normalize_provider_profile()` accepts arbitrary `extra_body` and does not validate `api_key_env` as an environment-variable name (`scripts/eval_utils.py:938-1043`).
  - `build_raw_record()` persists `request_extra_body` (`scripts/eval_utils.py:5122-5135`), and registry rows serialize it at `scripts/eval_utils.py:7540`.
  - `run_config_to_hydra_yaml()` writes normalized profiles directly in `scripts/hydra_bridge.py:210-249`.
  - This contradicts the module contract at `scripts/hydra_bridge.py:10-13`.
- **Consequence:** Tokens placed in fields such as `extra_body.authorization` or accidentally supplied as the value of `api_key_env` can enter raw JSONL, generated Hydra profiles, registries, or resolved configuration.
- **Improvement:** Validate `api_key_env` against an environment-variable-name pattern; reject credential-shaped keys in `extra_body`; apply a shared fail-closed secret validator to every durable exporter.
- **Scope:** Architectural security/provenance policy with localized validation changes.
- **Confidence:** High about the exposure path. The review did not establish that a real credential is currently present.

### 5. High — Runner execution is not governed by an atomic lifecycle protocol

- **Severity:** High
- **Boundary:** Planning, provider invocation, resume, registry state, and live progress.
- **Problem:** Locks protect writes but not the `read pending → claim cell → call provider` transaction. Resume is not required to resolve an existing compatible cell, and exceptions bypass terminal reconciliation.
- **Evidence:**
  - Both runners calculate pending work before execution (`scripts/run_experiment_from_config.py:238-327`; `scripts/run_task3_verification_from_config.py:340-445`).
  - Storage locking covers individual append/upsert operations, not ownership of provider calls (`scripts/eval_utils.py:1621-1669,7560-7585`).
  - Batches are eagerly submitted in `scripts/eval_utils.py:5838-5844`.
  - Resume validates only that a run ID was supplied; planning assigns a new start time and rewrites resolved provenance (`scripts/run_experiment_from_config.py:570-660`; corresponding Task 3 path at `scripts/run_task3_verification_from_config.py:693-754`).
  - Completion loops lack failure reconciliation, so exceptions can leave status `running`.
- **Consequence:** Overlapping resumes can duplicate paid requests and raw attempts; a typo can silently create a "resumed" run; interruptions leave stale lifecycle state and overwritten provenance.
- **Improvement:** Add a per-cell lease and explicit state machine: planned → claimed → running → complete/failed/interrupted. Validate resume identity before writes, preserve original provenance, and atomically reconcile terminal state.
- **Scope:** Architectural runner-control refactor.
- **Confidence:** High from control flow. No concurrent provider test was run, and current raw files contained no duplicate logical keys.

### 6. High — Prompt, schema, model, and parser contracts have diverged

- **Severity:** High
- **Boundary:** Task specification to provider request, response validation, and provenance.
- **Problem:** A task contract is represented independently in prompt files, hard-coded batch wrappers, JSON schemas, Pydantic models, and tolerant parsers.
- **Evidence:**
  - `prompts/README.md` acknowledges that batched prompts are implemented separately in `batch_prompt_for_completion_jobs()` (`scripts/eval_utils.py:2789-2935`); editing the canonical prompt does not update the actual batch wrapper.
  - Benchmark manifests hash prompt files, while the actual batch wrapper is only partially represented through job-level wrapper hashes.
  - Task 3 JSON Schema requires `brief_reason`, while `Task3Response` and the tolerant parser permit it to be absent (`scripts/eval_utils.py:3079-3102,4196-4205`; `scripts/structured_outputs.py:65-69`).
  - Plain parsing accepts aliases rejected by strict JSON Schema and Instructor paths.
  - Task 2 declares `requirement` as an unconstrained string (`scripts/eval_utils.py:3088-3094`); direct validation accepted a whitespace-only requirement as a successful extraction.
- **Consequence:** The same semantic response can pass or fail depending on provider adapter. Prompt changes may alter unbatched requests but not batched ones, while provenance suggests a single contract.
- **Improvement:** Define one declarative `TaskContract` per task and derive single/batch prompts, Pydantic models, JSON Schema, parser validation, and contract hashes from it. Record tolerant repairs distinctly from conforming responses.
- **Scope:** Architectural contract consolidation.
- **Confidence:** High.

### 7. High — Physical attempts and logical observations are reconciled inconsistently

- **Severity:** High
- **Boundary:** Append-only raw rows to registry, progress, and quality reporting.
- **Problem:** Some consumers deduplicate retries; others compute headline counts and quality metrics over every physical attempt.
- **Evidence:**
  - `dedupe_raw_rows()` and `run_progress_summary()` treat resumed attempts as one logical result (`scripts/eval_utils.py:3872-3898`).
  - `run_registry_summary()` calculates observed records, parse rate, API calls, and quality from undeduplicated rows (`scripts/eval_utils.py:7440-7503`).
  - Runner live counters and `scripts/show_run_progress.py:151-195` likewise use raw attempts for some headline metrics while task progress deduplicates internally.
- **Consequence:** A failed response followed by a successful retry can simultaneously appear as `2/1`, a degraded registry parse rate, and `1/1` successful task progress. Quality gates may reject a recovered run.
- **Improvement:** Introduce a shared attempt-ledger abstraction that exposes both `latest_logical_observations` and `all_attempts`. Require each metric to name which view it consumes.
- **Scope:** Shared utility plus registry, runner, and reporting callers.
- **Confidence:** High from static tracing. Current raw artifacts had no duplicate logical keys, so this is a recovery-path defect rather than an observed current-data discrepancy.

### 8. Medium — "Paired" bootstrap inference does not establish a complete-pair population

- **Severity:** Medium
- **Boundary:** Per-seed/model cell rows to paired contrast confidence intervals.
- **Problem:** Groups with only a baseline or only a comparison arm remain in the bootstrap population and are silently skipped after sampling.
- **Evidence:** `paired_cluster_bootstrap_delta_ci()` accepts any group where at least one arm exists and computes both means later (`scripts/eval_utils.py:3547-3610`). Missing-arm draws yield non-finite differences that are filtered afterward.
- **Consequence:** The effective number of pairs varies by bootstrap replicate, changing the interval relative to a defined complete-pair bootstrap. The reported method is not equivalent to its "paired" label.
- **Improvement:** Construct and report the complete-pair cohort before resampling. Alternatively, specify and test an explicit missing-arm estimand.
- **Scope:** Local metric correction and affected export regeneration.
- **Confidence:** High about the implementation; whether current paper cells contain missing arms was not exhaustively recomputed.

### 9. Medium — ACSE artifacts use directory conventions and status flags as cache validity

- **Severity:** Medium
- **Boundary:** Completed analyses to semantic-cache discovery and diagnostic plots.
- **Problem:** Cache identity is not bound to source contents, and discovery depends on metadata populated by one invocation convention.
- **Evidence:**
  - `completed_runs_from_analysis_dirs()` ignores otherwise valid analysis manifests whose `model_filter` is empty (`scripts/compute_acse_semantic_artifacts.py:139-180`), although `scripts/generate_evaluation_analysis.py:565` can write exactly that value.
  - Existing output is skipped when its manifest merely says `status="computed"`; raw rows, score CSVs, benchmark content, embedding revision, and code version are not fingerprinted.
  - Backend output directories are keyed only by backend name, and generated manifests contain absolute checkout paths.
  - A direct check confirmed that an empty-model-filter manifest is undiscoverable.
- **Consequence:** Valid analyses can disappear from aggregation, while stale embeddings can be silently reused after source changes. Moving the repository can invalidate manifest paths.
- **Improvement:** Discover explicit manifests rather than inferring validity from directory names. Fingerprint source raw/score/benchmark inputs and complete backend identity; use content-addressed or verified cache keys and paths relative to the manifest.
- **Scope:** ACSE artifact subsystem redesign.
- **Confidence:** High about the failure modes; no currently stale cache was demonstrated.

### 10. Medium — Core configuration leaks mutable process state and deployment assumptions

- **Severity:** Medium
- **Boundary:** Configuration loading and optional analysis backends.
- **Problem:** Configuration is an untyped, shallowly copied global dictionary inside a utility module that also imports and initializes unrelated scientific and provider stacks.
- **Evidence:**
  - `deep_update()` performs shallow copying, and missing-file `load_config()` returns `DEFAULT_CONFIG` itself (`scripts/eval_utils.py:443-461`).
  - A direct check found that mutating one returned configuration changed the result of the next call (`same_object=True`).
  - Provider booleans use `bool(value)`, so serialized values such as `"false"` become true (`scripts/eval_utils.py:963-968,1031-1041`).
  - Importing `scripts.run_provenance` transitively imports the 10,123-line `eval_utils.py`, creates cache directories, and triggered Matplotlib font-cache construction.
  - Default Hydra composition selects MLX embeddings, while the standard locked environment does not declare `mlx-embeddings`.
- **Consequence:** Caller mutation becomes process-global, configuration behavior depends on Python truthiness, lightweight tools acquire heavyweight side effects, and a default analysis path depends on an out-of-band installation.
- **Improvement:** Replace loose dictionaries with immutable validated configuration objects; deep-copy defaults during migration; move optional provider, plotting, and embedding imports behind subsystem boundaries and declare backend extras explicitly.
- **Scope:** Staged architectural decomposition.
- **Confidence:** High.

## Architectural conclusions

### Top three improvements

| Priority | Improvement | Impact | Effort |
|---|---|---:|---:|
| 1 | Introduce canonical run-cell and observation identities and migrate caches, Task 3 handoffs, registries, and exporter joins | Very high | Medium-high |
| 2 | Establish generated task and sampling contracts: one source for prompts, schemas, parsers, hashes, and expected samples | Very high | Medium |
| 3 | Correct the paper-facing embedding and paired-inference pipelines, then regenerate affected artifacts | High | Low-medium |

### Strongest boundary

The strongest boundary is the low-level persistence and isolation layer: registry writes use advisory locking and atomic replacement, raw appends are serialized and flushed, and smoke/fake artifacts are routed away from paper-facing files. Focused structured-output and shared-artifact concurrency tests passed. The principal defects sit above this layer in identity, lifecycle, and logical-record semantics.

### Weakest boundary

`scripts/eval_utils.py` is the weakest boundary. Its 10,123 lines and 318 top-level functions combine configuration, file storage, downloads, provider calls, parsing, metrics, embeddings, plotting, and exports. The issue is not size alone: it produces demonstrated mutable-default leakage, import-time side effects, duplicated task contracts, and broad transitive dependency coupling.

### Most serious representation leak and implicit protocol

- **Representation leak:** Task 2's predicted modality is inserted directly into the embedding text used to predict modality strengthening.
- **Implicit protocol:** Provider/profile/dataset/variant identity is expected to travel alongside rows but is not part of a single enforced identity value.

### Representative propagation path

```text
Task 2 contract change
  ├─ prompts/modality_extraction.txt
  ├─ hard-coded batch wrapper (may not change)
  ├─ JSON Schema / Pydantic model / tolerant parser
  ├─ job and manifest hashes
  └─ parsed_json
       → semantic_response_text()
       → UQ scores and embedding caches
       → paper joins and exported tables/figures
```

This path explains why a nominally local prompt or schema edit currently has non-obvious, partially untracked effects across the repository.

## Open questions

1. What is the numerical effect of adding dataset, variant, and run identity to the current pooled paper agreement export? The collision is confirmed, but corrected tables were not generated.
2. How much do requirement-only text and fold-local PCA change the reported strengthening AUROCs?
3. Have any private run configurations or raw artifacts ever contained credential-shaped `extra_body` fields? The exposure path exists, but no secret inventory was performed.
4. How many parse-valid Task 2 outputs are omitted from Task 3 because their heuristic textual modality is `unknown` or `negated`?
5. Should interrupted JSONL tails and shared live-progress CSVs support automatic recovery, or is fail-fast behavior intended?

## Verification

- Passed: all focused `tests.test_structured_outputs` tests and both `SharedArtifactConcurrencyTest` cases, eight tests total.
- Directly reproduced: cross-cell overwrite, provider/profile cache collision, omitted sampling-plan behavior, blank Task 2 acceptance, mutable default leakage, and empty-model-filter ACSE discovery failure.
- Inspected current raw artifacts: no duplicate logical completion keys and no same-run/model multi-profile groups.
- Broad test suite: not run.
- Repository modifications made by the review itself: none. This report is the only added file.
