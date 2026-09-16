# Python Repository Simplification Review

## Repository and Dependency Map

- **Reviewed baseline:** 36 tracked Python files:
  - 27 under `scripts/` — 21,963 lines
  - 9 under `tests/` — 10,823 lines
- `scripts/eval_utils.py` is the main shared module at 10,123 lines.
- Generated notebooks were assessed through their canonical generator, `scripts/populate_notebooks.py`.
- **Minimum Python:** 3.13.
- **Initial Git state:** `git status --short` and `git diff --stat` were both empty.
- **Baseline tests:** 314 tests passed in 19.99 seconds.

| Area | Declared minimum | Locked version |
|---|---:|---:|
| NumPy | 2.0 | 2.4.5 |
| SciPy | 1.14 | 1.17.1 |
| pandas | 2.2 | 3.0.3 |
| scikit-learn | 1.5 | 1.8.0 |
| Matplotlib | 3.10 | 3.10.9 |
| Pydantic | 2 | 2.13.4 |
| OpenAI | 1.0 | 2.37.0 |
| Instructor | 1.15, `<2` | 1.15.1 |
| Hydra | 1.3.6 | 1.3.6 |
| datasets | 2.20 | 4.8.5 |
| nbformat | 5.10 | 5.10.4 |
| Requests | 2.32 | 2.34.2 |
| tabulate | 0.9 | 0.10.0 |

No new dependency is justified. All package-backed recommendations use dependencies already declared by the repository.

There is no PyTorch, CUDA, JAX, or TensorFlow path. MLX support is optional and absent from `pyproject.toml` and `uv.lock`; the recommendations below do not introduce additional accelerator-to-CPU transfers.

## High-Confidence Simplification Opportunities

No Critical findings were identified.

The strongest opportunities are the shared runner lifecycle (**Finding 5**), Pydantic-backed response schemas (**Finding 2**), and unified ACSE clustering (**Finding 6**). **Finding 4** is primarily a metric-correctness improvement rather than a maintained-code reduction.

## Standard-Library Opportunities

### Finding 1 — Standardize hashing on `hashlib.file_digest`

- **Severity:** Low
- **Location:** `scripts/eval_utils.py:1436-1445`, direct hashes around `5615` and `5695`; duplicate `sha256_text` at `scripts/run_provenance.py:30-31`.
- **Current implementation:** Files are manually read in 1 MiB chunks. Text hashing is duplicated, and two batch paths bypass the shared helper.
- **Candidate replacement:** `hashlib.file_digest(handle, "sha256").hexdigest()`, with all text hashing routed through `eval_utils.sha256_text`.
- **Evidence:** File hashes feed benchmark manifests and external-probe provenance; text hashes feed job fingerprints, request provenance, batch wrappers, and Hydra resolved-config digests. A focused check produced identical digests for a 2.5 MB binary file.
- **Net benefit:** Removes the manual chunk loop and centralizes provenance hashing without adding a dependency.
- **Compatibility risks:** Equivalent for freshly opened, blocking binary files. Do not reuse partially read or nonblocking streams. Preserve UTF-8 encoding for text.
- **Recommendation:** **Replace.**
- **Validation:** Compare empty, small, and multi-chunk files against `hashlib.sha256(path.read_bytes())`; rerun manifest, job-fingerprint, batch-wrapper, and Hydra provenance tests.

## Established-Package Opportunities

### Finding 2 — Derive provider schemas from Pydantic models

- **Severity:** Medium
- **Location:** `scripts/eval_utils.py:3068-3137`; `scripts/structured_outputs.py:24-117`.
- **Current implementation:** The same response fields, enums, bounds, length limits, batch envelopes, and extra-field policy are maintained separately as handwritten JSON Schema and Pydantic models.
- **Candidate replacement:** Wrap `response_model_for_task(task, batched=...).model_json_schema()` in a provider-compatibility adapter that inlines references and removes unsupported metadata where required.
- **Evidence:** Handwritten schemas are consumed through `response_format_for_task`; Pydantic models already drive Instructor requests and response validation. The contracts have drifted: the handwritten Task 3 schema requires `brief_reason`, while `Task3Response` supplies a default and therefore omits it from Pydantic’s `required` list. Batched Pydantic schemas also emit `$defs`/`$ref`.
- **Net benefit:** One response-contract source of truth, approximately 60 fewer handwritten schema lines, and lower risk of validation/provider drift.
- **Compatibility risks:** Not a blind substitution. Strict providers may reject `$ref`, `default`, or descriptive metadata. Decide whether Task 3 `brief_reason` is actually required. Schema changes alter `job_config_sha` and intentionally invalidate resume caches.
- **Recommendation:** **Wrap.**
- **Validation:** Compare all six single/batched contracts field by field; test normalization of `$defs`; run JSON-schema provider smoke tests and existing structured-output/batch-schema tests.

### Finding 3 — Average centroids in projected space

- **Severity:** Low
- **Location:** `scripts/plot_acse_embedding_visualizations.py:216-228` and `599-607`.
- **Current implementation:** Each item’s full-width embeddings are copied and averaged, transformed through PCA again, and conditionally padded.
- **Candidate replacement:** `np.mean(projected[indices], axis=0)`.
- **Evidence:** PCA transformation is affine, so transforming the source-space mean is equivalent to averaging transformed rows. Focused checks across full-rank and padded 2D/3D projections differed by at most approximately `2.2e-16`.
- **Net benefit:** Removes repeated PCA transforms, padding branches, and full embedding-width advanced-index copies.
- **Compatibility risks:** Last-bit floating-point differences may change serialized coordinates. This operates on existing CPU NumPy arrays after embedding conversion; it does not affect MLX placement or synchronization.
- **Recommendation:** **Replace.**
- **Validation:** Compare generated projection CSVs with tight tolerances, including singleton items and PCA outputs padded to three dimensions.

### Finding 4 — Dispatch two-class probes to scikit-learn’s binary APIs

- **Severity:** Medium
- **Location:** `scripts/probe_acse_embedding_separability.py:170-205`, called at `265-268`.
- **Current implementation:** Custom multiclass wrappers send two-column probabilities through scikit-learn’s multiclass AUROC path. `label_binarize` produces one target column for two classes, while average precision receives two probability columns. Both failures are swallowed and reported as `NaN`.
- **Candidate replacement:** Retain a thin adapter:
  - two-class AUROC: `roc_auc_score(y_true, probabilities[:, positive_column])`;
  - two-class macro AP: construct an explicit two-column one-hot target and call `average_precision_score(..., average="macro")`;
  - retain current multiclass APIs for three or more classes.
- **Evidence:** On perfect two-class predictions, the current wrappers returned `NaN`; the corresponding scikit-learn binary/macro APIs returned `1.0`. Default targets such as `dataset_id` and `benchmark_variant` can contain exactly two classes.
- **Net benefit:** Produces valid metrics for legitimate two-class analyses and makes binary versus multiclass semantics explicit.
- **Compatibility risks:** Previously blank artifact cells become populated. Preserve estimator class-column ordering and confirm that two-class AP is intended to remain macro-averaged rather than positive-class-only.
- **Recommendation:** **Wrap.**
- **Validation:** Add two-, three-, and four-class fixtures; imbalanced classes; absent test classes; reordered estimator classes; and constant probabilities.

## Internal Duplication to Consolidate

### Finding 5 — Share the runner execution lifecycle and common CLI options

- **Severity:** High
- **Location:**
  - `scripts/run_experiment_from_config.py:113-150,154-230,337-378,406-678`
  - `scripts/run_task3_verification_from_config.py:168-202,206-277,460-499,528-772`
- **Current implementation:** Task 1/2 and Task 3 independently implement parser options, cell state, registry construction, execution, progress, warnings, events, resume handling, and finalization.
- **Candidate replacement:** A shared internal execution helper with task-specific callbacks/adapters, plus an `argparse.ArgumentParser(add_help=False)` parent for common options.
- **Evidence:** The two `execute_cell` functions are both 155 lines; 140 stripped lines align exactly. Both invoke the same `eval_utils` pending-job, batching, registry, progress, event, warning, completion, and finalization primitives.
- **Net benefit:** Removes roughly 140 duplicated lifecycle lines plus repeated CLI declarations, and prevents Task 1/2 and Task 3 operational behavior from drifting.
- **Compatibility risks:** Preserve Task 1/2’s embedding-backend stamp, Task 3 source/audit provenance, different item populations, event fields, registry notes, and dry-run/source-validation behavior. Planning should remain task-specific.
- **Recommendation:** **Consolidate.**
- **Validation:** Run runner-matrix, resume, warning-event, smoke isolation, dry-run, registry, and Task 3 source-validation tests; separately compare both `--help` outputs and parser defaults.

### Finding 6 — Compute ACSE clusters once and reuse the result

- **Severity:** Medium
- **Location:** `scripts/eval_utils.py:6205-6317`; `scripts/compute_acse_semantic_artifacts.py:470-477`; duplicate `scripts/plot_acse_embedding_visualizations.py:79-98,591-593`.
- **Current implementation:** Diagnostics reconstruct the normalized cosine-distance matrix and then invoke a helper that reconstructs it again. Artifact generation separately asks for labels, while the plotting script carries another implementation of the same clustering algorithm.
- **Candidate replacement:** One internal analysis operation returning both diagnostics and remapped cluster labels; retain public wrappers if needed, and have the plot call the canonical helper.
- **Evidence:** The artifact path currently performs three normalizations, three cosine-similarity calculations, and two `fit_predict` calls for one five-row item. The plot-local implementation duplicates clipped distances, noise removal, average-linkage clustering, and deterministic cluster remapping.
- **Net benefit:** Removes approximately 20 duplicate plot lines, avoids repeated allocations/clustering, and guarantees diagnostics and persisted labels come from the same fit.
- **Compatibility risks:** Preserve empty/singleton behavior, zero-vector semantics, threshold-boundary handling, and relabeling by `(-cluster_size, raw_label)`. Do not substitute SciPy cosine distance: zero and antipodal-vector behavior differs.
- **Recommendation:** **Consolidate.**
- **Validation:** Compare dictionaries and labels for empty, singleton, all-zero, identical, tied-cluster, threshold-boundary, and random matrices; assert one clustering fit per artifact item.

### Finding 7 — Share diagnostic projection-input preparation

- **Severity:** Medium
- **Location:** `scripts/plot_embedding_diagnostic_figure.py:416-462`; `scripts/plot_embedding_diagnostic_tsne_supp.py:77-118`.
- **Current implementation:** Both scripts independently resolve inputs, load the MLX cache and manifest, validate row counts, deduplicate requirement text, perform modality-balanced sampling, remap indices, and compute identical PCA/t-SNE coordinates.
- **Candidate replacement:** Extract the preparation block from the main figure module into a helper returning coordinates, selected rows/indices, and the correctly advanced RNG. Keep figure layout and styling separate.
- **Evidence:** The supplementary script already imports the canonical projection and plotting helpers from the main script and promises equivalent shared panels, but duplicates the data-selection path.
- **Net benefit:** Removes roughly 40 duplicated lines and prevents silent divergence in sampling or index mapping.
- **Compatibility risks:** Preserve iteration order and RNG consumption because they influence point ordering and pixel output. Do not consolidate distinct figure sizing, titles, or style.
- **Recommendation:** **Consolidate.**
- **Validation:** Test selected row IDs and coordinates on a synthetic cache with a fixed seed, then regenerate and image-diff the shared panels.

### Finding 8 — Route manifest JSON through the existing writer

- **Severity:** Low
- **Location:** `scripts/eval_utils.py:1487-1489`; `compare_context_ablation.py:324-338`; `generate_evaluation_analysis.py:602-605`; `diagnose_embedding_separability.py:401-403`; `probe_acse_embedding_separability.py:461-463`; `compute_acse_semantic_artifacts.py:603-605,707-710`; `plot_acse_embedding_visualizations.py:691-693`; `plot_acse_global_embedding_projection.py:364-366`.
- **Current implementation:** Nine sites directly repeat `json.dumps(..., indent=2, sort_keys=True) + "\n"` followed by `Path.write_text`.
- **Candidate replacement:** Existing `eval_utils.write_json`.
- **Evidence:** `write_json` uses the same ASCII escaping, indentation, key ordering, encoding, and trailing newline. A non-ASCII focused comparison was byte-identical.
- **Net benefit:** Removes roughly 20 repeated lines and centralizes manifest serialization and parent-directory creation.
- **Compatibility risks:** Manifests participate in hashing, so byte identity must be verified. Parent creation slightly broadens behavior where callers currently assume the directory exists.
- **Recommendation:** **Consolidate.**
- **Validation:** Regenerate all manifests, compare exact bytes—including non-ASCII values—and rerun provenance and artifact-integrity tests.

## Custom Implementations That Should Remain

### Finding 9 — Keep provider-boundary and recovery behavior

- **Location:** `scripts/eval_utils.py:1155-1208,1622-1671,2097-2210,3618-3648,4160-4526,5188-5844`; `scripts/show_run_progress.py:218-249`.
- **Custom behavior:** Narrow trailing-comma acceptance, locked/fsynced atomic artifact updates, PURE-specific XML flattening, tolerant extraction from model prose, recorded retry classification, deterministic resume-stable batching with per-item fallback, and incremental partial-line-safe JSONL watching.
- **Justification:** These encode observable research, provenance, provider-failure, and corpus semantics. Generic JSON5, retry, XML, or streaming packages would either broaden accepted behavior, hide retry attempts, alter batching/resume behavior, or add dependencies without reducing the project-specific adapter.

### Finding 10 — Keep the seed-clustered bootstrap

- **Location:** `scripts/eval_utils.py:6531-6609`; callers in `generate_evaluation_analysis.py`, `compare_context_ablation.py`, `compare_run_matrix.py`, and `export_paper_tables.py`.
- **Custom behavior:** Resamples complete `seed_id` clusters, repeats all rows for duplicate seed draws, applies paired draws over the union of two-arm seeds, uses percentile intervals, and drops undefined metric replicates.
- **Justification:** `scipy.stats.bootstrap` does not directly reconstruct variable-sized row clusters or the repository’s paired two-arm union semantics; its default BCa interval also differs. A direct substitution would change the study’s statistical unit and reported intervals.

## Prioritized Simplification Backlog

| Priority | Finding | Action |
|---:|---:|---|
| 1 | **#5** | Extract and test the common runner lifecycle first. |
| 2 | **#2** | Resolve the Task 3 contract, then prototype schema normalization. |
| 3 | **#6** | Introduce one ACSE analysis result and remove duplicate fits. |
| 4 | **#4** | Correct two-class metric dispatch before publishing probe results. |
| 5 | **#7** | Share projection preparation while preserving RNG order. |
| 6 | **#8** | Replace repeated manifest serialization with `write_json`. |
| 7 | **#3** | Simplify centroid projection with tolerance-based regression checks. |
| 8 | **#1** | Standardize hashing during a low-risk maintenance pass. |

The three best maintained-code reductions relative to migration risk are:

1. **#5 — Shared runner lifecycle**
2. **#8 — Existing JSON writer consolidation**
3. **#6 — Unified ACSE clustering**

Finding **#2** has greater architectural value than #8 or #6, but also requires provider-level compatibility testing.

## Repository Modification Check

Initial state:

```text
git status --short
# empty

git diff --stat
# empty
```

State at completion of the read-only review:

```text
git status --short
?? .pi/subagents.json
?? .pi/tasks/session-86081-86081/
?? .pi/tasks/session-90068-90068/
?? tests/test_pure_benchmark_builder.py
?? tests/test_risk_regressions.py
?? tests/test_semantic_artifacts.py

git diff --stat
# empty
```

**Constraint violation:** Yes. The repository changed from clean to six untracked paths during the review. There were no tracked-file diffs, but `git diff --stat` does not report untracked files. The unexpected files were not deleted because cleanup would itself modify the repository without authorization.
