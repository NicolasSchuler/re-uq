# Implementation tasks derived from the two reviews

Derived from `docs/abstraction_boundary_review.md` (AB) and
`docs/python_simplification_review.md` (PS) on 2026-09-04.

Global constraints for every task:

- `scripts/eval_utils.py` stays one module (documented deliberate choice).
- Bytes sent to providers (prompts, batch wrappers, JSON Schemas) and the
  pinned job-config / batch-wrapper fingerprints in `tests/` must not change.
  Existing raw rows must still resolve as complete under resume.
- Paper artifacts under `outputs/`, `data/`, `manuscript/` are not regenerated
  by these tasks; each task lists the regeneration command it invalidates.
- Regression tests are written before the restructuring they protect.
- `uv`-managed env: `.venv/bin/python -m unittest discover -s tests`,
  `.venv/bin/ruff check scripts tests`, `.venv/bin/ruff format --check`.

## T1 — Canonical observation identity and attempt ledger (AB#1, AB#7)

- Add a frozen `ObservationIdentity` (provider, profile, model, dataset,
  variant, task, item_id, sample_kind, sample_index) and derive
  `completion_record_key()` from it; legacy raw rows lacking fields fall back
  explicitly (documented) rather than silently.
- Remove the all-rows fallback in `source_rows_for_model()`; a Task 3 run
  whose requested profile has no Task 2 rows must fail or warn loudly, never
  audit another profile.
- Key `stochastic_rows_by_method()` and the deterministic-side join in
  `export_paper_tables.py` on dataset and variant as well as model/item.
- Add an attempt-ledger helper exposing `all_attempts` and
  `latest_logical_observations`; make `run_registry_summary()`,
  `run_progress_summary()`, runner live counters and `show_run_progress.py`
  name which view they consume, so a failed-then-retried item is `1/1`.

## T2 — Required sampling plan and complete-pair bootstrap (AB#2, AB#8)

- `build_uq_scores()` takes a validated `SamplingPlan`; inferring the
  denominator from observed samples is only allowed via an explicitly named
  exploratory flag. Thread the plan through `generate_evaluation_analysis.py`,
  the preliminary snapshot path, and `compare_run_matrix.py`.
- `paired_cluster_bootstrap_delta_ci()` builds and reports the complete-pair
  cohort before resampling; groups with a missing arm are counted and excluded,
  not silently dropped per replicate.

## T3 — Leak-free embedding diagnostic and binary-probe metrics (AB#3, PS#4)

- Primary strengthening probe uses requirement-only text with PCA fitted
  inside each fold; `prefixed` text becomes a labelled positive-control /
  leakage condition.
- Two-class targets dispatch to scikit-learn binary AUROC and explicit
  two-column macro AP instead of returning NaN.
- Figure v2 selects the requirement-only condition for target bars.

## T4 — ACSE clustering, centroid, projection-prep, and cache validity
(PS#6, PS#3, PS#7, AB#9)

- One internal ACSE analysis operation returns diagnostics and remapped
  cluster labels from a single fit; the plot script calls it.
- Centroids are averaged in projected space.
- Diagnostic figure and t-SNE supplement share one projection-input helper
  with identical RNG consumption.
- ACSE artifact discovery reads explicit manifests (empty `model_filter` is
  valid) and manifests carry input fingerprints and manifest-relative paths;
  a stale fingerprint triggers recomputation.

## T5 — Secret-safe provenance, config copying, hashing, JSON writer
(AB#4, AB#10 partial, PS#1, PS#8)

- Validate `api_key_env` as an environment-variable name; reject
  credential-shaped `extra_body` keys; one fail-closed secret validator applied
  by every durable exporter (raw records, registry rows, Hydra YAML).
- `load_config()`/`deep_update()` return deep copies; provider booleans parse
  string forms instead of `bool(value)`.
- File hashing uses `hashlib.file_digest`; all text hashing goes through
  `sha256_text`; the nine manifest writers use `write_json` (byte-identical).

## T6 — Shared runner lifecycle with resume and failure reconciliation
(AB#5, PS#5)

- Extract the common execute-cell lifecycle and a shared argparse parent from
  the two runners, keeping planning task-specific.
- Resume validates that the run ID exists and is compatible before any write,
  preserves original start time and resolved provenance, and reconciles
  terminal state (`complete` / `failed` / `interrupted`) on exceptions.

## T7 — Aligned task contracts (AB#6, PS#2)

- Resolve the Task 3 `brief_reason` and Task 2 blank-requirement drift so the
  JSON Schema, Pydantic models, and tolerant parser agree; parsed results
  record whether a tolerant repair was applied.
- Derive the provider JSON Schema from the Pydantic models through an adapter
  and assert the output is byte-identical to the current handwritten schema
  (so fingerprints are unchanged); then delete the handwritten copies.
