# Phased Evaluation Implementation Plan (Internal Build Log)

> This is the **internal build log** for how the evaluation pipeline was assembled. It is preserved for traceability of design decisions. Readers who want to **reproduce** the experiments should use `docs/reproduction.md` and `docs/architecture.md` instead. Readers who want the **paper-facing framing** should use `docs/paper_framing.md` and `docs/evaluation.md`.

This plan implements the modality-conditioned uncertainty evaluation described in `docs/evaluation.md`.

## Study Framing

The study tests whether LLMs and their uncertainty estimates respect requirement modality and stakeholder commitment. We hold the functional capability constant and vary only the source modality: `MUST`, `SHOULD`, `MAY`, and an operational weak stakeholder-intent class implemented as `nice_to_have`. We measure whether models correctly reject unsupported mandatory interpretations, preserve modality during extraction, and expose risk through uncertainty.

The central observable result is **high-confidence over-commitment of weak stakeholder intent**: an LLM can preserve functional content while incorrectly upgrading stakeholder commitment. Requirements-engineering evaluation should therefore measure high-confidence modality and priority upgrades, not only generic correctness.

Use **commitment normalization** as an explanatory shorthand, not as the primary empirical claim. The evidence supports output behavior, not an internal mechanism.

Treat Task 1 as a capability/control task and Task 2 as the main empirical task. In the current pilot, the model handled mandatory entailment correctly but failed to preserve weak stakeholder intent during extraction.

Add Task 3 as a source-grounded self-audit diagnostic over deterministic Task 2 outputs. Task 3 asks the same model whether its extraction preserved, strengthened, weakened, or changed the source statement; it audits fidelity but does not revise or repair outputs.

For paper-facing wording and caveats, see `docs/paper_framing.md`.

## Environment

- Use the existing `uv`-managed virtual environment at `.venv/`.
- Run Python commands with `.venv/bin/python` or activate the environment with `source .venv/bin/activate`.
- Dependencies are declared in `pyproject.toml` and locked in `uv.lock`; refresh with `uv sync --group dev`.
- Use the declared scientific stack (`pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `openai`, `requests`, and `nbformat`) instead of custom replacements for common CSV, metric, plotting, HTTP, and notebook-JSON tasks.

## Phase 1: Prepare Requirement Seeds

- Run `notebooks/00_prepare_data.ipynb`.
- Load or download the selected seed dataset. `DATASET_ID=nice` uses the NICE/PROMISE-derived CSV in `data/raw/`; `DATASET_ID=mlm_tapt` loads `limsc/mlm-tapt-requirements` through Hugging Face `datasets`.
- Generate the dataset-specific review table: `data/processed/seeds_review.csv` for NICE, `data/processed/seeds_review_mlm_tapt.csv` for HF.
- Manually review included seeds until the configured target count of accepted capabilities remains.
- Commit the final human-readable seed documents under `docs/final_seed_documents/` once the reviewed capability text is accepted.

## Dataset Scope

- NICE/PROMISE-derived requirements remain the backward-compatible default dataset.
- `limsc/mlm-tapt-requirements` is the second co-primary dataset path. Exclude `_PURE` sources, apply strict candidate filtering, and use deterministic source-weighted sampling before manual review.
- Raw PURE is not recommended as a benchmark source because its document structure is too heterogeneous for the short-communication scope.
- Dalpiaz user-story datasets are not primary paper data because of licensing/IP uncertainty; use them only for private sensitivity checks if licensing is resolved.
- For the IST short communication, compare NICE and `mlm_tapt` only after both reviewed seed lists pass the same manual-review gate.

## Phase 2: Build Modality Benchmark

- Run `notebooks/01_build_modality_benchmark.ipynb`.
- Generate four controlled variants for each accepted seed: mandatory, recommended, optional, and weak stakeholder intent.
- Keep `nice_to_have` as the implementation label, but paper-facing text should call this condition weak stakeholder intent or weak desiderative intent.
- Write `data/processed/benchmark_items.csv`.
- Preserve an existing reviewed benchmark by default; if regenerated content differs, write `data/processed/benchmark_items_candidate.csv` for comparison.
- Generate a secondary `SHALL` robustness benchmark at `data/processed/benchmark_items_shall.csv`. The main paper claim remains based on the `MUST` benchmark; `SHALL` is a robustness check.
- Write `outputs/benchmark_manifest.json` with SHA-256 hashes, row counts, and metadata for reviewed seeds, benchmark files, and prompt files.

## Strengthening Additions

- Add a deterministic rule-based modality baseline as a sanity comparator, not a competing ML method.
- Add high-confidence over-commitment metrics at thresholds `0.80` and `0.90`.
- Treat all `v2-conf01` and `v2-instructor-conf01` model outputs as probability-scale confidence in `[0.0, 1.0]`; new v2 raw rows must carry `confidence_scale=0_1`.
- Add Task 3 source-grounded modality self-audit after the full run, using `prompts/modality_verification.txt` and writing `data/processed/model_outputs_raw_task3_verification.jsonl`.
- Add one Task 1 prompt-sensitivity check on the pilot subset using `prompts/mandatory_entailment_strict.txt`.
- Add one focused Task 2 prompt-validity check for `nice_to_have` sources using `prompts/modality_extraction_labels_only.txt`, which states the allowed output labels without giving deterministic mapping rules or examples.
- Add `notebooks/02b_weak_modality_robustness_probe.ipynb` before full runs to test whether the `nice_to_have` failure is phrase-specific or robust across weak stakeholder-intent phrasings.
- Treat the weak-modality robustness probe as construct-validity support, not as a headline result by itself.
- Use `docs/weak_modality_construct_review.csv` as a two-reviewer paper-readiness gate for the weakest class. Full experimental runs may proceed, but paper-ready weak-intent claims require both reviewer slots for every template to mark `weaker_than_should=yes`.
- Export qualitative over-commitment examples after model outputs exist.
- Use `BENCHMARK_VARIANT=must` by default and `BENCHMARK_VARIANT=shall` for the secondary robustness path; `SHALL` artifacts use `_shall` suffixes.

## Survey-Aligned UQ Methods

- Use the ACM CSUR UQ survey taxonomy as framing: token-level, self-verbalized, semantic-similarity/distributional, and mechanistic UQ.
- Keep `verbalized_confidence` as the self-verbalized method.
- Keep `label_self_consistency` and `modality_consistency` as black-box consistency methods.
- Add `predictive_entropy` over the same stochastic label samples, normalized to `[0, 1]`.
- Add `variation_ratio`, defined as `1 - p_majority`, over the same stochastic label samples.
- Add `model_ensemble_disagreement` only when deterministic outputs from at least two local models are available for the same item and task.
- Treat `token_logprob_confidence` as optional and non-headline until the local OpenAI-compatible endpoint passes a logprob capability probe. For LM Studio, this probe uses `/v1/responses` with `include=["message.output_text.logprobs"]` and `top_logprobs`, because logprobs are not expected from the normal chat UI or necessarily from `/v1/chat/completions`.
- Keep the UQ claim narrow: consistency-based UQ may miss systematic stable errors in modality extraction. Do not claim that all lightweight UQ fails, because verbalized confidence and token/logprob signals may behave differently.
- Exclude mechanistic interpretability from this short communication because it requires model internals and would broaden the contribution beyond lightweight black-box UQ.

## Phase 3: Pilot Local LLM Calls

- Run `notebooks/02_pilot_local_llms.ipynb`.
- Configure the OpenAI-compatible local endpoint with `HOST` and `MODEL`.
- Set `RUN_PILOT=true` only after the endpoint is available.
- Continue only if JSON parse success is at least 95%.
- Inspect stochastic pilot diagnostics for self-consistency, predictive entropy, and variation ratio.
- Optionally set `RUN_PROMPT_SENSITIVITY=true` to compare the default and strict Task 1 prompts on the deterministic pilot subset.
- Optionally set `RUN_TASK2_PROMPT_SENSITIVITY=true` to compare the default and labels-only Task 2 extraction prompts on the `nice_to_have` pilot subset.
- Optionally set `RUN_LOGPROB_PROBE=true` to test whether the local endpoint supports token logprobs through `/v1/responses`; do not make token-level UQ a headline result unless this succeeds.

## Phase 3b: Weak-Modality Robustness Probe

- Run `notebooks/02b_weak_modality_robustness_probe.ipynb` before the full experiment.
- Use the same 20 pilot seeds and Task 2 prompt, but replace the weak source wording across four templates: `useful_if`, `nice_if`, `low_priority_enhancement`, and `future_enhancement`.
- Complete `outputs/weak_modality_template_sanity_check.csv` first by marking every template as weaker than `SHOULD`; the notebook skips model calls until this gate passes.
- Write separate probe items, raw outputs, and summaries under `data/processed/weak_modality_probe_items.csv`, `data/processed/model_outputs_raw_weak_modality_probe.jsonl`, and `outputs/weak_modality_probe_summary.*`.
- Proceed to full runs only after reviewing whether weak-modality collapse generalizes beyond the current `useful_if` wording.
- Current pilot finding: in run `weak-modality-probe-20260518-220538-6a39dfa8`, `nice_to_have` was never preserved across four weak-intent phrasings; the model assigned either `optional` or `recommended` with high confidence. Treat this as construct-validity support for proceeding to full runs, while noting that the exact upgraded class is wording-sensitive.

## Phase 4: Run Full Experiments

- Prefer the provider-aware CLI for GPU/server runs: copy `run_configs/full_matrix.example.json` to the ignored `run_configs/current_run.json`, edit provider profiles, then run `scripts/run_experiment_from_config.py`.
- Start each provider/model with `--mode smoke`; proceed to `--mode full` only after provider preflight and smoke parsing succeed.
- For llama.cpp, run one model at a time because the server must be restarted manually for each loaded model; pass `--model` for the currently loaded model.
- For Z.ai GLM Coding Plan keys, use `https://api.z.ai/api/coding/paas/v4`; the general `https://api.z.ai/api/paas/v4` endpoint requires standard API balance.
- Use profile-level `batch_size` to reduce API-call overhead. Batches only combine compatible jobs for the same task, sample kind, sample index, model, and provider settings, then split the response back into one JSONL row per item/sample.
- Use `structured_output=json_schema` for providers that accept strict OpenAI-style structured outputs; use `json_object` for loose JSON mode and `none` for local endpoints that do not support response formatting.
- For current v2 prompts, reject percentage-style confidence values such as `95` or `"95%"` during smoke checks instead of treating them as successful parses.
- Cache every raw response in the canonical `data/processed/model_outputs_raw*.jsonl` files and track completion in `data/processed/run_registry*.csv`.
- Use the live progress artifacts during long runs: `data/processed/run_progress_live*.csv` for current task coverage and `data/processed/run_events*.jsonl` for structured start/progress/warning/finish events. Monitor with `scripts/show_run_progress.py --dataset nice --run-id RUN_ID --watch 30`.
- `notebooks/03_run_experiments.ipynb` remains available for direct env-var driven local runs, but the CLI is the reproducible path for provider matrices.

Example:

```bash
.venv/bin/python scripts/run_experiment_from_config.py \
  --config run_configs/current_run.json \
  --profile local_llama_cpp \
  --model qwen/qwen3.5-9b \
  --dataset nice \
  --mode smoke
```

## Phase 4b: Run Source-Grounded Modality Verification

- Run `scripts/run_task3_verification_from_config.py` after a complete Task 2 full run. `notebooks/03b_run_modality_verification.ipynb` mirrors this diagnostic for inspection.
- Build `data/processed/task3_verification_items.csv` from valid deterministic Task 2 outputs.
- Run the same model as verifier for each extraction, with deterministic decoding plus stochastic verifier samples.
- Cache raw Task 3 responses in `data/processed/model_outputs_raw_task3_verification.jsonl`.

## Phase 5: Compute UQ and Metrics

- For publication artifacts, run `scripts/generate_evaluation_analysis.py`. `notebooks/04_compute_uq_and_metrics.ipynb` remains available for staged inspection.
- Analyze one full experiment run at a time. The notebook selects the latest `full-*` run by default; set `RUN_ID` or `ANALYSIS_RUN_ID` to reproduce an earlier run.
- For provider/model matrix comparison, run `scripts/compare_run_matrix.py --config run_configs/current_run.json` after the registry marks the relevant runs complete.
- Compute verbalized confidence, label self-consistency, modality consistency, relation consistency, predictive entropy, variation ratio, and model-ensemble disagreement when available.
- Export UQ scores, metric summaries, rule-baseline rows, bootstrap confidence intervals, high-confidence risk rates, error-detection AUROC, and the recommended-strength sensitivity check.
- Export seed-level bootstrap intervals for headline risks: unsupported mandatory acceptance, `HC-OC_overcommittable`, and `weak_strengthening`.
- Report selective error after deferring the most uncertain 10% and 20% of predictions as a compact triage diagnostic.
- Use precise denominator-specific names for high-confidence risks:
  - Task 1 `unsupported_mandatory_acceptance@tau`: among non-mandatory sources, the fraction assigned `p_yes >= tau` for the mandatory candidate.
  - Task 2 `HC-OC_all@tau`: among all valid Task 2 outputs, the fraction that strengthen the source and have confidence at least `tau`.
  - Task 2 `HC-OC_overcommittable@tau`: the same numerator, but excluding mandatory source rows from the denominator.
  - Task 2 `weak_strengthening@tau`: among weak stakeholder-intent sources, the fraction predicted as any stronger modality with confidence at least `tau`.
- Treat entropy, variation ratio, and consistency frequency as summaries of the same stochastic label distribution, not as independent prediction methods.
- Treat Task 3 as a source-grounded self-audit diagnostic, not independent verification.

## Phase 6: Export Paper Artifacts

- Use `scripts/generate_evaluation_analysis.py` to export the compact paper-facing result table, modality figure, qualitative examples, UQ method inventory, and provenance manifest under a local `outputs/evaluation_*` directory. `notebooks/05_analyze_and_export_results.ipynb` remains a companion export notebook.
- Inspect the figure and result notes before using them in the IST manuscript.
- Treat checked-in 468-row metric snapshots as stale until a clean post-fix full run replaces them.

## Verification

- Use `.venv/bin/python -m unittest discover -s tests -v` for utility tests.
- Use coverage before finalizing a code/data contract change: `.venv/bin/python -m coverage run --branch --source=scripts -m unittest discover -s tests -v` followed by `.venv/bin/python -m coverage report -m`.
- Validate notebook JSON after changes.
- Treat model results as paper-ready only after inspecting benchmark items, parse failures, metric summaries, and generated plots.
