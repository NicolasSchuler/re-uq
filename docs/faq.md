# FAQ

Short answers to questions reviewers and new collaborators commonly raise. For depth, follow the linked documents. The complete setup and study limitations are in [`docs/experimental_setup.md`](experimental_setup.md).

## I do not have a provider API key. Can I still verify anything?

Yes. Follow [`docs/reproduction_smoke.md`](reproduction_smoke.md): the `--fake-completion` path exercises the full request planner, runner, parser, and registry plumbing using locally synthesized responses. It does not make HTTP calls. It is not a substitute for a real benchmark run, but it confirms the pipeline is wired end-to-end.

## I have raw outputs already. How do I regenerate the paper-facing analysis without re-running the provider?

Use `scripts/generate_evaluation_analysis.py` directly, or the wrapper:

```bash
bash scripts/reproduce.sh analysis --run-id RUN_ID --task3-run-id TASK3_RUN_ID
```

The analysis script reads cached raw JSONL outputs and writes the paper-facing artifacts under `outputs/evaluation_<dataset>_<variant>_<run_id>/`. It fails closed if the run registry is incomplete, the construct-validity gate is not satisfied, or the confidence-scale contract is violated. To diagnose, use the documented opt-out flags (`--allow-partial`, `--skip-registry-check`, `--skip-construct-review-check`, `--max-parse-failure-rate`). Do not use them for paper-facing results.

## Why are there so many parallel files with `_shall`, `_mlm_tapt`, `_mlm_tapt_shall` suffixes?

Those are dataset-and-variant suffixes. The unsuffixed files are the **main** path (NICE/PROMISE seeds, `MUST` mandatory variant). The other suffixes encode the co-primary dataset (`mlm_tapt`), the secondary robustness variant (`shall`), or both. See the suffix table in [`docs/repository_layout.md`](repository_layout.md).

## Why is there a `.csv` *and* a `.md` for the same file in `outputs/`?

The CSV is the machine-readable evidence and the Markdown is the human-readable rendering used during review. Both are tracked deliberately. See the *CSV + Markdown pair convention* section in `docs/repository_layout.md`.

## How do I prepare or inspect the benchmark?

Use `scripts/prepare_benchmark.py` to prepare candidates and build the main benchmark from reviewed seed tables. The tracked CSVs and their Markdown renderings support inspection; [`benchmark_ground_truth.md`](benchmark_ground_truth.md) explains the transformations. The command sequence is in [`reproduction.md`](reproduction.md).

## What about Task 3? Is it part of the headline?

Task 3 is a **diagnostic** blind text audit, not the headline. It asks whether the deterministic Task 2 extracted text preserved, strengthened, weakened, or changed the source, without revealing the Task 2 declared modality. Declared-modality Task 3 runs are anchoring ablations. See `docs/architecture.md` and `docs/evaluation.md`.

## What if the construct-validity gate is incomplete?

Then `scripts/generate_evaluation_analysis.py` will refuse to write paper-facing artifacts and weak-intent results stay diagnostic. The gate requires at least two distinct reviewer IDs per template and affirmative judgments in every recorded row of [`docs/weak_modality_construct_review.csv`](weak_modality_construct_review.csv). That file retains two LLM-assisted reviews and separate author-confirmation rows. Passing the automated gate does not establish independent human agreement.

Human validation is complete; the authors repeated it before resubmission. The original LLM-assisted judgments remain separately identified. See [validation review](validation_review.md) for the scope and the wording checks' limitations; no independent two-human agreement is claimed.

## Are the prompts in `README.md` what the models actually received?

Yes. Every request of the reported campaign carries one benchmark item rendered from the frozen file in `prompts/`, with no system message, so the file is the request body. The batched wrapper built by `batch_prompt_for_completion_jobs` in `scripts/eval_utils.py` (several items per request, an array of results keyed by `request_index`) was the delivery mode of the archived May campaign and is the request-composition ablation; its prompt bodies are reproduced in [`docs/experimental_setup.md`](experimental_setup.md).

## Does sending several items per request change the results?

It does, and the paper measures by how much. The request-composition ablation repeats deterministic Task 2 on one cell at 4 and 16 items per request, with the four conditions of a capability kept together or spread across requests, against a single-item reference, for all nine models. For most models, several items per request lower the weak-intent strengthening rate by tens of percentage points, so single-item results do not transfer to batched use. The table is `outputs/batching_ablation_summary.md`.

## Does surrounding document context change the result?

That is what the document-context ablation measures. The `pure` cell takes 180 marked requirements from two PURE documents, keeps the minimal-pair templates, and shows each Task 2 item either bare or with its document, section, marker and neighbouring requirements (`item_context: bare|document`). The document arm also adds a context instruction, so it does not isolate information alone. Deltas compare the same eligible source items and resample capabilities in the reported exports. They are never pooled into the headline cells; see [`docs/context_ablation.md`](context_ablation.md).

## Why is `heuristic_system_verb` counted in broad strengthening but not strict?

The broad rule treats a bare `The system exports reports.` as an obligation, following an RE convention. That is a modelling assumption. In the reported campaign, 511 of 25,920 valid outputs (2.0%) receive this heuristic-only classification. A separate 1,092 (4.2%) have unclassified wording and are excluded from the readable-text denominator. The strict rule requires explicit modal or weak-phrase evidence. Both are reported, with eligible counts; neither establishes a bound on semantic error. See [`docs/evaluation.md`](evaluation.md).

## Is the model cohort mostly one family?

It is not: the reported campaign evaluates nine models from five independently developed families, two hosted GLM models and seven open-weight models served locally (see [`docs/experimental_setup.md`](experimental_setup.md) §5.1). The two hosted models share a developer, so the main results are reported per model and conclusions are restricted to the evaluated models. Example profiles for OpenAI, Mistral, Gemini, and Ollama exist so the cohort can be widened further; provider support is deliberately limited to OpenAI-compatible chat-completions endpoints, so any provider exposing one is a profile file away.

## Can I reproduce the exact requests of the reported runs?

Not byte for byte. Every request of the reported campaign carries a recorded seed (20260518 for the single-pass answers, 20260519 to 20260523 for the five samples), and the served model identifier is recorded on every response. Local outputs are still not byte-identical under concurrent serving, and the provider does not expose the revision behind a hosted model id. A rerun is a new run that tests stability; the archived outputs verify the reported numbers (README, reproduction tier 2).

## Will you fine-tune a model to fix this?

No. Fine-tuning is out of scope for this artifact and is not a to-do item. The study measures off-the-shelf behaviour of hosted instruction-tuned models under a frozen prompt contract. Whether fine-tuning removes modal-force strengthening is an open question and a different paper; it is recorded as a limitation.
