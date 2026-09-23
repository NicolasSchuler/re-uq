# When Weak Intent Becomes a Requirement

[![CI](https://github.com/NicolasSchuler/re-uq/actions/workflows/ci.yml/badge.svg)](https://github.com/NicolasSchuler/re-uq/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
<!-- The Zenodo DOI badge is added with the v2.0.0 release. -->

Replication package for the paper **"When Weak Intent Becomes a Requirement:
Limits of Uncertainty Signals in LLM-Assisted Requirements Engineering"**
(Nicolas Schuler, Vincenzo Scotti, Raffaela Mirandola; Karlsruhe Institute of
Technology), submitted to the *Journal of Systems and Software*.

Large language models can extract the right capability from a stakeholder
statement and still change how strongly the stakeholder meant it. The study
holds the capability fixed, varies only the modal force of the source
statement, and asks nine models to extract the requirement. It measures how
often the generated text strengthens the source, whether lightweight
uncertainty signals notice, and whether output-based checks catch it.

Everything the paper's numbers are generated from is tracked here: the frozen
prompts, the benchmark, the campaign configuration, the analysis code, the
tables and figures, and the provenance that names every run. The raw model
outputs (about one gigabyte) are archived separately on Zenodo; see
[Reproduce](#reproduce).

## Results at a glance

Every value below is read from a tracked file. A test keeps this table equal
to [`outputs/paper_numbers.tex`](outputs/paper_numbers.tex), which is itself
generated from the tables under `outputs/`; the Source column names the table.

| Quantity | Value | Source |
| --- | --- | --- |
| Models | 9, from five independently developed families | `paper_snapshot_provenance.json` <!-- num:numModels --> |
| Items per request, primary protocol | 1 | `paper_snapshot_provenance.json` <!-- num:numBatchSize --> |
| Benchmark | 360 capabilities x 4 source conditions = 1440 items per wording variant, 720 per cell | `benchmark_manifest*.json` <!-- num:numSeeds numItems numItemsPerCell --> |
| Task 2 answers scored | 25,920, of which 24,828 with a readable text modality | `paper_per_model_rq_table.csv` <!-- num:numTaskTwoAnswers numTaskTwoReadable --> |
| Task 1 mandatory-entailment accuracy | 97.5% | `paper_per_model_rq_table.csv` <!-- num:numTaskOneAcc --> |
| Task 2 declared-label accuracy | 91.8% | `paper_per_model_rq_table.csv` <!-- num:numLabelAcc --> |
| Text strengthened the source, strict wording check, all sources | 26.0% [25.9, 26.2]; per model 24.6 to 34.2 | `paper_headline_metrics.csv`, `paper_per_model_headline.csv` <!-- num:numStrictOverall numStrictOverallCI numStrictModelsRange --> |
| Text strengthened the source, broad wording check, all sources | 27.9% [27.6, 28.2]; per model 25.0 to 42.5 | `paper_headline_metrics.csv`, `paper_per_model_headline.csv` <!-- num:numBroadOverall numBroadOverallCI numBroadModelsRange --> |
| Weak-intent sources strengthened, strict, ungated | 99.7% (6320 of 6337 readable answers) | `paper_per_model_rq_table.csv` <!-- num:numWeakStrict numWeakStrictNum numWeakStrictDen --> |
| Weak-intent sources strengthened at verbalized confidence >= 0.90 | 82.8% | `paper_headline_metrics.csv` <!-- csv:paper_headline_metrics.csv:weak_strict_text_strengthening_90:value_macro_over_cells --> |
| Weak-intent answers rewritten to should/shall/must; kept could/may but dropped the wish framing | 68.9%; 30.8% | `paper_per_model_rq_table.csv` <!-- num:numWeakEscalation numWeakFrameOnly --> |
| Strict-strengthened outputs with verbalized confidence >= 0.90 | 83.4% [82.4, 84.2] | `paper_headline_metrics.csv` <!-- num:numHighConfShare numHighConfShareCI --> |
| Strict-strengthened outputs whose five sampled labels all agree | 67.5% | `paper_headline_metrics.csv` <!-- num:numSampleAgreement --> |
| Strengthening detectors, AUROC: meaning variation across samples; verbalized confidence | 0.719 [0.710, 0.728]; 0.935 | `paper_per_model_rq_table.csv` <!-- num:numMeaningVarAUROC numMeaningVarAUROCCI numVerbConfAUROC --> |
| Embedding classifier on the requirement text, held-out capabilities, AUROC | 0.810 [0.795, 0.824] | `embedding_diagnostic/probe_grid_summary.csv` <!-- num:numEmbGlobalAUROC numEmbGlobalAUROCCI --> |
| Blind preservation check: strict-strengthened outputs it flags | 94.8% [94.0, 95.5] | `paper_per_model_rq_table.csv` <!-- num:numBlindRecall numBlindRecallCI --> |

Intervals are 95% bootstrap intervals clustered on the capability. Per-model
values for every quantity are in
[`outputs/paper_per_model_rq_table.md`](outputs/paper_per_model_rq_table.md);
the ablation results are in
[`outputs/batching_ablation_summary.md`](outputs/batching_ablation_summary.md),
[`outputs/context_ablation_summary.md`](outputs/context_ablation_summary.md)
and the nine directories under
[`outputs/weak_modality_probe/`](outputs/weak_modality_probe/). How the
numbers are pooled is specified in [`docs/aggregation.md`](docs/aggregation.md).

## The study in one minute

- Source: "It would be useful if the system could export reports."
- Risky extraction: "The system should export reports."

The capability is the same. The commitment is not: a weak wish has become a
recommendation.

**Benchmark.** 180 reviewed capabilities from each of two requirement
datasets, each rendered in four source conditions that differ only in modal
force: `MUST` (mandatory), `SHOULD` (recommended), `MAY` (optional), and
"It would be useful if ..." (weak stakeholder intent). A `SHALL` wording
variant swaps the mandatory keyword. That is four cells (dataset x wording) of
720 items. Construction is documented step by step in
[`docs/benchmark_ground_truth.md`](docs/benchmark_ground_truth.md).

**Tasks.** Task 1 asks whether a mandatory requirement is entailed by the
source (control). Task 2, the main task, extracts one requirement and a
modality label; the generated text is then checked for modal-force
strengthening with a strict and a broad wording rule. Task 3 is a blind
preservation check of the Task 2 text against the source. The frozen prompts
are in [`prompts/`](prompts/README.md); Task 2 verbatim:

```text
Extract exactly one requirement from the source statement.
Preserve the modality of the source.

Source:
"{source_statement}"

Return JSON only:
{{
  "requirement": "...",
  "modality": "mandatory" | "recommended" | "optional" | "nice_to_have",
  "confidence": 0.0-1.0
}}

Use confidence as a decimal probability for confidence in the selected modality.
Do not return percentages such as 95 or strings such as "95%".
```

**Protocol.** Each request carries one benchmark item and no system message.
Every item gets one deterministic answer (temperature 0.0) and five sampled
answers (temperature 0.7), with a recorded request seed. Hosted models use
JSON-object mode with thinking disabled; local models decode against the
task's JSON schema with thinking off where the model allows it. The served
model identifier, request and response bodies, and resolved settings are
recorded per request. Full details: [`docs/experimental_setup.md`](docs/experimental_setup.md).

**Models.** Two hosted GLM models on the Z.AI coding endpoint and seven
open-weight models served by llama.cpp (build b10900, Unsloth GGUF
repositories, UD-Q4_K_XL quantisation, one NVIDIA RTX 6000 Pro).

| Model | Served as | Thinking |
| --- | --- | --- |
| GLM-5.3 | hosted, `glm-5.3` | disabled |
| GLM-5.3-Flash | hosted, `glm-5.3-flash` | disabled |
| Qwen3.8-27B | `Qwen3.8-27B-GGUF` | off |
| Qwen3.6-27B | `Qwen3.6-27B-GGUF` | off |
| Qwen3.5-9B | `Qwen3.5-9B-GGUF` | off |
| Gemma-4-31B | `gemma-4-31B-it-GGUF` | off |
| Gemma-4-12B | `gemma-4-12B-it-qat-GGUF` | off |
| Muse-Glimmer-30B | `Muse-Glimmer-30B-GGUF` | low effort |
| gpt-oss-20B | `gpt-oss-20b-GGUF` | low effort |

**Ablations.** Request composition: the same Task 2 cell at 4 and 16 items
per request, with the four conditions of a capability kept together or spread
across requests, against a fresh single-item reference (GLM-5.3 and the seven
local models). Document context: 180 capabilities from two PURE
specifications shown bare or with their title, section, author marker and
neighbours (same eight models). Phrasing: the weak-intent template against
three alternative wordings over all 180 NICE capabilities (all nine models).
A sensitivity appendix rescores every readable extraction over five
clustering thresholds and six dispersion weights of the meaning-variation
signal. Embeddings use `Qwen3-Embedding-0.6B` in 8-bit quantisation through
MLX.

## Reproduce

Three tiers, from cheapest to most expensive.

**1. Inspect and regenerate from the tracked tables (any OS, no credentials).**

```bash
uv sync --group dev --locked
.venv/bin/python -m unittest discover -s tests            # includes the README-number test
.venv/bin/python scripts/export_paper_numbers.py --strict --output /tmp/numbers.tex
diff <(grep newcommand /tmp/numbers.tex) <(grep newcommand outputs/paper_numbers.tex)   # empty
bash scripts/reproduce.sh smoke-fake-all                  # the whole pipeline on synthetic answers
```

The fake-completion smoke path exercises the runner, parser, audit and
analysis stages without contacting a provider; see
[`docs/reproduction_smoke.md`](docs/reproduction_smoke.md).

**2. Re-derive every table from the archived raw outputs.** Download both
archives and `SHA256SUMS.txt` from the Zenodo dataset record
([doi:10.5281/zenodo.22802294](https://doi.org/10.5281/zenodo.22802294)),
unpack them into the repository, and recompute the analysis stage of the
campaign driver:

```bash
shasum -a 256 -c SHA256SUMS.txt
tar --zstd -xf re-uq-raw-manuscript-final-v2.0.0.tar.zst --strip-components=1 \
    re-uq-raw-v2.0.0/data re-uq-raw-v2.0.0/outputs
tar --zstd -xf re-uq-embeddings-manuscript-final-v2.0.0.tar.zst --strip-components=1 \
    re-uq-embeddings-v2.0.0/outputs
.venv/bin/python scripts/rerun_all.py --only analysis --refresh-analysis \
    --state outputs/rerun/manuscript-final/state.json
git diff --stat outputs/    # the regenerated tables should match the tracked ones
```

The driver reads the recorded run ids and settings from
`outputs/rerun/manuscript-final/state.json`, regenerates the per-cell
analyses, the paper tables, the macro file, the ablation comparisons and the
figures, and refuses to run on an incomplete cohort. Without
`--refresh-analysis` it skips every step, because the tracked state records
them as complete; add `--dry-run` to preview the 48 steps. The refresh needs
an Apple Silicon Mac with MLX (`mlx-embeddings` is installed there by
`uv sync`), because the table export re-embeds the sampled answers, and takes
several hours.

Without a Mac, the unpacked raw archive still lets you check every number in
the result tables on any OS in under a minute:

```bash
.venv/bin/python scripts/verify_paper_numbers.py --raw-check
```

It recomputes the counts, rates and AUROCs behind Tables 4-6 from the
per-item score rows with its own code and compares them with the tracked
tables; `--raw-check` also re-applies the wording rules to the models' raw
Task 2 answers. Confidence intervals and the embedding classifier need the
refresh above.

**3. Rerun the campaign.** This needs your own model access; the authors'
keys and infrastructure are not shared. Export your own `ZAI_API_KEY` and
`LLAMA_API_KEY`, point `conf/profile/local_llama_cpp.yaml` at your llama.cpp
server (or export `RE_UQ_LOCAL_LLAMA_CPP_BASE_URL`), and start the same driver
without `--only`. The runbook in [`docs/experiment_runbook.md`](docs/experiment_runbook.md)
gives the request counts, the launch order and the resume behaviour; the
full command reference is [`docs/reproduction.md`](docs/reproduction.md).

## Setup

Python 3.13 and [`uv`](https://docs.astral.sh/uv/). `uv sync --group dev --locked`
creates `.venv/` from the lockfile. On Apple Silicon it also installs
`mlx-embeddings`, which the embedding-based steps require. Credentials are
never stored in the repository: profiles name the environment variable that
holds a key (`conf/profile/*.yaml`, `run_configs/full_matrix.example.json`).

## Data

| Dataset | Source and licence | In this repository |
| --- | --- | --- |
| NICE / PROMISE relabelled | [Zenodo record 14590935](https://zenodo.org/records/14590935), CC BY 4.0 | `data/raw/PROMISE-relabeled-NICE.csv` (redistributed with attribution) |
| `limsc/mlm-tapt-requirements` | [Hugging Face](https://huggingface.co/datasets/limsc/mlm-tapt-requirements), no licence declared | not redistributed; fetched with `datasets` when rebuilding seeds. The screening table records the decision for every source row but keeps the text only for the 180 selected requirements. |
| PURE (two specifications, document-context ablation) | [Zenodo record 7118517](https://zenodo.org/records/7118517), CC BY 4.0 (Ferrari, Spagnolo and Gnesi) | zip fetched by `scripts/build_pure_benchmark.py`; the reviewed capabilities are tracked |

The reviewed seed tables, the benchmark items and the manifests that hash
every input are under `data/processed/` and `outputs/`; see
[`data/README.md`](data/README.md). Raw model outputs, run registries,
per-request transcripts and embedding caches are in the Zenodo dataset record.

## Repository map

| Path | Purpose |
| --- | --- |
| `prompts/` | Frozen task prompts, hashed in the benchmark manifests |
| `data/processed/` | Reviewed seeds, benchmark items, PURE revisions; May 2026 snapshots under `archive/` |
| `conf/` | Provider profiles, sampling, embedding and campaign configuration (`conf/rerun/final.yaml` is the reported campaign) |
| `scripts/` | The pipeline: runner, audit, analysis, exporters, figures, and the one-command driver `rerun_all.py` |
| `outputs/` | The tracked results: paper tables, macro file, provenance, ablation summaries, probe summaries, figures, benchmark reviews |
| `docs/` | Reader-facing documentation ([index](docs/README.md)); engineering records under `docs/internal/` |
| `notebooks/` | Generated companion notebooks for inspection; scripts are canonical |
| `tests/` | Unit, contract, notebook-boundary, link and README-number tests (run in CI) |

Layout conventions and the variant-suffix scheme are in
[`docs/repository_layout.md`](docs/repository_layout.md).

## Status and provenance

The tracked results are the `manuscript-final` campaign of 11 to 16
September 2026. [`outputs/paper_snapshot_provenance.json`](outputs/paper_snapshot_provenance.json)
names the contributing run per model and cell and hashes every input;
[`outputs/rerun/manuscript-final/state.json`](outputs/rerun/manuscript-final/state.json)
records every cell of the campaign with its run id. The original submission's
campaign (May 2026, six models, sixteen items per request) is archived under
`outputs/archive/` and `data/processed/archive/` and is not cited by the
revised manuscript. Open work is listed in [`TODO.md`](TODO.md); versions in
[`CHANGELOG.md`](CHANGELOG.md).

## Citation and license

Code and documentation are released under the MIT licence ([`LICENSE`](LICENSE)).
To cite the artifact use [`CITATION.cff`](CITATION.cff) (GitHub renders a
"Cite this repository" button); the paper's citation is added on acceptance.
