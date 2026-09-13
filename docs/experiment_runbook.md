# Experiment rerun runbook

Use this guide to launch the prepared campaign from the repository root. The
entry point is `scripts/rerun_all.py --config conf/rerun/final.yaml`: it runs Task 1/2, audits those outputs
with Task 3, runs the configured ablations, and produces the analysis exports.
No notebook execution is required.

**Final-run gate (2026-09-11): not yet launched.** The author changed the
primary protocol to one item per request, using server concurrency for
parallelism. Batch sizes 4 and 16, both grouped and sibling-separated, are
prominent ablations. Use the explicit final configuration below, not the old
exploratory campaign state. Follow
[final-run readiness](final_run_readiness.md) before spending provider budget.

The commands below assume this checkout at `/Users/nicolas/re-uq` and macOS
with Apple Silicon and Metal GPU access, as required by the configured MLX
embedding backend. Adjust the initial `cd` if the checkout moves.

Generation itself goes over the network: the hosted profile calls Z.AI and the
local profile calls the llama-swap server on the lab GPU box, so the driver
runs on this Mac and nothing has to run on the box.

## 1. Know what will run

The configuration checked on 2026-09-11 selects:

| Setting | Prepared value |
| --- | --- |
| Campaign group | `manuscript-final`, from `conf/rerun/final.yaml` |
| Hosted profile | `zai`: `glm-5.3`, `glm-5.3-flash` |
| Local profile | `local_llama_cpp`: Qwen3.6-27B, Qwen3.8-27B, Qwen3.5-9B, Gemma4-31B, Gemma4-12B, Muse Glimmer-30B, GPT-OSS-20B; exact IDs in the profile |
| Benchmark cells | NICE/MUST, NICE/SHALL, MLM-TAPT/MUST, MLM-TAPT/SHALL |
| Items per cell | 720: 180 capabilities × four modalities |
| Main runs | Both Task 1 and Task 2, one deterministic answer plus five stochastic answers per item/task |
| Audit | Blind Task 3, tied to each recorded Task 1/2 source run |
| Batching ablation | Task 2 on MLM-TAPT/MUST: sizes 1, 4, 16; grouped and sibling-separated at sizes 4 and 16; deterministic only, single-item reference |
| Context ablation | Task 2 on revised PURE/MUST: bare versus document context, 720 single-item requests per arm; deterministic only |
| Weak-phrasing probe | Four weak templates over the 180 NICE capabilities, using the configured deterministic and stochastic sampling |
| Embeddings | `mlx-community/Qwen3-Embedding-0.6B-8bit` |
| Analysis resampling | 1,000 bootstrap samples in the driver-controlled table/comparison/embedding-diagnostic commands |

The final batching/context comparisons use `glm-5.3` and all seven local
models. Weak phrasing uses `glm-5.3`, `qwen3.6-27b`, and `muse-glimmer-30b`.
The single-item batching arm is a separate deterministic repeat, not a reuse
of the main run. Alternative embedding models are available as configurations but
are not automatically swept by this campaign.

There are 36 main Task 1/2 runs for the current nine models. Each plans 8,640 item answers
(`720 × 2 tasks × 6 answers`), or 311,040 across the main matrix. At batch size
1 that is 311,040 main-generation requests before provider probes, retries,
single-item fallbacks, audits, and ablations. This is a workload count, not a
price or duration estimate; use live smoke timings and provider usage to plan
the run.

For each model, the driver finishes its four main cells, their audits, and
its selected ablations before moving to the next model. Analysis follows all
generation. Hosted and local models retain separate table groups, but both
contribute to pooled estimates. PURE context results remain separate.

## 2. Prepare the environment and existing data

```bash
cd /Users/nicolas/re-uq
uv sync --group dev --locked
.venv/bin/python -m unittest discover -s tests -v
```

Use the locked environment. `pyproject.toml` requires Python 3.13 or newer;
`mlx-embeddings` is included on macOS ARM64. Activation is optional because all
commands explicitly use `.venv/bin/python`.

The prepared benchmark files should already exist:

```text
data/processed/benchmark_items.csv                 NICE/MUST
data/processed/benchmark_items_shall.csv           NICE/SHALL
data/processed/benchmark_items_mlm_tapt.csv         MLM-TAPT/MUST
data/processed/benchmark_items_mlm_tapt_shall.csv   MLM-TAPT/SHALL
data/processed/benchmark_items_pure.csv            PURE/MUST context ablation
```

Keep the reviewed seed tables, benchmark manifests, prompts, and
`docs/weak_modality_construct_review.csv` with these files. The driver does
not rebuild missing datasets. The weak probe requires the template construct
review to pass; the analysis also checks benchmark integrity and review
evidence. See [validation status](validation_review.md) and
[benchmark ground truth](benchmark_ground_truth.md). Do not regenerate the
reviewed inputs as a routine launch step.

## 3. Set credentials and start the local endpoint

Run these commands in the same terminal that will launch the experiments.
This zsh input form avoids putting the key itself into shell history:

```zsh
read -rs 'ZAI_API_KEY?Z.AI API key: '
echo
export ZAI_API_KEY
```

For an authenticated local server:

```zsh
read -rs 'LLAMA_API_KEY?Local server client API key: '
echo
export LLAMA_API_KEY
```

If your local server has authentication disabled, use a nonempty placeholder
instead; the driver still requires the variable to be set:

```bash
export LLAMA_API_KEY=local-no-auth
```

`ZAI_API_KEY` authenticates generation against Z.AI. `LLAMA_API_KEY` is the
client key accepted by your local chat server. It is not an embedding-model
credential. Store only the variable **names** in YAML; the repository does
not automatically load a `.env` file for this workflow.

The local profile points at the llama-swap server on the lab GPU box
(`ssh GPU`), published at `http://141.3.52.248:9292/v1` and reachable from
this Mac. It loads a model on the first request that names it and unloads it
after 15 idle minutes, so the first request of a run (and the first after a
pause) waits for the load; the profile's 300 s timeout covers a reload from
the box's cache. The exact request model ID is `qwen3.6-27b`; `GET /v1/models`
with the key lists what the box currently offers. If you list several local
models, the router loads them by request model name without manual
intervention.

Both keys are kept in `~/.zshenv` on the box. To use them from this Mac
without typing them, pull them into the launching shell:

```zsh
export LLAMA_API_KEY="$(ssh GPU 'source ~/.zshenv; printf %s "$LLAMA_API_KEY"')"
export ZAI_API_KEY="$(ssh GPU 'source ~/.zshenv; printf %s "$ZAI_API_KEY"')"
```

## 4. Review and freeze the configuration

The campaign reads **`conf/`**, with these responsibilities:

| File | What to set |
| --- | --- |
| `conf/profile/zai.yaml` | Hosted endpoint, exact model IDs, API-key variable name, concurrency, batching, timeout, output-token budget, retries, structured output, request seed |
| `conf/profile/local_llama_cpp.yaml` | Local endpoint and model IDs, local client-key variable name, the same request controls |
| `conf/rerun/default.yaml` | Campaign group, datasets, variants, hosted/local profile membership, ablation representatives, embedding group, audit mode, analysis settings |
| `conf/sampling/default.yaml` | Deterministic and stochastic temperatures, `top_p`, sample counts |
| `conf/config.yaml` | Base prompt version, run seed and other Hydra defaults |
| `conf/embedding/qwen3_06b.yaml` | Default neural embedding backend and model |
| `conf/logging/default.yaml` | Progress cadence and warning thresholds; the campaign forces progress, events and transcripts on |

For the normal rerun, review the endpoint/model lists in the two profiles and
leave the prepared protocol settings in place:

| Request setting | Z.AI | Local |
| --- | --- | --- |
| `base_url` | `https://api.z.ai/api/coding/paas/v4` | `http://141.3.52.248:9292/v1` |
| `api_key_env` | `ZAI_API_KEY` | `LLAMA_API_KEY` |
| `concurrency` | 2 | 8 (headline runs used 2 workers on 4 server slots) |
| `batch_size` / `batch_order` | 1 / `grouped` | 1 / `grouped` |
| `timeout_s` | 180 | 300 |
| `max_tokens` | 256 per item | 1024 per item (historical Qwen3.6 headline used 512) |
| `max_retries` | 3 | 3 |
| `seed` / `send_seed` | 20260518 / `true` | 20260518 / `true` |
| `json_mode` / `structured_output` | `true` / `json_object` | `false` / `json_schema` |
| `extra_body` | `thinking: disabled`, `response_format: json_object` | `chat_template_kwargs.enable_thinking: false`, `reasoning_effort: low` |

The Z.AI profile is configured for the Coding Plan endpoint. If your account
uses the standard paid API, the profile's documented alternative is
`https://api.z.ai/api/paas/v4`; select the endpoint supported by your account.
The hosted profile requests `thinking: disabled`. The local profile requests
disabled thinking where supported; Muse and GPT-OSS use low reasoning effort
and can still generate reasoning tokens. Qwen3.6 uses the chat template's
`enable_thinking: false`; without it, observed requests exhausted their budget on
reasoning and returns an empty answer. Z.AI also requests JSON there.

Z.AI lists more model IDs than it serves under their own name: on 2026-09-10
requests for `glm-4.5-air`, `glm-4.7` and `glm-5-turbo` came back from
`glm-5.3-flash`, and `glm-5`, `glm-5.1` and `glm-5.2` from `glm-5.3`. The
profile therefore lists only `glm-5.3` and `glm-5.3-flash`. After any smoke
cell, compare `served_model` in the transcript with the requested ID.

The batched runner multiplies `max_tokens` by the number of items: a full
16-item ablation request permits 4,096 output tokens on Z.AI and 16,384 locally.
Single-item primary requests permit 256 and 1,024 respectively. Ensure
the local server can accommodate the prompt and this output allowance.
If smoke results truncate, inspect the transcript and server limits before
changing the profile. The batching ablation pins its size/composition sweep;
both context arms explicitly pin size 1. Changing settings after a cell starts
requires a new campaign, not an in-place resume of historical grouped runs.

Sampling is temperature `0.0`, `top_p: 1.0`, one deterministic sample, and
temperature `0.7`, `top_p: 1.0`, five stochastic samples. Stochastic repetitions
use distinct seeds; reproducibility still depends on provider seed support.

Changing a profile's model order changes its default ablation representative.
To pin representatives, replace `profiles:` inside each relevant ablation
with an explicit `models:` list, for example:

```yaml
models:
  - profile: zai
    model: glm-5.3
  - profile: local_llama_cpp
    model: qwen3.6-27b
```

These models must also be listed in their selected cohort profiles. To omit
local generation, set `local_profiles: []` **and** remove the local profile
from every ablation's `profiles:` or explicit `models:` list.

`run_configs/current_run.json` and the `RE_UQ_PROFILE`/`RE_UQ_MODEL` variables
used by `scripts/reproduce.sh` do not configure this driver. It generates
`outputs/rerun/run_config.json` from `conf/`; do not edit that generated file.
Likewise, Hydra overrides such as `profile=zai` belong to `scripts/run.py`,
not `scripts/rerun_all.py --config conf/rerun/final.yaml`.

## 5. Check the plan and execution before the full launch

First print the whole plan without credentials, provider calls, or campaign
artifact writes:

```bash
.venv/bin/python scripts/rerun_all.py --config conf/rerun/final.yaml --dry-run
```

Check that the model list, dataset/variant combinations, ablation arms,
embedding model and output locations match your intended campaign.

Then exercise the synthetic chain:

```bash
.venv/bin/python scripts/rerun_all.py --config conf/rerun/final.yaml --fake-completion --smoke-items 8
```

This makes no provider calls. Raw records go to `data/processed/smoke/`;
campaign state, config and analysis go under `outputs/smoke/`. Per-run logs
use smoke run IDs. It uses TF-IDF embeddings and skips the neural embedding
diagnostic/figures, headline aggregation and LaTeX numbers export. It verifies
wiring, not model behavior or the complete neural analysis.

With keys set and the local server running, check the real prerequisites:

```bash
.venv/bin/python scripts/rerun_all.py --config conf/rerun/final.yaml --only preflight
```

This checks key presence, benchmark-file presence, profile constraints, and,
when the analysis stage is among the requested stages, MLX import/Metal
availability. It writes the generated config but does not send chat requests. It does **not** validate credentials, model availability,
or downloading/loading the embedding weights.

Run a small **real** smoke cell on each endpoint next. These commands consume
provider usage and may download the configured embedding model on first use:

```bash
for model in glm-5.3 glm-5.3-flash; do
  .venv/bin/python scripts/run.py \
    profile=zai model=$model dataset=mlm_tapt variant=must \
    task=both mode=smoke smoke_items=16 embedding=qwen3_06b \
    run_group_id=provider-smoke-2026-09
done

.venv/bin/python scripts/run.py \
  profile=local_llama_cpp model=qwen3.6-27b \
  dataset=mlm_tapt variant=must task=both mode=smoke smoke_items=16 \
  embedding=qwen3_06b run_group_id=provider-smoke-2026-09
```

Sixteen items exercise concurrent single-item requests. Also smoke-test the
size-4 and size-16 ablation overrides before the full sweep. For each command,
inspect the reported run ID, parse failures, truncation, retries and request
transcript. Repeat with the remaining hosted model IDs before relying on the
whole model list. Smoke rows remain separate and do not satisfy full campaign
cells. First model loading and embedding downloads can take longer than
subsequent requests.

## 6. Launch the experiments

Once smoke checks pass and the configuration is fixed:

```bash
.venv/bin/python -u scripts/rerun_all.py --config conf/rerun/final.yaml
```

On this Mac, you can instead keep the machine awake while the command runs:

```bash
caffeinate -i .venv/bin/python -u scripts/rerun_all.py --config conf/rerun/final.yaml
```

Keep the terminal and local server running. Use one driver at a time in this
checkout: generated configs, some reports, and stage-log destinations are
shared even between campaign groups.

To separate generation from analysis, use these two commands in order:

```bash
.venv/bin/python -u scripts/rerun_all.py --config conf/rerun/final.yaml \
  --only preflight --only cohort --only task3 --only ablations

.venv/bin/python -u scripts/rerun_all.py --config conf/rerun/final.yaml --only analysis
```

`--only` is repeatable and does not automatically add prerequisites. Analysis
requires all configured main runs, audits and ablations to be complete in the
same state file. Running `--only cohort` alone is therefore not enough for a
later complete analysis.

The generation command needs no Metal, so it can run unattended on the GPU
box from a synced copy of this checkout (`~/projects/re-uq-rerun`, built with
`uv sync --group dev --locked --python /usr/bin/python3.13`) inside `tmux`.
The 2026-09 campaign was started that way. Afterwards copy
`data/processed/` (without `smoke/`), `data/processed/logs/` and
`outputs/rerun/<run_group_id>/state.json` back to this Mac and run the
analysis command here.

The hosted and the local endpoint are independent, so their generation can
run at the same time as one driver per profile, each in its own `tmux`
session, against the same state file (records merge per cell under a lock):

```bash
.venv/bin/python -u scripts/rerun_all.py --config conf/rerun/final.yaml --profile zai \
  --only preflight --only cohort --only task3 --only ablations

.venv/bin/python -u scripts/rerun_all.py --config conf/rerun/final.yaml --profile local_llama_cpp \
  --only preflight --only cohort --only task3 --only ablations
```

Each driver reports and retries only its own profile's cells. Both write the
same generated run config, which is identical, and append to the same
per-dataset raw files and registries, which the runners already serialise
with file locks. Run the analysis stage afterwards without `--profile`.

`conf/rerun/headline.yaml` is an exploratory plan. Its historical outputs used
grouped requests; they are not single-item main evidence. Driver states do not
merge automatically, even when run groups match. Use the recorded state only
for an explicitly selected historical analysis refresh.

For a separate campaign plan, copy and edit the rerun YAML, then use its path
consistently for launch, monitoring and resume:

```bash
cp conf/rerun/final.yaml conf/rerun/my_campaign.yaml
# Edit my_campaign.yaml, including a distinct run_group_id, before starting.
.venv/bin/python scripts/rerun_all.py --config conf/rerun/my_campaign.yaml --dry-run
.venv/bin/python -u scripts/rerun_all.py --config conf/rerun/my_campaign.yaml
```

This still reads the shared profiles, sampling and embedding files in `conf/`.

## 7. Monitor progress

The driver prints its current model/cell and appends stage output to
`outputs/rerun/logs/`. To list campaign status and discover exact run IDs in a
second terminal:

```bash
cd /Users/nicolas/re-uq
.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path('outputs/rerun/manuscript-final/state.json')
if not path.exists():
    print('No campaign state yet; inspect the launching terminal.')
else:
    state = json.loads(path.read_text())
    for name, cell in state['cells'].items():
        print(cell['status'], name, cell.get('run_id', ''), sep='\t')
PY
```

Replace the group in that path if you changed it. For a Task 1/2 cell, copy
its run ID and use the progress reporter (replace `RUN_ID`):

```bash
.venv/bin/python scripts/show_run_progress.py \
  --dataset mlm_tapt --variant must \
  --profile zai --model glm-5.3 --run-id RUN_ID --watch 30
```

Match the dataset, variant, profile and model to the selected cell. For Task
3 and weak probes, use their stage/per-run logs and dedicated registry and
progress files. The Task 1/2 progress reporter is not their monitor.

## 8. Interrupt, recover and resume

To stop, press Ctrl-C in the driver terminal. Restart the same command with
the **same configuration and state path**:

```bash
.venv/bin/python -u scripts/rerun_all.py --config conf/rerun/final.yaml
```

Complete cells are skipped. Unfinished cells resume their recorded run IDs;
valid completed answers are reused, while failed transport/parsing attempts
can be requested again. The main/audit/ablation cell runner also tries one
automatic resume after a failure, separate from per-request retries. The weak
probe resumes when the driver is invoked again. Analysis failures stop later
analysis steps that depend on their output.

| Symptom | Action |
| --- | --- |
| Missing key | Export the named variable in the launching shell; rerun preflight. |
| Connection refused / model unavailable | Start or repair the server and confirm the exact model ID and URL; repeat the real smoke cell. |
| First local request slow or timed out | llama-swap is loading `qwen3.6-27b` after an idle unload; wait for the load or send one request by hand, then resume. |
| Authentication or rate-limit errors | Inspect the provider error in the transcript; correct credentials or wait for quota, then resume. |
| Truncated responses / repeated parse failures | Inspect `truncated_records`, response bodies and actual token limits. Test a corrected profile with smoke before starting a new campaign configuration. |
| MLX/Metal failure | Refresh the locked environment and use an Apple Silicon session with Metal access; inspect model loading/download errors in the per-run log. |
| Manifest or construct-review failure | Restore/resolve the reviewed inputs and evidence; do not bypass the checks for paper results. |
| State belongs to a different configuration | Preserve the old state. Give a changed experiment a new `run_group_id`, or deliberately select a fresh `--state` path. |
| Analysis reports a partial rerun | Complete the missing cells it lists by rerunning the full driver with the original state. |

A new group/state starts fresh work and can repeat provider usage; it is not
a resume shortcut. Do not delete raw outputs or edit state entries to make a
failed cell appear complete. Completed analysis steps are also skipped on
resume: `--only analysis` continues pending work, not a forced recomputation
of already completed exports. Use the exact per-step command recorded in
the stage log if you deliberately need to regenerate an existing analysis.

## 9. Find and check the results

| Artifact | Location |
| --- | --- |
| Campaign state and resolved plan | `outputs/rerun/<run_group_id>/state.json` |
| Generated legacy-tool config | `outputs/rerun/run_config.json` |
| Stage commands, stdout/stderr, exit status | `outputs/rerun/logs/*.log` |
| Per-run logs, transmitted requests/responses, resolved YAML | `data/processed/logs/<safe_run_id>.*` |
| Task 1/2 answers and attempts | `data/processed/model_outputs_raw*.jsonl` plus the compacted `*.parquet` sibling |
| Audit answers | `data/processed/model_outputs_raw_task3_verification*.jsonl` plus the compacted `*.parquet` sibling |
| Registries, events and progress | `data/processed/run_registry*.csv`, `run_events*.jsonl`, `run_progress_live*.csv` |
| Per-cell analysis and provenance | `outputs/evaluation_<dataset>_<variant>_<safe_run_id>/` |
| Selected embedding-cache manifest | `outputs/rerun/acse_selected_manifest.csv` |
| Embedding diagnostic | `outputs/embedding_diagnostic/` |
| Candidate figures | `outputs/rerun/figures/` |
| Main paper tables and headline metrics | `outputs/paper_*.csv` |
| Batching and context comparisons | `outputs/batching_ablation_summary*`, `outputs/context_ablation_summary*` |
| Per-run weak-phrasing summaries | `outputs/weak_modality_probe/<safe_run_id>/` |
| Candidate manuscript macros | `outputs/paper_numbers.tex` |

Raw files are append-only JSONL while runs are in flight. After a campaign, run
`.venv/bin/python scripts/compact_raw_store.py` (or `--dry-run` first) to move
the finished rows into zstd Parquet siblings, about 40x smaller; the JSONL is
left as an empty tail for later appends and every reader returns both halves.
Do not compact while a runner is appending to the same file.

**The real analysis refreshes shared paper snapshots in `outputs/`.** A
different campaign group isolates state and run selection, but does not give
every export a new directory. Preserve any prior exports you need to compare
before running analysis. Raw files contain multiple runs/attempts; use the
recorded IDs and deduplicated metrics, not raw line counts as sample counts.

Completion means the full command exits successfully, all planned cells are
complete, and the real analysis gates pass. Then inspect the plots and tables,
counts/denominators, missing values, parse failures, confidence/stability
results and selected-run provenance. Repeat the author review of generated
outputs described in [validation status](validation_review.md), including
flagged, unflagged, mixed-cue and unreadable cases.

The driver leaves `manuscript/numbers.tex` unchanged. After reviewing the
candidate exports, compare before intentionally integrating the macros:

```bash
diff -u manuscript/numbers.tex outputs/paper_numbers.tex
# Only after reviewing the new results:
cp outputs/paper_numbers.tex manuscript/numbers.tex
```

`diff` exits with status 1 when files differ. Candidate figures likewise need
review and deliberate integration. Pipeline success alone does not make the
results ready for publication.

For individual-run commands, analysis flags and publication checks, see
[reproduction](reproduction.md). For Hydra composition and alternate
configurations, see [configuration](configuration.md).
