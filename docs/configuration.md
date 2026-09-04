# Configuration

There are two ways to configure a run, and they end up in exactly the same
place: the normalized dictionary produced by
`eval_utils.normalize_run_config()`. Nothing downstream of that function knows
or cares which path you used.

| Path | Entry point | Status |
| --- | --- | --- |
| JSON run config | `scripts/run_experiment_from_config.py --config run_configs/*.json` | Legacy, unchanged, **the provenance record for the published runs** |
| Hydra composition | `scripts/run.py <overrides>` | Recommended for new experiments: composition, overrides, sweeps |

The JSON path is frozen on purpose. Every number in the paper was produced with
it, and it keeps working byte-for-byte as documented in
[`reproduction.md`](reproduction.md). The Hydra layer is additive: it builds the
same dictionary from composable YAML groups and then calls the same runner
functions.

## 1. Quick start

```bash
uv sync --group dev                      # installs hydra-core
# one cell
.venv/bin/python scripts/run.py profile=zai model=glm-5.1 dataset=nice variant=must mode=full
# no credentials needed: synthesize completions locally into data/processed/smoke/
.venv/bin/python scripts/run.py profile=zai model=glm-5.1 mode=smoke fake_completion=true
# what would run, without contacting a provider
.venv/bin/python scripts/run.py profile=zai model=glm-5.1 dry_run=true
# print the composed config and exit
.venv/bin/python scripts/run.py profile=openai --cfg job
```

`scripts/reproduce.sh hydra <overrides>` forwards to the same entry point.

Hydra never changes the process working directory (`hydra.job.chdir: false`).
Every path in this project resolves through `eval_utils.project_root()`, which
walks up from the working directory, so always run from the repository root.
Hydra's own job artifacts go to `outputs/hydra/<date>/<time>/` (single run) and
`outputs/hydra/multirun/<date>/<time>/<job>/` (sweep). They contain Hydra's
bookkeeping only; the run artifacts stay in `data/processed/` exactly as before.

## 2. Group layout

```
conf/
  config.yaml              defaults list + run-level and execution fields
  profile/                 one file per provider profile (the paper's provider matrix)
    local_llama_cpp.yaml  institutional_llm.yaml  zai.yaml  openai.yaml
    mistral.yaml          google_gemini.yaml      ollama_local.yaml
    kit_toolbox.yaml
  dataset/                 nice.yaml  mlm_tapt.yaml
  variant/                 must.yaml  shall.yaml
  sampling/                default.yaml  deterministic_only.yaml
  logging/                 default.yaml
  embedding/               qwen3_06b.yaml (default), qwen3_4b.yaml, bge_m3.yaml,
                           multilingual_e5_large.yaml, embeddinggemma_300m.yaml,
                           tfidf_proxy.yaml
  experiment/              paper_cohort.yaml  batching_ablation.yaml  context_ablation.yaml
                           diverse_families.yaml
```

The eight `conf/profile/*.yaml` files mirror
`run_configs/full_matrix.example.json` entry for entry. Every profile is an
OpenAI-compatible chat-completions endpoint (`base_url` + `api_key_env`);
that compatibility layer is the only provider integration the pipeline has,
so a new family is a new profile file, nothing more. Each carries the
per-profile knobs `seed`, `send_seed`, `max_retries`, `batch_order`,
`batch_size`, `concurrency`, `timeout_s`, `max_tokens`, `json_mode`,
`structured_output`, `extra_body`, `requires_manual_server` and `notes`. A test
(`tests/test_hydra_bridge.py`) asserts that composing each profile yields the
same normalized profile as loading the JSON file, so the two stay in sync.

Top-level fields in `conf/config.yaml`:

| Field | Meaning |
| --- | --- |
| `run_group_id`, `prompt_version`, `seed`, `batch_order` | Run-level provenance and defaults, same keys as the JSON config |
| `item_context` | `bare` (the paper condition) or `document`: show each Task 2 item inside its document context. Run-level only; `document` requires `dataset=pure` and `task=task2` ([`context_ablation.md`](context_ablation.md)). Recorded on every raw row and in the registry. |
| `acse_embedding_backend`, `acse_embedding_mlx_model` | Resolved `embedding=` selection, persisted on raw run rows for later analysis and cache generation |
| `model` | Model id to run. `null` runs every model of the selected profile sequentially (the `--all-models` behaviour) |
| `task` | `task1` \| `task2` \| `both` \| `task3`; `task3` dispatches to the Task 3 runner |
| `mode` | `smoke` \| `full` \| `resume` (`resume` requires `run_id`) |
| `run_id` | Explicit run id; required for `mode=resume` |
| `smoke_items` | Benchmark items per cell in `mode=smoke` |
| `fake_completion` | Synthesize completions locally; never reads a key, always writes to `data/processed/smoke/` |
| `dry_run` | Print planned job/batch/API-call counts and exit without provider calls or artifact writes, including for Task 3 |
| `log_level` | Level for the `re_uq` logger |
| `source_run_id`, `audit_mode`, `allow_partial_source` | Task 3 only |
| `allow_source_profile_mismatch` | Task 3 only: audit Task 2 rows written under a *different* provider profile. Off by default (the run fails instead); when on, the audited source profile is recorded in the registry `notes` |

### Secrets

Config files hold `api_key_env`, the **name** of the environment variable that
carries the key, never a key. The name itself is indirected through
`${oc.env:RE_UQ_<PROFILE>_API_KEY_ENV,<default>}`, so you can point a profile at
a different variable without editing the file:

```bash
RE_UQ_ZAI_API_KEY_ENV=MY_TEAM_ZAI_KEY .venv/bin/python scripts/run.py profile=zai model=glm-5.1
```

## 3. Override syntax

```bash
# select a config group
profile=openai  dataset=mlm_tapt  variant=shall  sampling=deterministic_only
# override a single value inside a group (dotted path)
profile.batch_order=shuffled  profile.batch_size=1  profile.concurrency=1
# override a top-level field
mode=full  task=task2  log_level=DEBUG  smoke_items=4
# add a field that is not in the config
+profile.extra_body.reasoning_effort=low
```

`model=` is validated against the selected profile. A model id the profile does
not list is rejected up front, with the profile's valid ids in the error, rather
than being sent to an endpoint that does not serve it. Select the profile that
owns the model first — `profile=kit_toolbox model=kit.gemma4-31b-it` — or leave
`model` at `null` to run every model of the profile.

`profile.batch_order=shuffled` is a *constrained* shuffle: it never places two
source variants of one seed in the same batch, and the permutation is derived
deterministically from the run seed, so the arm is reproducible and a resume
keeps the original batch membership. Pair it with `profile.batch_size=1` for the
true single-item arm.

Example — the batching ablation arm for one cell:

```bash
.venv/bin/python scripts/run.py \
  profile=zai model=glm-5.1 dataset=mlm_tapt variant=must \
  task=task2 sampling=deterministic_only profile.batch_order=shuffled mode=full
```

## 4. Sweeps

Add `--multirun` (`-m`) and give comma-separated values. The default (basic)
launcher runs the cells **sequentially** in one process, which is what you want
against a rate-limited provider:

```bash
.venv/bin/python scripts/run.py --multirun \
  profile=zai model=glm-5.1,glm-4.7 dataset=nice,mlm_tapt variant=must,shall \
  fake_completion=true mode=smoke
```

That is 2 x 2 x 2 = 8 cells; with `fake_completion=true` every cell lands in the
smoke tree (`data/processed/smoke/`) and no provider is contacted. Drop
`fake_completion` and set `mode=full` for a real sweep.

Sweeping across profiles only makes sense when the swept models exist in each
profile, because `model=` selects the model for the *selected* profile — and the
model id is validated against it, so a cross-profile sweep over a model list
fails fast instead of running the wrong cells. For a cohort spread over several
endpoints (the official cohort is `zai` plus `kit_toolbox`), run one sweep per
profile.

## 5. Experiment presets

`conf/experiment/*.yaml` are ready-made override bundles (`@package _global_`),
applied last and opted into with a leading `+`:

```bash
.venv/bin/python scripts/run.py --multirun +experiment=paper_cohort
.venv/bin/python scripts/run.py --multirun +experiment=batching_ablation
.venv/bin/python scripts/run.py --multirun +experiment=context_ablation
.venv/bin/python scripts/run.py --multirun +experiment=diverse_families
```

| Preset | What it pins |
| --- | --- |
| `paper_cohort` | Official cohort: `profile=zai`, all five GLM models (`glm-4.5-air,glm-4.7,glm-5,glm-5-turbo,glm-5.1`), datasets `nice,mlm_tapt`, both variants `must,shall`, Task 1 + Task 2, `mode=full`. The one non-GLM official model lives on `profile=kit_toolbox` and runs as a separate invocation (see the file header). |
| `batching_ablation` | [`TODO.md`](../TODO.md) section A: `mlm_tapt`/`must`, Task 2, deterministic pass only, sweeping `profile.batch_order=grouped,shuffled` with `profile.batch_size` pinned to 16 so the two arms differ only in batch membership. The `batch_size=1` arm and the `kit.gemma4-31b-it` half of the cohort are one extra override each, spelled out in the file header. |
| `context_ablation` | [`TODO.md`](../TODO.md) section B, minimal version: `pure`/`must`, Task 2, deterministic pass only, grouped 16-item batches, sweeping the run-level `item_context=bare,document`. Own `run_group_id` (`context-ablation-2026-09`) so the arms can never be selected into paper tables; the `kit.gemma4-31b-it` half is one extra override (file header). Table: `scripts/compare_context_ablation.py`; design and reading guide in [`context_ablation.md`](context_ablation.md). |
| `diverse_families` | [`TODO.md`](../TODO.md) section C: `openai`, `mistral`, `google_gemini`, `ollama_local`, every model of each profile, both datasets, variant `must`. All OpenAI-compatible endpoints; adding a family without one is out of scope. |

Presets compose with further overrides, e.g.
`+experiment=batching_ablation profile.batch_size=1`.

## 6. Resolved-config provenance

Every run launched through `scripts/run.py` writes its fully resolved
composition (`OmegaConf.to_yaml(cfg, resolve=True)`, credential-shaped values
masked) next to the run log:

```
data/processed/logs/<run_id>.resolved.yaml
```

and appends the file's SHA-256 to the run-registry `notes` column:

```
mode=smoke; resolved_config_sha=4cd38b5d…
```

Task 3 rows keep their existing `audit_mode=…; source_run_id=…` fields and
append the digest after them. `RUN_REGISTRY_FIELDS` is unchanged, so every
existing consumer of the registry keeps working. Runs launched from the JSON
CLI write no resolved config and their `notes` value is byte-identical to
before.

Verify a run's configuration after the fact with:

```bash
shasum -a 256 data/processed/logs/<run_id>.resolved.yaml
```

## 7. Migrating `current_run.json`

`scripts/hydra_bridge.py` exports an existing JSON run config into the `conf/`
groups:

```bash
.venv/bin/python scripts/hydra_bridge.py --config run_configs/current_run.json
# or: scripts/reproduce.sh hydra-export
```

It writes:

- `conf/profile/<profile_id>.yaml` for every profile in the JSON file,
- `conf/sampling/<name>.yaml` and `conf/logging/<name>.yaml`,
- `conf/experiment/<name>.yaml`, a preset that pins the run-level fields and
  sweeps the config's profile / dataset / variant matrix. It sets `model: null`,
  so each profile job runs only the models listed by that profile.

`<name>` defaults to the JSON file stem (`current_run`); override it with
`--name`. Existing files are only replaced with `--overwrite`, and the exporter
refuses to clobber a file whose content differs. Then:

```bash
.venv/bin/python scripts/run.py --multirun +experiment=current_run
```

The export goes through `load_run_config()`, so the YAML contains the
*normalized* profile (defaults made explicit). Sweeping profiles rather than a
union of model ids preserves every profile/model pairing while keeping the
single command above. A test asserts the round trip: composing the exported
groups reproduces the JSON config's normalized profiles, sampling blocks,
logging thresholds and run-level fields exactly.

## 8. Which path should I use?

- Reproducing a published number: the JSON path, as documented in
  [`reproduction.md`](reproduction.md).
- Anything new — ablations, new providers, sweeps: the Hydra path, and keep the
  `resolved_config_sha` with the results.
