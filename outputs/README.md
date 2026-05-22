# outputs/

Curated paper-facing summaries and review artifacts.

## Tracked here

- **Benchmark manifests** (`benchmark_manifest*.json`) with SHA-256 hashes of every seed, prompt, and benchmark CSV consumed by a run.
- **Benchmark statement reviews** (`benchmark_statements_review*.csv` and matching `.md`) — the human-readable review of generated variants per dataset and modality variant.
- **Included-capabilities reviews** (`included_capabilities_review*.csv` and matching `.md`).
- **Pilot and probe summaries**:
  - `pilot_results_summary.md`
  - `prompt_sensitivity_summary.csv`, `task2_prompt_sensitivity_summary.csv`
  - `weak_modality_probe_summary.csv` / `.md`
  - `weak_modality_template_sanity_check.csv` / `.md`
  - `logprob_probe.json`
- **External AI service probe** results under `outputs/external_ai_service_probe/` (evaluation Markdown, confusion matrices, source-condition summaries, comparison).

## Local-only

Run-specific analysis directories (`outputs/evaluation_<dataset>_<variant>_<run_id>/`), capability-seed agent-repair artifacts, and any preliminary tables are gitignored. They are reproducibility evidence and are promoted to Git only when they become paper-facing.

See `.gitignore` for the precise allow-list, and `docs/repository_layout.md` for the variant-suffix convention used across these files.
