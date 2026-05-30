# Repository hygiene

This project is a research-engineering artifact: Git should preserve code, prompts, benchmark definitions, reviewed seeds, manifests, and compact paper-facing summaries. Local runs may produce much larger raw or row-level artifacts, but those should stay local unless they are explicitly promoted for a paper or reproducibility reason.

## Git workflow

- Work on a branch for every coherent change. Use `codex/<topic>` for Codex-driven branches.
- Keep commits narrow enough to review: separate code/notebook behavior, artifact policy, and infrastructure changes.
- Stage intentionally. Prefer path-specific `git add` and inspect `git diff --cached --stat` before committing.
- Do not commit local secrets or machine-specific settings. Use `config.example.json` as the tracked template and keep `config.json` local.

## Artifact policy

- Track durable inputs and curated artifacts: prompts, stripped notebooks, tests, benchmark item CSVs, selected/reviewed seed files, benchmark manifests, final seed documents, and compact paper-facing summaries.
- Keep raw and run-level outputs out of Git by default: `model_outputs_raw*.jsonl`, run registries/progress files, `uq_scores*.csv`, Task 3 item CSVs, provider matrix current-run configs, generated `outputs/evaluation_*` directories, and scratch files under `tmp/`.
- If a generated output becomes paper-facing, promote it deliberately in a small commit whose message explains why it belongs in the repository.
- Remove accidental tracked generated files with `git rm --cached <path>` so the local file is preserved.
- Keep local manuscript drafts and assistant-orchestration scripts out of normal commits. In particular, `docs/paper_draft.*`, `o.sh`, and `*.local.sh` are ignored because they commonly contain author identity, local paths, or one-off prompt bundles. Promote an anonymized manuscript deliberately with `git add -f` only when packaging a publication artifact.

## Local verification before commit

Run these checks before pushing or asking for review:

```bash
git diff --check
.venv/bin/python -m unittest discover -s tests -v
```

The unit suite includes notebook boundary tests that verify checked-in notebooks match `scripts/populate_notebooks.py` and contain no stored execution outputs.

For a quick working-tree hygiene audit before committing, run:

```bash
git status --short
git status --ignored --short
git ls-files outputs data/processed docs | sort
```

The first command should be empty except for intentional tracked edits. The ignored-status view should contain local run outputs and caches, not files that you intend to publish. Run the content preflight in `docs/anonymization.md` before sharing an artifact; it only checks tracked files, so inspect any ignored or untracked manuscript candidate separately before forcing it into Git.

## Publication release checklist

Before tagging or archiving a publication artifact:

- Confirm `README.md`, `docs/evaluation.md`, `docs/reproduction.md`, and `docs/results_mapping.md` agree on Task 1/2 as the primary experiment and Task 3 as a diagnostic.
- Run the command-first reproduction path in `docs/reproduction.md` or document exactly which provider cells could not be rerun.
- Complete `docs/weak_modality_construct_review.csv` before making weak-intent paper claims.
- Generate final analysis with `scripts/generate_evaluation_analysis.py` and inspect the exported table, figure, qualitative examples, and provenance manifest.
- Audit tracked files with `git ls-files` and ignored local outputs with `git status --ignored --short` before committing curated artifacts.
- Treat raw JSONL outputs as local reproducibility evidence unless there is a deliberate archival reason to promote them.
