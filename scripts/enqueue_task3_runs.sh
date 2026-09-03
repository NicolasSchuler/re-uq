#!/usr/bin/env bash
set -euo pipefail

# Enqueue official blind Task 3 text-audit runs for completed Task 1/2 source runs.
#
# Profiles, models, datasets, and benchmark variants come from the run config
# (no hand-maintained matrix). For every config cell the newest complete
# `full-*` Task 2 run matching the configured group, sampling, and batching plan
# is used as the Task 3 source.
#
# Private/institutional model rows are intentionally omitted from the official
# cohort. They remain in the raw registries as diagnostic attempts, but should
# not be queued for official blind Task 3 or counted in paper-facing results.
#
# Overrides:
#   RE_UQ_CONFIG          run config path (default: run_configs/current_run.json)
#   TASK3_AUDIT_MODE      blind | declared_text | declared_source (default: blind)
#   TASK3_SKIP_PROFILES   comma-separated profile ids to skip (default: institutional_llm)
#   TASK3_DRY_RUN         1 to print the pueue commands instead of enqueuing them

cd "$(dirname "$0")/.."

CONFIG="${RE_UQ_CONFIG:-run_configs/current_run.json}"
AUDIT_MODE="${TASK3_AUDIT_MODE:-blind}"
SKIP_PROFILES="${TASK3_SKIP_PROFILES:-institutional_llm}"
PY=".venv/bin/python"

if [[ ! -f "$CONFIG" ]]; then
  echo "error: run config not found at $CONFIG" >&2
  exit 2
fi

# Emits one "dataset<TAB>variant<TAB>profile<TAB>model<TAB>source_run_id" line
# per config cell that has a complete full Task 2 source run.
resolve_cells() {
  RE_UQ_TASK3_CONFIG="$CONFIG" RE_UQ_TASK3_SKIP="$SKIP_PROFILES" "$PY" - <<'PY'
import os
import sys

sys.path.insert(0, "scripts")
import eval_utils as eu

config = eu.load_run_config(os.environ["RE_UQ_TASK3_CONFIG"])
skip = {value.strip() for value in os.environ.get("RE_UQ_TASK3_SKIP", "").split(",") if value.strip()}
root = eu.project_root()

for dataset_id in config["datasets"]:
    for variant in config["benchmark_variants"]:
        prefix = "full" if variant == "must" else f"full-{variant}"
        benchmark_path = eu.artifact_path(root / "data/processed/benchmark_items.csv", dataset_id, variant)
        benchmark = eu.read_csv_rows(benchmark_path)
        registry_path = eu.run_registry_path(root, dataset_id, variant)
        registry_rows = eu.read_csv_rows(registry_path) if registry_path.exists() else []
        for profile in config["profiles"]:
            if profile["profile_id"] in skip:
                continue
            for model in profile["models"]:
                candidates = [
                    row
                    for row in registry_rows
                    if str(row.get("profile_id", "")) == profile["profile_id"]
                    and str(row.get("model", "")) == model
                    and eu.run_id_matches_prefix(row.get("run_id", ""), prefix)
                    and not eu.registry_row_compatibility_issues(
                        row,
                        run_group_id=config["run_group_id"],
                        benchmark_item_count=len(benchmark),
                        expected_stochastic_samples=int(config["stochastic"]["samples"]),
                        required_tasks=("task2",),
                        expected_batch_order=profile.get("batch_order", config["batch_order"]),
                        expected_batch_size=int(profile["batch_size"]),
                    )
                ]
                if not candidates:
                    print(
                        f"skip: no complete {prefix}-* Task 2 run for "
                        f"{dataset_id}/{variant} {profile['profile_id']}/{model} in {registry_path}",
                        file=sys.stderr,
                    )
                    continue
                source_run_id = sorted(candidates, key=lambda row: str(row.get("started_at_utc", "")))[-1]["run_id"]
                print("\t".join([dataset_id, variant, profile["profile_id"], model, source_run_id]))
PY
}

while IFS=$'\t' read -r dataset variant profile model source_run_id; do
  [[ -z "${dataset:-}" ]] && continue
  cmd=".venv/bin/python scripts/run_task3_verification_from_config.py --config ${CONFIG} --profile ${profile} --model ${model} --dataset ${dataset} --variant ${variant} --source-run-id ${source_run_id} --audit-mode ${AUDIT_MODE} --mode full"
  if [[ "${TASK3_DRY_RUN:-0}" == "1" ]]; then
    echo "pueue add -- ${cmd}"
  else
    pueue add -- "${cmd}"
  fi
done < <(resolve_cells)
