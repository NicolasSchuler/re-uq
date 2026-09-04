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
# per config cell that has a complete full Task 2 source run. The resolution
# itself lives in scripts/task3_sources.py so this queue and any other driver
# pick the same source run.
resolve_cells() {
  local skip_args=()
  local profile
  IFS=',' read -ra skip_list <<< "$SKIP_PROFILES"
  for profile in "${skip_list[@]}"; do
    [[ -n "$profile" ]] && skip_args+=(--skip-profile "$profile")
  done
  "$PY" scripts/task3_sources.py --config "$CONFIG" "${skip_args[@]}"
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
