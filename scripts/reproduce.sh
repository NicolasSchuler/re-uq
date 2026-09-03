#!/usr/bin/env bash
# Thin convenience wrapper around the canonical reproduction commands.
#
# This script does not introduce new behavior. Every subcommand maps 1:1 to
# an existing scripts/*.py CLI documented in docs/reproduction.md. Defaults
# are intentionally conservative.
#
# Usage:
#   scripts/reproduce.sh <subcommand> [extra args]
#
# Subcommands:
#   env                  Refresh the uv environment and run unit tests.
#   smoke-fake           Smoke run with --fake-completion (no API key required).
#   smoke-fake-task3     Blind Task 3 over the latest fake smoke run (no API key).
#   smoke-fake-analysis  Analysis over the latest fake smoke run (no API key).
#   smoke-fake-all       smoke-fake -> smoke-fake-task3 -> smoke-fake-analysis.
#   smoke                Smoke run against the configured provider/model cell.
#   full                 Full Task 1 + Task 2 run.
#   task3                Blind Task 3 diagnostic over a complete Task 2 source run.
#   analysis             Generate paper-facing analysis artifacts.
#   verify               Publication verification gate (status, tests, coverage).
#   hydra                Hydra entry point: forwards all args to scripts/run.py.
#   hydra-export         Export a JSON run config into the conf/ groups.
#
# Required environment overrides (set via env or pass as extra args):
#   RE_UQ_CONFIG     Path to run config (default: run_configs/current_run.json)
#   RE_UQ_PROFILE    Provider profile name (default: zai)
#   RE_UQ_MODEL      Model id (default: glm-5.1)
#   RE_UQ_DATASET    Dataset id (default: mlm_tapt)
#   RE_UQ_VARIANT    Benchmark variant (default: must)
#   RE_UQ_MODE       Run mode for `task3` (default: full)
#   RE_UQ_RUN_ID     Explicit run id for analysis/task3 source lookups
#
# The `hydra` subcommand ignores the RE_UQ_* variables above: it is configured
# entirely by conf/ plus the overrides you pass, e.g.
#   scripts/reproduce.sh hydra profile=zai model=glm-5.1 dataset=nice mode=full
#   scripts/reproduce.sh hydra --multirun +experiment=paper_cohort
# See docs/configuration.md.
#
# The smoke-fake* subcommands synthesize completions locally: they never call a
# provider, never read an API key, and write only into data/processed/smoke/.

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${RE_UQ_CONFIG:-run_configs/current_run.json}"
PROFILE="${RE_UQ_PROFILE:-zai}"
MODEL="${RE_UQ_MODEL:-glm-5.1}"
DATASET="${RE_UQ_DATASET:-mlm_tapt}"
VARIANT="${RE_UQ_VARIANT:-must}"
MODE="${RE_UQ_MODE:-full}"
PY=".venv/bin/python"

usage() {
  sed -n '2,41p' "$0"
  exit "${1:-0}"
}

announce() {
  echo "resolved: subcommand=$1 PROFILE=${PROFILE} MODEL=${MODEL} DATASET=${DATASET} VARIANT=${VARIANT} CONFIG=${CONFIG} MODE=${MODE}" >&2
}

require_config() {
  if [[ ! -f "$CONFIG" ]]; then
    echo "error: run config not found at $CONFIG" >&2
    echo "       cp run_configs/full_matrix.example.json $CONFIG" >&2
    exit 2
  fi
}

# Newest run id with the given prefix in a raw-output tree ("smoke" or "full").
latest_run_id() {
  local tree="$1" prefix="$2"
  RE_UQ_TREE="$tree" RE_UQ_PREFIX="$prefix" RE_UQ_DS="$DATASET" RE_UQ_VAR="$VARIANT" "$PY" - <<'PY'
import os
import sys

sys.path.insert(0, "scripts")
import eval_utils as eu

smoke = os.environ["RE_UQ_TREE"] == "smoke"
prefix = os.environ["RE_UQ_PREFIX"]
resolve = eu.task3_raw_path if prefix.startswith("task3") else eu.model_outputs_raw_path
path = resolve(eu.project_root(), os.environ["RE_UQ_DS"], os.environ["RE_UQ_VAR"], smoke=smoke)
run_id = eu.latest_run_id(eu.read_jsonl(path), prefix=prefix)
if not run_id:
    raise SystemExit(f"error: no {os.environ['RE_UQ_PREFIX']}-* run found in {path}")
print(run_id)
PY
}

smoke_fake_run() {
  require_config
  # --fake-completion synthesizes responses locally; no provider key is read.
  "$PY" scripts/run_experiment_from_config.py \
    --config "$CONFIG" --profile "$PROFILE" --model "$MODEL" \
    --dataset "$DATASET" --task both --mode smoke --fake-completion "$@"
}

smoke_fake_task3() {
  require_config
  local source_run_id="${RE_UQ_RUN_ID:-$(latest_run_id smoke smoke)}"
  echo "resolved: source_run_id=${source_run_id}" >&2
  "$PY" scripts/run_task3_verification_from_config.py \
    --config "$CONFIG" --profile "$PROFILE" --model "$MODEL" \
    --dataset "$DATASET" --variant "$VARIANT" --mode smoke --fake-completion \
    --source-run-id "$source_run_id" --allow-partial-source "$@"
}

smoke_fake_analysis() {
  local run_id="${RE_UQ_RUN_ID:-$(latest_run_id smoke smoke)}"
  local task3_run_id
  task3_run_id="$(latest_run_id smoke task3-smoke || true)"
  echo "resolved: run_id=${run_id} task3_run_id=${task3_run_id:-none}" >&2
  # RE_UQ_SMOKE_TREE routes the read-only artifact lookups at data/processed/smoke/;
  # smoke analysis artifacts stay under outputs/smoke/ and never next to paper runs.
  RE_UQ_SMOKE_TREE=1 "$PY" scripts/generate_evaluation_analysis.py \
    --dataset "$DATASET" --variant "$VARIANT" --run-id "$run_id" \
    --output-dir "outputs/smoke/evaluation_${DATASET}_${VARIANT}_${run_id//-/_}" \
    ${task3_run_id:+--task3-run-id "$task3_run_id"} \
    --allow-partial --skip-registry-check --skip-construct-review-check "$@"
}

cmd="${1:-}"
shift || true

case "$cmd" in
  env)
    announce env
    uv sync --group dev --locked
    "$PY" -m unittest discover -s tests -v
    ;;
  smoke-fake)
    announce smoke-fake
    smoke_fake_run "$@"
    ;;
  smoke-fake-task3)
    announce smoke-fake-task3
    smoke_fake_task3 "$@"
    ;;
  smoke-fake-analysis)
    announce smoke-fake-analysis
    smoke_fake_analysis "$@"
    ;;
  smoke-fake-all)
    announce smoke-fake-all
    smoke_fake_run
    smoke_fake_task3
    smoke_fake_analysis
    ;;
  smoke)
    announce smoke
    require_config
    "$PY" scripts/run_experiment_from_config.py \
      --config "$CONFIG" --profile "$PROFILE" --model "$MODEL" \
      --dataset "$DATASET" --task both --mode smoke "$@"
    ;;
  full)
    announce full
    require_config
    "$PY" scripts/run_experiment_from_config.py \
      --config "$CONFIG" --profile "$PROFILE" --model "$MODEL" \
      --dataset "$DATASET" --task both --mode full "$@"
    ;;
  task3)
    announce task3
    require_config
    "$PY" scripts/run_task3_verification_from_config.py \
      --config "$CONFIG" --profile "$PROFILE" --model "$MODEL" \
      --dataset "$DATASET" --mode "$MODE" "$@"
    ;;
  analysis)
    announce analysis
    "$PY" scripts/generate_evaluation_analysis.py \
      --dataset "$DATASET" --variant "$VARIANT" "$@"
    ;;
  hydra)
    # Composition-based entry point; conf/ replaces --config for this path.
    echo "resolved: subcommand=hydra (configured by conf/ + overrides)" >&2
    "$PY" scripts/run.py "$@"
    ;;
  hydra-export)
    echo "resolved: subcommand=hydra-export CONFIG=${CONFIG}" >&2
    require_config
    "$PY" scripts/hydra_bridge.py --config "$CONFIG" "$@"
    ;;
  verify)
    announce verify
    git status --short
    git diff --check
    "$PY" - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, "scripts")
import eval_utils as eu
root = eu.project_root()
for manifest in sorted(root.glob("outputs/benchmark_manifest*.json")):
    summary = eu.verify_benchmark_manifest(manifest, root)
    print(f"manifest OK: {manifest.name} checked={summary['checked']} missing={summary['missing_count']}")
PY
    "$PY" -m unittest discover -s tests -v
    "$PY" -m coverage run --branch --source=scripts -m unittest discover -s tests -v
    "$PY" -m coverage report -m
    ;;
  -h|--help|help|"")
    usage 0
    ;;
  *)
    echo "error: unknown subcommand '$cmd'" >&2
    usage 2
    ;;
esac
