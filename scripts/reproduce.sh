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
#   env           Refresh the uv environment and run unit tests.
#   smoke-fake    Smoke run with --fake-completion (no API key required).
#   smoke         Smoke run against the configured provider/model cell.
#   full          Full Task 1 + Task 2 run.
#   task3         Task 3 diagnostic over a complete Task 2 source run.
#   analysis      Generate paper-facing analysis artifacts.
#   verify        Publication verification gate (status, tests, coverage).
#
# Required environment overrides (set via env or pass as extra args):
#   RE_UQ_CONFIG     Path to run config (default: run_configs/current_run.json)
#   RE_UQ_PROFILE    Provider profile name (default: zai)
#   RE_UQ_MODEL      Model id (default: glm-5.1)
#   RE_UQ_DATASET    Dataset id (default: mlm_tapt)
#   RE_UQ_VARIANT    Benchmark variant (default: must)

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${RE_UQ_CONFIG:-run_configs/current_run.json}"
PROFILE="${RE_UQ_PROFILE:-zai}"
MODEL="${RE_UQ_MODEL:-glm-5.1}"
DATASET="${RE_UQ_DATASET:-mlm_tapt}"
VARIANT="${RE_UQ_VARIANT:-must}"
PY=".venv/bin/python"

usage() {
  sed -n '2,26p' "$0"
  exit "${1:-0}"
}

require_config() {
  if [[ ! -f "$CONFIG" ]]; then
    echo "error: run config not found at $CONFIG" >&2
    echo "       cp run_configs/full_matrix.example.json $CONFIG" >&2
    exit 2
  fi
}

cmd="${1:-}"
shift || true

case "$cmd" in
  env)
    uv sync --group dev --locked
    "$PY" -m unittest discover -s tests -v
    ;;
  smoke-fake)
    require_config
    "$PY" scripts/run_experiment_from_config.py \
      --config "$CONFIG" --profile "$PROFILE" --model "$MODEL" \
      --dataset "$DATASET" --task both --mode smoke --fake-completion "$@"
    ;;
  smoke)
    require_config
    "$PY" scripts/run_experiment_from_config.py \
      --config "$CONFIG" --profile "$PROFILE" --model "$MODEL" \
      --dataset "$DATASET" --task both --mode smoke "$@"
    ;;
  full)
    require_config
    "$PY" scripts/run_experiment_from_config.py \
      --config "$CONFIG" --profile "$PROFILE" --model "$MODEL" \
      --dataset "$DATASET" --task both --mode full "$@"
    ;;
  task3)
    require_config
    "$PY" scripts/run_task3_verification_from_config.py \
      --config "$CONFIG" --profile "$PROFILE" --model "$MODEL" \
      --dataset "$DATASET" --mode full "$@"
    ;;
  analysis)
    "$PY" scripts/generate_evaluation_analysis.py \
      --dataset "$DATASET" --variant "$VARIANT" "$@"
    ;;
  verify)
    git status --short
    git diff --check
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
