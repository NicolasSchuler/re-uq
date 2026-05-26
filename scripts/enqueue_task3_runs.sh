#!/usr/bin/env bash
set -euo pipefail

# Enqueue official blind Task 3 text-audit runs for completed Task 1/2 source runs.
# The mlm_tapt azure.gpt-5.4 cells are intentionally omitted because their
# synced Task 2 parse/scoring state was not usable for Task 3 source auditing.
#
# Override for anchoring ablations, e.g.:
#   TASK3_AUDIT_MODE=declared_source bash scripts/enqueue_task3_runs.sh

queue_task3() {
  local dataset="$1"
  local variant="$2"
  local profile="$3"
  local model="$4"
  local source_run_id="$5"
  local audit_mode="${TASK3_AUDIT_MODE:-blind}"

  pueue add -- ".venv/bin/python scripts/run_task3_verification_from_config.py --config run_configs/current_run.json --profile ${profile} --model ${model} --dataset ${dataset} --variant ${variant} --source-run-id ${source_run_id} --audit-mode ${audit_mode} --mode full"
}

# nice / must
queue_task3 nice must kit_toolbox kit.gemma4-31b-it full-20260522-141406-962a3a84
queue_task3 nice must kit_toolbox azure.gpt-5.4 full-20260522-162229-a1d98608
queue_task3 nice must kit_toolbox azure.gpt-5-nano full-20260522-173231-29f884e0
queue_task3 nice must zai glm-5.1 full-20260522-184142-83579d93
queue_task3 nice must zai glm-5 full-20260522-200032-3a4c8216
queue_task3 nice must zai glm-5-turbo full-20260522-211601-62e27962
queue_task3 nice must zai glm-4.7 full-20260522-224445-fa160fed
queue_task3 nice must zai glm-4.5-air full-20260523-001117-467c466a

# nice / shall
queue_task3 nice shall kit_toolbox kit.gemma4-31b-it full-shall-20260522-151931-1f4cd32f
queue_task3 nice shall kit_toolbox azure.gpt-5.4 full-shall-20260522-165758-bbdf9ab5
queue_task3 nice shall kit_toolbox azure.gpt-5-nano full-shall-20260522-180736-b42c186e
queue_task3 nice shall zai glm-5.1 full-shall-20260522-192121-e5eab752
queue_task3 nice shall zai glm-5 full-shall-20260522-203850-9476f622
queue_task3 nice shall zai glm-5-turbo full-shall-20260522-220118-d2b1aafc
queue_task3 nice shall zai glm-4.7 full-shall-20260522-232742-55d639a2
queue_task3 nice shall zai glm-4.5-air full-shall-20260523-005713-bd7b7969

# mlm_tapt / must
queue_task3 mlm_tapt must kit_toolbox kit.gemma4-31b-it full-20260523-014135-78ceaa43
queue_task3 mlm_tapt must zai glm-5.1 full-20260523-052426-36e8cef4
queue_task3 mlm_tapt must zai glm-5 full-20260523-065237-ce677096
queue_task3 mlm_tapt must zai glm-5-turbo full-20260523-082104-e8b8f0ba
queue_task3 mlm_tapt must zai glm-4.7 full-20260523-120814-aa161ed3
queue_task3 mlm_tapt must zai glm-4.5-air full-20260523-141326-7124ae7d

# mlm_tapt / shall
queue_task3 mlm_tapt shall kit_toolbox kit.gemma4-31b-it full-shall-20260523-030706-5f16ae1d
queue_task3 mlm_tapt shall zai glm-5.1 full-shall-20260523-060859-8a3926a6
queue_task3 mlm_tapt shall zai glm-5 full-shall-20260523-073541-467f276c
queue_task3 mlm_tapt shall zai glm-5-turbo full-shall-20260523-102508-fe8e153c
queue_task3 mlm_tapt shall zai glm-4.7 full-shall-20260523-131535-655f057e
queue_task3 mlm_tapt shall zai glm-4.5-air full-shall-20260523-151636-f509d22b
