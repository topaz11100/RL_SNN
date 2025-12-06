#!/usr/bin/env bash
set -euo pipefail

RUN_BASE="smoke_small_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="log"
mkdir -p "${LOG_DIR}"

echo "Run base: ${RUN_BASE}"
echo "Logs in: ${LOG_DIR}/<scenario>_${RUN_BASE}.log"
echo

run_one() {
  local scenario="$1"
  local extra_args="$2"
  local run_name="${RUN_BASE}_${scenario}"
  local log_file="${LOG_DIR}/${scenario}_${RUN_BASE}.log"

  echo "===== [${scenario}] start (run_name=${run_name}) ====="
  python main.py \
    --scenario "${scenario}" \
    --run-name "${run_name}" \
    --num-epochs 1 \
    --batch-size-images 8 \
    --T-unsup1 50 \
    --T-unsup2 50 \
    --T-semi 50 \
    --T-sup 50 \
    --spike-array-len 10 \
    --N-E 32 \
    --N-hidden 64 \
    --max-rate 0.1 \
    ${extra_args} \
    > "${log_file}" 2>&1
  echo "===== [${scenario}] done -> ${log_file} ====="
  echo
}

# 필요하면 이 스크립트 자체를 nohup으로 돌리면 됨:
#   nohup ./run_smoke_all_seq.sh &


run_one "semi" ""
run_one "grad" "--log-gradient-stats"
run_one "unsup1" ""
run_one "unsup2" ""

echo "All scenarios finished."
