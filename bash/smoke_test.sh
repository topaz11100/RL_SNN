#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# [경로 설정]
# 스크립트 실행 위치(bash/)를 기준으로 상위 폴더(프로젝트 루트)를 찾습니다.
# -----------------------------------------------------------------------------
# 1. 프로젝트 루트 경로 확보 (현재 폴더의 상위 폴더)
PROJECT_ROOT="$(cd .. && pwd)"

# 2. src 폴더를 PYTHONPATH에 추가 (모듈 import 경로 문제 해결)
export PYTHONPATH="${PYTHONPATH:-}:${PROJECT_ROOT}/src"

# 3. 로그 저장 위치 설정 (프로젝트 루트의 log 폴더에 저장)
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "Run base: ${PROJECT_ROOT}"
echo "Log dir : ${LOG_DIR}"
echo

# 실행 시 식별을 위한 타임스탬프
RUN_ID="smoke_quick_$(date +%Y%m%d_%H%M%S)"

run_one() {
  local scenario="$1"
  local extra_args="$2"
  local run_name="${RUN_ID}_${scenario}"
  local log_file="${LOG_DIR}/${scenario}_${RUN_ID}.log"

  echo "===== [${scenario}] start (run_name=${run_name}) ====="

  # 에러 발생 시 스크립트가 멈추지 않고 상태 코드를 잡기 위해 set +e 사용
  set +e
  
  # [핵심 변경] ../src/main.py 경로로 실행
  python "${PROJECT_ROOT}/src/main.py" \
    --scenario "${scenario}" \
    --run-name "${run_name}" \
    --num-epochs 1 \
    --batch-size-images 2 \
    --event-batch-size 128 \
    --T-unsup1 5 \
    --T-unsup2 5 \
    --T-semi 5 \
    --T-sup 5 \
    --spike-array-len 5 \
    --N-E 10 \
    --N-hidden 16 \
    --max-rate 0.5 \
    ${extra_args} \
    > "${log_file}" 2>&1
    
  status=$?
  set -e

  if [ "${status}" -ne 0 ]; then
    echo "===== [${scenario}] FAILED (exit=${status}) -> Check ${log_file} ====="
    echo "--- Last 5 lines of log ---"
    tail -n 5 "${log_file}"
    echo "---------------------------"
  else
    echo "===== [${scenario}] PASSED ====="
  fi
  echo
}

# -----------------------------------------------------------------------------
# [시나리오별 테스트 실행]
# -----------------------------------------------------------------------------

run_one "grad" "--direct-input --log-gradient-stats"
# 4. Unsupervised Dual Policy
run_one "unsup2" ""


# 1. Gradient Mimicry (Poisson)
run_one "grad" "--log-gradient-stats"

# 2. Gradient Mimicry (Direct Input) - 새로 추가된 기능 테스트


# 3. Semi-supervised
run_one "semi" ""



# 5. Unsupervised Single Policy
run_one "unsup1" ""

echo "All smoke tests finished."