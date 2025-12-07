#!/bin/bash

# 스크립트가 에러 발생 시 중단되도록 설정
set -e

# 시나리오 및 클리핑 설정 배열
scenario=("unsup1" "unsup2")
# 포맷: exc_min exc_max inh_min inh_max
clip=("0.0 1.0 -1.0 0.0" "-1.0 1.0 -1.0 1.0")

# 하이퍼파라미터 설정
timestep=100
spike_L=25
max_rate=0.25

sparce=0.3
rho_target=$max_rate
divergence=0.4
stability=0.3

N_E=200

img_batch=1
ppo_batch=256
epoch=2
seed=0

echo "Running experiments from $(pwd)..."

for sc in "${scenario[@]}"; do
    for c_val in "${clip[@]}"; do
        # clip 문자열을 4개 변수로 분해
        read exc_min exc_max inh_min inh_max <<< "$c_val"
        
        # 현재 시간 가져오기 (예: 20251207_123000)
        # 루프 돌 때마다 시간이 갱신되어 폴더가 고유해집니다.
        timestamp=$(date "+%Y%m%d_%H%M%S")

        # 실행 이름 생성 (로그 폴더명용)
        # 예: unsup1_exc0.0_1.0_inh-1.0_0.0_20251207_123000
        run_name="${sc}_exc${exc_min}_${exc_max}_inh${inh_min}_${inh_max}_${timestamp}"

        echo "----------------------------------------------------------------"
        echo "[RUN] Scenario: $sc"
        echo "      Clip: E[$exc_min, $exc_max] I[$inh_min, $inh_max]"
        echo "      Timestamp: $timestamp"
        echo "----------------------------------------------------------------"

        # ../src/main.py 실행
        python ../src/main.py \
            --scenario "$sc" \
            --seed "$seed" \
            --batch-size-images "$img_batch" \
            --ppo-batch-size "$ppo_batch"\
            --num-epochs "$epoch" \
            --T-unsup1 "$timestep" \
            --T-unsup2 "$timestep" \
            --spike-array-len "$spike_L" \
            --max-rate "$max_rate" \
            --rho-target "$rho_target" \
            --alpha-sparse "$sparce" \
            --alpha-div "$divergence" \
            --alpha-stab "$stability" \
            --N-E "$N_E" \
            --exc-clip-min "$exc_min" \
            --exc-clip-max "$exc_max" \
            --inh-clip-min "$inh_min" \
            --inh-clip-max "$inh_max" \
            --run-name "$run_name"

        echo "[DONE] Finished $run_name"
        echo ""
    done
done