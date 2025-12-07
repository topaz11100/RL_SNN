#!/bin/bash
cd ..
# 에러 체크용 스크립트: 모든 시나리오를 최소 부하로 1회 실행
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EPOCHS=1
BATCH=3           # 최소 배치
K=3               # K=1 (Events per image)
T_SHORT=5        # 매우 짧은 시뮬레이션 시간

echo "Starting Error Check (Dry Run)..."

# 1. Unsup1
echo "[Checking Unsup1]"
python src/main.py --scenario unsup1 --run-name "debug_unsup1_${TIMESTAMP}" \
    --num-epochs ${EPOCHS} --batch-size-images ${BATCH} --events-per-image ${K} \
    --T-unsup1 ${T_SHORT} --exc-clip-max 1.0 --inh-clip-min -1.0 --log-interval 100

# 2. Unsup2
echo "[Checking Unsup2]"
python src/main.py --scenario unsup2 --run-name "debug_unsup2_${TIMESTAMP}" \
    --num-epochs ${EPOCHS} --batch-size-images ${BATCH} --events-per-image 30 \
    --T-unsup2 50 --exc-clip-max 1.0 --inh-clip-min -1.0 --log-interval 100

# 3. Semi
echo "[Checking Semi]"
python src/main.py --scenario semi --run-name "debug_semi_${TIMESTAMP}" \
    --num-epochs ${EPOCHS} --batch-size-images ${BATCH} --events-per-image ${K} \
    --T-semi ${T_SHORT} --w-clip-min -1.0 --w-clip-max 1.0 --N-hidden 32 --log-interval 100

# 4. Grad (Direct)
echo "[Checking Grad - Direct]"
python src/main.py --scenario grad --run-name "debug_grad_direct_${TIMESTAMP}" \
    --num-epochs ${EPOCHS} --batch-size-images ${BATCH} --events-per-image ${K} \
    --T-sup 2 --sup-input-encoding direct --w-clip-min -1.0 --w-clip-max 1.0 --log-interval 100

# 5. Grad (Poisson)
echo "[Checking Grad - Poisson]"
python src/main.py --scenario grad --run-name "debug_grad_poisson_${TIMESTAMP}" \
    --num-epochs ${EPOCHS} --batch-size-images ${BATCH} --events-per-image ${K} \
    --T-sup ${T_SHORT} --sup-input-encoding poisson --w-clip-min -1.0 --w-clip-max 1.0 --log-interval 100

echo "Error check completed. Check logs in results/ folder."