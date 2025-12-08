# ==============================================================================
# [사용 가이드]
# 이 스크립트는 프로젝트 루트의 'bash/' 디렉토리 안에서 실행해야 합니다.
# 백그라운드에서 실행하고 로그를 남기려면 아래 명령어를 사용하세요.
# (logs 폴더가 없다면 미리 생성: mkdir -p ../logs)
#
# nohup ./sup_semi_experiments.sh > ../logs/sup_semi_exp_$(date "+%Y%m%d_%H%M%S").log 2>&1 &
# ==============================================================================

# 에러 발생 시 스크립트 중단
set -e

# ------------------------------------------------------------------------------
# 주요 하이퍼파라미터 (공통)
# ------------------------------------------------------------------------------
SEED=0
NUM_EPOCHS=2
BATCH_SIZE=16
PPO_BATCH_SIZE=512
EVENT_PER_IMAGE=1024

MAX_RATE=0.25

# 가중치 클리핑 (준지도/지도 학습용 w-clip)
W_CLIP_MIN=-1.0
W_CLIP_MAX=1.0

# 학습률
LR_ACTOR=1e-3
LR_CRITIC=1e-3

echo "================================================================================"
echo " Starting Semi-Supervised & Supervised Experiments"
echo " Start : $(date "+%Y%m%d_%H%M%S")"
echo "================================================================================"

mkdir -p ../logs

# ------------------------------------------------------------------------------
# 1. Semi-Supervised Experiment
# ------------------------------------------------------------------------------
SCENARIO="semi"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
RUN_NAME="${SCENARIO}_${TIMESTAMP}"

# 네트워크 크기 설정 (준지도용)
N_HIDDEN=300

#타임스텝
TIMESTEP=100
#스파이크 배열 길이
SPIKE_L=25

echo ""
echo "----------------------------------------------------------------"
echo "[RUNNING] Scenario: $SCENARIO (Semi-Supervised)"
echo "          Run Name: $RUN_NAME"
echo "----------------------------------------------------------------"

python ../src/main.py \
    --scenario "$SCENARIO" \
    --seed "$SEED" \
    --events-per-image "$EVENT_PER_IMAGE"\
    --num-epochs "$NUM_EPOCHS" \
    --batch-size-images "$BATCH_SIZE" \
    --ppo-batch-size "$PPO_BATCH_SIZE" \
    --T-semi "$TIMESTEP" \
    --spike-array-len "$SPIKE_L" \
    --N-hidden "$N_HIDDEN" \
    --w-clip-min "$W_CLIP_MIN" \
    --w-clip-max "$W_CLIP_MAX" \
    --lr-actor "$LR_ACTOR" \
    --lr-critic "$LR_CRITIC" \
    --run-name "$RUN_NAME"

echo "[DONE] Finished $RUN_NAME"


# ------------------------------------------------------------------------------
# 2. Supervised Experiment (Direct Current Input)
# ------------------------------------------------------------------------------
SCENARIO="grad"
ENCODING="direct"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
RUN_NAME="${SCENARIO}_${ENCODING}_${TIMESTAMP}"

#타임스텝
TIMESTEP=5
#스파이크 배열 길이
SPIKE_L=3

echo ""
echo "----------------------------------------------------------------"
echo "[RUNNING] Scenario: $SCENARIO (Supervised)"
echo "          Encoding: $ENCODING (Direct Current)"
echo "          Run Name: $RUN_NAME"
echo "----------------------------------------------------------------"

python ../src/main.py \
    --scenario "$SCENARIO" \
    --sup-input-encoding "$ENCODING" \
    --seed "$SEED" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size-images "$BATCH_SIZE" \
    --ppo-batch-size "$PPO_BATCH_SIZE" \
    --T-sup "$TIMESTEP" \
    --spike-array-len "$SPIKE_L" \
    --w-clip-min "$W_CLIP_MIN" \
    --w-clip-max "$W_CLIP_MAX" \
    --lr-actor "$LR_ACTOR" \
    --lr-critic "$LR_CRITIC" \
    --run-name "$RUN_NAME"

echo "[DONE] Finished $RUN_NAME"


# ------------------------------------------------------------------------------
# 3. Supervised Experiment (Poisson Spike Input)
# ------------------------------------------------------------------------------
SCENARIO="grad"
ENCODING="poisson"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
RUN_NAME="${SCENARIO}_${ENCODING}_${TIMESTAMP}"

#타임스텝
TIMESTEP=25
#스파이크 배열 길이
SPIKE_L=6

echo ""
echo "----------------------------------------------------------------"
echo "[RUNNING] Scenario: $SCENARIO (Supervised)"
echo "          Encoding: $ENCODING (Poisson Spikes)"
echo "          Run Name: $RUN_NAME"
echo "----------------------------------------------------------------"

python ../src/main.py \
    --scenario "$SCENARIO" \
    --sup-input-encoding "$ENCODING" \
    --seed "$SEED" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size-images "$BATCH_SIZE" \
    --ppo-batch-size "$PPO_BATCH_SIZE" \
    --T-sup "$TIMESTEP" \
    --spike-array-len "$SPIKE_L" \
    --max-rate "$MAX_RATE" \
    --w-clip-min "$W_CLIP_MIN" \
    --w-clip-max "$W_CLIP_MAX" \
    --lr-actor "$LR_ACTOR" \
    --lr-critic "$LR_CRITIC" \
    --run-name "$RUN_NAME"

echo "[DONE] Finished $RUN_NAME"

echo ""
echo "All semi/supervised experiments completed."
echo "END : $(date "+%Y%m%d_%H%M%S")"