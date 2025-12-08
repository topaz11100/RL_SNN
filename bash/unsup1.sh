#!/usr/bin/env bash
set -e

# ------------------------------------------------------------------------------
# Unsupervised Scenario 1 (unsup1)
# ------------------------------------------------------------------------------

# 공통 설정
SEED=0
NUM_EPOCHS=5
BATCH_SIZE_IMAGES=16
EVENTS_PER_IMAGE=2048
EVENT_BATCH_SIZE=8196
PPO_BATCH_SIZE=8196

PPO_EPS=0.2
PPO_EPOCHS=2

T_UNSUP1=100
SPIKE_ARRAY_LEN=25
DT=1.0

N_E=300

SIGMA_UNSUP1=0.1

MAX_RATE=0.25
RHO_TARGET="$MAX_RATE"
ALPHA_SPARSE=0.4
ALPHA_DIV=0.4
ALPHA_STAB=0.2

LR_ACTOR=1e-3
LR_CRITIC=1e-3

LOG_INTERVAL=125

# E/I 클리핑 조합 (필요 없으면 하나만 남겨도 됨)
CLIPS=("0.0 1.0 -1.0 0.0" "-1.0 1.0 -1.0 1.0")

mkdir -p ../logs

for CLIP in "${CLIPS[@]}"; do
    read EXC_MIN EXC_MAX INH_MIN INH_MAX <<< "$CLIP"

    TS=$(date "+%Y%m%d_%H%M%S")
    RUN_NAME="unsup1_E${EXC_MIN}_${EXC_MAX}_I${INH_MIN}_${INH_MAX}_${TS}"

    echo "----------------------------------------------------------------"
    echo "[RUNNING] Scenario: unsup1  Clip: E[$EXC_MIN,$EXC_MAX] I[$INH_MIN,$INH_MAX]"
    echo "          Run Name: $RUN_NAME"
    echo "          Start   : $(date "+%Y%m%d_%H%M%S")"
    echo "----------------------------------------------------------------"

    python ../src/main.py \
        --scenario "unsup1" \
        --seed "$SEED" \
        --num-epochs "$NUM_EPOCHS" \
        --batch-size-images "$BATCH_SIZE_IMAGES" \
        --events-per-image "$EVENTS_PER_IMAGE" \
        --event-batch-size "$EVENT_BATCH_SIZE" \
        --ppo-batch-size "$PPO_BATCH_SIZE" \
        --ppo-eps "$PPO_EPS" \
        --ppo-epochs "$PPO_EPOCHS" \
        --T-unsup1 "$T_UNSUP1" \
        --spike-array-len "$SPIKE_ARRAY_LEN" \
        --dt "$DT" \
        --sigma-unsup1 "$SIGMA_UNSUP1" \
        --max-rate "$MAX_RATE" \
        --N-E "$N_E" \
        --exc-clip-min "$EXC_MIN" \
        --exc-clip-max "$EXC_MAX" \
        --inh-clip-min "$INH_MIN" \
        --inh-clip-max "$INH_MAX" \
        --rho-target "$RHO_TARGET" \
        --alpha-sparse "$ALPHA_SPARSE" \
        --alpha-div "$ALPHA_DIV" \
        --alpha-stab "$ALPHA_STAB" \
        --lr-actor "$LR_ACTOR" \
        --lr-critic "$LR_CRITIC" \
        --log-interval "$LOG_INTERVAL" \
        --run-name "$RUN_NAME"

    echo "[DONE] $RUN_NAME"
    echo "END : $(date "+%Y%m%d_%H%M%S")"
done
