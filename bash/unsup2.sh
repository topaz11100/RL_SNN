#!/usr/bin/env bash
set -e

# ------------------------------------------------------------------------------
# Unsupervised Scenario 2 (unsup2)
# ------------------------------------------------------------------------------

SEED=0
NUM_EPOCHS=15
BATCH_SIZE_IMAGES=4
EVENTS_PER_IMAGE=512
EVENT_BATCH_SIZE=2048
PPO_BATCH_SIZE=256

PPO_EPS=0.2
PPO_EPOCHS=2

T_UNSUP2=80
SPIKE_ARRAY_LEN=20
DT=1.0

N_E=250

SIGMA_UNSUP2=0.1

MAX_RATE=0.25
RHO_TARGET="$MAX_RATE"
ALPHA_SPARSE=0.4
ALPHA_DIV=0.4
ALPHA_STAB=0.2

LR_ACTOR=1e-3
LR_CRITIC=1e-3

LOG_INTERVAL=125

CLIPS=("-1.0 1.0 -1.0 1.0")

mkdir -p ../logs

for CLIP in "${CLIPS[@]}"; do
    read EXC_MIN EXC_MAX INH_MIN INH_MAX <<< "$CLIP"

    TS=$(date "+%Y%m%d_%H%M%S")
    RUN_NAME="unsup2_E${EXC_MIN}_${EXC_MAX}_I${INH_MIN}_${INH_MAX}_${TS}"

    echo "----------------------------------------------------------------"
    echo "[RUNNING] Scenario: unsup2  Clip: E[$EXC_MIN,$EXC_MAX] I[$INH_MIN,$INH_MAX]"
    echo "          Run Name: $RUN_NAME"
    echo "          Start   : $(date "+%Y%m%d_%H%M%S")"
    echo "----------------------------------------------------------------"

    python ../src/main.py \
        --scenario "unsup2" \
        --seed "$SEED" \
        --num-epochs "$NUM_EPOCHS" \
        --batch-size-images "$BATCH_SIZE_IMAGES" \
        --events-per-image "$EVENTS_PER_IMAGE" \
        --event-batch-size "$EVENT_BATCH_SIZE" \
        --ppo-batch-size "$PPO_BATCH_SIZE" \
        --ppo-eps "$PPO_EPS" \
        --ppo-epochs "$PPO_EPOCHS" \
        --T-unsup2 "$T_UNSUP2" \
        --spike-array-len "$SPIKE_ARRAY_LEN" \
        --dt "$DT" \
        --sigma-unsup2 "$SIGMA_UNSUP2" \
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
        --run-name "$RUN_NAME" \
        

    echo "[DONE] $RUN_NAME"
    echo "END : $(date "+%Y%m%d_%H%M%S")"
done
