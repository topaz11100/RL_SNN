#!/usr/bin/env bash
set -e

# ------------------------------------------------------------------------------
# Supervised Gradient Mimicry (grad) - Poisson Spike Input
# ------------------------------------------------------------------------------

SEED=0
NUM_EPOCHS=15
BATCH_SIZE_IMAGES=16
EVENTS_PER_IMAGE=1024
EVENT_BATCH_SIZE=16284
PPO_BATCH_SIZE=1024

PPO_EPS=0.2
PPO_EPOCHS=2

DT=1.0
T_SUP_POISSON=50
SPIKE_LEN_SUP_POISSON=16

MAX_RATE=0.25
SIGMA_SUP=0.1

ALPHA_ALIGN=0.1
LAYER_INDEX_SCALE=1.0
LOG_GRADIENT_STATS=125

W_CLIP_MIN=-1.0
W_CLIP_MAX=1.0

LR_ACTOR=1e-3
LR_CRITIC=1e-3

LOG_INTERVAL=125

mkdir -p ../logs

if [ "$LOG_GRADIENT_STATS" -eq 1 ]; then
    GRAD_STATS_FLAG="--log-gradient-stats"
else
    GRAD_STATS_FLAG=""
fi

TS=$(date "+%Y%m%d_%H%M%S")
RUN_NAME="grad_poisson_${TS}"

echo "----------------------------------------------------------------"
echo "[RUNNING] Scenario: grad (poisson input)"
echo "          Run Name: $RUN_NAME"
echo "          Start   : $(date "+%Y%m%d_%H%M%S")"
echo "----------------------------------------------------------------"

python ../src/main.py \
    --scenario "grad" \
    --sup-input-encoding "poisson" \
    --seed "$SEED" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size-images "$BATCH_SIZE_IMAGES" \
    --events-per-image "$EVENTS_PER_IMAGE" \
    --event-batch-size "$EVENT_BATCH_SIZE" \
    --ppo-batch-size "$PPO_BATCH_SIZE" \
    --ppo-eps "$PPO_EPS" \
    --ppo-epochs "$PPO_EPOCHS" \
    --T-sup "$T_SUP_POISSON" \
    --spike-array-len "$SPIKE_LEN_SUP_POISSON" \
    --dt "$DT" \
    --max-rate "$MAX_RATE" \
    --sigma-sup "$SIGMA_SUP" \
    --alpha-align "$ALPHA_ALIGN" \
    --layer-index-scale "$LAYER_INDEX_SCALE" \
    $GRAD_STATS_FLAG \
    --w-clip-min "$W_CLIP_MIN" \
    --w-clip-max "$W_CLIP_MAX" \
    --lr-actor "$LR_ACTOR" \
    --lr-critic "$LR_CRITIC" \
    --log-interval "$LOG_INTERVAL" \
    --run-name "$RUN_NAME"

echo "[DONE] $RUN_NAME"
echo "END : $(date "+%Y%m%d_%H%M%S")"