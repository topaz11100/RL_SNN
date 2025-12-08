#!/usr/bin/env bash
set -e

# ------------------------------------------------------------------------------
# Semi-Supervised Scenario (semi)
# ------------------------------------------------------------------------------

SEED=0
NUM_EPOCHS=2
BATCH_SIZE_IMAGES=16
EVENTS_PER_IMAGE=2048
EVENT_BATCH_SIZE=8196
PPO_BATCH_SIZE=8196

PPO_EPS=0.2
PPO_EPOCHS=2

T_SEMI=80
SPIKE_LEN_SEMI=20
DT=1.0

N_HIDDEN=300

MAX_RATE=0.25
SIGMA_SEMI=0.1
BETA_MARGIN=0.5

W_CLIP_MIN=-1.0
W_CLIP_MAX=1.0

LR_ACTOR=1e-3
LR_CRITIC=1e-3

LOG_INTERVAL=125

mkdir -p ../logs

TS=$(date "+%Y%m%d_%H%M%S")
RUN_NAME="semi_${TS}"

echo "----------------------------------------------------------------"
echo "[RUNNING] Scenario: semi"
echo "          Run Name: $RUN_NAME"
echo "          Start   : $(date "+%Y%m%d_%H%M%S")"
echo "----------------------------------------------------------------"

python ../src/main.py \
    --scenario "semi" \
    --seed "$SEED" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size-images "$BATCH_SIZE_IMAGES" \
    --events-per-image "$EVENTS_PER_IMAGE" \
    --event-batch-size "$EVENT_BATCH_SIZE" \
    --ppo-batch-size "$PPO_BATCH_SIZE" \
    --ppo-eps "$PPO_EPS" \
    --ppo-epochs "$PPO_EPOCHS" \
    --T-semi "$T_SEMI" \
    --spike-array-len "$SPIKE_LEN_SEMI" \
    --dt "$DT" \
    --N-hidden "$N_HIDDEN" \
    --max-rate "$MAX_RATE" \
    --sigma-semi "$SIGMA_SEMI" \
    --beta-margin "$BETA_MARGIN" \
    --w-clip-min "$W_CLIP_MIN" \
    --w-clip-max "$W_CLIP_MAX" \
    --lr-actor "$LR_ACTOR" \
    --lr-critic "$LR_CRITIC" \
    --log-interval "$LOG_INTERVAL" \
    --run-name "$RUN_NAME"

echo "[DONE] $RUN_NAME"
echo "END : $(date "+%Y%m%d_%H%M%S")"