# ==============================================================================
# [사용 가이드]
# 이 스크립트는 프로젝트 루트의 'bash/' 디렉토리 안에서 실행해야 합니다.
# 백그라운드에서 실행하고 로그를 남기려면 아래 명령어를 사용하세요.
# (logs 폴더가 없다면 미리 생성: mkdir -p ../logs)
#
# nohup ./unsup_experiments.sh > ../logs/unsup_exp_$(date "+%Y%m%d_%H%M%S").log 2>&1 &
# ==============================================================================

# 에러 발생 시 스크립트 중단
set -e

# ------------------------------------------------------------------------------
# 1. 실험 설정 (조합)
# ------------------------------------------------------------------------------
# 시나리오 목록
scenarios=("unsup1" "unsup2")

# 클리핑 설정 (포맷: "exc_min exc_max inh_min inh_max")
# 1) E[0,1] I[-1,0]
# 2) E[-1,1] I[-1,1]
clips=("0.0 1.0 -1.0 0.0" "-1.0 1.0 -1.0 1.0")

# ------------------------------------------------------------------------------
# 2. 주요 하이퍼파라미터 (변수로 추출)
# ------------------------------------------------------------------------------
SEED=0
NUM_EPOCHS=5
BATCH_SIZE=16           # 이미지 배치 크기
PPO_BATCH_SIZE=512      # PPO 업데이트용 이벤트 배치 크기
EVENT_PER_IMAGE=1024

TIMESTEP=100            # 시뮬레이션 시간
SPIKE_L=25              # 스파이크 히스토리 길이 (L)
MAX_RATE=0.25            # 푸아송 인코딩 최대 발화율 (Hz 아님, 비율)
N_E=300                 # 흥분성 뉴런 개수

# 보상(Reward) 관련 계수
ALPHA_SPARSE=0.4
RHO_TARGET=$MAX_RATE
ALPHA_DIV=0.4
ALPHA_STAB=0.2


echo "================================================================================"
echo " Starting Unsupervised Learning Experiments"
echo " Start : $(date "+%Y%m%d_%H%M%S")"
echo "================================================================================"

# ../logs 디렉토리 확인 및 생성
mkdir -p ../logs

for sc in "${scenarios[@]}"; do
    for c_val in "${clips[@]}"; do
        # 클리핑 값 파싱
        read exc_min exc_max inh_min inh_max <<< "$c_val"
        
        # 타임스탬프 및 실행 이름 생성
        timestamp=$(date "+%Y%m%d_%H%M%S")
        run_name="${sc}_E${exc_min}_${exc_max}_I${inh_min}_${inh_max}_${timestamp}"

        echo ""
        echo "----------------------------------------------------------------"
        echo "[RUNNING] Scenario: $sc"
        echo "          Clip: Exc[$exc_min, $exc_max], Inh[$inh_min, $inh_max]"
        echo "          Save Dir: $run_name"
        echo "----------------------------------------------------------------"

        python ../src/main.py \
            --scenario "$sc" \
            --events-per-image "$EVENT_PER_IMAGE"\
            --seed "$SEED" \
            --num-epochs "$NUM_EPOCHS" \
            --batch-size-images "$BATCH_SIZE" \
            --ppo-batch-size "$PPO_BATCH_SIZE" \
            --T-unsup1 "$TIMESTEP" \
            --T-unsup2 "$TIMESTEP" \
            --spike-array-len "$SPIKE_L" \
            --max-rate "$MAX_RATE" \
            --N-E "$N_E" \
            --exc-clip-min "$exc_min" \
            --exc-clip-max "$exc_max" \
            --inh-clip-min "$inh_min" \
            --inh-clip-max "$inh_max" \
            --rho-target "$RHO_TARGET" \
            --alpha-sparse "$ALPHA_SPARSE" \
            --alpha-div "$ALPHA_DIV" \
            --alpha-stab "$ALPHA_STAB" \
            --run-name "$run_name"

        echo "[DONE] Finished: $run_name"
    done
done

echo ""
echo "All unsupervised experiments completed."
echo "END : $(date "+%Y%m%d_%H%M%S")"