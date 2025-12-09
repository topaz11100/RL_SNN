#!/usr/bin/env bash
set -e

# ------------------------------------------------------------------------------
# Semi + Supervised(grad) 3개 병렬 실행
#   - semi
#   - grad_direct
#   - grad_poisson
# 이 파일도 bash/ 디렉토리에서 실행한다고 가정.
# ------------------------------------------------------------------------------

cd "$(dirname "$0")"
mkdir -p ../logs

TS=$(date "+%Y%m%d_%H%M%S")

echo "================================================================"
echo " Launching semi/supervised scenarios (semi, grad_direct, grad_poisson) in parallel  ($TS)"
echo "================================================================"

# 마찬가지로 GPU 하나뿐이면 CUDA_VISIBLE_DEVICES 생략 가능.
# 필요하면 앞에 CUDA_VISIBLE_DEVICES=0 붙여서 명시.

nohup ./semi.sh > "../logs/semi_${TS}.log" 2>&1 &

nohup ./grad_direct.sh > "../logs/grad_direct_${TS}.log" 2>&1 &
nohup ./grad_poisson.sh > "../logs/grad_poisson_${TS}.log" 2>&1 &

echo "Submitted semi, grad_direct, grad_poisson (3 jobs) with nohup."
echo "Check '../logs/semi_*.log' and '../logs/grad_*_${TS}.log' and 'nvidia-smi' for progress."
