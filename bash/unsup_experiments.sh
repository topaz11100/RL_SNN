#!/usr/bin/env bash
set -e

# ------------------------------------------------------------------------------
# Unsupervised만 2개 병렬 실행 (unsup1, unsup2)
# 이 파일은 bash/ 디렉토리에서 실행한다고 가정.
# ------------------------------------------------------------------------------

cd "$(dirname "$0")"
mkdir -p ../logs

TS=$(date "+%Y%m%d_%H%M%S")

echo "================================================================"
echo " Launching unsupervised scenarios (unsup1, unsup2) in parallel  ($TS)"
echo "================================================================"

# GPU가 하나뿐인 PC라면 CUDA_VISIBLE_DEVICES는 생략해도 됨.
# 필요하면 아래처럼 명시적으로 GPU 인덱스 지정:
#   CUDA_VISIBLE_DEVICES=0 nohup ...
# 아니면 그냥 nohup만 써도 됨.

nohup ./unsup1.sh \
  > "../logs/unsup1_${TS}.log" 2>&1 &

nohup ./unsup2.sh \
  > "../logs/unsup2_${TS}.log" 2>&1 &

echo "Submitted unsup1, unsup2 (2 jobs) with nohup."
echo "Check '../logs/unsup*_*.log' and 'nvidia-smi' for progress."
