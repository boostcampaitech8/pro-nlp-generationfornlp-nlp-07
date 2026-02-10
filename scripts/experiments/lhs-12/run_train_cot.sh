#!/bin/bash
set -euo pipefail

# (선택) HF 토큰
export HF_TOKEN=""

# CoT 학습 활성화 (기본 학습으로 돌리려면 USE_COT_TRAIN=0)
export USE_COT_TRAIN=1
export COT_TRAIN_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/train/train_with_cot.csv"

# 백그라운드 실행
nohup python3 -u rebuilding_cot.py > ./train_cot.log 2>&1 &

echo "Started: rebuilding_cot.py"
echo "Log: train_cot_v3.log"
