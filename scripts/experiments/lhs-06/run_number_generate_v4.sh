#!/bin/bash
#set -e

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export HF_HOME=/data/ephemeral/hf_cache
#mkdir -p "$HF_HOME"

# 필요시
# export MODEL_NAME="unsloth/Qwen3-30B-A3B-bnb-4bit"
# export TEST_PATH="data/test.csv"
# export OUT_PATH="submission_number_generate_v4.csv"

# v4 기본(성능 중심) 권장 시작값
export MARGIN_THRESHOLD=1.4
export ENTROPY_THRESHOLD=1.10
export TOP1P_THRESHOLD=0.60

export SC_SAMPLES_BASE=5
export SC_TEMPERATURE_BASE=0.8
export SC_SAMPLES_HARD=7
export SC_TEMPERATURE_HARD=0.95
export HARD_MARGIN=0.7

export MAX_SC_RATE=0.70
export SEED=42

nohup python3 -u inference_number_generate_v4.py > inference_number_generate_v4.log 2>&1 &
echo "PID=$!"
echo "tail -f inference_number_generate_v4.log"
