#!/bin/bash
#set -e

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export HF_HOME=/data/ephemeral/hf_cache
#mkdir -p "$HF_HOME"

# 필요시 오버라이드
# export MODEL_NAME="unsloth/Qwen3-30B-A3B-bnb-4bit"
# export TEST_PATH="data/test.csv"
# export OUT_PATH="submission_number_generate_v2_generate.csv"

# v2 튜닝 포인트(권장 시작값)
export MARGIN_THRESHOLD=1.0
export SC_SAMPLES=3
export SC_TEMPERATURE=0.8
export MAX_SC_RATE=0.25
export SEED=42

nohup python3 -u inference_number_generate_v2.py > inference_number_generate_v2.log 2>&1 &
echo "PID=$!"
echo "tail -f inference_number_generate_v2.log"
