#!/bin/bash
#set -e

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export HF_HOME=/data/ephemeral/hf_cache
#mkdir -p "$HF_HOME"

# 모델/경로 필요시 오버라이드
# export MODEL_NAME="unsloth/Qwen3-30B-A3B-bnb-4bit"
# export TEST_PATH="data/test.csv"
# export OUT_PATH="submission_number_generate.csv"

nohup python3 -u inference_number_generate_v1.py > inference_number_generate_v1.log 2>&1 &
echo "PID=$!"
echo "tail -f inference_number_generate_v1.log"
