#!/bin/bash
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export HF_HOME=/data/ephemeral/hf_cache
mkdir -p $HF_HOME

# 필요시 경로/모델명 오버라이드
# export MODEL_NAME="unsloth/Qwen3-30B-A3B-bnb-4bit"
# export TEST_PATH="data/test.csv"
# export OUT_PATH="submission.csv"

nohup python3 -u inference_qwen3_v2.py > inference.log 2>&1 &
echo "PID=$!"
echo "tail -f inference.log"
