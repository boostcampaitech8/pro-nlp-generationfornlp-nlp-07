#!/bin/bash
set -euo pipefail

export BASE_MODEL="unsloth/Qwen3-32B-bnb-4bit"
# 학습된 어댑터 HF repo 또는 로컬 경로 (없으면 빈 값)
export ADAPTER_MODEL="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/outputs/T8164/Qwen3-32B-bnb-4bit-CoT/best_model"

export TEST_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/test/test.csv"
export OUT_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/outputs/submission_logit_only.csv"
export SEED=42

nohup python3 -u inference_logit_only.py > infer_logit_only.log 2>&1 &
echo "Started logit-only inference. Log: infer_logit_only.log"
