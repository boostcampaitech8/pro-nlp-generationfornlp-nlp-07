#!/bin/bash
set -euo pipefail

export BASE_MODEL="unsloth/Qwen3-32B-bnb-4bit"
# 학습된 어댑터 HF repo 또는 로컬 경로
export ADAPTER_MODEL="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/outputs/T8164/Qwen3-32B-bnb-4bit-CoT/best_model"

export TEST_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/test/test.csv"
export OUT_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/outputs/submission_sc_budget.csv"
export SEED=42

# Gate thresholds
export MARGIN_THRESHOLD=0.75
export ENTROPY_THRESHOLD=1.40
export TOP1P_THRESHOLD=0.40

# v3 guard
export SC_TRIGGER_K=2
export CONF_MARGIN=2.0
export CONF_TOP1P=0.80
export CONF_ENTROPY=0.60
export HARD_MARGIN=0.45

# SC params
export SC_SAMPLES_BASE=5
export SC_TEMPERATURE_BASE=0.65
export SC_SAMPLES_HARD=7
export SC_TEMPERATURE_HARD=0.80

# SC accept
export SC_ACCEPT_MARGIN=1.0
export SC_ACCEPT_MARGIN_STRICT=1.6

# Budget
export MAX_SC_RATE=0.10

# Debug (optional)
export SC_DEBUG=0
export SC_DEBUG_MAX=200

nohup python3 -u inference_logit_sc.py > infer_sc.log 2>&1 &
echo "Started. Log: infer_sc_budget.log"
