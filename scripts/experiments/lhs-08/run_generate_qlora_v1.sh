#!/bin/bash
set -e

# HF org/private token
export HF_TOKEN="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"

# Base + QLoRA adapter
export BASE_MODEL="unsloth/Qwen3-32B-bnb-4bit"
export ADAPTER_MODEL="NLP-07-ODQA/qwen3-32b-qlora-v1"

# I/O
export TEST_PATH="data/test.csv"
export OUT_PATH="submission_number_generate_v4_qlora.csv"

# Repro
export SEED=42

# v4 3-gate thresholds (네가 쓰던 형태 그대로)
export MARGIN_THRESHOLD=1.4
export ENTROPY_THRESHOLD=1.10
export TOP1P_THRESHOLD=0.60

# Adaptive SC
export HARD_MARGIN=0.7
export SC_SAMPLES_BASE=5
export SC_TEMPERATURE_BASE=0.8
export SC_SAMPLES_HARD=7
export SC_TEMPERATURE_HARD=0.95

export MAX_SC_RATE=0.70

nohup python3 -u inference_generate_qlora_v1.py > generate_qlora_v1.log 2>&1 &