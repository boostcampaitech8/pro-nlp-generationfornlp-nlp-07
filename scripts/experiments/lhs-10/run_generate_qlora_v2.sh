#!/bin/bash
set -e

# HF org/private token
export HF_TOKEN=""

# Base + QLoRA adapter
export BASE_MODEL="unsloth/Qwen3-32B-bnb-4bit"
export ADAPTER_MODEL="NLP-07-ODQA/qwen3-32b-qlora-v1"

# I/O
export TEST_PATH="data/test.csv"
export OUT_PATH="submission_number_generate_v4_qlora_2.csv"

# Repro
export SEED=42

# v4 3-gate thresholds (네가 쓰던 형태 그대로)
export MARGIN_THRESHOLD=0.9
export ENTROPY_THRESHOLD=1.30
export TOP1P_THRESHOLD=0.45

# Adaptive SC
export HARD_MARGIN=0.45
export SC_SAMPLES_BASE=9
export SC_TEMPERATURE_BASE=0.65
export SC_SAMPLES_HARD=13
export SC_TEMPERATURE_HARD=0.80
export SC_ACCEPT_MARGIN=1.0
export MAX_SC_RATE=0.12

nohup python3 -u inference_generate_qlora_v2.py > generate_qlora_v2.log 2>&1 &