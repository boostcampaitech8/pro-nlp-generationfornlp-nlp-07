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
export MARGIN_THRESHOLD=0.75
export ENTROPY_THRESHOLD=1.40
export TOP1P_THRESHOLD=0.40

# v3 강화 게이트
export SC_TRIGGER_K=2
export CONF_MARGIN=2.0
export CONF_TOP1P=0.80
export CONF_ENTROPY=0.60

# SC 반영은 좀 더 빡세게
export SC_ACCEPT_MARGIN=1.0
export SC_ACCEPT_MARGIN_STRICT=1.6

# Adaptive SC
export HARD_MARGIN=0.45
export SC_SAMPLES_BASE=5
export SC_TEMPERATURE_BASE=0.65
export SC_SAMPLES_HARD=9
export SC_TEMPERATURE_HARD=0.80
export MAX_SC_RATE=0.10

export SC_DEBUG=1
export SC_DEBUG_MAX=200   # 처음엔 200줄만 찍고 늘리기

nohup python3 -u inference_generate_qlora_v3.py > generate_qlora_v3.log 2>&1 &