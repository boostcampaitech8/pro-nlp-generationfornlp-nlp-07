#!/bin/bash
set -e

export MODEL_NAME="unsloth/Qwen3-30B-A3B-bnb-4bit"
export TEST_PATH="data/test.csv"
export OUT_PATH="submission_number_generate_v5.csv"

export PROMPT_ENSEMBLE=2
export SEED=42

# Uncertainty thresholds (start from these; tune after 1 run)
export MARGIN_THRESHOLD=6.0
export P1_THRESHOLD=0.90
export H_THRESHOLD=0.08

# Adaptive SC
export MAX_SC_RATE=0.70
export HARD_MARGIN=3.0

export SC_SAMPLES_BASE=9
export SC_TEMPERATURE_BASE=0.8
export SC_TOP_P_BASE=0.90

export SC_SAMPLES_HARD=13
export SC_TEMPERATURE_HARD=0.95
export SC_TOP_P_HARD=0.90

export FORCE_SC=0

nohup python3 -u inference_number_generate_v5.py > inference_number_generate_v5.log 2>&1 &