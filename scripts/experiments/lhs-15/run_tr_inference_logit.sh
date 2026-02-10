#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== 실험 식별 =====
export CAMPER_ID="T8164"
export EXP_NAME="Qwen2.5-32B-Instruct-bnb-4bit__8"
export HF_ORG="NLP-07-ODQA"
export OUTPUT_ROOT="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/outputs/"

# ===== 모델/학습 하이퍼 (CoT 추천 스타터) =====
# (1) 학습 시작점(이어학습 시작 체크포인트/어댑터)
export TRAIN_BASE_MODEL="unsloth/Qwen2.5-32B-Instruct-bnb-4bit"

# (2) 추론용 '진짜' 베이스
export INFER_BASE_MODEL="unsloth/Qwen2.5-32B-Instruct-bnb-4bit"

export MAX_SEQ_LENGTH=4096
export LORA_R=64
export LORA_ALPHA=128
export LEARNING_RATE=5e-5
export EPOCHS=2
export GRAD_ACCUM=8
export WEIGHT_DECAY=0.0

# ===== CoT 학습 on/off =====
export USE_COT_TRAIN=1
export COT_TRAIN_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/train/train_with_cot.csv"


TRAIN_LOG="train_${EXP_NAME}.log"
INFER_LOG="infer_sc_${EXP_NAME}.log"

# ===== 1) Train =====
export BASE_MODEL="${TRAIN_BASE_MODEL}"
#nohup python3 -u train_cot.py > "${TRAIN_LOG}" 2>&1
echo "[OK] Train done. Log: ${TRAIN_LOG}"

# 학습된 어댑터 HF repo 또는 로컬 경로
export BASE_MODEL="${INFER_BASE_MODEL}"
export ADAPTER_MODEL="${OUTPUT_ROOT}/${CAMPER_ID}/${EXP_NAME}/best_model"
export TEST_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/test/test.csv"
export OUT_PATH="${OUTPUT_ROOT}/submission_sc_${EXP_NAME}.csv"
export SEED=42

# Gate thresholds
export MARGIN_THRESHOLD=0.75
export ENTROPY_THRESHOLD=1.40
export TOP1P_THRESHOLD=0.40

# v3 guard
export MAX_SC_RATE=0.10
export SC_TRIGGER_K=3
export CONF_MARGIN=2.0
export CONF_TOP1P=0.80
export CONF_ENTROPY=0.60
export HARD_MARGIN=0.40

# SC params
export SC_SAMPLES_BASE=9
export SC_TEMPERATURE_BASE=0.65
export SC_SAMPLES_HARD=13
export SC_TEMPERATURE_HARD=0.80

# SC accept
export SC_ACCEPT_MARGIN=1.0
export SC_ACCEPT_MARGIN_STRICT=1.8

# Debug (optional)
export FIRST_SAMPLE_DEBUG=1
export SC_DEBUG=0
export SC_DEBUG_MAX=200

nohup python3 -u inference_logit_sc.py > "${INFER_LOG}" 2>&1 &
echo "[OK] Infer logit+SC done -> ${OUT_PATH} (log: ${INFER_LOG})"
