#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== 실험 식별 (공통) =====
export CAMPER_ID="T8164"
export HF_ORG="NLP-07-ODQA"
export OUTPUT_ROOT="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/outputs/"

# ===== 모델/학습 하이퍼 (공통) =====
# (1) 학습 시작점(이어학습 시작 체크포인트/어댑터)
export TRAIN_BASE_MODEL="unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
# (2) 추론용 '진짜' 베이스
export INFER_BASE_MODEL="unsloth/Qwen2.5-32B-Instruct-bnb-4bit"

export MAX_SEQ_LENGTH=4096
export EPOCHS=2
export GRAD_ACCUM=8
export WEIGHT_DECAY=0.02

# ===== CoT 학습 on/off (공통) =====
export USE_COT_TRAIN=1
export COT_TRAIN_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/train/train_with_cot.csv"

# ===== 추론 공통 =====
export TEST_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/test/test.csv"
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


run_one () {
  local EXP_SUFFIX="$1"
  local LR="$2"
  local R="$3"
  local ALPHA="$4"

  export EXP_NAME="Qwen2.5-32B-Instruct-bnb-4bit_${EXP_SUFFIX}"
  export LEARNING_RATE="${LR}"
  export LORA_R="${R}"
  export LORA_ALPHA="${ALPHA}"

  local TRAIN_LOG="train_${EXP_NAME}.log"
  local INFER_LOG="infer_sc_${EXP_NAME}.log"

  echo "============================================================"
  echo "[RUN] EXP_NAME=${EXP_NAME}"
  echo "      LR=${LEARNING_RATE}  R=${LORA_R}  ALPHA=${LORA_ALPHA}"
  echo "============================================================"

  # ===== 1) Train =====
  export BASE_MODEL="${TRAIN_BASE_MODEL}"
  nohup python3 -u train_cot.py > "${TRAIN_LOG}" 2>&1
  echo "[OK] Train done. Log: ${TRAIN_LOG}"

  # ===== 2) Infer =====
  export BASE_MODEL="${INFER_BASE_MODEL}"
  export ADAPTER_MODEL="${OUTPUT_ROOT}/${CAMPER_ID}/${EXP_NAME}/best_model"
  export OUT_PATH="${OUTPUT_ROOT}/submission_sc_${EXP_NAME}.csv"

  # (주의) inference는 백그라운드(&)로 돌리면 2개가 동시에 떠서 VRAM 터질 수 있음
  #       순차 실행을 원하면 '&' 없이 실행하세요.
  nohup python3 -u inference_logit_sc.py > "${INFER_LOG}" 2>&1
  echo "[OK] Infer logit+SC done -> ${OUT_PATH} (log: ${INFER_LOG})"
}


# ===== 실험 1) LR=5e-5, r=64, alpha=128 =====
#run_one "8" "1e-4" "128" "128"

# ===== 실험 2) r=128, alpha=128 (LR은 동일 5e-5 유지) =====
#run_one "9" "1.3e-4" "128" "128"

#run_one "10" "1e-4" "128" "192"

#run_one "11" "8e-5" "128" "256"


#run_one "12" "9e-5" "128" "128"
#run_one "13" "1e-4" "144" "128"
#run_one "14" "1.05e-4" "128" "128"
#run_one "15" "1e-4" "128" "128" # WEIGHT_DECAY=0.01
run_one "16" "1e-4" "128" "128" # WEIGHT_DECAY=0.02

echo "[DONE] All experiments completed."
