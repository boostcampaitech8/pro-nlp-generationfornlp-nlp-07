#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== 실험 식별 =====
export CAMPER_ID="T8164"
export EXP_NAME="Qwen3-32B-bnb-4bit-CoT_2.3"

# (옵션) outputs 루트 변경
export OUTPUT_ROOT="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/outputs/"

# ===== adapter 경로 (학습 결과를 그대로 사용) =====
#ADAPTER_DIR="${OUTPUT_ROOT}/${CAMPER_ID}/${EXP_NAME}/best_model"
export ADAPTER_MODEL="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/outputs/T8164/Qwen3-32B-bnb-4bit-CoT_2/best_model"

# ===== Inference I/O =====
export TEST_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/test/test.csv"
export OUT_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/outputs/submission_sc_${EXP_NAME}.csv"
export SEED=42

INFER_LOG="infer_sc_${EXP_NAME}.log"

# ===== Gate thresholds =====
export MARGIN_THRESHOLD=0.75
export ENTROPY_THRESHOLD=1.40
export TOP1P_THRESHOLD=0.40

# ===== v3 guard / budget =====
export MAX_SC_RATE=0.10
export SC_TRIGGER_K=3
export CONF_MARGIN=2.0
export CONF_TOP1P=0.80
export CONF_ENTROPY=0.60
export HARD_MARGIN=0.35

# ===== SC params =====
export SC_SAMPLES_BASE=9
export SC_TEMPERATURE_BASE=0.65
export SC_SAMPLES_HARD=13
export SC_TEMPERATURE_HARD=0.80

# ===== SC accept =====
export SC_ACCEPT_MARGIN=1.0
export SC_ACCEPT_MARGIN_STRICT=2.0

# ===== Debug (optional) =====
export FIRST_SAMPLE_DEBUG=1
export SC_DEBUG=0
export SC_DEBUG_MAX=200

# ===== Run inference only =====
# NOTE: inference_logit_sc.py 내부에서 BASE_MODEL을 기본값으로 로드하도록 되어있으면 BASE_MODEL export 불필요.
# 만약 필요하면 아래 한 줄을 추가:
# export BASE_MODEL="unsloth/Qwen3-32B-bnb-4bit"

nohup python3 -u inference_logit_sc.py > "${INFER_LOG}" 2>&1 &
echo "[OK] Inference started -> ${OUT_PATH}"
echo "[INFO] Adapter: ${ADAPTER_MODEL}"
echo "[INFO] Log: ${INFER_LOG}"
