#!/bin/bash
set -euo pipefail

# ===== CUDA 메모리 안정화 =====
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== 경로/실험 식별 =====
export EXP_NAME="Qwen2.5-32B-RAG-gated"
export WORKDIR="$(pwd)"

# ===== 모델 설정 =====
export BASE_MODEL="unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
# (선택) 학습된 LoRA 어댑터가 있을 경우
export ADAPTER_MODEL="NLP-07-ODQA/Qwen2.5-32B-Instruct-bnb-4bit_15"

# ===== 데이터/출력 =====
export TEST_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/test/test.csv"
export OUT_PATH="./submission_rag.csv"

# ===== 공통 =====
export SEED=42
export MAX_SEQ_LENGTH=4096

# ===== 로그 =====
LOG_FILE="./infer_rag_${EXP_NAME}.log"

echo "[INFO] Workdir: ${WORKDIR}"
echo "[INFO] Base model: ${BASE_MODEL}"
echo "[INFO] Adapter: ${ADAPTER_MODEL}"
echo "[INFO] Test: ${TEST_PATH}"
echo "[INFO] Output: ${OUT_PATH}"
echo "[INFO] Log: ${LOG_FILE}"

# ===== 실행 =====
nohup python3 -u inference_logit_sc_rag_gate.py > "${LOG_FILE}" 2>&1 

echo "[OK] Inference started."
echo "[TIP] Tail log: tail -f ${LOG_FILE}"
