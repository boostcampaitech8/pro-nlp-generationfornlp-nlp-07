#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export EXP_NAME="Qwen2.5-32B-LogitSC-ChoiceRAG"
export OUTPUT_ROOT="$(pwd)"

# ===== 모델 =====
export BASE_MODEL="unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
export ADAPTER_MODEL="NLP-07-ODQA/Qwen2.5-32B-Instruct-bnb-4bit_15"

# ===== 데이터/출력 =====
export TEST_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/test/test.csv"
export OUT_PATH="./submission_choice_rag_ver16.csv"

# ===== 공통 =====
export SEED=42
export MAX_SEQ_LENGTH=4096

# ===== SC (기존과 동일) =====
export MARGIN_THRESHOLD=0.75
export ENTROPY_THRESHOLD=1.40
export TOP1P_THRESHOLD=0.40

export MAX_SC_RATE=0.10
export SC_TRIGGER_K=3
export CONF_MARGIN=2.0
export CONF_TOP1P=0.80
export CONF_ENTROPY=0.60
export HARD_MARGIN=0.40

export SC_SAMPLES_BASE=9
export SC_TEMPERATURE_BASE=0.65
export SC_SAMPLES_HARD=13
export SC_TEMPERATURE_HARD=0.80

export SC_ACCEPT_MARGIN=1.0
export SC_ACCEPT_MARGIN_STRICT=1.8

# ===== Choice-conditioned RAG =====
export RAG_MAX_DOCS=30
export RAG_TOP_DOCS=3
export RAG_PER_DOC_SENT=4
export RAG_SENTENCES=10
export RAG_CTX_CHARS=1600
export RAG_MIN_DOC_SCORE=1.5

# 선택지별 컨텍스트 후보 중 몇 개까지 비교(속도/성능 trade-off)
export CHOICE_RAG_TOPK=2
# best_by_margin | best_by_top1p
export CHOICE_RAG_MODE="best_by_margin"

# RAG 채택 최소 개선폭(노이즈 플립 방지)
export RAG_ACCEPT_D_MARGIN=0.10
export RAG_ACCEPT_D_TOP1P=0.02
export RAG_ACCEPT_D_ENT=0.05

# ===== Debug =====
export FIRST_SAMPLE_DEBUG=1
export SC_DEBUG=0
export SC_DEBUG_MAX=200
export RAG_DEBUG=0
export RAG_DEBUG_MAX=200
export RAG_DEBUG_JSONL="${OUTPUT_ROOT}/rag_debug_choice.jsonl"

LOG_FILE="${OUTPUT_ROOT}/infer_${EXP_NAME}.log"

nohup python3 -u inference_logit_sc_rag_gate_ver16.py > "${LOG_FILE}" 2>&1 &
echo "[OK] started -> ${OUT_PATH}"
echo "[OK] log -> ${LOG_FILE}"
echo "[OK] rag jsonl -> ${RAG_DEBUG_JSONL}"
