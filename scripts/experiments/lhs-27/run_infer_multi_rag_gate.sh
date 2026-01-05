#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== 실험 식별 =====
export EXP_NAME="Qwen2.5-32B-LogitSC-RAGGate_ver20"
export OUTPUT_ROOT="$(pwd)"

# ===== 모델 =====
export BASE_MODEL="unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
export ADAPTER_MODEL="NLP-07-ODQA/Qwen2.5-32B-Instruct-bnb-4bit_15"

# ===== 데이터/출력 =====
export TEST_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/test/test.csv"
export OUT_PATH="./submission_sc_rag_ver20.csv"

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

# ===== RAG 문서 정제/컨텍스트 =====
export RAG_MAX_DOCS=20
export RAG_SENTENCES=10
export RAG_PER_DOC_SENT=3
export RAG_CTX_CHARS=1800

export RAG_USE_LLM_GATE=1                  # 0=AUTO(추천), 1=LLM gate
export RAG_TRY_ONLY_IF_SC_NOT_ACCEPTED=1   # SC가 이미 확정이면 RAG 금지
export RAG_AUTO_TRIGGER_K=2                # AUTO일 때 triggers>=2면 RAG 시도

export RAG_QUERY_EMBED=1
export RAG_EMBED_MODEL=intfloat/multilingual-e5-base   # FAISS 만들 때 쓴 모델과 동일해야 함
export RAG_EMBED_DEVICE=cpu                            # cuda도 가능

# 멀티쿼리/검색 폭
export RAG_MULTI_Q=2

# 영문 타이틀 컷 강도(0.55~0.70 사이 추천)
export RAG_MIN_HANGUL_RATIO=0.55

# ===== Debug =====
export FIRST_SAMPLE_DEBUG=1
export SC_DEBUG=0
export SC_DEBUG_MAX=200
export RAG_DEBUG=0
export RAG_DEBUG_MAX=200
export RAG_DEBUG_JSONL="${OUTPUT_ROOT}/rag_debug_denoised_5.jsonl"
export OUT_PROBS_PATH="avg_probs_ver20.csv"
LOG_FILE="${OUTPUT_ROOT}/infer_${EXP_NAME}.log"

nohup python3 -u inference_logit_sc_rag_gate.py > "${LOG_FILE}" 2>&1 
echo "[OK] started -> ${OUT_PATH}"
echo "[OK] log -> ${LOG_FILE}"
echo "[OK] rag jsonl -> ${RAG_DEBUG_JSONL}"


