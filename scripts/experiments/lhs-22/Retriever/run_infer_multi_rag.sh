#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== 실험 식별 =====
export CAMPER_ID="T8164"
export EXP_NAME="ENS_rag_gate_ver4"
export HF_ORG="NLP-07-ODQA"
export OUTPUT_ROOT="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/outputs/"

# ===== 베이스 모델 (추론용) =====
export BASE_MODEL="unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
export TEST_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/test/test.csv"
export SEED=42

# ===== 프롬프트/토큰 설정 =====
export USE_CHAT_TEMPLATE=1
export MAX_SEQ_LENGTH=4096
export TRUNCATION_SIDE=left

# ===== Base confidence / SC gate (원본과 동일) =====
export MARGIN_THRESHOLD=0.75
export ENTROPY_THRESHOLD=1.40
export TOP1P_THRESHOLD=0.40

# ===== v3 guard =====
export MAX_SC_RATE=0.10
export SC_TRIGGER_K=3
export CONF_MARGIN=2.0
export CONF_TOP1P=0.80
export CONF_ENTROPY=0.60
export HARD_MARGIN=0.40

# ===== SC params =====
export SC_SAMPLES_BASE=9
export SC_TEMPERATURE_BASE=0.65
export SC_SAMPLES_HARD=13
export SC_TEMPERATURE_HARD=0.80

# ===== SC accept =====
export SC_ACCEPT_MARGIN=1.0
export SC_ACCEPT_MARGIN_STRICT=1.8

# ===== RAG gate (NEW: 판단만, retrieval은 아직 없음) =====
# RAG 사용 최대 비율 (로그/분석 및 추후 비용 제어용)
export RAG_MAX_RATE=0.30

# RAG confidence 기준 (낮을수록 RAG)
export RAG_CONF_THR=1.15

# RAG 분포 기준 (지식 부족 신호)
export RAG_ENTROPY_NORM_THR=0.62
export RAG_MARGIN_THR=0.82
export RAG_TOP1P_THR=0.57

# 너무 hard한 케이스는 RAG보다 SC 우선
export RAG_AVOID_HARD=1
export RAG_RUN=1
export RAG_DENSE_DEVICE=cpu
export RAG_RERANK_DEVICE=cpu

export RAG_INDEX=/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/scripts/experiments/lhs-22/Retriever/wiki.faiss
export RAG_DOCSTORE=/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/scripts/experiments/lhs-22/Retriever/docstore.pt
export RAG_BM25_VOCAB=/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/scripts/experiments/lhs-22/Retriever/bm25_vocab.joblib
export RAG_BM25_TF=/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/scripts/experiments/lhs-22/Retriever/bm25_tf_matrix.npz
export RAG_BM25_META=/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/scripts/experiments/lhs-22/Retriever/bm25_meta.npz

export RAG_DEBUG=1
export RAG_DEBUG_MAX=5
# ===== Debug =====
export FIRST_SAMPLE_DEBUG=0
export SC_DEBUG=0
export SC_DEBUG_MAX=200
export RAG_DENSE_MODEL="intfloat/multilingual-e5-base"

# ===== 여러 어댑터를 한 번에 추론 =====
ADAPTER_MODELS=(
  "NLP-07-ODQA/Qwen2.5-32B-Instruct-bnb-4bit_15"
)

PROB_DIR="./${EXP_NAME}"
mkdir -p "${PROB_DIR}"

for ADP in "${ADAPTER_MODELS[@]}"; do
  export ADAPTER_MODEL="${ADP}"

  SAFE_NAME=$(echo "${ADP}" | sed 's#[/:]#_#g')
  PROB_OUT="${PROB_DIR}/probs_${SAFE_NAME}.csv"
  SUB_OUT="${PROB_DIR}/submission_${SAFE_NAME}.csv"
  LOG_OUT="${PROB_DIR}/infer_${SAFE_NAME}.log"

  export OUT_PATH="${PROB_OUT}"

  echo "[RUN] adapter=${ADAPTER_MODEL}"
  nohup python3 -u inference_rag_ver4.py > "${LOG_OUT}" 2>&1
  echo "[OK] wrote probs: ${PROB_OUT} (log: ${LOG_OUT})"

  SUB_OUT="${SUB_OUT}" python3 - << 'PY'
import os, pandas as pd
prob_out = os.environ["OUT_PATH"]
sub_out  = os.environ["SUB_OUT"]
df = pd.read_csv(prob_out)
df[["id","answer"]].to_csv(sub_out, index=False)
print("[OK] wrote submission:", sub_out)
PY
done

echo "[DONE] RAG gate inference finished"
