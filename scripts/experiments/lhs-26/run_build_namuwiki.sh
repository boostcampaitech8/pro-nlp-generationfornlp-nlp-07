#!/bin/bash
set -euo pipefail

# Build docstore + TF-IDF(CPU) + FAISS(GPU) for heegyu/namuwiki-sentences
# TF-IDF and FAISS run concurrently AFTER docstore is finished.

DOCSTORE_OUT="docstore.pt"
TFIDF_VEC="tfidf_vectorizer.joblib"
TFIDF_MAT="tfidf_matrix.npz"
FAISS_OUT="faiss_index.index"

# ====== 추천 기본값(조정 포인트) ======
MAX_DOCS="${MAX_DOCS:-1500000}"          
CHUNK_SENTS="${CHUNK_SENTS:-6}"
CHUNK_CHARS="${CHUNK_CHARS:-1200}"
STREAMING="${STREAMING:-1}"

# TF-IDF (CPU)
TFIDF_MAX_CHARS="${TFIDF_MAX_CHARS:-800}"
TFIDF_MAX_FEATURES="${TFIDF_MAX_FEATURES:-300000}"
TFIDF_MIN_DF="${TFIDF_MIN_DF:-3}"
TFIDF_NGRAM_MAX="${TFIDF_NGRAM_MAX:-1}"

# FAISS (GPU)
DENSE_MODEL="${DENSE_MODEL:-intfloat/multilingual-e5-base}"
FAISS_BATCH_SIZE="${FAISS_BATCH_SIZE:-256}"   # 128~256 사이. OOM 나면 128로
FAISS_DEVICE="${FAISS_DEVICE:-cuda}"
FAISS_MAX_CHARS="${FAISS_MAX_CHARS:-1200}"    # embedding은 너무 길면 느려짐

# CPU thread tuning (TF-IDF에만 주로 영향)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

echo "[CFG] MAX_DOCS=${MAX_DOCS}  TFIDF(max_features=${TFIDF_MAX_FEATURES}, min_df=${TFIDF_MIN_DF})  FAISS(bs=${FAISS_BATCH_SIZE})"
echo

# 1) docstore (must finish first)
echo "[1/3] Building docstore -> ${DOCSTORE_OUT}"
python3 step1_build_docstore_namuwiki.py \
  --out "${DOCSTORE_OUT}" \
  --chunk_sents "${CHUNK_SENTS}" \
  --chunk_chars "${CHUNK_CHARS}" \
  --max_docs "${MAX_DOCS}" \
  --streaming "${STREAMING}"

echo
echo "[2/3] Building TF-IDF (CPU) and FAISS (GPU) concurrently..."

# 2) TF-IDF (CPU) in background
(
  set -euo pipefail
  echo "[TFIDF] start"
  python3 step4_build_sparse_fast_tfidf_namuwiki.py \
    --docstore "${DOCSTORE_OUT}" \
    --vec_out "${TFIDF_VEC}" \
    --mat_out "${TFIDF_MAT}" \
    --max_chars "${TFIDF_MAX_CHARS}" \
    --max_features "${TFIDF_MAX_FEATURES}" \
    --min_df "${TFIDF_MIN_DF}" \
    --ngram_max "${TFIDF_NGRAM_MAX}"
  echo "[TFIDF] done"
) > "build_tfidf.log" 2>&1 &

TFIDF_PID=$!

# 3) FAISS dense (GPU) in background
(
  set -euo pipefail
  echo "[FAISS] start"
  python3 step3_build_faiss_dense_namuwiki.py \
    --docstore "${DOCSTORE_OUT}" \
    --out_faiss "${FAISS_OUT}" \
    --model "${DENSE_MODEL}" \
    --batch_size "${FAISS_BATCH_SIZE}" \
    --device "${FAISS_DEVICE}" \
    --use_ip 1 \
    --use_e5_prefix 1 \
    --max_docs "${MAX_DOCS}" \
    --max_chars "${FAISS_MAX_CHARS}"
  echo "[FAISS] done"
) > "build_faiss.log" 2>&1 &

FAISS_PID=$!

# Wait for both
wait "${TFIDF_PID}"
wait "${FAISS_PID}"

echo
echo "[OK] built:"
echo "  ${DOCSTORE_OUT}"
echo "  ${TFIDF_VEC}    (log: build_tfidf.log)"
echo "  ${TFIDF_MAT}    (log: build_tfidf.log)"
echo "  ${FAISS_OUT}    (log: build_faiss.log)"

echo
echo "[RUN] starting inference (ver15)..."

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== 실험 식별 =====
export EXP_NAME="Qwen2.5-32B-LogitSC-RAGGate_ver18"
export OUTPUT_ROOT="$(pwd)"

# ===== 모델 =====
export BASE_MODEL="unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
export ADAPTER_MODEL="NLP-07-ODQA/Qwen2.5-32B-Instruct-bnb-4bit_15"

# ===== 데이터/출력 =====
export TEST_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/test/test.csv"
export OUT_PATH="./submission_sc_rag_ver18.csv"

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
export RAG_SENTENCES=5 #10
export RAG_PER_DOC_SENT=2 #3
export RAG_CTX_CHARS=1400 #1800

# ===== Debug =====
export FIRST_SAMPLE_DEBUG=1
export SC_DEBUG=0
export SC_DEBUG_MAX=200
export RAG_DEBUG=0
export RAG_DEBUG_MAX=200
export RAG_DEBUG_JSONL="${OUTPUT_ROOT}/rag_debug_denoised_4.jsonl"
export OUT_PROBS_PATH="avg_probs_ver18.csv"
LOG_FILE="${OUTPUT_ROOT}/infer_${EXP_NAME}.log"

nohup python3 -u inference_logit_sc_rag_gate_ver16.py > "${LOG_FILE}" 2>&1 
echo "[OK] started -> ${OUT_PATH}"
echo "[OK] log -> ${LOG_FILE}"
echo "[OK] rag jsonl -> ${RAG_DEBUG_JSONL}"
