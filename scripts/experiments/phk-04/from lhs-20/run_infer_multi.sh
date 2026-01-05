#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== 경로 설정 =====
OUTPUT_DIR="../../../../submissions"
CACHE_DIR="/data/ephemeral/home/.cache/huggingface/hub/"

# ===== 테스트 데이터 및 시드 =====
export TEST_PATH="../../../../data/test/test.csv"
export SEED=42

# ===== 프롬프트/토큰 설정 =====
export MAX_SEQ_LENGTH=4096

# ===== Gate thresholds =====
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

# ===== Debug =====
export FIRST_SAMPLE_DEBUG=0
export SC_DEBUG=0
export SC_DEBUG_MAX=200

# ===== 캐시 정리 함수 =====
clear_cache() {
  local reason="${1:-cleanup}"
  echo ""
  echo "[CACHE] Clearing HuggingFace cache (${reason})..."
  CACHE_SIZE=$(du -sh "${CACHE_DIR}" 2>/dev/null | cut -f1 || echo "0B")
  echo "        Current size: ${CACHE_SIZE}"
  rm -rf "${CACHE_DIR}"*
  echo "[OK] Cache cleared"
}

# ===== 여러 어댑터를 한 번에 추론 =====
ADAPTER_MODELS=(
  "NLP-07-ODQA/qwen2.5-32b-it-qlora-v4.1"
  "NLP-07-ODQA/qwen2.5-32b-it-qlora-v4.2"
  "NLP-07-ODQA/qwen3-32b-qlora-v4.2"
  "NLP-07-ODQA/qwen3-32b-qlora-v5"
  "NLP-07-ODQA/qwen3-32b-qlora-v6.1"
  "NLP-07-ODQA/qwen3-32b-qlora-v6.2"
  "NLP-07-ODQA/qwen3-32b-qlora-v7.1"
  "NLP-07-ODQA/qwen3-32b-qlora-v7.2"
)

mkdir -p "${OUTPUT_DIR}"

# ===== 초기 캐시 정리 =====
clear_cache "initial"

PROB_FILES=()

for ADP in "${ADAPTER_MODELS[@]}"; do
  export ADAPTER_MODEL="${ADP}"

  # 파일명 안전화 - 마지막 / 이후만 추출
  SAFE_NAME=$(basename "${ADP}")
  PROB_OUT="${OUTPUT_DIR}/${SAFE_NAME}_sc_probs.csv"
  SUB_OUT="${OUTPUT_DIR}/${SAFE_NAME}_sc.csv"
  LOG_OUT="${OUTPUT_DIR}/infer_${SAFE_NAME}.log"

  export OUT_PATH="${PROB_OUT}"

  echo ""
  echo "============================================"
  echo "[RUN] Adapter: ${ADAPTER_MODEL}"
  echo "      Output: ${PROB_OUT}"
  echo "============================================"
  
  python3 -u inference_logit_sc_multi.py > "${LOG_OUT}" 2>&1
  
  echo "[OK] Inference complete"
  echo "     Probs: ${PROB_OUT}"
  echo "     Log: ${LOG_OUT}"

  # 제출용(id,answer)만 따로 저장
  SUB_OUT="${SUB_OUT}" python3 - << 'PY'
import os, pandas as pd
prob_out = os.environ["OUT_PATH"]
sub_out  = os.environ["SUB_OUT"]
df = pd.read_csv(prob_out)
df[["id","answer"]].to_csv(sub_out, index=False)
print(f"[OK] Submission saved: {sub_out}")
PY

  PROB_FILES+=("${PROB_OUT}")
  
  echo "[OK] Completed: ${SAFE_NAME}"
  
  # ===== 다음 모델 전 캐시 정리 =====
  clear_cache "next model"
done

echo ""
echo "============================================"
echo "[DONE] All adapters processed"
echo "       Total models: ${#ADAPTER_MODELS[@]}"
echo "       Output directory: ${OUTPUT_DIR}"
echo "============================================"
