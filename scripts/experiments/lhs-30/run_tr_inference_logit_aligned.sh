#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== 실험 식별 =====
export CAMPER_ID="T8164"
export EXP_NAME="Qwen2.5_fixed"
export HF_ORG="NLP-07-ODQA"
export OUTPUT_ROOT="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/outputs/"

# ===== 베이스 모델 (추론용) =====
export BASE_MODEL="unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
export TEST_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/test/test.csv"
export SEED=42

# ===== 프롬프트/토큰 설정 (기존과 동일) =====
export USE_CHAT_TEMPLATE=1
export MAX_SEQ_LENGTH=4096
export TRUNCATION_SIDE=left

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

# ===== 여러 어댑터를 한 번에 추론 =====
# 여기에 HF repo / 로컬 경로를 추가
ADAPTER_MODELS=(
  #"NLP-07-ODQA/Qwen2.5-32B-Instruct-bnb-4bit_8"
  "NLP-07-ODQA/Qwen2.5-32B-Instruct-bnb-4bit_15"
  #"NLP-07-ODQA/Qwen3-32B-bnb-4bit-CoT_2"
  #"NLP-07-ODQA/Qwen2.5-32B-Instruct-bnb-4bit"
  #"NLP-07-ODQA/Qwen2.5-32B-Instruct-bnb-4bit_noCoT"
  #"NLP-07-ODQA/Qwen2.5-32B-Instruct-bnb-4bit_11"
  #"NLP-07-ODQA/Qwen2.5-32B-Instruct-bnb-4bit_10"
  #"NLP-07-ODQA/qwen3-32b-qlora-v4.1"
  #"NLP-07-ODQA/qwen3-32b-qlora-v6.2"
)

PROB_DIR="${OUTPUT_ROOT}/prob_dumps/${EXP_NAME}"
mkdir -p "${PROB_DIR}"

#PROB_FILES=()

for ADP in "${ADAPTER_MODELS[@]}"; do
  export ADAPTER_MODEL="${ADP}"

  # 파일명 안전화
  SAFE_NAME=$(echo "${ADP}" | sed 's#[/:]#_#g')
  PROB_OUT="${PROB_DIR}/probs_${SAFE_NAME}.csv"
  SUB_OUT="${PROB_DIR}/submission_${SAFE_NAME}.csv"
  LOG_OUT="${PROB_DIR}/infer_${SAFE_NAME}.log"

  export OUT_PATH="${PROB_OUT}"

  echo "[RUN] adapter=${ADAPTER_MODEL}"
  nohup python3 -u inference_logit_sc_aligned.py > "${LOG_OUT}" 2>&1

  echo "[OK] wrote probs: ${PROB_OUT} (log: ${LOG_OUT})"

  # 제출용(id,answer)만 따로 저장
  SUB_OUT="${SUB_OUT}" python3 - << 'PY'
import os, pandas as pd
prob_out = os.environ["OUT_PATH"]
sub_out  = os.environ["SUB_OUT"]
df = pd.read_csv(prob_out)
df[["id","answer"]].to_csv(sub_out, index=False)
print("[OK] wrote submission:", sub_out)
PY
    #PROB_FILES+=("${PROB_OUT}")
done

# ===== Soft voting (평균 확률) =====
#ENS_SUB="${OUTPUT_ROOT}/submission_ens_${EXP_NAME}.csv"
#ENS_AVG="${OUTPUT_ROOT}/avg_probs_${EXP_NAME}.csv"
#export ENS_PROB_OUT="${ENS_AVG}"

#python3 ensemble_soft_voting.py --out "${ENS_SUB}" "${PROB_FILES[@]}"

#echo "[DONE] Ensemble submission: ${ENS_SUB}"
#echo "[DONE] Avg probs dump: ${ENS_AVG}"
