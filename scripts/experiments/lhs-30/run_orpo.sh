#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07"
OUT="${ROOT}/outputs/orpo_run"
#mkdir -p "${OUT}"



# =========================
# (0) 공통 설정
# =========================
export BASE_MODEL="unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
export MAX_SEQ_LENGTH=4096
export SEED=42

# =========================
# (1) train 전체 확률 덤프(채굴) - SC 끔
# =========================
# 기존 best adapter(없으면 빈 값으로 두면 base로만 채굴)
export ADAPTER_MODEL="NLP-07-ODQA/Qwen2.5-32B-Instruct-bnb-4bit_15"

export TEST_PATH="${ROOT}/data/train/train.csv"
export OUT_PATH="${OUT}/probs_train.csv"

export MAX_SC_RATE=0
export FIRST_SAMPLE_DEBUG=0
export SC_DEBUG=0

#python3 -u inference_logit_sc_aligned.py
echo "[OK] mined train probs -> ${OUT_PATH}"

# =========================
# (2) ORPO pair(jsonl) 생성
# =========================
#python3 -u make_orpo_pairs_from_probs.py \
#  --train_csv "${ROOT}/data/train/train.csv" \
#  --probs_csv "${OUT}/probs_train.csv" \
#  --out_jsonl "${OUT}/pairs.jsonl"

echo "[OK] built pairs -> ${OUT}/pairs.jsonl"

# =========================
# (3) ORPO 학습 (answer-only)
# =========================
export PAIRS_JSONL="${OUT}/pairs.jsonl"
export OUTPUT_DIR="${OUT}/orpo_model"

# ORPO 시작점: 위에서 채굴에 쓴 adapter 이어학습 추천(없으면 ""로)
export START_ADAPTER_MODEL=""

export LORA_R=64
export LORA_ALPHA=64
export LORA_DROPOUT=0.0

export ORPO_LR=5e-6
export ORPO_EPOCHS=2
export ORPO_GRAD_ACCUM=8
export ORPO_BSZ=1
export ORPO_BETA=0.1

#python3 -u train_orpo_answer_only.py
echo "[OK] ORPO trained -> ${OUTPUT_DIR}/orpo_adapter"

# =========================
# (4) ORPO 어댑터로 test 추론 (probs + submission 2개 출력)
# =========================
export ADAPTER_MODEL="${OUTPUT_DIR}/checkpoint-254"
export TEST_PATH="${ROOT}/data/test/test.csv"

# (원하면 SC 켤 수 있지만, 우선은 base logit 성능 확인을 위해 SC 끄는 걸 추천)
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

PROB_TEST="./probs_test_orpo_1.csv"
SUB_TEST="./submission_test_orpo_1.csv"
export OUT_PATH="${PROB_TEST}"

python3 -u inference_logit_sc_aligned.py
echo "[OK] test probs -> ${PROB_TEST}"

# 제출용(id,answer)만 추출
SUB_OUT="${SUB_TEST}" python3 - << 'PY'
import os, pandas as pd
prob_out = os.environ["OUT_PATH"]
sub_out  = os.environ["SUB_OUT"]
df = pd.read_csv(prob_out)
df[["id","answer"]].to_csv(sub_out, index=False)
print("[OK] wrote submission:", sub_out)
PY

echo "[DONE] submission -> ${SUB_TEST}"
