#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== 실험 식별 =====
export CAMPER_ID="T8164"
export EXP_NAME="Qwen2.5-32B_tapt_then_cot"
export OUTPUT_ROOT="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/outputs"

# ===== 원본 베이스 모델 =====
export TRAIN_BASE_MODEL_ORIG="unsloth/Qwen2.5-32B-Instruct-bnb-4bit"

# ===== TAPT 어댑터 저장 경로(중간 체크포인트 없이 최종 1회 저장) =====
export TAPT_ADAPTER_DIR="${OUTPUT_ROOT}/${CAMPER_ID}/tapt_adapter_final"

# ===== 최종(CoT) 어댑터 저장 경로 =====
export COT_OUT_DIR="${OUTPUT_ROOT}/${CAMPER_ID}/cot_final"

mkdir -p "${OUTPUT_ROOT}/${CAMPER_ID}"

# ===== CoT 학습 설정 (기존 네 sh 값 유지) =====
export MAX_SEQ_LENGTH=4096
export LORA_R=128
export LORA_ALPHA=128
export LEARNING_RATE=1e-4
export EPOCHS=2
export GRAD_ACCUM=8
export WEIGHT_DECAY=0.01

export USE_COT_TRAIN=1
export COT_TRAIN_PATH="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/train/train_with_cot.csv"

TAPT_LOG="${OUTPUT_ROOT}/${CAMPER_ID}/tapt.log"
COT_LOG="${OUTPUT_ROOT}/${CAMPER_ID}/cot.log"

echo "[1/2] TAPT(adapter) start -> ${TAPT_ADAPTER_DIR}"
# python3 -u train_tapt_qlora.py \
#   --train_csv /data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/data/train/train.csv \
#   --base_model "${TRAIN_BASE_MODEL_ORIG}" \
#   --out_dir "${TAPT_ADAPTER_DIR}" \
#   --seq_len 1024 \
#   --batch_size 1 \
#   --grad_accum 16 \
#   --lr 2e-4 \
#   --epochs 1.0 \
#   --use_qlora 1 \
#   --lora_r "${LORA_R}" \
#   --lora_alpha "${LORA_ALPHA}" \
#   > "${TAPT_LOG}" 2>&1

echo "[1/2] TAPT done. Resume adapter for CoT training."

# ===== (중요) train_cot.py가 이 변수를 읽어서 TAPT 어댑터를 로드하게 함 =====
export BASE_MODEL="${TRAIN_BASE_MODEL_ORIG}"
export RESUME_ADAPTER_DIR="${TAPT_ADAPTER_DIR}"
export OUTPUT_DIR="${COT_OUT_DIR}"

echo "[2/2] CoT train start -> ${COT_OUT_DIR}"
nohup python3 -u train_cot.py > "${COT_LOG}" 2>&1

echo "Launched. logs:"
echo " - ${TAPT_LOG}"
echo " - ${COT_LOG}"
