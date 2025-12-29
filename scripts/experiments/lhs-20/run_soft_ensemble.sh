#!/bin/bash
set -euo pipefail

export OUTPUT_ROOT="/data/ephemeral/home/pro-nlp-generationfornlp-nlp-07/outputs/"
export EXP_NAME="ENS_softvote_aligned_7"

PROB_DIR="${OUTPUT_ROOT}/prob_dumps/ENS_softvote_aligned"

ENS_SUB="${OUTPUT_ROOT}/submission_ens_${EXP_NAME}.csv"
ENS_AVG="${OUTPUT_ROOT}/avg_probs_${EXP_NAME}.csv"
export ENS_PROB_OUT="${ENS_AVG}"

# probs*.csv 전부 수집
shopt -s nullglob
PROB_FILES=("${PROB_DIR}"/probs*.csv)
shopt -u nullglob

if [ ${#PROB_FILES[@]} -eq 0 ]; then
  echo "[ERROR] No prob files found: ${PROB_DIR}/probs*.csv"
  exit 1
fi

echo "[INFO] Found ${#PROB_FILES[@]} prob files"
for f in "${PROB_FILES[@]}"; do
  echo " - ${f}"
done

python3 -u ensemble_soft_voting.py --out "${ENS_SUB}" "${PROB_FILES[@]}"

echo "[DONE] Ensemble submission: ${ENS_SUB}"
echo "[DONE] Avg probs dump: ${ENS_AVG}"
