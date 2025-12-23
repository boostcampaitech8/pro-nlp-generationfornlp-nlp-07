#!/bin/bash
set -e

export GEMINI_API_KEY=""
export GEMINI_MODEL="gemini-3-flash-preview"
export GEMINI_ENDPOINT_BASE="https://generativelanguage.googleapis.com"

nohup python3 -u generate_cot_train_gemini.py \
  --input_csv "./data/train.csv" \
  --output_csv "./data/train_with_cot.csv" \
  --api_key "$GEMINI_API_KEY" \
  --model_name "$GEMINI_MODEL" \
  --sleep_sec 1.2 \
  --max_attempts 2 \
  --save_every 50 \
  --preview_every 1 \
  --preview_chars 450 \
  > generate_cot_csv.log 2>&1 &

