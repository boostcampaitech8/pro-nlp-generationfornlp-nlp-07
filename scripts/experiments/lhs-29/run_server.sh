#!/bin/bash
set -euo pipefail

LLAMA_SERVER="/data/ephemeral/home/llama.cpp/build/bin/llama-server"

MODEL="${MODEL_PATH:-./hf_models/models--unsloth--Qwen3-Next-80B-A3B-Instruct-GGUF/snapshots/9c071f8464c6736af88a78da6194b91a718c4ad0/Qwen3-Next-80B-A3B-Instruct-Q2_K_L.gguf}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

THREADS="${THREADS:-8}"
CTX="${CTX:-2048}"

mkdir -p logs

echo "[INFO] Starting llama-server"
echo "  bin   : $LLAMA_SERVER"
echo "  model : $MODEL"
echo "  addr  : http://$HOST:$PORT"
echo "  th/ctx: $THREADS / $CTX"

# 주의: 서버가 떠있는 동안 이 터미널은 점유됩니다.
# 필요하면 nohup으로 백그라운드 실행하세요.
exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  -t "$THREADS" \
  -tb 8 \
  -b 1024
  -c "$CTX" \
  -ub 256 \
  --prio 2
  --log-disable \
  2>&1 | tee "logs/llama_server_${PORT}.log"


