#!/usr/bin/env bash
set -euo pipefail

# The model still predicts its trained 50-step action chunk. Deployment only
# returns the first 25 steps; after executing them, the client sends a fresh
# camera/state observation and requests the next chunk.

model_path="${MODEL_PATH:?Set MODEL_PATH to the selected hf_ckpt directory}"
port="${PORT:-8006}"
execution_horizon="${EXECUTION_HORIZON:-25}"

if (( execution_horizon < 1 || execution_horizon > 50 )); then
  echo "EXECUTION_HORIZON must be between 1 and the trained chunk size (50)." >&2
  exit 2
fi

test -s "${model_path}/model.safetensors.index.json"

exec python -m deploy.lingbot_vla_v2_policy \
  --model_path "${model_path}" \
  --use_length "${execution_horizon}" \
  --chunk_ret true \
  --use_bf16 true \
  --use_fp32 false \
  --use_compile true \
  --port "${port}"
