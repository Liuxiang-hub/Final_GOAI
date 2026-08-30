#!/usr/bin/env bash
set -euo pipefail

# The model and server return the full trained 50-step action chunk. The robot
# client executes 15 steps, retains steps 16–20, then requests a fresh 50-step
# chunk and blends the retained five with the new chunk's first five.

model_path="${MODEL_PATH:?Set MODEL_PATH to the selected hf_ckpt directory}"
port="${PORT:-8006}"
execution_horizon="${EXECUTION_HORIZON:-15}"
blend_steps="${CHUNK_BLEND_STEPS:-5}"
prediction_horizon=50
required_horizon=$((execution_horizon + blend_steps))

if (( execution_horizon < 1 || blend_steps < 0 || required_horizon > prediction_horizon )); then
  echo "Require EXECUTION_HORIZON >= 1, CHUNK_BLEND_STEPS >= 0, and their sum <= 50." >&2
  exit 2
fi

test -s "${model_path}/model.safetensors.index.json"

exec python -m deploy.lingbot_vla_v2_policy \
  --model_path "${model_path}" \
  --use_length "${prediction_horizon}" \
  --chunk_ret true \
  --use_bf16 true \
  --use_fp32 false \
  --use_compile true \
  --port "${port}"
