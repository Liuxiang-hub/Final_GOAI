#!/usr/bin/env bash
set -euo pipefail

# The model still predicts its trained 50-step action chunk. The server returns
# execution_horizon + blend_steps so the robot client can execute 15 steps,
# retain the following 5, and blend those with the next chunk after replanning.

model_path="${MODEL_PATH:?Set MODEL_PATH to the selected hf_ckpt directory}"
port="${PORT:-8006}"
execution_horizon="${EXECUTION_HORIZON:-15}"
blend_steps="${CHUNK_BLEND_STEPS:-5}"
return_horizon=$((execution_horizon + blend_steps))

if (( execution_horizon < 1 || blend_steps < 0 || return_horizon > 50 )); then
  echo "Require EXECUTION_HORIZON >= 1, CHUNK_BLEND_STEPS >= 0, and their sum <= 50." >&2
  exit 2
fi

test -s "${model_path}/model.safetensors.index.json"

exec python -m deploy.lingbot_vla_v2_policy \
  --model_path "${model_path}" \
  --use_length "${return_horizon}" \
  --chunk_ret true \
  --use_bf16 true \
  --use_fp32 false \
  --use_compile true \
  --port "${port}"
