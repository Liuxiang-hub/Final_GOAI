#!/usr/bin/env bash
set -euo pipefail

# The model server returns the full trained 50-step action chunk. The robot
# client executes 15 steps, observes again, and applies the timestamp-aligned
# filter configured in configs/deploy_temporal_adaptive.yaml.

model_path="${MODEL_PATH:?Set MODEL_PATH to the selected hf_ckpt directory}"
port="${PORT:-8006}"
execution_horizon="${EXECUTION_HORIZON:-15}"
prediction_horizon=50

if (( execution_horizon < 1 || execution_horizon > prediction_horizon )); then
  echo "Require 1 <= EXECUTION_HORIZON <= 50." >&2
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
