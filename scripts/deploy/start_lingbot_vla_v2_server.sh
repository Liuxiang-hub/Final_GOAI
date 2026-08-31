#!/usr/bin/env bash
set -euo pipefail

# The model server returns the full trained 50-step action chunk. The robot
# client executes 15 steps, observes again, and applies the timestamp-aligned
# filter configured in configs/deploy_temporal_adaptive.yaml.

model_path="${MODEL_PATH:?Set MODEL_PATH to the selected hf_ckpt directory}"
port="${PORT:-8006}"
prediction_horizon=50

test -s "${model_path}/model.safetensors.index.json"

exec python -m deploy.lingbot_vla_v2_policy \
  --model_path "${model_path}" \
  --use_length "${prediction_horizon}" \
  --chunk_ret true \
  --use_bf16 true \
  --use_fp32 false \
  --use_compile true \
  --port "${port}"
