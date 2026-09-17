#!/usr/bin/env bash
set -euo pipefail

# Legacy rollback server: returns a full 50-step chunk for the pre-RTC temporal
# ensemble client. The current entry is start_lingbot_vla_v2_rtc_server.sh.

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
