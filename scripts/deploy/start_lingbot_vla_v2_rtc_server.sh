#!/usr/bin/env bash
set -euo pipefail

# Run from the LingBot-VLA 2.0 repository after applying the RTC patch.
# Guided RTC requests intentionally use eager sampling because PiGDM requires
# an action-space vector-Jacobian product. GPU latency must be benchmarked on
# the target RTX 6000D before connecting a powered robot.

model_path="${MODEL_PATH:?Set MODEL_PATH to the selected global_step_8884 hf_ckpt directory}"
port="${PORT:-8006}"
prediction_horizon=50

test -s "${model_path}/model.safetensors.index.json"

exec python -m deploy.lingbot_vla_v2_policy \
  --model_path "${model_path}" \
  --use_length "${prediction_horizon}" \
  --chunk_ret true \
  --use_bf16 true \
  --use_fp32 false \
  --use_compile false \
  --port "${port}"
