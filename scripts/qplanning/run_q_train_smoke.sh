#!/usr/bin/env bash
set -euo pipefail

# GOAI Q-function compatibility/performance smoke test.
# It intentionally trains only on the frozen 510-episode training split.

Q_ENV="${Q_ENV:-/root/autodl-tmp/conda_envs/qplanning}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/25w/data/lerobot_v30_joint}"
SPLIT_FILE="${SPLIT_FILE:-${DATASET_ROOT}/splits/train_episodes.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/qplanning_artifacts/goai_q_smoke}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/qplanning_hf}"
STEPS="${STEPS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
USE_BF16="${USE_BF16:-1}"

export HF_HOME
unset OMP_NUM_THREADS

mapfile -t EPISODES < <(sed '/^[[:space:]]*$/d' "${SPLIT_FILE}")
if [[ "${#EPISODES[@]}" -ne 510 ]]; then
  echo "Expected 510 training episodes, found ${#EPISODES[@]} in ${SPLIT_FILE}" >&2
  exit 2
fi

EPISODE_LIST="$(IFS=,; echo "${EPISODES[*]}")"

if [[ "${USE_BF16}" == "1" ]]; then
  LAUNCHER=("${Q_ENV}/bin/accelerate" launch --num_processes=1 --mixed_precision=bf16 -m lerobot.scripts.lerobot_train)
else
  LAUNCHER=("${Q_ENV}/bin/python" -m lerobot.scripts.lerobot_train)
fi

exec "${LAUNCHER[@]}" \
  --policy.type=q_function \
  --policy.push_to_hub=false \
  --job_name=goai_q_smoke \
  --output_dir="${OUTPUT_DIR}" \
  --policy.dino_model_name=facebook/dinov2-large \
  --policy.text_encoder_model=google/t5-v1_1-base \
  --policy.use_text_conditioning=true \
  --policy.dim_model=1024 \
  --policy.n_heads=16 \
  --policy.dim_feedforward=4096 \
  --policy.n_decoder_layers=18 \
  --policy.camera_keys='[observation.images.cam_high,observation.images.cam_left_wrist,observation.images.cam_right_wrist]' \
  --policy.h=50 \
  --policy.gamma=0.99 \
  --policy.num_bins=101 \
  --policy.v_min=-0.01 \
  --policy.v_max=1.01 \
  --policy.hl_gauss_sigma=0.0075 \
  --policy.target_tau=0.005 \
  --policy.reward_mode=all_success \
  --policy.step_reward=0.0 \
  --policy.optimizer_lr=3e-4 \
  --policy.optimizer_lr_backbone=9e-5 \
  --policy.optimizer_weight_decay=1e-4 \
  --policy.lr_scheduler=cosine_decay_with_warmup \
  --policy.lr_warmup_steps=2000 \
  --policy.lr_decay_steps=40000 \
  --policy.lr_decay_min=1e-6 \
  --policy.device=cuda \
  --dataset.repo_id=local/goai_piper_x \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.episodes="[${EPISODE_LIST}]" \
  --dataset.use_imagenet_stats=false \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_checkpoint=false \
  --log_freq=1 \
  --num_workers="${NUM_WORKERS}" \
  --seed=42 \
  --cudnn_deterministic=false \
  --test_split_ratio=0.0 \
  --test_freq=0 \
  --eval_freq=0 \
  --wandb.enable=false
