#!/usr/bin/env bash
set -euo pipefail

# Formal offline Q-function initialization for GOAI.
# The frozen 60-episode validation and 30-episode test splits are never passed
# to the training dataset. They are evaluated independently after training.

Q_ENV="${Q_ENV:-/root/autodl-tmp/conda_envs/qplanning}"
DATASET_ROOT="${DATASET_ROOT:-/root/autodl-tmp/25w/data/lerobot_v30_joint}"
TRAIN_SPLIT="${TRAIN_SPLIT:-${DATASET_ROOT}/splits/train_episodes.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/qplanning_artifacts/goai_q_formal_20260905}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/qplanning_hf}"
STEPS="${STEPS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-48}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-25}"

export HF_HOME
# Both pretrained encoders are pre-cached. Avoid fragile network HEAD requests
# interrupting or delaying a long formal run.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset OMP_NUM_THREADS

mapfile -t EPISODES < <(sed '/^[[:space:]]*$/d' "${TRAIN_SPLIT}")
if [[ "${#EPISODES[@]}" -ne 510 ]]; then
  echo "Expected 510 training episodes, found ${#EPISODES[@]} in ${TRAIN_SPLIT}" >&2
  exit 2
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing formal output: ${OUTPUT_DIR}" >&2
  exit 3
fi

EPISODE_LIST="$(IFS=,; echo "${EPISODES[*]}")"

exec "${Q_ENV}/bin/accelerate" launch \
  --num_processes=1 \
  --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --policy.type=q_function \
  --policy.push_to_hub=false \
  --job_name=goai_q_formal \
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
  --policy.lr_decay_steps="${STEPS}" \
  --policy.lr_decay_min=1e-6 \
  --policy.device=cuda \
  --dataset.repo_id=local/goai_piper_x \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.episodes="[${EPISODE_LIST}]" \
  --dataset.use_imagenet_stats=false \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --num_workers="${NUM_WORKERS}" \
  --seed=42 \
  --cudnn_deterministic=false \
  --test_split_ratio=0.0 \
  --test_freq=0 \
  --eval_freq=0 \
  --wandb.enable=false
