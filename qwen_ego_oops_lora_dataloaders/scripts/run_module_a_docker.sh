#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-qwen-omd-dataloaders:latest}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/amit/online_mistake_detection/data/videos-processed-720p}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/amit/online_mistake_detection/outputs}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/wandb}"
REPORT_TO="${REPORT_TO:-tensorboard}"

mkdir -p "${OUTPUT_ROOT}" "${HF_CACHE}" "${WANDB_DIR}"

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${PROJECT_DIR}:/workspace/qwen_ego_oops_lora_dataloaders" \
  -v "${DATA_ROOT}:${DATA_ROOT}:ro" \
  -v "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
  -v "${HF_CACHE}:/cache/huggingface" \
  -v "${WANDB_DIR}:${WANDB_DIR}" \
  -e WANDB_DIR="${WANDB_DIR}" \
  -e WANDB_PROJECT="${WANDB_PROJECT:-qwen-omd}" \
  -e WANDB_MODE="${WANDB_MODE:-online}" \
  -w /workspace/qwen_ego_oops_lora_dataloaders \
  "${IMAGE_NAME}" \
  python scripts/train_module_a_unsloth.py \
    --max-videos 50 \
    --max-steps 100 \
    --video-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/module_a_qwen35_2b_lora" \
    --report-to "${REPORT_TO}"
