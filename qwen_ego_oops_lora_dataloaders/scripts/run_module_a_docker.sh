#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-qwen-omd-dataloaders:latest}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EGO_OOPS_ROOT="${EGO_OOPS_ROOT:-$(cd "${PROJECT_DIR}/../ego_oops" && pwd)}"
DATA_ROOT="${DATA_ROOT:-$(cd "${PROJECT_DIR}/../data/videos-processed-720p" && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/amit/online_mistake_detection/outputs}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/wandb}"
REPORT_TO="${REPORT_TO:-tensorboard}"
MAX_VIDEOS="${MAX_VIDEOS:-50}"
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-50}"
FPS="${FPS:-1.0}"
MIN_FRAMES="${MIN_FRAMES:-2}"
MAX_FRAMES="${MAX_FRAMES:-32}"
VISION_RESIZE="${VISION_RESIZE:-512}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-6144}"
FINETUNE_VISION_LAYERS="${FINETUNE_VISION_LAYERS:-false}"
VIDEO_READER="${VIDEO_READER:-decord}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/module_a_qwen35_2b_lora}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

mkdir -p "${OUTPUT_ROOT}" "${HF_CACHE}" "${WANDB_DIR}"

if [[ ! -f "${EGO_OOPS_ROOT}/EgoOops-annotations/meta/metadata_edited.json" ]]; then
  echo "Missing EgoOops annotations under EGO_OOPS_ROOT=${EGO_OOPS_ROOT}" >&2
  echo "Expected: ${EGO_OOPS_ROOT}/EgoOops-annotations/meta/metadata_edited.json" >&2
  exit 1
fi

if ! compgen -G "${DATA_ROOT}"'/*/*.MP4' > /dev/null; then
  echo "No EgoOops videos found under DATA_ROOT=${DATA_ROOT}" >&2
  echo "Expected paths like: ${DATA_ROOT}/blacklight/S1800001.MP4" >&2
  echo "Set DATA_ROOT=/path/to/videos-processed-720p when running this script." >&2
  exit 1
fi

if (( MAX_FRAMES >= 32 && MAX_SEQ_LENGTH < 4096 )); then
  echo "MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH} is too small for MAX_FRAMES=${MAX_FRAMES}." >&2
  echo "Use MAX_SEQ_LENGTH=6144, or reduce MAX_FRAMES/VISION_RESIZE." >&2
  exit 1
fi

TRAIN_ARGS=(
  python scripts/train_module_a_unsloth.py
  --max-videos "${MAX_VIDEOS}"
  --max-steps "${MAX_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --fps "${FPS}"
  --min-frames "${MIN_FRAMES}"
  --max-frames "${MAX_FRAMES}"
  --vision-resize "${VISION_RESIZE}"
  --max-seq-length "${MAX_SEQ_LENGTH}"
  --video-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --report-to "${REPORT_TO}"
)

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

case "${FINETUNE_VISION_LAYERS}" in
  1|true|TRUE|yes|YES)
    TRAIN_ARGS+=(--finetune-vision-layers)
    ;;
  0|false|FALSE|no|NO)
    TRAIN_ARGS+=(--no-finetune-vision-layers)
    ;;
  *)
    echo "FINETUNE_VISION_LAYERS must be true or false, got: ${FINETUNE_VISION_LAYERS}" >&2
    exit 1
    ;;
esac

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${PROJECT_DIR}:/workspace/qwen_ego_oops_lora_dataloaders" \
  -v "${EGO_OOPS_ROOT}:/workspace/ego_oops:ro" \
  -v "${DATA_ROOT}:${DATA_ROOT}:ro" \
  -v "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
  -v "${HF_CACHE}:/cache/huggingface" \
  -v "${WANDB_DIR}:${WANDB_DIR}" \
  -e HF_HOME=/cache/huggingface \
  -e TRANSFORMERS_CACHE=/cache/huggingface \
  -e FORCE_UNSLOTH_VIDEO_READER="${VIDEO_READER}" \
  -e WANDB_DIR="${WANDB_DIR}" \
  -e WANDB_PROJECT="${WANDB_PROJECT:-qwen-omd}" \
  -e WANDB_MODE="${WANDB_MODE:-online}" \
  -w /workspace/qwen_ego_oops_lora_dataloaders \
  "${IMAGE_NAME}" \
  "${TRAIN_ARGS[@]}"
