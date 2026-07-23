#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-qwen-omd-dataloaders:latest}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

load_env_defaults() {
  local env_file="$1"
  local line key
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* || "${line}" != *=* ]] && continue
    key="${line%%=*}"
    if [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ && -z "${!key+x}" ]]; then
      export "${line}"
    fi
  done < "${env_file}"
}

if [[ -f "${PROJECT_DIR}/../.env" ]]; then
  load_env_defaults "${PROJECT_DIR}/../.env"
fi

EGO_OOPS_ROOT="${EGO_OOPS_ROOT:-$(cd "${PROJECT_DIR}/../ego_oops" && pwd)}"
if [[ -z "${DATA_ROOT:-}" ]]; then
  if [[ -d "${PROJECT_DIR}/../../data/videos-processed-720p" ]]; then
    DATA_ROOT="$(cd "${PROJECT_DIR}/../../data/videos-processed-720p" && pwd)"
  elif [[ -d "${PROJECT_DIR}/../data/videos-processed-720p" ]]; then
    DATA_ROOT="$(cd "${PROJECT_DIR}/../data/videos-processed-720p" && pwd)"
  else
    DATA_ROOT="${PROJECT_DIR}/../data/videos-processed-720p"
  fi
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/amit/online_mistake_detection/outputs}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
MAX_VIDEOS="${MAX_VIDEOS:-50}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
VAL_VIDEOS_PER_TASK="${VAL_VIDEOS_PER_TASK:-2}"
SPLIT_FILE="${SPLIT_FILE:-}"
REGENERATE_SPLIT="${REGENERATE_SPLIT:-false}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.5-2B}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"
LOAD_IN_16BIT="${LOAD_IN_16BIT:-true}"
FPS="${FPS:-1.0}"
MIN_FRAMES="${MIN_FRAMES:-2}"
MAX_FRAMES_A="${MAX_FRAMES_A:-16}"
MAX_FRAMES_B="${MAX_FRAMES_B:-16}"
MAX_FRAMES_C="${MAX_FRAMES_C:-16}"
VISION_RESIZE_A="${VISION_RESIZE_A:-336}"
VISION_RESIZE_B="${VISION_RESIZE_B:-384}"
VISION_RESIZE_C="${VISION_RESIZE_C:-384}"
MAX_SEQ_LENGTH_A="${MAX_SEQ_LENGTH_A:-3072}"
MAX_SEQ_LENGTH_B="${MAX_SEQ_LENGTH_B:-5120}"
MAX_SEQ_LENGTH_C="${MAX_SEQ_LENGTH_C:-3072}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/online_pipeline_eval/full_16gb_docker}"
REUSE_EVENTS_JSON="${REUSE_EVENTS_JSON:-}"
MODULE_A_LABEL_MODE="${MODULE_A_LABEL_MODE:-step_id}"
MODULE_A_CHECKPOINT="${MODULE_A_CHECKPOINT:-${OUTPUT_ROOT}/module_a_qwen35_2b_lora_wait_complete_vision/runs/module_a_recall_loss2_from_ep8_to_ep50_v8/best_ep10}"
MODULE_B_CHECKPOINT="${MODULE_B_CHECKPOINT:-${OUTPUT_ROOT}/wandb_artifacts/module-b-9kahxivi-best_v1/best_ep7}"
MODULE_C_CHECKPOINT="${MODULE_C_CHECKPOINT:-${OUTPUT_ROOT}/module_c_qwen35_lora_reasoning/runs/module_c_16gb_2b_r16_16f384_seq3072/best_ep1}"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_DIR}" "${HF_CACHE}"

if [[ ! -f "${EGO_OOPS_ROOT}/EgoOops-annotations/meta/metadata_edited.json" ]]; then
  echo "Missing EgoOops annotations under EGO_OOPS_ROOT=${EGO_OOPS_ROOT}" >&2
  exit 1
fi
if ! compgen -G "${DATA_ROOT}"'/*/*.MP4' > /dev/null; then
  echo "No EgoOops videos found under DATA_ROOT=${DATA_ROOT}" >&2
  exit 1
fi

DOCKER_TTY_ARGS=()
if [[ -t 0 ]]; then
  DOCKER_TTY_ARGS=(-it)
fi

PIPELINE_ARGS=(
  python3 scripts/run_online_pipeline_eval.py
  --metadata /workspace/ego_oops/EgoOops-annotations/meta/metadata_edited.json
  --mistake-classes /workspace/ego_oops/EgoOops-annotations/meta/mistake_classes.json
  --video-root "${DATA_ROOT}"
  --module-a-checkpoint "${MODULE_A_CHECKPOINT}"
  --module-b-checkpoint "${MODULE_B_CHECKPOINT}"
  --module-c-checkpoint "${MODULE_C_CHECKPOINT}"
  --module-a-label-mode "${MODULE_A_LABEL_MODE}"
  --model-name "${MODEL_NAME}"
  --max-videos "${MAX_VIDEOS}"
  --val-fraction "${VAL_FRACTION}"
  --val-videos-per-task "${VAL_VIDEOS_PER_TASK}"
  --fps "${FPS}"
  --min-frames "${MIN_FRAMES}"
  --max-frames-a "${MAX_FRAMES_A}"
  --max-frames-b "${MAX_FRAMES_B}"
  --max-frames-c "${MAX_FRAMES_C}"
  --vision-resize-a "${VISION_RESIZE_A}"
  --vision-resize-b "${VISION_RESIZE_B}"
  --vision-resize-c "${VISION_RESIZE_C}"
  --max-seq-length-a "${MAX_SEQ_LENGTH_A}"
  --max-seq-length-b "${MAX_SEQ_LENGTH_B}"
  --max-seq-length-c "${MAX_SEQ_LENGTH_C}"
  --max-samples "${MAX_SAMPLES}"
  --output-dir "${OUTPUT_DIR}"
)

case "${LOAD_IN_4BIT}" in
  1|true|TRUE|yes|YES) PIPELINE_ARGS+=(--load-in-4bit) ;;
  0|false|FALSE|no|NO) PIPELINE_ARGS+=(--no-load-in-4bit) ;;
  *) echo "LOAD_IN_4BIT must be true or false, got: ${LOAD_IN_4BIT}" >&2; exit 1 ;;
esac
case "${LOAD_IN_16BIT}" in
  1|true|TRUE|yes|YES) PIPELINE_ARGS+=(--load-in-16bit) ;;
  0|false|FALSE|no|NO) PIPELINE_ARGS+=(--no-load-in-16bit) ;;
  *) echo "LOAD_IN_16BIT must be true or false, got: ${LOAD_IN_16BIT}" >&2; exit 1 ;;
esac
case "${REGENERATE_SPLIT}" in
  1|true|TRUE|yes|YES) PIPELINE_ARGS+=(--regenerate-split) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "REGENERATE_SPLIT must be true or false, got: ${REGENERATE_SPLIT}" >&2; exit 1 ;;
esac
if [[ -n "${SPLIT_FILE}" ]]; then
  PIPELINE_ARGS+=(--split-file "${SPLIT_FILE}")
fi
if [[ -n "${REUSE_EVENTS_JSON}" ]]; then
  PIPELINE_ARGS+=(--reuse-events-json "${REUSE_EVENTS_JSON}")
fi

docker run --rm "${DOCKER_TTY_ARGS[@]}" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${PROJECT_DIR}:/workspace/qwen_ego_oops_lora_dataloaders" \
  -v "${EGO_OOPS_ROOT}:/workspace/ego_oops:ro" \
  -v "${DATA_ROOT}:${DATA_ROOT}:ro" \
  -v "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
  -v "${HF_CACHE}:/cache/huggingface" \
  -e HF_HOME=/cache/huggingface \
  -e TRANSFORMERS_CACHE=/cache/huggingface \
  -e FORCE_UNSLOTH_VIDEO_READER="${VIDEO_READER:-decord}" \
  -e PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
  -e UNSLOTH_RETURN_LOGITS=0 \
  -e UNSLOTH_RETURN_HIDDEN_STATES=0 \
  -w /workspace/qwen_ego_oops_lora_dataloaders \
  "${IMAGE_NAME}" \
  "${PIPELINE_ARGS[@]}"
