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
# Intentionally do not load .env.module_c here: that file often has training-time
# frame/resize/seq defaults that do not match a specific eval checkpoint.

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
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.5-2B}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"
LOAD_IN_16BIT="${LOAD_IN_16BIT:-true}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
VAL_VIDEOS_PER_TASK="${VAL_VIDEOS_PER_TASK:-2}"
SPLIT_FILE="${SPLIT_FILE:-${PROJECT_DIR}/splits/module_c_video_split_seed3407.json}"
MODULE_C_MIN_DURATION_SECONDS="${MODULE_C_MIN_DURATION_SECONDS:-0.5}"
MODULE_C_JITTER_RATIO_MIN="${MODULE_C_JITTER_RATIO_MIN:-0.05}"
MODULE_C_JITTER_RATIO_MAX="${MODULE_C_JITTER_RATIO_MAX:-0.10}"
EVAL_GENERATION_MAX_SAMPLES="${EVAL_GENERATION_MAX_SAMPLES:--1}"
EVAL_GENERATION_MAX_NEW_TOKENS="${EVAL_GENERATION_MAX_NEW_TOKENS:-128}"
FPS="${FPS:-1.0}"
MIN_FRAMES="${MIN_FRAMES:-2}"
MAX_FRAMES="${MAX_FRAMES:-16}"
EVAL_MAX_FRAMES="${EVAL_MAX_FRAMES:-16}"
VISION_RESIZE="${VISION_RESIZE:-384}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-3072}"
VIDEO_READER="${VIDEO_READER:-decord}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-auto}"
FINETUNE_VISION_LAYERS="${FINETUNE_VISION_LAYERS:-false}"
FINETUNE_LANGUAGE_LAYERS="${FINETUNE_LANGUAGE_LAYERS:-true}"
FINETUNE_ATTENTION_MODULES="${FINETUNE_ATTENTION_MODULES:-true}"
FINETUNE_MLP_MODULES="${FINETUNE_MLP_MODULES:-true}"
EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-${OUTPUT_ROOT}/module_c_qwen35_lora_reasoning/runs/module_c_16gb_2b_r16_16f384_seq3072/best_ep1}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/module_c_eval/best_ep1_16f384}"
GENERATION_EVAL_OUTPUT_JSON="${GENERATION_EVAL_OUTPUT_JSON:-${OUTPUT_DIR}/module_c_generation_eval.json}"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_DIR}" "${HF_CACHE}"

if [[ ! -f "${EGO_OOPS_ROOT}/EgoOops-annotations/meta/metadata_edited.json" ]]; then
  echo "Missing EgoOops annotations under EGO_OOPS_ROOT=${EGO_OOPS_ROOT}" >&2
  exit 1
fi
if ! compgen -G "${DATA_ROOT}"'/*/*.MP4' > /dev/null; then
  echo "No EgoOops videos found under DATA_ROOT=${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${EVAL_CHECKPOINT}" ]]; then
  echo "Missing EVAL_CHECKPOINT=${EVAL_CHECKPOINT}" >&2
  exit 1
fi

DOCKER_TTY_ARGS=()
if [[ -t 0 ]]; then
  DOCKER_TTY_ARGS=(-it)
fi

EVAL_ARGS=(
  python3 scripts/train_module_c_unsloth.py
  --generation-eval-only
  --eval-checkpoint "${EVAL_CHECKPOINT}"
  --generation-eval-output-json "${GENERATION_EVAL_OUTPUT_JSON}"
  --model-name "${MODEL_NAME}"
  --max-videos "${MAX_VIDEOS}"
  --val-fraction "${VAL_FRACTION}"
  --val-videos-per-task "${VAL_VIDEOS_PER_TASK}"
  --module-c-min-duration-seconds "${MODULE_C_MIN_DURATION_SECONDS}"
  --module-c-jitter-ratio-min "${MODULE_C_JITTER_RATIO_MIN}"
  --module-c-jitter-ratio-max "${MODULE_C_JITTER_RATIO_MAX}"
  --eval-generation-max-samples "${EVAL_GENERATION_MAX_SAMPLES}"
  --eval-generation-max-new-tokens "${EVAL_GENERATION_MAX_NEW_TOKENS}"
  --fps "${FPS}"
  --min-frames "${MIN_FRAMES}"
  --max-frames "${MAX_FRAMES}"
  --eval-max-frames "${EVAL_MAX_FRAMES}"
  --vision-resize "${VISION_RESIZE}"
  --max-seq-length "${MAX_SEQ_LENGTH}"
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --lora-target-modules "${LORA_TARGET_MODULES}"
  --video-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --report-to none
)

if [[ -n "${SPLIT_FILE}" ]]; then
  EVAL_ARGS+=(--split-file "${SPLIT_FILE}")
fi

case "${LOAD_IN_4BIT}" in
  1|true|TRUE|yes|YES) EVAL_ARGS+=(--load-in-4bit) ;;
  0|false|FALSE|no|NO) EVAL_ARGS+=(--no-load-in-4bit) ;;
  *) echo "LOAD_IN_4BIT must be true or false, got: ${LOAD_IN_4BIT}" >&2; exit 1 ;;
esac
case "${LOAD_IN_16BIT}" in
  1|true|TRUE|yes|YES) EVAL_ARGS+=(--load-in-16bit) ;;
  0|false|FALSE|no|NO) EVAL_ARGS+=(--no-load-in-16bit) ;;
  *) echo "LOAD_IN_16BIT must be true or false, got: ${LOAD_IN_16BIT}" >&2; exit 1 ;;
esac
case "${FINETUNE_VISION_LAYERS}" in
  1|true|TRUE|yes|YES) EVAL_ARGS+=(--finetune-vision-layers) ;;
  0|false|FALSE|no|NO) EVAL_ARGS+=(--no-finetune-vision-layers) ;;
  *) echo "FINETUNE_VISION_LAYERS must be true or false, got: ${FINETUNE_VISION_LAYERS}" >&2; exit 1 ;;
esac
case "${FINETUNE_LANGUAGE_LAYERS}" in
  1|true|TRUE|yes|YES) EVAL_ARGS+=(--finetune-language-layers) ;;
  0|false|FALSE|no|NO) EVAL_ARGS+=(--no-finetune-language-layers) ;;
  *) echo "FINETUNE_LANGUAGE_LAYERS must be true or false, got: ${FINETUNE_LANGUAGE_LAYERS}" >&2; exit 1 ;;
esac
case "${FINETUNE_ATTENTION_MODULES}" in
  1|true|TRUE|yes|YES) EVAL_ARGS+=(--finetune-attention-modules) ;;
  0|false|FALSE|no|NO) EVAL_ARGS+=(--no-finetune-attention-modules) ;;
  *) echo "FINETUNE_ATTENTION_MODULES must be true or false, got: ${FINETUNE_ATTENTION_MODULES}" >&2; exit 1 ;;
esac
case "${FINETUNE_MLP_MODULES}" in
  1|true|TRUE|yes|YES) EVAL_ARGS+=(--finetune-mlp-modules) ;;
  0|false|FALSE|no|NO) EVAL_ARGS+=(--no-finetune-mlp-modules) ;;
  *) echo "FINETUNE_MLP_MODULES must be true or false, got: ${FINETUNE_MLP_MODULES}" >&2; exit 1 ;;
esac

echo "Running Module C validation-only eval"
echo "  checkpoint: ${EVAL_CHECKPOINT}"
echo "  frames/resize: ${EVAL_MAX_FRAMES} / ${VISION_RESIZE}"
echo "  output: ${GENERATION_EVAL_OUTPUT_JSON}"

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
  -e FORCE_UNSLOTH_VIDEO_READER="${VIDEO_READER}" \
  -e PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
  -e UNSLOTH_RETURN_LOGITS=0 \
  -e UNSLOTH_RETURN_HIDDEN_STATES=0 \
  -w /workspace/qwen_ego_oops_lora_dataloaders \
  "${IMAGE_NAME}" \
  "${EVAL_ARGS[@]}"
