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
if [[ -f "${PROJECT_DIR}/../.env.module_c" ]]; then
  load_env_defaults "${PROJECT_DIR}/../.env.module_c"
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
WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/wandb}"
REPORT_TO="${REPORT_TO:-tensorboard}"
MAX_VIDEOS="${MAX_VIDEOS:-50}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.5-2B}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"
LOAD_IN_16BIT="${LOAD_IN_16BIT:-true}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-auto}"
TRAIN_MODE="${TRAIN_MODE:-steps}"
MAX_STEPS="${MAX_STEPS:-100}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3.0}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-1}"
EVAL_EPOCHS="${EVAL_EPOCHS:-1}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-4}"
KEEP_BEST_CHECKPOINTS="${KEEP_BEST_CHECKPOINTS:-4}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
VAL_VIDEOS_PER_TASK="${VAL_VIDEOS_PER_TASK:-2}"
SPLIT_FILE="${SPLIT_FILE:-}"
REGENERATE_SPLIT="${REGENERATE_SPLIT:-false}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
EARLY_STOPPING_THRESHOLD="${EARLY_STOPPING_THRESHOLD:-0.0}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-eval_mistake/f1}"
GREATER_IS_BETTER="${GREATER_IS_BETTER:-true}"
WANDB_LOG_BEST_CHECKPOINTS="${WANDB_LOG_BEST_CHECKPOINTS:-false}"
WANDB_ARTIFACT_PREFIX="${WANDB_ARTIFACT_PREFIX:-module-c}"
MODULE_C_MIN_DURATION_SECONDS="${MODULE_C_MIN_DURATION_SECONDS:-0.5}"
MODULE_C_JITTER_RATIO_MIN="${MODULE_C_JITTER_RATIO_MIN:-0.05}"
MODULE_C_JITTER_RATIO_MAX="${MODULE_C_JITTER_RATIO_MAX:-0.10}"
EVAL_GENERATION_MAX_SAMPLES="${EVAL_GENERATION_MAX_SAMPLES:--1}"
EVAL_GENERATION_MAX_NEW_TOKENS="${EVAL_GENERATION_MAX_NEW_TOKENS:-128}"
FPS="${FPS:-1.0}"
MIN_FRAMES="${MIN_FRAMES:-2}"
MAX_FRAMES="${MAX_FRAMES:-32}"
EVAL_MAX_FRAMES="${EVAL_MAX_FRAMES:-${MAX_FRAMES}}"
VISION_RESIZE="${VISION_RESIZE:-512}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
FINETUNE_VISION_LAYERS="${FINETUNE_VISION_LAYERS:-false}"
FINETUNE_LANGUAGE_LAYERS="${FINETUNE_LANGUAGE_LAYERS:-true}"
FINETUNE_ATTENTION_MODULES="${FINETUNE_ATTENTION_MODULES:-true}"
FINETUNE_MLP_MODULES="${FINETUNE_MLP_MODULES:-true}"
VIDEO_READER="${VIDEO_READER:-decord}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-module_c_qwen35_lora_reasoning_${RUN_TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/module_c_qwen35_lora_reasoning/runs/${RUN_NAME}}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
DRY_RUN="${DRY_RUN:-false}"
PRINT_EXAMPLES="${PRINT_EXAMPLES:-2}"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_DIR}" "${HF_CACHE}" "${WANDB_DIR}"

if [[ ! -f "${EGO_OOPS_ROOT}/EgoOops-annotations/meta/metadata_edited.json" ]]; then
  echo "Missing EgoOops annotations under EGO_OOPS_ROOT=${EGO_OOPS_ROOT}" >&2
  exit 1
fi

if ! compgen -G "${DATA_ROOT}"'/*/*.MP4' > /dev/null; then
  echo "No EgoOops videos found under DATA_ROOT=${DATA_ROOT}" >&2
  echo "Set DATA_ROOT=/path/to/videos-processed-720p when running this script." >&2
  exit 1
fi

DOCKER_TTY_ARGS=()
if [[ -t 0 ]]; then
  DOCKER_TTY_ARGS=(-it)
fi

TRAIN_ARGS=(
  python scripts/train_module_c_unsloth.py
  --model-name "${MODEL_NAME}"
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --lora-target-modules "${LORA_TARGET_MODULES}"
  --max-videos "${MAX_VIDEOS}"
  --train-mode "${TRAIN_MODE}"
  --max-steps "${MAX_STEPS}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS}"
  --checkpoint-epochs "${CHECKPOINT_EPOCHS}"
  --eval-epochs "${EVAL_EPOCHS}"
  --keep-last-checkpoints "${KEEP_LAST_CHECKPOINTS}"
  --keep-best-checkpoints "${KEEP_BEST_CHECKPOINTS}"
  --val-fraction "${VAL_FRACTION}"
  --val-videos-per-task "${VAL_VIDEOS_PER_TASK}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --early-stopping-threshold "${EARLY_STOPPING_THRESHOLD}"
  --metric-for-best-model "${METRIC_FOR_BEST_MODEL}"
  --wandb-artifact-prefix "${WANDB_ARTIFACT_PREFIX}"
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
  --video-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --report-to "${REPORT_TO}"
)

case "${LOAD_IN_4BIT}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--load-in-4bit) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-load-in-4bit) ;;
  *) echo "LOAD_IN_4BIT must be true or false, got: ${LOAD_IN_4BIT}" >&2; exit 1 ;;
esac
case "${LOAD_IN_16BIT}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--load-in-16bit) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-load-in-16bit) ;;
  *) echo "LOAD_IN_16BIT must be true or false, got: ${LOAD_IN_16BIT}" >&2; exit 1 ;;
esac
case "${WANDB_LOG_BEST_CHECKPOINTS}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--wandb-log-best-checkpoints) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-wandb-log-best-checkpoints) ;;
  *) echo "WANDB_LOG_BEST_CHECKPOINTS must be true or false, got: ${WANDB_LOG_BEST_CHECKPOINTS}" >&2; exit 1 ;;
esac
case "${DRY_RUN}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--dry-run --print-examples "${PRINT_EXAMPLES}") ;;
  0|false|FALSE|no|NO) ;;
  *) echo "DRY_RUN must be true or false, got: ${DRY_RUN}" >&2; exit 1 ;;
esac
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ -n "${SPLIT_FILE}" ]]; then
  TRAIN_ARGS+=(--split-file "${SPLIT_FILE}")
fi
case "${GREATER_IS_BETTER}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--greater-is-better) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-greater-is-better) ;;
  "") ;;
  *) echo "GREATER_IS_BETTER must be true, false, or empty, got: ${GREATER_IS_BETTER}" >&2; exit 1 ;;
esac
case "${REGENERATE_SPLIT}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--regenerate-split) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "REGENERATE_SPLIT must be true or false, got: ${REGENERATE_SPLIT}" >&2; exit 1 ;;
esac
case "${FINETUNE_VISION_LAYERS}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--finetune-vision-layers) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-finetune-vision-layers) ;;
  *) echo "FINETUNE_VISION_LAYERS must be true or false, got: ${FINETUNE_VISION_LAYERS}" >&2; exit 1 ;;
esac
case "${FINETUNE_LANGUAGE_LAYERS}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--finetune-language-layers) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-finetune-language-layers) ;;
  *) echo "FINETUNE_LANGUAGE_LAYERS must be true or false, got: ${FINETUNE_LANGUAGE_LAYERS}" >&2; exit 1 ;;
esac
case "${FINETUNE_ATTENTION_MODULES}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--finetune-attention-modules) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-finetune-attention-modules) ;;
  *) echo "FINETUNE_ATTENTION_MODULES must be true or false, got: ${FINETUNE_ATTENTION_MODULES}" >&2; exit 1 ;;
esac
case "${FINETUNE_MLP_MODULES}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--finetune-mlp-modules) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-finetune-mlp-modules) ;;
  *) echo "FINETUNE_MLP_MODULES must be true or false, got: ${FINETUNE_MLP_MODULES}" >&2; exit 1 ;;
esac

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
  -v "${WANDB_DIR}:${WANDB_DIR}" \
  -e HF_HOME=/cache/huggingface \
  -e TRANSFORMERS_CACHE=/cache/huggingface \
  -e FORCE_UNSLOTH_VIDEO_READER="${VIDEO_READER}" \
  -e PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
  -e UNSLOTH_RETURN_LOGITS=0 \
  -e UNSLOTH_RETURN_HIDDEN_STATES=0 \
  -e WANDB_DIR="${WANDB_DIR}" \
  -e WANDB_PROJECT="${WANDB_PROJECT:-qwen-omd}" \
  -e WANDB_MODE="${WANDB_MODE:-online}" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e WANDB_ENTITY="${WANDB_ENTITY:-}" \
  -e WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-}" \
  -e WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}" \
  -w /workspace/qwen_ego_oops_lora_dataloaders \
  "${IMAGE_NAME}" \
  "${TRAIN_ARGS[@]}"
