#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

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

if [[ -f "${REPO_DIR}/.env" ]]; then
  load_env_defaults "${REPO_DIR}/.env"
fi

if [[ -n "${VAST_API_KEY:-}" ]]; then
  mkdir -p "${HOME}/.config/vastai"
  printf "%s\n" "${VAST_API_KEY}" > "${HOME}/.config/vastai/vast_api_key"
  chmod 600 "${HOME}/.config/vastai/vast_api_key"
fi

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-module_a_vastai_${RUN_TIMESTAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/module_a_qwen35_2b_lora_wait_complete_vision/runs/${RUN_NAME}}"
EGO_OOPS_ROOT="${EGO_OOPS_ROOT:-${REPO_DIR}/ego_oops}"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/videos-processed-720p}"
COPY_VIDEOS_TO_SHM="${COPY_VIDEOS_TO_SHM:-false}"
SHM_VIDEO_ROOT="${SHM_VIDEO_ROOT:-/dev/shm/ego_oops_videos}"
MAX_VIDEOS="${MAX_VIDEOS:-50}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.5-2B}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"
LOAD_IN_16BIT="${LOAD_IN_16BIT:-true}"
TRAIN_MODE="${TRAIN_MODE:-steps}"
MAX_STEPS="${MAX_STEPS:-100}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3.0}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-2}"
EVAL_EPOCHS="${EVAL_EPOCHS:-2}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-4}"
KEEP_BEST_CHECKPOINTS="${KEEP_BEST_CHECKPOINTS:-4}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
VAL_VIDEOS_PER_TASK="${VAL_VIDEOS_PER_TASK:-2}"
SPLIT_FILE="${SPLIT_FILE:-}"
REGENERATE_SPLIT="${REGENERATE_SPLIT:-false}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
EARLY_STOPPING_THRESHOLD="${EARLY_STOPPING_THRESHOLD:-0.0}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-eval_loss}"
GREATER_IS_BETTER="${GREATER_IS_BETTER:-}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen-omd}"
WANDB_ARTIFACT_PREFIX="${WANDB_ARTIFACT_PREFIX:-module-a}"
EVAL_GENERATION_MAX_SAMPLES="${EVAL_GENERATION_MAX_SAMPLES:--1}"
EVAL_GENERATION_MAX_NEW_TOKENS="${EVAL_GENERATION_MAX_NEW_TOKENS:-8}"
GENERATION_EVAL_MODE="${GENERATION_EVAL_MODE:-subprocess}"
FPS="${FPS:-1.0}"
MIN_FRAMES="${MIN_FRAMES:-2}"
MAX_FRAMES="${MAX_FRAMES:-32}"
VISION_RESIZE="${VISION_RESIZE:-512}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-6144}"
FINETUNE_VISION_LAYERS="${FINETUNE_VISION_LAYERS:-false}"
UPLOAD_FINAL_CHECKPOINTS_TO_WANDB="${UPLOAD_FINAL_CHECKPOINTS_TO_WANDB:-true}"
UPLOAD_FINAL_CHECKPOINTS_ON_FAILURE="${UPLOAD_FINAL_CHECKPOINTS_ON_FAILURE:-false}"
DESTROY_VAST_INSTANCE_ON_EXIT="${DESTROY_VAST_INSTANCE_ON_EXIT:-false}"
DESTROY_VAST_INSTANCE_ON_FAILURE="${DESTROY_VAST_INSTANCE_ON_FAILURE:-false}"
VAST_INSTANCE_ID="${VAST_INSTANCE_ID:-}"

export RUN_TIMESTAMP RUN_NAME OUTPUT_ROOT OUTPUT_DIR
export REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT WANDB_ARTIFACT_PREFIX
export WANDB_LOG_BEST_CHECKPOINTS="${WANDB_LOG_BEST_CHECKPOINTS:-true}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/wandb}"
export HF_HOME="${HF_HOME:-/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export FORCE_UNSLOTH_VIDEO_READER="${VIDEO_READER:-decord}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_DIR}" "${WANDB_DIR}" "${HF_HOME}"

if [[ ! -f "${EGO_OOPS_ROOT}/EgoOops-annotations/meta/metadata_edited.json" ]]; then
  echo "Missing EgoOops annotations under EGO_OOPS_ROOT=${EGO_OOPS_ROOT}" >&2
  echo "Expected: ${EGO_OOPS_ROOT}/EgoOops-annotations/meta/metadata_edited.json" >&2
  exit 1
fi

case "${COPY_VIDEOS_TO_SHM}" in
  1|true|TRUE|yes|YES)
    if ! compgen -G "${DATA_ROOT}"'/*/*.MP4' > /dev/null; then
      echo "Cannot copy videos to /dev/shm; no videos found under DATA_ROOT=${DATA_ROOT}" >&2
      exit 1
    fi
    if [[ "${DATA_ROOT}" != "${SHM_VIDEO_ROOT}" ]]; then
      mkdir -p "${SHM_VIDEO_ROOT}"
      if ! compgen -G "${SHM_VIDEO_ROOT}"'/*/*.MP4' > /dev/null; then
        echo "Copying EgoOops videos to shared memory: ${SHM_VIDEO_ROOT}"
        cp -a "${DATA_ROOT}/." "${SHM_VIDEO_ROOT}/"
      else
        echo "Using existing shared-memory video copy: ${SHM_VIDEO_ROOT}"
      fi
      DATA_ROOT="${SHM_VIDEO_ROOT}"
    fi
    ;;
  0|false|FALSE|no|NO)
    ;;
  *)
    echo "COPY_VIDEOS_TO_SHM must be true or false, got: ${COPY_VIDEOS_TO_SHM}" >&2
    exit 1
    ;;
esac

if ! compgen -G "${DATA_ROOT}"'/*/*.MP4' > /dev/null; then
  echo "No EgoOops videos found under DATA_ROOT=${DATA_ROOT}" >&2
  echo "Expected paths like: ${DATA_ROOT}/blacklight/S1800001.MP4" >&2
  exit 1
fi

upload_final_checkpoints() {
  if [[ ! "${UPLOAD_FINAL_CHECKPOINTS_TO_WANDB}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
    return 0
  fi
  if [[ ! -d "${OUTPUT_DIR}" ]]; then
    echo "No output directory to upload: ${OUTPUT_DIR}" >&2
    return 0
  fi
  if ! python3 -c "import wandb" >/dev/null 2>&1; then
    echo "wandb is not installed on the host; skipping final checkpoint upload." >&2
    return 0
  fi

  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_ENTITY="${WANDB_ENTITY:-}" \
  WANDB_RUN_NAME="${RUN_NAME}-final-upload" \
  WANDB_ARTIFACT_NAME="${WANDB_ARTIFACT_PREFIX}-${RUN_NAME}-all-checkpoints" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  python3 - <<'PY'
import os
from pathlib import Path

import wandb

def sanitize(name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    cleaned = "".join(char if char in allowed else "-" for char in name)
    return cleaned.strip(".-") or "module-a-all-checkpoints"

output_dir = Path(os.environ["OUTPUT_DIR"])
entity = os.environ.get("WANDB_ENTITY") or None
run = wandb.init(
    project=os.environ["WANDB_PROJECT"],
    entity=entity,
    job_type="final-checkpoint-upload",
    name=os.environ["WANDB_RUN_NAME"],
)
artifact = wandb.Artifact(
    name=sanitize(os.environ["WANDB_ARTIFACT_NAME"]),
    type="model",
    metadata={"output_dir": str(output_dir)},
)
artifact.add_dir(str(output_dir))
run.log_artifact(artifact, aliases=["final", "all-checkpoints"])
run.finish()
PY
}

destroy_vast_instance() {
  if [[ ! "${DESTROY_VAST_INSTANCE_ON_EXIT}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
    return 0
  fi
  if [[ -z "${VAST_INSTANCE_ID}" ]]; then
    echo "DESTROY_VAST_INSTANCE_ON_EXIT=true but VAST_INSTANCE_ID is empty; not destroying." >&2
    return 0
  fi
  if ! command -v vastai >/dev/null 2>&1; then
    echo "vastai CLI is not installed; not destroying instance ${VAST_INSTANCE_ID}." >&2
    return 0
  fi
  vastai destroy instance "${VAST_INSTANCE_ID}"
}

cleanup() {
  exit_code=$?
  set +e
  if [[ "${exit_code}" -eq 0 || "${UPLOAD_FINAL_CHECKPOINTS_ON_FAILURE}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
    upload_final_checkpoints
  else
    echo "Training exited with code ${exit_code}; skipping final W&B checkpoint upload."
  fi
  if [[ "${exit_code}" -eq 0 || "${DESTROY_VAST_INSTANCE_ON_FAILURE}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
    destroy_vast_instance
  else
    echo "Training exited with code ${exit_code}; not destroying Vast.ai instance."
  fi
  exit "${exit_code}"
}
trap cleanup EXIT

cd "${PROJECT_DIR}"

TRAIN_ARGS=(
  python scripts/train_module_a_unsloth.py
  --model-name "${MODEL_NAME}"
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
  --eval-generation-max-samples "${EVAL_GENERATION_MAX_SAMPLES}"
  --eval-generation-max-new-tokens "${EVAL_GENERATION_MAX_NEW_TOKENS}"
  --generation-eval-mode "${GENERATION_EVAL_MODE}"
  --fps "${FPS}"
  --min-frames "${MIN_FRAMES}"
  --max-frames "${MAX_FRAMES}"
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

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
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

"${TRAIN_ARGS[@]}"
