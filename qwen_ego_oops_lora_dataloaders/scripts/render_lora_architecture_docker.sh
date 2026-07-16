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

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/amit/online_mistake_detection/outputs}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/wandb}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.5-2B}"
ARCH_MODULE="${ARCH_MODULE:-}"
ARCH_CHECKPOINT="${ARCH_CHECKPOINT:-}"
ARCH_OUTPUT_DIR="${ARCH_OUTPUT_DIR:-}"
ARCH_TITLE="${ARCH_TITLE:-}"
ARCH_LOG_TO_WANDB="${ARCH_LOG_TO_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen-omd}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME="${WANDB_NAME:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ARTIFACT_NAME="${WANDB_ARTIFACT_NAME:-}"

if [[ -z "${ARCH_MODULE}" ]]; then
  echo "Set ARCH_MODULE=A, B, or C." >&2
  exit 1
fi
if [[ -z "${ARCH_CHECKPOINT}" ]]; then
  echo "Set ARCH_CHECKPOINT=/path/to/best_epX or adapter checkpoint directory." >&2
  exit 1
fi
if [[ ! -d "${ARCH_CHECKPOINT}" ]]; then
  echo "ARCH_CHECKPOINT does not exist or is not a directory: ${ARCH_CHECKPOINT}" >&2
  exit 1
fi
ARCH_CHECKPOINT_PARENT="$(cd "$(dirname "${ARCH_CHECKPOINT}")" && pwd)"
ARCH_CHECKPOINT="${ARCH_CHECKPOINT_PARENT}/$(basename "${ARCH_CHECKPOINT}")"
if [[ -z "${ARCH_OUTPUT_DIR}" ]]; then
  checkpoint_name="$(basename "${ARCH_CHECKPOINT}")"
  run_name="$(basename "$(dirname "${ARCH_CHECKPOINT}")")"
  ARCH_OUTPUT_DIR="${OUTPUT_ROOT}/lora_architecture/module_${ARCH_MODULE,,}_${run_name}_${checkpoint_name}"
fi

mkdir -p "${OUTPUT_ROOT}" "${ARCH_OUTPUT_DIR}" "${HF_CACHE}" "${WANDB_DIR}"

DOCKER_TTY_ARGS=()
if [[ -t 0 ]]; then
  DOCKER_TTY_ARGS=(-it)
fi

RENDER_ARGS=(
  python scripts/render_lora_architecture.py
  --module "${ARCH_MODULE}"
  --checkpoint "${ARCH_CHECKPOINT}"
  --model-name "${MODEL_NAME}"
  --output-dir "${ARCH_OUTPUT_DIR}"
  --wandb-project "${WANDB_PROJECT}"
)

if [[ -n "${ARCH_TITLE}" ]]; then
  RENDER_ARGS+=(--title "${ARCH_TITLE}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  RENDER_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_NAME}" ]]; then
  RENDER_ARGS+=(--wandb-run-name "${WANDB_NAME}")
fi
if [[ -n "${WANDB_MODE}" ]]; then
  RENDER_ARGS+=(--wandb-mode "${WANDB_MODE}")
fi
if [[ -n "${WANDB_ARTIFACT_NAME}" ]]; then
  RENDER_ARGS+=(--wandb-artifact-name "${WANDB_ARTIFACT_NAME}")
fi
case "${ARCH_LOG_TO_WANDB}" in
  1|true|TRUE|yes|YES) RENDER_ARGS+=(--log-to-wandb) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "ARCH_LOG_TO_WANDB must be true or false, got: ${ARCH_LOG_TO_WANDB}" >&2; exit 1 ;;
esac

docker run --rm "${DOCKER_TTY_ARGS[@]}" \
  --ipc=host \
  -v "${PROJECT_DIR}:/workspace/qwen_ego_oops_lora_dataloaders" \
  -v "${ARCH_CHECKPOINT_PARENT}:${ARCH_CHECKPOINT_PARENT}:ro" \
  -v "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
  -v "${HF_CACHE}:/cache/huggingface" \
  -v "${WANDB_DIR}:${WANDB_DIR}" \
  -e HF_HOME=/cache/huggingface \
  -e TRANSFORMERS_CACHE=/cache/huggingface \
  -e WANDB_DIR="${WANDB_DIR}" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e WANDB_PROJECT="${WANDB_PROJECT}" \
  -e WANDB_ENTITY="${WANDB_ENTITY}" \
  -e WANDB_MODE="${WANDB_MODE}" \
  -w /workspace/qwen_ego_oops_lora_dataloaders \
  "${IMAGE_NAME}" \
  "${RENDER_ARGS[@]}"
