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
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.5-2B}"
MAX_VIDEOS="${MAX_VIDEOS:-50}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
VAL_VIDEOS_PER_TASK="${VAL_VIDEOS_PER_TASK:-2}"
SPLIT_FILE="${SPLIT_FILE:-}"
REGENERATE_SPLIT="${REGENERATE_SPLIT:-false}"
MODULE_C_MIN_DURATION_SECONDS="${MODULE_C_MIN_DURATION_SECONDS:-0.5}"
MODULE_C_JITTER_RATIO_MIN="${MODULE_C_JITTER_RATIO_MIN:-0.05}"
MODULE_C_JITTER_RATIO_MAX="${MODULE_C_JITTER_RATIO_MAX:-0.10}"
FPS="${FPS:-1.0}"
MIN_FRAMES="${MIN_FRAMES:-2}"
MEASURE_MAX_FRAMES="${MEASURE_MAX_FRAMES:-16 24 32}"
MEASURE_EVAL_MAX_FRAMES="${MEASURE_EVAL_MAX_FRAMES:-16}"
VISION_RESIZE="${VISION_RESIZE:-384}"
MEASURE_SEQ_LENGTHS="${MEASURE_SEQ_LENGTHS:-3072 4096 5120 6144 8192}"
MEASURE_MAX_SAMPLES_PER_SPLIT="${MEASURE_MAX_SAMPLES_PER_SPLIT:--1}"
MEASURE_OUTPUT_JSON="${MEASURE_OUTPUT_JSON:-${OUTPUT_ROOT}/module_c_seq_lengths_16gb.json}"
MEASURE_OUTPUT_CSV="${MEASURE_OUTPUT_CSV:-${OUTPUT_ROOT}/module_c_seq_lengths_16gb.csv}"
MEASURE_HISTOGRAM_DIR="${MEASURE_HISTOGRAM_DIR:-${OUTPUT_ROOT}/module_c_seq_length_histograms_16gb}"
MEASURE_INSTALL_MATPLOTLIB="${MEASURE_INSTALL_MATPLOTLIB:-true}"
VIDEO_READER="${VIDEO_READER:-decord}"

mkdir -p "${OUTPUT_ROOT}" "${HF_CACHE}" "$(dirname "${MEASURE_OUTPUT_JSON}")" "$(dirname "${MEASURE_OUTPUT_CSV}")" "${MEASURE_HISTOGRAM_DIR}"

if [[ ! -f "${EGO_OOPS_ROOT}/EgoOops-annotations/meta/metadata_edited.json" ]]; then
  echo "Missing EgoOops annotations under EGO_OOPS_ROOT=${EGO_OOPS_ROOT}" >&2
  exit 1
fi

if ! compgen -G "${DATA_ROOT}"'/*/*.MP4' > /dev/null; then
  echo "No EgoOops videos found under DATA_ROOT=${DATA_ROOT}" >&2
  echo "Set DATA_ROOT=/path/to/videos-processed-720p when running this script." >&2
  exit 1
fi

read -r -a MAX_FRAMES_VALUES <<< "${MEASURE_MAX_FRAMES}"
read -r -a EVAL_MAX_FRAMES_VALUES <<< "${MEASURE_EVAL_MAX_FRAMES}"
read -r -a SEQ_LENGTH_VALUES <<< "${MEASURE_SEQ_LENGTHS}"

DOCKER_TTY_ARGS=()
if [[ -t 0 ]]; then
  DOCKER_TTY_ARGS=(-it)
fi

MEASURE_ARGS=(
  python scripts/measure_module_c_seq_lengths.py
  --model-name "${MODEL_NAME}"
  --max-videos "${MAX_VIDEOS}"
  --val-fraction "${VAL_FRACTION}"
  --val-videos-per-task "${VAL_VIDEOS_PER_TASK}"
  --module-c-min-duration-seconds "${MODULE_C_MIN_DURATION_SECONDS}"
  --module-c-jitter-ratio-min "${MODULE_C_JITTER_RATIO_MIN}"
  --module-c-jitter-ratio-max "${MODULE_C_JITTER_RATIO_MAX}"
  --fps "${FPS}"
  --min-frames "${MIN_FRAMES}"
  --max-frames "${MAX_FRAMES_VALUES[@]}"
  --eval-max-frames "${EVAL_MAX_FRAMES_VALUES[@]}"
  --vision-resize "${VISION_RESIZE}"
  --seq-lengths "${SEQ_LENGTH_VALUES[@]}"
  --max-samples-per-split "${MEASURE_MAX_SAMPLES_PER_SPLIT}"
  --video-root "${DATA_ROOT}"
  --output-json "${MEASURE_OUTPUT_JSON}"
  --output-csv "${MEASURE_OUTPUT_CSV}"
  --histogram-dir "${MEASURE_HISTOGRAM_DIR}"
)

if [[ -n "${SPLIT_FILE}" ]]; then
  MEASURE_ARGS+=(--split-file "${SPLIT_FILE}")
fi
case "${REGENERATE_SPLIT}" in
  1|true|TRUE|yes|YES) MEASURE_ARGS+=(--regenerate-split) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "REGENERATE_SPLIT must be true or false, got: ${REGENERATE_SPLIT}" >&2; exit 1 ;;
esac

CONTAINER_ARGS=("${MEASURE_ARGS[@]}")
case "${MEASURE_INSTALL_MATPLOTLIB}" in
  1|true|TRUE|yes|YES)
    CONTAINER_ARGS=(
      bash
      -lc
      'python -m pip install --quiet matplotlib && exec "$@"'
      bash
      "${MEASURE_ARGS[@]}"
    )
    ;;
  0|false|FALSE|no|NO)
    ;;
  *)
    echo "MEASURE_INSTALL_MATPLOTLIB must be true or false, got: ${MEASURE_INSTALL_MATPLOTLIB}" >&2
    exit 1
    ;;
esac

docker run --rm "${DOCKER_TTY_ARGS[@]}" \
  --ipc=host \
  -v "${PROJECT_DIR}:/workspace/qwen_ego_oops_lora_dataloaders" \
  -v "${EGO_OOPS_ROOT}:/workspace/ego_oops:ro" \
  -v "${DATA_ROOT}:${DATA_ROOT}:ro" \
  -v "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
  -v "${HF_CACHE}:/cache/huggingface" \
  -e HF_HOME=/cache/huggingface \
  -e TRANSFORMERS_CACHE=/cache/huggingface \
  -e FORCE_UNSLOTH_VIDEO_READER="${VIDEO_READER}" \
  -w /workspace/qwen_ego_oops_lora_dataloaders \
  "${IMAGE_NAME}" \
  "${CONTAINER_ARGS[@]}"
