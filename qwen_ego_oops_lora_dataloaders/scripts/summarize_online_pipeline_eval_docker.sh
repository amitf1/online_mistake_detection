#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-qwen-omd-dataloaders:latest}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/amit/online_mistake_detection/outputs}"
PIPELINE_METRICS_JSON="${PIPELINE_METRICS_JSON:-${OUTPUT_ROOT}/online_pipeline_eval/full_16gb_docker/pipeline_metrics.json}"
SUMMARY_OUTPUT_DIR="${SUMMARY_OUTPUT_DIR:-${OUTPUT_ROOT}/online_pipeline_eval/full_16gb_docker/summary}"
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

mkdir -p "${SUMMARY_OUTPUT_DIR}"

DOCKER_TTY_ARGS=()
if [[ -t 0 ]]; then
  DOCKER_TTY_ARGS=(-it)
fi

docker run --rm "${DOCKER_TTY_ARGS[@]}" \
  -v "${PROJECT_DIR}:/workspace/qwen_ego_oops_lora_dataloaders" \
  -v "${EGO_OOPS_ROOT}:/workspace/ego_oops:ro" \
  -v "${DATA_ROOT}:${DATA_ROOT}:ro" \
  -v "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
  -w /workspace/qwen_ego_oops_lora_dataloaders \
  "${IMAGE_NAME}" \
  bash -lc 'python3 -c "import matplotlib" 2>/dev/null || pip install matplotlib; python3 scripts/summarize_online_pipeline_eval.py --pipeline-metrics-json "'"${PIPELINE_METRICS_JSON}"'" --output-dir "'"${SUMMARY_OUTPUT_DIR}"'"'
