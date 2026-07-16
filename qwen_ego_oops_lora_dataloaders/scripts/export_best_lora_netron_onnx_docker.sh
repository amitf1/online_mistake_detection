#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-qwen-omd-dataloaders:latest}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/amit/online_mistake_detection/outputs}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
ONNX_AUTO_INSTALL="${ONNX_AUTO_INSTALL:-true}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.5-2B}"

MODULE_A_CHECKPOINT="${MODULE_A_CHECKPOINT:-${OUTPUT_ROOT}/module_a_qwen35_2b_lora_recall_loss_weighted_r8_frames32/runs/module_a_recall_loss2_r8_frames32_seq6144_prompt_v2/best_ep9}"
MODULE_B_CHECKPOINT="${MODULE_B_CHECKPOINT:-${OUTPUT_ROOT}/experiment_documentation/module_b_wandb_checkpoints/module_b__model=unsloth-Qwen3.5-2B__r=32__alpha=64__frames=24__evalframes=unknown__resize=512__seq=8192__negratio=0.1__incomplete=true__eval_temporal_mean_iou=0.735739__ep=7__artifact=module-b-9kahxivi-best-v1/best_ep7}"

MODULE_A_OUTPUT_DIR="${MODULE_A_OUTPUT_DIR:-${OUTPUT_ROOT}/netron_onnx/module_a_recall_r8_best_ep9}"
MODULE_B_OUTPUT_DIR="${MODULE_B_OUTPUT_DIR:-${OUTPUT_ROOT}/netron_onnx/module_b_wandb_9kahxivi_best_ep7}"

for checkpoint in "${MODULE_A_CHECKPOINT}" "${MODULE_B_CHECKPOINT}"; do
  if [[ ! -d "${checkpoint}" ]]; then
    echo "Checkpoint does not exist: ${checkpoint}" >&2
    exit 1
  fi
  if [[ ! -f "${checkpoint}/adapter_config.json" ]]; then
    echo "Checkpoint is missing adapter_config.json: ${checkpoint}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${MODULE_A_OUTPUT_DIR}" "${MODULE_B_OUTPUT_DIR}" "${HF_CACHE}"

run_export() {
  local module="$1"
  local checkpoint="$2"
  local output_dir="$3"
  local title="$4"

  docker run --rm \
    --ipc=host \
    -v "${PROJECT_DIR}:/workspace/qwen_ego_oops_lora_dataloaders" \
    -v "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
    -v "${HF_CACHE}:/cache/huggingface" \
    -e HF_HOME=/cache/huggingface \
    -e TRANSFORMERS_CACHE=/cache/huggingface \
    -e ONNX_AUTO_INSTALL="${ONNX_AUTO_INSTALL}" \
    -w /workspace/qwen_ego_oops_lora_dataloaders \
    --entrypoint /bin/bash \
    "${IMAGE_NAME}" \
    -lc 'if [[ "${ONNX_AUTO_INSTALL}" == "true" ]]; then python - <<'"'"'PY'"'"' || python -m pip install onnx
try:
    import onnx
except ModuleNotFoundError:
    raise SystemExit(1)
PY
fi
python scripts/export_lora_netron_onnx.py "$@"' \
    _ \
    --module "${module}" \
    --checkpoint "${checkpoint}" \
    --model-name "${MODEL_NAME}" \
    --output-dir "${output_dir}" \
    --title "${title}" \
    --view summary
}

run_export "A" "${MODULE_A_CHECKPOINT}" "${MODULE_A_OUTPUT_DIR}" "Module A LoRA topology for Netron"
run_export "B" "${MODULE_B_CHECKPOINT}" "${MODULE_B_OUTPUT_DIR}" "Module B LoRA topology for Netron"

echo "Wrote Netron ONNX exports:"
echo "- ${MODULE_A_OUTPUT_DIR}/lora_netron_simplified.onnx"
echo "- ${MODULE_B_OUTPUT_DIR}/lora_netron_simplified.onnx"
