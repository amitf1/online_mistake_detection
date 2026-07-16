# LoRA Architecture Rendering

Render a flowchart-style diagram and LoRA module tables from a saved PEFT adapter checkpoint.
The renderer does not retrain and does not export ONNX.

## Module A Example

```bash
ARCH_MODULE=A \
ARCH_CHECKPOINT=/home/amit/online_mistake_detection/outputs/module_a_qwen35_2b_lora_recall_loss_weighted_r16_frames32/runs/module_a_recall_loss2_r16_frames32_seq6144_prompt_v2/best_ep2 \
ARCH_OUTPUT_DIR=/home/amit/online_mistake_detection/outputs/lora_architecture/module_a_best_ep2 \
bash qwen_ego_oops_lora_dataloaders/scripts/render_lora_architecture_docker.sh
```

## Module B Example

```bash
ARCH_MODULE=B \
ARCH_CHECKPOINT=/home/amit/online_mistake_detection/outputs/module_b_qwen35_lora_grounding/runs/module_b_24f_336vr_with_no_action/best_ep1 \
ARCH_OUTPUT_DIR=/home/amit/online_mistake_detection/outputs/lora_architecture/module_b_24f_336vr_best_ep1 \
bash qwen_ego_oops_lora_dataloaders/scripts/render_lora_architecture_docker.sh
```

## Module C Example

After Module C has a best checkpoint:

```bash
ARCH_MODULE=C \
ARCH_CHECKPOINT=/home/amit/online_mistake_detection/outputs/module_c_qwen35_lora_reasoning/runs/<run_name>/best_ep<epoch> \
ARCH_OUTPUT_DIR=/home/amit/online_mistake_detection/outputs/lora_architecture/module_c_best \
bash qwen_ego_oops_lora_dataloaders/scripts/render_lora_architecture_docker.sh
```

## Optional W&B Upload

Add these variables to any command above:

```bash
ARCH_LOG_TO_WANDB=true \
WANDB_PROJECT=qwen-omd \
WANDB_NAME=module_a_best_lora_architecture
```

Outputs:
- `lora_architecture.svg`
- `lora_architecture.png` when SVG-to-PNG conversion is available in the container
- `lora_modules.csv`
- `lora_modules.json`
- `lora_summary.json`

## Netron ONNX Export

Create lightweight ONNX topology graphs for the current best Module A and Module B adapters:

```bash
bash qwen_ego_oops_lora_dataloaders/scripts/export_best_lora_netron_onnx_docker.sh
```

Outputs:
- `/home/amit/online_mistake_detection/outputs/netron_onnx/module_a_recall_r8_best_ep9/lora_netron_simplified.onnx`
- `/home/amit/online_mistake_detection/outputs/netron_onnx/module_b_wandb_9kahxivi_best_ep7/lora_netron_simplified.onnx`

These ONNX files are for Netron architecture inspection only. The default export is simplified into major blocks with LoRA counts, ranks, alpha/scaling, and projection summaries as node attributes; it is not a full executable Qwen inference graph.
