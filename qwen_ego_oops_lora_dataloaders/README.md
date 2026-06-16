# Qwen OMD Dataloaders

PyTorch datasets and Qwen ChatML collators for three-stage online procedural
mistake detection:

1. Module A: online gatekeeper / step completion detection.
2. Module B: temporal bounding from overshot positive buffers.
3. Module C: mistake detection and short reasoning generation.

The first implementation supports EgoOops. MATT-Bench, Assembly101-O,
Epic-Tent-O, and EgoPER are represented by placeholders that document the
normalization contract.

## Install

```bash
cd /nvcr/users/afeldman/experiments/online_mistake_detection/qwen_ego_oops_lora_dataloaders
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For 16GB GPU experiments, start with the 2B Qwen3.5 vision model through the
Unsloth Module A training script. Its defaults are conservative: bf16/16-bit
LoRA instead of QLoRA, batch size 1, gradient accumulation 16, LoRA rank 8,
1 FPS video messages, 1024 token context, and vision LoRA disabled by default.

## Smoke Test

```bash
python scripts/smoke_dataloaders.py \
  --video-root /nvcr/users/afeldman/data/exper/videos-processed-720p \
  --video-ids S1800001 S1800002
```

Add `--load-processor --model-id <model-or-local-path>` to validate real Qwen
processor batches. Without that flag the script validates dataset examples only.

## Module A Unsloth Training

Dataset-only dry run:

```bash
python scripts/train_module_a_unsloth.py --dry-run --max-videos 2
```

Tiny training smoke run:

```bash
python scripts/train_module_a_unsloth.py \
  --max-videos 2 \
  --max-steps 2 \
  --output-dir /nvcr/users/afeldman/omd/module_a_qwen35_2b_smoke
```

TensorBoard logging is enabled by default. To watch loss curves:

```bash
tensorboard --logdir /nvcr/users/afeldman/omd/module_a_qwen35_2b_smoke
```

To try the 4B model after the pipeline is stable, override the defaults:

```bash
python scripts/train_module_a_unsloth.py \
  --model-name unsloth/Qwen3.5-4B \
  --max-videos 2 \
  --max-steps 2 \
  --output-dir /nvcr/users/afeldman/omd/module_a_qwen35_4b_smoke
```
