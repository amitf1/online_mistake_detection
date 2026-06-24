# Installation

This project supports two setup paths:

- Docker for a reproducible GPU environment.
- A local Python virtual environment for direct development.

Clone the repository:

```bash
git clone https://github.com/amitf1/online_mistake_detection.git
cd online_mistake_detection/qwen_ego_oops_lora_dataloaders
```

The pinned package set is chosen for Qwen3.5 support with Unsloth:

- `torch>=2.10.0,<2.11.0` from a CUDA PyTorch index (`cu128` by default).
- `transformers==5.5.0`.
- `unsloth==2026.6.7`.
- `unsloth_zoo==2026.6.5`.
- `datasets>=3.4.1,!=4.0.*,!=4.1.0,<4.4.0`.

## Docker

Prerequisites:

- NVIDIA driver installed on the host.
- Docker installed.
- NVIDIA Container Toolkit installed so `docker run --gpus all ...` works.

Build the image from the project root:

```bash
docker build -t qwen-omd-dataloaders:latest .
```

The default Docker build uses a CUDA 12.8 image and `cu128` PyTorch wheels. This
works on hosts with a new enough NVIDIA driver, including hosts where
`nvidia-smi` reports CUDA 13.2.

For a CUDA 12.4 host, build with a CUDA 12.4 base image and `cu124` PyTorch
wheels:

```bash
docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  --build-arg 'TORCH_SPEC=torch>=2.4.0,<2.11.0' \
  -t qwen-omd-dataloaders:cu124 .
```

Run the CUDA 12.4 image:

```bash
IMAGE_NAME=qwen-omd-dataloaders:cu124 bash scripts/run_module_a_docker.sh
```

Verify GPU access inside the image:

```bash
docker run --rm --gpus all qwen-omd-dataloaders:latest \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Run Module A training with the default paths:

```bash
bash scripts/run_module_a_docker.sh
```

The script runs:

```bash
python scripts/train_module_a_unsloth.py \
  --max-videos 50 \
  --max-steps 100 \
  --video-root /home/amit/online_mistake_detection/data/videos-processed-720p \
  --output-dir /home/amit/online_mistake_detection/outputs/module_a_qwen35_2b_lora \
  --report-to tensorboard
```

Override paths if needed:

```bash
DATA_ROOT=/path/to/videos \
OUTPUT_ROOT=/path/to/outputs \
HF_CACHE=/path/to/huggingface-cache \
bash scripts/run_module_a_docker.sh
```

Use Weights & Biases instead of TensorBoard:

```bash
wandb login
REPORT_TO=wandb WANDB_PROJECT=qwen-omd bash scripts/run_module_a_docker.sh
```

Log to both TensorBoard and W&B:

```bash
REPORT_TO=tensorboard,wandb WANDB_PROJECT=qwen-omd bash scripts/run_module_a_docker.sh
```

Run W&B in offline mode:

```bash
REPORT_TO=wandb WANDB_MODE=offline bash scripts/run_module_a_docker.sh
```

## Virtual Environment

Create and activate a venv:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install PyTorch first, then Unsloth without dependency resolution, then the rest:

```bash
pip install --upgrade pip setuptools wheel
pip install "torch>=2.10.0,<2.11.0" torchvision --index-url https://download.pytorch.org/whl/cu128
pip install unsloth==2026.6.7 unsloth_zoo==2026.6.5 --no-deps
pip install -r requirements.txt
pip install -e .
```

For CUDA 12.4, install PyTorch from the `cu124` index instead:

```bash
pip install --upgrade pip setuptools wheel
pip install "torch>=2.4.0,<2.11.0" torchvision --index-url https://download.pytorch.org/whl/cu124
pip install unsloth==2026.6.7 unsloth_zoo==2026.6.5 --no-deps
pip install -r requirements.txt
pip install -e .
```

Verify the install:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from transformers.models.auto.configuration_auto import CONFIG_MAPPING; print('qwen3_5' in CONFIG_MAPPING)"
pip check
```

Run Module A training:

```bash
python scripts/train_module_a_unsloth.py \
  --max-videos 50 \
  --max-steps 100 \
  --video-root /home/amit/online_mistake_detection/data/videos-processed-720p \
  --output-dir /home/amit/online_mistake_detection/outputs/module_a_qwen35_2b_lora \
  --report-to tensorboard
```

## Logging

The training script uses Hugging Face/TRL logging through `--report-to`.
TensorBoard is the default:

```bash
python scripts/train_module_a_unsloth.py ... --report-to tensorboard
tensorboard --logdir /home/amit/online_mistake_detection/outputs/module_a_qwen35_2b_lora
```

Weights & Biases is also installed. Log in once, then run with W&B reporting:

```bash
wandb login
WANDB_PROJECT=qwen-omd python scripts/train_module_a_unsloth.py ... --report-to wandb
```

To use both:

```bash
WANDB_PROJECT=qwen-omd python scripts/train_module_a_unsloth.py ... --report-to tensorboard,wandb
```

For offline W&B logging:

```bash
WANDB_MODE=offline python scripts/train_module_a_unsloth.py ... --report-to wandb
```
