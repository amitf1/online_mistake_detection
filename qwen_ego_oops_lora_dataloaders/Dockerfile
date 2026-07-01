ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_SPEC=torch>=2.10.0,<2.11.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    byobu \
    build-essential \
    ca-certificates \
    ffmpeg \
    git \
    libglib2.0-0 \
    libgl1 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel

# Install torch first so Unsloth does not choose a CPU or incompatible wheel.
RUN pip install "${TORCH_SPEC}" torchvision --index-url "${TORCH_INDEX_URL}"

# Install Unsloth without dependency resolution to preserve the CUDA torch wheel.
RUN pip install unsloth==2026.6.7 unsloth_zoo==2026.6.5 --no-deps

COPY requirements.txt ./

RUN pip install -r requirements.txt
RUN pip install vastai

CMD ["bash"]
