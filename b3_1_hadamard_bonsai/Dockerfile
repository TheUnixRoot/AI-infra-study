# syntax=docker/dockerfile:1
# ==============================================================================
# B3.1 Hadamard + Ternary Bonsai (Qwen3.5-2B vs Ternary-Bonsai-2-2B)
# Extiende la imagen local ai-infra-study:cuda12.6 para reutilizar capas en C:
# y mantener clones, compilaciones (.build/), modelos (models/) y logits en D:.
#
# Build previo (una vez si no existe):
#   docker build -t ai-infra-study:cuda12.6 --build-arg INSTALL_TORCH=0 .
# Build de esta practica:
#   docker build -t ai-infra-study:llama-cuda12.6 -f b3_1_hadamard_bonsai/Dockerfile b3_1_hadamard_bonsai
# ==============================================================================
ARG BASE_IMAGE=ai-infra-study:cuda12.6
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PATH=/opt/hf-venv/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        ninja-build \
        libssl-dev \
        libgomp1 \
        unzip \
        time \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/hf-venv \
    && /opt/hf-venv/bin/pip install --no-cache-dir huggingface_hub hf_transfer

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvml-dev-12-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
CMD ["bash"]
