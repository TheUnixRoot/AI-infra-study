# syntax=docker/dockerfile:1
# ==============================================================================
# AI-infra-study: entorno reproducible para las practicas B1.x (local y Cloud)
# Herramientas: nvcc, cuBLAS, Nsight Systems (nsys), Nsight Compute (ncu), perf,
# Python + PyTorch (CUDA) con Triton incluido en la wheel de Linux.
#
# BUILD
#   Local, GTX 1050 (sm_61). Imagen ligera, torch en una venv montada (.venv-linux):
#     docker build -t ai-infra-study:cuda12.6 --build-arg INSTALL_TORCH=0 .
#   Cloud, autocontenida (torch dentro). Ejemplos de CUDA_ARCH:
#     L4=sm_89  L40S=sm_89  H100=sm_90  A100=sm_80  GTX1050=sm_61
#     docker build -t ai-infra-study:cloud --build-arg CUDA_ARCH=sm_90 .
#   Nota: CUDA >= 13 ya no compila sm_61. Para GPUs nuevas puedes subir CUDA_VERSION
#   (p. ej. 12.8.1, CUDA_PKG=12-8, TORCH_INDEX=cu128).
#
# RUN (requiere NVIDIA Container Toolkit; Docker Desktop/WSL2 ya lo trae)
#   docker run --rm --gpus all -e CUDA_ARCH=sm_90 -v "$PWD":/work ai-infra-study:cloud \
#       bash -c "cd b1_2_cuda_gemm && ./run.sh --full"
#   Profiling (nsys/ncu pueden necesitar capacidades extra):
#       docker run --rm --gpus all --cap-add SYS_ADMIN -v "$PWD":/work ai-infra-study:cloud ...
#   Nota: nsys dentro de WSL2 captura pero no puede importar la traza con esta GPU (usa nsys nativo en Windows o Linux).
#   Cloud: recuerda el auto-kill de la VM antes de nada:  sudo shutdown -h +90
# ==============================================================================
ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

ARG CUDA_PKG=12-6
ARG TORCH_INDEX=cu126
ARG INSTALL_TORCH=1
# Arquitectura por defecto que usan los runners (run.sh / Makefile de B1.2)
ARG CUDA_ARCH=sm_61

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_ARCH=${CUDA_ARCH} \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-${CUDA_PKG} cuda-cudart-dev-${CUDA_PKG} libcublas-dev-${CUDA_PKG} \
        cuda-nsight-systems-${CUDA_PKG} cuda-nsight-compute-${CUDA_PKG} \
        build-essential make git curl ca-certificates \
        python3 python3-venv python3-pip \
        linux-tools-common linux-tools-generic \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA) + dependencias de las practicas, en una venv propia de la imagen.
# --no-cache-dir: no dejar la cache de pip (varios GB) dentro de la capa.
RUN if [ "${INSTALL_TORCH}" = "1" ]; then \
        python3 -m venv /opt/venv \
        && /opt/venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/${TORCH_INDEX} \
        && /opt/venv/bin/pip install --no-cache-dir numpy huggingface_hub; \
    fi

WORKDIR /work
CMD ["bash"]
