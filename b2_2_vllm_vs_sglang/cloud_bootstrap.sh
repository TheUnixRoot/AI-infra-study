#!/usr/bin/env bash
# ==============================================================================
# cloud_bootstrap.sh - PLANTILLA de arranque para la VM Cloud-2 (1x L40S o 1x H100)
# Ejecutar en la VM, tras clonar el repo:  bash b2_2_vllm_vs_sglang/cloud_bootstrap.sh
# No se ejecuta en local. No llama a ningun servicio cloud: solo prepara la VM ya alquilada.
# ==============================================================================
set -euo pipefail

# 1) AUTO-KILL PRIMERO (AGENTS.md 2): nunca dejes una GPU encendida.
echo "[1/5] Programando apagado automatico en 90 min ..."
sudo shutdown -h +90 || echo "[WARN] NO se pudo programar el apagado: apaga la VM a mano al terminar."
echo "      (cancelar: sudo shutdown -c | ahora: sudo shutdown -h now)"

# 2) GPU
echo "[2/5] Verificando GPU ..."
command -v nvidia-smi >/dev/null || { echo "[ERROR] nvidia-smi no esta: imagen sin driver NVIDIA." >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
case "${CC}" in
    8.9|9.0) echo "      sm_${CC/./} OK (L40S/H100)" ;;
    *) echo "[ERROR] compute capability ${CC}: se esperaba 8.9 (L40S) o 9.0 (H100)." >&2; exit 1 ;;
esac

# 3) Venvs separadas: vLLM y SGLang fijan versiones distintas de torch/flashinfer.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"
echo "[3/5] Creando venvs (pip --no-cache-dir) ..."
python3 -m venv .venv-vllm
python3 -m venv .venv-sglang
.venv-vllm/bin/pip install --no-cache-dir -U pip vllm aiohttp flashinfer-python
.venv-sglang/bin/pip install --no-cache-dir -U pip "sglang[all]" aiohttp
# Si el wheel de FlashInfer no coincide con tu CUDA/torch, sigue la guia oficial de FlashInfer
# (indice https://flashinfer.ai/whl/cu1xx/torchx.y) y fija versiones; anotalas en el informe.

# 4) Modelo (descarga una vez, ~15 GB). Token HF solo por variable de entorno; nunca al repo.
echo "[4/5] Descargando modelo ..."
.venv-vllm/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-Coder-7B-Instruct')"

# 5) Versiones usadas (para reproducibilidad) y siguiente paso
echo "[5/5] Versiones:"
.venv-vllm/bin/pip list 2>/dev/null | grep -Ei '^(vllm|torch|flashinfer)' || true
.venv-sglang/bin/pip list 2>/dev/null | grep -Ei '^(sglang|torch|flashinfer)' || true
cat <<'EOF'

Listo. Ejecuta (tmux recomendado):
  PATH="$PWD/.venv-vllm/bin:$PATH" VLLM_BIN=.venv-vllm/bin/vllm SGLANG_PY=.venv-sglang/bin/python \
    bash b2_2_vllm_vs_sglang/run_compare.sh
Recoge artifacts/b2_2_* y APAGA la VM.
EOF
