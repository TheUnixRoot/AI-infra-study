#!/usr/bin/env bash
# ==============================================================================
# GPU Systems Reproducible Benchmark Runner: B1.2 CUDA GEMM & Warp Softmax
# Target Hardware: NVIDIA GeForce GTX 1050 (sm_61, 2GB VRAM, ~112 GB/s)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARTIFACTS_DIR="${ROOT_DIR}/artifacts"
JSON_OUTPUT="${ARTIFACTS_DIR}/b1_2_results.json"
CSV_OUTPUT="${ARTIFACTS_DIR}/b1_2_results.csv"

# Binarios destino
GEMM_BIN="${SCRIPT_DIR}/gemm_bench"
SOFTMAX_BIN="${SCRIPT_DIR}/warp_online_softmax"

NVCC_FLAGS="-O3 -arch=${CUDA_ARCH:-sm_61} -std=c++14 -Xcompiler -fPIC,-Wall,-Wextra"

print_header() {
    echo "=============================================================================="
    echo "  RUNNER B1.2: CUDA C++ GEMM & Warp Online Softmax (GTX 1050 / sm_61)"
    echo "=============================================================================="
}

show_help() {
    cat << EOF
Uso: $0 [OPCIONES]

Opciones:
  --smoke, --smoke-test   Ejecuta verificacion rapida (< 30s) en tamano 256x256.
  --full                  Ejecuta benchmark completo (1024x1024 y 2048x2048).
  --clean                 Elimina binarios compilados y temporales.
  -h, --help              Muestra este mensaje de ayuda.

Salidas:
  - Stdout:               Tablas Markdown limpias para cuaderno de estudio.
  - JSON:                 ${JSON_OUTPUT}
  - CSV:                  ${CSV_OUTPUT}
EOF
}

clean_artifacts() {
    echo "[INFO] Limpiando binarios generados..."
    rm -f "${GEMM_BIN}" "${SOFTMAX_BIN}"
    echo "[INFO] Limpieza completada."
}

# 1. Parse de banderas
MODE="smoke"
CLEAN_ONLY=false

for arg in "$@"; do
    case "${arg}" in
        --smoke|--smoke-test)
            MODE="smoke"
            ;;
        --full)
            MODE="full"
            ;;
        --clean)
            CLEAN_ONLY=true
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "[WARNING] Argumento no reconocido: ${arg}"
            ;;
    esac
done

if [ "${CLEAN_ONLY}" = true ]; then
    clean_artifacts
    exit 0
fi

print_header

# 2. Idempotencia: Asegurar existencia de directorio de artefactos
mkdir -p "${ARTIFACTS_DIR}"

# 3. Diagnóstico de herramientas CUDA y hardware GPU
echo "[1/4] Verificando subsistema CUDA y GPU..."

if ! command -v nvcc >/dev/null 2>&1; then
    echo "------------------------------------------------------------------------------"
    echo "[ERROR CRITICO] 'nvcc' no fue encontrado en el PATH del sistema."
    echo ""
    echo "Instrucciones de resolucion para ejecutar en el entorno destino (Linux x86_64 / CUDA):"
    echo "  1. Asegurate de tener instalado el NVIDIA CUDA Toolkit (v11.x o v12.x):"
    echo "     sudo apt-get install -y nvidia-cuda-toolkit"
    echo "  2. Agrega el compilador a tu variable de entorno PATH:"
    echo "     export PATH=/usr/local/cuda/bin:\$PATH"
    echo "     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    echo "  3. Vuelve a ejecutar este script:"
    echo "     ./run.sh --smoke"
    echo "------------------------------------------------------------------------------"
    exit 1
fi

echo "  - Compilador: $(nvcc --version | grep "release" | sed 's/^[ \t]*//')"

GPU_NAME="NVIDIA GeForce GTX 1050 (Detectada por defecto sm_61)"
GPU_DRIVER="N/A"
GPU_CLOCKS="N/A"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  - nvidia-smi: DISPONIBLE"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || echo "${GPU_NAME}")
    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 || echo "N/A")
    GPU_CLOCKS=$(nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv,noheader | head -n1 || echo "N/A")
    echo "  - GPU: ${GPU_NAME}"
    echo "  - Driver NVIDIA: ${GPU_DRIVER}"
    echo "  - Relojes activos (SM / Mem): ${GPU_CLOCKS}"
else
    echo "  - nvidia-smi: NO DISPONIBLE (usando configuracion teorica sm_61 / Pascal)"
fi

# 4. Compilacion de binarios con verificacion estricta
echo ""
echo "[2/4] Compilando kernels CUDA C++ (-arch=sm_61 -O3 -lcublas)..."

echo "  -> Compilando gemm_bench.cu..."
nvcc ${NVCC_FLAGS} "${SCRIPT_DIR}/gemm_bench.cu" -lcublas -o "${GEMM_BIN}"

echo "  -> Compilando warp_online_softmax.cu..."
nvcc ${NVCC_FLAGS} "${SCRIPT_DIR}/warp_online_softmax.cu" -o "${SOFTMAX_BIN}"

echo "  [OK] Binarios compilados exitosamente en ${SCRIPT_DIR}/"

# 5. Ejecucion de Smoke-Test (<30s)
echo ""
echo "[3/4] Ejecutando Smoke-Test de verificacion rapida (dim 256x256)..."
RESULTS_TMP="$(mktemp -d)"
trap 'rm -rf "${RESULTS_TMP}"' EXIT
"${GEMM_BIN}" --smoke > "${RESULTS_TMP}/gemm.out"; cat "${RESULTS_TMP}/gemm.out"
"${SOFTMAX_BIN}" --smoke > "${RESULTS_TMP}/softmax.out"; cat "${RESULTS_TMP}/softmax.out"

if [ "${MODE}" = "smoke" ]; then
    echo ""
    echo "[INFO] Modo --smoke finalizado con exito. Todos los checks numericos pasaron."
fi

# 6. Ejecucion de Benchmark Completo si se especifico --full
if [ "${MODE}" = "full" ]; then
    echo ""
    echo "[4/4] Ejecutando Benchmark Completo (1024x1024 y 2048x2048)..."
    "${GEMM_BIN}" --full > "${RESULTS_TMP}/gemm.out"; cat "${RESULTS_TMP}/gemm.out"

    echo ""
    "${SOFTMAX_BIN}" --rows 1024 --cols 1024 --warmup 20 --iters 100 > "${RESULTS_TMP}/softmax.out"; cat "${RESULTS_TMP}/softmax.out"
fi

# 7. Generacion de Artefactos (CSV y JSON)
echo ""
echo "[INFO] Generando artefactos en ${ARTIFACTS_DIR}..."

# CSV + JSON con las medidas reales parseadas de la salida de los binarios
python3 "${SCRIPT_DIR}/parse_results.py" "${RESULTS_TMP}/gemm.out" "${RESULTS_TMP}/softmax.out"     "${MODE}" "${GPU_NAME}" "${GPU_DRIVER}" "${JSON_OUTPUT}" "${CSV_OUTPUT}"

echo "  -> JSON generado: ${JSON_OUTPUT}"
echo "  -> CSV generado:  ${CSV_OUTPUT}"
echo ""
echo "=============================================================================="
echo "  [RESUMEN FINAL] Experimento B1.2 completado exitosamente."
echo "=============================================================================="
