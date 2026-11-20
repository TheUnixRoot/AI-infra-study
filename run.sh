#!/bin/sh
# ==============================================================================
# GPU Systems Reproducible Benchmark Runner: B1.1 Strides & Memory Layout
# Target Hardware: NVIDIA GeForce GTX 1050 (sm_61, 2GB VRAM, ~112 GB/s)
# ==============================================================================

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

ARTIFACTS_DIR="${SCRIPT_DIR}/artifacts"
METRICS_FILE="${ARTIFACTS_DIR}/b1_1_metrics.json"
PYTHON_SCRIPT="${SCRIPT_DIR}/b1_1_strides_bench.py"

print_header() {
    echo "=============================================================================="
    echo "  RUNNER B1.1: GPU Strides & Memory Layout Benchmark (GTX 1050 / sm_61)"
    echo "=============================================================================="
}

show_help() {
    cat << EOF
Uso: ./run.sh [OPCIONES]

Opciones:
  --smoke-test         Ejecuta una verificación rápida (< 30s) con iteraciones reducidas.
  --size N             Dimensión N de la matriz NxN (por defecto: 2048 -> 16 MiB FP32).
  --iterations N       Número de iteraciones cronometradas (por defecto: 50).
  --warmup N           Número de iteraciones de warmup (por defecto: 20).
  --output PATH        Ruta personalizada para el JSON de métricas (por defecto: ./artifacts/b1_1_metrics.json).
  --force-simulation   Fuerza la ejecución en modo simulación aunque haya CUDA disponible.
  -h, --help           Muestra este mensaje de ayuda.

Salidas:
  - Stdout:            Tabla resumen con análisis de arquitectura y saturación de bus DRAM.
  - Archivo JSON:      ${METRICS_FILE}
EOF
}

# 1. Parse help flag early
for arg in "$@"; do
    if [ "${arg}" = "-h" ] || [ "${arg}" = "--help" ]; then
        show_help
        exit 0
    fi
done

print_header

# 2. Idempotencia: Asegurar existencia de directorio de artefactos
mkdir -p "${ARTIFACTS_DIR}"

# 3. Detección de intérprete Python
PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif [ -x "/data/data/com.termux/files/usr/bin/python3" ]; then
    PYTHON_BIN="/data/data/com.termux/files/usr/bin/python3"
elif [ -x "/data/data/com.termux/files/usr/bin/python" ]; then
    PYTHON_BIN="/data/data/com.termux/files/usr/bin/python"
else
    echo "[FATAL ERROR] No se encontró intérprete de Python (python3 o python) en el PATH." >&2
    exit 1
fi

echo "[INFO] Usando intérprete: $("${PYTHON_BIN}" --version 2>&1)"
echo "[INFO] Directorio de trabajo: ${SCRIPT_DIR}"
echo "[INFO] Archivo de salida configurado: ${METRICS_FILE}"

# 4. Diagnóstico del subsistema CUDA y PyTorch
echo "[INFO] Verificando subsistema CUDA y PyTorch..."
"${PYTHON_BIN}" -c '
import sys
try:
    import torch
    print(f"  - PyTorch: {torch.__version__} [INSTALADO]")
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024*1024)
        print(f"  - CUDA: DISPONIBLE")
        print(f"  - GPU Dispositivo 0: {dev_name} (sm_{cap[0]}{cap[1]}, {vram_mb:.0f} MB VRAM)")
    else:
        print("  - CUDA: NO DISPONIBLE (El benchmark usará verificación CPU y modelado de hardware)")
except ImportError:
    print("  - PyTorch: NO INSTALADO en este entorno (El benchmark ejecutará el modelo analítico de arquitectura)")
' || true

# 5. Ejecución del benchmark
echo "[INFO] Iniciando benchmark B1.1..."
START_TIME=$(date +%s)

"${PYTHON_BIN}" "${PYTHON_SCRIPT}" --output "${METRICS_FILE}" "$@"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "[INFO] Benchmark finalizado en ${ELAPSED} segundos."

# 6. Verificación de integridad del artefacto generado
if [ -f "${METRICS_FILE}" ]; then
    if "${PYTHON_BIN}" -c "import json; json.load(open('${METRICS_FILE}'))" >/dev/null 2>&1; then
        echo "[SUCCESS] Artefacto verificado correctamente: ${METRICS_FILE}"
        echo "[SUCCESS] Tamaño del JSON: $(wc -c < "${METRICS_FILE}") bytes"
    else
        echo "[ERROR] El archivo de métricas ${METRICS_FILE} no contiene un JSON válido." >&2
        exit 1
    fi
else
    echo "[ERROR] No se encontró el archivo de métricas esperado en ${METRICS_FILE}." >&2
    exit 1
fi

exit 0
