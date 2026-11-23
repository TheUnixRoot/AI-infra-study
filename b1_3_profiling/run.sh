#!/bin/sh
# ==============================================================================
# GPU Systems Reproducible Benchmark Runner: B1.3 Profiling nsys & Linux perf
# Target Hardware: NVIDIA GeForce GTX 1050 (sm_61, 2GB VRAM) + Linux CPU
# ==============================================================================

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

ARTIFACTS_DIR="${ROOT_DIR}/artifacts"
SUMMARY_FILE="${ARTIFACTS_DIR}/b1_3_summary.json"
BAD_TRACE="${ARTIFACTS_DIR}/b1_3_bad.nsys-rep"
OPT_TRACE="${ARTIFACTS_DIR}/b1_3_opt.nsys-rep"
PYTHON_SCRIPT="${SCRIPT_DIR}/b1_3_antipatterns.py"

print_header() {
    echo "=============================================================================="
    echo "  RUNNER B1.3: GPU Profiling (nsys & Linux perf) - Antipatterns Benchmark"
    echo "  Target Hardware: NVIDIA GeForce GTX 1050 (sm_61, 2GB VRAM) + Linux CPU"
    echo "=============================================================================="
}

show_help() {
    cat << EOF
Uso: ./run.sh [OPCIONES]

Opciones:
  --smoke-test         Ejecuta una verificación rápida (< 30s) con iteraciones reducidas.
  --mode [bad|opt|all] Modo de ejecución (por defecto: all).
  --dim N              Dimensión N del tensor NxN (por defecto: 2048 -> 16 MiB FP32).
  --num-ops N          Número de operaciones element-wise diminutas para Antipatrón C (por defecto: 200).
  --iterations N       Número de iteraciones cronometradas (por defecto: 20).
  --warmup N           Número de iteraciones de warmup (por defecto: 5).
  --output PATH        Ruta personalizada para el JSON de resumen (por defecto: ./artifacts/b1_3_summary.json).
  --force-simulation   Fuerza la ejecución en modo modelo de hardware sin invocar nsys/CUDA.
  -h, --help           Muestra este mensaje de ayuda.

Salidas:
  - Stdout:            Tablas de perfilado cuda_api_sum, cuda_gpu_kern_sum y resumen comparativo.
  - Trazas Nsys:       ${BAD_TRACE} y ${OPT_TRACE}
  - Resumen JSON:      ${SUMMARY_FILE}
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
    echo "[FATAL ERROR] No se encontró intérprete de Python en el PATH." >&2
    exit 1
fi

echo "[INFO] Usando intérprete: $("${PYTHON_BIN}" --version 2>&1)"
echo "[INFO] Directorio de trabajo: ${SCRIPT_DIR}"
echo "[INFO] Archivo de resumen configurado: ${SUMMARY_FILE}"

# 4. Detección de nsys (NVIDIA Nsight Systems)
NSYS_BIN=""
if command -v nsys >/dev/null 2>&1; then
    NSYS_BIN="nsys"
else
    # Buscar en rutas estándar de CUDA y Nsight
    for cand in /usr/local/cuda/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys /usr/bin/nsys; do
        if [ -x "${cand}" ]; then
            NSYS_BIN="${cand}"
            break
        fi
    done
fi

if [ -n "${NSYS_BIN}" ]; then
    echo "[INFO] Profiler NVIDIA Nsight Systems: ${NSYS_BIN} [DISPONIBLE]"
else
    echo "[AVISO] 'nsys' no detectado en el PATH ni en rutas estándar de CUDA."
fi

# 5. Detección de Linux perf
PERF_BIN=""
if command -v perf >/dev/null 2>&1; then
    PERF_BIN="perf"
else
    for cand in /usr/bin/perf /usr/lib/linux-tools/*/perf; do
        if [ -x "${cand}" ]; then
            PERF_BIN="${cand}"
            break
        fi
    done
fi

if [ -n "${PERF_BIN}" ]; then
    echo "[INFO] Linux perf: ${PERF_BIN} [DISPONIBLE]"
else
    echo "[AVISO] 'perf' no detectado en el host."
fi

# 6. Verificación de PyTorch y CUDA en Python
CUDA_READY="false"
CUDA_DIAG=$("${PYTHON_BIN}" -c '
import sys
try:
    import torch
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024*1024)
        print(f"READY:{dev} (sm_{cap[0]}{cap[1]}, {vram:.0f} MB VRAM)")
    else:
        print("NO_CUDA:CUDA no disponible en torch")
except ImportError:
    print("NO_TORCH:PyTorch no instalado")
' 2>/dev/null || echo "ERROR:Fallo en sonda Python")

case "${CUDA_DIAG}" in
    READY:*)
        CUDA_READY="true"
        echo "[INFO] GPU Física detectada: ${CUDA_DIAG#READY:}"
        ;;
    NO_CUDA:*)
        echo "[INFO] PyTorch presente pero sin aceleración CUDA en este entorno."
        ;;
    NO_TORCH:*)
        echo "[INFO] PyTorch no instalado en este entorno. Usando emulador de hardware Pascal."
        ;;
    *)
        echo "[INFO] Diagnóstico de hardware: ${CUDA_DIAG}"
        ;;
esac

# 7. Ejecución con nsys real vs Emulación Arquitectural
START_TIME=$(date +%s)

FORCE_SIM=0
for arg in "$@"; do
    if [ "${arg}" = "--force-simulation" ]; then
        FORCE_SIM=1
    fi
done

if [ -n "${NSYS_BIN}" ] && [ "${CUDA_READY}" = "true" ] && [ "${FORCE_SIM}" -eq 0 ]; then
    echo "------------------------------------------------------------------------------"
    echo "[EXEC] Ejecutando nsys profile sobre MODO BAD..."
    echo "------------------------------------------------------------------------------"
    "${NSYS_BIN}" profile \
        --trace=cuda,nvtx,osrt \
        --stats=true \
        --force-overwrite=true \
        -o "${ARTIFACTS_DIR}/b1_3_bad" \
        "${PYTHON_BIN}" "${PYTHON_SCRIPT}" --mode bad --output "${SUMMARY_FILE}" "$@" || echo "[AVISO] nsys termino con error en modo BAD."
    [ -f "${ARTIFACTS_DIR}/b1_3_bad.nsys-rep" ] || echo "[AVISO] No se genero b1_3_bad.nsys-rep (nsys no pudo trazar CUDA; p.ej. WSL2 + Pascal). Sin traza real no hay nsys stats."

    echo "------------------------------------------------------------------------------"
    echo "[EXEC] Ejecutando nsys profile sobre MODO OPT..."
    echo "------------------------------------------------------------------------------"
    "${NSYS_BIN}" profile \
        --trace=cuda,nvtx,osrt \
        --stats=true \
        --force-overwrite=true \
        -o "${ARTIFACTS_DIR}/b1_3_opt" \
        "${PYTHON_BIN}" "${PYTHON_SCRIPT}" --mode opt --output "${SUMMARY_FILE}" "$@" || echo "[AVISO] nsys termino con error en modo OPT."
    [ -f "${ARTIFACTS_DIR}/b1_3_opt.nsys-rep" ] || echo "[AVISO] No se genero b1_3_opt.nsys-rep (nsys no pudo trazar CUDA; p.ej. WSL2 + Pascal). Sin traza real no hay nsys stats."

    echo "------------------------------------------------------------------------------"
    echo "[EXEC] Generando consolidado y tablas estadísticas cuda_api_sum / cuda_gpu_kern_sum..."
    echo "------------------------------------------------------------------------------"
    "${PYTHON_BIN}" "${PYTHON_SCRIPT}" --mode all --output "${SUMMARY_FILE}" "$@"

else
    echo "------------------------------------------------------------------------------"
    echo "[AVISO] Sin nsys real: se miden tiempos con CUDA events, pero NO se generan trazas .nsys-rep."
    echo "        Instala Nsight Systems para obtener nsys stats (cuda_api_sum / cuda_gpu_kern_sum)."
    echo "------------------------------------------------------------------------------"

    "${PYTHON_BIN}" "${PYTHON_SCRIPT}" --mode all --output "${SUMMARY_FILE}" "$@"
fi

# 8. Evaluación de Linux perf si está disponible y habilitado
if [ -n "${PERF_BIN}" ] && [ "${FORCE_SIM}" -eq 0 ]; then
    echo "[INFO] Capturando contadores de hardware de CPU con Linux perf..."
    "${PERF_BIN}" stat -e cycles,instructions,cache-misses,page-faults,context-switches \
        "${PYTHON_BIN}" "${PYTHON_SCRIPT}" --smoke-test --quiet 2>"${ARTIFACTS_DIR}/b1_3_perf_stat.txt" || true
    if [ -f "${ARTIFACTS_DIR}/b1_3_perf_stat.txt" ]; then
        echo "[OK] Contadores de hardware perf guardados en: ${ARTIFACTS_DIR}/b1_3_perf_stat.txt"
    fi
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "=============================================================================="
echo "  REPORTE FINAL B1.3"
echo "=============================================================================="
if [ -f "${BAD_TRACE}" ] && [ -f "${OPT_TRACE}" ]; then echo "  - Estado:               COMPLETADO (con trazas nsys reales)"; else echo "  - Estado:               COMPLETADO SIN TRAZAS NSYS (solo tiempos CUDA events)"; fi
echo "  - Tiempo total ejec:    ${ELAPSED} segundos"
[ -f "${BAD_TRACE}" ] && echo "  - Traza BAD (.nsys-rep): ${BAD_TRACE}"
[ -f "${OPT_TRACE}" ] && echo "  - Traza OPT (.nsys-rep): ${OPT_TRACE}"
echo "  - Resumen JSON:         ${SUMMARY_FILE}"
echo "=============================================================================="

# 9. Verificación de integridad del JSON
if [ -f "${SUMMARY_FILE}" ]; then
    "${PYTHON_BIN}" -c '
import json, sys
with open(sys.argv[1], "r") as f:
    d = json.load(f)
speedup = d["speedup_metrics"]["step_time_speedup_x"]
launches_red = d["speedup_metrics"]["kernel_launch_reduction_x"]
print(f"[VERIFICACIÓN] JSON válido. Speedup de Paso: {speedup}x | Reducción de Kernels: {launches_red}x")
' "${SUMMARY_FILE}"
else
    echo "[ERROR] El archivo de resumen ${SUMMARY_FILE} no fue generado." >&2
    exit 1
fi
