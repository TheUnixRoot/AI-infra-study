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
  --cpu-work-ms MS     Trabajo CPU sintético por paso en ms (por defecto: 0.0).
  --output PATH        Ruta personalizada para el JSON de resumen (por defecto: ../artifacts/b1_3_summary.json).
  --check-only         Verifica entorno/argumentos en CPU sin requerir CUDA ni escribir en artifacts/.
  -h, --help           Muestra este mensaje de ayuda.

Salidas:
  - Stdout:            Tablas de perfilado cuda_api_sum, cuda_gpu_kern_sum y resumen comparativo.
  - Trazas Nsys:       ${BAD_TRACE} y ${OPT_TRACE}
  - Resumen JSON:      ${SUMMARY_FILE}
EOF
}

# 1. Parse flags (extract --mode, --output, --check-only cleanly so passthrough args never override them)
REQ_MODE="all"
CHECK_ONLY=0
TARGET_SUMMARY="${SUMMARY_FILE}"
PASS_ARGS=""
expect_val=""
for arg in "$@"; do
    if [ -n "${expect_val}" ]; then
        case "${expect_val}" in
            mode) REQ_MODE="${arg}" ;;
            output) TARGET_SUMMARY="${arg}" ;;
        esac
        expect_val=""
        continue
    fi
    case "${arg}" in
        -h|--help)
            show_help
            exit 0
            ;;
        --mode)
            expect_val="mode"
            ;;
        --mode=*)
            REQ_MODE="${arg#--mode=}"
            ;;
        --output)
            expect_val="output"
            ;;
        --output=*)
            TARGET_SUMMARY="${arg#--output=}"
            ;;
        --check-only)
            CHECK_ONLY=1
            PASS_ARGS="${PASS_ARGS} --check-only"
            ;;
        *)
            PASS_ARGS="${PASS_ARGS} ${arg}"
            ;;
    esac
done
if [ -n "${expect_val}" ]; then
    echo "[ERROR] Falta valor para --${expect_val}." >&2
    exit 2
fi
SUMMARY_FILE="${TARGET_SUMMARY}"

print_header

# 2. Idempotencia: Asegurar existencia de directorio de artefactos
mkdir -p "${ARTIFACTS_DIR}"

# 3. Detección de intérprete Python
# Se puede forzar el intérprete con PYTHON_BIN=/ruta/python (p. ej. python.exe de una venv en
# Windows, necesario para que nsys.exe pueda lanzarlo).
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -n "${PYTHON_BIN}" ]; then
    :
elif command -v python3 >/dev/null 2>&1; then
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
        echo "[INFO] PyTorch no instalado en este entorno (sin aceleración GPU; no se fabrican métricas)."
        ;;
    *)
        echo "[INFO] Diagnóstico de hardware: ${CUDA_DIAG}"
        ;;
esac

# 7. Ejecución real con CUDA / nsys (sin simulación ni fabricación de datos)
START_TIME=$(date +%s)

if [ "${CHECK_ONLY}" -eq 1 ]; then
    # shellcheck disable=SC2086
    "${PYTHON_BIN}" "${PYTHON_SCRIPT}" --mode "${REQ_MODE}" --output "${SUMMARY_FILE}" ${PASS_ARGS}
    exit 0
fi

if [ "${CUDA_READY}" != "true" ]; then
    echo "------------------------------------------------------------------------------" >&2
    echo "[ERROR] CUDA no está disponible en este entorno. B1.3 no fabrica datos simulados (AGENTS.md §5.1)." >&2
    echo "        Los artefactos existentes en ${ARTIFACTS_DIR} se mantienen intactos." >&2
    echo "        Usa '--check-only' para validar dependencias sin GPU." >&2
    echo "------------------------------------------------------------------------------" >&2
    exit 1
fi

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) NSYS_TRACE="cuda,nvtx" ;;   # osrt es solo Linux
    *) NSYS_TRACE="cuda,nvtx,osrt" ;;
esac

if [ -n "${NSYS_BIN}" ]; then
    if [ "${REQ_MODE}" = "bad" ] || [ "${REQ_MODE}" = "all" ]; then
        echo "------------------------------------------------------------------------------"
        echo "[EXEC] Ejecutando nsys profile sobre MODO BAD..."
        echo "------------------------------------------------------------------------------"
        # shellcheck disable=SC2086
        "${NSYS_BIN}" profile \
            --trace="${NSYS_TRACE}" \
            --stats=true \
            --force-overwrite=true \
            -o "${ARTIFACTS_DIR}/b1_3_bad" \
            "${PYTHON_BIN}" "${PYTHON_SCRIPT}" --mode bad --output "${SUMMARY_FILE}" ${PASS_ARGS} || echo "[AVISO] nsys termino con error en modo BAD."
        [ -f "${ARTIFACTS_DIR}/b1_3_bad.nsys-rep" ] || echo "[AVISO] No se genero b1_3_bad.nsys-rep (nsys fallo; en WSL2 no identifica la GPU: usa nsys nativo de Windows o Linux). Sin traza real no hay nsys stats."
    fi

    if [ "${REQ_MODE}" = "opt" ] || [ "${REQ_MODE}" = "all" ]; then
        echo "------------------------------------------------------------------------------"
        echo "[EXEC] Ejecutando nsys profile sobre MODO OPT..."
        echo "------------------------------------------------------------------------------"
        # shellcheck disable=SC2086
        "${NSYS_BIN}" profile \
            --trace="${NSYS_TRACE}" \
            --stats=true \
            --force-overwrite=true \
            -o "${ARTIFACTS_DIR}/b1_3_opt" \
            "${PYTHON_BIN}" "${PYTHON_SCRIPT}" --mode opt --output "${SUMMARY_FILE}" ${PASS_ARGS} || echo "[AVISO] nsys termino con error en modo OPT."
        [ -f "${ARTIFACTS_DIR}/b1_3_opt.nsys-rep" ] || echo "[AVISO] No se genero b1_3_opt.nsys-rep (nsys fallo; en WSL2 no identifica la GPU: usa nsys nativo de Windows o Linux). Sin traza real no hay nsys stats."
    fi

    echo "------------------------------------------------------------------------------"
    echo "[EXEC] Generando consolidado (${REQ_MODE})..."
    echo "------------------------------------------------------------------------------"
    # shellcheck disable=SC2086
    "${PYTHON_BIN}" "${PYTHON_SCRIPT}" --mode "${REQ_MODE}" --output "${SUMMARY_FILE}" ${PASS_ARGS}

else
    echo "------------------------------------------------------------------------------"
    echo "[AVISO] Sin nsys real: se miden tiempos con CUDA events, pero NO se generan trazas .nsys-rep."
    echo "        Instala Nsight Systems para obtener nsys stats (cuda_api_sum / cuda_gpu_kern_sum)."
    echo "------------------------------------------------------------------------------"

    # shellcheck disable=SC2086
    "${PYTHON_BIN}" "${PYTHON_SCRIPT}" --mode "${REQ_MODE}" --output "${SUMMARY_FILE}" ${PASS_ARGS}
fi

# 8. Evaluación de Linux perf si está disponible y habilitado
if [ -n "${PERF_BIN}" ]; then
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
OUT_JSON="${SUMMARY_FILE}"
if [ "${REQ_MODE}" != "all" ]; then
    OUT_JSON="$(printf '%s' "${SUMMARY_FILE}" | sed "s/\\.json$/_${REQ_MODE}.json/")"
fi
if [ -f "${OUT_JSON}" ]; then
    if [ "${REQ_MODE}" = "all" ]; then
        "${PYTHON_BIN}" -c '
import json, sys
with open(sys.argv[1], "r") as f:
    d = json.load(f)
speedup = d["speedup_metrics"]["step_time_speedup_x"]
launches_red = d["speedup_metrics"]["kernel_launch_reduction_x"]
print(f"[VERIFICACIÓN] JSON válido. Speedup de Paso: {speedup}x | Reducción de Kernels: {launches_red}x")
' "${OUT_JSON}"
    else
        "${PYTHON_BIN}" -c "import json,sys; json.load(open(sys.argv[1]))" "${OUT_JSON}"
        echo "[VERIFICACIÓN] JSON válido para modo ${REQ_MODE}: ${OUT_JSON}"
    fi
else
    echo "[ERROR] El archivo de resumen ${OUT_JSON} no fue generado." >&2
    exit 1
fi
