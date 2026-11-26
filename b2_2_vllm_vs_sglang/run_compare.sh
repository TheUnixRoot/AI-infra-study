#!/usr/bin/env bash
# ==============================================================================
# run_compare.sh - B2.2: vLLM (V1, FlashInfer) vs SGLang (RadixAttention, FlashInfer)
#
#   --dry-run   CPU local, <15 s, sin GPU ni servidores: valida sintaxis, dependencias
#               del cliente y el parser de reportes con datos SINTETICOS
#               (salida en artifacts/b2_2_dryrun/, nunca en artifacts/b2_2_results.json).
#   (sin flag)  Modo real, Cloud-2 (1x L40S sm_89 o 1x H100 sm_90): levanta cada motor
#               por separado, espera /health, lanza la carga y lo para antes del siguiente.
#
# Opciones: --engines "vllm sglang"  --concurrency "1 8 32 64"  --force (repite motor ya medido)
# Variables: MODEL PORT INPUT_LEN OUTPUT_LEN MAX_MODEL_LEN GPU_MEM_UTIL HEALTH_TIMEOUT
#            VLLM_BIN (def. "vllm")  SGLANG_PY (def. python3; usa otra venv si hace falta)
#            HARDWARE (texto libre para el informe)
# ==============================================================================
set -euo pipefail
export PYTHONUTF8=1   # Windows: stdout cp1252 no soporta ✓

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARTIFACTS_DIR="${ROOT_DIR}/artifacts"
LOG_DIR="${ARTIFACTS_DIR}/b2_2_logs"

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
PORT="${PORT:-8000}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"   # la primera carga descarga ~15 GB
VLLM_BIN="${VLLM_BIN:-vllm}"
SGLANG_PY="${SGLANG_PY:-python3}"
HARDWARE="${HARDWARE:-}"
ENGINES="vllm sglang"
CONCURRENCY="1 8 32 64"
DRY_RUN=0
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|--smoke-test) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        --engines)
            [[ $# -ge 2 ]] || { echo "[ERROR] falta valor para --engines" >&2; exit 2; }
            ENGINES="$2"; shift ;;
        --concurrency)
            [[ $# -ge 2 ]] || { echo "[ERROR] falta valor para --concurrency" >&2; exit 2; }
            CONCURRENCY="$2"; shift ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "[ERROR] opcion desconocida: $1" >&2; exit 2 ;;
    esac
    shift
done

# python3 real (en Windows el alias de la Store no cuenta)
PY=""
for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import sys; assert sys.version_info >= (3,9)" 2>/dev/null; then
        PY="$(command -v "$cand")"; break
    fi
done
if [[ -z "${PY}" ]]; then
    echo "[ERROR] se necesita python >= 3.9 en el PATH (ver AGENTS.md 3)." >&2
    exit 1
fi
CLIENT=("${PY}" "${SCRIPT_DIR}/bench_client.py")

# ------------------------------------------------------------------------------ dry-run
if [[ "${DRY_RUN}" == 1 ]]; then
    START=$(date +%s)
    echo "[dry-run] B2.2 - validacion local (sin GPU, sin servidores)"
    bash -n "${BASH_SOURCE[0]}" && echo "  ✓ sintaxis run_compare.sh"
    bash -n "${SCRIPT_DIR}/cloud_bootstrap.sh" && echo "  ✓ sintaxis cloud_bootstrap.sh"
    "${PY}" -I -m py_compile "${SCRIPT_DIR}/bench_client.py" && echo "  ✓ sintaxis bench_client.py"
    echo "  ✓ Python: $("${PY}" --version 2>&1)"
    if "${PY}" -c "import aiohttp" 2>/dev/null; then
        echo "  ✓ aiohttp: $("${PY}" -c 'import aiohttp;print(aiohttp.__version__)')"
    else
        echo "  ! aiohttp no instalado (necesario en Cloud-2: pip install aiohttp). El dry-run no lo usa."
    fi
    for t in vllm nvidia-smi; do
        command -v "$t" >/dev/null 2>&1 && echo "  ✓ ${t} presente" || echo "  - ${t} ausente (esperado en local; solo hace falta en Cloud-2)"
    done
    DRY_DIR="${ARTIFACTS_DIR}/b2_2_dryrun"
    "${CLIENT[@]}" dry --out-dir "${DRY_DIR}" --model "${MODEL}" --input-len "${INPUT_LEN}" \
        --output-len "${OUTPUT_LEN}" --concurrency ${CONCURRENCY} --hardware "SYNTHETIC (dry-run)"
    # Verificacion estructural del reporte generado
    "${PY}" -I -X utf8 - "${DRY_DIR}/b2_2_results.json" "${DRY_DIR}/b2_2_summary.md" <<'PYEOF'
import json, sys
res = json.load(open(sys.argv[1], encoding="utf-8"))
md = open(sys.argv[2], encoding="utf-8").read()
assert res["synthetic"] is True and "SINTETICOS" in md
for e in ("vllm", "sglang"):
    for B, lv in res["engines"][e].items():
        for k in ("ttft_ms", "tpot_ms", "itl_ms"):
            assert set(lv[k]) == {"p50", "p95", "p99"} and None not in lv[k].values(), (e, B, k)
        assert lv["output_tok_s"] > 0 and lv["total_tok_s"] > lv["output_tok_s"]
        assert "kv_cache_pct_max" in lv["metrics"] and "preemptions" in lv["metrics"]
print("  ✓ estructura de results.json / summary.md correcta")
PYEOF
    # Parser Prometheus: muestras reales del formato de vLLM y SGLang
    "${PY}" -I -X utf8 - "${SCRIPT_DIR}" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
import bench_client as b
vllm = '# HELP x\nvllm:kv_cache_usage_perc{engine="0"} 0.42\nvllm:num_preemptions_total{engine="0"} 3.0\n'
sgl = 'sglang:token_usage{tp_rank="0"} 0.31\nsglang:num_queue_reqs 2\n'
a, c = b.metrics_summary([b.parse_prom(vllm)]), b.metrics_summary([b.parse_prom(sgl)])
assert a["kv_cache_pct_max"] == 42.0 and a["preemptions"] == 0
assert c["kv_cache_pct_max"] == 31.0 and c["preemptions"] is None
assert abs(b.percentile([1, 2, 3, 4, 5], 50) - 3) < 1e-9
print("  ✓ parser /metrics (vLLM y SGLang) y percentiles")
PYEOF
    echo "[dry-run] OK en $(( $(date +%s) - START )) s -> ${DRY_DIR}/ (sinteticos, NO son resultados)"
    exit 0
fi

# ------------------------------------------------------------------------------ modo real
command -v nvidia-smi >/dev/null 2>&1 || { echo "[ERROR] sin nvidia-smi: el modo real es solo para Cloud-2 (usa --dry-run en local)." >&2; exit 1; }
"${PY}" -c "import aiohttp" 2>/dev/null || { echo "[ERROR] falta aiohttp (pip install aiohttp)." >&2; exit 1; }
VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if (( VRAM_MB < 40000 )); then
    echo "[ERROR] GPU con ${VRAM_MB} MiB: B2.2 requiere L40S (48 GB) o H100 (80 GB). La GTX 1050 no vale (AGENTS.md 2)." >&2
    exit 1
fi
[[ -z "${HARDWARE}" ]] && HARDWARE="$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1)"
mkdir -p "${ARTIFACTS_DIR}" "${LOG_DIR}"
if ! pgrep -f "shutdown" >/dev/null 2>&1 && [[ ! -e /run/systemd/shutdown/scheduled ]]; then
    echo "[WARN] no hay apagado programado. Ejecuta: sudo shutdown -h +90"
fi

SERVER_PID=""
stop_server() {
    [[ -z "${SERVER_PID}" ]] && return 0
    echo "[server] parando (pgid ${SERVER_PID}) ..."
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "${SERVER_PID}" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
    # esperar a que la VRAM se libere (evita OOM al arrancar el siguiente motor)
    for _ in $(seq 1 60); do
        used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
        (( used < 1500 )) && break
        sleep 2
    done
    echo "[server] VRAM en uso tras parar: ${used:-?} MiB"
}
trap stop_server EXIT INT TERM

wait_health() {
    local t0=$SECONDS
    until curl -fs "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
        kill -0 "${SERVER_PID}" 2>/dev/null || { echo "[ERROR] el servidor murio; ver ${LOG}" >&2; tail -n 30 "${LOG}" >&2; return 1; }
        (( SECONDS - t0 > HEALTH_TIMEOUT )) && { echo "[ERROR] timeout esperando /health" >&2; return 1; }
        sleep 3
    done
    echo "[server] /health OK tras $(( SECONDS - t0 )) s"
}

start_engine() {
    case "$1" in
        vllm)  # V1 es el motor por defecto en vLLM recientes; se fuerza FlashInfer
            VLLM_USE_V1=1 VLLM_ATTENTION_BACKEND=FLASHINFER setsid "${VLLM_BIN}" serve "${MODEL}" \
                --port "${PORT}" --dtype bfloat16 --max-model-len "${MAX_MODEL_LEN}" \
                --gpu-memory-utilization "${GPU_MEM_UTIL}" --disable-log-requests >"${LOG}" 2>&1 & ;;
        sglang)  # RadixAttention activo por defecto (no se pasa --disable-radix-cache)
            setsid "${SGLANG_PY}" -m sglang.launch_server --model-path "${MODEL}" \
                --port "${PORT}" --dtype bfloat16 --context-length "${MAX_MODEL_LEN}" \
                --mem-fraction-static "${GPU_MEM_UTIL}" --attention-backend flashinfer \
                --enable-metrics >"${LOG}" 2>&1 & ;;
        *) echo "[ERROR] motor desconocido: $1" >&2; return 2 ;;
    esac
    SERVER_PID=$!
}

for eng in ${ENGINES}; do
    RAW="${ARTIFACTS_DIR}/b2_2_raw_${eng}.json"
    if [[ -s "${RAW}" && "${FORCE}" != 1 ]]; then
        echo "[skip] ${eng}: ya existe ${RAW} (usa --force para repetir)"; continue
    fi
    LOG="${LOG_DIR}/${eng}_server.log"
    echo "=== ${eng} ==="
    start_engine "${eng}"
    wait_health
    "${CLIENT[@]}" run --engine "${eng}" --port "${PORT}" --model "${MODEL}" --input-len "${INPUT_LEN}" \
        --output-len "${OUTPUT_LEN}" --concurrency ${CONCURRENCY} --out-dir "${ARTIFACTS_DIR}"
    stop_server
done

"${CLIENT[@]}" report --out-dir "${ARTIFACTS_DIR}" --model "${MODEL}" --input-len "${INPUT_LEN}" \
    --output-len "${OUTPUT_LEN}" --concurrency ${CONCURRENCY} --hardware "${HARDWARE}"
echo "[OK] ${ARTIFACTS_DIR}/b2_2_results.json y b2_2_summary.md. Ahora: sudo shutdown -h now"
