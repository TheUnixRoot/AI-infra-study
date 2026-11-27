#!/usr/bin/env bash
# ==============================================================================
# run.sh - B3.1: Qwen3.5-2B vs Ternary-Bonsai-2-2B (Hadamard + llama.cpp Docker)
#
# Ejecuta compilacion y benchmarks SIEMPRE dentro de Docker (ai-infra-study:llama-cuda12.6)
# montando este repo de D: en /work. Las descargas de modelos (--download) pueden correr
# directo en host Python hacia D: (offline) o dentro del contenedor si falta huggingface_hub.
#
# Opciones:
#   --download        Descarga los 7 GGUF + wikitext-2 en b3_1_hadamard_bonsai/{models,data}
#   --build           Clona y compila ggml-org/llama.cpp + PrismML-Eng/llama.cpp en Docker
#   --smoke-test      Validacion rapida (<30 s) en Docker GPU
#   --full            Benchmark completo (llama-bench + llama-perplexity PPL/KLD + RSS/VRAM)
#   --all             --download + --build + --full
#   --clean           Borra .build/ (no borra models/)
#   -h, --help        Muestra esta ayuda
#
# Variables: CUDA_ARCH [61]  PPL_CHUNKS [8]  PPL_CTX [512]  BENCH_REPS [3]  THREADS [4]
#            DOCKER_IMAGE [ai-infra-study:llama-cuda12.6]  FORCE_REBUILD [0]
# ==============================================================================
set -euo pipefail

export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_IN_CONTAINER="/work/b3_1_hadamard_bonsai"
DOCKER_IMAGE="${DOCKER_IMAGE:-ai-infra-study:llama-cuda12.6}"
BASE_IMAGE="${BASE_IMAGE:-ai-infra-study:cuda12.6}"
CUDA_ARCH="${CUDA_ARCH:-61}"

DO_DOWNLOAD=0
DO_BUILD=0
DO_SMOKE=0
DO_FULL=0
DO_CLEAN=0
IN_DOCKER=0
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --download|--download-only) DO_DOWNLOAD=1 ;;
        --build|--build-only) DO_BUILD=1 ;;
        --smoke|--smoke-test) DO_SMOKE=1 ;;
        --full) DO_FULL=1 ;;
        --all) DO_DOWNLOAD=1; DO_BUILD=1; DO_FULL=1 ;;
        --clean) DO_CLEAN=1 ;;
        --in-docker) IN_DOCKER=1 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "[ERROR] Opcion desconocida: $1" >&2; exit 2 ;;
    esac
    [[ "$1" != "--in-docker" ]] && FORWARD_ARGS+=("$1")
    shift
done

if (( DO_DOWNLOAD + DO_BUILD + DO_SMOKE + DO_FULL + DO_CLEAN == 0 )); then
    DO_SMOKE=1
    FORWARD_ARGS=(--smoke-test)
fi

if [[ -f /.dockerenv ]]; then
    IN_DOCKER=1
fi

host_path() {
    local p="$1"
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -w "$p"
    else
        printf '%s\n' "$p"
    fi
}

ensure_docker_image() {
    command -v docker >/dev/null 2>&1 || { echo "[ERROR] docker no esta en el PATH." >&2; exit 1; }
    if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
        echo "[docker] Construyendo imagen base ${BASE_IMAGE} (INSTALL_TORCH=0)..."
        docker build -t "${BASE_IMAGE}" --build-arg INSTALL_TORCH=0 "${ROOT_DIR}"
    fi
    if [[ "${FORCE_REBUILD:-0}" == "1" ]] || ! docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
        echo "[docker] Construyendo imagen ${DOCKER_IMAGE}..."
        docker build -t "${DOCKER_IMAGE}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"
    fi
}

# ------------------------------------------------------------------------------
# Descarga directa en D: (host si hay python+huggingface_hub, sino via Docker -> D:)
# ------------------------------------------------------------------------------
if (( IN_DOCKER == 0 && DO_DOWNLOAD == 1 )); then
    HOST_PY=""
    for cand in "${ROOT_DIR}/.venv/Scripts/python.exe" "${ROOT_DIR}/.venv-linux/bin/python" python3 python; do
        if [[ -x "$cand" ]] || command -v "$cand" >/dev/null 2>&1; then
            if "$cand" -c "import huggingface_hub" >/dev/null 2>&1; then
                HOST_PY="$cand"
                break
            fi
        fi
    done
    if [[ -n "${HOST_PY}" ]]; then
        echo "[download] Descargando modelos en D: via ${HOST_PY} (offline, sin Docker)..."
        PYTHONUNBUFFERED=1 "${HOST_PY}" "${SCRIPT_DIR}/download_models.py" --only all
        # Si solo se pidio --download, terminamos aqui sin levantar Docker
        if (( DO_BUILD + DO_SMOKE + DO_FULL + DO_CLEAN == 0 )); then
            exit 0
        fi
        # Reconstruir FORWARD_ARGS sin --download para la fase Docker
        NEW_ARGS=()
        for a in "${FORWARD_ARGS[@]}"; do
            [[ "$a" == "--download" || "$a" == "--download-only" || "$a" == "--all" ]] && continue
            NEW_ARGS+=("$a")
        done
        (( DO_BUILD == 1 && DO_FULL == 1 )) && NEW_ARGS=(--build --full)
        FORWARD_ARGS=("${NEW_ARGS[@]}")
    fi
fi

# ------------------------------------------------------------------------------
# Re-entrada dentro de Docker (compilacion y benchmarks 100% en contenedor)
# ------------------------------------------------------------------------------
if (( IN_DOCKER == 0 )); then
    ensure_docker_image
    HOST_ROOT="$(host_path "${ROOT_DIR}")"
    GPU_ARGS=()
    if (( DO_SMOKE == 1 || DO_FULL == 1 )); then
        GPU_ARGS=(--gpus all)
    fi
    echo "[docker] Ejecutando en ${DOCKER_IMAGE} (mount: ${HOST_ROOT} -> /work)..."
    exec docker run --rm "${GPU_ARGS[@]}" \
        -v "${HOST_ROOT}:/work" \
        -e "CUDA_ARCH=${CUDA_ARCH}" \
        -e "PPL_CHUNKS=${PPL_CHUNKS:-8}" \
        -e "PPL_CTX=${PPL_CTX:-512}" \
        -e "BENCH_REPS=${BENCH_REPS:-3}" \
        -e "THREADS=${THREADS:-4}" \
        -e "UPSTREAM_REF=${UPSTREAM_REF:-master}" \
        -e "PRISM_REF=${PRISM_REF:-prism}" \
        "${DOCKER_IMAGE}" \
        bash "${WORK_IN_CONTAINER}/run.sh" --in-docker "${FORWARD_ARGS[@]}"
fi

# ==============================================================================
# A partir de aqui estamos DENTRO de Docker (/work/b3_1_hadamard_bonsai)
# ==============================================================================
cd "${SCRIPT_DIR}"
BUILD_ROOT="${SCRIPT_DIR}/.build"
SRC_UPSTREAM="${BUILD_ROOT}/src-upstream"
SRC_PRISM="${BUILD_ROOT}/src-prism"
BIN_UPSTREAM="${BUILD_ROOT}/bin-upstream"
BIN_PRISM="${BUILD_ROOT}/bin-prism"
ARTIFACTS_DIR="${ROOT_DIR}/artifacts"
mkdir -p "${BUILD_ROOT}" "${ARTIFACTS_DIR}"

if (( DO_CLEAN == 1 )); then
    echo "[clean] Eliminando ${BUILD_ROOT} ..."
    rm -rf "${BUILD_ROOT}"
    echo "[clean] OK (models/ y data/ conservados)."
fi

if (( DO_DOWNLOAD == 1 )); then
    echo "[download] Ejecutando download_models.py dentro de Docker hacia D:..."
    PYTHONUNBUFFERED=1 python3 "${SCRIPT_DIR}/download_models.py" --only all
fi

build_variant() {
    local name="$1"
    local repo_url="$2"
    local ref="$3"
    local out_bin_dir="$4"
    local tmp_src="/tmp/llama-src-${name}"
    local tmp_bld="/tmp/llama-bld-${name}"

    local targets=(llama-bench llama-perplexity llama-quantize llama-cli llama-server)
    local ready=1
    if [[ ! -x "${out_bin_dir}/llama-bench" || ! -x "${out_bin_dir}/llama-perplexity" ]]; then
        ready=0
    fi
    if [[ "${name}" == "prism" ]]; then
        targets+=(llama-kv-mean-center)
        if [[ ! -x "${out_bin_dir}/llama-kv-mean-center" ]]; then
            ready=0
        fi
    fi

    if (( ready == 1 )) && [[ "${FORCE_REBUILD:-0}" != "1" ]]; then
        echo "[skip] ${name}: binarios ya compilados en ${out_bin_dir}"
        return 0
    fi

    rm -rf "${tmp_src}" "${tmp_bld}"
    echo "=== [${name}] Clonando ${repo_url} (${ref}) en /tmp (ext4 rapido) ==="
    git clone --depth 1 --branch "${ref}" "${repo_url}" "${tmp_src}"
    local rev
    rev="$(git -C "${tmp_src}" rev-parse --short HEAD)"
    echo "[${name}] Commit: ${rev}"

    echo "=== [${name}] Compilando CUDA (sm_${CUDA_ARCH}) con CMake + Ninja ==="
    cmake -S "${tmp_src}" -B "${tmp_bld}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_UI=OFF \
        -DCMAKE_BUILD_RPATH='$ORIGIN' \
        -DCMAKE_INSTALL_RPATH='$ORIGIN' \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs"

    cmake --build "${tmp_bld}" --target "${targets[@]}" -j"$(nproc)"

    rm -rf "${out_bin_dir}"
    mkdir -p "${out_bin_dir}"
    cp -a "${tmp_bld}/bin/"* "${out_bin_dir}/"
    find "${tmp_bld}" -maxdepth 4 -name "*.so*" -exec cp -P -n {} "${out_bin_dir}/" \; 2>/dev/null || true
    printf '%s\n' "${rev}" > "${out_bin_dir}/GIT_REV"
    rm -rf "${tmp_src}" "${tmp_bld}"
    echo "[${name}] Binarios listos en ${out_bin_dir} (rev ${rev})"
}

if (( DO_BUILD == 1 )); then
    rm -rf "${BUILD_ROOT}/src-upstream" "${BUILD_ROOT}/src-prism" "${BUILD_ROOT}/cmake-upstream" "${BUILD_ROOT}/cmake-prism"
    build_variant "upstream" "https://github.com/ggml-org/llama.cpp.git" "${UPSTREAM_REF:-master}" "${BIN_UPSTREAM}"
    build_variant "prism" "https://github.com/PrismML-Eng/llama.cpp.git" "${PRISM_REF:-prism}" "${BIN_PRISM}"
fi

if (( DO_SMOKE == 1 )); then
    echo "=== [smoke-test] Validando entorno Docker CUDA + binarios llama.cpp ==="
    nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
    for d in "${BIN_UPSTREAM}" "${BIN_PRISM}"; do
        [[ -x "${d}/llama-bench" && -x "${d}/llama-perplexity" ]] || {
            echo "[ERROR] Faltan binarios en ${d}. Ejecuta primero: ./run.sh --build" >&2
            exit 1
        }
    done
    echo "  upstream llama-bench: $(LD_LIBRARY_PATH="${BIN_UPSTREAM}" "${BIN_UPSTREAM}/llama-bench" --version 2>&1 | head -n1)"
    echo "  prism    llama-bench: $(LD_LIBRARY_PATH="${BIN_PRISM}" "${BIN_PRISM}/llama-bench" --version 2>&1 | head -n1)"

    SMOKE_MODEL="${SCRIPT_DIR}/models/Ternary-Bonsai-2-2B-PTQ1_0.gguf"
    SMOKE_DIR="${ARTIFACTS_DIR}/b3_1_smoke"
    mkdir -p "${SMOKE_DIR}"
    if [[ -f "${SMOKE_MODEL}" ]]; then
        echo "  Ejecutando micro llama-bench GPU (ngl=99, p=32, n=16) con PrismML PTQ1_0 (Hadamard)..."
        LD_LIBRARY_PATH="${BIN_PRISM}" "${BIN_PRISM}/llama-bench" \
            -m "${SMOKE_MODEL}" -ngl 99 -fa 1 -p 32 -n 16 -r 1 -o json \
            > "${SMOKE_DIR}/smoke_prism_ptq1.json"
        python3 - "${SMOKE_DIR}/smoke_prism_ptq1.json" <<'PYEOF'
import json, sys
rows = json.load(open(sys.argv[1]))
assert len(rows) >= 2, rows
for r in rows:
    kind = f"pp{r['n_prompt']}" if r.get("n_prompt") else f"tg{r['n_gen']}"
    print(f"    {kind}: {r['avg_ts']:.2f} tok/s (model={r.get('model_type')})")
PYEOF
    else
        echo "  [WARN] ${SMOKE_MODEL} aun no existe; ejecuta ./run.sh --download para probar inferencia GPU."
    fi
    echo "[smoke-test] OK"
fi

if (( DO_FULL == 1 )); then
    echo "=== [full] Ejecutando suite completa en Docker ==="
    python3 "${SCRIPT_DIR}/bench_suite.py" \
        --base-dir "${SCRIPT_DIR}" \
        --artifacts-dir "${ARTIFACTS_DIR}" \
        --ppl-chunks "${PPL_CHUNKS:-8}" \
        --ppl-ctx "${PPL_CTX:-512}" \
        --reps "${BENCH_REPS:-3}" \
        --threads "${THREADS:-4}"
fi
