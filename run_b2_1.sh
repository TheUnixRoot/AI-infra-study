#!/usr/bin/env bash
# ==============================================================================
# run_b2_1.sh - Entrypoint for Experiment B2.1: KV-Cache Arithmetic & VRAM Budget
# Module of Project A: agentic-inference-opt-lab
# Hardware Target: CPU Local (0 EUR, no GPU required)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ARTIFACTS_DIR="${SCRIPT_DIR}/artifacts"
mkdir -p "${ARTIFACTS_DIR}"

REPORT_JSON="${ARTIFACTS_DIR}/b2_1_kv_report.json"
REPORT_CSV="${ARTIFACTS_DIR}/b2_1_kv_report.csv"

echo "======================================================================"
echo " [B2.1] KV-Cache Arithmetic & VRAM Budgeting Experiment Runner"
echo " Workspace: ${SCRIPT_DIR}"
echo " Target HW: CPU Local (0 EUR, purely analytical)"
echo "======================================================================"

# ------------------------------------------------------------------------------
# 1. Dependency Verification (Idempotent)
# ------------------------------------------------------------------------------
echo "[1/4] Verifying dependencies..."

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 is required but not installed in PATH." >&2
    exit 1
fi
PYTHON_BIN="$(command -v python3)"
echo "  ✓ Python: ${PYTHON_BIN} ($(${PYTHON_BIN} --version))"

# Check huggingface_hub
if "${PYTHON_BIN}" -c "import huggingface_hub" >/dev/null 2>&1; then
    HF_VER=$("${PYTHON_BIN}" -c "import huggingface_hub; print(huggingface_hub.__version__)")
    echo "  ✓ huggingface_hub: installed (v${HF_VER})"
else
    echo "  ! huggingface_hub: not detected in python environment."
    echo "    (kv_calc.py provides verified embedded configs for offline execution in <5s)"
fi

# ------------------------------------------------------------------------------
# 2. Fast Smoke-Test Execution (<5s target, <30s requirement)
# ------------------------------------------------------------------------------
echo ""
echo "[2/4] Running fast smoke-test (<30s target)..."
START_SMOKE=$(date +%s)
"${PYTHON_BIN}" "${SCRIPT_DIR}/kv_calc.py" --smoke-test --seq-len 8192 >/dev/null
END_SMOKE=$(date +%s)
SMOKE_DURATION=$((END_SMOKE - START_SMOKE))
echo "  ✓ Smoke-test completed cleanly in ${SMOKE_DURATION}s."

# ------------------------------------------------------------------------------
# 3. Full Benchmark Matrix Execution
# ------------------------------------------------------------------------------
echo ""
echo "[3/4] Running full KV-Cache & VRAM budgeting matrix across models..."
echo "  Models: meta-llama/Meta-Llama-3-8B, Qwen/Qwen2.5-72B-Instruct, deepseek-ai/DeepSeek-V3"
echo "  Sequence Length: 8192 | GPU Util: 0.90 | Overhead: 2.0 GB"
echo ""

"${PYTHON_BIN}" "${SCRIPT_DIR}/kv_calc.py" \
    --models meta-llama/Meta-Llama-3-8B Qwen/Qwen2.5-72B-Instruct deepseek-ai/DeepSeek-V3 \
    --seq-len 8192 \
    --gpu-mem-util 0.90 \
    --overhead-gb 2.0 \
    --output-json "${REPORT_JSON}" \
    --output-csv "${REPORT_CSV}"

# ------------------------------------------------------------------------------
# 4. Output Summary & Verification
# ------------------------------------------------------------------------------
echo ""
echo "[4/4] Artifacts successfully generated:"
echo "  ✓ JSON Report: ${REPORT_JSON}"
echo "  ✓ CSV Report:  ${REPORT_CSV}"
echo "======================================================================"
