#!/usr/bin/env bash
# ==============================================================================
# termux_energy_bench.sh - Companion script for Android (Termux, no root)
# Measures real energy (mWh/token), RSS RAM, pp512 and tg128 on mobile CPU NEON
# and Vulkan using /sys/class/power_supply/battery/{current_now,voltage_now}.
#
# Prerequisites inside Termux:
#   pkg update && pkg install -y git cmake ninja clang vulkan-headers vulkan-loader-android
#   # Build upstream llama.cpp (b11556+):
#   git clone --depth 1 https://github.com/ggml-org/llama.cpp llama-upstream
#   cmake -S llama-upstream -B build-upstream -G Ninja -DCMAKE_BUILD_TYPE=Release \
#     -DGGML_VULKAN=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_TESTS=OFF
#   cmake --build build-upstream --target llama-bench llama-perplexity -j$(nproc)
#   # Build PrismML fork (PQ2_0, PTQ1_0 + Hadamard):
#   git clone --depth 1 -b prism https://github.com/PrismML-Eng/llama.cpp llama-prism
#   cmake -S llama-prism -B build-prism -G Ninja -DCMAKE_BUILD_TYPE=Release \
#     -DGGML_VULKAN=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_TESTS=OFF
#   cmake --build build-prism --target llama-bench llama-perplexity -j$(nproc)
#
# Usage:
#   MODEL=~/models/Ternary-Bonsai-2-2B-PTQ1_0.gguf BIN=~/build-prism/bin/llama-bench \
#     NGL=0 ./termux_energy_bench.sh
# ==============================================================================
set -euo pipefail

: "${MODEL:?Define MODEL=path/to/model.gguf}"
BIN="${BIN:-llama-bench}"
NGL="${NGL:-0}"         # 0 = CPU NEON, 99 = Vulkan GPU offload
THREADS="${THREADS:-4}"
REPS="${REPS:-3}"
PROMPT_N="${PROMPT_N:-512}"
GEN_N="${GEN_N:-128}"
SAMPLE_MS="${SAMPLE_MS:-100}"

BAT_DIR="${BAT_DIR:-/sys/class/power_supply/battery}"
CUR_FILE="${BAT_DIR}/current_now"
VOL_FILE="${BAT_DIR}/voltage_now"

if [[ ! -r "${CUR_FILE}" || ! -r "${VOL_FILE}" ]]; then
    echo "[ERROR] Cannot read ${CUR_FILE} or ${VOL_FILE} on this device." >&2
    echo "        Try setting BAT_DIR=/sys/class/power_supply/<battery_node>." >&2
    exit 1
fi

command -v "${BIN}" >/dev/null 2>&1 || { echo "[ERROR] Binary not found: ${BIN}" >&2; exit 1; }

POWER_LOG="$(mktemp)"
TIME_LOG="$(mktemp)"
BENCH_JSON="$(mktemp)"
SAMPLER_PID=""
cleanup() {
    [[ -n "${SAMPLER_PID}" ]] && kill "${SAMPLER_PID}" 2>/dev/null || true
    rm -f "${POWER_LOG}" "${TIME_LOG}" "${BENCH_JSON}"
}
trap cleanup EXIT INT TERM

echo "[info] Unplug USB charger for accurate battery discharge current!"
echo "[info] Sampling ${BAT_DIR} every ${SAMPLE_MS}ms while running ${BIN} ..."

python3 - "${CUR_FILE}" "${VOL_FILE}" "${SAMPLE_MS}" "${POWER_LOG}" <<'PYEOF' &
import sys, time
cur_f, vol_f, ms, out_f = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
dt = ms / 1000.0
with open(out_f, "w") as out:
    while True:
        try:
            t = time.monotonic()
            with open(cur_f) as fc, open(vol_f) as fv:
                c = abs(int(fc.read().strip()))
                v = abs(int(fv.read().strip()))
            out.write(f"{t:.6f} {c} {v}\n")
            out.flush()
        except Exception:
            pass
        time.sleep(dt)
PYEOF
SAMPLER_PID=$!

T0=$(date +%s.%N)
if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -v -o "${TIME_LOG}" \
        "${BIN}" -m "${MODEL}" -ngl "${NGL}" -t "${THREADS}" -p "${PROMPT_N}" -n "${GEN_N}" -r "${REPS}" -o json \
        > "${BENCH_JSON}"
else
    "${BIN}" -m "${MODEL}" -ngl "${NGL}" -t "${THREADS}" -p "${PROMPT_N}" -n "${GEN_N}" -r "${REPS}" -o json \
        > "${BENCH_JSON}"
fi
T1=$(date +%s.%N)

kill "${SAMPLER_PID}" 2>/dev/null || true
wait "${SAMPLER_PID}" 2>/dev/null || true
SAMPLER_PID=""

python3 - "${POWER_LOG}" "${BENCH_JSON}" "${TIME_LOG}" "${T0}" "${T1}" "${PROMPT_N}" "${GEN_N}" "${REPS}" <<'PYEOF'
import json, re, sys

power_log, bench_json, time_log, t0, t1, pn, gn, reps = sys.argv[1:]
elapsed_s = float(t1) - float(t0)

samples_w = []
with open(power_log) as f:
    for line in f:
        parts = line.split()
        if len(parts) != 3:
            continue
        c_raw, v_raw = float(parts[1]), float(parts[2])
        # Linux sysfs power_supply units are microamps (uA) and microvolts (uV)
        # Heuristic: if voltage > 100000, it's uV; if > 1000, it's mV
        v_v = v_raw / 1e6 if v_raw > 100_000 else (v_raw / 1e3 if v_raw > 100 else v_raw)
        c_a = c_raw / 1e6 if c_raw > 10_000 else (c_raw / 1e3 if c_raw > 50 else c_raw)
        samples_w.append(v_v * c_a)

avg_w = sum(samples_w) / len(samples_w) if samples_w else 0.0
total_mwh = avg_w * (elapsed_s / 3600.0) * 1000.0
total_tokens = (int(pn) + int(gn)) * int(reps)
mwh_per_token = total_mwh / total_tokens if total_tokens > 0 else 0.0

rows = json.load(open(bench_json))
pp = next((r for r in rows if r.get("n_prompt", 0) > 0), {})
tg = next((r for r in rows if r.get("n_gen", 0) > 0), {})

rss_mb = None
try:
    txt = open(time_log).read()
    m = re.search(r"Maximum resident set size.*?:\s*(\d+)", txt)
    if m:
        rss_mb = int(m.group(1)) / 1024.0
except FileNotFoundError:
    pass

print(f"Elapsed:       {elapsed_s:.2f} s ({len(samples_w)} power samples)")
print(f"Avg Power:     {avg_w:.2f} W")
print(f"Total Energy:  {total_mwh:.3f} mWh (tokens={total_tokens})")
print(f"Energy/Token:  {mwh_per_token:.5f} mWh/token")
if rss_mb is not None:
    print(f"Peak RSS:      {rss_mb:.1f} MiB")
if pp:
    print(f"pp{pn}:         {pp.get('avg_ts', 0):.2f} +- {pp.get('stddev_ts', 0):.2f} tok/s")
if tg:
    print(f"tg{gn}:         {tg.get('avg_ts', 0):.2f} +- {tg.get('stddev_ts', 0):.2f} tok/s")
PYEOF
