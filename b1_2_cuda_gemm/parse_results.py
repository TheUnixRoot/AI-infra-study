#!/usr/bin/env python3
"""Parse the markdown tables printed by gemm_bench / warp_online_softmax into real
CSV + JSON artifacts (no placeholders: every number comes from the binaries' output)."""
import csv, json, re, sys, datetime

def rows(path):
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("| **"):
            yield [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]

def num(s):
    return float(re.sub(r"[^0-9.eE+-]", "", s.replace("x", "")))

def main(gemm_out, softmax_out, mode, gpu, driver, json_path, csv_path):
    gemm = []
    for c in rows(gemm_out):
        m = re.search(r"err:\s*([0-9.eE+-]+)", c[5])
        gemm.append({"kernel": c[0], "dimension": c[1].replace(" ", ""), "latency_ms": num(c[2]),
                     "gflops": num(c[3]), "speedup_vs_naive": num(c[4]),
                     "verified": "PASS" in c[5], "max_abs_error": float(m.group(1)) if m else None})
    soft = [{"kernel": c[0], "dimension": c[1], "latency_ms": num(c[2]), "effective_bw_gb_s": num(c[3]),
             "speedup_vs_3pass": num(c[4]), "max_abs_error": num(c[5]), "verified": "PASS" in c[6]}
            for c in rows(softmax_out)]
    if not gemm or not soft:
        sys.exit("[ERROR] No se pudieron parsear resultados de los binarios.")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "kernel", "dimension", "latency_ms", "throughput", "throughput_unit", "speedup", "verified"])
        for r in gemm:
            w.writerow(["gemm", r["kernel"], r["dimension"], r["latency_ms"], r["gflops"], "GFLOP/s", r["speedup_vs_naive"], r["verified"]])
        for r in soft:
            w.writerow(["softmax", r["kernel"], r["dimension"], r["latency_ms"], r["effective_bw_gb_s"], "GB/s", r["speedup_vs_3pass"], r["verified"]])
    with open(json_path, "w") as f:
        json.dump({"experiment": "B1.2 - CUDA C++ GEMM & Warp Online Softmax",
                   "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "target_hardware": {"gpu": gpu, "architecture": "Pascal (sm_61)", "driver_version": driver},
                   "execution_mode": mode, "gemm": gemm, "softmax": soft,
                   "verification_status": "ALL_TESTS_PASSED" if all(r["verified"] for r in gemm + soft) else "FAILED"},
                  f, indent=2)

if __name__ == "__main__":
    main(*sys.argv[1:8])
