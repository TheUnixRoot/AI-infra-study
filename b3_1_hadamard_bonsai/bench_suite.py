#!/usr/bin/env python3
"""B3.1 Benchmark Suite (inside Docker): Qwen3.5-2B vs Ternary-Bonsai-2-2B.

Measures four real quantities across upstream llama.cpp (b11556+) and
PrismML-Eng/llama.cpp (Hadamard + PQ2_0 / PTQ1_0):
  1. Perplexity (PPL) and KL-Divergence (KLD vs Qwen3.5-2B-F16) via llama-perplexity
  2. Speed (pp512 and tg128) via llama-bench
  3. Real Peak Memory (Host RSS via /usr/bin/time -v and GPU VRAM via nvidia-smi)
  4. Energy (mWh/token): null on GTX 1050 Docker unless board power is exposed;
     Android Termux energy script is provided separately.

All artifact outputs avoid local absolute filesystem paths (AGENTS.md 6.2).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchTarget:
    key: str
    family: str
    quant: str
    filename: str
    engines: tuple[str, ...]
    ngl_bench: int
    ngl_ppl: int
    ppl_batch: int = 128
    is_baseline: bool = False


TARGETS: tuple[BenchTarget, ...] = (
    BenchTarget(
        key="qwen3.5-2b-f16",
        family="Qwen3.5-2B",
        quant="F16",
        filename="Qwen3.5-2B-F16.gguf",
        engines=("upstream", "prism"),
        ngl_bench=0,  # 3.6 GiB > 2 GB GTX 1050 VRAM
        ngl_ppl=0,
        ppl_batch=128,
        is_baseline=True,
    ),
    BenchTarget(
        key="qwen3.5-2b-q4_k_m",
        family="Qwen3.5-2B",
        quant="Q4_K_M",
        filename="Qwen3.5-2B-Q4_K_M.gguf",
        engines=("upstream", "prism"),
        ngl_bench=99,
        ngl_ppl=99,
        ppl_batch=128,
    ),
    BenchTarget(
        key="qwen3.5-2b-iq2_xxs",
        family="Qwen3.5-2B",
        quant="UD-IQ2_XXS",
        filename="Qwen3.5-2B-UD-IQ2_XXS.gguf",
        engines=("upstream", "prism"),
        ngl_bench=99,
        ngl_ppl=99,
        ppl_batch=128,
    ),
    BenchTarget(
        key="qwen3.5-2b-tq1_0",
        family="Qwen3.5-2B",
        quant="TQ1_0",
        filename="Qwen3.5-2B-TQ1_0.gguf",
        engines=("upstream", "prism"),
        ngl_bench=99,
        ngl_ppl=99,
        ppl_batch=128,
    ),
    BenchTarget(
        key="bonsai2-2b-q2_0",
        family="Ternary-Bonsai-2-2B",
        quant="Q2_0",
        filename="Ternary-Bonsai-2-2B-Q2_0.gguf",
        engines=("upstream", "prism"),
        ngl_bench=99,
        ngl_ppl=99,
        ppl_batch=128,
    ),
    BenchTarget(
        key="bonsai2-2b-pq2_0",
        family="Ternary-Bonsai-2-2B",
        quant="PQ2_0",
        filename="Ternary-Bonsai-2-2B-PQ2_0.gguf",
        engines=("prism",),
        ngl_bench=99,
        ngl_ppl=99,
        ppl_batch=128,
    ),
    BenchTarget(
        key="bonsai2-2b-ptq1_0",
        family="Ternary-Bonsai-2-2B",
        quant="PTQ1_0",
        filename="Ternary-Bonsai-2-2B-PTQ1_0.gguf",
        engines=("prism",),
        ngl_bench=99,
        ngl_ppl=99,
        # PTQ1_0 MMQ requires Turing+ (mmq.cu: turing_mma_available); keep ubatch <= 8
        # on Pascal sm_61 so the tied 248k-vocab head stays on mmvq instead of F16 cuBLAS OOM.
        ppl_batch=8,
    ),
)


def query_gpu_info() -> dict[str, object]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip().splitlines()[0]
    parts = [p.strip() for p in out.split(",")]
    return {
        "name": parts[0],
        "vram_total": parts[1],
        "compute_cap": parts[2],
        "driver": parts[3],
    }


def query_vram_used_mib() -> float | None:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


def run_monitored(
    cmd: list[str],
    env: dict[str, str],
    time_log: Path,
    raw_log: Path | None = None,
) -> tuple[int, str, str, float | None, float | None, float | None]:
    """Run command wrapped in /usr/bin/time -v while sampling GPU VRAM used."""
    idle_vram = query_vram_used_mib()
    stop_event = threading.Event()
    vram_samples: list[float] = []
    if idle_vram is not None:
        vram_samples.append(idle_vram)

    def poll_vram() -> None:
        while not stop_event.is_set():
            v = query_vram_used_mib()
            if v is not None:
                vram_samples.append(v)
            stop_event.wait(0.15)

    t = threading.Thread(target=poll_vram, daemon=True)
    t.start()
    wrapped = ["/usr/bin/time", "-v", "-o", str(time_log), *cmd]
    proc = subprocess.run(wrapped, env=env, capture_output=True, text=True)
    stop_event.set()
    t.join(timeout=2)

    if raw_log is not None:
        raw_log.write_text(
            f"CMD: {' '.join(cmd)}\nRC: {proc.returncode}\n\n--- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}\n",
            encoding="utf-8",
            errors="replace",
        )

    rss_mib: float | None = None
    if time_log.is_file():
        txt = time_log.read_text(errors="replace")
        m = re.search(r"Maximum resident set size.*?:\s*(\d+)", txt)
        if m:
            rss_mib = round(int(m.group(1)) / 1024.0, 1)

    peak_vram = round(max(vram_samples), 1) if vram_samples else None
    delta_vram = round(max(0.0, peak_vram - idle_vram), 1) if (peak_vram is not None and idle_vram is not None) else None
    return proc.returncode, proc.stdout, proc.stderr, rss_mib, peak_vram, delta_vram


def run_llama_bench(
    bin_dir: Path,
    model_path: Path,
    ngl: int,
    reps: int,
    threads: int,
    tmp_dir: Path,
    raw_log: Path,
) -> dict[str, object]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    cmd = [
        str(bin_dir / "llama-bench"),
        "-m", str(model_path),
        "-lm", "none",
        "-ngl", str(ngl),
        "-fa", "1",
        "-t", str(threads),
        "-p", "512",
        "-n", "128",
        "-r", str(reps),
        "-o", "json",
    ]
    rc, stdout, stderr, rss_mib, peak_vram_mib, delta_vram_mib = run_monitored(
        cmd, env, tmp_dir / "bench_time.txt", raw_log=raw_log
    )
    if rc != 0:
        raise RuntimeError(f"llama-bench failed (rc={rc}): {stderr[-500:]}")
    rows = json.loads(stdout)
    pp = next((r for r in rows if r.get("n_prompt", 0) == 512), {})
    tg = next((r for r in rows if r.get("n_gen", 0) == 128), {})
    return {
        "pp512_tok_s": round(pp.get("avg_ts", 0.0), 2) if pp else None,
        "pp512_std": round(pp.get("stddev_ts", 0.0), 2) if pp else None,
        "tg128_tok_s": round(tg.get("avg_ts", 0.0), 2) if tg else None,
        "tg128_std": round(tg.get("stddev_ts", 0.0), 2) if tg else None,
        "peak_rss_mib": rss_mib,
        "peak_vram_mib": peak_vram_mib,
        "delta_vram_mib": delta_vram_mib,
    }


def parse_ppl_output(text: str) -> dict[str, float | None]:
    def find_float(pattern: str) -> float | None:
        m = re.search(pattern, text)
        if not m:
            return None
        val = float(m.group(1))
        return round(val, 6)

    ppl = find_float(r"Mean PPL\(Q\)\s*:\s*([0-9]+(?:\.[0-9]+)?)")
    if ppl is None:
        ppl = find_float(r"Final estimate:\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)")
    ppl_std = find_float(r"Mean PPL\(Q\)\s*:\s*[0-9.]+\s*\+/-\s*([0-9]+(?:\.[0-9]+)?)")
    if ppl_std is None:
        ppl_std = find_float(r"Mean PPL\(Q\)\s*:\s*[0-9.]+\s*±\s*([0-9]+(?:\.[0-9]+)?)")
    if ppl_std is None:
        ppl_std = find_float(r"Final estimate:\s*PPL\s*=\s*[0-9.]+\s*\+/-\s*([0-9]+(?:\.[0-9]+)?)")

    ppl_base = find_float(r"Mean PPL\(base\)\s*:\s*([0-9]+(?:\.[0-9]+)?)")
    kld_mean = find_float(r"Mean\s+KLD:\s*([0-9]+(?:\.[0-9]+)?)")
    kld_median = find_float(r"Median\s+KLD:\s*([0-9]+(?:\.[0-9]+)?)")
    kld_p99 = find_float(r"99\.0%\s+KLD:\s*([0-9]+(?:\.[0-9]+)?)")
    same_top_pct = find_float(r"Same top p\s*:\s*([0-9]+(?:\.[0-9]+)?)")
    return {
        "ppl": ppl,
        "ppl_unc": ppl_std,
        "ppl_base": ppl_base,
        "kld_mean": kld_mean,
        "kld_median": kld_median,
        "kld_p99": kld_p99,
        "same_top_pct": same_top_pct,
    }


def run_perplexity(
    bin_dir: Path,
    model_path: Path,
    corpus_path: Path,
    ngl: int,
    ctx: int,
    batch: int,
    chunks: int,
    threads: int,
    tmp_dir: Path,
    raw_log: Path,
    save_logits: Path | None = None,
    kld_base: Path | None = None,
) -> dict[str, float | None]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    cmd = [
        str(bin_dir / "llama-perplexity"),
        "-m", str(model_path),
        "-lm", "none",
        "-f", str(corpus_path),
        "-ngl", str(ngl),
        "-fa", "on",
        "-c", str(ctx),
        "-b", str(batch),
        "-ub", str(batch),
        "--chunks", str(chunks),
        "-t", str(threads),
    ]
    if save_logits is not None:
        cmd.extend(["--save-all-logits", str(save_logits)])
    elif kld_base is not None:
        cmd.extend(["--kl-divergence", "--kl-divergence-base", str(kld_base)])

    rc, stdout, stderr, rss_mib, peak_vram_mib, delta_vram_mib = run_monitored(
        cmd, env, tmp_dir / "ppl_time.txt", raw_log=raw_log
    )
    combined = stdout + "\n" + stderr
    if rc != 0:
        raise RuntimeError(f"llama-perplexity failed (rc={rc}): {combined[-800:]}")
    metrics = parse_ppl_output(combined)
    metrics["ppl_peak_rss_mib"] = rss_mib
    metrics["ppl_peak_vram_mib"] = peak_vram_mib
    metrics["ppl_delta_vram_mib"] = delta_vram_mib
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run B3.1 Docker benchmark suite")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--ppl-chunks", type=int, default=8)
    parser.add_argument("--ppl-ctx", type=int, default=512)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    models_dir = args.base_dir / "models"
    corpus_path = args.base_dir / "data" / "wikitext-2-raw" / "wiki.test.raw"
    logits_dir = args.base_dir / "logits"
    tmp_dir = args.base_dir / ".build" / "tmp"
    logs_dir = args.base_dir / ".build" / "logs"
    logits_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.is_file():
        raise FileNotFoundError(f"Missing corpus {corpus_path}. Run ./run.sh --download first.")

    bins = {
        "upstream": args.base_dir / ".build" / "bin-upstream",
        "prism": args.base_dir / ".build" / "bin-prism",
    }
    revs = {}
    for eng, bdir in bins.items():
        rev_file = bdir / "GIT_REV"
        revs[eng] = rev_file.read_text().strip() if rev_file.is_file() else "unknown"

    gpu = query_gpu_info()
    logits_path = logits_dir / f"qwen35_2b_f16_ctx{args.ppl_ctx}_chunks{args.ppl_chunks}.logits"

    baseline_model = models_dir / "Qwen3.5-2B-F16.gguf"
    if not baseline_model.is_file():
        raise FileNotFoundError(f"Missing baseline {baseline_model}. Run ./run.sh --download first.")

    if not logits_path.is_file():
        print(f"[ppl-base] Computing F16 baseline logits ({args.ppl_chunks} chunks, ctx={args.ppl_ctx})...", flush=True)
        base_ppl = run_perplexity(
            bins["upstream"],
            baseline_model,
            corpus_path,
            ngl=0,
            ctx=args.ppl_ctx,
            batch=128,
            chunks=args.ppl_chunks,
            threads=args.threads,
            tmp_dir=tmp_dir,
            raw_log=logs_dir / "ppl_base_f16.log",
            save_logits=logits_path,
        )
    else:
        print(f"[ppl-base] Reusing existing logits file {logits_path.name}", flush=True)
        base_ppl = run_perplexity(
            bins["upstream"],
            baseline_model,
            corpus_path,
            ngl=0,
            ctx=args.ppl_ctx,
            batch=128,
            chunks=args.ppl_chunks,
            threads=args.threads,
            tmp_dir=tmp_dir,
            raw_log=logs_dir / "ppl_base_f16.log",
        )

    results: list[dict[str, object]] = []
    for target in TARGETS:
        mpath = models_dir / target.filename
        if not mpath.is_file():
            print(f"[warn] Skipping missing model {target.filename}", flush=True)
            continue
        size_mib = round(mpath.stat().st_size / (1024 ** 2), 1)
        for eng in target.engines:
            print(
                f"=== {target.key} [{eng}] ({size_mib} MiB, ngl={target.ngl_bench}, ppl_batch={target.ppl_batch}) ===",
                flush=True,
            )
            bench = run_llama_bench(
                bins[eng],
                mpath,
                target.ngl_bench,
                args.reps,
                args.threads,
                tmp_dir,
                raw_log=logs_dir / f"bench_{target.key}_{eng}.log",
            )
            if target.is_baseline:
                ppl_res = {
                    **base_ppl,
                    "ppl_base": base_ppl["ppl"],
                    "kld_mean": 0.0,
                    "kld_median": 0.0,
                    "kld_p99": 0.0,
                    "same_top_pct": 100.0,
                }
            else:
                ppl_res = run_perplexity(
                    bins[eng],
                    mpath,
                    corpus_path,
                    ngl=target.ngl_ppl,
                    ctx=args.ppl_ctx,
                    batch=target.ppl_batch,
                    chunks=args.ppl_chunks,
                    threads=args.threads,
                    tmp_dir=tmp_dir,
                    raw_log=logs_dir / f"ppl_{target.key}_{eng}.log",
                    kld_base=logits_path,
                )
            row = {
                "model_key": target.key,
                "family": target.family,
                "quant": target.quant,
                "file_mib": size_mib,
                "engine": eng,
                "engine_rev": revs[eng],
                "ngl": target.ngl_bench,
                "ppl_batch": target.ppl_batch,
                **bench,
                **ppl_res,
                "energy_mwh_per_token": None,
            }
            results.append(row)
            print(
                f"  pp512={row['pp512_tok_s']} tg128={row['tg128_tok_s']} "
                f"RSS={row['peak_rss_mib']}MiB VRAM={row['peak_vram_mib']}MiB(+{row['delta_vram_mib']}) "
                f"PPL={row['ppl']} KLD={row['kld_mean']} SameTop={row['same_top_pct']}%",
                flush=True,
            )

    payload = {
        "gpu": gpu,
        "config": {
            "ppl_ctx": args.ppl_ctx,
            "ppl_chunks": args.ppl_chunks,
            "bench_reps": args.reps,
            "threads": args.threads,
            "load_mode": "none",
            "corpus": "wikitext-2-raw/wiki.test.raw",
        },
        "revs": revs,
        "results": results,
    }
    json_out = args.artifacts_dir / "b3_1_results.json"
    csv_out = args.artifacts_dir / "b3_1_results.csv"
    md_out = args.artifacts_dir / "b3_1_summary.md"

    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if results:
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    lines = [
        "# B3.1 Resultados: Qwen3.5-2B vs Ternary-Bonsai-2-2B (Docker CUDA)",
        "",
        f"- GPU: `{gpu['name']}` ({gpu['vram_total']}, `sm_{gpu['compute_cap'].replace('.', '')}`, driver `{gpu['driver']}`)",
        f"- Engines: `upstream` (`{revs['upstream']}`), `prism` (`{revs['prism']}`)",
        f"- PPL/KLD: `wikitext-2-raw/wiki.test.raw`, `n_ctx={args.ppl_ctx}`, `chunks={args.ppl_chunks}`; `llama-bench`: `pp512`/`tg128`, `reps={args.reps}`, `-lm none`",
        "",
        "| Modelo | Quant | Tamaño (MiB) | Motor | NGL | pp512 (tok/s) | tg128 (tok/s) | RSS Pico (MiB) | VRAM Pico (MiB) | PPL | KLD vs F16 | SameTop (%) |",
        "| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| {r['family']} | `{r['quant']}` | {r['file_mib']} | `{r['engine']}` | {r['ngl']} | "
            f"{r['pp512_tok_s']} | {r['tg128_tok_s']} | {r['peak_rss_mib']} | {r['peak_vram_mib']} | "
            f"{r['ppl']} | {r['kld_mean']} | {r['same_top_pct']} |"
        )
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] Wrote {json_out.name}, {csv_out.name}, {md_out.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
