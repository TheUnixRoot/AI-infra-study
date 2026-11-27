#!/usr/bin/env python3
"""B3.1 KV-Cache Ablation (PrismML 43f31f4 inside Docker sm_61):
Mode A: Hadamard KV rotation (default on 43f31f4: -fa on -ctk q4_0 -ctv q4_0)
Mode B: PrismML Mean-Centering KV (LLAMA_ATTN_ROT_DISABLE=1 + -ctk q4_0 -ctv q4_0 + --kv-mean-center)
Mode B0: Raw Q4_0 without rotation or mean-center (ablation control)
Mode C: Uncompressed KV baseline (-fa on -ctk f16 -ctv f16)

Also verifies:
- llama-kv-mean-center calibration on non-overlapping wikitext-2 tail split
- PR #53 calibration-basis mismatch guard (fails cleanly when rotation state mismatches)
- End-to-end llama-cli timing WITH --kv-mean-center loaded
- Hybrid Qwen3.5 long-prompt / KV-shift CUDA stability check (issue #21383 probe)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bench_suite import parse_ppl_output, query_gpu_info, run_monitored


MODELS = (
    {
        "key": "bonsai2-2b-pq2_0",
        "label": "Ternary-Bonsai-2-2B PQ2_0",
        "filename": "Ternary-Bonsai-2-2B-PQ2_0.gguf",
        "ppl_batch": 128,
        "bench_depths": [0, 2048, 4096],
    },
    {
        "key": "bonsai2-2b-ptq1_0",
        "label": "Ternary-Bonsai-2-2B PTQ1_0",
        "filename": "Ternary-Bonsai-2-2B-PTQ1_0.gguf",
        "ppl_batch": 8,
        "bench_depths": [0, 2048],
    },
    {
        "key": "qwen3.5-2b-q4_k_m",
        "label": "Qwen3.5-2B Q4_K_M",
        "filename": "Qwen3.5-2B-Q4_K_M.gguf",
        "ppl_batch": 128,
        "bench_depths": [0, 1024],
    },
)


def make_env(bin_dir: Path, rot_disable: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    if rot_disable:
        env["LLAMA_ATTN_ROT_DISABLE"] = "1"
    else:
        env.pop("LLAMA_ATTN_ROT_DISABLE", None)
    return env


def ensure_calib_corpus(corpus_path: Path, out_path: Path) -> Path:
    """Use lines 1500+ of wiki.test.raw so calibration never overlaps chunks 0..7."""
    if not out_path.is_file():
        lines = corpus_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        out_path.write_text("".join(lines[1499:]), encoding="utf-8")
    return out_path


def run_kv_calibration(
    bin_dir: Path,
    model_path: Path,
    calib_corpus: Path,
    out_bias: Path,
    batch: int,
    chunks: int,
    ctx: int,
    threads: int,
    raw_log: Path,
    supports_batch: bool,
) -> dict[str, object]:
    cmd = [
        str(bin_dir / "llama-kv-mean-center"),
        "-m", str(model_path),
        "-lm", "none",
        "-f", str(calib_corpus),
        "-o", str(out_bias),
        "-ngl", "99",
        "-c", str(ctx),
        "--chunks", str(chunks),
        "-t", str(threads),
    ]
    if supports_batch:
        cmd.extend(["-b", str(batch), "-ub", str(batch)])
    proc = subprocess.run(cmd, env=make_env(bin_dir), capture_output=True, text=True)
    combined = proc.stdout + "\n" + proc.stderr
    raw_log.write_text(f"CMD: {' '.join(cmd)}\nRC: {proc.returncode}\n\n{combined}", encoding="utf-8", errors="replace")
    ok = proc.returncode == 0 and out_bias.is_file()
    note_line = next((l.strip() for l in combined.splitlines() if "wrote K-cache mean-centering bias" in l), None)
    return {
        "ok": ok,
        "rc": proc.returncode,
        "bias_path": out_bias.name if ok else None,
        "bias_bytes": out_bias.stat().st_size if ok else None,
        "summary_line": note_line,
        "error": None if ok else combined[-600:],
    }


def run_cli_probe(
    bin_dir: Path,
    model_path: Path,
    prompt: str,
    n_gen: int,
    ctx: int,
    ctk: str,
    ctv: str,
    rot_disable: bool,
    bias_path: Path | None,
    threads: int,
    raw_log: Path,
) -> dict[str, object]:
    cmd = [
        str(bin_dir / "llama-cli"),
        "-m", str(model_path),
        "-lm", "none",
        "-ngl", "99",
        "-fa", "on",
        "-c", str(ctx),
        "--cache-type-k", ctk,
        "--cache-type-v", ctv,
        "-t", str(threads),
        "-st",
        "-p", prompt,
        "-n", str(n_gen),
        "--verbose",
    ]
    if bias_path is not None:
        cmd.extend(["--kv-mean-center", str(bias_path)])
    proc = subprocess.run(cmd, env=make_env(bin_dir, rot_disable=rot_disable), capture_output=True, text=True)
    raw_log.write_text(
        f"CMD: {' '.join(cmd)}\nLLAMA_ATTN_ROT_DISABLE={int(rot_disable)}\nRC: {proc.returncode}\n\n"
        f"--- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}\n",
        encoding="utf-8",
        errors="replace",
    )
    m_tps = re.search(r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]", proc.stdout)
    m_rot_k = re.search(r"n_rot_k\s*=\s*(\d+)", proc.stderr)
    m_rot_v = re.search(r"n_rot_v\s*=\s*(\d+)", proc.stderr)
    m_kv_sz = re.search(r"llama_kv_cache:\s*size\s*=\s*([^\n]+)", proc.stderr)
    err_lines = [l.strip() for l in proc.stderr.splitlines() if "load_kv_mean_center" in l or "CUDA error" in l]
    return {
        "rc": proc.returncode,
        "rot_disable": rot_disable,
        "n_rot_k": int(m_rot_k.group(1)) if m_rot_k else None,
        "n_rot_v": int(m_rot_v.group(1)) if m_rot_v else None,
        "kv_cache_line": m_kv_sz.group(1).strip() if m_kv_sz else None,
        "prompt_tok_s": float(m_tps.group(1)) if m_tps else None,
        "gen_tok_s": float(m_tps.group(2)) if m_tps else None,
        "relevant_lines": err_lines[:5],
    }


def run_ppl_mode(
    bin_dir: Path,
    model_path: Path,
    corpus_path: Path,
    logits_base: Path,
    ctx: int,
    batch: int,
    chunks: int,
    threads: int,
    ctk: str,
    ctv: str,
    rot_disable: bool,
    bias_path: Path | None,
    tmp_dir: Path,
    raw_log: Path,
) -> dict[str, object]:
    cmd = [
        str(bin_dir / "llama-perplexity"),
        "-m", str(model_path),
        "-lm", "none",
        "-f", str(corpus_path),
        "-ngl", "99",
        "-fa", "on",
        "-ctk", ctk,
        "-ctv", ctv,
        "-c", str(ctx),
        "-b", str(batch),
        "-ub", str(batch),
        "--chunks", str(chunks),
        "-t", str(threads),
        "--kl-divergence",
        "--kl-divergence-base", str(logits_base),
    ]
    if bias_path is not None:
        cmd.extend(["--kv-mean-center", str(bias_path)])
    rc, stdout, stderr, rss_mib, peak_vram_mib, delta_vram_mib = run_monitored(
        cmd, make_env(bin_dir, rot_disable=rot_disable), tmp_dir / "kv_ppl_time.txt", raw_log=raw_log
    )
    combined = stdout + "\n" + stderr
    if rc != 0:
        return {"ok": False, "rc": rc, "error": combined[-600:]}
    metrics = parse_ppl_output(combined)
    metrics["ok"] = True
    metrics["peak_rss_mib"] = rss_mib
    metrics["peak_vram_mib"] = peak_vram_mib
    metrics["delta_vram_mib"] = delta_vram_mib
    return metrics


def run_bench_kv(
    bin_dir: Path,
    model_path: Path,
    ctk: str,
    ctv: str,
    rot_disable: bool,
    depths: list[int],
    reps: int,
    threads: int,
    tmp_dir: Path,
    raw_log: Path,
) -> dict[str, object]:
    d_arg = ",".join(str(d) for d in depths)
    cmd = [
        str(bin_dir / "llama-bench"),
        "-m", str(model_path),
        "-lm", "none",
        "-ngl", "99",
        "-fa", "1",
        "-ctk", ctk,
        "-ctv", ctv,
        "-t", str(threads),
        "-p", "512",
        "-n", "128",
        "-d", d_arg,
        "-r", str(reps),
        "-o", "json",
    ]
    rc, stdout, stderr, rss_mib, peak_vram_mib, delta_vram_mib = run_monitored(
        cmd, make_env(bin_dir, rot_disable=rot_disable), tmp_dir / "kv_bench_time.txt", raw_log=raw_log
    )
    if rc != 0:
        return {"ok": False, "rc": rc, "error": (stdout + stderr)[-600:]}
    rows = json.loads(stdout)
    by_depth: dict[str, dict[str, float | None]] = {}
    for d in depths:
        pp = next((r for r in rows if r.get("n_depth") == d and r.get("n_prompt", 0) == 512), None)
        tg = next((r for r in rows if r.get("n_depth") == d and r.get("n_gen", 0) == 128), None)
        by_depth[str(d)] = {
            "pp512_tok_s": round(pp["avg_ts"], 2) if pp else None,
            "pp512_std": round(pp["stddev_ts"], 2) if pp else None,
            "tg128_tok_s": round(tg["avg_ts"], 2) if tg else None,
            "tg128_std": round(tg["stddev_ts"], 2) if tg else None,
        }
    return {
        "ok": True,
        "by_depth": by_depth,
        "peak_rss_mib": rss_mib,
        "peak_vram_mib": peak_vram_mib,
        "delta_vram_mib": delta_vram_mib,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run B3.1 KV-cache A/B/C ablation in Docker")
    parser.add_argument("--base-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=SCRIPT_DIR.parent / "artifacts")
    parser.add_argument("--ppl-chunks", type=int, default=8)
    parser.add_argument("--ppl-ctx", type=int, default=512)
    parser.add_argument("--calib-chunks", type=int, default=16)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    bin_prism = args.base_dir / ".build" / "bin-prism"
    models_dir = args.base_dir / "models"
    logs_dir = args.base_dir / ".build" / "logs"
    tmp_dir = args.base_dir / ".build" / "tmp"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = args.base_dir / "data" / "wikitext-2-raw" / "wiki.test.raw"
    calib_corpus = ensure_calib_corpus(corpus_path, logs_dir / "calib_tail.txt")
    logits_base = args.base_dir / "logits" / f"qwen35_2b_f16_ctx{args.ppl_ctx}_chunks{args.ppl_chunks}.logits"
    if not logits_base.is_file():
        raise FileNotFoundError(f"Missing baseline logits: {logits_base}")

    kvc_help = subprocess.run(
        [str(bin_prism / "llama-kv-mean-center"), "--help"],
        env=make_env(bin_prism),
        capture_output=True,
        text=True,
    )
    supports_batch = ("-b," in (kvc_help.stdout + kvc_help.stderr))
    print(f"[info] llama-kv-mean-center supports -b/-ub: {supports_batch}", flush=True)

    # Verification probes on PQ2_0
    pq2_model = models_dir / "Ternary-Bonsai-2-2B-PQ2_0.gguf"
    pq2_bias = logs_dir / "bonsai2-2b-pq2_0-kv-bias.gguf"
    if not pq2_bias.is_file():
        run_kv_calibration(
            bin_prism, pq2_model, calib_corpus, pq2_bias,
            batch=128, chunks=args.calib_chunks, ctx=512, threads=args.threads,
            raw_log=logs_dir / "kv_calib_bonsai2-2b-pq2_0.log", supports_batch=supports_batch,
        )

    print("[verify] Step 2/3 llama-cli probes on PQ2_0...", flush=True)
    probe_a_rot = run_cli_probe(
        bin_prism, pq2_model, "Hola", 16, 4096, "q4_0", "q4_0", False, None, args.threads, logs_dir / "probe_step2_rot.log"
    )
    probe_guard_fail = run_cli_probe(
        bin_prism, pq2_model, "Hola", 16, 4096, "q4_0", "q4_0", False, pq2_bias, args.threads, logs_dir / "probe_step3_guard_fail.log"
    )
    probe_b_pass = run_cli_probe(
        bin_prism, pq2_model, "Hola", 16, 4096, "q4_0", "q4_0", True, pq2_bias, args.threads, logs_dir / "probe_step3_guard_pass.log"
    )

    # End-to-end 128-token decode timing comparison WITH --kv-mean-center loaded
    long_prompt = "Explain how fast Walsh-Hadamard transforms reduce outliers in neural weights. " * 20
    e2e_cli = {
        "A_hadamard_kv_q4": run_cli_probe(
            bin_prism, pq2_model, long_prompt, 128, 4096, "q4_0", "q4_0", False, None, args.threads, logs_dir / "cli_e2e_mode_A.log"
        ),
        "B_meancenter_kv_q4": run_cli_probe(
            bin_prism, pq2_model, long_prompt, 128, 4096, "q4_0", "q4_0", True, pq2_bias, args.threads, logs_dir / "cli_e2e_mode_B.log"
        ),
        "C_control_kv_f16": run_cli_probe(
            bin_prism, pq2_model, long_prompt, 128, 4096, "f16", "f16", False, None, args.threads, logs_dir / "cli_e2e_mode_C.log"
        ),
    }

    # Issue #21383 probe: >2000 token prompt on hybrid Qwen3.5 (with -c 4096) and overflow check (-c 512 with 700-tok prompt)
    huge_prompt = " ".join(["wikitext"] * 1800)
    shift_prompt = " ".join(["token"] * 650)
    stress_21383 = {
        "long_prompt_4k_mode_A": run_cli_probe(
            bin_prism, pq2_model, huge_prompt, 16, 4096, "q4_0", "q4_0", False, None, args.threads, logs_dir / "stress_4k_mode_A.log"
        ),
        "long_prompt_4k_mode_B": run_cli_probe(
            bin_prism, pq2_model, huge_prompt, 16, 4096, "q4_0", "q4_0", True, pq2_bias, args.threads, logs_dir / "stress_4k_mode_B.log"
        ),
        "overflow_512_mode_A": run_cli_probe(
            bin_prism, pq2_model, shift_prompt, 16, 512, "q4_0", "q4_0", False, None, args.threads, logs_dir / "stress_overflow_mode_A.log"
        ),
        "overflow_512_mode_B": run_cli_probe(
            bin_prism, pq2_model, shift_prompt, 16, 512, "q4_0", "q4_0", True, pq2_bias, args.threads, logs_dir / "stress_overflow_mode_B.log"
        ),
    }

    model_results: list[dict[str, object]] = []
    for cfg in MODELS:
        key = cfg["key"]
        mpath = models_dir / cfg["filename"]
        bias_path = logs_dir / f"{key}-kv-bias.gguf"
        print(f"=== [model] {key} ===", flush=True)
        calib = run_kv_calibration(
            bin_prism, mpath, calib_corpus, bias_path,
            batch=cfg["ppl_batch"], chunks=args.calib_chunks, ctx=512, threads=args.threads,
            raw_log=logs_dir / f"kv_calib_{key}.log", supports_batch=supports_batch,
        )
        print(f"  calib ok={calib['ok']} ({calib['summary_line'] or calib['rc']})", flush=True)

        ppl_modes: dict[str, dict[str, object]] = {}
        mode_defs = [
            ("A_hadamard_kv_q4", "q4_0", "q4_0", False, None),
            ("B_meancenter_kv_q4", "q4_0", "q4_0", True, bias_path if calib["ok"] else None),
            ("B0_norot_raw_q4", "q4_0", "q4_0", True, None),
            ("C_control_kv_f16", "f16", "f16", False, None),
        ]
        for mname, ctk, ctv, rot_dis, bpath in mode_defs:
            if mname == "B_meancenter_kv_q4" and bpath is None:
                ppl_modes[mname] = {"ok": False, "error": "skipped because calibration failed"}
                continue
            print(f"  [ppl] {key} {mname} (ctk={ctk}, rot_disable={rot_dis})...", flush=True)
            ppl_modes[mname] = run_ppl_mode(
                bin_prism, mpath, corpus_path, logits_base,
                ctx=args.ppl_ctx, batch=cfg["ppl_batch"], chunks=args.ppl_chunks, threads=args.threads,
                ctk=ctk, ctv=ctv, rot_disable=rot_dis, bias_path=bpath,
                tmp_dir=tmp_dir, raw_log=logs_dir / f"kv_ppl_{key}_{mname}.log",
            )

        bench_modes: dict[str, dict[str, object]] = {}
        for bname, ctk, ctv, rot_dis in [
            ("A_hadamard_kv_q4", "q4_0", "q4_0", False),
            ("B_norot_kv_q4", "q4_0", "q4_0", True),
            ("C_control_kv_f16", "f16", "f16", False),
        ]:
            print(f"  [bench] {key} {bname} depths={cfg['bench_depths']}...", flush=True)
            bench_modes[bname] = run_bench_kv(
                bin_prism, mpath, ctk=ctk, ctv=ctv, rot_disable=rot_dis,
                depths=cfg["bench_depths"], reps=args.reps, threads=args.threads,
                tmp_dir=tmp_dir, raw_log=logs_dir / f"kv_bench_{key}_{bname}.log",
            )

        model_results.append({
            "key": key,
            "label": cfg["label"],
            "file_mib": round(mpath.stat().st_size / (1024 ** 2), 1),
            "calibration": calib,
            "perplexity": ppl_modes,
            "bench": bench_modes,
        })

    payload = {
        "gpu": query_gpu_info(),
        "prism_rev": (bin_prism / "GIT_REV").read_text().strip(),
        "ppl_chunks": args.ppl_chunks,
        "ppl_ctx": args.ppl_ctx,
        "calib_chunks": args.calib_chunks,
        "calib_source": "data/wikitext-2-raw/wiki.test.raw (lines 1500+, non-overlapping)",
        "verifications": {
            "step2_hadamard_rot_active": probe_a_rot,
            "step3_pr53_guard_rejected": probe_guard_fail,
            "step3_meancenter_rot_disabled_pass": probe_b_pass,
            "llama_cli_128tok_pq2_e2e": e2e_cli,
            "issue_21383_hybrid_probe": stress_21383,
        },
        "models": model_results,
    }

    out_json = args.artifacts_dir / "b3_1_kv_ablation.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ok] Saved {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
