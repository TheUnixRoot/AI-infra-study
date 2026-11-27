#!/usr/bin/env python3
"""Execute B3.1 Hadamard Ablation Ladder (N1 unrotated vs N2 Hadamard in PQ2_0) inside Docker.

Runs against PrismML-Eng/llama.cpp (`bin-prism`, rev 43f31f4):
  1. F16 Identity Tests (`qwen3.5-2b-F16-N1.gguf`, `qwen3.5-2b-F16-N2.gguf`)
     at `ngl=0` vs baseline `Qwen3.5-2B-F16` logits (`KLD ~= 0` confirms exact math).
  2. `llama-quantize ... PQ2_0 --output-tensor-type pq2_0 --token-embedding-type pq2_0`
     producing `qwen3.5-2b-N1-raw-PQ2_0.gguf` and `qwen3.5-2b-N2-hadamard-PQ2_0.gguf`.
  3. GPU evaluation (`ngl=99`) via `llama-bench`, `llama-perplexity` (PPL/KLD/SameTop),
     and a quick sanity text generation via `llama-cli`.
  4. Saves real measured outputs to `artifacts/b3_1_ablation_ladder.json`.
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

from bench_suite import parse_ppl_output, query_gpu_info, run_llama_bench, run_perplexity  # noqa: E402


def run_quantize_pq2(bin_dir: Path, src_f16: Path, dst_pq2: Path, raw_log: Path, force: bool = False) -> float:
    if dst_pq2.is_file() and dst_pq2.stat().st_size > 100_000_000 and not force:
        mb = round(dst_pq2.stat().st_size / (1024 ** 2), 1)
        print(f"[skip] {dst_pq2.name} already exists ({mb} MiB)", flush=True)
        return mb

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    tmp_dst = dst_pq2.with_suffix(".gguf.tmp")
    if tmp_dst.exists():
        tmp_dst.unlink()

    cmd = [
        str(bin_dir / "llama-quantize"),
        "--output-tensor-type", "pq2_0",
        "--token-embedding-type", "pq2_0",
        "--tensor-type", "ssm_alpha=f32",
        "--tensor-type", "ssm_beta=f32",
        str(src_f16),
        str(tmp_dst),
        "PQ2_0",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    combined = proc.stdout + "\n" + proc.stderr
    raw_log.write_text(f"CMD: {' '.join(cmd)}\nRC: {proc.returncode}\n\n{combined}", encoding="utf-8", errors="replace")
    if proc.returncode != 0 or not tmp_dst.is_file():
        raise RuntimeError(f"llama-quantize failed for {src_f16.name} (rc={proc.returncode}): {combined[-600:]}")
    if dst_pq2.exists():
        dst_pq2.unlink()
    tmp_dst.replace(dst_pq2)
    return round(dst_pq2.stat().st_size / (1024 ** 2), 1)


def run_cli_sample(bin_dir: Path, model_path: Path, prompt: str, threads: int, raw_log: Path) -> str:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    cmd = [
        str(bin_dir / "llama-cli"),
        "-m", str(model_path),
        "-lm", "none",
        "-ngl", "99",
        "-fa", "on",
        "-c", "1024",
        "-t", str(threads),
        "-st",
        "-p", prompt,
        "-n", "64",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    raw_log.write_text(
        f"CMD: {' '.join(cmd)}\nRC: {proc.returncode}\n\n--- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}\n",
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        return f"[error rc={proc.returncode}]"
    # Extract reply after echoed prompt
    out = proc.stdout
    idx = out.find(prompt)
    if idx != -1:
        out = out[idx + len(prompt):]
    out = re.sub(r"\[\s*Prompt:.*?\]", "", out, flags=re.S)
    out = out.replace("Exiting...", "").strip()
    return out[:360]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run B3.1 Hadamard Ablation Ladder (N1 vs N2)")
    parser.add_argument("--base-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=SCRIPT_DIR.parent / "artifacts")
    parser.add_argument("--ppl-chunks", type=int, default=8)
    parser.add_argument("--ppl-ctx", type=int, default=512)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip-identity", action="store_true", help="Skip F16 identity PPL checks")
    parser.add_argument("--force-quantize", action="store_true")
    args = parser.parse_args()

    bin_prism = args.base_dir / ".build" / "bin-prism"
    models_dir = args.base_dir / "models"
    logs_dir = args.base_dir / ".build" / "logs"
    tmp_dir = args.base_dir / ".build" / "tmp"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = args.base_dir / "data" / "wikitext-2-raw" / "wiki.test.raw"
    logits_base = args.base_dir / "logits" / f"qwen35_2b_f16_ctx{args.ppl_ctx}_chunks{args.ppl_chunks}.logits"
    if not logits_base.is_file():
        raise FileNotFoundError(f"Missing baseline logits: {logits_base}")

    gpu = query_gpu_info()
    prism_rev = (bin_prism / "GIT_REV").read_text(encoding="utf-8").strip()

    f16_n1 = models_dir / "qwen3.5-2b-F16-N1.gguf"
    f16_n2 = models_dir / "qwen3.5-2b-F16-N2.gguf"
    f16_n1_twn = models_dir / "qwen3.5-2b-F16-N1-twn.gguf"
    f16_n2_twn = models_dir / "qwen3.5-2b-F16-N2-twn.gguf"

    pq2_n1 = models_dir / "qwen3.5-2b-N1-raw-PQ2_0.gguf"
    pq2_n2 = models_dir / "qwen3.5-2b-N2-hadamard-PQ2_0.gguf"
    pq2_n1_twn = models_dir / "qwen3.5-2b-N1-twn-PQ2_0.gguf"
    pq2_n2_twn = models_dir / "qwen3.5-2b-N2-twn-PQ2_0.gguf"

    identity_tests: dict[str, dict[str, object]] = {}
    for tag, mpath in (("F16_N1_identity", f16_n1), ("F16_N2_hadamard", f16_n2)):
        id_log = logs_dir / f"identity_{tag}.log"
        if args.skip_identity:
            if id_log.is_file():
                identity_tests[tag] = parse_ppl_output(id_log.read_text(encoding="utf-8", errors="replace"))
            continue
        print(f"=== [identity] {tag} ({mpath.name}, ngl=0, chunks={args.ppl_chunks}) ===", flush=True)
        res = run_perplexity(
            bin_prism, mpath, corpus_path,
            ngl=0, ctx=args.ppl_ctx, batch=128, chunks=args.ppl_chunks, threads=args.threads,
            tmp_dir=tmp_dir, raw_log=id_log, kld_base=logits_base,
        )
        identity_tests[tag] = res
        print(f"  PPL={res['ppl']} (base={res['ppl_base']}) KLD={res['kld_mean']} SameTop={res['same_top_pct']}%", flush=True)

    levels: list[dict[str, object]] = []
    prompt = "Explain what a hash table is in two sentences:"

    variants = [
        ("N1_amax", "Qwen3.5-2B N1 Raw PQ2_0 (sin Hadamard, amax-RTN)", f16_n1, pq2_n1),
        ("N2_amax", "Qwen3.5-2B N2 Hadamard + amax-RTN PQ2_0", f16_n2, pq2_n2),
        ("N1_twn", "Qwen3.5-2B N1 sin Hadamard + TWN MSE-RTN PQ2_0", f16_n1_twn, pq2_n1_twn),
        ("N2_twn", "Qwen3.5-2B N2 Hadamard + TWN MSE-RTN PQ2_0", f16_n2_twn, pq2_n2_twn),
    ]

    for lvl, label, src_f16, dst_pq2 in variants:
        if not src_f16.is_file() and not dst_pq2.is_file():
            print(f"[skip] Missing {src_f16.name} for {lvl}", flush=True)
            continue
        print(f"=== [quantize] {lvl}: {src_f16.name} -> {dst_pq2.name} ===", flush=True)
        size_mib = run_quantize_pq2(
            bin_prism, src_f16, dst_pq2,
            raw_log=logs_dir / f"quant_{lvl}.log",
            force=args.force_quantize,
        )
        print(f"  Quantized size: {size_mib} MiB", flush=True)

        print(f"=== [bench] {lvl} (ngl=99) ===", flush=True)
        bench = run_llama_bench(
            bin_prism, dst_pq2, ngl=99, reps=args.reps, threads=args.threads,
            tmp_dir=tmp_dir, raw_log=logs_dir / f"bench_{lvl}_pq2.log",
        )
        print(f"=== [ppl] {lvl} (ngl=99, chunks={args.ppl_chunks}) ===", flush=True)
        ppl = run_perplexity(
            bin_prism, dst_pq2, corpus_path,
            ngl=99, ctx=args.ppl_ctx, batch=128, chunks=args.ppl_chunks, threads=args.threads,
            tmp_dir=tmp_dir, raw_log=logs_dir / f"ppl_{lvl}_pq2.log", kld_base=logits_base,
        )
        sample = run_cli_sample(
            bin_prism, dst_pq2, prompt, threads=args.threads,
            raw_log=logs_dir / f"cli_{lvl}_pq2.log",
        )
        row = {
            "level": lvl,
            "label": label,
            "filename": dst_pq2.name,
            "file_mib": size_mib,
            "engine": "prism",
            "engine_rev": prism_rev,
            "ngl": 99,
            **bench,
            **ppl,
            "sanity_prompt": prompt,
            "sanity_output": sample,
        }
        levels.append(row)
        print(
            f"  pp512={row['pp512_tok_s']} tg128={row['tg128_tok_s']} "
            f"PPL={row['ppl']} KLD={row['kld_mean']} SameTop={row['same_top_pct']}%",
            flush=True,
        )

    payload = {
        "gpu": gpu,
        "prism_rev": prism_rev,
        "ppl_chunks": args.ppl_chunks,
        "ppl_ctx": args.ppl_ctx,
        "bench_reps": args.reps,
        "identity_tests": identity_tests,
        "levels": levels,
    }
    out_json = args.artifacts_dir / "b3_1_ablation_ladder.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ok] Saved {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
