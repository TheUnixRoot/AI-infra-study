#!/usr/bin/env python3
"""B2.2 - Cliente de carga asincrono para servidores OpenAI-compatibles (vLLM / SGLang).

Subcomandos:
  run     Lanza la carga contra un servidor ya levantado (modo real, Cloud-2).
          Escribe artifacts/b2_2_raw_<engine>.json
  report  Fusiona los raw por motor y genera b2_2_results.json + b2_2_summary.md.
  dry     Datos SINTETICOS para validar el parser de reportes sin GPU (<15 s, CPU).
          Salida SIEMPRE en un directorio aparte y marcada "synthetic": true
          (AGENTS.md 5.1: nunca mezclar cifras simuladas con resultados reales).

Metricas por concurrencia B: TTFT p50/p95/p99, TPOT e ITL p50/p95/p99, throughput
(output tok/s y total tok/s), KV-cache % y preemptions (scrapeo de /metrics).
Solo stdlib en `report`/`dry`; `run` necesita aiohttp.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS = ROOT / "artifacts"
PCTS = (50, 95, 99)

# Nombres de metricas Prometheus por motor (V1 renombro varias; se prueban candidatas).
KV_METRICS = ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "sglang:token_usage"]
PREEMPT_METRICS = ["vllm:num_preemptions", "sglang:num_retracted_reqs"]


# ----------------------------------------------------------------------------- stats
def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def pct_dict(values: list[float]) -> dict:
    out = {}
    for p in PCTS:
        v = percentile(values, p)
        out[f"p{p}"] = round(v, 3) if v is not None else None
    return out


def summarize(records: list[dict], wall_s: float, nominal_in: int) -> dict:
    ok = [r for r in records if r.get("ok")]
    ttft = [r["ttft_ms"] for r in ok]
    tpot = [r["tpot_ms"] for r in ok if r.get("tpot_ms") is not None]
    itl = [x for r in ok for x in r.get("itl_ms", [])]
    out_tok = sum(r["output_tokens"] for r in ok)
    in_tok = sum(r.get("input_tokens") or nominal_in for r in ok)
    return {
        "requests_ok": len(ok),
        "requests_failed": len(records) - len(ok),
        "duration_s": round(wall_s, 3),
        "ttft_ms": pct_dict(ttft),
        "tpot_ms": pct_dict(tpot),
        "itl_ms": pct_dict(itl),
        "output_tok_s": round(out_tok / wall_s, 2) if wall_s > 0 else None,
        "total_tok_s": round((in_tok + out_tok) / wall_s, 2) if wall_s > 0 else None,
        "mean_ttft_ms": round(statistics.fmean(ttft), 3) if ttft else None,
    }


# --------------------------------------------------------------------------- metrics
_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|[+-]?Inf)\s*$")


def parse_prom(text: str) -> dict[str, float]:
    """Suma por nombre de metrica (agrega etiquetas). Ignora histogramas _bucket."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _LINE.match(line)
        if not m or m.group(1).endswith("_bucket"):
            continue
        try:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
        except ValueError:
            pass
    return out


def pick(metrics: dict[str, float], candidates: list[str]) -> float | None:
    for c in candidates:
        for name in (c, c + "_total"):
            if name in metrics:
                return metrics[name]
    return None


def metrics_summary(samples: list[dict[str, float]]) -> dict:
    if not samples:
        return {"available": False, "kv_cache_pct_max": None, "kv_cache_pct_mean": None, "preemptions": None}
    kv = [v * 100 for s in samples if (v := pick(s, KV_METRICS)) is not None]
    pre = [v for s in samples if (v := pick(s, PREEMPT_METRICS)) is not None]
    return {
        "available": True,
        "kv_cache_pct_max": round(max(kv), 2) if kv else None,
        "kv_cache_pct_mean": round(statistics.fmean(kv), 2) if kv else None,
        # contador acumulado: delta entre primera y ultima muestra de la ronda
        "preemptions": int(pre[-1] - pre[0]) if pre else None,
    }


# --------------------------------------------------------------------------- real run
def make_prompt(rng: random.Random, n_tokens: int, vocab_hi: int, shared: list[int]) -> list[int]:
    # IDs aleatorios unicos por peticion: evita que el prefix-cache (Radix/APC) infle resultados.
    # Con --prefix-share > 0 una parte inicial es comun (caso favorable a RadixAttention).
    tail = [rng.randrange(1000, vocab_hi) for _ in range(n_tokens - len(shared))]
    return shared + tail


async def one_request(session, url, model, prompt, out_len, timeout_s) -> dict:
    payload = {"model": model, "prompt": prompt, "max_tokens": out_len, "min_tokens": out_len,
               "temperature": 0.0, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    last = t0
    ttft = None
    itl: list[float] = []
    chunks = 0
    usage: dict = {}
    try:
        async with session.post(url, json=payload, timeout=timeout_s) as resp:
            if resp.status != 200:
                return {"ok": False, "error": f"HTTP {resp.status}: {(await resp.text())[:200]}"}
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                obj = json.loads(body)
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if choices and choices[0].get("text"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = (now - t0) * 1000
                    else:
                        itl.append((now - last) * 1000)
                    last = now
                    chunks += 1
    except Exception as e:  # noqa: BLE001 - se registra y cuenta como fallo
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if ttft is None:
        return {"ok": False, "error": "sin tokens"}
    out_tok = usage.get("completion_tokens") or chunks
    e2e = (last - t0) * 1000
    return {"ok": True, "ttft_ms": ttft, "e2e_ms": e2e, "output_tokens": out_tok,
            "input_tokens": usage.get("prompt_tokens"),
            "tpot_ms": (e2e - ttft) / (out_tok - 1) if out_tok > 1 else None, "itl_ms": itl}


async def poll_metrics(session, base, stop: asyncio.Event, samples: list, period=1.0):
    while not stop.is_set():
        try:
            async with session.get(f"{base}/metrics", timeout=5) as r:
                if r.status == 200:
                    samples.append(parse_prom(await r.text()))
        except Exception:  # noqa: BLE001 - /metrics es opcional
            pass
        try:
            await asyncio.wait_for(stop.wait(), period)
        except asyncio.TimeoutError:
            pass


async def run_level(session, args, base, B: int, rng: random.Random) -> dict:
    n = max(args.min_requests, B * args.requests_per_slot)
    shared = [rng.randrange(1000, args.vocab_hi) for _ in range(int(args.input_len * args.prefix_share))]
    prompts = [make_prompt(rng, args.input_len, args.vocab_hi, shared) for _ in range(n)]
    sem = asyncio.Semaphore(B)
    samples: list = []
    stop = asyncio.Event()
    poller = asyncio.create_task(poll_metrics(session, base, stop, samples))

    async def guarded(p):
        async with sem:
            return await one_request(session, f"{base}/v1/completions", args.model, p, args.output_len, args.timeout)

    t0 = time.perf_counter()
    recs = await asyncio.gather(*(guarded(p) for p in prompts))
    wall = time.perf_counter() - t0
    stop.set()
    await poller
    res = summarize(recs, wall, args.input_len)
    res["concurrency"] = B
    res["num_requests"] = n
    res["metrics"] = metrics_summary(samples)
    errs = sorted({r["error"] for r in recs if not r.get("ok")})[:3]
    if errs:
        res["errors_sample"] = errs
    return res


def config_of(a) -> dict:
    return {"model": a.model, "input_len": a.input_len, "output_len": a.output_len,
            "concurrency": list(a.concurrency), "prefix_share": a.prefix_share, "seed": a.seed}


async def cmd_run_async(args) -> int:
    try:
        import aiohttp
    except ImportError:
        print("[ERROR] falta aiohttp: pip install aiohttp", file=sys.stderr)
        return 2
    base = f"http://{args.host}:{args.port}"
    rng = random.Random(args.seed)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=None)) as session:
        print(f"[run] warmup x{args.warmup} ...", flush=True)
        wsem = asyncio.Semaphore(min(args.warmup, 8))

        async def warm():
            async with wsem:
                return await one_request(session, f"{base}/v1/completions", args.model,
                                         make_prompt(rng, args.input_len, args.vocab_hi, []),
                                         args.output_len, args.timeout)
        wres = await asyncio.gather(*(warm() for _ in range(args.warmup)))
        if not any(r.get("ok") for r in wres):
            print(f"[ERROR] warmup fallo: {wres[0].get('error')}", file=sys.stderr)
            return 1
        levels = []
        for B in args.concurrency:
            print(f"[run] {args.engine} B={B} ...", flush=True)
            lv = await run_level(session, args, base, B, rng)
            print(f"      TTFT p50={lv['ttft_ms']['p50']}ms  TPOT p50={lv['tpot_ms']['p50']}ms  "
                  f"out={lv['output_tok_s']} tok/s  fail={lv['requests_failed']}", flush=True)
            levels.append(lv)
    out = Path(args.out_dir) / f"b2_2_raw_{args.engine}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"engine": args.engine, "synthetic": False, "config": config_of(args),
                               "levels": levels}, indent=2), encoding="utf-8")
    print(f"[run] guardado {out}")
    return 0 if all(lv["requests_failed"] == 0 for lv in levels) else 1


# ------------------------------------------------------------------------ dry (synth)
def synth_records(rng: random.Random, B: int, n: int, in_len: int, out_len: int, base_ttft: float, base_tpot: float):
    recs = []
    for _ in range(n):
        ttft = base_ttft * (1 + 0.05 * B) * rng.lognormvariate(0, 0.25)
        itl = [base_tpot * (1 + 0.01 * B) * rng.lognormvariate(0, 0.15) for _ in range(out_len - 1)]
        recs.append({"ok": True, "ttft_ms": ttft, "output_tokens": out_len, "input_tokens": in_len,
                     "tpot_ms": statistics.fmean(itl), "itl_ms": itl})
    return recs


def cmd_dry(args) -> int:
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("b2_2_raw_*.json"):  # el dry-run nunca hereda datos previos
        stale.unlink()
    for engine, (bt, bp) in {"vllm": (40.0, 11.0), "sglang": (38.0, 10.5)}.items():
        levels = []
        for B in args.concurrency:
            n = max(args.min_requests, B * args.requests_per_slot)
            recs = synth_records(rng, B, n, args.input_len, args.output_len, bt, bp)
            wall = n * args.output_len * bp / 1000 / max(B, 1)
            lv = summarize(recs, wall, args.input_len)
            samples = [{"vllm:kv_cache_usage_perc": min(0.99, 0.01 * B * rng.random() + 0.02),
                        "vllm:num_preemptions_total": float(i) if B >= 64 else 0.0}
                       for i in range(3)]
            lv.update({"concurrency": B, "num_requests": n, "metrics": metrics_summary(samples)})
            levels.append(lv)
        (out_dir / f"b2_2_raw_{engine}.json").write_text(json.dumps(
            {"engine": engine, "synthetic": True, "config": config_of(args), "levels": levels}, indent=2),
            encoding="utf-8")
    print(f"[dry] datos sinteticos en {out_dir}")
    return cmd_report(args)


# -------------------------------------------------------------------------- report
def fmt(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def render_md(res: dict) -> str:
    c = res["config"]
    L = []
    if res["synthetic"]:
        L += ["> **DATOS SINTETICOS (dry-run): validan el parser, NO son mediciones. "
              "No citar en INFORME_RESULTADOS.md.**", ""]
    L += ["# B2.2 vLLM vs SGLang (FlashInfer)", "",
          f"Modelo: `{c['model']}` · entrada {c['input_len']} tok · salida {c['output_len']} tok · "
          f"prefix-share {c['prefix_share']}", ""]
    if res.get("hardware"):
        L += [f"Hardware: {res['hardware']}", ""]
    engines = res["engines"]
    for title, key in (("TTFT (ms)", "ttft_ms"), ("TPOT (ms)", "tpot_ms"), ("ITL (ms)", "itl_ms")):
        L += [f"## {title}", "", "| B | motor | p50 | p95 | p99 |", "| ---: | :--- | ---: | ---: | ---: |"]
        for B in c["concurrency"]:
            for e, lvls in engines.items():
                lv = lvls.get(str(B))
                if lv:
                    d = lv[key]
                    L.append(f"| {B} | {e} | {fmt(d['p50'])} | {fmt(d['p95'])} | {fmt(d['p99'])} |")
        L.append("")
    L += ["## Throughput y KV-cache", "",
          "| B | motor | out tok/s | total tok/s | KV-cache % (max) | preemptions | fallos |",
          "| ---: | :--- | ---: | ---: | ---: | ---: | ---: |"]
    for B in c["concurrency"]:
        for e, lvls in engines.items():
            lv = lvls.get(str(B))
            if lv:
                m = lv["metrics"]
                pre = "n/d" if m["preemptions"] is None else str(m["preemptions"])
                L.append(f"| {B} | {e} | {fmt(lv['output_tok_s'])} | {fmt(lv['total_tok_s'])} | "
                         f"{fmt(m['kv_cache_pct_max'])} | {pre} | {lv['requests_failed']} |")
    L += ["", "`n/d` = el motor no expone esa metrica en `/metrics` "
          "(SGLang no publica preemptions en todas las versiones).", ""]
    return "\n".join(L)


def cmd_report(args) -> int:
    d = Path(args.out_dir)
    raws = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("b2_2_raw_*.json"))]
    if not raws:
        print(f"[ERROR] no hay b2_2_raw_*.json en {d}", file=sys.stderr)
        return 1
    flags = {r["synthetic"] for r in raws}
    if len(flags) > 1:
        print("[ERROR] mezcla de datos sinteticos y reales; abortando", file=sys.stderr)
        return 1
    res = {"synthetic": flags.pop(), "config": raws[0]["config"], "hardware": args.hardware or None,
           "engines": {r["engine"]: {str(lv["concurrency"]): lv for lv in r["levels"]} for r in raws}}
    (d / "b2_2_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (d / "b2_2_summary.md").write_text(render_md(res), encoding="utf-8")
    print(f"[report] {d / 'b2_2_results.json'}\n[report] {d / 'b2_2_summary.md'}")
    return 0


# ----------------------------------------------------------------------------- cli
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
        p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 64])
        p.add_argument("--input-len", type=int, default=1024)
        p.add_argument("--output-len", type=int, default=256)
        p.add_argument("--requests-per-slot", type=int, default=4, help="peticiones = max(min, B*este)")
        p.add_argument("--min-requests", type=int, default=16)
        p.add_argument("--prefix-share", type=float, default=0.0,
                       help="fraccion del prompt comun (0 = sin reutilizacion)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out-dir", default=str(DEFAULT_ARTIFACTS))
        p.add_argument("--hardware", default="", help="texto libre, p. ej. 'L40S 48GB sm_89'")

    r = sub.add_parser("run")
    common(r)
    r.add_argument("--engine", required=True, choices=["vllm", "sglang"])
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=8000)
    r.add_argument("--warmup", type=int, default=8)
    r.add_argument("--vocab-hi", type=int, default=100000, help="IDs aleatorios < este valor (Qwen2.5: vocab 151k)")
    r.add_argument("--timeout", type=float, default=600)
    common(sub.add_parser("report"))
    common(sub.add_parser("dry"))
    a = ap.parse_args()
    if a.cmd == "run":
        return asyncio.run(cmd_run_async(a))
    return cmd_dry(a) if a.cmd == "dry" else cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
