#!/usr/bin/env python3
"""Prepare non-overlapping N3A chat+think calibration text (disjoint from PROMPT_DUEL & wiki val)."""
from __future__ import annotations

import random
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent

CHAT_TRACES: list[tuple[str, str, str]] = [
    (
        "Implement a Python function `merge_intervals(intervals)` that merges overlapping [start, end] pairs.",
        "Thinking Process:\n\n1. Sort intervals by start time.\n2. Maintain a result list starting with the first interval.\n3. If next interval overlaps (next[0] <= last[1]), extend last[1] = max(last[1], next[1]).\n4. Otherwise append the new interval.",
        "```python\ndef merge_intervals(intervals):\n    if not intervals:\n        return []\n    intervals.sort(key=lambda x: x[0])\n    merged = [list(intervals[0])]\n    for s, e in intervals[1:]:\n        if s <= merged[-1][1]:\n            merged[-1][1] = max(merged[-1][1], e)\n        else:\n            merged.append([s, e])\n    return merged\n```",
    ),
    (
        "Return ONLY valid JSON for an HTTP 503 health check response with fields `status`, `service`, `retry_after_sec`, and `checks`.",
        "The user requires raw valid JSON only, with keys status, service, retry_after_sec, and checks representing HTTP 503 unavailable.",
        '{\n  "status": "unavailable",\n  "service": "inference-gateway",\n  "retry_after_sec": 30,\n  "checks": {"gpu_worker": "overloaded", "kv_cache": "healthy"}\n}',
    ),
    (
        "Write a SQL query using a window function to rank products by revenue within each category.",
        "1. Aggregate order_items joined with products or sum price*qty per product.\n2. Use DENSE_RANK() OVER (PARTITION BY category ORDER BY total_rev DESC).\n3. Output product_id, category, total_rev, and rank.",
        "```sql\nSELECT\n    p.category,\n    p.id AS product_id,\n    SUM(oi.quantity * oi.unit_price) AS total_rev,\n    DENSE_RANK() OVER (PARTITION BY p.category ORDER BY SUM(oi.quantity * oi.unit_price) DESC) AS rev_rank\nFROM products p\nJOIN order_items oi ON oi.product_id = p.id\nGROUP BY p.category, p.id;\n```",
    ),
    (
        "A batch job processes 45,000 records at 300 records/sec, pauses 25 seconds for checkpointing, then processes 30,000 more at 250 records/sec. What is total elapsed time?",
        "1. Phase 1 time = 45000 / 300 = 150 seconds.\n2. Checkpoint pause = 25 seconds.\n3. Phase 2 time = 30000 / 250 = 120 seconds.\n4. Total time = 150 + 25 + 120 = 295 seconds (4 min 55 sec).",
        "Total elapsed time is **295 seconds** (4 minutes and 55 seconds): 150 s for the first batch, 25 s checkpoint pause, and 120 s for the second batch.",
    ),
    (
        "Show a concise C++ shared_memory tiled transpose kernel signature and explain why padding [TILE][TILE+1] helps.",
        "1. Column reads from shared memory without padding hit the same memory bank across threads in a warp (32-way bank conflict on 32-bit floats).\n2. Adding +1 column padding shifts consecutive rows by one bank, eliminating column-read bank conflicts.",
        "```cpp\n__shared__ float tile[32][33]; // +1 padding breaks 32-way bank conflicts\n```\nAdding one extra column shifts consecutive row elements into distinct shared-memory banks, so reading transposed columns along a warp finishes in a single transaction instead of serializing.",
    ),
    (
        "All pods in namespace `prod-infer` enforce `runAsNonRoot: true`. `worker-0` runs in `prod-infer` with UID 0. Is this configuration valid? Answer Yes or No and explain briefly.",
        "Check premise: runAsNonRoot requires UID != 0. UID 0 is root, violating the policy.",
        "No. Because `prod-infer` enforces `runAsNonRoot: true`, a container running as UID `0` (root) violates the security context and will be rejected.",
    ),
    (
        "Implement a Python token-bucket rate limiter class with `allow(now)` method.",
        "1. Store capacity, refill_rate, tokens, last_ts.\n2. On allow(now), refill (now - last_ts) * refill_rate capped at capacity.\n3. Update last_ts = now. If tokens >= 1, subtract 1 and return True; else return False.",
        "```python\nclass TokenBucket:\n    def __init__(self, rate: float, capacity: float):\n        self.rate = rate\n        self.capacity = capacity\n        self.tokens = capacity\n        self.last_ts = 0.0\n\n    def allow(self, now: float) -> bool:\n        if self.last_ts > 0.0:\n            self.tokens = min(self.capacity, self.tokens + (now - self.last_ts) * self.rate)\n        self.last_ts = now\n        if self.tokens >= 1.0:\n            self.tokens -= 1.0\n            return True\n        return False\n```",
    ),
    (
        "Return ONLY valid JSON representing a Kubernetes Pod resource request for 1 GPU and 4 GiB RAM.",
        "Format strict JSON with containers list, resources.requests and resources.limits using nvidia.com/gpu and memory 4Gi.",
        '{\n  "kind": "Pod",\n  "spec": {\n    "containers": [\n      {\n        "name": "vllm-worker",\n        "resources": {\n          "requests": {"memory": "4Gi", "cpu": "2", "nvidia.com/gpu": "1"},\n          "limits": {"memory": "4Gi", "nvidia.com/gpu": "1"}\n        }\n      }\n    ]\n  }\n}',
    ),
    (
        "Why does memory-coalesced global access improve GPU throughput in CUDA kernels?",
        "1. Hardware issues 32B/128B cache-line memory transactions per warp.\n2. When consecutive threads (threadIdx.x = 0..31) load adjacent addresses, a single burst satisfies all 32 lanes.\n3. Strided access wastes most of each fetched cache line, cutting effective bandwidth.",
        "Consecutive threads reading contiguous global addresses coalesce into a minimal number of 32-byte/128-byte DRAM transactions. Strided accesses force multiple wasted transfers per warp, slashing effective HBM/GDDR bandwidth.",
    ),
    (
        "According to Amdahl's Law, if 80% of a workload is parallelizable and you run it on 4 identical processors, what is the maximum speedup?",
        "1. Serial fraction s = 1 - 0.80 = 0.20, parallel fraction p = 0.80, N = 4.\n2. Speedup = 1 / (s + p / N) = 1 / (0.20 + 0.80 / 4) = 1 / (0.20 + 0.20) = 1 / 0.40 = 2.5.",
        "The theoretical maximum speedup is **2.5x** (`1 / (0.20 + 0.80 / 4) = 2.5`).",
    ),
    (
        "Write a minimal C++ RAII lock guard using `std::mutex` without copying or moving.",
        "Delete copy constructor and copy assignment; acquire mutex in ctor, unlock in dtor.",
        "```cpp\nstruct ScopeLock {\n    std::mutex& m;\n    explicit ScopeLock(std::mutex& mtx) : m(mtx) { m.lock(); }\n    ~ScopeLock() { m.unlock(); }\n    ScopeLock(const ScopeLock&) = delete;\n    ScopeLock& operator=(const ScopeLock&) = delete;\n};\n```",
    ),
    (
        "Explain in two sentences why Cholesky inverse GPTQ compensates quantization error better than round-to-nearest.",
        "GPTQ uses curvature X^T X via Hessian inverse to shift unquantized weights when a column rounds, whereas RTN treats every coordinate independently.",
        "Cholesky GPTQ measures second-order output sensitivity via $(X^T X)^{-1}$ and propagates each weight's rounding error onto remaining columns. Round-to-nearest ignores cross-weight correlations, letting correlated errors accumulate across layers.",
    ),
]


def wrap_chat(user: str, think: str, ans: str) -> str:
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n{think}\n</think>\n\n{ans}<|im_end|>\n"
    )


def build_repo_code_chat() -> list[str]:
    out = []
    targets = [
        ("Explain what this Python benchmarking utility measures and list its main inputs/outputs:", REPO / "b1_1_strides/bench_strides.py"),
        ("Summarize the CUDA shared-memory tiling logic implemented in this kernel file:", REPO / "b1_2_cuda_gemm/sgemm.cu"),
        ("Inspect this KV-cache VRAM calculator script and describe the main memory bottleneck formula:", REPO / "b2_1_kv_cache/kv_calc.py"),
        ("Review this continuous-batching benchmark configuration and highlight key throughput vs latency signals:", REPO / "b2_2_vllm_vs_sglang/README.md"),
        ("Summarize how this profiling wrapper parses kernel traces:", REPO / "b1_3_profiling/profile_trace.py"),
    ]
    for prompt, path in targets:
        if not path.is_file():
            continue
        txt = path.read_text(encoding="utf-8", errors="replace")
        for i in range(0, min(len(txt), 10800), 1600):
            seg = txt[i : i + 1600].strip()
            if len(seg) < 180:
                continue
            lines = [ln.strip() for ln in seg.splitlines() if ln.strip() and not ln.strip().startswith("#")]
            syms = [ln[:80] for ln in lines[:4]]
            user = f"{prompt}\n\n```\n{seg}\n```"
            think = (
                f"1. Inspect snippet from `{path.name}` ({len(seg)} chars).\n"
                f"2. Identify key constructs: {'; '.join(syms)}.\n"
                f"3. Formulate a concise explanation grounded in the code."
            )
            ans = "This code segment handles:\n" + "\n".join(f"- `{s}`" for s in syms[:3])
            out.append(wrap_chat(user, think, ans))
    return out


def build_playbook_chat() -> list[str]:
    out = []
    for rel in (
        "docs/temario/00-master-playbook.md",
        "docs/temario/01-agentic-mlops-playbook.md",
        "docs/temario/02-tronco-comun-y-itinerarios.md",
        "docs/campus/curso3/e0-e1-infra-baseline.md",
        "docs/campus/curso3/e2-ptq-hadamard-gptq.md",
    ):
        p = REPO / rel
        if not p.is_file():
            continue
        sections = re.split(r"(?=^##\s+)", p.read_text(encoding="utf-8", errors="replace"), flags=re.M)
        for sec in sections:
            sec = sec.strip()
            if len(sec) < 300 or len(sec) > 3000:
                continue
            first = sec.splitlines()[0].lstrip("# ").strip()
            body_preview = " ".join(ln.strip() for ln in sec.splitlines()[1:6] if ln.strip())[:320]
            out.append(
                wrap_chat(
                    f"Summarize the learning objectives and technical focus of \"{first}\":\n\n{sec[:1100]}",
                    f"1. Inspect heading `{first}`.\n2. Extract architecture constraints and deliverables.\n3. Return a crisp technical summary.",
                    f"**{first}** covers: {body_preview}",
                )
            )
    return out


def main() -> int:
    rng = random.Random(2026)
    chats = [wrap_chat(u, th, a) for u, th, a in CHAT_TRACES] * 6
    chats.extend(build_repo_code_chat() * 3)
    chats.extend(build_playbook_chat() * 3)
    rng.shuffle(chats)

    chat_out = SCRIPT_DIR / ".build" / "tmp" / "calib_n3a_chat.txt"
    chat_out.parent.mkdir(parents=True, exist_ok=True)
    chat_out.write_text("\n".join(chats), encoding="utf-8")
    print(f"[ok] Wrote {len(chats)} chat+think turns to {chat_out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
