#!/usr/bin/env python3
"""Idempotent downloader for B3.1 (Qwen3.5-2B vs Ternary-Bonsai-2-2B GGUFs + wikitext-2).

Downloads files directly to D: (b3_1_hadamard_bonsai/models and data) via
resumable HTTP streaming (curl -fL -C - or urllib with Range resume) ordered
smallest-first, avoiding hf_xet memory buffering/stalls.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    quant: str
    repo_id: str
    filename: str
    sha256: str | None = None


# Ordered smallest-first so small models are usable immediately
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="bonsai2-2b-ptq1_0",
        family="Ternary-Bonsai-2-2B",
        quant="PTQ1_0",
        repo_id="prism-ml/Ternary-Bonsai-2-2B-gguf",
        filename="Ternary-Bonsai-2-2B-PTQ1_0.gguf",
        sha256="36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa",
    ),
    ModelSpec(
        key="bonsai2-2b-pq2_0",
        family="Ternary-Bonsai-2-2B",
        quant="PQ2_0",
        repo_id="prism-ml/Ternary-Bonsai-2-2B-gguf",
        filename="Ternary-Bonsai-2-2B-PQ2_0.gguf",
        sha256="d5f01a4ecaf49239653308ee732ad1a0c5f802c0ddb997549fc7b694c8b45928",
    ),
    ModelSpec(
        key="bonsai2-2b-q2_0",
        family="Ternary-Bonsai-2-2B",
        quant="Q2_0",
        repo_id="prism-ml/Ternary-Bonsai-2-2B-gguf",
        filename="Ternary-Bonsai-2-2B-Q2_0.gguf",
        sha256="c2893fa6f951daee8b7263d5f49b455e57989ef56417a22125b03566dc560650",
    ),
    ModelSpec(
        key="qwen3.5-2b-tq1_0",
        family="Qwen3.5-2B",
        quant="TQ1_0",
        repo_id="titan0115/Qwen3.5-2B",
        filename="Qwen3.5-2B-TQ1_0.gguf",
    ),
    ModelSpec(
        key="qwen3.5-2b-iq2_xxs",
        family="Qwen3.5-2B",
        quant="UD-IQ2_XXS",
        repo_id="unsloth/Qwen3.5-2B-GGUF",
        filename="Qwen3.5-2B-UD-IQ2_XXS.gguf",
    ),
    ModelSpec(
        key="qwen3.5-2b-q4_k_m",
        family="Qwen3.5-2B",
        quant="Q4_K_M",
        repo_id="titan0115/Qwen3.5-2B",
        filename="Qwen3.5-2B-Q4_K_M.gguf",
    ),
    ModelSpec(
        key="qwen3.5-2b-f16",
        family="Qwen3.5-2B",
        quant="F16",
        repo_id="titan0115/Qwen3.5-2B",
        filename="Qwen3.5-2B-F16.gguf",
    ),
)

WIKITEXT_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_gguf_magic(path: Path) -> None:
    with path.open("rb") as f:
        magic = f.read(4)
    if magic != b"GGUF":
        raise RuntimeError(f"Invalid GGUF header in {path}: {magic!r}")


def remote_file_info(url: str) -> tuple[int | None, str | None]:
    headers = {"User-Agent": "ai-infra-study/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, method="HEAD", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            cl = resp.headers.get("Content-Length")
            etag = resp.headers.get("X-Linked-ETag") or resp.headers.get("ETag")
            return (int(cl) if cl and cl.isdigit() else None), etag.strip('"') if etag else None
    except Exception:
        return None, None


def stream_download(url: str, part_path: Path, expected_size: int | None) -> None:
    curl_bin = shutil.which("curl.exe") or shutil.which("curl")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if curl_bin:
        cmd = [
            curl_bin,
            "-fL",
            "-C", "-",
            "--retry", "5",
            "--retry-delay", "3",
            "--connect-timeout", "20",
            "--speed-time", "30",
            "--speed-limit", "10240",
            "-sS",
            "-o", str(part_path),
            url,
        ]
        if token:
            cmd[1:1] = ["-H", f"Authorization: Bearer {token}"]
        proc = subprocess.Popen(cmd)
        last_report = 0.0
        while proc.poll() is None:
            time.sleep(2.0)
            now = time.monotonic()
            if now - last_report >= 8.0 and part_path.exists():
                cur = part_path.stat().st_size
                if expected_size:
                    pct = 100.0 * cur / expected_size
                    print(f"    ... {cur / (1024**2):.1f} / {expected_size / (1024**2):.1f} MiB ({pct:.1f}%)", flush=True)
                else:
                    print(f"    ... {cur / (1024**2):.1f} MiB", flush=True)
                last_report = now
        if proc.returncode != 0:
            raise RuntimeError(f"curl failed with exit code {proc.returncode} for {url}")
        return

    # Pure-Python fallback with HTTP Range resume
    existing = part_path.stat().st_size if part_path.exists() else 0
    headers = {"User-Agent": "ai-infra-study/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        mode = "ab" if resp.status == 206 else "wb"
        downloaded = existing if resp.status == 206 else 0
        last_report = time.monotonic()
        with part_path.open(mode) as f:
            while True:
                chunk = resp.read(2 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 8.0:
                    if expected_size:
                        pct = 100.0 * downloaded / expected_size
                        print(f"    ... {downloaded / (1024**2):.1f} / {expected_size / (1024**2):.1f} MiB ({pct:.1f}%)", flush=True)
                    else:
                        print(f"    ... {downloaded / (1024**2):.1f} MiB", flush=True)
                    last_report = now


def download_wikitext(data_dir: Path) -> Path:
    target = data_dir / "wikitext-2-raw" / "wiki.test.raw"
    if target.is_file() and target.stat().st_size > 100_000:
        print(f"[skip] wikitext-2 already present: {target.name} ({target.stat().st_size / 1024:.1f} KiB)", flush=True)
        return target

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "wikitext-2-raw-v1.zip"
    print(f"[download] {WIKITEXT_URL} -> {zip_path}", flush=True)
    urllib.request.urlretrieve(WIKITEXT_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    zip_path.unlink(missing_ok=True)
    if not target.is_file():
        raise RuntimeError(f"Expected {target} after extracting zip")
    print(f"[ok] extracted {target.name} ({target.stat().st_size / 1024:.1f} KiB) and removed zip", flush=True)
    return target


def download_model(spec: ModelSpec, models_dir: Path) -> Path:
    dest = models_dir / spec.filename
    url = f"https://huggingface.co/{spec.repo_id}/resolve/main/{spec.filename}"
    expected_size, _ = remote_file_info(url)

    if dest.is_file() and dest.stat().st_size > 1_000_000:
        if expected_size is None or dest.stat().st_size == expected_size:
            verify_gguf_magic(dest)
            if spec.sha256:
                digest = sha256_file(dest)
                if digest != spec.sha256:
                    raise RuntimeError(f"SHA256 mismatch for existing {dest}: {digest} != {spec.sha256}")
            print(f"[skip] {spec.filename} ({dest.stat().st_size / (1024**2):.1f} MiB, verified)", flush=True)
            return dest

    part = models_dir / f"{spec.filename}.part"
    size_label = f"{expected_size / (1024**2):.1f} MiB" if expected_size else "unknown size"
    print(f"[download] {spec.repo_id}/{spec.filename} ({size_label}) ...", flush=True)
    t0 = time.monotonic()
    stream_download(url, part, expected_size)
    if expected_size is not None and part.stat().st_size != expected_size:
        raise RuntimeError(f"Incomplete download for {dest.name}: {part.stat().st_size} != {expected_size}")
    verify_gguf_magic(part)
    if spec.sha256:
        digest = sha256_file(part)
        if digest != spec.sha256:
            raise RuntimeError(f"SHA256 mismatch for {dest.name}: {digest} != {spec.sha256}")
    if dest.exists():
        dest.unlink()
    part.replace(dest)
    dt = max(time.monotonic() - t0, 0.1)
    mb_s = (dest.stat().st_size / (1024**2)) / dt
    print(f"[ok] {dest.name} ({dest.stat().st_size / (1024**2):.1f} MiB in {dt:.1f}s, {mb_s:.1f} MiB/s)", flush=True)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="Download B3.1 models & wikitext-2 into D:")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--only",
        choices=("all", "bonsai", "qwen", "smoke"),
        default="all",
        help="Subset to download (smoke downloads only Ternary-Bonsai-2-2B-PTQ1_0)",
    )
    parser.add_argument("--skip-f16", action="store_true", help="Skip 3.6 GB F16 baseline")
    parser.add_argument("--no-wikitext", action="store_true", help="Skip wikitext-2 download")
    args = parser.parse_args()

    models_dir = args.base_dir / "models"
    data_dir = args.base_dir / "data"
    models_dir.mkdir(parents=True, exist_ok=True)
    # Remove any abandoned HF .cache leftovers
    hf_cache = models_dir / ".cache"
    if hf_cache.exists():
        shutil.rmtree(hf_cache, ignore_errors=True)

    selected = []
    for m in MODELS:
        if args.only == "bonsai" and m.family != "Ternary-Bonsai-2-2B":
            continue
        if args.only == "qwen" and m.family != "Qwen3.5-2B":
            continue
        if args.only == "smoke" and m.key != "bonsai2-2b-ptq1_0":
            continue
        if args.skip_f16 and m.quant == "F16":
            continue
        selected.append(m)

    if not args.no_wikitext:
        download_wikitext(data_dir)

    for spec in selected:
        download_model(spec, models_dir)

    print("[done] All requested assets are ready.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
