#!/usr/bin/env python3
"""Prepare N1 (unrotated) and N2 (Hadamard-rotated) F16 GGUFs from Qwen3.5-2B-F16.

Generates F16 GGUF files carrying the PrismML `prism.hadamard.*` contract
required by `PrismML-Eng/llama.cpp` (`llama-quantize PQ2_0` and runtime FWHT):

  - Mode `n1` (`--mode n1`):
      Unrotated baseline targeting the exact same 151 `PQ2_0` tensors, with
      `block_size = 1` and `sign_mode = "identity"` so weights stay unrotated
      and the runtime rotation is identity (`H_1 = [1]`).
  - Mode `n2` (`--mode n2`):
      Pure Hadamard + random +/-1 diagonal sign flip (`block_size = 512`,
      `sign_mode = "explicit"`, normalized Sylvester-Walsh-Hadamard), folded
      into all 150 projection weights (`weight_names`) and `token_embd.weight`
      (`tied_output = true`, `version = 2`).

Uses fast direct GGUF v3 streaming I/O (inspired by `gguf_dspark_to_dflash.py`)
so copying 500k tokenizer entries and 3.6 GiB of tensors completes in seconds
with bounded memory.

Usage:
    python make_bonsai_ablation.py --mode n1 -o models/qwen3.5-2b-F16-N1.gguf
    python make_bonsai_ablation.py --mode n2 -o models/qwen3.5-2b-F16-N2.gguf
"""
from __future__ import annotations

import argparse
import math
import struct
import time
from pathlib import Path

import numpy as np

GGUF_MAGIC = b"GGUF"

# GGUF v3 value types (ggml.h / gguf_dspark_to_dflash.py)
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)

_FIXED_FMT = {
    U8: "<B",
    I8: "<b",
    U16: "<H",
    I16: "<h",
    U32: "<I",
    I32: "<i",
    F32: "<f",
    BOOL: "<?",
    U64: "<Q",
    I64: "<q",
    F64: "<d",
}
_FIXED_SZ = {k: struct.calcsize(v) for k, v in _FIXED_FMT.items()}

# GGML tensor types present in F16/F32 GGUFs (ggml.h)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

_NP_DTYPE = {
    GGML_TYPE_F32: np.dtype("<f4"),
    GGML_TYPE_F16: np.dtype("<f2"),
}

# Exact set of matmul weight suffixes folded in Ternary-Bonsai-2-2B-PQ2_0.gguf
# (verified against src/llama-model.cpp `is_foldable_weight`)
FOLDABLE_SUFFIXES: tuple[str, ...] = (
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_qkv.weight",
    "attn_gate.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "ssm_out.weight",
)


def is_foldable_weight(name: str) -> bool:
    if not name.startswith("blk."):
        return False
    parts = name.split(".", 2)
    if len(parts) != 3 or not parts[1].isdigit():
        return False
    return parts[2] in FOLDABLE_SUFFIXES


def _read_bytes_str(f) -> bytes:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n)


def _read_kv_val(f, t: int):
    if t == STR:
        return _read_bytes_str(f)
    if t == ARR:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if et == STR:
            return (et, [_read_bytes_str(f) for _ in range(n)])
        sz = _FIXED_SZ[et]
        raw = f.read(sz * n)
        return (et, raw)
    fmt = _FIXED_FMT[t]
    return struct.unpack(fmt, f.read(_FIXED_SZ[t]))[0]


def read_gguf_header(path: Path) -> dict:
    """Read GGUF header, KV entries, and tensor table quickly."""
    with path.open("rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        kv = []
        for _ in range(n_kv):
            k = _read_bytes_str(f).decode("utf-8", "replace")
            t = struct.unpack("<I", f.read(4))[0]
            v = _read_kv_val(f, t)
            kv.append((k, t, v))

        tensors = []
        for _ in range(n_tensors):
            name = _read_bytes_str(f).decode("utf-8", "replace")
            nd = struct.unpack("<I", f.read(4))[0]
            dims = list(struct.unpack(f"<{nd}Q", f.read(8 * nd)))
            ttype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})

        header_end = f.tell()

    alignment = 32
    for k, _, v in kv:
        if k == "general.alignment":
            alignment = int(v)
    pad = (-header_end) % alignment
    data_start = header_end + pad
    return {
        "version": version,
        "kv": kv,
        "tensors": tensors,
        "alignment": alignment,
        "data_start": data_start,
    }


def _enc_str(b: bytes | str) -> bytes:
    raw = b.encode("utf-8") if isinstance(b, str) else b
    return struct.pack("<Q", len(raw)) + raw


def _enc_kv_val(t: int, v) -> bytes:
    if t == STR:
        return _enc_str(v)
    if t == ARR:
        et, items = v
        if et == STR:
            parts = [struct.pack("<IQ", et, len(items))]
            parts.extend(_enc_str(it) for it in items)
            return b"".join(parts)
        if isinstance(items, (bytes, bytearray)):
            n = len(items) // _FIXED_SZ[et]
            return struct.pack("<IQ", et, n) + bytes(items)
        n = len(items)
        return struct.pack("<IQ", et, n) + struct.pack("<" + _FIXED_FMT[et][1:] * n, *items)
    return struct.pack(_FIXED_FMT[t], v)


def sylvester_hadamard(block_size: int) -> np.ndarray:
    """Normalized Sylvester-Walsh-Hadamard matrix matching llama-model.cpp.

    Entry (r, c) = (-1)^popcount(r & c) / sqrt(block_size), in float32.
    Symmetric and orthogonal: H == H.T and H @ H == I.
    """
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError(f"block_size must be a power of 2, got {block_size}")
    if block_size == 1:
        return np.ones((1, 1), dtype=np.float32)

    idx = np.arange(block_size, dtype=np.uint32)
    and_mat = idx[:, None] & idx[None, :]
    if hasattr(np, "bitwise_count"):
        parity = np.bitwise_count(and_mat) & 1
    else:
        parity = np.zeros_like(and_mat, dtype=np.uint32)
        for b in range(block_size.bit_length()):
            parity ^= (and_mat >> b) & 1
    scale = np.float32(1.0 / math.sqrt(float(block_size)))
    return np.where(parity, -scale, scale).astype(np.float32)


def snap_ternary_mse(weight: np.ndarray, group_size: int = 128) -> np.ndarray:
    """Exact per-group-128 MSE-optimal ternary projection (no Hessian/QAT).

    For each group of 128 weights sorted descending by |w|, finds k in 1..128
    maximizing (sum_{j<=k} |w_{(j)}|)^2 / k, which yields optimal scale d* = mean(|w_top_k|)
    and threshold 0.5 * d*. Snaps w -> {-d*, 0, +d*} in float16 so
    llama-quantize PQ2_0 packs it with d = amax = d* losslessly.
    """
    orig_dtype = weight.dtype
    orig_shape = weight.shape
    w = weight.astype(np.float32, copy=False).reshape(-1, group_size)
    n_groups = w.shape[0]
    out = np.empty_like(w)

    ks = np.arange(1, group_size + 1, dtype=np.float32)[None, :]
    chunk_groups = 65536
    for start in range(0, n_groups, chunk_groups):
        end = min(start + chunk_groups, n_groups)
        blk = w[start:end]
        abs_blk = np.abs(blk)
        abs_sorted = np.sort(abs_blk, axis=1)[:, ::-1]
        csum = np.cumsum(abs_sorted, axis=1)
        scores = (csum * csum) / ks
        best_idx = np.argmax(scores, axis=1)
        rows = np.arange(end - start)
        d_opt = csum[rows, best_idx] / (best_idx + 1).astype(np.float32)
        # Cast d_opt to float16 first so llama-quantize PQ2_0 sees exact fp16 scale d*
        d_fp16 = d_opt.astype(np.float16).astype(np.float32)[:, None]
        inv_d = np.divide(1.0, d_fp16, out=np.zeros_like(d_fp16), where=d_fp16 > 0.0)
        q = np.clip(np.rint(blk * inv_d), -1.0, 1.0)
        out[start:end] = q * d_fp16

    return out.reshape(orig_shape).astype(orig_dtype, copy=False)


def fold_hadamard(weight: np.ndarray, signs: np.ndarray | None, H: np.ndarray) -> np.ndarray:
    """Fold W_f = (W * D) @ H along the last (input) dimension."""
    orig_dtype = weight.dtype
    block_size = H.shape[0]
    w = weight.astype(np.float32, copy=True)
    if signs is not None:
        if w.shape[-1] != signs.shape[0]:
            raise ValueError(f"Weight input dim {w.shape[-1]} != signs len {signs.shape[0]}")
        w *= signs[None, :]

    if block_size > 1:
        if w.shape[-1] % block_size != 0:
            raise ValueError(f"Input dim {w.shape[-1]} not divisible by block_size {block_size}")
        lead_shape = w.shape[:-1]
        n_blocks = w.shape[-1] // block_size
        w_blk = w.reshape(-1, block_size)
        chunk_rows = 32768
        out = np.empty_like(w_blk)
        for start in range(0, w_blk.shape[0], chunk_rows):
            end = min(start + chunk_rows, w_blk.shape[0])
            out[start:end] = w_blk[start:end] @ H
        w = out.reshape(*lead_shape, n_blocks * block_size)

    return w.astype(orig_dtype, copy=False)


def build_ablation_gguf(
    input_path: Path,
    output_path: Path,
    mode: str,
    block_size: int,
    seed: int,
    tie_embd: bool,
    snap: str = "none",
) -> None:
    t0 = time.monotonic()
    hdr = read_gguf_header(input_path)
    tensors = hdr["tensors"]
    alignment = hdr["alignment"]
    data_start = hdr["data_start"]

    has_output = any(t["name"] == "output.weight" for t in tensors)
    if tie_embd and has_output:
        raise RuntimeError("tied_output=True requires output.weight to be absent")

    weight_names = sorted(t["name"] for t in tensors if is_foldable_weight(t["name"]))
    if not weight_names:
        raise RuntimeError("No foldable matmul weights found")

    fold_set = set(weight_names)
    if tie_embd:
        if not any(t["name"] == "token_embd.weight" for t in tensors):
            raise RuntimeError("token_embd.weight not found for tied_output=True")
        fold_set.add("token_embd.weight")

    widths = sorted({int(t["dims"][0]) for t in tensors if t["name"] in fold_set})
    effective_block = 1 if mode == "n1" else block_size
    sign_mode = "identity" if mode == "n1" else "explicit"

    signs_by_width: dict[int, np.ndarray] = {}
    if mode == "n2":
        for w in widths:
            if w % effective_block != 0:
                raise ValueError(f"Foldable width {w} not divisible by block_size {effective_block}")
        rng = np.random.default_rng(seed)
        for w in widths:
            signs_by_width[w] = rng.choice(np.array([-1, 1], dtype=np.int32), size=w).astype(np.float32)

    H = sylvester_hadamard(effective_block)
    if mode == "n2":
        ident_err = float(np.max(np.abs(H @ H - np.eye(effective_block, dtype=np.float32))))
        assert ident_err < 1e-5, f"Hadamard orthogonality failed: {ident_err}"

    out_kv = [
        (k, t, v)
        for (k, t, v) in hdr["kv"]
        if not (k == "prism" or k.startswith("prism.") or k.startswith("split."))
    ]

    version = 2 if tie_embd else 1
    out_kv.extend([
        ("prism.hadamard.tied_output", BOOL, tie_embd),
        ("prism.hadamard.version", U32, version),
        ("prism.hadamard.block_size", U32, effective_block),
        ("prism.hadamard.transform", STR, b"normalized-sylvester-walsh-hadamard"),
        ("prism.hadamard.axis", STR, b"input-last-dimension"),
        ("prism.hadamard.sign_mode", STR, sign_mode.encode("ascii")),
        ("prism.hadamard.weight_names", ARR, (STR, [n.encode("utf-8") for n in weight_names])),
    ])
    if sign_mode == "explicit":
        flat_signs = []
        for w in widths:
            flat_signs.extend(int(x) for x in signs_by_width[w].astype(np.int32).tolist())
        out_kv.append(("prism.hadamard.sign_widths", ARR, (I32, [int(w) for w in widths])))
        out_kv.append(("prism.hadamard.sign_values", ARR, (I32, flat_signs)))
    if tie_embd:
        out_kv.append(("prism.hadamard.inverse_weight_names", ARR, (STR, [b"token_embd.weight"])))

    kv_chunks = []
    for k, t, v in out_kv:
        kv_chunks.append(_enc_str(k))
        kv_chunks.append(struct.pack("<I", t))
        kv_chunks.append(_enc_kv_val(t, v))
    kv_blob = b"".join(kv_chunks)

    ti_chunks = []
    rel_off = 0
    for t in tensors:
        dtype = _NP_DTYPE[t["type"]]
        n_elem = int(math.prod(t["dims"]))
        nbytes = n_elem * dtype.itemsize
        t["nbytes"] = nbytes
        t["out_offset"] = rel_off
        ti_chunks.append(_enc_str(t["name"]))
        ti_chunks.append(struct.pack("<I", len(t["dims"])))
        ti_chunks.append(struct.pack(f"<{len(t['dims'])}Q", *t["dims"]))
        ti_chunks.append(struct.pack("<IQ", t["type"], rel_off))
        rel_off += ((nbytes + alignment - 1) // alignment) * alignment
    ti_blob = b"".join(ti_chunks)

    head_blob = struct.pack("<4sIQQ", GGUF_MAGIC, hdr["version"], len(tensors), len(out_kv))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_path.with_suffix(".gguf.tmp")
    if tmp_out.exists():
        tmp_out.unlink()

    rotated = 0
    snapped = 0
    with input_path.open("rb") as fin, tmp_out.open("wb") as fout:
        fout.write(head_blob)
        fout.write(kv_blob)
        fout.write(ti_blob)
        cur_pos = fout.tell()
        pad0 = (-cur_pos) % alignment
        if pad0:
            fout.write(b"\x00" * pad0)

        for t in tensors:
            fin.seek(data_start + t["offset"])
            nbytes = t["nbytes"]
            needs_transform = t["name"] in fold_set and (mode == "n2" or snap != "none")
            if needs_transform:
                np_dtype = _NP_DTYPE[t["type"]]
                np_shape = tuple(reversed(t["dims"]))
                arr = np.fromfile(fin, dtype=np_dtype, count=int(math.prod(t["dims"]))).reshape(np_shape)
                if mode == "n2":
                    arr = fold_hadamard(arr, signs_by_width[int(t["dims"][0])], H)
                    rotated += 1
                if snap == "twn":
                    arr = snap_ternary_mse(arr, group_size=128)
                    snapped += 1
                fout.write(arr.tobytes(order="C"))
            else:
                remaining = nbytes
                while remaining > 0:
                    chunk = fin.read(min(remaining, 32 * 1024 * 1024))
                    if not chunk:
                        raise EOFError(f"Unexpected EOF reading tensor {t['name']}")
                    fout.write(chunk)
                    remaining -= len(chunk)

            tpad = (-nbytes) % alignment
            if tpad:
                fout.write(b"\x00" * tpad)

    if output_path.exists():
        output_path.unlink()
    tmp_out.replace(output_path)

    dt = time.monotonic() - t0
    mb = output_path.stat().st_size / (1024 ** 2)
    print(
        f"[ok] mode={mode} snap={snap} block_size={effective_block} sign_mode={sign_mode} tied={tie_embd} "
        f"weight_names={len(weight_names)} rotated={rotated} snapped={snapped} -> {output_path.name} ({mb:.1f} MiB in {dt:.1f}s)",
        flush=True,
    )


def main() -> int:
    _here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build N1 (unrotated) / N2 (Hadamard) F16 GGUF for PQ2_0 ablation")
    parser.add_argument("--mode", choices=("n1", "n2"), required=True, help="n1=identity (no rotation), n2=Hadamard+signs")
    parser.add_argument("-i", "--input", type=Path, default=_here / "models" / "Qwen3.5-2B-F16.gguf")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=512, help="Sylvester-Walsh-Hadamard block size for n2")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for random +/-1 signs in n2")
    parser.add_argument("--snap", choices=("none", "twn"), default="none", help="Pre-snap foldable tensors to MSE-optimal {-d,0,+d} (g128)")
    parser.add_argument("--no-tie-embd", action="store_true", help="Use version=1 without folding token_embd.weight")
    args = parser.parse_args()

    build_ablation_gguf(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        block_size=args.block_size,
        seed=args.seed,
        tie_embd=not args.no_tie_embd,
        snap=args.snap,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
