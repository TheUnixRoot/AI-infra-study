#!/usr/bin/env python3
"""N3 (Hadamard + Ternary GPTQ) for Qwen3.5-2B -> PrismML PQ2_0 on GTX 1050 CUDA.

Reads raw FP16/FP32 weights from `Qwen3.5-2B-F16.gguf`, runs exact Qwen3.5
hybrid sub-blocks (18 GatedDeltaNet + 6 GQA FullAttn + SwiGLU FFN) one layer
at a time on GTX 1050 CUDA (< 1.2 GB peak VRAM), applies true sequential
sub-block GPTQ in the N2 Sylvester-Walsh-Hadamard basis (`group_size=128`),
and writes a PQ2_0-ready F16 GGUF.

Sub-commands / flags:
  --verify-fp16   Validate PyTorch Qwen3.5 FP16 PPL against llama-perplexity (13.41)
  (default)       Run full sequential GPTQ and write N3 F16-rotated-snapped GGUF
"""
from __future__ import annotations

import argparse
import gc
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from make_bonsai_ablation import (  # noqa: E402
    ARR,
    BOOL,
    GGUF_MAGIC,
    I32,
    STR,
    U32,
    _NP_DTYPE,
    _enc_kv_val,
    _enc_str,
    is_foldable_weight,
    read_gguf_header,
    sylvester_hadamard,
)


def gguf_kv_dict(kv_list: list[tuple[str, int, object]]) -> dict[str, object]:
    return {
        k: (v.decode("utf-8", "replace") if isinstance(v, bytes) else v)
        for k, _, v in kv_list
    }


class GGUFTensorStore:
    """Lazy reader for individual tensors from Qwen3.5-2B-F16.gguf."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.hdr = read_gguf_header(path)
        self.meta = gguf_kv_dict(self.hdr["kv"])
        self.data_start = self.hdr["data_start"]
        self.by_name = {t["name"]: t for t in self.hdr["tensors"]}
        self._fp = path.open("rb")

    def close(self) -> None:
        self._fp.close()

    def get_numpy(self, name: str) -> np.ndarray:
        t = self.by_name[name]
        self._fp.seek(self.data_start + t["offset"])
        dtype = _NP_DTYPE[t["type"]]
        shape = tuple(reversed(t["dims"]))
        return np.fromfile(self._fp, dtype=dtype, count=int(math.prod(t["dims"]))).reshape(shape)

    def get_torch(self, name: str, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
        arr = self.get_numpy(name)
        ten = torch.from_numpy(np.array(arr, copy=True))
        if dtype is not None:
            ten = ten.to(dtype=dtype)
        return ten.to(device=device)


# ------------------------------------------------------------------------------
# Exact PrismML Qwen3.5 primitives (verified against qwen35.cpp + delta-net-base.cpp)
# ------------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * scale * weight.float()).to(x.dtype)


def apply_partial_neox_rope(
    x: torch.Tensor,    # [B, T, n_heads, head_dim]
    cos: torch.Tensor,  # [T, rot_dim // 2]
    sin: torch.Tensor,  # [T, rot_dim // 2]
) -> torch.Tensor:
    rot_dim = cos.shape[-1] * 2
    x_rot = x[..., :rot_dim].float()
    x_pass = x[..., rot_dim:]
    half = rot_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    y1 = x1 * c - x2 * s
    y2 = x2 * c + x1 * s
    return torch.cat([torch.cat([y1, y2], dim=-1).to(x.dtype), x_pass], dim=-1)


def delta_net_chunked_torch(
    q: torch.Tensor,     # [B, T, H, D]
    k: torch.Tensor,     # [B, T, H, D]
    v: torch.Tensor,     # [B, T, H, D]
    g: torch.Tensor,     # [B, T, H]
    beta: torch.Tensor,  # [B, T, H]
    chunk_size: int = 64,
) -> torch.Tensor:
    """Exact PyTorch equivalent of llm_build_delta_net_base::build_delta_net_chunking."""
    B, T, H, D = q.shape
    pad = (chunk_size - (T % chunk_size)) % chunk_size
    if pad:
        q = F.pad(q, (0, 0, 0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
    T_pad = q.shape[1]
    n_chunks = T_pad // chunk_size
    dev = q.device

    qf = q.float() * (1.0 / math.sqrt(D))
    kf = k.float()
    vf = v.float()
    bf = beta.float()
    gf = g.float()

    def to_chunks(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x.permute(0, 2, 1, 3).reshape(B, H, n_chunks, chunk_size, D)
        return x.permute(0, 2, 1).reshape(B, H, n_chunks, chunk_size)

    qc = to_chunks(qf)
    kc = to_chunks(kf)
    vc = to_chunks(vf)
    bc = to_chunks(bf)[..., None]
    gc = to_chunks(gf)

    v_b = vc * bc
    k_b = kc * bc

    g_cs = torch.cumsum(gc, dim=-1)
    m_tril = torch.tril(torch.ones(chunk_size, chunk_size, device=dev, dtype=torch.bool), diagonal=0)
    m_strict = torch.tril(torch.ones(chunk_size, chunk_size, device=dev, dtype=torch.bool), diagonal=-1)
    eye = torch.eye(chunk_size, device=dev, dtype=torch.float32)

    g_diff = (g_cs[..., :, None] - g_cs[..., None, :]).masked_fill(~m_tril, -1e9)
    decay_mask = torch.exp(g_diff) * m_tril

    kb_k = torch.matmul(k_b, kc.transpose(-1, -2)) * decay_mask * m_strict
    lhs = kb_k + eye
    A = torch.linalg.solve_triangular(lhs, eye.expand_as(lhs), upper=False, unitriangular=True)

    v_u = torch.matmul(A, v_b)
    k_cd = torch.matmul(A, k_b * torch.exp(g_cs)[..., None])
    kq = torch.matmul(qc, kc.transpose(-1, -2)) * decay_mask

    q_g = qc * torch.exp(g_cs)[..., None]
    g_last = g_cs[..., -1:]
    k_gdiff = kc * torch.exp(g_last - g_cs)[..., None]
    g_last_exp = torch.exp(g_last)[..., None]

    state = torch.zeros(B, H, D, D, device=dev, dtype=torch.float32)
    outs = []
    for c in range(n_chunks):
        v_prime = torch.matmul(k_cd[:, :, c], state)
        v_new = v_u[:, :, c] - v_prime
        v_attn = torch.matmul(kq[:, :, c], v_new)
        attn_inter = torch.matmul(q_g[:, :, c], state)
        outs.append(attn_inter + v_attn)
        state = state * g_last_exp[:, :, c] + torch.matmul(k_gdiff[:, :, c].transpose(-1, -2), v_new)

    out = torch.cat(outs, dim=2)
    return out[:, :, :T, :].permute(0, 2, 1, 3).contiguous()


class Qwen35Config:
    def __init__(self, meta: dict[str, object]) -> None:
        self.n_layers = int(meta["qwen35.block_count"])
        self.n_embd = int(meta["qwen35.embedding_length"])
        self.n_ff = int(meta["qwen35.feed_forward_length"])
        self.n_head = int(meta["qwen35.attention.head_count"])
        self.n_head_kv = int(meta["qwen35.attention.head_count_kv"])
        self.head_dim = int(meta["qwen35.attention.key_length"])
        self.rot_dim = int(meta["qwen35.rope.dimension_count"])
        self.freq_base = float(meta["qwen35.rope.freq_base"])
        self.rms_eps = float(meta["qwen35.attention.layer_norm_rms_epsilon"])
        self.ssm_conv = int(meta["qwen35.ssm.conv_kernel"])
        self.ssm_state = int(meta["qwen35.ssm.state_size"])
        self.ssm_groups = int(meta["qwen35.ssm.group_count"])
        self.ssm_dt_rank = int(meta["qwen35.ssm.time_step_rank"])
        self.full_attn_interval = int(meta["qwen35.full_attention_interval"])

    def is_recurrent(self, il: int) -> bool:
        return (il + 1) % self.full_attn_interval != 0


def rope_cache(max_len: int, rot_dim: int, freq_base: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (freq_base ** (torch.arange(0, rot_dim, 2, device=device, dtype=torch.float32) / rot_dim))
    pos = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def compute_attn_core(
    il: int,
    cur: torch.Tensor,  # [B, T, n_embd] on device
    cfg: Qwen35Config,
    weights: dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Compute attention/GDN pre-out-projection tensor `[B, T, n_embd]`."""
    B, T, _ = cur.shape
    dev = cur.device
    if cfg.is_recurrent(il):
        qkv_mixed = F.linear(cur, weights["attn_qkv.weight"])
        z = F.linear(cur, weights["attn_gate.weight"])
        beta = torch.sigmoid(F.linear(cur, weights["ssm_beta.weight"]))
        alpha = F.linear(cur, weights["ssm_alpha.weight"])
        g = F.softplus(alpha + weights["ssm_dt.bias"]) * weights["ssm_a"]

        qkv_p = F.pad(qkv_mixed.transpose(1, 2), (cfg.ssm_conv - 1, 0))
        conv_w = weights["ssm_conv1d.weight"][:, None, :]
        conv_silu = F.silu(F.conv1d(qkv_p, conv_w, groups=qkv_mixed.shape[-1]).transpose(1, 2))

        key_dim = cfg.ssm_state * cfg.ssm_groups
        val_dim = cfg.ssm_state * cfg.ssm_dt_rank
        q_c, k_c, v_c = torch.split(conv_silu, [key_dim, key_dim, val_dim], dim=-1)
        q_c = q_c.view(B, T, cfg.ssm_groups, cfg.ssm_state)
        k_c = k_c.view(B, T, cfg.ssm_groups, cfg.ssm_state)
        v_c = v_c.view(B, T, cfg.ssm_dt_rank, cfg.ssm_state)

        q_c = q_c / q_c.norm(dim=-1, keepdim=True).clamp(min=cfg.rms_eps)
        k_c = k_c / k_c.norm(dim=-1, keepdim=True).clamp(min=cfg.rms_eps)
        if cfg.ssm_dt_rank != cfg.ssm_groups:
            rep = cfg.ssm_dt_rank // cfg.ssm_groups
            q_c = q_c.repeat_interleave(rep, dim=2)
            k_c = k_c.repeat_interleave(rep, dim=2)

        gdn_out = delta_net_chunked_torch(q_c, k_c, v_c, g, beta, chunk_size=64)
        normed = rms_norm(gdn_out, weights["ssm_norm.weight"], cfg.rms_eps)
        z_4d = z.view(B, T, cfg.ssm_dt_rank, cfg.ssm_state)
        return (normed * F.silu(z_4d)).reshape(B, T, val_dim)

    q_full = F.linear(cur, weights["attn_q.weight"]).view(B, T, cfg.n_head, 2, cfg.head_dim)
    q = q_full[:, :, :, 0, :]
    gate = q_full[:, :, :, 1, :].reshape(B, T, cfg.n_head * cfg.head_dim)
    k = F.linear(cur, weights["attn_k.weight"]).view(B, T, cfg.n_head_kv, cfg.head_dim)
    v = F.linear(cur, weights["attn_v.weight"]).view(B, T, cfg.n_head_kv, cfg.head_dim)

    q = rms_norm(q, weights["attn_q_norm.weight"], cfg.rms_eps)
    k = rms_norm(k, weights["attn_k_norm.weight"], cfg.rms_eps)
    q = apply_partial_neox_rope(q, cos[:T], sin[:T])
    k = apply_partial_neox_rope(k, cos[:T], sin[:T])

    n_rep = cfg.n_head // cfg.n_head_kv
    k_rep = k.repeat_interleave(n_rep, dim=2).permute(0, 2, 1, 3)
    v_rep = v.repeat_interleave(n_rep, dim=2).permute(0, 2, 1, 3)
    q_p = q.permute(0, 2, 1, 3)

    scores = torch.matmul(q_p, k_rep.transpose(-1, -2)) * (1.0 / math.sqrt(cfg.head_dim))
    causal = torch.triu(torch.full((T, T), float("-inf"), device=dev), diagonal=1)
    probs = F.softmax(scores + causal, dim=-1)
    pre_gate = torch.matmul(probs, v_rep).permute(0, 2, 1, 3).reshape(B, T, cfg.n_head * cfg.head_dim)
    return pre_gate * torch.sigmoid(gate)


def forward_qwen35_layer(
    il: int,
    x_cpu: torch.Tensor,
    cfg: Qwen35Config,
    weights: dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    batch_seqs: int = 4,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    n_seqs, _, _ = x_cpu.shape
    out_cpu = torch.empty_like(x_cpu)
    out_key = "ssm_out.weight" if cfg.is_recurrent(il) else "attn_output.weight"
    with torch.no_grad():
        for s0 in range(0, n_seqs, batch_seqs):
            s1 = min(s0 + batch_seqs, n_seqs)
            inpL = x_cpu[s0:s1].to(device=device, dtype=torch.float32)
            cur = rms_norm(inpL, weights["attn_norm.weight"], cfg.rms_eps)
            attn_pre = compute_attn_core(il, cur, cfg, weights, cos, sin)
            h = inpL + F.linear(attn_pre, weights[out_key])
            cur_ffn = rms_norm(h, weights["post_attention_norm.weight"], cfg.rms_eps)
            interm = F.silu(F.linear(cur_ffn, weights["ffn_gate.weight"])) * F.linear(cur_ffn, weights["ffn_up.weight"])
            out_cpu[s0:s1] = (h + F.linear(interm, weights["ffn_down.weight"])).to(dtype=out_cpu.dtype, device="cpu")
    return out_cpu


def load_layer_weights(store: GGUFTensorStore, il: int, cfg: Qwen35Config, device: torch.device) -> dict[str, torch.Tensor]:
    prefix = f"blk.{il}."
    names = [
        "attn_norm.weight",
        "post_attention_norm.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ]
    if cfg.is_recurrent(il):
        names.extend([
            "attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight",
            "ssm_dt.bias", "ssm_a", "ssm_conv1d.weight", "ssm_norm.weight", "ssm_out.weight",
        ])
    else:
        names.extend([
            "attn_q.weight", "attn_k.weight", "attn_v.weight",
            "attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight",
        ])
    return {n: store.get_torch(prefix + n, device=device, dtype=torch.float32) for n in names}


def eval_fp16_ppl(store: GGUFTensorStore, tokens_path: Path, n_chunks: int = 8, ctx: int = 512) -> float:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Qwen35Config(store.meta)
    toks = np.fromfile(tokens_path, dtype=np.int32)[: n_chunks * ctx].reshape(n_chunks, ctx)

    tok_emb = store.get_torch("token_embd.weight", device=torch.device("cpu"), dtype=torch.float16)
    x_cpu = tok_emb[torch.from_numpy(toks).long()].to(torch.float32)
    cos, sin = rope_cache(ctx, cfg.rot_dim, cfg.freq_base, dev)

    t0 = time.monotonic()
    for il in range(cfg.n_layers):
        w = load_layer_weights(store, il, cfg, dev)
        x_cpu = forward_qwen35_layer(il, x_cpu, cfg, w, cos, sin, batch_seqs=4, device=dev)
        del w
        torch.cuda.empty_cache()

    out_norm = store.get_torch("output_norm.weight", device=dev, dtype=torch.float32)
    nll = 0.0
    count = 0
    with torch.no_grad():
        for c in range(n_chunks):
            h = rms_norm(x_cpu[c].to(dev), out_norm, cfg.rms_eps)
            h_eval = h[ctx // 2 - 1 : ctx - 1]
            targets = torch.from_numpy(toks[c, ctx // 2 : ctx]).long().to(dev)
            vocab_n = tok_emb.shape[0]
            logits_parts = []
            for v0 in range(0, vocab_n, 32768):
                v1 = min(v0 + 32768, vocab_n)
                logits_parts.append(h_eval @ tok_emb[v0:v1].to(dev, dtype=torch.float32).T)
            logits = torch.cat(logits_parts, dim=-1)
            nll += float(F.cross_entropy(logits, targets, reduction="sum").item())
            count += int(targets.numel())
            del logits, logits_parts

    ppl = math.exp(nll / count)
    print(f"[verify-fp16] chunks={n_chunks} count={count} PPL={ppl:.4f} (in {time.monotonic() - t0:.1f}s)", flush=True)
    return ppl


# ------------------------------------------------------------------------------
# Vectorized MSE-optimal TWN ternary grid + Block-128 GPTQ Solver
# ------------------------------------------------------------------------------

def snap_twn_block_torch(w_blk: torch.Tensor) -> torch.Tensor:
    """Exact per-row MSE-optimal ternary projection for one `[M, 128]` block."""
    M, G = w_blk.shape
    ks = torch.arange(1, G + 1, device=w_blk.device, dtype=torch.float32)[None, :]
    abs_sorted, _ = torch.sort(torch.abs(w_blk), dim=1, descending=True)
    csum = torch.cumsum(abs_sorted, dim=1)
    scores = (csum * csum) / ks
    best_idx = torch.argmax(scores, dim=1)
    rows = torch.arange(M, device=w_blk.device)
    d_opt = csum[rows, best_idx] / (best_idx + 1).to(torch.float32)
    d_fp16 = d_opt.to(torch.float16).to(torch.float32)[:, None]
    inv_d = torch.where(d_fp16 > 0.0, 1.0 / d_fp16, torch.zeros_like(d_fp16))
    q = torch.clamp(torch.round(w_blk * inv_d), -1.0, 1.0)
    return q * d_fp16


def prepare_hessian_cholesky(
    H_raw: torch.Tensor,
    n_tokens: int,
    damp_pct: float = 0.01,
    perm: torch.Tensor | None = None,
    keep_h: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute Cholesky inverse upper factor U (and optionally normalized primal H)."""
    K = H_raw.shape[0]
    dev = H_raw.device
    H = (H_raw * (2.0 / float(n_tokens))).clone()

    diag_idx = torch.arange(K, device=dev)
    dead = H[diag_idx, diag_idx] <= 0
    if dead.any():
        H[dead, dead] = 1.0

    H = 0.5 * (H + H.T)
    H_keep = H.clone() if keep_h else None
    if perm is not None:
        H = H[perm][:, perm]
        dead_out = dead[perm]
    else:
        dead_out = dead

    base_diag_mean = torch.mean(torch.diag(H)).clamp(min=1e-6)
    Hinv_upper = None
    for mult in (1.0, 2.0, 5.0, 10.0, 50.0, 200.0):
        try:
            H_d = H.clone()
            H_d[diag_idx, diag_idx] += (damp_pct * mult) * base_diag_mean
            L = torch.linalg.cholesky(H_d)
            del H_d
            Hinv = torch.cholesky_inverse(L)
            del L
            Hinv = 0.5 * (Hinv + Hinv.T)
            Hinv_upper = torch.linalg.cholesky(Hinv, upper=True)
            del Hinv
            break
        except RuntimeError:
            continue

    del H
    if Hinv_upper is None:
        raise RuntimeError(f"Cholesky failed even at 200x damping for shape ({K}, {K})")
    return Hinv_upper, dead_out, H_keep


def solve_asymmetric_target(
    H_qq: torch.Tensor,  # [K, K] = X_q^T @ X_q
    H_qf: torch.Tensor,  # [K, K] = X_q^T @ X_fp16
    ridge_pct: float = 0.08,
) -> torch.Tensor:
    """Compute right-projection matrix M^T so W_target = W_orig @ M^T minimizes
    || X_q W_target^T - X_f W_orig^T ||_F^2 + lam || W_target - W_orig ||_F^2.
    """
    H_reg = 0.5 * (H_qq + H_qq.T)
    lam = float(torch.mean(torch.diag(H_reg)).clamp(min=1e-6)) * ridge_pct
    H_reg.diagonal().add_(lam)
    rhs = H_qf.clone()
    rhs.diagonal().add_(lam)
    M = torch.linalg.solve(H_reg, rhs)
    del H_reg, rhs
    return M.T.contiguous()


def precompute_static_scales_n3a(
    W_blk_rows: torch.Tensor,  # [M, K] float32
    diag_H: torch.Tensor,      # [K] float32
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hessian-weighted per-group-128 scale grid search around TWN optimal d*."""
    M, K = W_blk_rows.shape
    G = K // group_size
    W_g = W_blk_rows.view(M, G, group_size)
    h_g = diag_H.clamp(min=0.0).view(1, G, group_size)

    d_twn = torch.max(torch.abs(snap_twn_block_torch(W_blk_rows.reshape(-1, group_size))), dim=1).values.view(M, G)
    best_d = d_twn.clone()
    best_err = torch.full((M, G), float("inf"), device=W_blk_rows.device, dtype=torch.float32)

    alphas = (0.72, 0.78, 0.84, 0.88, 0.92, 0.96, 1.00, 1.05, 1.12, 1.20)
    for alpha in alphas:
        d_cand = (d_twn * alpha).to(torch.float16).to(torch.float32)
        d3 = d_cand[:, :, None]
        inv_d = torch.where(d3 > 0.0, 1.0 / d3, torch.zeros_like(d3))
        q = torch.clamp(torch.round(W_g * inv_d), -1.0, 1.0) * d3
        err = ((W_g - q).pow(2) * h_g).sum(dim=-1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_d = torch.where(better, d_cand, best_d)

    inv_best_d = torch.where(best_d > 0.0, 1.0 / best_d, torch.zeros_like(best_d))
    return best_d, inv_best_d


def refine_global_scales_n3a(
    W_ref: torch.Tensor,   # [M, K] float32 on device
    H_full: torch.Tensor,  # [K, K] float32 symmetric PD on device
    Q_phys: torch.Tensor,  # [M, K] float32 on device
    group_size: int = 128,
    n_iters: int = 3,
) -> torch.Tensor:
    """Monotonic coordinate refinement of fp16 block scales minimizing tr((W_ref - W_hat) H (W_ref - W_hat)^T)."""
    M, K = W_ref.shape
    G = K // group_size
    Q_g = Q_phys.view(M, G, group_size)
    d_g = torch.max(torch.abs(Q_g), dim=2).values.to(torch.float16).to(torch.float32)
    inv_d = torch.where(d_g > 0.0, 1.0 / d_g, torch.zeros_like(d_g))
    T_g = torch.clamp(torch.round(Q_g * inv_d[:, :, None]), -1.0, 1.0)

    W_hat = (T_g * d_g[:, :, None]).view(M, K).clone()
    R = W_ref - W_hat
    G_mat = R @ H_full

    for _ in range(n_iters):
        for g in range(G):
            c0 = g * group_size
            c1 = c0 + group_size
            t_blk = T_g[:, g, :]
            active = t_blk.abs().sum(dim=1) > 0
            H_gg = H_full[c0:c1, c0:c1]
            den = ((t_blk @ H_gg) * t_blk).sum(dim=1)
            num = (G_mat[:, c0:c1] * t_blk).sum(dim=1)
            valid = active & (den > 1e-12)
            delta = torch.where(valid, num / den.clamp(min=1e-12), torch.zeros_like(num))
            d_new = torch.clamp(d_g[:, g] + delta, min=1e-7).to(torch.float16).to(torch.float32)
            step = d_new - d_g[:, g]
            # Exact quadratic objective change in tr(R H R^T) after fp16 rounding of d_new
            delta_loss = -2.0 * step * num + (step * step) * den
            accept = valid & (step != 0.0) & (delta_loss < -1e-12)
            if accept.any():
                actual = torch.where(accept, step, torch.zeros_like(step))
                d_g[:, g] += actual
                diff_blk = actual[:, None] * t_blk
                W_hat[:, c0:c1] += diff_blk
                R[:, c0:c1] -= diff_blk
                G_mat -= diff_blk @ H_full[c0:c1, :]

    return (T_g * d_g[:, :, None]).view(M, K)


def gptq_ternary_weight(
    W_rot: torch.Tensor,    # [M, K] float32 on device
    H_raw: torch.Tensor,    # [K, K] float32 on device
    n_tokens: int,
    group_size: int = 128,
    damp_pct: float = 0.01,
    row_chunk: int = 16384,
) -> torch.Tensor:
    """Block-128 GPTQ over Hadamard-rotated weights onto {-d_fp16, 0, +d_fp16}."""
    M, K = W_rot.shape
    Hinv_upper, dead, _ = prepare_hessian_cholesky(H_raw, n_tokens, damp_pct=damp_pct)

    out_f16 = torch.empty((M, K), dtype=torch.float16, device="cpu")
    for r0 in range(0, M, row_chunk):
        r1 = min(r0 + row_chunk, M)
        W = W_rot[r0:r1].clone().float()
        if dead.any():
            W[:, dead] = 0.0
        Q = torch.zeros_like(W)

        for i1 in range(0, K, group_size):
            i2 = min(i1 + group_size, K)
            cnt = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv_upper[i1:i2, i1:i2]

            Q_init = snap_twn_block_torch(W1)
            d_blk = torch.max(torch.abs(Q_init), dim=1, keepdim=True).values
            inv_d = torch.where(d_blk > 0.0, 1.0 / d_blk, torch.zeros_like(d_blk))

            for j in range(cnt):
                w = W1[:, j]
                d_inv_jj = Hinv1[j, j]
                q = torch.clamp(torch.round(w * inv_d[:, 0]), -1.0, 1.0) * d_blk[:, 0]
                Q1[:, j] = q
                err = (w - q) / d_inv_jj
                W1[:, j:] -= err[:, None] * Hinv1[j, j:][None, :]
                Err1[:, j] = err

            Q[:, i1:i2] = Q1
            if i2 < K:
                W[:, i2:] -= Err1 @ Hinv_upper[i1:i2, i2:]

        out_f16[r0:r1] = Q.to(dtype=torch.float16, device="cpu")
        del W, Q

    del Hinv_upper
    return out_f16


def gptq_ternary_weight_n3a(
    W_rot: torch.Tensor,          # [M, K] float32 on device (or CPU float16/float32 for streaming embd)
    H_raw: torch.Tensor,          # [K, K] float32 on device
    n_tokens: int,
    group_size: int = 128,
    damp_pct: float = 0.01,
    row_chunk: int = 8192,
    actorder: bool = False,
    refine_scales: bool = False,
    n_refine_iters: int = 3,
) -> torch.Tensor:
    """N3A GPTQ solver supporting Hessian-ordered static groups + monotonic WLS scale refinement."""
    M, K = W_rot.shape
    dev = H_raw.device
    H_norm = (H_raw * (2.0 / float(n_tokens)))
    diag_H = torch.diag(H_norm).clamp(min=1e-8)

    if actorder:
        perm = torch.argsort(diag_H, descending=True)
        inv_perm = torch.argsort(perm)
        g_idx_perm = (perm // group_size).tolist()
    else:
        perm = None
        inv_perm = None
        g_idx_perm = None

    Hinv_upper, dead_perm, H_full = prepare_hessian_cholesky(
        H_raw, n_tokens, damp_pct=damp_pct, perm=perm, keep_h=refine_scales
    )

    out_f16 = torch.empty((M, K), dtype=torch.float16, device="cpu")
    for r0 in range(0, M, row_chunk):
        r1 = min(r0 + row_chunk, M)
        W_orig_chunk = W_rot[r0:r1].to(device=dev, dtype=torch.float32).clone()

        if actorder:
            d_static, inv_d_static = precompute_static_scales_n3a(W_orig_chunk, diag_H, group_size=group_size)
            W = W_orig_chunk[:, perm].clone()
            if dead_perm.any():
                W[:, dead_perm] = 0.0
            Q_p = torch.zeros_like(W)
            for i1 in range(0, K, group_size):
                i2 = min(i1 + group_size, K)
                cnt = i2 - i1
                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Hinv1 = Hinv_upper[i1:i2, i1:i2]
                for j in range(cnt):
                    g = g_idx_perm[i1 + j]
                    w = W1[:, j]
                    d_col = d_static[:, g]
                    q = torch.clamp(torch.round(w * inv_d_static[:, g]), -1.0, 1.0) * d_col
                    Q1[:, j] = q
                    err = (w - q) / Hinv1[j, j]
                    W1[:, j:] -= err[:, None] * Hinv1[j, j:][None, :]
                    Err1[:, j] = err
                Q_p[:, i1:i2] = Q1
                if i2 < K:
                    W[:, i2:] -= Err1 @ Hinv_upper[i1:i2, i2:]
            Q_phys = Q_p[:, inv_perm]
            del W, Q_p, d_static, inv_d_static
        else:
            W = W_orig_chunk.clone()
            if dead_perm.any():
                W[:, dead_perm] = 0.0
            Q_phys = torch.zeros_like(W)
            for i1 in range(0, K, group_size):
                i2 = min(i1 + group_size, K)
                cnt = i2 - i1
                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Hinv1 = Hinv_upper[i1:i2, i1:i2]
                Q_init = snap_twn_block_torch(W1)
                d_blk = torch.max(torch.abs(Q_init), dim=1, keepdim=True).values
                inv_d = torch.where(d_blk > 0.0, 1.0 / d_blk, torch.zeros_like(d_blk))
                for j in range(cnt):
                    w = W1[:, j]
                    q = torch.clamp(torch.round(w * inv_d[:, 0]), -1.0, 1.0) * d_blk[:, 0]
                    Q1[:, j] = q
                    err = (w - q) / Hinv1[j, j]
                    W1[:, j:] -= err[:, None] * Hinv1[j, j:][None, :]
                    Err1[:, j] = err
                Q_phys[:, i1:i2] = Q1
                if i2 < K:
                    W[:, i2:] -= Err1 @ Hinv_upper[i1:i2, i2:]
            del W

        if refine_scales and H_full is not None:
            Q_phys = refine_global_scales_n3a(W_orig_chunk, H_full, Q_phys, group_size=group_size, n_iters=n_refine_iters)

        out_f16[r0:r1] = Q_phys.to(dtype=torch.float16, device="cpu")
        del W_orig_chunk, Q_phys

    del Hinv_upper, H_full
    return out_f16


def transform_x_to_had(X_raw: torch.Tensor, signs: torch.Tensor, H_had: torch.Tensor) -> torch.Tensor:
    """Transform primal covariance X^T X -> H_had D (X^T X) D H_had."""
    width = X_raw.shape[0]
    B = H_had.shape[0]
    nb = width // B
    X_s = X_raw * signs[:, None] * signs[None, :]
    tmp = torch.matmul(H_had, X_s.view(nb, B, width)).view(width, width)
    del X_s
    res = torch.matmul(tmp.view(width, nb, B), H_had).view(width, width)
    del tmp
    return 0.5 * (res + res.T)


def self_tests() -> None:
    """Verify Cholesky GPTQ, Hadamard covariance, asymmetric ridge, and N3A scale refinement."""
    torch.manual_seed(0)
    X = torch.randn(64, 8)
    H = (X.T @ X) + 0.05 * torch.eye(8)
    W = torch.randn(3, 8)

    L = torch.linalg.cholesky(H)
    U = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    Wc = W.clone()
    Qc = torch.zeros_like(Wc)
    for j in range(8):
        qj = torch.round(Wc[:, j])
        Qc[:, j] = qj
        Wc[:, j:] -= ((Wc[:, j] - qj) / U[j, j])[:, None] * U[j, j:][None, :]

    W_obs = W.clone()
    for j in range(8):
        qj = Qc[:, j]
        S = torch.linalg.inv(H[j:, j:])
        W_obs[:, j:] -= ((W_obs[:, j] - qj) / S[0, 0])[:, None] * S[0, :][None, :]
        W_obs[:, j] = qj

    chol_err = float((Wc - W_obs).abs().max())
    assert chol_err < 1e-4, f"GPTQ Cholesky test failed: {chol_err}"

    # Verify Hadamard covariance rotation matches explicit rotated X_rot^T X_rot
    B = 4
    width = 8
    H_had = torch.from_numpy(sylvester_hadamard(B))
    signs = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
    X_big = torch.randn(32, width)
    X_rot = ((X_big * signs[None, :]).view(32, 2, B) @ H_had).view(32, width)
    cov_fast = transform_x_to_had(X_big.T @ X_big, signs, H_had)
    cov_ref = X_rot.T @ X_rot
    had_err = float((cov_fast - cov_ref).abs().max())
    assert had_err < 1e-4, f"Hadamard covariance test failed: {had_err}"

    # Verify asymmetric ridge: exact identity when X_q == X_f, strictly better match when X_q != X_f
    M_id = solve_asymmetric_target(cov_ref, cov_ref, ridge_pct=0.08)
    ridge_id_err = float((W @ M_id - W).abs().max())
    assert ridge_id_err < 1e-4, f"Asymmetric ridge identity failed: {ridge_id_err}"
    X_q = X_big + 0.15 * torch.randn_like(X_big)
    M_asym = solve_asymmetric_target(X_q.T @ X_q, X_q.T @ X_big, ridge_pct=0.05)
    W_asym = W @ M_asym
    err_orig = float(((X_q @ W.T) - (X_big @ W.T)).pow(2).sum())
    err_asym = float(((X_q @ W_asym.T) - (X_big @ W.T)).pow(2).sum())
    assert err_asym < err_orig, f"Asymmetric target did not reduce distortion: {err_asym} >= {err_orig}"

    # Verify N3A solver (actorder + static-groups + global monotonic WLS refinement + exact PQ2_0 grid)
    X_n3a = torch.randn(256, 256)
    X_n3a *= torch.linspace(0.5, 2.0, 256)[None, :]
    W_n3a = torch.randn(16, 256) * 0.05
    H_n3a = X_n3a.T @ X_n3a
    Q_n3a = gptq_ternary_weight_n3a(
        W_n3a, H_n3a, 256, group_size=128, damp_pct=0.01, actorder=True, refine_scales=True
    ).float()
    Q_twn = snap_twn_block_torch(W_n3a.reshape(-1, 128)).reshape(16, 256)
    loss_n3a = float(((W_n3a - Q_n3a) @ H_n3a @ (W_n3a - Q_n3a).T).trace())
    loss_twn = float(((W_n3a - Q_twn) @ H_n3a @ (W_n3a - Q_twn).T).trace())
    assert loss_n3a < loss_twn, f"N3A loss did not beat TWN: {loss_n3a} >= {loss_twn}"
    # Check exact PQ2_0 zero-double-rounding invariant on every 128-group
    Q_g = Q_n3a.view(-1, 128)
    d_max = Q_g.abs().max(dim=1, keepdim=True).values
    ratios = torch.where(d_max > 0, Q_g / d_max, torch.zeros_like(Q_g))
    assert float((ratios - ratios.round()).abs().max()) < 1e-3, "N3A output escaped fp16 ternary PQ2_0 grid"

    print(
        f"[self-test] Cholesky={chol_err:.2e}, HadCov={had_err:.2e}, RidgeId={ridge_id_err:.2e}, "
        f"N3A_loss={loss_n3a:.3f} < TWN={loss_twn:.3f} OK",
        flush=True,
    )


def build_calib_batch(
    calib_path: Path,
    n_chunks: int,
    ctx: int,
    seed: int,
    chat_path: Path | None = None,
    chat_frac: float = 0.40,
) -> np.ndarray:
    """Sample non-overlapping prose + optional Qwen3.5 chat/think calibration chunks."""
    all_toks = np.fromfile(calib_path, dtype=np.int32)
    total_avail = len(all_toks) // ctx
    mat_main = all_toks[: total_avail * ctx].reshape(total_avail, ctx)

    if chat_path is not None and chat_path.is_file():
        chat_toks = np.fromfile(chat_path, dtype=np.int32)
        total_chat = len(chat_toks) // ctx
        n_chat = max(1, int(round(n_chunks * chat_frac)))
        n_prose = n_chunks - n_chat
        if total_avail < n_prose or total_chat < n_chat:
            raise RuntimeError(
                f"Insufficient chunks: prose={total_avail}/{n_prose}, chat={total_chat}/{n_chat}"
            )
        p_idx = np.linspace(0, total_avail - 1, num=n_prose, dtype=np.int64)
        c_idx = np.linspace(0, total_chat - 1, num=n_chat, dtype=np.int64)
        batch = np.concatenate([mat_main[p_idx], chat_toks[: total_chat * ctx].reshape(total_chat, ctx)[c_idx]], axis=0)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(batch))
        return batch[perm]

    if total_avail < n_chunks:
        raise RuntimeError(f"Only {total_avail} chunks available in {calib_path}, need {n_chunks}")

    # First ~87% of calib_mix.txt is WikiText-2 tail (lines 1500+); last ~13% is Python/JSON code.
    split_code = int(total_avail * 0.87)
    n_code = max(4, n_chunks // 8)
    n_wiki = n_chunks - n_code

    wiki_idx = np.linspace(0, split_code - 1, num=n_wiki, dtype=np.int64)
    code_idx = np.linspace(split_code, total_avail - 1, num=n_code, dtype=np.int64)
    chosen = np.concatenate([wiki_idx, code_idx])
    rng = np.random.default_rng(seed)
    rng.shuffle(chosen)
    return mat_main[chosen]


def run_gptq_pipeline(args: argparse.Namespace) -> None:
    self_tests()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GGUFTensorStore(args.input)
    try:
        if args.verify_fp16:
            eval_fp16_ppl(store, args.val_tokens, n_chunks=args.val_chunks, ctx=args.ctx)
            return

        t0 = time.monotonic()
        cfg = Qwen35Config(store.meta)
        calib_toks = build_calib_batch(args.calib_tokens, args.calib_chunks, args.ctx, args.seed)
        total_tok = args.calib_chunks * args.ctx

        weight_names = sorted(t["name"] for t in store.hdr["tensors"] if is_foldable_weight(t["name"]))
        fold_set = set(weight_names) | {"token_embd.weight"}
        widths = sorted({int(t["dims"][0]) for t in store.hdr["tensors"] if t["name"] in fold_set})

        rng = np.random.default_rng(args.seed)
        signs_np = {
            w: rng.choice(np.array([-1, 1], dtype=np.int32), size=w).astype(np.float32)
            for w in widths
        }
        signs_gpu = {w: torch.from_numpy(s).to(dev) for w, s in signs_np.items()}
        H_had_gpu = torch.from_numpy(sylvester_hadamard(args.block_size)).to(dev)

        def fold_w_gpu(W: torch.Tensor, width: int) -> torch.Tensor:
            B = args.block_size
            Ws = W.float() * signs_gpu[width][None, :]
            return (Ws.view(-1, B) @ H_had_gpu).view_as(Ws)

        def unfold_w_gpu(W_rot: torch.Tensor, width: int) -> torch.Tensor:
            B = args.block_size
            return (W_rot.float().view(-1, B) @ H_had_gpu).view_as(W_rot) * signs_gpu[width][None, :]

        cos, sin = rope_cache(args.ctx, cfg.rot_dim, cfg.freq_base, dev)

        tok_emb_np = store.get_numpy("token_embd.weight")
        quantized_rot_tensors: dict[str, np.ndarray] = {}

        if args.embd_snap == "twn_first":
            print("[embd] Pre-snapping token_embd.weight with TWN g128 before layer 0...", flush=True)
            tok_rot = np.empty_like(tok_emb_np)
            tok_unrot = np.empty_like(tok_emb_np)
            for r0 in range(0, tok_emb_np.shape[0], 16384):
                r1 = min(r0 + 16384, tok_emb_np.shape[0])
                w_g = torch.from_numpy(tok_emb_np[r0:r1].astype(np.float32)).to(dev)
                w_q = snap_twn_block_torch(fold_w_gpu(w_g, cfg.n_embd).view(-1, 128)).view(-1, cfg.n_embd)
                tok_rot[r0:r1] = w_q.cpu().numpy().astype(np.float16)
                tok_unrot[r0:r1] = unfold_w_gpu(w_q, cfg.n_embd).cpu().numpy().astype(np.float16)
            quantized_rot_tensors["token_embd.weight"] = tok_rot
            tok_for_input = torch.from_numpy(tok_unrot)
            del tok_unrot
        else:
            tok_for_input = torch.from_numpy(np.array(tok_emb_np, copy=True))

        # Store calibration hidden states as float16 on CPU (~128 MiB for 64x512x2048)
        x_cpu = tok_for_input[torch.from_numpy(calib_toks).long()].to(torch.float16)
        del tok_for_input

        print(
            f"=== [gptq] Starting 24-layer sub-block sequential GPTQ "
            f"(chunks={args.calib_chunks}, tokens={total_tok}, damp={args.damp}, embd={args.embd_snap}) ===",
            flush=True,
        )

        for il in range(cfg.n_layers):
            lt0 = time.monotonic()
            w = load_layer_weights(store, il, cfg, dev)
            is_recr = cfg.is_recurrent(il)

            # ---- Sub-block A1: attention/GDN input projections ----
            H_in = torch.zeros((cfg.n_embd, cfg.n_embd), device=dev, dtype=torch.float32)
            with torch.no_grad():
                for s0 in range(0, args.calib_chunks, args.batch_seqs):
                    s1 = min(s0 + args.batch_seqs, args.calib_chunks)
                    inpL = x_cpu[s0:s1].to(dev, dtype=torch.float32)
                    cur2d = rms_norm(inpL, w["attn_norm.weight"], cfg.rms_eps).reshape(-1, cfg.n_embd)
                    H_in.add_(cur2d.T @ cur2d)
            H_in_rot = transform_x_to_had(H_in, signs_gpu[cfg.n_embd], H_had_gpu)
            del H_in

            in_names = ["attn_qkv.weight", "attn_gate.weight"] if is_recr else ["attn_q.weight", "attn_k.weight", "attn_v.weight"]
            for wn in in_names:
                Q_f16 = gptq_ternary_weight(fold_w_gpu(w[wn], cfg.n_embd), H_in_rot, total_tok, damp_pct=args.damp)
                quantized_rot_tensors[f"blk.{il}.{wn}"] = Q_f16.numpy()
                w[wn] = unfold_w_gpu(Q_f16.to(dev), cfg.n_embd)
            del H_in_rot

            # ---- Sub-block A2: ssm_out / attn_output (using QUANTIZED QKV/gate) ----
            out_wn = "ssm_out.weight" if is_recr else "attn_output.weight"
            H_ao = torch.zeros((cfg.n_embd, cfg.n_embd), device=dev, dtype=torch.float32)
            with torch.no_grad():
                for s0 in range(0, args.calib_chunks, args.batch_seqs):
                    s1 = min(s0 + args.batch_seqs, args.calib_chunks)
                    inpL = x_cpu[s0:s1].to(dev, dtype=torch.float32)
                    cur = rms_norm(inpL, w["attn_norm.weight"], cfg.rms_eps)
                    ao2d = compute_attn_core(il, cur, cfg, w, cos, sin).reshape(-1, cfg.n_embd)
                    H_ao.add_(ao2d.T @ ao2d)
            H_ao_rot = transform_x_to_had(H_ao, signs_gpu[cfg.n_embd], H_had_gpu)
            del H_ao

            Q_ao = gptq_ternary_weight(fold_w_gpu(w[out_wn], cfg.n_embd), H_ao_rot, total_tok, damp_pct=args.damp)
            quantized_rot_tensors[f"blk.{il}.{out_wn}"] = Q_ao.numpy()
            w[out_wn] = unfold_w_gpu(Q_ao.to(dev), cfg.n_embd)
            del H_ao_rot

            # Compute post-attention hidden state `h_cpu` with QUANTIZED attention
            h_cpu = torch.empty_like(x_cpu)
            H_fg = torch.zeros((cfg.n_embd, cfg.n_embd), device=dev, dtype=torch.float32)
            with torch.no_grad():
                for s0 in range(0, args.calib_chunks, args.batch_seqs):
                    s1 = min(s0 + args.batch_seqs, args.calib_chunks)
                    inpL = x_cpu[s0:s1].to(dev, dtype=torch.float32)
                    cur = rms_norm(inpL, w["attn_norm.weight"], cfg.rms_eps)
                    h_b = inpL + F.linear(compute_attn_core(il, cur, cfg, w, cos, sin), w[out_wn])
                    h_cpu[s0:s1] = h_b.to(dtype=h_cpu.dtype, device="cpu")
                    ffn_in = rms_norm(h_b, w["post_attention_norm.weight"], cfg.rms_eps).reshape(-1, cfg.n_embd)
                    H_fg.add_(ffn_in.T @ ffn_in)

            # ---- Sub-block B1: ffn_gate + ffn_up ----
            H_fg_rot = transform_x_to_had(H_fg, signs_gpu[cfg.n_embd], H_had_gpu)
            del H_fg
            for wn in ("ffn_gate.weight", "ffn_up.weight"):
                Q_f16 = gptq_ternary_weight(fold_w_gpu(w[wn], cfg.n_embd), H_fg_rot, total_tok, damp_pct=args.damp)
                quantized_rot_tensors[f"blk.{il}.{wn}"] = Q_f16.numpy()
                w[wn] = unfold_w_gpu(Q_f16.to(dev), cfg.n_embd)
            del H_fg_rot

            # ---- Sub-block B2: ffn_down (using QUANTIZED gate & up) ----
            # Free unneeded attention weights before allocating 6144x6144 Hessian on 2 GB VRAM
            for k in list(w.keys()):
                if k not in ("post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
                    del w[k]
            torch.cuda.empty_cache()

            H_fd = torch.zeros((cfg.n_ff, cfg.n_ff), device=dev, dtype=torch.float32)
            with torch.no_grad():
                for s0 in range(0, args.calib_chunks, args.batch_seqs):
                    s1 = min(s0 + args.batch_seqs, args.calib_chunks)
                    h_b = h_cpu[s0:s1].to(dev, dtype=torch.float32)
                    cur_ffn = rms_norm(h_b, w["post_attention_norm.weight"], cfg.rms_eps)
                    interm = (F.silu(F.linear(cur_ffn, w["ffn_gate.weight"])) * F.linear(cur_ffn, w["ffn_up.weight"])).reshape(-1, cfg.n_ff)
                    H_fd.add_(interm.T @ interm)
            H_fd_rot = transform_x_to_had(H_fd, signs_gpu[cfg.n_ff], H_had_gpu)
            del H_fd
            torch.cuda.empty_cache()

            Q_fd = gptq_ternary_weight(fold_w_gpu(w["ffn_down.weight"], cfg.n_ff), H_fd_rot, total_tok, damp_pct=args.damp)
            quantized_rot_tensors[f"blk.{il}.ffn_down.weight"] = Q_fd.numpy()
            w["ffn_down.weight"] = unfold_w_gpu(Q_fd.to(dev), cfg.n_ff)
            del H_fd_rot

            # Advance x_cpu using quantized FFN
            with torch.no_grad():
                for s0 in range(0, args.calib_chunks, args.batch_seqs):
                    s1 = min(s0 + args.batch_seqs, args.calib_chunks)
                    h_b = h_cpu[s0:s1].to(dev, dtype=torch.float32)
                    cur_ffn = rms_norm(h_b, w["post_attention_norm.weight"], cfg.rms_eps)
                    interm = F.silu(F.linear(cur_ffn, w["ffn_gate.weight"])) * F.linear(cur_ffn, w["ffn_up.weight"])
                    x_cpu[s0:s1] = (h_b + F.linear(interm, w["ffn_down.weight"])).to(dtype=x_cpu.dtype, device="cpu")

            del w, h_cpu
            torch.cuda.empty_cache()
            gc.collect()
            print(f"  layer {il:2d}/{cfg.n_layers - 1} ({'GDN' if is_recr else 'ATT'}) done in {time.monotonic() - lt0:.1f}s", flush=True)

        # ---- Tied token_embd.weight via final output_norm Hessian (when gptq_head) ----
        if args.embd_snap == "gptq_head":
            print("[embd] Quantizing tied token_embd.weight with GPTQ against output_norm Hessian...", flush=True)
            out_norm = store.get_torch("output_norm.weight", device=dev, dtype=torch.float32)
            H_out = torch.zeros((cfg.n_embd, cfg.n_embd), device=dev, dtype=torch.float32)
            with torch.no_grad():
                for s0 in range(0, args.calib_chunks, args.batch_seqs):
                    s1 = min(s0 + args.batch_seqs, args.calib_chunks)
                    h_b = rms_norm(x_cpu[s0:s1].to(dev, dtype=torch.float32), out_norm, cfg.rms_eps).reshape(-1, cfg.n_embd)
                    H_out.add_(h_b.T @ h_b)
            H_out_rot = transform_x_to_had(H_out, signs_gpu[cfg.n_embd], H_had_gpu)
            del H_out, out_norm

            tok_rot = np.empty_like(tok_emb_np)
            for v0 in range(0, tok_emb_np.shape[0], 16384):
                v1 = min(v0 + 16384, tok_emb_np.shape[0])
                w_g = torch.from_numpy(tok_emb_np[v0:v1].astype(np.float32)).to(dev)
                Q_f16 = gptq_ternary_weight(fold_w_gpu(w_g, cfg.n_embd), H_out_rot, total_tok, damp_pct=args.damp, row_chunk=16384)
                tok_rot[v0:v1] = Q_f16.numpy()
                del w_g, Q_f16
            quantized_rot_tensors["token_embd.weight"] = tok_rot
            del H_out_rot

        del x_cpu
        torch.cuda.empty_cache()

        write_gguf_from_dict(store, args.output, quantized_rot_tensors, weight_names, widths, signs_np, args.block_size)
        print(f"[ok] N3 GPTQ F16 saved to {args.output.name} ({args.output.stat().st_size / (1024**2):.1f} MiB in {time.monotonic() - t0:.1f}s)", flush=True)
    finally:
        store.close()


def _mm_unlink(mm_obj: np.memmap | None, path: Path) -> None:
    if mm_obj is not None:
        try:
            mm_obj.flush()
            del mm_obj
        except Exception:
            pass
    gc.collect()
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def run_gptq_n3a_pipeline(args: argparse.Namespace) -> None:
    """Memory-safe (<1.2 GiB RAM, <850 MiB VRAM) N3A-1 / N3A-2 sequential pipeline."""
    import shutil

    use_actorder = args.algo == "n3a2"
    use_refine = args.algo == "n3a2"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GGUFTensorStore(args.input)
    tmp_root = SCRIPT_DIR / ".build" / "tmp" / f"n3a_work_{args.algo}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tensor_dir = tmp_root / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    p_xf = tmp_root / "x_fp16.mm"
    p_xq = tmp_root / "x_q.mm"
    p_xtmp = tmp_root / "x_tmp.mm"
    p_tokrot = tmp_root / "tok_rot.mm"
    p_tokunrot = tmp_root / "tok_unrot.mm"

    try:
        t0 = time.monotonic()
        cfg = Qwen35Config(store.meta)
        calib_toks = build_calib_batch(
            args.calib_tokens, args.calib_chunks, args.ctx, args.seed,
            chat_path=args.calib_chat, chat_frac=args.chat_frac,
        )
        total_tok = args.calib_chunks * args.ctx
        b_seqs = min(args.batch_seqs, 2) if dev.type == "cuda" else args.batch_seqs

        weight_names = sorted(t["name"] for t in store.hdr["tensors"] if is_foldable_weight(t["name"]))
        fold_set = set(weight_names) | {"token_embd.weight"}
        widths = sorted({int(t["dims"][0]) for t in store.hdr["tensors"] if t["name"] in fold_set})

        rng = np.random.default_rng(args.seed)
        signs_np = {
            w: rng.choice(np.array([-1, 1], dtype=np.int32), size=w).astype(np.float32)
            for w in widths
        }
        signs_gpu = {w: torch.from_numpy(s).to(dev) for w, s in signs_np.items()}
        H_had_gpu = torch.from_numpy(sylvester_hadamard(args.block_size)).to(dev)

        def fold_w_gpu(W: torch.Tensor, width: int) -> torch.Tensor:
            B = args.block_size
            Ws = W.float() * signs_gpu[width][None, :]
            return (Ws.view(-1, B) @ H_had_gpu).view_as(Ws)

        def unfold_w_gpu(W_rot: torch.Tensor, width: int) -> torch.Tensor:
            B = args.block_size
            return (W_rot.float().view(-1, B) @ H_had_gpu).view_as(W_rot) * signs_gpu[width][None, :]

        cos, sin = rope_cache(args.ctx, cfg.rot_dim, cfg.freq_base, dev)
        mm_shape = (args.calib_chunks, args.ctx, cfg.n_embd)

        # ---- Step 0: Collect clean FP16 output_norm Hessian H_out via streaming memmap ----
        print(
            f"=== [{args.algo}] Step 0: Clean FP16 forward for H_out "
            f"(chunks={args.calib_chunks}, tokens={total_tok}, batch_seqs={b_seqs}) ===",
            flush=True,
        )
        tok_emb_np = store.get_numpy("token_embd.weight")
        x_tmp_mm = np.memmap(p_xtmp, dtype=np.float16, mode="w+", shape=mm_shape)
        x_fp16_mm = np.memmap(p_xf, dtype=np.float16, mode="w+", shape=mm_shape)
        for c0 in range(0, args.calib_chunks, 16):
            c1 = min(c0 + 16, args.calib_chunks)
            gathered = tok_emb_np[calib_toks[c0:c1]].astype(np.float16, copy=False)
            x_fp16_mm[c0:c1] = gathered
            x_tmp_mm[c0:c1] = gathered
        x_fp16_mm.flush()
        x_tmp_mm.flush()

        for il in range(cfg.n_layers):
            w_fp = load_layer_weights(store, il, cfg, dev)
            out_key = "ssm_out.weight" if cfg.is_recurrent(il) else "attn_output.weight"
            with torch.no_grad():
                for s0 in range(0, args.calib_chunks, b_seqs):
                    s1 = min(s0 + b_seqs, args.calib_chunks)
                    inpL = torch.from_numpy(np.array(x_tmp_mm[s0:s1], copy=True)).to(dev, dtype=torch.float32)
                    cur = rms_norm(inpL, w_fp["attn_norm.weight"], cfg.rms_eps)
                    attn_pre = compute_attn_core(il, cur, cfg, w_fp, cos, sin)
                    h = inpL + F.linear(attn_pre, w_fp[out_key])
                    cur_ffn = rms_norm(h, w_fp["post_attention_norm.weight"], cfg.rms_eps)
                    interm = F.silu(F.linear(cur_ffn, w_fp["ffn_gate.weight"])) * F.linear(cur_ffn, w_fp["ffn_up.weight"])
                    x_tmp_mm[s0:s1] = (h + F.linear(interm, w_fp["ffn_down.weight"])).to(dtype=torch.float16, device="cpu").numpy()
            del w_fp
            if dev.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                time.sleep(0.01)

        out_norm = store.get_torch("output_norm.weight", device=dev, dtype=torch.float32)
        H_out = torch.zeros((cfg.n_embd, cfg.n_embd), device=dev, dtype=torch.float32)
        with torch.no_grad():
            for s0 in range(0, args.calib_chunks, b_seqs):
                s1 = min(s0 + b_seqs, args.calib_chunks)
                h_b = rms_norm(torch.from_numpy(np.array(x_tmp_mm[s0:s1], copy=True)).to(dev, dtype=torch.float32), out_norm, cfg.rms_eps).reshape(-1, cfg.n_embd)
                H_out.add_(h_b.T @ h_b)
        H_out_rot = transform_x_to_had(H_out, signs_gpu[cfg.n_embd], H_had_gpu)
        del H_out, out_norm
        _mm_unlink(x_tmp_mm, p_xtmp)
        x_tmp_mm = None
        print(f"[step0] Clean H_out collected in {time.monotonic() - t0:.1f}s", flush=True)

        # ---- Step 1: Fix #3 - Quantize tied token_embd.weight FIRST with protected input damping ----
        print(
            f"=== [{args.algo}] Step 1: Quantizing token_embd.weight first "
            f"(embd_damp={args.embd_damp}, actorder={use_actorder}, refine={use_refine}, row_chunk={args.row_chunk}) ===",
            flush=True,
        )
        t1 = time.monotonic()
        V = tok_emb_np.shape[0]
        tok_rot_mm = np.memmap(p_tokrot, dtype=np.float16, mode="w+", shape=(V, cfg.n_embd))
        tok_unrot_mm = np.memmap(p_tokunrot, dtype=np.float16, mode="w+", shape=(V, cfg.n_embd))
        slice_sz = max(args.row_chunk, 8192)
        # Compute Hessian Cholesky/perm ONCE on CPU/GPU by running gptq_ternary_weight_n3a per 32768-row block
        for v0 in range(0, V, 32768):
            v1 = min(v0 + 32768, V)
            # Fold in row_chunk GPU micro-batches into CPU float16 buffer (~128 MiB) so VRAM stays <250 MiB
            rot_buf = torch.empty((v1 - v0, cfg.n_embd), dtype=torch.float16, device="cpu")
            for u0 in range(v0, v1, args.row_chunk):
                u1 = min(u0 + args.row_chunk, v1)
                wg = torch.from_numpy(tok_emb_np[u0:u1].astype(np.float32)).to(dev)
                rot_buf[u0 - v0 : u1 - v0] = fold_w_gpu(wg, cfg.n_embd).to(dtype=torch.float16, device="cpu")
                del wg
            q_rot_cpu = gptq_ternary_weight_n3a(
                rot_buf, H_out_rot, total_tok,
                damp_pct=args.embd_damp, row_chunk=args.row_chunk,
                actorder=use_actorder, refine_scales=use_refine,
            )
            del rot_buf
            tok_rot_mm[v0:v1] = q_rot_cpu.numpy()
            for u0 in range(v0, v1, args.row_chunk):
                u1 = min(u0 + args.row_chunk, v1)
                qr = q_rot_cpu[u0 - v0 : u1 - v0].to(dev)
                tok_unrot_mm[u0:u1] = unfold_w_gpu(qr, cfg.n_embd).to(dtype=torch.float16, device="cpu").numpy()
                del qr
            del q_rot_cpu
            if dev.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                time.sleep(0.01)

        tok_rot_mm.flush()
        tok_unrot_mm.flush()
        del H_out_rot, tok_emb_np

        x_q_mm = np.memmap(p_xq, dtype=np.float16, mode="w+", shape=mm_shape)
        for c0 in range(0, args.calib_chunks, 16):
            c1 = min(c0 + 16, args.calib_chunks)
            x_q_mm[c0:c1] = tok_unrot_mm[calib_toks[c0:c1]].astype(np.float16, copy=False)
        x_q_mm.flush()
        _mm_unlink(tok_unrot_mm, p_tokunrot)
        tok_unrot_mm = None
        del tok_rot_mm
        gc.collect()
        tensor_file_map: dict[str, Path] = {"token_embd.weight": p_tokrot}
        print(f"[step1] token_embd quantized in {time.monotonic() - t1:.1f}s; starting 24-layer asymmetric GPTQ", flush=True)

        # ---- Step 2: 24 layers with asymmetric ridge target vs clean FP16 trajectory ----
        def save_rot_tensor(name: str, q_f16: torch.Tensor) -> None:
            safe = name.replace("/", "_") + ".npy"
            out_p = tensor_dir / safe
            np.save(out_p, q_f16.numpy())
            tensor_file_map[name] = out_p

        for il in range(cfg.n_layers):
            lt0 = time.monotonic()
            w_fp = load_layer_weights(store, il, cfg, dev)
            w_q = {k: v.clone() for k, v in w_fp.items()}
            is_recr = cfg.is_recurrent(il)

            # Sub-block A1: attn input projections
            H_qq = torch.zeros((cfg.n_embd, cfg.n_embd), device=dev, dtype=torch.float32)
            H_qf = torch.zeros((cfg.n_embd, cfg.n_embd), device=dev, dtype=torch.float32)
            with torch.no_grad():
                for s0 in range(0, args.calib_chunks, b_seqs):
                    s1 = min(s0 + b_seqs, args.calib_chunks)
                    xf = torch.from_numpy(np.array(x_fp16_mm[s0:s1], copy=True)).to(dev, dtype=torch.float32)
                    xq = torch.from_numpy(np.array(x_q_mm[s0:s1], copy=True)).to(dev, dtype=torch.float32)
                    cf = rms_norm(xf, w_fp["attn_norm.weight"], cfg.rms_eps).reshape(-1, cfg.n_embd)
                    cq = rms_norm(xq, w_q["attn_norm.weight"], cfg.rms_eps).reshape(-1, cfg.n_embd)
                    H_qq.add_(cq.T @ cq)
                    H_qf.add_(cq.T @ cf)
            M_a1 = solve_asymmetric_target(H_qq, H_qf, ridge_pct=args.ridge)
            del H_qf
            H_qq_rot = transform_x_to_had(H_qq, signs_gpu[cfg.n_embd], H_had_gpu)
            del H_qq

            in_names = ["attn_qkv.weight", "attn_gate.weight"] if is_recr else ["attn_q.weight", "attn_k.weight", "attn_v.weight"]
            for wn in in_names:
                W_tgt = w_fp[wn] @ M_a1
                Q_f16 = gptq_ternary_weight_n3a(
                    fold_w_gpu(W_tgt, cfg.n_embd), H_qq_rot, total_tok,
                    damp_pct=args.damp, row_chunk=args.row_chunk,
                    actorder=use_actorder, refine_scales=use_refine,
                )
                save_rot_tensor(f"blk.{il}.{wn}", Q_f16)
                w_q[wn] = unfold_w_gpu(Q_f16.to(dev), cfg.n_embd)
                del W_tgt, Q_f16
            del M_a1, H_qq_rot

            # Sub-block A2: ssm_out / attn_output
            out_wn = "ssm_out.weight" if is_recr else "attn_output.weight"
            H_ao_qq = torch.zeros((cfg.n_embd, cfg.n_embd), device=dev, dtype=torch.float32)
            H_ao_qf = torch.zeros((cfg.n_embd, cfg.n_embd), device=dev, dtype=torch.float32)
            with torch.no_grad():
                for s0 in range(0, args.calib_chunks, b_seqs):
                    s1 = min(s0 + b_seqs, args.calib_chunks)
                    xf = torch.from_numpy(np.array(x_fp16_mm[s0:s1], copy=True)).to(dev, dtype=torch.float32)
                    xq = torch.from_numpy(np.array(x_q_mm[s0:s1], copy=True)).to(dev, dtype=torch.float32)
                    cf = rms_norm(xf, w_fp["attn_norm.weight"], cfg.rms_eps)
                    cq = rms_norm(xq, w_q["attn_norm.weight"], cfg.rms_eps)
                    af = compute_attn_core(il, cf, cfg, w_fp, cos, sin).reshape(-1, cfg.n_embd)
                    aq = compute_attn_core(il, cq, cfg, w_q, cos, sin).reshape(-1, cfg.n_embd)
                    H_ao_qq.add_(aq.T @ aq)
                    H_ao_qf.add_(aq.T @ af)
            M_a2 = solve_asymmetric_target(H_ao_qq, H_ao_qf, ridge_pct=args.ridge)
            del H_ao_qf
            H_ao_rot = transform_x_to_had(H_ao_qq, signs_gpu[cfg.n_embd], H_had_gpu)
            del H_ao_qq

            W_tgt_ao = w_fp[out_wn] @ M_a2
            Q_ao = gptq_ternary_weight_n3a(
                fold_w_gpu(W_tgt_ao, cfg.n_embd), H_ao_rot, total_tok,
                damp_pct=args.damp, row_chunk=args.row_chunk,
                actorder=use_actorder, refine_scales=use_refine,
            )
            save_rot_tensor(f"blk.{il}.{out_wn}", Q_ao)
            w_q[out_wn] = unfold_w_gpu(Q_ao.to(dev), cfg.n_embd)
            del M_a2, H_ao_rot, W_tgt_ao, Q_ao

            # Advance x_fp16_mm and x_q_mm in-place to post-attention states while building FFN B1 Hessians
            H_fg_qq = torch.zeros((cfg.n_embd, cfg.n_embd), device=dev, dtype=torch.float32)
            H_fg_qf = torch.zeros((cfg.n_embd, cfg.n_embd), device=dev, dtype=torch.float32)
            with torch.no_grad():
                for s0 in range(0, args.calib_chunks, b_seqs):
                    s1 = min(s0 + b_seqs, args.calib_chunks)
                    xf = torch.from_numpy(np.array(x_fp16_mm[s0:s1], copy=True)).to(dev, dtype=torch.float32)
                    xq = torch.from_numpy(np.array(x_q_mm[s0:s1], copy=True)).to(dev, dtype=torch.float32)
                    cf = rms_norm(xf, w_fp["attn_norm.weight"], cfg.rms_eps)
                    cq = rms_norm(xq, w_q["attn_norm.weight"], cfg.rms_eps)
                    hf = xf + F.linear(compute_attn_core(il, cf, cfg, w_fp, cos, sin), w_fp[out_wn])
                    hq = xq + F.linear(compute_attn_core(il, cq, cfg, w_q, cos, sin), w_q[out_wn])
                    x_fp16_mm[s0:s1] = hf.to(dtype=torch.float16, device="cpu").numpy()
                    x_q_mm[s0:s1] = hq.to(dtype=torch.float16, device="cpu").numpy()
                    ff_f = rms_norm(hf, w_fp["post_attention_norm.weight"], cfg.rms_eps).reshape(-1, cfg.n_embd)
                    ff_q = rms_norm(hq, w_q["post_attention_norm.weight"], cfg.rms_eps).reshape(-1, cfg.n_embd)
                    H_fg_qq.add_(ff_q.T @ ff_q)
                    H_fg_qf.add_(ff_q.T @ ff_f)
            x_fp16_mm.flush()
            x_q_mm.flush()

            # Sub-block B1: ffn_gate + ffn_up
            M_b1 = solve_asymmetric_target(H_fg_qq, H_fg_qf, ridge_pct=args.ridge)
            del H_fg_qf
            H_fg_rot = transform_x_to_had(H_fg_qq, signs_gpu[cfg.n_embd], H_had_gpu)
            del H_fg_qq

            for wn in ("ffn_gate.weight", "ffn_up.weight"):
                W_tgt = w_fp[wn] @ M_b1
                Q_f16 = gptq_ternary_weight_n3a(
                    fold_w_gpu(W_tgt, cfg.n_embd), H_fg_rot, total_tok,
                    damp_pct=args.damp, row_chunk=args.row_chunk,
                    actorder=use_actorder, refine_scales=use_refine,
                )
                save_rot_tensor(f"blk.{il}.{wn}", Q_f16)
                w_q[wn] = unfold_w_gpu(Q_f16.to(dev), cfg.n_embd)
                del W_tgt, Q_f16
            del M_b1, H_fg_rot

            # Free attention weights before allocating 6144x6144 FFN Hessians on 2 GB VRAM
            for dict_w in (w_fp, w_q):
                for k in list(dict_w.keys()):
                    if k not in ("post_attention_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
                        del dict_w[k]
            if dev.type == "cuda":
                torch.cuda.empty_cache()

            # Sub-block B2: ffn_down (6144x6144)
            H_fd_qq = torch.zeros((cfg.n_ff, cfg.n_ff), device=dev, dtype=torch.float32)
            H_fd_qf = torch.zeros((cfg.n_ff, cfg.n_ff), device=dev, dtype=torch.float32)
            with torch.no_grad():
                for s0 in range(0, args.calib_chunks, b_seqs):
                    s1 = min(s0 + b_seqs, args.calib_chunks)
                    hf = torch.from_numpy(np.array(x_fp16_mm[s0:s1], copy=True)).to(dev, dtype=torch.float32)
                    hq = torch.from_numpy(np.array(x_q_mm[s0:s1], copy=True)).to(dev, dtype=torch.float32)
                    cf = rms_norm(hf, w_fp["post_attention_norm.weight"], cfg.rms_eps)
                    cq = rms_norm(hq, w_q["post_attention_norm.weight"], cfg.rms_eps)
                    itf = (F.silu(F.linear(cf, w_fp["ffn_gate.weight"])) * F.linear(cf, w_fp["ffn_up.weight"])).reshape(-1, cfg.n_ff)
                    itq = (F.silu(F.linear(cq, w_q["ffn_gate.weight"])) * F.linear(cq, w_q["ffn_up.weight"])).reshape(-1, cfg.n_ff)
                    H_fd_qq.add_(itq.T @ itq)
                    H_fd_qf.add_(itq.T @ itf)
            M_b2 = solve_asymmetric_target(H_fd_qq, H_fd_qf, ridge_pct=args.ridge)
            del H_fd_qf
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            H_fd_rot = transform_x_to_had(H_fd_qq, signs_gpu[cfg.n_ff], H_had_gpu)
            del H_fd_qq
            if dev.type == "cuda":
                torch.cuda.empty_cache()

            W_tgt_fd = w_fp["ffn_down.weight"] @ M_b2
            del M_b2
            Q_fd = gptq_ternary_weight_n3a(
                fold_w_gpu(W_tgt_fd, cfg.n_ff), H_fd_rot, total_tok,
                damp_pct=args.damp, row_chunk=args.row_chunk,
                actorder=use_actorder, refine_scales=use_refine,
            )
            save_rot_tensor(f"blk.{il}.ffn_down.weight", Q_fd)
            w_q["ffn_down.weight"] = unfold_w_gpu(Q_fd.to(dev), cfg.n_ff)
            del H_fd_rot, W_tgt_fd, Q_fd

            # Advance both streams across B2 into x_fp16_mm / x_q_mm for next layer
            with torch.no_grad():
                for s0 in range(0, args.calib_chunks, b_seqs):
                    s1 = min(s0 + b_seqs, args.calib_chunks)
                    hf = torch.from_numpy(np.array(x_fp16_mm[s0:s1], copy=True)).to(dev, dtype=torch.float32)
                    hq = torch.from_numpy(np.array(x_q_mm[s0:s1], copy=True)).to(dev, dtype=torch.float32)
                    cf = rms_norm(hf, w_fp["post_attention_norm.weight"], cfg.rms_eps)
                    cq = rms_norm(hq, w_q["post_attention_norm.weight"], cfg.rms_eps)
                    itf = F.silu(F.linear(cf, w_fp["ffn_gate.weight"])) * F.linear(cf, w_fp["ffn_up.weight"])
                    itq = F.silu(F.linear(cq, w_q["ffn_gate.weight"])) * F.linear(cq, w_q["ffn_up.weight"])
                    x_fp16_mm[s0:s1] = (hf + F.linear(itf, w_fp["ffn_down.weight"])).to(dtype=torch.float16, device="cpu").numpy()
                    x_q_mm[s0:s1] = (hq + F.linear(itq, w_q["ffn_down.weight"])).to(dtype=torch.float16, device="cpu").numpy()
            x_fp16_mm.flush()
            x_q_mm.flush()

            del w_fp, w_q
            if dev.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                time.sleep(0.01)
            gc.collect()
            print(f"  layer {il:2d}/{cfg.n_layers - 1} ({'GDN' if is_recr else 'ATT'}) done in {time.monotonic() - lt0:.1f}s", flush=True)

        _mm_unlink(x_fp16_mm, p_xf)
        _mm_unlink(x_q_mm, p_xq)
        x_fp16_mm = None
        x_q_mm = None

        write_gguf_from_dict(
            store, args.output, {}, weight_names, widths, signs_np, args.block_size,
            tensor_files=tensor_file_map,
        )
        print(f"[ok] {args.algo} F16 saved to {args.output.name} ({args.output.stat().st_size / (1024**2):.1f} MiB in {time.monotonic() - t0:.1f}s)", flush=True)
    finally:
        store.close()
        shutil.rmtree(tmp_root, ignore_errors=True)


def write_gguf_from_dict(
    store: GGUFTensorStore,
    output_path: Path,
    rot_tensors: dict[str, np.ndarray],
    weight_names: list[str],
    widths: list[int],
    signs_np: dict[int, np.ndarray],
    block_size: int,
    tensor_files: dict[str, Path] | None = None,
) -> None:
    hdr = store.hdr
    tensors = hdr["tensors"]
    alignment = hdr["alignment"]
    data_start = hdr["data_start"]

    out_kv = [
        (k, t, v)
        for (k, t, v) in hdr["kv"]
        if not (k == "prism" or k.startswith("prism.") or k.startswith("split."))
    ]
    flat_signs = []
    for w in widths:
        flat_signs.extend(int(x) for x in signs_np[w].astype(np.int32).tolist())

    out_kv.extend([
        ("prism.hadamard.tied_output", BOOL, True),
        ("prism.hadamard.version", U32, 2),
        ("prism.hadamard.block_size", U32, block_size),
        ("prism.hadamard.transform", STR, b"normalized-sylvester-walsh-hadamard"),
        ("prism.hadamard.axis", STR, b"input-last-dimension"),
        ("prism.hadamard.sign_mode", STR, b"explicit"),
        ("prism.hadamard.weight_names", ARR, (STR, [n.encode("utf-8") for n in weight_names])),
        ("prism.hadamard.sign_widths", ARR, (I32, [int(w) for w in widths])),
        ("prism.hadamard.sign_values", ARR, (I32, flat_signs)),
        ("prism.hadamard.inverse_weight_names", ARR, (STR, [b"token_embd.weight"])),
    ])

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
        nbytes = int(math.prod(t["dims"])) * dtype.itemsize
        t["nbytes"] = nbytes
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

    with store.path.open("rb") as fin, tmp_out.open("wb") as fout:
        fout.write(head_blob)
        fout.write(kv_blob)
        fout.write(ti_blob)
        pad0 = (-fout.tell()) % alignment
        if pad0:
            fout.write(b"\x00" * pad0)

        for t in tensors:
            nbytes = t["nbytes"]
            if tensor_files and t["name"] in tensor_files:
                tf = tensor_files[t["name"]]
                if tf.suffix == ".npy":
                    arr = np.load(tf, mmap_mode="r").astype(_NP_DTYPE[t["type"]], copy=False)
                else:
                    shape = tuple(reversed(t["dims"]))
                    arr = np.memmap(tf, dtype=_NP_DTYPE[t["type"]], mode="r", shape=shape)
                assert arr.nbytes == nbytes, f"{t['name']} size mismatch: {arr.nbytes} != {nbytes}"
                for r0 in range(0, arr.shape[0], 16384):
                    fout.write(np.ascontiguousarray(arr[r0 : r0 + 16384]).tobytes())
                del arr
            elif t["name"] in rot_tensors:
                arr = rot_tensors[t["name"]].astype(_NP_DTYPE[t["type"]], copy=False)
                assert arr.nbytes == nbytes, f"{t['name']} size mismatch: {arr.nbytes} != {nbytes}"
                fout.write(arr.tobytes(order="C"))
            else:
                fin.seek(data_start + t["offset"])
                rem = nbytes
                while rem > 0:
                    chunk = fin.read(min(rem, 32 * 1024 * 1024))
                    if not chunk:
                        raise EOFError(t["name"])
                    fout.write(chunk)
                    rem -= len(chunk)
            tpad = (-nbytes) % alignment
            if tpad:
                fout.write(b"\x00" * tpad)

    if output_path.exists():
        output_path.unlink()
    tmp_out.replace(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="N3 / N3A Hadamard + Ternary GPTQ for Qwen3.5-2B PQ2_0")
    parser.add_argument("-i", "--input", type=Path, default=SCRIPT_DIR / "models" / "Qwen3.5-2B-F16.gguf")
    parser.add_argument("-o", "--output", type=Path, default=SCRIPT_DIR / "models" / "qwen3.5-2b-F16-N3-gptq.gguf")
    parser.add_argument("--calib-tokens", type=Path, default=SCRIPT_DIR / ".build" / "tmp" / "calib_tokens.i32")
    parser.add_argument("--calib-chat", type=Path, default=None, help="Optional chat/think .i32 token stream")
    parser.add_argument("--chat-frac", type=float, default=0.40)
    parser.add_argument("--val-tokens", type=Path, default=SCRIPT_DIR / ".build" / "tmp" / "val_tokens.i32")
    parser.add_argument("--calib-chunks", type=int, default=64)
    parser.add_argument("--val-chunks", type=int, default=8)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--batch-seqs", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--embd-damp", type=float, default=0.08, help="Diagonal damping for N3A token_embd Hessian")
    parser.add_argument("--ridge", type=float, default=0.08, help="Asymmetric ridge fraction for N3A decoder targets")
    parser.add_argument("--row-chunk", type=int, default=8192, help="Row batch size for GPU GPTQ (safe on WDDM 2GB)")
    parser.add_argument("--algo", choices=("n3", "n3a1", "n3a2"), default="n3")
    parser.add_argument("--embd-snap", choices=("twn_first", "gptq_head"), default="gptq_head")
    parser.add_argument("--verify-fp16", action="store_true", help="Verify PyTorch Qwen3.5 FP16 PPL against llama-perplexity")
    parser.add_argument("--self-test", action="store_true", help="Run fast mathematical invariant checks and exit")
    args = parser.parse_args()

    if args.self_test:
        self_tests()
        return 0
    if args.algo in ("n3a1", "n3a2"):
        self_tests()
        run_gptq_n3a_pipeline(args)
        return 0
    run_gptq_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
