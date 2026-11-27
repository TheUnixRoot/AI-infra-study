#!/usr/bin/env python3
"""Generate publication-ready PNG charts from real artifacts/b3_1_results.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def short_label(row: dict) -> str:
    fam = "Bonsai2" if "Bonsai" in row["family"] else "Qwen3.5"
    return f"{fam}\n{row['quant']}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot B3.1 benchmark results")
    parser.add_argument("--input", type=Path, default=Path("artifacts/b3_1_results.json"))
    parser.add_argument("--kv-input", type=Path, default=Path("artifacts/b3_1_kv_ablation.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data["results"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    })

    # Preferred single row per model (prism includes all 7 models; also plot upstream vs prism separately)
    prism_rows = [r for r in rows if r["engine"] == "prism"]

    # ------------------------------------------------------------------
    # 1) Pareto: Model Size (MiB) vs Perplexity & KLD (excluding collapsed TQ1_0 from zoomed inset)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2))
    colors = {"Qwen3.5-2B": "#2563eb", "Ternary-Bonsai-2-2B": "#059669"}
    markers = {"Qwen3.5-2B": "o", "Ternary-Bonsai-2-2B": "D"}
    ann_offsets = {
        "PTQ1_0": {"xytext": (-12, -52), "ha": "left", "va": "top"},
        "PQ2_0": {"xytext": (44, -28), "ha": "left", "va": "top"},
        "Q2_0": {"xytext": (-12, 16), "ha": "left", "va": "bottom"},
        "UD-IQ2_XXS": {"xytext": (48, 8), "ha": "left", "va": "bottom"},
        "Q4_K_M": {"xytext": (10, 12), "ha": "left", "va": "bottom"},
        "F16": {"xytext": (-10, 14), "ha": "right", "va": "bottom"},
    }
    arrow_kw = dict(arrowstyle="-", color="#94a3b8", lw=0.8)

    for r in prism_rows:
        fam = r["family"]
        x = r["file_mib"]
        ppl = r["ppl"]
        kld = r["kld_mean"]
        is_tq1 = r["quant"] == "TQ1_0" and ppl and ppl > 100
        off = ann_offsets.get(r["quant"], {"xytext": (8, 8), "ha": "left", "va": "bottom"})
        if ppl is not None and not is_tq1:
            ax1.scatter(x, ppl, color=colors[fam], marker=markers[fam], s=85, zorder=4)
            ax1.annotate(
                f"{r['quant']}\n({ppl:.2f})",
                (x, ppl),
                textcoords="offset points",
                xytext=off["xytext"],
                ha=off["ha"],
                va=off["va"],
                arrowprops=arrow_kw,
                fontsize=8,
            )
        if kld is not None and not is_tq1:
            ax2.scatter(x, kld, color=colors[fam], marker=markers[fam], s=85, zorder=4)
            ax2.annotate(
                f"{r['quant']}\n({kld:.3f})",
                (x, kld),
                textcoords="offset points",
                xytext=off["xytext"],
                ha=off["ha"],
                va=off["va"],
                arrowprops=arrow_kw,
                fontsize=8,
            )

    # Note collapsed TQ1_0 explicitly in subtitle/box if present
    tq1 = next((r for r in prism_rows if r["quant"] == "TQ1_0"), None)
    if tq1 and tq1.get("ppl"):
        ax1.text(
            0.97, 0.96,
            f"Omitido de escala por colapso:\nQwen3.5 TQ1_0 ({tq1['file_mib']} MiB)\nPPL={tq1['ppl']:,.0f}, KLD={tq1['kld_mean']:.2f}",
            transform=ax1.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.35", fc="#fee2e2", ec="#ef4444", alpha=0.92),
        )

    for fam, col in colors.items():
        ax1.scatter([], [], color=col, marker=markers[fam], s=80, label=fam)
    ax1.legend(loc="upper right", bbox_to_anchor=(0.97, 0.72), frameon=True)
    ax1.set_ylim(11.5, 30.2)
    ax1.set_title("Frontera Tamaño vs Perplejidad (menor es mejor)")
    ax1.set_xlabel("Tamaño GGUF en disco (MiB)")
    ax1.set_ylabel("Perplejidad (wikitext-2)")

    ax2.set_ylim(-0.07, 0.82)
    ax2.set_title("Tamaño vs Divergencia KL frente a F16 (menor es mejor)")
    ax2.set_xlabel("Tamaño GGUF en disco (MiB)")
    ax2.set_ylabel("Mean KLD vs Qwen3.5-2B F16")

    fig.suptitle("B3.1 Calidad vs Compresión: Qwen3.5-2B vs Ternary-Bonsai-2-2B (PrismML fork)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out_dir / "b3_1_pareto_quality.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2) Speed: pp512 & tg128 (upstream vs prism)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    keys = []
    for r in prism_rows:
        if r["model_key"] not in keys:
            keys.append(r["model_key"])

    x = np.arange(len(keys))
    width = 0.36
    up_map = {r["model_key"]: r for r in rows if r["engine"] == "upstream"}
    pr_map = {r["model_key"]: r for r in rows if r["engine"] == "prism"}

    labels = [short_label(pr_map[k]) + ("\n(CPU)" if pr_map[k]["ngl"] == 0 else "") for k in keys]
    up_pp = [up_map[k]["pp512_tok_s"] if k in up_map and up_map[k]["pp512_tok_s"] else 0 for k in keys]
    pr_pp = [pr_map[k]["pp512_tok_s"] if pr_map[k]["pp512_tok_s"] else 0 for k in keys]
    up_tg = [up_map[k]["tg128_tok_s"] if k in up_map and up_map[k]["tg128_tok_s"] else 0 for k in keys]
    pr_tg = [pr_map[k]["tg128_tok_s"] if pr_map[k]["tg128_tok_s"] else 0 for k in keys]

    b1 = ax1.bar(x - width / 2, up_pp, width, label="upstream", color="#64748b")
    b2 = ax1.bar(x + width / 2, pr_pp, width, label="prism (Hadamard)", color="#0d9488")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Tokens / s")
    ax1.set_title("Prefill (pp512)")
    ax1.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax1.annotate(f"{h:.0f}", (bar.get_x() + bar.get_width() / 2, h),
                             ha="center", va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points")

    b3 = ax2.bar(x - width / 2, up_tg, width, label="upstream", color="#64748b")
    b4 = ax2.bar(x + width / 2, pr_tg, width, label="prism (Hadamard)", color="#0d9488")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Tokens / s")
    ax2.set_title("Decode (tg128)")
    ax2.legend()
    for bars in (b3, b4):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax2.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                             ha="center", va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points")

    fig.suptitle("B3.1 Throughput en GTX 1050 (Docker CUDA sm_61, F16 en CPU)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out_dir / "b3_1_speed_throughput.png", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3) Memory footprint: File size vs Peak VRAM vs Peak Host RSS
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(prism_rows))
    w = 0.27
    f_mib = [r["file_mib"] for r in prism_rows]
    v_mib = [r["peak_vram_mib"] or 0 for r in prism_rows]
    rss_mib = [r["peak_rss_mib"] or 0 for r in prism_rows]

    ax.bar(x - w, f_mib, w, label="Fichero GGUF (MiB)", color="#94a3b8")
    ax.bar(x, v_mib, w, label="VRAM Pico GPU (MiB)", color="#0ea5e9")
    ax.bar(x + w, rss_mib, w, label="RSS Pico Host RAM (MiB)", color="#f59e0b")
    ax.axhline(2048, color="#dc2626", linestyle="--", linewidth=1.2, label="Límite VRAM GTX 1050 (2048 MiB)")

    ax.set_xticks(x)
    ax.set_xticklabels([short_label(r) + ("\n(ngl=0)" if r["ngl"] == 0 else "") for r in prism_rows], fontsize=8)
    ax.set_ylabel("Memoria (MiB)")
    ax.set_title("B3.1 Consumo real de memoria: Disco vs VRAM GPU vs RSS Host (PrismML fork)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "b3_1_memory_footprint.png", bbox_inches="tight")
    plt.close(fig)

    generated = [
        "b3_1_pareto_quality.png",
        "b3_1_speed_throughput.png",
        "b3_1_memory_footprint.png",
    ]

    # ------------------------------------------------------------------
    # 4) KV-Cache A/B/B0/C Ablation (Quality delta vs F16 + Decode Depth)
    # ------------------------------------------------------------------
    if args.kv_input.is_file():
        kv = json.loads(args.kv_input.read_text(encoding="utf-8"))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.1))

        m_labels = ["Bonsai2\nPQ2_0", "Bonsai2\nPTQ1_0", "Qwen3.5\nQ4_K_M"]
        mode_specs = [
            ("A_hadamard_kv_q4", "A: Hadamard KV q4_0", "#0d9488"),
            ("B_meancenter_kv_q4", "B: Mean-Center KV q4_0", "#2563eb"),
            ("B0_norot_raw_q4", "B0: Raw KV q4_0 (sin mitig.)", "#f97316"),
        ]
        x = np.arange(len(kv["models"]))
        w = 0.25
        for idx, (mkey, mlabel, col) in enumerate(mode_specs):
            deltas = []
            for m in kv["models"]:
                c_ppl = m["perplexity"]["C_control_kv_f16"]["ppl"]
                m_ppl = m["perplexity"][mkey]["ppl"]
                deltas.append(m_ppl - c_ppl)
            bars = ax1.bar(x + (idx - 1) * w, deltas, w, label=mlabel, color=col)
            for bar in bars:
                h = bar.get_height()
                ax1.annotate(
                    f"+{h:.3f}",
                    (bar.get_x() + bar.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=7.5, xytext=(0, 3), textcoords="offset points",
                )
        ax1.set_xticks(x)
        ax1.set_xticklabels(m_labels, fontsize=8.5)
        ax1.set_ylim(0, 0.31)
        ax1.set_ylabel("Δ Perplejidad vs KV F16 (menor es mejor)")
        ax1.set_title("Degradación PPL al comprimir KV a q4_0 (wikitext-2)")
        ax1.legend(loc="upper right", fontsize=8)

        pq2 = next(m for m in kv["models"] if m["key"] == "bonsai2-2b-pq2_0")
        depths = [0, 2048, 4096]
        curves = [
            ("C_control_kv_f16", "C: KV F16 (48 MiB @ 4k)", "#64748b", "o", "--", (0, 8)),
            ("B_norot_kv_q4", "B (sin FWHT): KV q4_0 (13,5 MiB)", "#2563eb", "s", "-", (8, 5)),
            ("A_hadamard_kv_q4", "A: Hadamard KV q4_0 (13,5 MiB)", "#0d9488", "D", "-", (-6, -15)),
        ]
        for ckey, clabel, col, mk, ls, off in curves:
            ys = [pq2["bench"][ckey]["by_depth"][str(d)]["tg128_tok_s"] for d in depths]
            ax2.plot(depths, ys, color=col, marker=mk, linestyle=ls, linewidth=2.0, markersize=7, label=clabel)
            for dx, dy in zip(depths, ys):
                ax2.annotate(
                    f"{dy:.1f}",
                    (dx, dy),
                    textcoords="offset points",
                    xytext=off,
                    ha="center",
                    fontsize=8,
                    color=col,
                    fontweight="bold" if ckey != "B_norot_kv_q4" else "normal",
                )
        ax2.set_xticks(depths)
        ax2.set_xlim(-300, 4400)
        ax2.set_ylim(44, 75)
        ax2.set_xlabel("Profundidad previa de contexto (tokens)")
        ax2.set_ylabel("Decode tg128 (tokens/s)")
        ax2.set_title("PQ2_0 Decode vs Longitud de KV-Cache (GTX 1050 sm_61)")
        ax2.legend(loc="lower left", fontsize=8)

        fig.suptitle("B3.1 KV-Cache Ablation (PrismML 43f31f4): Calidad vs Overhead en sm_61", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(args.out_dir / "b3_1_kv_ablation.png", bbox_inches="tight")
        plt.close(fig)
        generated.append("b3_1_kv_ablation.png")

    print(f"[ok] Generated {', '.join(generated)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
