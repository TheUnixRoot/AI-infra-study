#!/usr/bin/env python3
"""
kv_calc.py - KV-Cache Arithmetic & VRAM Budgeting CLI for Large Language Models
Part of Project A: agentic-inference-opt-lab (Module B2.1)

Supported Architectures:
  - MHA / GQA: 2 * L * n_kv * d_head * S * B * bytes_per_elem
  - MLA (Multi-Head Latent Attention - DeepSeek-V2 / DeepSeek-V3):
      L * (kv_lora_rank + qk_rope_head_dim) * S * B * bytes_per_elem
"""

import os
import sys
import json
import math
import argparse
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# ==============================================================================
# Verified Embedded Offline Configurations (Guarantee <5s execution offline)
# ==============================================================================

EMBEDDED_CONFIGS: Dict[str, Dict[str, Any]] = {
    "meta-llama/Meta-Llama-3-8B": {
        "_name_or_path": "meta-llama/Meta-Llama-3-8B",
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "max_position_embeddings": 8192,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "torch_dtype": "bfloat16",
        "vocab_size": 128256,
        "num_parameters": 8030261248,
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "_name_or_path": "Qwen/Qwen2.5-72B-Instruct",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 8192,
        "intermediate_size": 29568,
        "max_position_embeddings": 32768,
        "model_type": "qwen2",
        "num_attention_heads": 64,
        "num_hidden_layers": 80,
        "num_key_value_heads": 8,
        "torch_dtype": "bfloat16",
        "vocab_size": 152064,
        "num_parameters": 72706449408,
    },
    "deepseek-ai/DeepSeek-V3": {
        "_name_or_path": "deepseek-ai/DeepSeek-V3",
        "architectures": ["DeepseekV3ForCausalLM"],
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "kv_lora_rank": 512,
        "max_position_embeddings": 4096,
        "model_type": "deepseek_v3",
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_attention_heads": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 61,
        "num_key_value_heads": 128,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1536,
        "v_head_dim": 128,
        "torch_dtype": "bfloat16",
        "vocab_size": 129280,
        "num_parameters": 671027000000,
    },
}

# Aliases for convenience
CONFIG_ALIASES = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "meta-llama/meta-llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-r1": "deepseek-ai/DeepSeek-V3",
}

# Precision configurations: (Name, Weight bytes, KV bytes)
PRECISION_COMBOS = [
    ("Pesos FP16 + KV FP16", 2, 2),
    ("Pesos FP8 + KV FP16", 1, 2),
    ("Pesos FP8 + KV FP8", 1, 1),
]

# Standard GPU profiles for VRAM budgeting: Total VRAM in GiB and GB
GPU_PROFILES = {
    "L40S (48 GB)": {"total_gib": 48.0, "total_gb": 48.0, "arch": "Ada Lovelace", "memory_type": "GDDR6"},
    "H100 (80 GB)": {"total_gib": 80.0, "total_gb": 80.0, "arch": "Hopper", "memory_type": "HBM3"},
}


def resolve_model_key(model_name: str) -> Optional[str]:
    """Finds canonical key in EMBEDDED_CONFIGS or aliases."""
    if model_name in EMBEDDED_CONFIGS:
        return model_name
    lower_name = model_name.lower().strip()
    if lower_name in CONFIG_ALIASES:
        return CONFIG_ALIASES[lower_name]
    for key in EMBEDDED_CONFIGS:
        if lower_name in key.lower() or key.lower() in lower_name:
            return key
    return None


def fetch_or_load_config(model_id_or_path: str, smoke_test: bool = False) -> Tuple[Dict[str, Any], str]:
    """
    Loads config.json from local path, huggingface_hub, or embedded fallback.
    Guarantees fast offline execution.
    """
    source_info = "embedded"

    # If smoke test is requested, prefer embedded verified configs directly
    canonical_key = resolve_model_key(model_id_or_path)
    if smoke_test and canonical_key:
        return EMBEDDED_CONFIGS[canonical_key], f"verified-embedded ({canonical_key})"

    # Check if local path exists
    local_p = Path(model_id_or_path)
    if local_p.is_file():
        try:
            with open(local_p, "r", encoding="utf-8") as f:
                return json.load(f), f"local-file ({local_p.resolve()})"
        except Exception as e:
            print(f"[WARN] Failed to read local file {local_p}: {e}", file=sys.stderr)
    elif local_p.is_dir() and (local_p / "config.json").is_file():
        try:
            with open(local_p / "config.json", "r", encoding="utf-8") as f:
                return json.load(f), f"local-dir ({(local_p / 'config.json').resolve()})"
        except Exception as e:
            print(f"[WARN] Failed to read local directory config: {e}", file=sys.stderr)

    # Attempt to fetch only config.json via huggingface_hub if available
    try:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(
            repo_id=model_id_or_path,
            filename="config.json",
            local_files_only=False,
        )
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f), f"huggingface_hub ({model_id_or_path})"
    except Exception as e:
        # Fall back gracefully to embedded configs if network fails or gated repo
        if canonical_key:
            print(
                f"[INFO] HF download unavailable/gated for '{model_id_or_path}' ({e.__class__.__name__}). "
                f"Falling back to verified embedded config '{canonical_key}'.",
                file=sys.stderr,
            )
            return EMBEDDED_CONFIGS[canonical_key], f"verified-embedded-fallback ({canonical_key})"
        raise RuntimeError(
            f"Could not load config for model '{model_id_or_path}'. "
            f"Not in local filesystem, HF Hub download failed ({e}), and no embedded config matched."
        )


def analyze_model_architecture(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes model architecture and computes attention & KV cache properties:
    - MHA (Multi-Head Attention)
    - GQA (Grouped-Query Attention)
    - MLA (Multi-Head Latent Attention - DeepSeek-V3)
    """
    model_type = config.get("model_type", "").lower()
    architectures = [str(a).lower() for a in config.get("architectures", [])]

    # Check for MLA (Multi-Head Latent Attention)
    is_mla = (
        "kv_lora_rank" in config
        or "deepseek_v3" in model_type
        or "deepseek_v2" in model_type
        or any("deepseekv3" in a or "deepseekv2" in a for a in architectures)
    )

    num_hidden_layers = config.get("num_hidden_layers", 32)

    if is_mla:
        # DeepSeek Multi-Head Latent Attention
        # Instead of storing raw uncompressed heads for all layers, MLA caches:
        # 1) Compressed latent KV vector c^{KV} of dimension kv_lora_rank (e.g. 512)
        # 2) Decoupled RoPE Key k^R of dimension qk_rope_head_dim (e.g. 64)
        # Total cached vector per token per layer = kv_lora_rank + qk_rope_head_dim
        # (NO factor of 2, because c^{KV} holds joint compressed representation for both K and V!)
        kv_lora_rank = config.get("kv_lora_rank", 512)
        qk_rope_head_dim = config.get("qk_rope_head_dim", 64)
        num_attention_heads = config.get("num_attention_heads", 128)

        elems_per_token_per_layer = kv_lora_rank + qk_rope_head_dim
        total_kv_elems_per_token = num_hidden_layers * elems_per_token_per_layer

        attn_type = "MLA"
        attn_details = (
            f"MLA (kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim}, "
            f"heads={num_attention_heads}, L={num_hidden_layers})"
        )
        head_dim_desc = f"latent({kv_lora_rank}) + rope({qk_rope_head_dim}) = {elems_per_token_per_layer}"
        n_kv_heads = 1  # In MLA, latent vector is cached per token per layer
    else:
        # Standard MHA or GQA
        # Formula: 2 * L * n_kv * d_head
        num_attention_heads = config.get("num_attention_heads", 32)
        num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
        hidden_size = config.get("hidden_size", 4096)

        if "head_dim" in config:
            head_dim_val = config["head_dim"]
        elif "kv_channels" in config:
            head_dim_val = config["kv_channels"]
        else:
            head_dim_val = hidden_size // num_attention_heads

        attn_type = "GQA" if num_key_value_heads < num_attention_heads else "MHA"
        # 2 for Key and Value
        elems_per_token_per_layer = 2 * num_key_value_heads * head_dim_val
        total_kv_elems_per_token = num_hidden_layers * elems_per_token_per_layer

        attn_details = (
            f"{attn_type} (q_heads={num_attention_heads}, kv_heads={num_key_value_heads}, "
            f"d_head={head_dim_val}, L={num_hidden_layers})"
        )
        head_dim_desc = str(head_dim_val)
        n_kv_heads = num_key_value_heads

    # Parameter count estimation
    param_count = config.get("num_parameters")
    if param_count is None:
        param_count = estimate_parameters_from_config(config)

    return {
        "attn_type": attn_type,
        "attn_details": attn_details,
        "is_mla": is_mla,
        "num_hidden_layers": num_hidden_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim_desc": head_dim_desc,
        "elems_per_token_per_layer": elems_per_token_per_layer,
        "total_kv_elems_per_token": total_kv_elems_per_token,
        "param_count": param_count,
        "param_count_billions": round(param_count / 1e9, 2),
    }


def estimate_parameters_from_config(config: Dict[str, Any]) -> int:
    """Analytical parameter estimator for models when num_parameters is not explicitly given."""
    name_or_type = (config.get("_name_or_path", "") + " " + config.get("model_type", "")).lower()
    if "llama-3-8b" in name_or_type or "llama_3_8b" in name_or_type:
        return 8030261248
    if "qwen2.5-72b" in name_or_type or "qwen_72b" in name_or_type:
        return 72706449408
    if "deepseek-v3" in name_or_type or "deepseek_v3" in name_or_type:
        return 671027000000

    # Generic transformer parameter counter
    vocab_size = config.get("vocab_size", 32000)
    hidden_size = config.get("hidden_size", 4096)
    intermediate_size = config.get("intermediate_size", hidden_size * 4)
    num_layers = config.get("num_hidden_layers", 32)
    num_attention_heads = config.get("num_attention_heads", 32)
    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
    tie_word_embeddings = config.get("tie_word_embeddings", False)

    # Embeddings
    embedding_params = vocab_size * hidden_size
    lm_head_params = 0 if tie_word_embeddings else (vocab_size * hidden_size)

    # Attention per layer
    head_dim = hidden_size // num_attention_heads
    q_proj = hidden_size * (num_attention_heads * head_dim)
    k_proj = hidden_size * (num_key_value_heads * head_dim)
    v_proj = hidden_size * (num_key_value_heads * head_dim)
    o_proj = (num_attention_heads * head_dim) * hidden_size
    attn_per_layer = q_proj + k_proj + v_proj + o_proj

    # MLP per layer (SwiGLU has gate, up, down)
    mlp_per_layer = 3 * hidden_size * intermediate_size

    # LayerNorms
    ln_per_layer = 4 * hidden_size

    total = embedding_params + lm_head_params + num_layers * (attn_per_layer + mlp_per_layer + ln_per_layer)
    return total


def calculate_vram_and_concurrency(
    arch_info: Dict[str, Any],
    seq_len: int,
    gpu_mem_util: float,
    overhead_gb: float,
) -> List[Dict[str, Any]]:
    """
    Computes weight VRAM, KV Cache size, and maximum concurrency across
    the 3 required precision configurations for L40S and H100.
    """
    param_count = arch_info["param_count"]
    total_kv_elems = arch_info["total_kv_elems_per_token"]

    results = []

    for combo_name, weight_bytes, kv_bytes in PRECISION_COMBOS:
        # Weight VRAM calculation
        weight_total_bytes = param_count * weight_bytes
        weight_vram_gb = weight_total_bytes / 1e9          # Decimal GB (vendor spec)
        weight_vram_gib = weight_total_bytes / (1024**3)   # Binary GiB (CUDA allocation)

        # KV-Cache calculation
        kv_bytes_per_token = total_kv_elems * kv_bytes
        kv_kib_per_token = kv_bytes_per_token / 1024.0

        kv_bytes_per_seq = kv_bytes_per_token * seq_len
        kv_gib_per_seq = kv_bytes_per_seq / (1024**3)
        kv_gb_per_seq = kv_bytes_per_seq / 1e9

        # Hardware concurrency analysis
        hardware_results = {}
        for gpu_name, gpu_spec in GPU_PROFILES.items():
            total_gib = gpu_spec["total_gib"]

            # Usable VRAM formula: Total * mem_util - overhead
            usable_gib = (total_gib * gpu_mem_util) - overhead_gb

            # Available for KV cache: Usable - Weights
            available_kv_gib = usable_gib - weight_vram_gib

            if weight_vram_gib > usable_gib:
                # Weights alone exceed usable single-GPU memory
                max_concurrency = 0
                status = f"OOM: Weights ({weight_vram_gib:.1f} GiB) exceed usable VRAM ({usable_gib:.1f} GiB)"
                min_gpus_weights = math.ceil(weight_vram_gib / usable_gib)
            else:
                if available_kv_gib <= 0 or kv_gib_per_seq <= 0:
                    max_concurrency = 0
                    status = "OOM: No remaining VRAM for KV-Cache"
                else:
                    max_concurrency = math.floor(available_kv_gib / kv_gib_per_seq)
                    # Less than one full sequence of KV-cache fits -> not servable
                    if max_concurrency >= 1:
                        status = "OK"
                    else:
                        status = "OOM: Available KV < 1 seq"
                min_gpus_weights = 1

            hardware_results[gpu_name] = {
                "total_vram_gib": total_gib,
                "usable_vram_gib": round(usable_gib, 2),
                "available_kv_gib": round(max(0.0, available_kv_gib), 2),
                "max_concurrency": max_concurrency,
                "status": status,
                "min_gpus_weights": min_gpus_weights,
            }

        results.append({
            "combo_name": combo_name,
            "weight_bytes": weight_bytes,
            "kv_bytes": kv_bytes,
            "weight_vram_gb": round(weight_vram_gb, 2),
            "weight_vram_gib": round(weight_vram_gib, 2),
            "kv_kib_per_token": round(kv_kib_per_token, 3),
            "kv_gib_per_seq": round(kv_gib_per_seq, 3),
            "kv_gb_per_seq": round(kv_gb_per_seq, 3),
            "hardware": hardware_results,
        })

    return results


def format_markdown_table(all_model_results: List[Dict[str, Any]], seq_len: int) -> str:
    """Formats clean, publication-ready markdown table for study notebook."""
    lines = []
    lines.append(f"### KV-Cache Arithmetic & Concurrency Benchmark Matrix (Seq Len = {seq_len})")
    lines.append("")
    headers = [
        "Model",
        "Arch",
        "Params",
        "Precision Setup",
        "Weight VRAM (GB / GiB)",
        "KV (KiB/tok)",
        f"KV/Seq ({seq_len})",
        "L40S (48GB) Streams",
        "H100 (80GB) Streams",
        "Min H100 GPUs",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for m in all_model_results:
        m_name = m["model_name"]
        arch = m["arch_info"]["attn_type"]
        params = f"{m['arch_info']['param_count_billions']}B"

        for row in m["combinations"]:
            combo = row["combo_name"]
            w_vram = f"{row['weight_vram_gb']:.1f} GB ({row['weight_vram_gib']:.1f} GiB)"
            kib_tok = f"{row['kv_kib_per_token']:.2f} KiB"
            kv_seq = f"{row['kv_gib_per_seq']:.2f} GiB"

            l40s_info = row["hardware"]["L40S (48 GB)"]
            h100_info = row["hardware"]["H100 (80 GB)"]

            l40s_disp = str(l40s_info["max_concurrency"]) if l40s_info["max_concurrency"] > 0 else "0 (OOM)"
            h100_disp = str(h100_info["max_concurrency"]) if h100_info["max_concurrency"] > 0 else "0 (OOM)"
            min_h100 = f"{h100_info['min_gpus_weights']}x"

            lines.append(
                f"| {m_name} | {arch} | {params} | {combo} | {w_vram} | {kib_tok} | {kv_seq} | {l40s_disp} | {h100_disp} | {min_h100} |"
            )

    return "\n".join(lines)


def export_csv(all_model_results: List[Dict[str, Any]], seq_len: int, output_path: str):
    """Exports structured metrics to CSV."""
    import csv

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_name",
            "arch_type",
            "parameters_billions",
            "precision_combo",
            "weight_bytes",
            "kv_bytes",
            "weight_vram_gb",
            "weight_vram_gib",
            "kv_kib_per_token",
            "seq_len",
            "kv_gib_per_seq",
            "kv_gb_per_seq",
            "l40s_max_concurrency",
            "l40s_status",
            "l40s_min_gpus",
            "h100_max_concurrency",
            "h100_status",
            "h100_min_gpus",
        ])

        for m in all_model_results:
            m_name = m["model_name"]
            arch = m["arch_info"]["attn_type"]
            params = m["arch_info"]["param_count_billions"]

            for row in m["combinations"]:
                l40s = row["hardware"]["L40S (48 GB)"]
                h100 = row["hardware"]["H100 (80 GB)"]

                writer.writerow([
                    m_name,
                    arch,
                    params,
                    row["combo_name"],
                    row["weight_bytes"],
                    row["kv_bytes"],
                    row["weight_vram_gb"],
                    row["weight_vram_gib"],
                    row["kv_kib_per_token"],
                    seq_len,
                    row["kv_gib_per_seq"],
                    row["kv_gb_per_seq"],
                    l40s["max_concurrency"],
                    l40s["status"],
                    l40s["min_gpus_weights"],
                    h100["max_concurrency"],
                    h100["status"],
                    h100["min_gpus_weights"],
                ])


def export_json(all_model_results: List[Dict[str, Any]], cli_args: Dict[str, Any], output_path: str):
    """Exports structured metrics and metadata to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": {
            "title": "B2.1 KV-Cache Arithmetic & VRAM Budgeting Report",
            "module": "Project A: agentic-inference-opt-lab",
            "cli_args": cli_args,
            "gpu_profiles": GPU_PROFILES,
            "precision_combos": [c[0] for c in PRECISION_COMBOS],
        },
        "results": all_model_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="KV-Cache Arithmetic and VRAM Budget Calculator (MHA, GQA, MLA)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["meta-llama/Meta-Llama-3-8B", "Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-V3"],
        help="List of model Hugging Face IDs or local paths",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8192,
        help="Target sequence length in tokens (default: 8192)",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.90,
        help="Target GPU memory utilization ratio (default: 0.90)",
    )
    parser.add_argument(
        "--overhead-gb",
        type=float,
        default=2.0,
        help="VRAM overhead buffer in GB for CUDA runtime/kernels (default: 2.0)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast smoke-test using verified embedded configs (execution in <5s)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save JSON metrics report (e.g. ./artifacts/b2_1_kv_report.json)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save CSV metrics report (e.g. ./artifacts/b2_1_kv_report.csv)",
    )

    args = parser.parse_args()

    all_model_results = []

    for model_name in args.models:
        try:
            config, source_desc = fetch_or_load_config(model_name, smoke_test=args.smoke_test)
            arch_info = analyze_model_architecture(config)
            combos = calculate_vram_and_concurrency(
                arch_info=arch_info,
                seq_len=args.seq_len,
                gpu_mem_util=args.gpu_mem_util,
                overhead_gb=args.overhead_gb,
            )
            all_model_results.append({
                "model_name": model_name,
                "config_source": source_desc,
                "arch_info": arch_info,
                "combinations": combos,
            })
        except Exception as e:
            print(f"[ERROR] Failed to process model '{model_name}': {e}", file=sys.stderr)
            sys.exit(1)

    # Format table for notebook
    md_table = format_markdown_table(all_model_results, args.seq_len)
    print("\n" + md_table + "\n")

    # Save exports if requested
    cli_args_dict = {
        "models": args.models,
        "seq_len": args.seq_len,
        "gpu_mem_util": args.gpu_mem_util,
        "overhead_gb": args.overhead_gb,
        "smoke_test": args.smoke_test,
    }

    if args.output_json:
        export_json(all_model_results, cli_args_dict, args.output_json)
        print(f"[INFO] Saved JSON report to: {args.output_json}")

    if args.output_csv:
        export_csv(all_model_results, args.seq_len, args.output_csv)
        print(f"[INFO] Saved CSV report to: {args.output_csv}")


if __name__ == "__main__":
    main()
