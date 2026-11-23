#!/usr/bin/env python3
"""
================================================================================
Benchmark B1.1: GPU Memory Layout, Strides, Transpose vs Contiguous
Target Hardware: NVIDIA GeForce GTX 1050 (sm_61, 2GB VRAM, ~112 GB/s Peak BW)
================================================================================

This benchmark quantifies the architectural difference between:
  (a) Zero-copy metadata view manipulation: x.transpose(0, 1)
  (b) Physical VRAM copy and rearrangement: x.transpose(0, 1).contiguous()

Systems Engineering Highlights:
  - Tensor Size: 2048 x 2048 FP32 = 16.0 MiB (< 64 MB OOM threshold for 2GB VRAM).
  - L2 Cache Impact: GTX 1050 has 1024 KB (1 MB) L2 cache. At 16 MiB, memory traffic
    guarantees DRAM bus saturation rather than L2 cache residency.
  - Measurement: Hardware CUDA Events with explicit stream synchronization.
  - Robustness: Graceful fallback and smoke-test support if CUDA is unavailable.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Detect PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


THEORETICAL_PEAK_BW_GB_S = 112.0  # NVIDIA GeForce GTX 1050 (128-bit GDDR5 @ ~7 Gbps)
TARGET_GPU_NAME = "NVIDIA GeForce GTX 1050"
TARGET_ARCH = "sm_61 (Pascal GP107)"
MAX_VRAM_SAFE_MB = 64.0


class BenchmarkResult:
    def __init__(self) -> None:
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.device_name: str = TARGET_GPU_NAME
        self.compute_cap: str = "6.1"
        self.cuda_available: bool = False
        self.is_simulation: bool = False
        self.pytorch_version: str = torch.__version__ if TORCH_AVAILABLE else "N/A"
        
        # Tensor specs
        self.dim_n: int = 2048
        self.dtype_str: str = "torch.float32"
        self.element_size: int = 4
        self.num_elements: int = 2048 * 2048
        self.tensor_bytes: int = 2048 * 2048 * 4
        self.tensor_mb: float = self.tensor_bytes / (1024 * 1024)
        
        # Verification specs
        self.x_shape: Tuple[int, ...] = ()
        self.x_stride: Tuple[int, ...] = ()
        self.x_contiguous: bool = True
        self.x_data_ptr: int = 0
        
        # Transpose specs
        self.y_shape: Tuple[int, ...] = ()
        self.y_stride: Tuple[int, ...] = ()
        self.y_contiguous: bool = False
        self.y_data_ptr: int = 0
        self.y_ptr_match: bool = False
        self.y_gpu_time_ms: Dict[str, float] = {}
        self.y_cpu_time_us: Dict[str, float] = {}
        
        # Contiguous specs
        self.z_shape: Tuple[int, ...] = ()
        self.z_stride: Tuple[int, ...] = ()
        self.z_contiguous: bool = True
        self.z_data_ptr: int = 0
        self.z_ptr_match: bool = False
        self.z_traffic_bytes: int = 0
        self.z_gpu_time_ms: Dict[str, float] = {}
        self.effective_bw_gb_s: Dict[str, float] = {}
        self.effective_bw_gib_s: Dict[str, float] = {}
        self.bus_saturation_pct: Dict[str, float] = {}
        # Sequential-copy baseline (x.clone()): practical bandwidth ceiling of the card.
        self.clone_gpu_time_ms: Optional[float] = None
        self.clone_bw_gb_s: Optional[float] = None

        self.warmup_iters: int = 20
        self.bench_iters: int = 50


def compute_stats(data: List[float]) -> Dict[str, float]:
    """Compute basic statistics: mean, median, min, max, std."""
    if not data:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    sorted_data = sorted(data)
    n = len(data)
    mean_val = sum(data) / n
    median_val = (
        sorted_data[n // 2]
        if n % 2 != 0
        else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2.0
    )
    variance = sum((x - mean_val) ** 2 for x in data) / n if n > 1 else 0.0
    std_val = math.sqrt(variance)
    return {
        "mean": mean_val,
        "median": median_val,
        "min": sorted_data[0],
        "max": sorted_data[-1],
        "std": std_val,
    }


def run_cuda_benchmark(size: int, warmup_iters: int, bench_iters: int) -> BenchmarkResult:
    """Execute benchmark natively on CUDA GPU with torch.cuda.Event."""
    res = BenchmarkResult()
    res.cuda_available = True
    res.is_simulation = False
    res.dim_n = size
    res.num_elements = size * size
    res.tensor_bytes = res.num_elements * 4
    res.tensor_mb = res.tensor_bytes / (1024 * 1024)
    res.warmup_iters = warmup_iters
    res.bench_iters = bench_iters

    device_idx = torch.cuda.current_device()
    res.device_name = torch.cuda.get_device_name(device_idx)
    major, minor = torch.cuda.get_device_capability(device_idx)
    res.compute_cap = f"{major}.{minor}"

    print(f"[INFO] Initializing CUDA context on {res.device_name} (sm_{major}{minor})...")
    torch.cuda.init()
    torch.cuda.synchronize()

    # 1. Base Tensor Creation
    x = torch.randn(size, size, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()

    res.x_shape = tuple(x.shape)
    res.x_stride = tuple(x.stride())
    res.x_contiguous = x.is_contiguous()
    res.x_data_ptr = x.data_ptr()

    # 2. Transpose (Zero-Copy Metadata) Verification
    y = x.transpose(0, 1)
    res.y_shape = tuple(y.shape)
    res.y_stride = tuple(y.stride())
    res.y_contiguous = y.is_contiguous()
    res.y_data_ptr = y.data_ptr()
    res.y_ptr_match = (y.data_ptr() == x.data_ptr())

    # Warmup Transpose
    for _ in range(warmup_iters):
        _ = x.transpose(0, 1)
    torch.cuda.synchronize()

    # Bench Transpose (CUDA Events & CPU overhead)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    y_gpu_samples: List[float] = []
    y_cpu_samples: List[float] = []

    for _ in range(bench_iters):
        t0 = time.perf_counter_ns()
        start_event.record()
        y_tmp = x.transpose(0, 1)
        end_event.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()

        y_gpu_samples.append(start_event.elapsed_time(end_event))
        y_cpu_samples.append((t1 - t0) / 1000.0)  # microseconds

    res.y_gpu_time_ms = compute_stats(y_gpu_samples)
    res.y_cpu_time_us = compute_stats(y_cpu_samples)

    # 3. Contiguous (Physical VRAM Copy) Verification
    z = y.contiguous()
    torch.cuda.synchronize()

    res.z_shape = tuple(z.shape)
    res.z_stride = tuple(z.stride())
    res.z_contiguous = z.is_contiguous()
    res.z_data_ptr = z.data_ptr()
    res.z_ptr_match = (z.data_ptr() == x.data_ptr())
    # Total memory traffic: read input tensor (16MB) + write output tensor (16MB)
    res.z_traffic_bytes = 2 * res.tensor_bytes

    # Warmup Contiguous
    for _ in range(warmup_iters):
        _ = y.contiguous()
    torch.cuda.synchronize()

    # Bench Contiguous
    z_gpu_samples: List[float] = []
    for _ in range(bench_iters):
        torch.cuda.synchronize()
        start_event.record()
        z_tmp = y.contiguous()
        end_event.record()
        torch.cuda.synchronize()
        z_gpu_samples.append(start_event.elapsed_time(end_event))

    res.z_gpu_time_ms = compute_stats(z_gpu_samples)

    # Effective Bandwidth Calculation: (2 * tensor.nbytes) / (time_s)
    # 1 GB = 10^9 Bytes (standard memory bus spec)
    # 1 GiB = 2^30 Bytes
    mean_time_s = res.z_gpu_time_ms["mean"] / 1000.0
    min_time_s = res.z_gpu_time_ms["min"] / 1000.0

    mean_bw_gb_s = (res.z_traffic_bytes / (mean_time_s * 1e9)) if mean_time_s > 0 else 0.0
    peak_bw_gb_s = (res.z_traffic_bytes / (min_time_s * 1e9)) if min_time_s > 0 else 0.0

    mean_bw_gib_s = (res.z_traffic_bytes / (mean_time_s * (1024**3))) if mean_time_s > 0 else 0.0
    peak_bw_gib_s = (res.z_traffic_bytes / (min_time_s * (1024**3))) if min_time_s > 0 else 0.0

    res.effective_bw_gb_s = {"mean": mean_bw_gb_s, "peak": peak_bw_gb_s}
    res.effective_bw_gib_s = {"mean": mean_bw_gib_s, "peak": peak_bw_gib_s}
    res.bus_saturation_pct = {
        "mean": (mean_bw_gb_s / THEORETICAL_PEAK_BW_GB_S) * 100.0,
        "peak": (peak_bw_gb_s / THEORETICAL_PEAK_BW_GB_S) * 100.0,
    }

    # 4. Baseline: x.clone() is a fully coalesced sequential copy (same traffic as the
    # transposed copy). It shows what the card can really sustain, so the gap to
    # y.contiguous() isolates the cost of the strided (non-coalesced) access pattern.
    for _ in range(warmup_iters):
        _ = x.clone()
    torch.cuda.synchronize()
    clone_samples: List[float] = []
    for _ in range(bench_iters):
        torch.cuda.synchronize()
        start_event.record()
        _c = x.clone()
        end_event.record()
        torch.cuda.synchronize()
        clone_samples.append(start_event.elapsed_time(end_event))
    res.clone_gpu_time_ms = compute_stats(clone_samples)["mean"]
    res.clone_bw_gb_s = res.z_traffic_bytes / (res.clone_gpu_time_ms / 1000.0 * 1e9)

    return res


def run_cpu_or_simulation_benchmark(size: int, warmup_iters: int, bench_iters: int) -> BenchmarkResult:
    """
    Fallback execution when CUDA is not present.
    Executes real stride and pointer logic on CPU (using PyTorch CPU or pure-Python Mock),
    and models GTX 1050 Pascal memory architecture expectations.
    """
    res = BenchmarkResult()
    res.cuda_available = False
    res.is_simulation = True
    res.dim_n = size
    res.num_elements = size * size
    res.tensor_bytes = res.num_elements * 4
    res.tensor_mb = res.tensor_bytes / (1024 * 1024)
    res.warmup_iters = warmup_iters
    res.bench_iters = bench_iters

    if TORCH_AVAILABLE:
        print("[NOTICE] PyTorch detected but CUDA is unavailable. Running CPU stride verification...")
        x = torch.randn(size, size, dtype=torch.float32, device="cpu")
        res.x_shape = tuple(x.shape)
        res.x_stride = tuple(x.stride())
        res.x_contiguous = x.is_contiguous()
        res.x_data_ptr = x.data_ptr()

        y = x.transpose(0, 1)
        res.y_shape = tuple(y.shape)
        res.y_stride = tuple(y.stride())
        res.y_contiguous = y.is_contiguous()
        res.y_data_ptr = y.data_ptr()
        res.y_ptr_match = (y.data_ptr() == x.data_ptr())

        z = y.contiguous()
        res.z_shape = tuple(z.shape)
        res.z_stride = tuple(z.stride())
        res.z_contiguous = z.is_contiguous()
        res.z_data_ptr = z.data_ptr()
        res.z_ptr_match = (z.data_ptr() == x.data_ptr())
    else:
        print("[NOTICE] PyTorch is not installed. Running pure-Python architectural model verification...")
        # Pure Python simulation of PyTorch Strided Tensor metadata
        res.x_shape = (size, size)
        res.x_stride = (size, 1)
        res.x_contiguous = True
        res.x_data_ptr = 0x7F1000000000

        res.y_shape = (size, size)
        res.y_stride = (1, size)
        res.y_contiguous = False
        res.y_data_ptr = res.x_data_ptr  # Zero-copy view shares exact pointer!
        res.y_ptr_match = True

        res.z_shape = (size, size)
        res.z_stride = (size, 1)
        res.z_contiguous = True
        res.z_data_ptr = 0x7F1001000000  # New physical allocation
        res.z_ptr_match = False

    res.z_traffic_bytes = 2 * res.tensor_bytes

    # Projected/Simulated timings for NVIDIA GTX 1050 (sm_61 GP107, 112 GB/s theoretical)
    # Typical transpose copy kernel on Pascal achieves ~60-70 GB/s due to strided DRAM access
    simulated_contiguous_ms = 0.495  # ~67.8 GB/s
    res.y_gpu_time_ms = {"mean": 0.0001, "median": 0.0001, "min": 0.0001, "max": 0.0002, "std": 0.00002}
    res.y_cpu_time_us = {"mean": 2.40, "median": 2.30, "min": 1.80, "max": 4.10, "std": 0.35}

    res.z_gpu_time_ms = {
        "mean": simulated_contiguous_ms,
        "median": simulated_contiguous_ms,
        "min": simulated_contiguous_ms * 0.96,
        "max": simulated_contiguous_ms * 1.05,
        "std": 0.015,
    }

    mean_time_s = res.z_gpu_time_ms["mean"] / 1000.0
    min_time_s = res.z_gpu_time_ms["min"] / 1000.0

    mean_bw_gb_s = res.z_traffic_bytes / (mean_time_s * 1e9)
    peak_bw_gb_s = res.z_traffic_bytes / (min_time_s * 1e9)

    mean_bw_gib_s = res.z_traffic_bytes / (mean_time_s * (1024**3))
    peak_bw_gib_s = res.z_traffic_bytes / (min_time_s * (1024**3))

    res.effective_bw_gb_s = {"mean": mean_bw_gb_s, "peak": peak_bw_gb_s}
    res.effective_bw_gib_s = {"mean": mean_bw_gib_s, "peak": peak_bw_gib_s}
    res.bus_saturation_pct = {
        "mean": (mean_bw_gb_s / THEORETICAL_PEAK_BW_GB_S) * 100.0,
        "peak": (peak_bw_gb_s / THEORETICAL_PEAK_BW_GB_S) * 100.0,
    }

    return res


def print_summary_table(res: BenchmarkResult) -> None:
    """Print an ANSI / ASCII high-precision engineering summary table to stdout."""
    term_width = 100
    sep_double = "=" * term_width
    sep_single = "-" * term_width

    exec_status = "CUDA HARDWARE [NATIVE]" if res.cuda_available else "SIMULATION / CPU FALLBACK"

    print("\n" + sep_double)
    print("                 GPU SYSTEMS BENCHMARK B1.1: STRIDES & MEMORY LAYOUT")
    print(sep_double)
    print(f" Target Hardware   : {TARGET_GPU_NAME} ({TARGET_ARCH}, 2GB VRAM, Peak BW: {THEORETICAL_PEAK_BW_GB_S} GB/s)")
    print(f" Execution Mode    : {exec_status}")
    print(f" Active Device     : {res.device_name} (Capability: {res.compute_cap})")
    print(f" PyTorch Version   : {res.pytorch_version}")
    print(f" Matrix Dimension  : {res.dim_n}x{res.dim_n} | Dtype: {res.dtype_str} (4 B/elem) | Elements: {res.num_elements:,}")
    print(f" Tensor Memory     : {res.tensor_mb:.2f} MiB ({res.tensor_bytes / 1e6:.2f} MB) [Threshold < {MAX_VRAM_SAFE_MB} MB: SAFE]")
    print(f" Profile Setup     : {res.warmup_iters} warmup iterations | {res.bench_iters} timed loops (cudaEventSynchronize)")
    print(sep_single)
    
    # Table Header
    print(f"{'Operation':<24} | {'Shape & Stride':<22} | {'Contig?':<7} | {'Storage Ptr':<12} | {'Traffic':<10} | {'Latency (ms)':<12} | {'Effective BW':<14} | {'% Peak':<8}")
    print(sep_single)

    # Base Tensor Row
    base_shape_stride = f"{res.x_shape} {res.x_stride}"
    print(f"{'Base Tensor x':<24} | {base_shape_stride:<22} | {'YES':<7} | {'0x...':<12} | {'0 B':<10} | {'-':<12} | {'-':<14} | {'-':<8}")

    # Transpose Row
    t_shape_stride = f"{res.y_shape} {res.y_stride}"
    t_ptr_str = "MATCH (0-copy)" if res.y_ptr_match else "MISMATCH"
    t_lat_str = f"{res.y_gpu_time_ms['mean']:.4f} ms" if res.cuda_available else "~0.000 ms"
    print(f"{'y = x.transpose(0,1)':<24} | {t_shape_stride:<22} | {'NO':<7} | {t_ptr_str:<12} | {'0 B':<10} | {t_lat_str:<12} | {'N/A (Metadata)':<14} | {'0.0%':<8}")

    # Contiguous Row
    c_shape_stride = f"{res.z_shape} {res.z_stride}"
    c_ptr_str = "NEW ALLOC" if not res.z_ptr_match else "OVERWRITE"
    c_lat_str = f"{res.z_gpu_time_ms['mean']:.4f} ms"
    traffic_mb_str = f"{res.z_traffic_bytes / 1e6:.2f} MB"
    bw_str = f"{res.effective_bw_gb_s['mean']:.2f} GB/s"
    sat_str = f"{res.bus_saturation_pct['mean']:.1f}%"
    print(f"{'z = y.contiguous()':<24} | {c_shape_stride:<22} | {'YES':<7} | {c_ptr_str:<12} | {traffic_mb_str:<10} | {c_lat_str:<12} | {bw_str:<14} | {sat_str:<8}")
    if res.clone_bw_gb_s is not None:
        cl_pct = res.clone_bw_gb_s / THEORETICAL_PEAK_BW_GB_S * 100.0
        print(f"{'x.clone() (baseline)':<24} | {'sequential, coalesced':<22} | {'YES':<7} | {'NEW ALLOC':<12} | {traffic_mb_str:<10} | "
              f"{res.clone_gpu_time_ms:.4f} ms{'':<4} | {res.clone_bw_gb_s:.2f} GB/s{'':<3} | {cl_pct:.1f}%")
    print(sep_single)

    # Engineering Analysis Notes
    print(" Systems Engineering Analysis:")
    print("  1. Transpose: Zero-copy metadata mutation. Shape & stride change (2048, 1) -> (1, 2048).")
    print(f"     Storage pointer is identical: &y[0,0] == &x[0,0] ({res.y_ptr_match}). Memory traffic = 0 Bytes.")
    print("  2. Contiguous: Physical DRAM transfer. Re-strides memory back to (2048, 1) row-major.")
    print(f"     DRAM traffic = Read ({res.tensor_mb:.1f} MB) + Write ({res.tensor_mb:.1f} MB) = {res.z_traffic_bytes / 1e6:.2f} MB ({res.z_traffic_bytes / (1024*1024):.1f} MiB).")
    print(f"     Achieved Bandwidth: {res.effective_bw_gb_s['mean']:.2f} GB/s (Peak: {res.effective_bw_gb_s['peak']:.2f} GB/s) "
          f"vs Theoretical Peak: {THEORETICAL_PEAK_BW_GB_S} GB/s ({res.bus_saturation_pct['mean']:.1f}% bus saturation).")
    print(f"  3. Cache Behavior: Tensor size ({res.tensor_mb:.1f} MiB) >> GTX 1050 L2 Cache (1.0 MiB), guaranteeing true DRAM bus testing.")
    print(sep_double + "\n")


def export_metrics_json(res: BenchmarkResult, output_path: str) -> None:
    """Export complete benchmark metrics to a standardized JSON artifact."""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    metrics_payload = {
        "experiment_id": "B1.1_strides_bench",
        "timestamp": res.timestamp,
        "target_hardware": {
            "model": TARGET_GPU_NAME,
            "architecture": TARGET_ARCH,
            "vram_total_mb": 2048,
            "theoretical_bandwidth_gb_s": THEORETICAL_PEAK_BW_GB_S,
            "memory_bus_width_bits": 128,
            "memory_type": "GDDR5",
            "l2_cache_kb": 1024,
        },
        "runtime_environment": {
            "cuda_available": res.cuda_available,
            "is_simulation": res.is_simulation,
            "active_device": res.device_name,
            "compute_capability": res.compute_cap,
            "pytorch_version": res.pytorch_version,
            "python_version": sys.version.split()[0],
        },
        "tensor_spec": {
            "dimension_n": res.dim_n,
            "shape": list(res.x_shape),
            "dtype": res.dtype_str,
            "element_size_bytes": res.element_size,
            "num_elements": res.num_elements,
            "tensor_bytes": res.tensor_bytes,
            "tensor_size_mib": res.tensor_mb,
            "max_safe_vram_mb": MAX_VRAM_SAFE_MB,
            "oom_safe": res.tensor_mb < MAX_VRAM_SAFE_MB,
        },
        "operations": {
            "transpose": {
                "operation": "y = x.transpose(0, 1)",
                "description": "Zero-copy metadata view modification",
                "input_shape": list(res.x_shape),
                "input_stride": list(res.x_stride),
                "input_contiguous": res.x_contiguous,
                "output_shape": list(res.y_shape),
                "output_stride": list(res.y_stride),
                "output_contiguous": res.y_contiguous,
                "zero_copy_verified": res.y_ptr_match,
                "same_data_pointer": res.y_ptr_match,
                "vram_allocated_bytes": 0,
                "memory_traffic_bytes": 0,
                "warmup_iterations": res.warmup_iters,
                "benchmark_iterations": res.bench_iters,
                "gpu_time_ms": res.y_gpu_time_ms,
                "cpu_dispatch_us": res.y_cpu_time_us,
            },
            "contiguous": {
                "operation": "z = x.transpose(0, 1).contiguous()",
                "description": "Physical copy in VRAM allocating contiguous memory buffer",
                "input_shape": list(res.y_shape),
                "input_stride": list(res.y_stride),
                "input_contiguous": res.y_contiguous,
                "output_shape": list(res.z_shape),
                "output_stride": list(res.z_stride),
                "output_contiguous": res.z_contiguous,
                "zero_copy_verified": False,
                "same_data_pointer": res.z_ptr_match,
                "new_buffer_allocated": not res.z_ptr_match,
                "read_bytes": res.tensor_bytes,
                "write_bytes": res.tensor_bytes,
                "total_traffic_bytes": res.z_traffic_bytes,
                "total_traffic_mb": res.z_traffic_bytes / 1e6,
                "total_traffic_mib": res.z_traffic_bytes / (1024 * 1024),
                "warmup_iterations": res.warmup_iters,
                "benchmark_iterations": res.bench_iters,
                "gpu_time_ms": res.z_gpu_time_ms,
                "effective_bandwidth_gb_s": res.effective_bw_gb_s,
                "effective_bandwidth_gib_s": res.effective_bw_gib_s,
                "theoretical_peak_gb_s": THEORETICAL_PEAK_BW_GB_S,
                "bus_saturation_pct": res.bus_saturation_pct,
            },
        },
        "summary": {
            "transpose_is_zero_copy": res.y_ptr_match,
            "contiguous_allocates_physical_memory": not res.z_ptr_match,
            "effective_bandwidth_gb_s": res.effective_bw_gb_s["mean"],
            "bus_saturation_pct": res.bus_saturation_pct["mean"],
            "clone_baseline_gpu_time_ms": res.clone_gpu_time_ms,
            "clone_baseline_bandwidth_gb_s": res.clone_bw_gb_s,
            "transpose_copy_vs_clone_slowdown_x": (
                res.z_gpu_time_ms["mean"] / res.clone_gpu_time_ms if res.clone_gpu_time_ms else None
            ),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"[ARTIFACT] Metrics successfully exported to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="B1.1 Strides Benchmark: Transpose vs Contiguous on GPU"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2048,
        help="Matrix dimension N (N x N FP32 matrix, default: 2048)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup iterations (default: 20)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of benchmark iterations (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./artifacts/b1_1_metrics.json",
        help="Path to output JSON metrics artifact",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run fast smoke-test (< 30s) with reduced iterations",
    )
    parser.add_argument(
        "--force-simulation",
        action="store_true",
        help="Force simulation mode even if CUDA is detected",
    )

    args = parser.parse_args()

    # Smoke test settings adjustment
    warmup_iters = 5 if args.smoke_test else args.warmup
    bench_iters = 10 if args.smoke_test else args.iterations
    size = args.size

    # Validate memory footprint (< 64 MB constraint)
    mem_mb = (size * size * 4) / (1024 * 1024)
    if mem_mb > MAX_VRAM_SAFE_MB:
        print(
            f"[ERROR] Requested tensor size {mem_mb:.2f} MiB exceeds safety threshold "
            f"({MAX_VRAM_SAFE_MB} MiB) for 2GB VRAM target. Aborting to prevent OOM."
        )
        return 1

    use_cuda = (
        TORCH_AVAILABLE
        and torch.cuda.is_available()
        and not args.force_simulation
    )

    try:
        if use_cuda:
            result = run_cuda_benchmark(
                size=size, warmup_iters=warmup_iters, bench_iters=bench_iters
            )
        else:
            result = run_cpu_or_simulation_benchmark(
                size=size, warmup_iters=warmup_iters, bench_iters=bench_iters
            )

        print_summary_table(result)
        export_metrics_json(result, args.output)
        return 0

    except Exception as e:
        print(f"[FATAL ERROR] Benchmark execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
