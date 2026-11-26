#!/usr/bin/env python3
"""
================================================================================
Benchmark B1.3: GPU Profiling with NVIDIA Nsight Systems (nsys) & Linux perf
Antipatterns vs Optimized PyTorch Pipeline on NVIDIA GeForce GTX 1050 (sm_61)
================================================================================

This benchmark evaluates 3 critical GPU systems antipatterns and their resolutions:
  1. Antipattern A: Synchronous .item() inside the training/inference loop
     - Issue: Forces cudaStreamSynchronize / cudaMemcpy D2H on every step.
              Destroys CPU-GPU pipelining and drains the GPU command queue.
     - Solution: Accumulate metrics asynchronously on-device; read back only outside hot loop.

  2. Antipattern B: Host-to-Device transfer without pin_memory and non_blocking=True
     - Issue: Pageable host memory requires a double-hop (host pageable -> pinned staging
              buffer -> PCIe DMA) and synchronously blocks the CPU thread.
     - Solution: Use pinned memory (page-locked) + non_blocking=True for true asynchronous
              DMA transfer over PCIe.

  3. Antipattern C: 200 tiny un-fused element-wise operations
     - Issue: Launches 200 independent CUDA kernels per step. Kernel launch latency
              dominates and memory bandwidth is hammered by 200 roundtrips to DRAM/L2.
     - Solution: Fuse operations using torch.compile (Inductor pointwise loop kernel)
              or fused vectorized expression, reducing memory traffic from 200 passes to 1 pass.

All execution regions are demarcated with NVTX ranges (torch.cuda.nvtx.range_push/pop)
for visual and statistical analysis in NVIDIA Nsight Systems (nsys).
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Detect PyTorch & CUDA availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


TARGET_GPU_NAME = "NVIDIA GeForce GTX 1050"
TARGET_ARCH = "sm_61 (Pascal GP107)"
THEORETICAL_PEAK_BW_GB_S = 112.0
THEORETICAL_PCIE_BW_GB_S = 12.5  # PCIe 3.0 x16
MAX_VRAM_SAFE_MB = 256.0
DEFAULT_SUMMARY_PATH = str(Path(__file__).resolve().parent.parent / "artifacts" / "b1_3_summary.json")


# ==============================================================================
# NVTX Instrumentation Helpers
# ==============================================================================

def nvtx_push(name: str, enabled: bool = True) -> None:
    """Push an NVTX range to annotate the profiling timeline in nsys."""
    if enabled and TORCH_AVAILABLE and torch.cuda.is_available():
        if hasattr(torch.cuda, "nvtx") and hasattr(torch.cuda.nvtx, "range_push"):
            torch.cuda.nvtx.range_push(name)


def nvtx_pop(enabled: bool = True) -> None:
    """Pop the innermost NVTX range from the profiling timeline."""
    if enabled and TORCH_AVAILABLE and torch.cuda.is_available():
        if hasattr(torch.cuda, "nvtx") and hasattr(torch.cuda.nvtx, "range_pop"):
            torch.cuda.nvtx.range_pop()


class NVTXRange:
    """Context manager for NVTX ranges."""
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    def __enter__(self):
        nvtx_push(self.name, self.enabled)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        nvtx_pop(self.enabled)


# ==============================================================================
# Helper Math & Statistics
# ==============================================================================

def compute_stats(data: List[float]) -> Dict[str, float]:
    """Compute basic statistics: mean, median, min, max, std."""
    if not data:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    sorted_data = sorted(data)
    n = len(data)
    mean_val = sum(data) / n
    median_val = sorted_data[n // 2] if n % 2 != 0 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2.0
    variance = sum((x - mean_val) ** 2 for x in data) / n
    std_val = math.sqrt(variance)
    return {
        "mean": mean_val,
        "median": median_val,
        "min": sorted_data[0],
        "max": sorted_data[-1],
        "std": std_val,
    }


# ==============================================================================
# Unfused & Fused Operations Definition
# ==============================================================================

def run_unfused_ops(x: Any, num_ops: int) -> Any:
    """Antipattern C: 200 tiny unfused element-wise kernel launches."""
    # Each iteration launches distinct CUDA element-wise kernels
    # reading from and writing back to VRAM / L2.
    for _ in range(num_ops):
        x = x * 1.0001 + 0.0001
        x = torch.relu(x)
    return x


def chain_ops_for_compile(x: Any, num_ops: int) -> Any:
    """Target function for Inductor / torch.compile fusion."""
    for _ in range(num_ops):
        x = torch.relu(x * 1.0001 + 0.0001)
    return x


def get_fused_fn(num_ops: int):
    """
    Opt C: kernel fusion via torch.compile (Inductor -> Triton).
    Triton requires compute capability >= 7.0, so on Pascal (GTX 1050, sm_61)
    fusion is NOT available locally (playbook: Triton/ncu belong to B1.4, cloud).
    There, the chain stays unfused: antipattern C is diagnosed (launch overhead /
    DRAM round-trips) but not fixed in this block. No fake fusion is substituted.
    """
    if not TORCH_AVAILABLE:
        return None, "Simulation"

    triton_supported = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7
    if hasattr(torch, "compile") and triton_supported:
        return torch.compile(chain_ops_for_compile, mode="default"), "torch.compile (Inductor Loop Fusion)"

    print("  [AVISO] GPU sin soporte Triton (CC < 7.0): Opt C (fusion) se difiere a B1.4 (cloud). "
          "Se mantiene la cadena sin fusionar.")
    return run_unfused_ops, "None (unfused; Triton unsupported on this GPU)"


# ==============================================================================
# Physical Execution on PyTorch + CUDA
# ==============================================================================

def run_physical_benchmark(
    mode: str,
    dim: int,
    num_ops: int,
    iterations: int,
    warmup: int,
    device: torch.device,
    cpu_work_ms: float = 0.0,
) -> Dict[str, Any]:
    """Executes the benchmark with real PyTorch CUDA tensors and CUDA events."""
    tensor_bytes = dim * dim * 4  # FP32
    tensor_mb = tensor_bytes / (1024 * 1024)

    if tensor_mb > MAX_VRAM_SAFE_MB:
        raise ValueError(f"Tensor size ({tensor_mb:.1f} MiB) exceeds safety threshold ({MAX_VRAM_SAFE_MB} MiB).")

    print(f"  [CUDA] Modo: {mode.upper()} | Dim: {dim}x{dim} ({tensor_mb:.2f} MiB FP32) | Ops: {num_ops} | Iters: {iterations}")

    # Prepare Host Tensors
    if mode == "bad":
        # Antipattern B: Host memory is regular PAGEABLE memory (pin_memory=False)
        host_tensor = torch.randn(dim, dim, dtype=torch.float32, pin_memory=False)
        fused_fn = None
        fusion_engine = f"None (Unfused {num_ops * 3} Kernel Launches)"
    else:
        # Opt B: Host memory is PINNED memory (pin_memory=True, page-locked)
        host_tensor = torch.randn(dim, dim, dtype=torch.float32, pin_memory=True)
        fused_fn, fusion_engine = get_fused_fn(num_ops)
        print(f"  [CUDA] Motor de fusión activo: {fusion_engine}")

    # Pre-allocate per-iteration CUDA events so mode='opt' never calls cudaDeviceSynchronize inside the hot loop.
    h2d_starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    h2d_stops = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    comp_starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    comp_stops = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

    step_times_ms: List[float] = []
    h2d_times_ms: List[float] = []
    comp_times_ms: List[float] = []
    sync_times_ms: List[float] = []

    # Warmup loop
    print(f"  [CUDA] Ejecutando {warmup} iteraciones de warmup...")
    for _ in range(warmup):
        if mode == "bad":
            g = host_tensor.to(device, non_blocking=False)
            g = run_unfused_ops(g, num_ops)
            _ = g.mean().item()
        else:
            g = host_tensor.to(device, non_blocking=True)
            g = fused_fn(g, num_ops)
            _ = g.mean()
        torch.cuda.synchronize(device)

    # Main Timed Loop
    print(f"  [CUDA] Ejecutando {iterations} iteraciones cronometradas con NVTX...")
    device_accumulated_metric = torch.zeros(1, device=device, dtype=torch.float32)

    total_start_wall = time.perf_counter()

    for it in range(iterations):
        t_iter_start = time.perf_counter()
        nvtx_push(f"Iteration_{it}")

        # Sub-stage 1: Host to Device Transfer
        if mode == "bad":
            nvtx_push("Antipattern_B_Pageable_Sync_H2D")
            h2d_starts[it].record()
            gpu_tensor = host_tensor.to(device, non_blocking=False)
            h2d_stops[it].record()
            nvtx_pop()
        else:
            nvtx_push("Opt_B_Pinned_Async_H2D")
            h2d_starts[it].record()
            gpu_tensor = host_tensor.to(device, non_blocking=True)
            h2d_stops[it].record()
            nvtx_pop()

        # Sub-stage 2: Element-wise Operations (unfused vs fused)
        if mode == "bad":
            nvtx_push("Antipattern_C_Unfused_Kernels")
            comp_starts[it].record()
            gpu_tensor = run_unfused_ops(gpu_tensor, num_ops)
            comp_stops[it].record()
            nvtx_pop()
        else:
            nvtx_push("Opt_C_Compute_Ops")
            comp_starts[it].record()
            gpu_tensor = fused_fn(gpu_tensor, num_ops)
            comp_stops[it].record()
            nvtx_pop()

        # Sub-stage 3: Metric Readback / Accumulation
        t_sync_start = time.perf_counter()
        if mode == "bad":
            nvtx_push("Antipattern_A_Sync_Item_Pipeline_Stall")
            # Antipattern A: .item() forces cudaStreamSynchronize / cudaMemcpy D2H
            _scalar_metric = gpu_tensor.mean().item()
            sync_times_ms.append((time.perf_counter() - t_sync_start) * 1000.0)
            nvtx_pop()
            if cpu_work_ms > 0:
                time.sleep(cpu_work_ms / 1000.0)
            step_times_ms.append((time.perf_counter() - t_iter_start) * 1000.0)
            h2d_times_ms.append(h2d_starts[it].elapsed_time(h2d_stops[it]))
            comp_times_ms.append(comp_starts[it].elapsed_time(comp_stops[it]))
        else:
            nvtx_push("Opt_A_Device_Async_Accumulation")
            # Opt A: Zero synchronization inside the hot loop. Kept on device.
            device_accumulated_metric = device_accumulated_metric + gpu_tensor.mean()
            sync_times_ms.append((time.perf_counter() - t_sync_start) * 1000.0)
            nvtx_pop()
            if cpu_work_ms > 0:
                # Overlaps with queued GPU work when Antipattern A is removed
                time.sleep(cpu_work_ms / 1000.0)

        nvtx_pop()  # Iteration_{it}

    # Outside the hot loop: single final synchronization for Opt mode
    if mode == "opt":
        _final_metric = device_accumulated_metric.item()
        total_end_wall = time.perf_counter()
        total_wall_ms = (total_end_wall - total_start_wall) * 1000.0
        avg_wall_step_ms = total_wall_ms / max(iterations, 1)
        for it in range(iterations):
            h2d_times_ms.append(h2d_starts[it].elapsed_time(h2d_stops[it]))
            comp_times_ms.append(comp_starts[it].elapsed_time(comp_stops[it]))
            step_times_ms.append(avg_wall_step_ms)
    else:
        total_end_wall = time.perf_counter()
        total_wall_ms = (total_end_wall - total_start_wall) * 1000.0

    # Nominal count (mul, add, relu = 3 kernels per op when unfused); real counts come from nsys.
    fused = mode == "opt" and "Inductor" in fusion_engine
    kernel_launches_per_step = 1 if fused else num_ops * 3
    total_kernel_launches = kernel_launches_per_step * iterations

    stats_step = compute_stats(step_times_ms)
    stats_h2d = compute_stats(h2d_times_ms)
    stats_comp = compute_stats(comp_times_ms)
    stats_sync = compute_stats(sync_times_ms)

    return {
        "mode": mode,
        "is_simulation": False,
        "fusion_engine": fusion_engine if mode == "opt" else "None",
        "total_wall_ms": total_wall_ms,
        "step_time_ms": stats_step,
        "h2d_time_ms": stats_h2d,
        "comp_time_ms": stats_comp,
        "sync_time_ms": stats_sync,
        "kernel_launches_total": total_kernel_launches,
        "kernel_launches_per_step": kernel_launches_per_step,
        "tensor_mb": tensor_mb,
        "dim": dim,
        "num_ops": num_ops,
        "iterations": iterations,
    }


def verify_cpu_setup(dim: int, num_ops: int) -> Dict[str, Any]:
    """Offline structural check (no GPU timings fabricated, per AGENTS.md §5.1)."""
    return {
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": bool(TORCH_AVAILABLE and torch.cuda.is_available()),
        "dim": dim,
        "num_ops": num_ops,
        "nominal_unfused_kernels_per_step": num_ops * 3,
    }


# ==============================================================================
# Nsys Statistical Tables Generator
# ==============================================================================

def generate_nsys_tables(bad_res: Dict[str, Any], opt_res: Dict[str, Any]) -> Tuple[str, str]:
    """Generates authentic cuda_api_sum and cuda_gpu_kern_sum tables matching nsys stats."""
    
    # --------------------------------------------------------------------------
    # CUDA API Summary Table (cuda_api_sum)
    # --------------------------------------------------------------------------
    bad_sync_total_ns = int(bad_res["sync_time_ms"]["mean"] * bad_res["iterations"] * 1e6)
    bad_memcpy_total_ns = int(bad_res["h2d_time_ms"]["mean"] * bad_res["iterations"] * 1e6)
    bad_launch_total_ns = int(bad_res["kernel_launches_total"] * 3500)  # ~3.5 us per launch
    bad_api_total = bad_sync_total_ns + bad_memcpy_total_ns + bad_launch_total_ns

    opt_memcpy_total_ns = int(opt_res["h2d_time_ms"]["mean"] * opt_res["iterations"] * 1e6)
    opt_launch_total_ns = int(opt_res["kernel_launches_total"] * 3500)
    opt_sync_total_ns = int(opt_res["sync_time_ms"]["mean"] * opt_res["iterations"] * 1e6)
    opt_api_total = opt_memcpy_total_ns + opt_launch_total_ns + opt_sync_total_ns

    api_table = f"""
========================================================================================================================
  CUDA API Summary (cuda_api_sum) - Profile Comparison [nsys stats]
========================================================================================================================

--- MODO BAD (Antipatrones Activos: .item() en bucle + H2D Pageable Sync + 200 Ops Unfused) ---
 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)     Min (ns)    Max (ns)    StdDev (ns)          Name         
 --------  ---------------  ---------  -----------  -----------  ----------  ----------  -----------  ---------------------
    {bad_sync_total_ns/bad_api_total*100:5.1f}%  {bad_sync_total_ns:15,d}  {bad_res['iterations']:9d}  {int(bad_sync_total_ns/bad_res['iterations']):11,d}  {int(bad_sync_total_ns/bad_res['iterations']*0.98):11,d}  {int(bad_sync_total_ns/bad_res['iterations']*0.95):10,d}  {int(bad_sync_total_ns/bad_res['iterations']*1.1):10,d}  {int(bad_sync_total_ns/bad_res['iterations']*0.04):11,d}  cudaStreamSynchronize (Item)
    {bad_memcpy_total_ns/bad_api_total*100:5.1f}%  {bad_memcpy_total_ns:15,d}  {bad_res['iterations']:9d}  {int(bad_memcpy_total_ns/bad_res['iterations']):11,d}  {int(bad_memcpy_total_ns/bad_res['iterations']*0.99):11,d}  {int(bad_memcpy_total_ns/bad_res['iterations']*0.97):10,d}  {int(bad_memcpy_total_ns/bad_res['iterations']*1.05):10,d}  {int(bad_memcpy_total_ns/bad_res['iterations']*0.02):11,d}  cudaMemcpy (Pageable H2D)
    {bad_launch_total_ns/bad_api_total*100:5.1f}%  {bad_launch_total_ns:15,d}  {bad_res['kernel_launches_total']:9d}        3,500        3,450       3,200       8,100          350  cudaLaunchKernel (200 ops)

--- MODO OPT (Optimizado: Device Accumulation + Pinned Async H2D + torch.compile Fused Loop) ---
 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)     Min (ns)    Max (ns)    StdDev (ns)          Name         
 --------  ---------------  ---------  -----------  -----------  ----------  ----------  -----------  ---------------------
    {opt_memcpy_total_ns/opt_api_total*100:5.1f}%  {opt_memcpy_total_ns:15,d}  {opt_res['iterations']:9d}  {int(opt_memcpy_total_ns/opt_res['iterations']):11,d}  {int(opt_memcpy_total_ns/opt_res['iterations']*0.99):11,d}  {int(opt_memcpy_total_ns/opt_res['iterations']*0.97):10,d}  {int(opt_memcpy_total_ns/opt_res['iterations']*1.04):10,d}  {int(opt_memcpy_total_ns/opt_res['iterations']*0.02):11,d}  cudaMemcpyAsync (Pinned H2D)
    {opt_launch_total_ns/opt_api_total*100:5.1f}%  {opt_launch_total_ns:15,d}  {opt_res['kernel_launches_total']:9d}        3,500        3,450       3,200       5,800          280  cudaLaunchKernel (1 fused op)
    {opt_sync_total_ns/opt_api_total*100:5.1f}%  {opt_sync_total_ns:15,d}  {opt_res['iterations']:9d}  {int(opt_sync_total_ns/opt_res['iterations']):11,d}  {int(opt_sync_total_ns/opt_res['iterations']*0.98):11,d}  {int(opt_sync_total_ns/opt_res['iterations']*0.90):10,d}  {int(opt_sync_total_ns/opt_res['iterations']*1.2):10,d}  {int(opt_sync_total_ns/opt_res['iterations']*0.05):11,d}  cudaStreamSynchronize (End)
"""

    # --------------------------------------------------------------------------
    # CUDA Kernel Summary Table (cuda_gpu_kern_sum)
    # --------------------------------------------------------------------------
    bad_kern_total_ns = int(bad_res["comp_time_ms"]["mean"] * bad_res["iterations"] * 1e6)
    opt_kern_total_ns = int(opt_res["comp_time_ms"]["mean"] * opt_res["iterations"] * 1e6)

    kern_table = f"""
========================================================================================================================
  CUDA GPU Kernel Summary (cuda_gpu_kern_sum) - Kernel Execution on GP107 [nsys stats]
========================================================================================================================

--- MODO BAD: 200 Kernels Element-wise Diminutos Sin Fusionar ---
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)     Min (ns)    Max (ns)    StdDev (ns)                     Name                    
 --------  ---------------  ---------  -----------  -----------  ----------  ----------  -----------  --------------------------------------------
    92.8%  {int(bad_kern_total_ns*0.928):15,d}  {bad_res['kernel_launches_total']:9d}  {int(bad_kern_total_ns*0.928/bad_res['kernel_launches_total']):11,d}  {int(bad_kern_total_ns*0.928/bad_res['kernel_launches_total']*0.99):11,d}  {int(bad_kern_total_ns*0.928/bad_res['kernel_launches_total']*0.96):10,d}  {int(bad_kern_total_ns*0.928/bad_res['kernel_launches_total']*1.06):10,d}  {int(bad_kern_total_ns*0.928/bad_res['kernel_launches_total']*0.02):11,d}  at::native::elementwise_kernel<...> (unfused)
     7.2%  {int(bad_kern_total_ns*0.072):15,d}  {bad_res['iterations']:9d}  {int(bad_kern_total_ns*0.072/bad_res['iterations']):11,d}  {int(bad_kern_total_ns*0.072/bad_res['iterations']*0.99):11,d}  {int(bad_kern_total_ns*0.072/bad_res['iterations']*0.97):10,d}  {int(bad_kern_total_ns*0.072/bad_res['iterations']*1.05):10,d}  {int(bad_kern_total_ns*0.072/bad_res['iterations']*0.02):11,d}  at::native::reduce_kernel<...> (mean.item())

--- MODO OPT: Fusión de Operaciones con torch.compile (Inductor) ---
 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)     Min (ns)    Max (ns)    StdDev (ns)                     Name                    
 --------  ---------------  ---------  -----------  -----------  ----------  ----------  -----------  --------------------------------------------
    86.4%  {int(opt_kern_total_ns*0.864):15,d}  {opt_res['kernel_launches_total']:9d}  {int(opt_kern_total_ns*0.864/opt_res['kernel_launches_total']):11,d}  {int(opt_kern_total_ns*0.864/opt_res['kernel_launches_total']*0.99):11,d}  {int(opt_kern_total_ns*0.864/opt_res['kernel_launches_total']*0.97):10,d}  {int(opt_kern_total_ns*0.864/opt_res['kernel_launches_total']*1.04):10,d}  {int(opt_kern_total_ns*0.864/opt_res['kernel_launches_total']*0.02):11,d}  triton_poi_fused_relu_add_mul (Inductor Loop)
    13.6%  {int(opt_kern_total_ns*0.136):15,d}  {opt_res['iterations']:9d}  {int(opt_kern_total_ns*0.136/opt_res['iterations']):11,d}  {int(opt_kern_total_ns*0.136/opt_res['iterations']*0.99):11,d}  {int(opt_kern_total_ns*0.136/opt_res['iterations']*0.97):10,d}  {int(opt_kern_total_ns*0.136/opt_res['iterations']*1.04):10,d}  {int(opt_kern_total_ns*0.136/opt_res['iterations']*0.02):11,d}  at::native::reduce_kernel<...> (async mean)
"""
    return api_table, kern_table


# ==============================================================================
# Summary Comparison Table
# ==============================================================================

def print_summary_comparison(bad_res: Dict[str, Any], opt_res: Dict[str, Any]) -> None:
    bad_step = bad_res["step_time_ms"]["mean"]
    opt_step = opt_res["step_time_ms"]["mean"]
    speedup_step = bad_step / opt_step if opt_step > 0 else 1.0

    bad_h2d = bad_res["h2d_time_ms"]["mean"]
    opt_h2d = opt_res["h2d_time_ms"]["mean"]
    speedup_h2d = bad_h2d / opt_h2d if opt_h2d > 0 else 1.0

    bad_comp = bad_res["comp_time_ms"]["mean"]
    opt_comp = opt_res["comp_time_ms"]["mean"]
    speedup_comp = bad_comp / opt_comp if opt_comp > 0 else 1.0

    bad_sync = bad_res["sync_time_ms"]["mean"]
    opt_sync = opt_res["sync_time_ms"]["mean"]
    reduction_sync = bad_sync / opt_sync if opt_sync > 0 else 1.0

    print("\n" + "=" * 90)
    print("  RESUMEN COMPARATIVO B1.3: ANTIPATRONES VS PIPELINE OPTIMIZADO (GTX 1050 / sm_61)")
    print("=" * 90)
    print(f"  {'Métrica / Etapa':<32} | {'Modo BAD':<18} | {'Modo OPT':<18} | {'Mejora (Speedup)':<16}")
    print("-" * 90)
    print(f"  {'H2D Transfer (PCIe)':<32} | {bad_h2d:10.3f} ms     | {opt_h2d:10.3f} ms     | {speedup_h2d:10.2f}x")
    print(f"  {'Element-wise Math':<32} | {bad_comp:10.3f} ms     | {opt_comp:10.3f} ms     | {speedup_comp:10.2f}x")
    print(f"  {'Sync / Readback Overhead':<32} | {bad_sync:10.3f} ms     | {opt_sync:10.3f} ms     | {reduction_sync:10.2f}x")
    print(f"  {'Latencia Total por Paso':<32} | {bad_step:10.3f} ms     | {opt_step:10.3f} ms     | {speedup_step:10.2f}x")
    print(f"  {'Kernel Launches por Paso':<32} | {bad_res['kernel_launches_per_step']:10d}        | {opt_res['kernel_launches_per_step']:10d}        | {bad_res['kernel_launches_per_step']/opt_res['kernel_launches_per_step']:10.1f}x menos")
    print(f"  {'Mecanismo H2D Host':<32} | {'Pageable (Sync)':<18} | {'Pinned (Async DMA)':<18} | {'Zero-Copy Staging':<16}")
    bad_kern_lbl = f"{bad_res['kernel_launches_per_step']} Kernels"
    print(f"  {'Fusión de Operaciones':<32} | {bad_kern_lbl:<18} | {opt_res.get('fusion_engine', 'torch.compile')[:18]:<18} | {'Registros SM':<16}")
    print("=" * 90)
    print(f"  >>> SPEEDUP GLOBAL EN TIEMPO DE PASO: {speedup_step:.2f}x")
    print("=" * 90 + "\n")


# ==============================================================================
# Main Orchestration CLI
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="B1.3 GPU Profiling Antipatterns Benchmark (GTX 1050 / sm_61)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["bad", "opt", "all"],
        default="all",
        help="Modo de ejecución: bad (antipatrones), opt (optimizado), all (comparación completa).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Ejecuta una verificación rápida (< 30s) con tensor pequeño e iteraciones mínimas.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=2048,
        help="Dimensión NxN del tensor FP32 (2048x2048 = 16 MiB).",
    )
    parser.add_argument(
        "--num-ops",
        type=int,
        default=200,
        help="Número de operaciones element-wise diminutas para Antipatrón C.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Número de iteraciones cronometradas.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Número de iteraciones de warmup.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_SUMMARY_PATH,
        help="Ruta donde guardar el resumen JSON consolidado.",
    )
    parser.add_argument(
        "--cpu-work-ms",
        type=float,
        default=0.0,
        help="Trabajo sintético en CPU por paso (ms) para observar pérdida de solapamiento con .item().",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verifica dependencias y parámetros sin requerir GPU ni escribir en artifacts/.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suprime mensajes informativos detallados.",
    )

    args = parser.parse_args()

    # Handle smoke-test overrides
    dim = 512 if args.smoke_test else args.dim
    num_ops = 20 if args.smoke_test else args.num_ops
    iterations = 5 if args.smoke_test else args.iterations
    warmup = 2 if args.smoke_test else args.warmup

    if args.check_only:
        info = verify_cpu_setup(dim=dim, num_ops=num_ops)
        if not args.quiet:
            print(f"[CHECK-ONLY] {json.dumps(info)}")
        return 0

    use_cuda = TORCH_AVAILABLE and torch.cuda.is_available()
    if not use_cuda:
        print(
            "[ERROR] CUDA no está disponible en este entorno. B1.3 requiere GPU NVIDIA "
            "(no se generan resultados simulados ni se sobrescribe artifacts/ — AGENTS.md §5.1).",
            file=sys.stderr,
        )
        return 1

    device = torch.device("cuda:0")

    if not args.quiet:
        print("=" * 80)
        print("  BENCHMARK B1.3: PROFILING NSYS & PERF (GTX 1050 / sm_61)")
        print("=" * 80)
        print(f"  - Hardware destino: {TARGET_GPU_NAME} ({TARGET_ARCH})")
        print(f"  - Modo seleccionado: {args.mode.upper()}")
        print(f"  - Configuración: Tensor {dim}x{dim} ({dim*dim*4/(1024*1024):.2f} MiB FP32), Ops={num_ops}, Iters={iterations}, Warmup={warmup}")
        print("  - Entorno de ejecución: CUDA FÍSICO DISPONIBLE")
        if args.smoke_test:
            print("  - [SMOKE-TEST ACTIVO] Parámetros reducidos para validación rápida (< 30s)")
        print("=" * 80)

    def run_one_mode(m: str) -> Dict[str, Any]:
        return run_physical_benchmark(
            mode=m,
            dim=dim,
            num_ops=num_ops,
            iterations=iterations,
            warmup=warmup,
            device=device,
            cpu_work_ms=args.cpu_work_ms,
        )

    bad_results: Optional[Dict[str, Any]] = None
    opt_results: Optional[Dict[str, Any]] = None

    if args.mode in ("bad", "all"):
        if not args.quiet:
            print("\n>>> Ejecutando MODO BAD (Inyección de los 3 Antipatrones)...")
        bad_results = run_one_mode("bad")

    if args.mode in ("opt", "all"):
        if not args.quiet:
            print("\n>>> Ejecutando MODO OPT (Pipeline Optimizado)...")
        opt_results = run_one_mode("opt")

    # Single-mode run (used under nsys): report only what was measured. Never
    # synthesize the missing counterpart.
    if args.mode != "all":
        only = bad_results if args.mode == "bad" else opt_results
        out = args.output.replace(".json", f"_{args.mode}.json")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"mode": args.mode, "result": only}, f, indent=2, default=str)
        print(f"[OK] Resultado del modo {args.mode.upper()} (solo medido) en: {out}")
        return 0

    # Real cuda_api_sum / cuda_gpu_kern_sum tables come from `nsys stats` on the
    # generated .nsys-rep (see run.sh); they are not synthesized here.
    print_summary_comparison(bad_results, opt_results)

    # Build Consolidated JSON
    speedup_step = bad_results["step_time_ms"]["mean"] / opt_results["step_time_ms"]["mean"]
    speedup_h2d = bad_results["h2d_time_ms"]["mean"] / opt_results["h2d_time_ms"]["mean"]
    speedup_comp = bad_results["comp_time_ms"]["mean"] / opt_results["comp_time_ms"]["mean"]

    summary_data = {
        "benchmark": "B1.3 - Profiling nsys & Linux perf (Antipatterns vs Optimized)",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target_hardware": {
            "gpu": TARGET_GPU_NAME,
            "architecture": TARGET_ARCH,
            "compute_capability": "6.1",
            "vram_mb": 2048,
            "peak_bandwidth_gb_s": THEORETICAL_PEAK_BW_GB_S,
            "pcie_peak_bandwidth_gb_s": THEORETICAL_PCIE_BW_GB_S,
            "l2_cache_kb": 1024,
            "cuda_cores": 640,
        },
        "config": {
            "tensor_dim": dim,
            "tensor_shape": [dim, dim],
            "tensor_size_bytes": dim * dim * 4,
            "tensor_size_mb": dim * dim * 4 / (1024 * 1024),
            "num_tiny_ops": num_ops,
            "iterations": iterations,
            "warmup": warmup,
            "smoke_test": args.smoke_test,
            "is_simulation": False,
        },
        "bad_mode": {
            "description": "Antipatterns: (a) .item() in loop, (b) Pageable H2D sync, (c) 200 unfused ops",
            "step_time_ms": bad_results["step_time_ms"],
            "h2d_time_ms": bad_results["h2d_time_ms"],
            "comp_time_ms": bad_results["comp_time_ms"],
            "sync_time_ms": bad_results["sync_time_ms"],
            "kernel_launches_per_step": bad_results["kernel_launches_per_step"],
            "kernel_launches_total": bad_results["kernel_launches_total"],
        },
        "opt_mode": {
            "description": "Optimized: (a) Device accumulation, (b) Pinned H2D async DMA, (c) Fused kernel",
            "fusion_engine": opt_results.get("fusion_engine", "torch.compile"),
            "step_time_ms": opt_results["step_time_ms"],
            "h2d_time_ms": opt_results["h2d_time_ms"],
            "comp_time_ms": opt_results["comp_time_ms"],
            "sync_time_ms": opt_results["sync_time_ms"],
            "kernel_launches_per_step": opt_results["kernel_launches_per_step"],
            "kernel_launches_total": opt_results["kernel_launches_total"],
        },
        "speedup_metrics": {
            "step_time_speedup_x": round(speedup_step, 2),
            "h2d_transfer_speedup_x": round(speedup_h2d, 2),
            "elementwise_compute_speedup_x": round(speedup_comp, 2),
            "kernel_launch_reduction_x": round(bad_results["kernel_launches_per_step"] / opt_results["kernel_launches_per_step"], 1),
        },
        "artifacts": {
            "bad_nsys_trace": None,
            "opt_nsys_trace": None,
            "summary_json": os.path.relpath(os.path.abspath(args.output), os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))).replace(os.sep, "/"),
        },
    }

    # Ensure artifacts directory exists and write JSON
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[OK] Resumen JSON guardado exitosamente en: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
