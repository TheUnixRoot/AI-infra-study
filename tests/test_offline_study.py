#!/usr/bin/env python3
"""Offline unit tests for AI-infra-study (CPU-only, Termux-safe, no GPU required)."""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


kv_calc = load_module("kv_calc", "b2_1_kv_cache/kv_calc.py")
b1_1 = load_module("b1_1_strides_bench", "b1_1_strides/b1_1_strides_bench.py")
b1_2_parser = load_module("parse_results", "b1_2_cuda_gemm/parse_results.py")
b1_3 = load_module("b1_3_antipatterns", "b1_3_profiling/b1_3_antipatterns.py")
b2_2_client = load_module("bench_client", "b2_2_vllm_vs_sglang/bench_client.py")


class TestB21KvCalc(unittest.TestCase):
    def test_gqa_and_mla_arithmetic(self) -> None:
        llama_arch = kv_calc.analyze_model_architecture(
            kv_calc.EMBEDDED_CONFIGS["meta-llama/Meta-Llama-3-8B"]
        )
        self.assertEqual(llama_arch["attn_type"], "GQA")
        self.assertEqual(llama_arch["total_kv_elems_per_token"], 2 * 32 * 8 * 128)

        ds_arch = kv_calc.analyze_model_architecture(
            kv_calc.EMBEDDED_CONFIGS["deepseek-ai/DeepSeek-V3"]
        )
        self.assertEqual(ds_arch["attn_type"], "MLA")
        self.assertEqual(ds_arch["elems_per_token_per_layer"], 512 + 64)
        self.assertEqual(ds_arch["total_kv_elems_per_token"], 61 * 576)

    def test_qwen72b_oom_when_less_than_one_seq(self) -> None:
        qwen_arch = kv_calc.analyze_model_architecture(
            kv_calc.EMBEDDED_CONFIGS["Qwen/Qwen2.5-72B-Instruct"]
        )
        combos = kv_calc.calculate_vram_and_concurrency(
            qwen_arch, seq_len=8192, gpu_mem_util=0.90, overhead_gb=2.0
        )
        fp8_fp16 = next(c for c in combos if c["combo_name"] == "Pesos FP8 + KV FP16")
        h100 = fp8_fp16["hardware"]["H100 (80 GB)"]
        self.assertEqual(h100["max_concurrency"], 0)
        self.assertEqual(h100["status"], "OOM: Available KV < 1 seq")

        fp8_fp8 = next(c for c in combos if c["combo_name"] == "Pesos FP8 + KV FP8")
        self.assertEqual(fp8_fp8["hardware"]["H100 (80 GB)"]["max_concurrency"], 1)
        self.assertEqual(fp8_fp8["hardware"]["H100 (80 GB)"]["status"], "OK")

    def test_parameter_estimator_head_dim_and_moe(self) -> None:
        moe_cfg = {
            "vocab_size": 1000,
            "hidden_size": 256,
            "intermediate_size": 512,
            "moe_intermediate_size": 128,
            "num_hidden_layers": 4,
            "first_k_dense_replace": 1,
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
        }
        dense_cfg = {
            "vocab_size": 1000,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
        }
        self.assertGreater(
            kv_calc.estimate_parameters_from_config(moe_cfg),
            kv_calc.estimate_parameters_from_config(dense_cfg),
        )

    def test_seq_len_warning_against_max_position_embeddings(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "b2_1_kv_cache" / "kv_calc.py"),
                "--smoke-test",
                "--models",
                "llama3-8b",
                "--seq-len",
                "16384",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("max_position_embeddings", proc.stderr)

    def test_artifacts_b2_1_consistency(self) -> None:
        report = json.loads((ROOT / "artifacts" / "b2_1_kv_report.json").read_text(encoding="utf-8"))
        self.assertEqual(len(report["results"]), 3)
        with (ROOT / "artifacts" / "b2_1_kv_report.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 9)
        qwen_row = next(
            r for r in rows
            if r["model_name"] == "Qwen/Qwen2.5-72B-Instruct"
            and r["precision_combo"] == "Pesos FP8 + KV FP16"
        )
        self.assertEqual(qwen_row["h100_max_concurrency"], "0")
        self.assertEqual(qwen_row["h100_status"], "OOM: Available KV < 1 seq")


class TestB11StridesOffline(unittest.TestCase):
    def test_smoke_test_uses_512_unless_explicit_size(self) -> None:
        self.assertEqual(b1_1.resolve_run_config(True, None, 20, 50), (512, 5, 10))
        self.assertEqual(b1_1.resolve_run_config(True, 1024, 20, 50), (1024, 5, 10))
        self.assertEqual(b1_1.resolve_run_config(False, None, 20, 50), (2048, 20, 50))

    def test_cpu_verification_no_fake_gpu_timings(self) -> None:
        res = b1_1.run_cpu_verification(size=64, warmup_iters=2, bench_iters=4)
        self.assertFalse(res.cuda_available)
        self.assertFalse(res.is_simulation)
        self.assertTrue(res.y_ptr_match)
        self.assertFalse(res.z_ptr_match)
        self.assertEqual(res.y_stride, (1, 64))
        self.assertEqual(res.z_traffic_bytes, 2 * 64 * 64 * 4)
        self.assertEqual(res.z_gpu_time_ms, {})


class TestB12ParserAndCudaSource(unittest.TestCase):
    def test_parse_results_supports_16x16_variant(self) -> None:
        gemm_md = "\n".join([
            "| Kernel GEMM | Dimensión (MxNxK) | Latencia (ms) | Rendimiento (GFLOP/s) | Speedup vs Naive | Verificación (Tol 1e-3) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            "| **cublasSgemm (Ref) ** | 1024x1024x1024 |      1.373 |            1563.73 |         53.51x | PASS [OK] (err: 0.00e+00) |",
            "| **gemm_naive        ** | 1024x1024x1024 |     73.490 |              29.22 |          1.00x | PASS [OK] (err: 0.00e+00) |",
            "| **gemm_coalesced    ** | 1024x1024x1024 |     17.955 |             119.60 |          4.09x | PASS [OK] (err: 0.00e+00) |",
            "| **gemm_smem_tiled_16** | 1024x1024x1024 |      8.100 |             265.12 |          9.07x | PASS [OK] (err: 0.00e+00) |",
            "| **gemm_smem_tiled   ** | 1024x1024x1024 |      7.589 |             282.97 |          9.68x | PASS [OK] (err: 0.00e+00) |",
        ])
        soft_md = "\n".join([
            "| **Standard 3-Pass (Warp Shfl)** | 1024x1024 |     0.1752 |           47.88 |           1.00x |            1.86e-09 | PASS [OK] |",
            "| **Online 1-Pass (Warp Shfl)**   | 1024x1024 |     0.1339 |           62.67 |           1.31x |            2.79e-09 | PASS [OK] |",
        ])
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            (td_p / "g.out").write_text(gemm_md, encoding="utf-8")
            (td_p / "s.out").write_text(soft_md, encoding="utf-8")
            b1_2_parser.main(
                str(td_p / "g.out"),
                str(td_p / "s.out"),
                "full",
                "NVIDIA L40S",
                "550.0",
                str(td_p / "res.json"),
                str(td_p / "res.csv"),
                arch="sm_89",
            )
            data = json.loads((td_p / "res.json").read_text(encoding="utf-8"))
            self.assertEqual(data["target_hardware"]["architecture"], "sm_89")
            kernels = [r["kernel"] for r in data["gemm"]]
            self.assertIn("gemm_smem_tiled_16", kernels)
            self.assertEqual(len(kernels), 5)

    def test_cuda_sources_contain_fixes(self) -> None:
        gemm_src = (ROOT / "b1_2_cuda_gemm" / "gemm_bench.cu").read_text(encoding="utf-8")
        self.assertIn("gemm_smem_tiled_16", gemm_src)
        self.assertIn("cudaGetLastError", gemm_src)
        soft_src = (ROOT / "b1_2_cuda_gemm" / "warp_online_softmax.cu").read_text(encoding="utf-8")
        self.assertIn("(d == 0.0f)", soft_src)


class TestB13AndB22Offline(unittest.TestCase):
    def test_b1_3_check_only_and_kernel_count(self) -> None:
        info = b1_3.verify_cpu_setup(dim=512, num_ops=200)
        self.assertEqual(info["nominal_unfused_kernels_per_step"], 600)
        proc = subprocess.run(
            [sys.executable, str(ROOT / "b1_3_profiling" / "b1_3_antipatterns.py"), "--check-only"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("CHECK-ONLY", proc.stdout)

    def test_b2_2_prometheus_and_mix_guard(self) -> None:
        prom = 'sglang:token_usage{tp="0"} 0.25\n'
        parsed = b2_2_client.parse_prom(prom)
        summary = b2_2_client.metrics_summary([parsed])
        self.assertEqual(summary["kv_cache_pct_max"], 25.0)
        self.assertIsNone(summary["preemptions"])


class TestDocsAndSiteHygiene(unittest.TestCase):
    def test_katex_delimiters_escaped(self) -> None:
        js = (ROOT / "docs" / "javascripts" / "katex.js").read_text(encoding="utf-8")
        self.assertIn('left: "\\\\("', js)
        self.assertIn('left: "\\\\["', js)

    def test_mkdocs_and_agents_consistent(self) -> None:
        mk = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        for mod in ("b1_1_strides", "b1_2_cuda_gemm", "b1_3_profiling", "b2_1_kv_cache", "b2_2_vllm_vs_sglang"):
            self.assertIn(mod, mk)
        for doc in (
            "docs/temario/00-master-playbook.md",
            "docs/temario/01-agentic-mlops-playbook.md",
            "docs/temario/02-tronco-comun-y-itinerarios.md",
            "docs/temario/03-edge-ai-ternary-playbook.md",
        ):
            self.assertTrue((ROOT / doc).exists(), f"Missing syllabus file: {doc}")
            self.assertIn(doc, mk)
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertNotIn("_config.yml", agents)
        self.assertIn("mkdocs.yml", agents)

    def test_typst_manuals_and_pdfs_present(self) -> None:
        pdf_dir = ROOT / "docs" / "pdf"
        required_typ = [
            pdf_dir / "theme.typ",
            pdf_dir / "master_ai_infra_y_mlops.typ",
            pdf_dir / "curso1_ai_infra.typ",
            pdf_dir / "curso2_agentic_mlops.typ",
            pdf_dir / "curso3_edge_ai.typ",
            pdf_dir / "sections" / "00_tronco_comun.typ",
            *[pdf_dir / "sections" / f"c1_b{i}.typ" for i in range(1, 6)],
            *[pdf_dir / "sections" / f"c2_b{i}.typ" for i in range(1, 5)],
            *[pdf_dir / "sections" / f"c3_{s}.typ" for s in ("e0_e1", "e2", "e3", "e4", "e5_e6")],
        ]
        for typ_file in required_typ:
            self.assertTrue(typ_file.exists(), f"Missing Typst source: {typ_file}")
            self.assertGreater(typ_file.stat().st_size, 500)

        required_pdfs = [
            "Master_AI_Infra_y_Agentic_MLOps.pdf",
            "Curso_1_AI_Infrastructure_GPU.pdf",
            "Curso_2_Agentic_Platform_MLOps.pdf",
            "Curso_3_Edge_AI_Sub2Bit_Kernels.pdf",
        ]
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for pdf_name in required_pdfs:
            pdf_path = pdf_dir / pdf_name
            self.assertTrue(pdf_path.exists(), f"Missing compiled PDF: {pdf_path}")
            header = pdf_path.read_bytes()[:5]
            self.assertEqual(header, b"%PDF-")
            self.assertGreater(pdf_path.stat().st_size, 50_000)
            self.assertIn(pdf_name, readme)

    def test_campus_assets_and_strict_links_valid(self) -> None:
        mk = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        required_assets = [
            "docs/stylesheets/campus.css",
            "docs/javascripts/campus.js",
            "docs/campus/index.md",
            "docs/campus/tronco-comun.md",
            "docs/campus/progreso.md",
            "docs/campus/biblioteca-pdf.md",
            "docs/campus/curso1/index.md",
            "docs/campus/curso1/b1-arquitectura-cuda.md",
            "docs/campus/curso1/b2-motores-kv-cache.md",
            "docs/campus/curso1/b3-cuantizacion-pareto.md",
            "docs/campus/curso1/b4-paralelismo-moe-nccl.md",
            "docs/campus/curso1/b5-cluster-benchmarking.md",
            "docs/campus/curso2/index.md",
            "docs/campus/curso2/b1-databricks-mlflow-uc.md",
            "docs/campus/curso2/b2-evals-rag-observabilidad.md",
            "docs/campus/curso2/b3-serving-gitops-crossplane.md",
            "docs/campus/curso2/b4-sre-slos-multiregion.md",
            "docs/campus/curso3/index.md",
            "docs/campus/curso3/e0-e1-infra-baseline.md",
            "docs/campus/curso3/e2-ptq-hadamard-gptq.md",
            "docs/campus/curso3/e3-gemma4-evals-pareto.md",
            "docs/campus/curso3/e4-kernels-cuda-vulkan.md",
            "docs/campus/curso3/e5-e6-edge-server-logfit-qat.md",
        ]
        for rel in required_assets:
            self.assertTrue((ROOT / rel).exists(), f"Missing Campus file: {rel}")
            self.assertIn(rel, mk, f"Unregistered Campus file in mkdocs.yml: {rel}")

        campus_js = (ROOT / "docs" / "javascripts" / "campus.js").read_text(encoding="utf-8")
        for track_id in ('track: "tronco"', 'track: "c1"', 'track: "c2"', 'track: "c3"'):
            self.assertIn(track_id, campus_js)
        for sprint_id in ("C3-E0.1", "C3-E1.4", "C3-E2.6", "C3-E3.5", "C3-E4.7", "C3-E5.4", "C3-E6.3"):
            self.assertIn(sprint_id, campus_js)

        files = set(
            subprocess.check_output(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=ROOT,
                text=True,
            ).splitlines()
        )
        skip_files = {"CLAUDE.md", "GEMINI.md", "mkdocs.yml", ".gitattributes", ".gitignore", ".dockerignore"}
        copied = {
            f for f in files if f not in skip_files and not f.startswith(("tools/", ".github/")) and (ROOT / f).is_file()
        }
        link_re = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
        errors: list[str] = []
        for f in sorted(copied):
            if not f.endswith(".md"):
                continue
            text = (ROOT / f).read_text(encoding="utf-8")
            in_fence = False
            for idx, line in enumerate(text.splitlines(), 1):
                if line.lstrip().startswith("```"):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                for m in link_re.finditer(line):
                    target = m.group(1).strip().split("#")[0]
                    if not target or target.startswith(("http://", "https://", "mailto:", "file://")):
                        continue
                    resolved = (ROOT / f).parent.joinpath(target).resolve()
                    try:
                        rel = resolved.relative_to(ROOT.resolve()).as_posix()
                    except ValueError:
                        rel = str(resolved)
                    if resolved.is_dir():
                        errors.append(f"{f}:{idx} links to directory {target}")
                    elif rel not in copied:
                        errors.append(f"{f}:{idx} links to missing/untracked {target} ({rel})")
        self.assertEqual(errors, [], "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
