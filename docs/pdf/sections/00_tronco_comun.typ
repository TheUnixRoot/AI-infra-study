#import "../theme.typ": *

= Tronco Común: Lo Común, Lo Específico y Tres Itinerarios

Partiendo de un perfil *Senior DevOps / SRE* (Kubernetes, CI/CD, IaC en AWS/Azure, gestión de secretos/identidades, SLOs y Prometheus/Grafana) junto con base en sistemas C++/CUDA (`LogFit` y bloque local `B1.1`–`B2.1`), los tres itinerarios evitan duplicar fundamentos y concentran el esfuerzo en cerrar brechas específicas de *inferencia y plataformas de IA* sobre un *Tronco Común de 6 pilares* (~35–40 % transversal).

#infografia(
  "Mapa Unificado: Baseline Técnico -> Tronco Común (6 Pilares) -> Tres Itinerarios",
  "Lo común (6 pilares transversales en el centro) alimenta las tres especializaciones: Datacenter GPU (Curso 1), Plataforma Agéntica (Curso 2) y Edge-AI Sub-2-Bit & Kernels Portables (Curso 3).",
  height: 245pt,
)[
  // Baseline superior
  #diag-node(120pt, 6pt, 230pt, 28pt, "BASELINE: Senior DevOps/SRE + C++/CUDA", sub: "K8s · CI/CD · Prometheus · CUDA sm_61 (B1.1–B2.1) · LogFit", fill: bg-light, col: slate)
  #arrow-d(235pt, 34pt, 50pt, col: slate)

  // Tronco común central
  #place(dx: 10pt, dy: 50pt, rect(width: 450pt, height: 68pt, fill: rgb("#F8FAFC"), stroke: 0.9pt + teal, radius: 5pt))
  #place(dx: 18pt, dy: 54pt, text(fill: teal, weight: "bold", size: 7.8pt)[TRONCO COMÚN · LO COMÚN (6 PILARES TRANSVERSALES)])
  #diag-node(18pt, 68pt, 140pt, 20pt, "1. Prefill/Decode & KV-Cache", fill: bg-teal, col: teal)
  #diag-node(165pt, 68pt, 140pt, 20pt, "2. Agentes, Tools & JSON", fill: bg-teal, col: teal)
  #diag-node(312pt, 68pt, 140pt, 20pt, "3. Rigor Científico & Pareto", fill: bg-teal, col: teal)
  #diag-node(18pt, 92pt, 140pt, 20pt, "4. Trazabilidad & FinOps/mWh", fill: bg-teal, col: teal)
  #diag-node(165pt, 92pt, 140pt, 20pt, "5. Serving & Portabilidad", fill: bg-teal, col: teal)
  #diag-node(312pt, 92pt, 140pt, 20pt, "6. Micro-Sprints 20m + Handover", fill: bg-teal, col: teal)

  // Trifurcación hacia Cursos 1, 2 y 3
  #arrow-d(84pt, 118pt, 140pt, col: navy)
  #arrow-d(235pt, 118pt, 140pt, col: purple)
  #arrow-d(386pt, 118pt, 140pt, col: amber)

  // Curso 1
  #diag-node(10pt, 140pt, 146pt, 96pt,
    "CURSO 1: AI Infra GPU",
    sub: "B1 CUDA/nsys -> B2 vLLM/SGLang\nB3 FP8/AWQ/SpecDecode\nB4 MoE/WideEP/PDD/NCCL\nB5 DCGM/MFU/K8s MIG\nProj A + B (Cloud H100/L40S)",
    fill: bg-navy, col: navy)

  // Curso 2
  #diag-node(162pt, 140pt, 146pt, 96pt,
    "CURSO 2: Agentic MLOps",
    sub: "B1 Databricks/MLflow/UC\nB2 Evals/RAG/OTel/FinOps\nB3 Seldon/Istio/Argo/Crossplane\nB4 SRE SLOs/Multi-Región UE\nProj: agentic-golden-path",
    fill: bg-purple, col: purple)

  // Curso 3
  #diag-node(314pt, 140pt, 146pt, 96pt,
    "CURSO 3: Edge-AI (0 EUR)",
    sub: "E0-E1 mWh/tok + Bonsai 2\nE2 Hadamard + GPTQ CPU\nE3 Gemma 4 E2B (<1 GB)\nE4 CUDA dp4a + Vulkan GLSL\nE5 <8W + LogFit-Edge + UMA",
    fill: bg-amber, col: amber)
]

== Matriz Comparativa de los Tres Itinerarios

#styled-table(
  columns: (0.85fr, 1.35fr, 1.35fr, 1.35fr),
  headers: ("Eje", "Curso 1: AI Infra & GPU", "Curso 2: Agentic MLOps", "Curso 3: Edge-AI Sub-2-Bit"),
  rows: (
    ([*Roles Objetivo*], [GPU Platform Benchmarking · Senior MLE Inference], [Staff/Senior MLOps · Agentic Platform Engineer], [Senior Edge Inference MLE · Portable Kernel Engineer (HPAC/UMA)]),
    ([*Hardware / Entorno*], [GTX 1050 `sm_61` (`B1`–`B2.1`) + Cloud (`L4`/`L40S`/`H100`)], [CPU local (Docker, Kind, MLflow) + Databricks Trial], [*0 EUR:* GTX 1050 `sm_61` + CPU 16GB+ + Android Termux (`ARM64` + Vulkan)]),
    ([*Métrica Reina*], [TTFT, TPOT, tok/s/GPU, `busbw`, MFU, Pareto `\$/1M`], [Task Success, Tool Reliability, `Recall@K`, SLOs, Audit], [`mWh/token` (W netos), `% BW pico`, `PPL`, `KLD`, `GSM8K`/`IFEval`, `€/año`]),
    ([*Entregables*], [`agentic-inference-opt-lab` + `gpu-cluster-acceptance`], [`agentic-platform-golden-path` (Audit-Ready + GitOps)], [`edge-ternary-lab` + `Logfit` README + Dossier UMA 1 pág.]),
  ),
)

== Los 6 Pilares del Tronco Común (Lo Común)

#styled-table(
  columns: (0.8fr, 1.25fr, 1.25fr, 1.25fr),
  headers: ("Pilar", "Curso 1 (Datacenter GPU)", "Curso 2 (Enterprise MLOps)", "Curso 3 (Edge-AI Sub-2-Bit)"),
  rows: (
    ([*1. Prefill/Decode & KV*],
     [Roofline $I="FLOPs"/"Bytes"$, KV FP8, MLA (`DeepSeek`) y `PagedAttention`.],
     [Capacity planning (`TPM`/`RPM` vs VRAM) y latencia cross-region.],
     [Techo $"BW"/"Bytes"$ en `112 GB/s` y móvil; `1.73 bpp` + KV `q4_0` (`< 1 GB`).]),
    ([*2. Agentes, Tools & JSON*],
     [*Prefix Caching* (`APC`/`Radix`) y `XGrammar` FSM en concurrencia alta.],
     [Trayectorias agénticas, `mlflow.evaluate()` y salud RAG (`index_lag`).],
     [Calidad real (`GSM8K`, `IFEval`, JSON, `<|think|>`) y fallos `lm_head`/`PLE`.]),
    ([*3. Rigor y Pareto*],
     [Warmup, carga abierta (sin *coordinated omission*) y Pareto `\$/1M`.],
     [Gates bloqueantes CI (`Tool >= 95%`) y Canary auto-rollback en Argo.],
     [Muestreo `sysfs`/`NVML` con base *idle* (var. `< 5%`) y Pareto `mWh/tok` vs `KLD`.]),
    ([*4. Trazas y Economía*],
     [`nsys` + NVTX, `ncu` roofline, `perf` y métrica `USD / 1M tokens`.],
     [OTel GenAI spans, MLflow Tracing, OpenSearch y *circuit breakers* FinOps.],
     [`nsys` sobre `GEMV` `sm_61` (`> 65–75%` pico), potencia neta (W) y `€/año`.]),
    ([*5. Serving & Portabilidad*],
     [K8s GPU Operator, `MIG`, `LeaderWorkerSet`, `PDD`/`NIXL` y *cold starts*.],
     [Seldon v2, Knative KPA, Istio shadow/canary, Crossplane XRDs y SRE UE.],
     [CUDA `sm_61` $arrow.r$ Vulkan GLSL (*zero-copy* móvil), *LogFit-Edge* y API `< 8 W`.]),
    ([*6. Sistema 20m*],
     [`CHAT-B1..B5`, `[PREP]`$arrow.r$`[CLD]`, `run.sh --smoke-test`, `shutdown +90`.],
     [`CHAT-B1..B4`, `[MOB]/[LOC]/[LAB]`, `verify.sh < 60s` y `audit-evidence`.],
     [`[MOB]/[LOC-PC]/[LOC-AND]/[AGENT]`, Pre-Flight con tests y 1 commit/sprint.]),
  ),
)

== Lo Específico y Enganche Temporal (`Curso 1` $arrow.r$ `Curso 3`)

#concepto-clave(title: "Frontera entre lo Específico de Datacenter (C1), Plataforma (C2) y Edge (C3)")[
  - *Exclusivo Curso 1:* `OpenAI Triton`, `ncu` (`sm_70+`), `vLLM`/`SGLang`/`TensorRT-LLM`, `FP8` en Tensor Cores, `Speculative Decoding` (`EAGLE-3`/`MTP`), paralelismo multi-GPU (`TP`/`PP`/`EP`, `NCCL`, `PDD`) y aceptación `dcgmi`/`Xid`/`Megatron MFU`.
  - *Exclusivo Curso 2:* `Databricks Mosaic AI`, `MLflow 3` (`LoggedModel` + aliases), `Unity Catalog` (`lineage`, `RBAC`/`ABAC`), `Seldon`/`Knative`/`Istio`/`Argo`/`Crossplane` y *failover* soberano RGPD en la UE.
  - *Exclusivo Curso 3 (`B3-Edge`):* empaquetado base-3 `TQ1_0` sin divisiones (`(xi * 3) >> 8`), rotación ortogonal *Fast Hadamard Transform* (`FHT`), `GPTQ` con Cholesky del Hessiano en CPU (`< 10 GB` RAM), anatomía `Gemma 4 E2B` (`PLE`, `vision_tower`, `audio_tower`, `lm_head`), kernels desde cero con *`__dp4a`* (`sm_61`) y *Vulkan Compute GLSL* (*subgroups* + memoria unificada *zero-copy*), despliegue doméstico `< 8 W` y puente académico con `TheUnixRoot/Logfit` (dossier HPAC/UMA).
]

#practica-box(title: "Checklist de Enganche sin Quemarte (Curso 1 ↔ Curso 3)")[
  1. *Mientras cierras Bloque 2 de Curso 1 (`B2.1`–`B2.4`):* ejecuta solo `E0` y `E1` en ratos sueltos (compilar `llama.cpp`/PrismML + calibrar `mWh/token`).
  2. *Al arrancar Bloque 3 (`B3.1` Cuantización) de Curso 1:* ejecuta `E2` y `E3` en CPU local (`0 EUR`) antes de alquilar `Cloud-3` (`H100`).
  3. *Tras cerrar Bloque 3 de Curso 1:* ejecuta `E4` (kernels CUDA `sm_61` + Vulkan GLSL) y `E5` (servidor `< 8 W` + *LogFit-Edge* + dossier para Rafael Asenjo).
]
