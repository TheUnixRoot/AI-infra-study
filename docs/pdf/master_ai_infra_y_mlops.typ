#import "theme.typ": *

#show: master-setup.with(
  title: "AI Infrastructure, Agentic MLOps & Edge-AI Sub-2-Bit",
  subtitle: "Edición Máster Unificada · Tronco Común + Curso 1 + Curso 2 + Curso 3",
  edition: "Edición Máster Unificada",
)

#cover-page(
  "Manual Técnico de Nivel Máster · Tres Especializaciones Integradas",
  "AI Infrastructure, Agentic MLOps & Edge-AI",
  "Del hardware CUDA, KV-Cache y paralelismo multi-GPU a plataformas agénticas empresariales e inferencia Edge sub-2-bit con kernels portables CUDA/Vulkan",
  [
    Este volumen unifica tres itinerarios complementarios construidos sobre un *Tronco Común de Ingeniería de IA (6 pilares)* y pensados para ejecutarse en *micro-sprints de 20 minutos* (`[MOB]`, `[LOC]`, `[LOC-PC]`, `[LOC-AND]`, `[AGENT]`, `[PREP]`, `[CLD]`, `[LAB]`):
    - *Curso 1 (Infraestructura GPU y Motores Cloud):* Anatomía HBM/L2, CUDA C++, `nsys`/`ncu`, cuantización FP8/AWQ, vLLM/SGLang/TRT-LLM y clústeres multi-GPU con NVLink/NCCL.
    - *Curso 2 (Enterprise MLOps & Agentic Platform):* Databricks Mosaic AI, MLflow 3, Unity Catalog, evaluación automática de agentes/RAG, OTel/FinOps, Seldon/Istio/Argo/Crossplane y SRE UE.
    - *Curso 3 (Edge-AI Sub-2-Bit & Kernels Portables · 0 EUR):* Réplica *open-source* de PrismML (`Qwen3.5-2B` $arrow.r$ `Gemma 4 E2B < 1 GB`), Hadamard + GPTQ en CPU, kernels `sm_61` `__dp4a` + Vulkan Compute GLSL, API `< 8 W` y puente HPAC/UMA (`LogFit-Edge`).
  ],
  (
    (
      title: "0 · Tronco Común (Lo Común vs Específico)",
      bg: bg-navy,
      col: navy,
      body: [
        6 pilares transversales (Prefill/Decode, Agentes/Tools, Rigor Pareto, Trazabilidad/FinOps, Serving/Portabilidad, OS 20m) y matriz comparativa de los 3 itinerarios.
      ],
    ),
    (
      title: "1 · Curso 1: AI Infrastructure (B1–B5)",
      bg: bg-teal,
      col: teal,
      body: [
        20 unidades (`B1.1`–`B5.4`) con mediciones reales en GTX 1050 (2 GB) y diseño `[PREP]` $arrow.r$ `[CLD]` para L4, L40S y H100 (`sm_89`/`sm_90`).
      ],
    ),
    (
      title: "2 · Curso 2: Agentic MLOps (B1–B4)",
      bg: bg-purple,
      col: purple,
      body: [
        12 unidades (`B1.1`–`B4.3`) cerrando al 100 % la brecha entre Senior DevOps/SRE y plataformas agénticas reguladas con `agentic-platform-golden-path`.
      ],
    ),
    (
      title: "3 · Curso 3: Edge-AI Sub-2-Bit (E0–E6)",
      bg: bg-amber,
      col: amber,
      body: [
        7 módulos (33 sprints a 0 EUR) en PC + Android: `mWh/token`, `TQ1_0`/`FHT`/`GPTQ`, `Gemma 4 E2B`, CUDA `__dp4a` + Vulkan GLSL y dossier UMA.
      ],
    ),
  ),
)

#outline(title: "Índice General de Contenidos", depth: 2, indent: 1.4em)

#pagebreak(weak: true)
#include "sections/00_tronco_comun.typ"

#pagebreak(weak: true)
= Parte I · Curso 1: AI Infrastructure, GPU Systems & Inference Engines
#block(width: 100%, fill: bg-teal, stroke: (left: 4pt + teal, rest: 0.5pt + border-gray), inset: 10pt, radius: 4pt)[
  *Objetivo del Itinerario:* Dominar el stack por debajo del modelo — desde los *strides* de PyTorch y el *tiling* en memoria compartida CUDA hasta *PagedAttention*, *FP8*, paralelismo tensorial/pipeline (`TP`/`PP`/`EP`) y diagnóstico multi-nodo con NCCL/RoCEv2 — combinando laboratorio local controlado (`GTX 1050 2 GB`) con ráfagas cloud preparadas de antemano (`[PREP]` $arrow.r$ `[CLD]`).
]
#v(6pt)
#include "sections/c1_b1.typ"
#pagebreak(weak: true)
#include "sections/c1_b2.typ"
#pagebreak(weak: true)
#include "sections/c1_b3.typ"
#pagebreak(weak: true)
#include "sections/c1_b4.typ"
#pagebreak(weak: true)
#include "sections/c1_b5.typ"

#pagebreak(weak: true)
= Parte II · Curso 2: Enterprise MLOps & Agentic Platform Engineering
#block(width: 100%, fill: bg-purple, stroke: (left: 4pt + purple, rest: 0.5pt + border-gray), inset: 10pt, radius: 4pt)[
  *Objetivo del Itinerario:* Transformar un perfil *Senior DevOps / SRE* en *Staff MLOps & Agentic Platform Engineer* para entornos regulados, construyendo el repositorio integrador `agentic-platform-golden-path`: gobernanza en Databricks Mosaic AI + MLflow 3 + Unity Catalog, evaluaciones agénticas/RAG, FinOps, serving en Seldon/Knative/Istio/ArgoCD/Crossplane y resiliencia multi-región.
]
#v(6pt)
#include "sections/c2_b1.typ"
#pagebreak(weak: true)
#include "sections/c2_b2.typ"
#pagebreak(weak: true)
#include "sections/c2_b3.typ"
#pagebreak(weak: true)
#include "sections/c2_b4.typ"

#pagebreak(weak: true)
= Parte III · Curso 3: Edge-AI Sub-2-Bit & Portable Kernels (`B3-Edge`)
#block(width: 100%, fill: bg-amber, stroke: (left: 4pt + amber, rest: 0.5pt + border-gray), inset: 10pt, radius: 4pt)[
  *Objetivo del Itinerario:* Replicar a presupuesto *0 EUR* el rendimiento y la eficiencia energética (`mWh/token`) de `Ternary Bonsai` sobre `Qwen3.5-2B` y `Gemma 4 E2B` (`< 1 GB`), construyendo cuantizadores CPU (`FHT + GPTQ`), kernels portables `CUDA sm_61` (`__dp4a`) y `Vulkan Compute (GLSL/SPIR-V)`, un servidor local `< 8 W` y el puente académico `LogFit-Edge` para HPAC/UMA.
]
#v(6pt)
#include "sections/c3_e0_e1.typ"
#pagebreak(weak: true)
#include "sections/c3_e2.typ"
#pagebreak(weak: true)
#include "sections/c3_e3.typ"
#pagebreak(weak: true)
#include "sections/c3_e4.typ"
#pagebreak(weak: true)
#include "sections/c3_e5_e6.typ"

#pagebreak(weak: true)
= Anexo · Mapa Consolidado de Unidades y Proyectos Integradores

A continuación se resume la matriz completa de los *3 cursos (65 unidades/sprints)* y su alineación con artefactos reales y repositorios de portafolio.

#styled-table(
  columns: (0.7fr, 1.0fr, 2.3fr, 1.5fr),
  headers: ([Curso], [Bloque/Mód.], [Unidades / Sprints], [Entregable Principal]),
  rows: (
    ([Tronco], [Cap. 0], [`TC.1`–`TC.6` · Fundamentos Compartidos], [Lo Común vs Específico + Enganche C1/C3 + OS 20m]),
    ([Curso 1], [B1 (`LOC`/`CLD`)], [`B1.1` Strides, `B1.2` CUDA GEMM, `B1.3` `nsys`, `B1.4` Triton/Roofline], [Benchmarks GTX 1050 + kernels SMEM 32x32 y 16x16]),
    ([Curso 1], [B2 (`LOC`/`CLD`)], [`B2.1` KV-Cache, `B2.2` vLLM vs SGLang, `B2.3` Dynamo, `B2.4` Agentes], [Calculadora KV + dry-run/serving vLLM & SGLang]),
    ([Curso 1], [B3 (`CLD`)], [`B3.1` FP8/AWQ/NVFP4, `B3.2` Calibración, `B3.3` SpecDecode, `B3.4` Pareto], [Proyecto A: `agentic-inference-opt-lab`]),
    ([Curso 1], [B4 (`CLD`)], [`B4.1` TP/PP/DP, `B4.2` MoE/WideEP, `B4.3` PDD/LMCache, `B4.4` NCCL], [Diseño 3D/WideEP + diagnóstico `busbw`]),
    ([Curso 1], [B5 (`LOC`/`CLD`)], [`B5.1` Harness, `B5.2` DCGM/Xid, `B5.3` Megatron MFU, `B5.4` K8s/MIG], [Proyecto B: `gpu-cluster-acceptance-profiler`]),
    ([Curso 2], [B1 (`MOB`/`LOC`/`LAB`)], [`B1.1` Databricks vs Seldon, `B1.2` MLflow 3, `B1.3` Unity Catalog], [ADR-001 + pipeline `@dev->@champion` + audit pack]),
    ([Curso 2], [B2 (`MOB`/`LOC`)], [`B2.1` Agentic Evals, `B2.2` RAG/OpenSearch, `B2.3` OTel & FinOps], [`mlflow.genai.evaluate()` gate + panel OTel/Prom]),
    ([Curso 2], [B3 (`MOB`/`LOC`)], [`B3.1` Seldon/Knative, `B3.2` Istio/Argo, `B3.3` Crossplane], [`AnalysisTemplate` canary + `XAgenticEnvironment`]),
    ([Curso 2], [B4 (`MOB`/`LOC`)], [`B4.1` SLOs/Error Budget, `B4.2` GameDay, `B4.3` Multi-Region], [`agentic-platform-golden-path` + ADR UE]),
    ([Curso 3], [E0–E1 (`0 EUR`)], [`E0.1–E0.4` `llama.cpp`/PrismML + `sysfs`, `E1.1–E1.4` Baseline], [Harness `mWh/token` + `02_bonsai_vs_qwen.md` (H0–H1)]),
    ([Curso 3], [E2–E3 (`0 EUR`)], [`E2.1–E2.6` FHT+GPTQ CPU, `E3.1–E3.5` Gemma 4 E2B + Evals], [`Gemma-4-E2B-Ternary < 1 GB` + Pareto (H2–H3)]),
    ([Curso 3], [E4–E6 (`0 EUR`)], [`E4.1–E4.7` CUDA `dp4a` + Vulkan GLSL, `E5` `<8W`+UMA, `E6` QAT], [Kernels portables + PR `ggml-vulkan` + Dossier UMA (H4–H6)]),
  ),
)
