#import "theme.typ": *

#show: master-setup.with(
  title: "AI Infrastructure & Agentic Platform MLOps",
  subtitle: "Edición Máster Unificada · Tronco Común + Curso 1 + Curso 2",
  edition: "Edición Máster Unificada",
)

#cover-page(
  "Manual Técnico de Nivel Máster · Dos Especializaciones Integradas",
  "AI Infrastructure & Agentic Platform MLOps",
  "Del hardware CUDA, KV-Cache y paralelismo multi-GPU a plataformas agénticas corporativas con Databricks, MLflow 3, Seldon, Istio y SRE",
  [
    Este volumen unifica dos itinerarios complementarios construidos sobre un *Tronco Común de Ingeniería de IA* y pensados para ejecutarse mediante *micro-sprints de 20 minutos* (`[MOB]`, `[LOC]`, `[PREP]`, `[CLD]` y `[LAB]`):
    - *Curso 1 (Infraestructura GPU y Motores de Inferencia):* Anatomía de memoria HBM/L2, kernels CUDA C++, profiling `nsys`/`ncu`, cuantización FP8, TensorRT-LLM, vLLM/SGLang y clústeres multi-GPU con NVLink/NCCL.
    - *Curso 2 (Enterprise MLOps & Agentic Platform Engineering):* Evolución desde Senior DevOps/SRE hacia Databricks Mosaic AI, MLflow 3, Unity Catalog, evaluación automática de agentes/RAG, observabilidad OTel/OpenSearch, Seldon/Knative/Istio/ArgoCD/Crossplane y SLOs agénticos.
  ],
  (
    (
      title: "0 · Tronco Común (6 Pilares)",
      bg: bg-navy,
      col: navy,
      body: [
        Autoregresión Prefill vs Decode, matemáticas de capacidad/FinOps, reproducibilidad y linaje, evaluación y puertas de calidad, observabilidad de cola (`TTFT`/`TPOT`/`p99`) y seguridad Zero-Trust + PII.
      ],
    ),
    (
      title: "1 · Curso 1: AI Infrastructure (B1–B5)",
      bg: bg-teal,
      col: teal,
      body: [
        20 unidades didácticas (`B1.1`–`B5.4`). Incluye mediciones reales en GTX 1050 (2 GB) y diseño reproducible para L4, L40S y H100 (`sm_89`/`sm_90`).
      ],
    ),
    (
      title: "2 · Curso 2: Agentic MLOps (B1–B4)",
      bg: bg-purple,
      col: purple,
      body: [
        12 unidades didácticas (`B1.1`–`B4.3`) cerrando al 100% la brecha entre Senior DevOps/SRE y plataformas agénticas reguladas con proyecto *Golden Path*.
      ],
    ),
    (
      title: "3 · Sistema Operativo y Prácticas",
      bg: bg-amber,
      col: amber,
      body: [
        Infografías vectoriales de arquitectura, código reproducible (`verify.sh` / `run.sh`), ejercicios de entrevista resueltos y bibliografía oficial verificable.
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
= Anexo · Mapa Consolidado de Unidades y Proyecto Integrador

A continuación se resume la matriz completa de las *32 unidades* cubiertas por ambos cursos y su alineación con artefactos reales del repositorio y portafolio técnico.

#styled-table(
  columns: (0.7fr, 1.1fr, 2.2fr, 1.5fr),
  headers: ([Curso], [Bloque], [Unidades], [Entregable Principal]),
  rows: (
    ([Tronco], [Cap. 0], [`0.1`–`0.6` · Fundamentos Compartidos], [Modelo mental común Prefill/Decode, FinOps, Linaje, SLOs]),
    ([Curso 1], [B1 (`LOC`/`CLD`)], [`B1.1` Strides, `B1.2` CUDA GEMM, `B1.3` `nsys`, `B1.4` Triton/Roofline], [Benchmarks GTX 1050 + kernels SMEM 32x32 y 16x16]),
    ([Curso 1], [B2 (`LOC`/`CLD`)], [`B2.1` KV-Cache, `B2.2` vLLM vs SGLang, `B2.3` Disagg, `B2.4` SpecDecode], [Calculadora KV + dry-run/serving vLLM & SGLang]),
    ([Curso 1], [B3 (`CLD`)], [`B3.1` AWQ/GPTQ/FP8, `B3.2` TensorRT-LLM, `B3.3` Calibración], [Plan cuantización FP8 + engines TRT-LLM]),
    ([Curso 1], [B4 (`CLD`)], [`B4.1` LoRA/QLoRA, `B4.2` Megatron/FSDP, `B4.3` MoE, `B4.4` DeepSeek/RL], [Diseño 3D Parallelism + análisis `All-to-All`]),
    ([Curso 1], [B5 (`LOC`/`CLD`)], [`B5.1` K8s/MIG, `B5.2` NCCL/NVLink, `B5.3` RoCEv2, `B5.4` Burn-in/DCGM], [Runbook `dcgmi` + calculadora BW y stragglers]),
    ([Curso 2], [B1 (`MOB`/`LOC`/`LAB`)], [`B1.1` Databricks vs Seldon, `B1.2` MLflow 3, `B1.3` Unity Catalog], [ADR-001 + pipeline `@dev->@champion` + audit pack]),
    ([Curso 2], [B2 (`MOB`/`LOC`)], [`B2.1` Agentic Evals, `B2.2` RAG/OpenSearch, `B2.3` OTel & FinOps], [`mlflow.evaluate()` gate + panel OTel/Prom]),
    ([Curso 2], [B3 (`MOB`/`LOC`)], [`B3.1` Seldon/Knative, `B3.2` Istio/Argo, `B3.3` Crossplane], [`AnalysisTemplate` canary + `XAgenticEnvironment`]),
    ([Curso 2], [B4 (`MOB`/`LOC`)], [`B4.1` SLOs/Error Budget, `B4.2` Post-mortem, `B4.3` Multi-Region], [Post-mortem blameless + ADR enrutamiento EU]),
  ),
)
