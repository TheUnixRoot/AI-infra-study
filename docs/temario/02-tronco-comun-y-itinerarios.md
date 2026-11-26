# Tronco Común y Dos Itinerarios Profesionales (AI Infra & Agentic MLOps)

Este documento extrae la intersección técnica entre los dos playbooks del repositorio y organiza el estudio en **un tronco común de 6 pilares** seguido de **dos itinerarios especializados**, editados como manuales didácticos en Typst/PDF dentro de la [Biblioteca PDF](../campus/biblioteca-pdf.md).

## 1. Mapa de Dos Puestos desde un Baseline DevOps/SRE

| Dimensión | Curso 1: AI Infrastructure & GPU Systems | Curso 2: Enterprise MLOps & Agentic Platform |
| :--- | :--- | :--- |
| **Fuente maestra** | [`00-master-playbook.md`](00-master-playbook.md) | [`01-agentic-mlops-playbook.md`](01-agentic-mlops-playbook.md) |
| **Roles objetivo** | GPU Platform Benchmarking · Senior MLE Inference (Token Factory) | Staff/Senior MLOps Engineer · Agentic Platform Engineer |
| **Hardware / Entorno** | GTX 1050 `sm_61` + Cloud GPU (`L4`, `L40S`, `H100`) | CPU local (Docker, Kind, MLflow, OpenSearch) + Databricks Sandbox |
| **Proyectos Portfolio** | `agentic-inference-opt-lab` (A) y `gpu-cluster-acceptance-profiler` (B) | `agentic-platform-golden-path` |
| **Foco diferencial** | Kernels CUDA/Triton, `nsys`/`ncu`, KV-Cache, FP8/AWQ, NCCL, Megatron MFU | Databricks Mosaic AI, MLflow 3, Unity Catalog, Agentic Evals, Seldon/Istio/Argo/Crossplane |

## 2. Los 6 Pilares del Tronco Común

Estudiar ambos perfiles por separado duplicaría ~40 % del temario. El tronco común unifica:

1. **Economía de Inferencia LLM (Prefill vs Decode):** Intensidad aritmética, TTFT vs TPOT/ITL, aritmética de KV-Cache y planificación de concurrencia/capacidad (TPM/RPM vs VRAM).
2. **Anatomía de Cargas Agénticas y RAG:** Prompts largos compartidos (*system* + *tools*), *Prefix Caching*, *structured outputs* y separación de fallos de *Retrieval* vs *Generation*.
3. **Metodología Científica y Gates en CI/CD:** Warmup, percentiles p50/p95/p99, evitar *coordinated omission*, fronteras de Pareto y *evaluation/regression gates* automáticos antes de promover a producción.
4. **Observabilidad End-to-End y FinOps:** Correlación de trazas (`nsys` / NVTX en capa GPU; OpenTelemetry / MLflow Tracing en capa aplicación), detección de degradación y cálculo de $\$ / 1\text{M tokens}$.
5. **Serving Cloud-Native en Kubernetes y Operación SRE:** Autoescalado por concurrencia, aislamiento, gestión de *cold starts*, observabilidad Prometheus/Grafana, SLOs y respuesta a incidentes.
6. **Sistema Operativo de Micro-Sprints (20 min):** Regla 1 chat por bloque, modalidades `[MOB]`/`[LOC]`/`[PREP]`/`[CLD|LAB]`, *Handover* YAML y *auto-kill* en cloud.

## 3. Edición Didáctica en Typst y PDF (`docs/pdf/`)

Los archivos fuente modulares en Typst y sus PDFs compilados cubren cada bloque con **infografías vectoriales**, **desarrollo teórico nivel máster**, **prácticas guiadas**, **ejercicios con solucionario** y **fuentes primarias**:

- **Volumen Maestro Unificado (Tronco Común + Curso 1 + Curso 2):** [`master_ai_infra_y_mlops.typ`](../pdf/master_ai_infra_y_mlops.typ) → `Master_AI_Infra_y_Agentic_MLOps.pdf`
- **Curso 1 (Actual — AI Infrastructure & GPU Systems):** [`curso1_ai_infra.typ`](../pdf/curso1_ai_infra.typ) → `Curso_1_AI_Infrastructure_GPU.pdf`
- **Curso 2 (Nuevo — Agentic Platform MLOps):** [`curso2_agentic_mlops.typ`](../pdf/curso2_agentic_mlops.typ) → `Curso_2_Agentic_Platform_MLOps.pdf`
