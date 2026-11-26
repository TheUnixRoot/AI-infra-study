#import "../theme.typ": *

= Tronco Común y Arquitectura de los Dos Itinerarios

Partiendo de un perfil *Senior DevOps / SRE* (Kubernetes, CI/CD, IaC en AWS/Azure, gestión de secretos/identidades, SLOs, respuesta a incidentes y Prometheus/Grafana), ambos itinerarios evitan repetir fundamentos básicos de infraestructura generalista y concentran el estudio en cerrar brechas específicas de *sistemas de IA en producción*.

Al contrastar el currículo de *AI Infrastructure & GPU Systems* (Curso 1) con el de *Enterprise MLOps & Agentic Platform Engineering* (Curso 2), emerge un *tronco común técnico de 6 pilares* que representa aproximadamente el 40 % de las competencias transversales y evita duplicar esfuerzo entre ambos perfiles.

#infografia(
  "Mapa Unificado: Baseline DevOps/SRE -> Tronco Común -> Dos Itinerarios",
  "Arquitectura curricular conjunta: los 6 pilares comunes alimentan tanto la especialización de bajo nivel GPU/Inferencia (Curso 1) como la plataforma agéntica empresarial (Curso 2).",
  height: 225pt,
)[
  // Baseline superior
  #diag-node(140pt, 6pt, 190pt, 30pt, "BASELINE: Senior DevOps / SRE", sub: "K8s · CI/CD · IaC (AWS/Azure) · Prometheus · SLOs", fill: bg-light, col: slate)
  #arrow-d(235pt, 36pt, 54pt, col: slate)

  // Tronco común central
  #place(dx: 12pt, dy: 54pt, rect(width: 446pt, height: 68pt, fill: rgb("#F8FAFC"), stroke: 0.9pt + teal, radius: 5pt))
  #place(dx: 20pt, dy: 58pt, text(fill: teal, weight: "bold", size: 8pt)[TRONCO COMÚN (6 PILARES TRANSVERSALES)])
  #diag-node(20pt, 73pt, 138pt, 20pt, "1. Economía Prefill/Decode & KV", fill: bg-teal, col: teal)
  #diag-node(166pt, 73pt, 138pt, 20pt, "2. Cargas Agénticas & RAG", fill: bg-teal, col: teal)
  #diag-node(312pt, 73pt, 138pt, 20pt, "3. Eval Científica & CI Gates", fill: bg-teal, col: teal)
  #diag-node(20pt, 97pt, 138pt, 20pt, "4. Trazabilidad & FinOps $/1M", fill: bg-teal, col: teal)
  #diag-node(166pt, 97pt, 138pt, 20pt, "5. K8s Serving, SLOs & Cold Start", fill: bg-teal, col: teal)
  #diag-node(312pt, 97pt, 138pt, 20pt, "6. Sistema 20m Micro-Sprints", fill: bg-teal, col: teal)

  // Bifurcación hacia Curso 1 y Curso 2
  #arrow-d(120pt, 122pt, 146pt, col: navy)
  #arrow-d(350pt, 122pt, 146pt, col: purple)

  // Curso 1
  #diag-node(12pt, 146pt, 216pt, 72pt,
    "CURSO 1: AI Infra & GPU Systems",
    sub: "B1 GPU/CUDA/nsys -> B2 KV/vLLM/SGLang -> B3 FP8/SpecDecode -> B4 MoE/WideEP/PDD/NCCL -> B5 DCGM/MFU/K8s\nProyectos: agentic-inference-opt-lab + gpu-cluster-acceptance",
    fill: bg-navy, col: navy)

  // Curso 2
  #diag-node(242pt, 146pt, 216pt, 72pt,
    "CURSO 2: Agentic Platform MLOps",
    sub: "B1 Databricks/MLflow/Unity Catalog -> B2 Agent Evals/RAG/OTel -> B3 Seldon/Knative/Istio/Argo/Crossplane -> B4 SRE/Multi-Region\nProyecto: agentic-platform-golden-path",
    fill: bg-purple, col: purple)
]

== Matriz Comparativa de Ambos Puestos

#styled-table(
  columns: (1.1fr, 1.5fr, 1.5fr),
  headers: ("Eje de Comparación", "Itinerario 1: AI Infra & GPU Systems", "Itinerario 2: Agentic Platform MLOps"),
  rows: (
    ([*Roles Objetivo*], [GPU Platform Benchmarking · Senior MLE Inference (Token Factory)], [MLOps Engineer (Agentic Platform) · AI Platform Engineer]),
    ([*Capa Principal*], [Hardware GPU, CUDA/Triton, motores vLLM/SGLang/TRT-LLM, NCCL, clúster], [Plataforma Databricks/MLflow, gobernanza Unity Catalog, evals, K8s serving, SRE]),
    ([*Hardware / Entorno*], [GTX 1050 `sm_61` + Ráfagas Cloud (`L4`, `L40S`, `2x–8x H100`)], [CPU local (Docker, Kind, MLflow, OpenSearch) + Databricks Free/Trial]),
    ([*Métrica Reina*], [TTFT, TPOT, tok/s/GPU, `busbw`, MFU, frontera de Pareto `$/1M`], [Task Success, Tool Reliability, Recall\@K, Availability SLO, Audit Readiness]),
    ([*Entregables*], [`agentic-inference-opt-lab` + `gpu-cluster-acceptance-profiler`], [`agentic-platform-golden-path` (ADRs + MLflow + UC + Seldon/Istio/Crossplane)]),
  ),
)

== Los 6 Pilares del Tronco Común

#styled-table(
  columns: (0.9fr, 1.5fr, 1.5fr),
  headers: ("Pilar Común", "Materialización en Curso 1 (GPU/Inferencia)", "Materialización en Curso 2 (Agentic MLOps)"),
  rows: (
    ([*1. Física Prefill vs Decode & KV-Cache*],
     [Roofline $I = "FLOPs"/"Bytes"$, fórmula $2 dot L dot n_"kv" dot d_"head" dot S dot B dot b$, cuantización FP8 del KV y MLA.],
     [Capacity planning (TPM/RPM), dimensionamiento de endpoints serverless vs K8s, presupuesto de contexto largo y latencia cross-region.]),
    ([*2. Cargas Agénticas y RAG*],
     [Prefix Caching (vLLM APC / SGLang RadixAttention) sobre prompts de herramientas repetidos y overhead CPU de XGrammar/FSM.],
     [Evaluación de trayectorias agénticas (*tool selection*, JSON válido, no bucles) y salud de índices vectoriales (`Recall@K`, *index lag*).]),
    ([*3. Metodología Científica y Gates*],
     [Warmup obligatorio, percentiles p50/p95/p99 sin *coordinated omission*, `lm-eval` / EvalPlus y detección de regresiones en CI.],
     [`mlflow.genai.evaluate()` como *Evaluation Gate* bloqueante en CI/CD (`Tool >= 95%`, `Task >= 90%`) y Argo Rollouts Canary auto-rollback.]),
    ([*4. Trazabilidad y FinOps*],
     [Profiling GPU/CPU con `nsys` (NVTX, `cuda_gpu_kern_sum`), `ncu`, Linux `perf` y coste `$/1M tokens`.],
     [MLflow Tracing + OpenTelemetry (*spans* `user->RAG->LLM->tool`), OpenSearch + Grafana, cuotas por sesión y *circuit breakers* FinOps.]),
    ([*5. K8s Serving y Operación SRE*],
     [NVIDIA GPU Operator, MIG vs Time-Slicing, LeaderWorkerSet, DCGM Exporter y aceleración de *cold starts* (`safetensors` + NVMe).],
     [Seldon Core v2 (`Model`/`Pipeline`), Knative (*scale-to-zero*), Istio (mTLS, *mirroring*), Crossplane XRDs y *Post-Mortems blameless*.]),
    ([*6. Sistema Operativo 20m*],
     [Chats 1-a-1 `CHAT-B1..B5`, sprints `[MOB]/[LOC]/[PREP]/[CLD]`, *Handover* YAML y `sudo shutdown -h +90` en VMs GPU.],
     [Chats 1-a-1 `CHAT-B1..B4`, sprints `[MOB]/[LOC]/[PREP]/[LAB]`, *Handover* YAML y `verify.sh < 60s` por cada entregable.]),
  ),
)

== Sistema Operativo de Estudio (Higiene, Handover y Fork)

Para evitar que sesiones largas degraden el contexto técnico o mezclen objetivos de distintos bloques, ambos cursos comparten el mismo protocolo operativo en micro-sprints de 20 minutos:

1. *Regla 1-a-1:* un único chat activo por bloque (`CHAT-Bx`). Al alargarse un hilo, se cierra con un bloque *Handover* y se reinicia mediante *Fork*.
2. *Separación Tutor vs Agente:* el cuaderno LLM actúa como tutor socrático, revisor de trazas/ADRs y diseñador de rúbricas; el agente de código construye los scripts reproducibles (`run.sh` / `verify.sh`), Dockerfiles y manifests.
3. *Regla Pre-Flight `[PREP]`:* prohibido alquilar GPU cloud (`[CLD]`) o consumir cuota de laboratorio sin haber validado antes el script en seco (`--smoke-test` / `--dry-run`) en entorno local.
4. *Auto-Kill obligatorio en Cloud:* toda VM GPU alquilada en Nebius o Scaleway ejecuta `sudo shutdown -h +90` como primer comando.

#practica-box(title: "Plantillas Estándar de Control (Handover y Delegación)")[
```yaml
HANDOVER:
  itinerario: "Curso 1 (AI Infra) | Curso 2 (Agentic MLOps)"
  unidad_actual: "Bx.y"
  estado: "en_progreso" # pendiente | en_progreso | done
  aprendido_o_verificado:
    - "Invariante o métrica verificada con evidencia real"
  artefacto_generado: "ruta/al/script_o_adr"
  bloqueos_o_dudas: "Detalle concreto pendiente"
  siguiente_micro_sprint_20m: "Bx.y [MODO] - Siguiente paso exacto"
```
]
