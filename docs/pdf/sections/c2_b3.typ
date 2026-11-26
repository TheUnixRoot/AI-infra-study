#import "../theme.typ": *

= Curso 2 · Bloque 3: Serving Cloud-Native, GitOps y Golden Paths (`CHAT-B3`)

#block-banner(
  "C2-B3",
  "Seldon Core v2, Knative, Istio + Argo Rollouts y Crossplane Golden Paths",
  "Core Platform Engineering: K8s AI serving, progressive delivery con auto-rollback y autoservicio",
  "~2 semanas · 10 sprints de 20m",
)[
  Diseña e implementa la arquitectura completa de plataforma agéntica sobre Kubernetes: *serving* elástico (*Seldon Core v2 / Knative*), entrega progresiva con malla de servicios (*Istio + ArgoCD / Argo Rollouts*) y abstracciones de autoservicio (*Crossplane XRDs*) con seguridad *Zero-Trust*.
]

#infografia(
  "Arquitectura Cloud-Native: Crossplane Golden Path + GitOps Canary (Istio + Argo Rollouts)",
  "Izquierda: plataforma de autoservicio vía Crossplane. Derecha: entrega progresiva en producción gobernada por métricas de MLflow/Prometheus.",
  height: 192pt,
)[
  // Bloque izquierdo: Crossplane Golden Path
  #place(dx: 8pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[Golden Path Autoservicio (B3.3)])
  #diag-node(8pt, 20pt, 140pt, 32pt, "Claim YAML del Equipo", sub: "AgenticWorkspace (dev/prod)", fill: bg-light, col: slate)
  #arrow-d(78pt, 52pt, 66pt, col: slate)
  #diag-node(8pt, 66pt, 140pt, 42pt, "Crossplane XRD + Composition", sub: "Control Plane K8s\nAprovisiona infra gobernada", fill: bg-purple, col: purple)
  #arrow-d(78pt, 108pt, 122pt, col: purple)
  #diag-node(8pt, 122pt, 140pt, 60pt, "Recursos Cloud Seguros", sub: "• Bucket S3/ADLS + Vector DB\n• PrivateLink a Databricks/LLM\n• IRSA / Workload Identity\n  (0 secretos estáticos)", fill: bg-teal, col: teal)

  // Separador vertical
  #place(dx: 158pt, dy: 6pt, line(length: 178pt, angle: 90deg, stroke: 0.5pt + border-gray))

  // Bloque derecho: GitOps Serving & Canary
  #place(dx: 168pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[Serving & Progressive Delivery GitOps (B3.1 - B3.2)])
  #diag-node(168pt, 20pt, 106pt, 40pt, "ArgoCD (GitOps)", sub: "Sync declarativo\nApproval prod", fill: bg-navy, col: navy)
  #arrow-r(274pt, 40pt, 290pt, col: navy)
  #diag-node(290pt, 20pt, 178pt, 40pt, "Istio Gateway + VirtualService", sub: "mTLS estricto · Timeouts tool calls\n95% @champion / 5% @challenger + Mirror", fill: bg-navy, col: navy)

  #arrow-d(335pt, 60pt, 78pt, col: teal)
  #arrow-d(425pt, 60pt, 78pt, col: amber)

  #diag-node(168pt, 78pt, 185pt, 52pt, "Serving Stable (@champion)", sub: "Knative (KPA concurrencia / scale-to-zero dev)\no Seldon Core v2 (Pipeline CRDs: Guard -> LLM)", fill: bg-teal, col: teal)
  #diag-node(362pt, 78pt, 106pt, 52pt, "Canary (@challenger)", sub: "Nueva versión vN+1\n5% tráfico real", fill: bg-amber, col: amber)

  #diag-node(168pt, 140pt, 300pt, 42pt, "Argo Rollouts AnalysisTemplate (Prometheus / MLflow)", sub: "Evalúa cada 1 min: Latencia p99 < 1.5s AND Error Rate < 1% AND Tool Reliability >= 95%\nSi falla umbral -> Auto-Rollback inmediato a @champion (100% tráfico)", fill: bg-red, col: crimson)
]

== Unidad B3.1 · Serving de Modelos y Agentes en Kubernetes: Seldon Core v2 y Knative

#unit-header(
  "B3.1",
  "Serving Cloud-Native en K8s: Seldon Core v2 y Knative",
  "[MOB] 2s + [LOC] 2s",
  "B1.1",
  "B3.2, B4.1",
  "Clúster local Kind (Kubernetes in Docker)",
  "Manifiestos funcionales en Kind con autoscaling por concurrencia y pruebas de carga.",
)

#concepto-clave[
  1. *Seldon Core v2 (Data-Centric Pipelines):* Diseñado para flujos multi-etapa con alto throughput mediante un plano de datos desacoplado (Kafka / gRPC / REST) y dos CRDs clave: `Model` (despliegue sobre runtimes Triton/MLServer/custom) y `Pipeline` (grafo declarativo que encadena pasos: *PII Guardrail $arrow.r$ Embedding/RAG $arrow.r$ Modelo/Agente $arrow.r$ Verificador de salida*). Soporta *multi-model serving* compartiendo servidores cuando los modelos son pequeños.
  2. *Knative Serving (Concurrencia y Scale-to-Zero):* Ideal para servicios HTTP/SSE de agentes en Python (LangGraph / FastAPI) y endpoints de inferencia donde escalar por CPU (`HPA`) falla porque un pod esperando respuesta del LLM o de una tool tiene CPU al 5% mientras satura conexiones. El *Knative Pod Autoscaler (KPA)* escala por *concurrencia en vuelo* (`containerConcurrency` / `target-utilization`) y gestiona *Revisions* inmutables con *scale-to-zero* en `dev`/`test` para ahorrar GPU/CPU fuera de horario.
]

#practica-box(title: "Práctica C2-B3.1 ([LOC] Kind) · Servicio Knative con Autoscaling por Concurrencia")[
  ```yaml
  apiVersion: serving.knative.dev/v1
  kind: Service
  metadata: { name: agentic-orchestrator, namespace: ai-staging }
  spec:
    template:
      metadata:
        annotations:
          autoscaling.knative.dev/metric: "concurrency"
          autoscaling.knative.dev/target: "10"
          autoscaling.knative.dev/min-scale: "0"   # scale-to-zero en staging
          autoscaling.knative.dev/max-scale: "8"
      spec:
        containers:
          - image: ghcr.io/org/agentic-orchestrator:sha-1a2b3c
            ports: [{ containerPort: 8080 }]
  ```
]

== Unidad B3.2 · Despliegues Progresivos y Service Mesh: Istio + ArgoCD

#unit-header(
  "B3.2",
  "Istio Service Mesh, ArgoCD y Rollouts con Auto-Rollback",
  "[MOB] 2s + [LOC] 2s",
  "B2.1, B2.3, B3.1",
  "B4.1",
  "Clúster local Kind (Istio + Argo Rollouts + Prometheus)",
  "Canary 90/10 reproducible que ejecuta rollback automático ante degradación simulada.",
)

#concepto-clave[
  1. *Istio en Plataformas Agénticas:*
     - *Weighted Splitting & Shadow Mirroring:* Antes de dar tráfico real al `@challenger`, Istio `VirtualService.mirror` duplica tráfico de producción de forma asíncrona (*fire-and-forget*, descartando la respuesta) para validar latencia y parseo de herramientas sin afectar al usuario final; después pasa a canary 95/5.
     - *Seguridad y Resiliencia:* mTLS estricto entre orquestador, vector DB y tool servers; *timeouts* acotados y *circuit breakers* (`outlierDetection`) en `DestinationRule` para que una tool caída no bloquee todos los workers. *(Cuidado: evitar retries automáticos ciegos de Istio en tools no idempotentes como `create_ticket`!)*.
  2. *ArgoCD + Argo Rollouts:* Flujo GitOps `dev -> test -> prod` donde `AnalysisTemplate` consulta Prometheus cada 60 s (`p99_latency`, `error_rate`, `tool_reliability`); ante cualquier violación, Argo Rollouts revierte el peso del `VirtualService` al 100% hacia `@champion` en segundos.
]

#practica-box(title: "Práctica C2-B3.2 ([LOC] Kind) · AnalysisTemplate de Rollback Automático")[
  ```yaml
  apiVersion: argoproj.io/v1alpha1
  kind: AnalysisTemplate
  metadata: { name: agent-slo-gate }
  spec:
    metrics:
      - name: tool-reliability
        interval: 30s
        successCondition: result[0] >= 0.95
        failureLimit: 2
        provider:
          prometheus:
            address: http://prometheus.monitoring:9090
            query: avg(agent_tool_reliability_ratio{version="canary"})
  ```
]

== Unidad B3.3 · Golden Paths con Crossplane + Cloud Networking

#unit-header(
  "B3.3",
  "Crossplane XRDs, PrivateLink y Zero-Trust Identity",
  "[MOB] 2s + [LOC] 1s",
  "B3.2",
  "B4.3",
  "Clúster local Kind + Crossplane",
  "XRD + Composition funcional (XAgenticEnvironment) abstraída para equipos de producto.",
)

#concepto-clave[
  1. *Platform Engineering con Crossplane:* En lugar de que cada equipo de producto copie 600 líneas de Terraform para crear un bucket S3/ADLS, una base vectorial, secretos KMS e IAM roles, el equipo de plataforma define una *CompositeResourceDefinition (XRD)* minimalista (`XAgenticEnvironment`) y una *Composition* con políticas de seguridad embebidas (cifrado obligatorio, retención de auditoría, etiquetas FinOps).
  2. *Cloud Networking & Zero-Trust Identity:* Tráfico privado obligatorio hacia Databricks Mosaic AI y bases vectoriales mediante *AWS PrivateLink / Azure Private Endpoints* (cero salida por Internet público), e identidades federadas efímeras (*IRSA* en EKS / *Workload Identity* en AKS) eliminando secretos estáticos de larga duración en los pods.
]

#ejercicio-box[
  *Ejercicio C2-B3-A (Resiliencia Service Mesh):* Un equipo activa `retries: attempts: 3` en el `VirtualService` de Istio para todas las rutas de herramientas del agente, incluida `POST /tools/refund_payment` con timeout de `2s`. Explica dos fallos graves que provoca esta configuración en producción y cómo debes configurarla correctamente.
]

#solucion-box[
  - *Sol. C2-B3-A:* (1) *Efecto duplicado no idempotente:* si el backend de pagos tarda 2.1 s en procesar un reembolso, Istio corta a los 2 s y reintenta hasta 3 veces, ejecutando múltiples reembolsos reales para la misma petición. (2) *Amplificación de tormenta de reintentos (Retry Storm):* bajo sobrecarga, multiplicar por 4 las llamadas colapsa definitivamente el servicio aguas abajo. *Corrección:* desactivar retries automáticos de mesh en métodos mutantes (`POST`), exigir `Idempotency-Key` en las herramientas mutantes, subir el timeout por ruta al presupuesto real + margen, y aplicar `outlierDetection` (circuit breaker) en `DestinationRule`.
]

#fuentes-box[
  1. Seldon Core v2 Architecture & Concepts: `https://docs.seldon.io/projects/seldon-core/en/v2/`
  2. Knative Serving Docs — *Autoscaling & Traffic Management*: `https://knative.dev/docs/serving/`
  3. Istio Docs — *Traffic Management, Mirroring & Security*: `https://istio.io/latest/docs/tasks/traffic-management/`
  4. Argo Rollouts Docs — *Canary & Prometheus Analysis*: `https://argo-rollouts.readthedocs.io/`
  5. Crossplane Docs — *Compositions & Platform Engineering*: `https://docs.crossplane.io/`
]
