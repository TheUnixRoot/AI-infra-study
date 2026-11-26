#import "../theme.typ": *

= Curso 2 · Bloque 4: Enterprise SRE, Multi-Cloud y Portfolio (`CHAT-B4`)

#block-banner(
  "C2-B4",
  "SLOs Agénticos, Capacity Planning, Multi-Region UE y Cierre Portfolio",
  "Core SRE & Platform Operations: fiabilidad en producción, soberanía de datos y defensa técnica",
  "~1.5 semanas · 7 sprints de 20m",
)[
  Traduce la experiencia Senior DevOps/SRE a la operación crítica de plataformas agénticas: SLIs/SLOs específicos de IA, planificación de capacidad (RPM/TPM), *runbooks* de incidentes agénticos, topologías multi-región bajo soberanía europea (RGPD) y consolidación final del repositorio `agentic-platform-golden-path`.
]

#infografia(
  "Topología Multi-Región con Soberanía de Datos UE y Operación SRE por SLOs",
  "Enrutamiento geo-consciente respetando residencia de datos en la UE (Frankfurt/Dublín), caché semántico local, cuotas TPM/RPM y flujo de failover.",
  height: 185pt,
)[
  // Top: Gateway / DNS
  #diag-node(140pt, 6pt, 190pt, 28pt, "Global API Gateway / GeoDNS (UE)", sub: "Latency-based routing + Health Checks + WAF", fill: bg-navy, col: navy)
  #arrow-d(130pt, 34pt, 54pt, col: teal)
  #arrow-d(340pt, 34pt, 54pt, col: purple)

  // Región Primaria UE
  #place(dx: 8pt, dy: 54pt, rect(width: 222pt, height: 86pt, fill: bg-teal, stroke: 0.9pt + teal, radius: 5pt))
  #place(dx: 16pt, dy: 58pt, text(fill: teal, weight: "bold", size: 8pt)[Región Activa 1: EU-Central (Frankfurt)])
  #diag-node(16pt, 72pt, 98pt, 30pt, "K8s / Seldon + Istio", sub: "Semantic Cache local", fill: white, col: teal)
  #diag-node(122pt, 72pt, 100pt, 30pt, "Vector DB Colocated", sub: "0 ms cruce de región", fill: white, col: teal)
  #diag-node(16pt, 106pt, 206pt, 26pt, "Mosaic AI / LLM Endpoint (PrivateLink)", sub: "Datos & Trazas retenidos dentro de UE (RGPD)", fill: white, col: navy)

  // Región Secundaria UE
  #place(dx: 240pt, dy: 54pt, rect(width: 228pt, height: 86pt, fill: bg-purple, stroke: 0.9pt + purple, radius: 5pt))
  #place(dx: 248pt, dy: 58pt, text(fill: purple, weight: "bold", size: 8pt)[Región Activa/Failover 2: EU-West (Dublín)])
  #diag-node(248pt, 72pt, 100pt, 30pt, "K8s / Seldon + Istio", sub: "Réplica caliente / Canary", fill: white, col: purple)
  #diag-node(356pt, 72pt, 104pt, 30pt, "Vector DB Replicada", sub: "Delta async < 5 min lag", fill: white, col: purple)
  #diag-node(248pt, 106pt, 212pt, 26pt, "LLM Endpoint Secundario (Failover TPM)", sub: "Degradación controlada ante 429 / caída regional", fill: white, col: navy)

  // Barra inferior SLOs
  #place(dx: 8pt, dy: 148pt, box(width: 460pt, fill: bg-amber, stroke: 0.6pt + amber, inset: 6pt, radius: 4pt)[
    #set text(size: 7.6pt, fill: ink)
    *Contrato SRE de Plataforma Agéntica (SLOs):* Disponibilidad >= `99.9%` · TTFT p95 <= `450 ms` · E2E p99 <= `4.0 s` · Tool Execution Success >= `98%` · Quality Score (online sample) >= `90%`.
  ])
]

== Unidad B4.1 · Operación por SLOs, Capacity Planning e Incident Response

#unit-header(
  "B4.1",
  "SLOs Agénticos, Capacity Planning (RPM/TPM) y Post-Mortems",
  "[MOB] 2s + [LOC] 1s",
  "B2.3, B3.2",
  "B4.2, B4.3",
  "Móvil + Docs/Markdown",
  "2 Runbooks operativos y 1 Post-Mortem Blameless aplicados a un fallo agéntico realista.",
)

#concepto-clave[
  1. *SLIs y SLOs Específicos de Agentes:* Un servicio puede devolver `HTTP 200 OK` y seguir violando su SLO funcional. Se definen cuatro pilares de Error Budget: (a) *Availability SLI* (sin 5xx), (b) *Latency SLI* (`TTFT p95` y `E2E p99` según número medio de pasos), (c) *Tool Execution Success Rate*, y (d) *Online Quality SLI* (muestreo asíncrono de fidelidad).
  2. *Capacity Planning (RPM / TPM vs Réplicas Provisionadas):* En endpoints gestionados, el cuello suele ser la cuota de *Tokens Per Minute (TPM)* y *Requests Per Minute (RPM)* antes que la CPU. Para $lambda$ peticiones/s con longitud media $(S_"in" + S_"out")$ y $k_"pasos"$ turnos por conversación:
     $ "TPM requerido" = lambda times 60 times k_"pasos" times (S_"in" + S_"out") $
     Esto determina cuándo compensa contratar *Provisioned Throughput* o réplicas GPU en K8s frente a *pay-per-token* con *backoff* exponencial y *fallback* entre proveedores.
  3. *Runbooks de Incidentes Típicos de Agentes:* (1) Caída o cambio de contrato de una API externa de herramienta, (2) Saturación o lag severo del índice vectorial, (3) Bucle infinito de reintentos/tokens tras nuevo prompt, (4) Deriva silenciosa tras actualización menor de un modelo fundacional externo (fijar siempre versiones/snapshots exactos).
]

#practica-box(title: "Práctica C2-B4.1 ([LOC]) · Runbooks y Post-Mortem Blameless")[
  - Redactar `docs/runbooks/01-tool-schema-drift.md`, `docs/runbooks/02-vector-lag-spike.md` y `docs/postmortems/2026-incident-tool-loop.md` (línea temporal, impacto en *error budget*, causa raíz en los 5 porqués y acciones preventivas automatizadas en el gate de B2.1/B3.2).
]

== Unidad B4.2 · Operaciones Multi-Cloud y Gestión de Latencia Cross-Region

#unit-header(
  "B4.2",
  "Arquitecturas Multi-Región, Latencia y Soberanía de Datos UE",
  "[MOB] 2s",
  "B4.1",
  "B4.3",
  "Móvil (Diseño en pizarra)",
  "Diseñar topología multi-región con failover automático y residencia de datos europea.",
)

#concepto-clave[
  1. *Colocación Geográfica y Presupuesto de Latencia:* En un agente RAG de 3 pasos con 2 llamadas vectoriales y 2 llamadas LLM, cruzar el Atlántico (UE $arrow.r$ US-East, ~90 ms RTT + TLS) cuatro veces añade $> 400"–"600 "ms"$ de pura latencia física antes de generar el primer token. Regla de oro: *colocar orquestador, base vectorial y endpoint LLM en la misma región*.
  2. *Soberanía UE (RGPD / Sector Regulado):* Garantizar mediante políticas de enrutamiento y endpoints regionales (p. ej. `eu-central-1` Frankfurt + `eu-west-1` Dublín / Azure Sweden Central + West Europe) que ningún prompt con PII ni trazas OTel cruce fronteras fuera del Espacio Económico Europeo incluso durante un *failover*.
  3. *Activo-Activo vs Activo-Pasivo y Caché Semántico:* En *activo-activo*, el *caching* semántico local por región reduce llamadas repetidas al LLM y el coste de replicación cross-region se limita a deltas asíncronos del índice vectorial.
]

== Unidad B4.3 · Ensamblaje Final y Preparación de Entrevista (Inglés)

#unit-header(
  "B4.3",
  "Ensamblaje `agentic-platform-golden-path` y Defensa STAR (EN)",
  "[PREP] 1s + [LOC] 2s",
  "B1.3, B2.3, B3.3, B4.2",
  "Portfolio Completo Entrevista",
  "Local Git + Docker/Kind",
  "Repositorio público impecable y guion técnico en inglés listo para rondas técnicas.",
)

#practica-box(title: "Checklist de Cierre agentic-platform-golden-path")[
  1. `README.md` ejecutivo en inglés con diagrama de arquitectura (Mermaid), tabla de SLOs y guía `verify.sh`.
  2. `docs/adr/001-serving-topology-databricks-vs-seldon.md` + `docs/runbooks/` + `docs/postmortems/`.
  3. `pipelines/` (MLflow 3 promotion gate + `audit-evidence.json`) y `evals/` (`mlflow.genai.evaluate` + RAG health).
  4. `infra/` (Knative/Seldon, Istio `VirtualService`, Argo Rollouts `AnalysisTemplate`, Crossplane XRD `XAgenticEnvironment`).
  5. *4 historias STAR en inglés* conectando tu trayectoria DevOps/SRE con: (a) Governance-by-Design, (b) Agentic Evals & CI Gates, (c) Istio/Argo Canary Auto-Rollback, (d) Multi-Region Failover & FinOps Incident.
]

#ejercicio-box[
  *Ejercicio C2-B4-A (Capacity Planning TPM):* Tu plataforma agéntica atiende un pico de `12 peticiones/s` de usuarios. Cada petición ejecuta una media de `2.5` llamadas al LLM con `2,200` tokens de entrada y `300` de salida por llamada. Calcula el TPM requerido y decide si basta una cuota estándar de `1,000,000 TPM` o necesitas repartir carga entre dos regiones/endpoints.
]

#solucion-box[
  - *Sol. C2-B4-A:* Tokens por conversación = $2.5 times (2200 + 300) = 6250 "tokens/req"$. Con `12 req/s` (= `720 req/min`): $"TPM requerido" = 720 times 6250 = 4500000 "TPM"$ (`4.5M TPM`). Una cuota de `1M TPM` sufriría rechazo masivo (`HTTP 429` en $>77\%$ del tráfico pico). Solución: combinar *Prompt/Prefix Caching* (reduce tokens facturados/procesados en entrada repetida), aprovisionar *Provisioned Throughput* o réplicas K8s dedicadas para la carga base y configurar enrutamiento multi-región dentro de la UE para desbordamiento (*overflow*).
]

#fuentes-box[
  1. Google SRE Book — *Service Level Objectives & Postmortem Culture*: `https://sre.google/sre-book/table-of-contents/`
  2. Databricks Docs — *Provisioned Throughput, Rate Limits & EU Compliance*: `https://docs.databricks.com/en/machine-learning/foundation-models/`
  3. OpenTelemetry & Grafana — *LLM / GenAI Semantic Conventions*: `https://opentelemetry.io/docs/specs/semconv/gen-ai/`
]
