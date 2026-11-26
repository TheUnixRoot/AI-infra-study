#import "../theme.typ": *

= Curso 2 · Bloque 1: Databricks, MLflow y Gobernanza por Diseño (`CHAT-B1`)

#block-banner(
  "C2-B1",
  "Databricks Mosaic AI, MLflow 3 Lifecycle y Unity Catalog Governance",
  "Core Enterprise MLOps: plataformas gestionadas vs K8s, promoción inmutable y auditoría automática",
  "~2 semanas · 12 sprints de 20m",
)[
  Domina la arquitectura empresarial de *Databricks Data Intelligence Platform* frente a *Kubernetes/Seldon*, implementa un ciclo de vida declarativo `dev -> test -> prod` con *MLflow 3* (`LoggedModel`, aliases, provenance) y automatiza la generación de evidencias regulatorias mediante *Unity Catalog*.
]

#infografia(
  "Arquitectura Corporativa: Promoción MLflow 3, Unity Catalog Lineage y Decisión de Serving",
  "Flujo Governance-by-Design desde el código y fuentes vectoriales en Unity Catalog hasta la promoción inmutable y el paquete Audit-Ready Evidence.",
  height: 192pt,
)[
  // Fila superior: Pipeline promoción
  #place(dx: 8pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[Golden Path CI/CD con Promoción Inmutable por Aliases (B1.2 - B1.3)])
  #diag-node(8pt, 20pt, 96pt, 48pt, "Git + Dataset UC", sub: "Commit SHA · Hash datos\ncatalog.schema.table\nEnd-to-End Lineage", fill: bg-light, col: slate)
  #arrow-r(104pt, 44pt, 118pt, col: slate)
  #diag-node(118pt, 20pt, 108pt, 48pt, "MLflow LoggedModel", sub: "Firma (Signature) + Deps\nAlias @dev -> @staging\nProvenance inmutable", fill: bg-navy, col: navy)
  #arrow-r(226pt, 44pt, 240pt, col: navy)
  #diag-node(240pt, 20pt, 108pt, 48pt, "Gates Técnicos + CAB", sub: "mlflow.evaluate() >= SLO\nRBAC/ABAC + Seguridad\nAprobación trazable", fill: bg-amber, col: amber)
  #arrow-r(348pt, 44pt, 362pt, col: amber)
  #diag-node(362pt, 20pt, 106pt, 48pt, "@champion en Prod", sub: "audit-evidence.json\nsha256 + system.access\nRollback inmediato", fill: bg-teal, col: teal)

  // Separador
  #place(dx: 8pt, dy: 78pt, line(length: 460pt, stroke: 0.5pt + border-gray))

  // Parte inferior: Comparativa Mosaic AI vs Seldon
  #place(dx: 8pt, dy: 84pt, text(fill: navy, weight: "bold", size: 8pt)[Decisión Arquitectónica (ADR-001): Databricks Mosaic AI vs Kubernetes / Seldon (B1.1)])
  #diag-node(8pt, 100pt, 224pt, 84pt,
    "Databricks Mosaic AI Model Serving",
    sub: "• Control Plane gestionado + Compute Plane en VPC cliente\n• Serverless Compute: 0 nodos K8s, autoscale rápido\n• Integración nativa Unity Catalog (linaje, audit, PII)\n• Ideal para: equipos rápidos, datos ya en Lakehouse, RAG/agentes estándar con gobernanza inmediata",
    fill: bg-purple, col: purple)
  #diag-node(244pt, 100pt, 224pt, 84pt,
    "Kubernetes (Seldon Core v2 / Knative + Istio)",
    sub: "• Control total de red (mTLS, egress), GPUs custom y kernel flags\n• Portabilidad multi-cloud estricta (AWS EKS + Azure AKS)\n• Shadow testing / Canary fino (Istio) + multi-model pipelines\n• Ideal para: serving de alta escala con optimización agresiva, latencia estricta o restricción anti-lock-in",
    fill: bg-navy, col: navy)
]

== Unidad B1.1 · Arquitectura Databricks Mosaic AI y Topologías de Serving

#unit-header(
  "B1.1",
  "Databricks Mosaic AI y Topologías de Serving (ADRs)",
  "[MOB] 3s + [LAB] 1s",
  "Baseline DevOps/SRE",
  "B1.2, B3.1",
  "Móvil + Databricks Free Edition / Trial",
  "Defender en 3 min cuándo usar Kubernetes/Seldon frente a Databricks Mosaic AI Serverless.",
)

#concepto-clave[
  1. *Anatomía de Databricks Data Intelligence Platform:* Separa estrictamente el *Control Plane* (alojado en Databricks: UI, API, scheduler, metadatos) del *Compute Plane* (*Classic* dentro de la VPC/VNet del cliente vía PrivateLink, o *Serverless* en cuentas aisladas administradas por Databricks con enlace privado).
  2. *Mosaic AI Model Serving:* Unifica tres tipos de *endpoints* bajo la misma capa de gobernanza: (a) *Foundation Model APIs* (pay-per-token o *provisioned throughput* con garantías TPM), (b) *Custom Models* (contenedores MLflow), y (c) *Compound AI / Agents* (orquestación + herramientas + Vector Search con trazabilidad MLflow Tracing).
  3. *Cuándo migrar de Seldon/K8s a Mosaic AI (y viceversa):*
     - *Hacia Mosaic AI:* cuando el dato maestro y los índices vectoriales ya residen en Unity Catalog, el equipo de plataforma es pequeño y la prioridad es *time-to-market* con auditoría y *scale-to-zero* sin operar GPUs ni upgrades de K8s/Istio.
     - *Mantener en K8s/Seldon:* cuando se requiere control granular del *scheduler* vLLM/SGLang, GPUs heterogéneas o *spot*, pipelines complejos de pre/post-procesado con Kafka/gRPC, políticas Istio *shadow traffic*, o portabilidad obligatoria entre nubes sin acoplamiento a un proveedor SaaS.
]

#practica-box(title: "Práctica C2-B1.1 · docs/adr/001-serving-topology-databricks-vs-seldon.md")[
  - Redactar el ADR bajo formato Michael Nygard (*Contexto, Drivers, Opciones evaluadas, Matriz ponderada por SLO/Coste/Gobernanza/Portabilidad, Decisión y Criterios de Reversión*), definiendo umbrales cuantitativos (p. ej., migración a K8s si el gasto mensual supera el umbral de *break-even* frente a réplicas dedicadas o si el p99 exige *traffic mirroring* previo en producción).
]

== Unidad B1.2 · Ciclo de Vida de Modelos y Agentes con MLflow (Promoción `dev -> test -> prod`)

#unit-header(
  "B1.2",
  "MLflow 3 GenAI Lifecycle, Provenance y Gates de Promoción",
  "[MOB] 2s + [LOC] 2s",
  "B1.1",
  "B1.3, B2.1",
  "Docker Compose local (MLflow Tracking + Registry)",
  "Pipeline CI que bloquea promoción a prod si falta firma, hash Git o inmutabilidad.",
)

#concepto-clave[
  1. *MLflow 3 & `LoggedModel` como Ciudadano de Primera Clase:* En GenAI, un agente no es solo un archivo `.pkl`: combina código del grafo (LangGraph/Python), *system prompts*, esquemas de herramientas, versión del modelo LLM y configuración del *retriever*. MLflow modela esto mediante un `LoggedModel` versionado con:
     - *Model Signature:* contrato estricto de entrada/salida (`ChatCompletionRequest` / JSON Schema).
     - *Pinned Dependencies:* `requirements.txt` / `uv.lock` congelado para reproducibilidad determinista.
     - *Artifact Provenance:* metadatos obligatorios (`git_commit_sha`, `dataset_uc_version`, `run_id`, `artifact_sha256`).
  2. *Aliases frente a Stages Legacy:* MLflow reemplaza los antiguos *stages* rígidos (`Staging/Production`) por *aliases mutables sobre versiones inmutables*: `@dev` $arrow.r$ `@staging` $arrow.r$ `@challenger` (tráfico canary 5%) $arrow.r$ `@champion` (producción). Nunca se reescribe una versión ya registrada; una corrección siempre genera una versión nueva (`vN+1`).
  3. *Golden Path de Promoción:* Ninguna promoción a producción ocurre por clic manual sin trazabilidad: CI valida firma + ejecuta *evaluation gate* + exige aprobación humana (*environment protection rule*) antes de mover el alias `@champion`.
]

#practica-box(title: "Práctica C2-B1.2 ([LOC]) · Pipeline MLflow Local y Gate de Inmutabilidad")[
  ```python
  # promote_gate.py - Validación obligatoria antes de mover alias @champion
  from mlflow.tracking import MlflowClient
  client = MlflowClient()
  mv = client.get_model_version(name=model_name, version=version)
  required = {"git_commit_sha", "artifact_sha256", "eval_passed", "approved_by"}
  missing = required - set(mv.tags.keys())
  assert not missing and mv.tags["eval_passed"] == "true", f"Gate bloqueado: {missing}"
  client.set_registered_model_alias(name=model_name, alias="champion", version=version)
  ```
]

== Unidad B1.3 · Governance-by-Design: Unity Catalog Lineage, Políticas y Auditoría

#unit-header(
  "B1.3",
  "Unity Catalog Lineage, RBAC/ABAC y Audit-Ready Evidence",
  "[MOB] 2s + [LOC] 1s + [LAB] 1s",
  "B1.2",
  "B2.2, B2.3",
  "Local Python + Databricks Sandbox",
  "Generador automático de audit-evidence.json + reporte firmado integrado en el pipeline.",
)

#concepto-clave[
  1. *Namespace de 3 Niveles y Gobernanza Unificada:* Todo activo en Unity Catalog sigue la jerarquía `catalog.schema.object` (tablas Delta, volúmenes de archivos no estructurados, índices vectoriales, funciones/herramientas SQL/Python y modelos registrados).
  2. *Control de Acceso RBAC/ABAC y Tablas del Sistema:* Políticas granulares (*Row-Level Security*, *Column Masking* para PII en prompts/documentos RAG) combinadas con auditoría nativa en `system.access.audit` y atribución de costes en `system.billing.usage`.
  3. *End-to-End Lineage & Marketplace Readiness:* Trazabilidad automática desde el documento fuente ingerido en el *Volume*, pasando por la tabla de *chunks* y el índice Vector Search, hasta la versión exacta `@champion` del agente. En entornos regulados (banca/seguros, EU AI Act), el cumplimiento deja de ser un formulario manual lento y se convierte en un *subproducto automático del pipeline* (*Governance-by-Design*).
]

#practica-box(title: "Práctica C2-B1.3 ([LOC]) · Generador de Evidence Bundles (audit-evidence.json)")[
  - Extender el pipeline de B1.2 para emitir tras cada despliegue un paquete inmutable con: `model_uri`, `artifact_sha256`, tabla origen de datos y versión Delta/hash, resultados del *evaluation gate*, comprobación de políticas de enmascaramiento PII, identidad del aprobador (`approved_by`) y firma SHA-256 del propio resumen en Markdown.
]

#ejercicio-box[
  *Ejercicio C2-B1-A (Diseño de Gobernanza):* Un auditor externo te pregunta qué documentos concretos alimentaban el índice RAG del agente de reclamaciones el día 12 a las 14:00 UTC cuando emitió una respuesta errónea, qué versión de prompt/código respondió y quién autorizó su paso a producción. Enumera qué objetos/metadatos de MLflow + Unity Catalog responden cada pregunta sin depender de logs efímeros de consola.

  *Ejercicio C2-B1-B (Promoción Segura):* Un desarrollador sube un ajuste rápido de *system prompt* directamente editando los tags de la versión `v4` ya marcada como `@champion`. ¿Por qué rompe esto la auditoría y cómo debe configurarse el pipeline para impedirlo?
]

#solucion-box[
  - *Sol. C2-B1-A:* (1) *Documentos:* tabla fuente en `catalog.schema.docs` con Delta *Time Travel* (versión/timestamp exacto) + linaje automático al Vector Search Index; (2) *Prompt/código:* `run_id` / `LoggedModel` asociado a la versión con alias activo en ese instante en `system.access.audit` (`git_commit_sha` + `MLmodel` artifacts); (3) *Aprobación:* los tags de versión y alias de MLflow son mutables, por lo que la prueba es el *PR merge SHA* (inmutable en Git) + `audit-evidence.json` con SHA-256 archivado en almacenamiento WORM/versionado; `approved_by` queda como índice de consulta.
  - *Sol. C2-B1-B:* Mutar una versión ya desplegada invalida el `artifact_sha256` y hace imposible reproducir fallos históricos ni hacer *rollback* determinista. Los privilegios de Unity Catalog sobre el modelo de prod (`CREATE MODEL VERSION`, `APPLY TAG`, propiedad/`MANAGE`) deben quedar solo en la *Service Principal* del CI, que registra `v5`, pasa los gates y reasigna `@champion`; los humanos conservan `EXECUTE`. Verifica en tu workspace qué privilegio controla cambiar alias y tags.
]

#fuentes-box[
  1. Databricks Docs — *Mosaic AI Model Serving & Deploy AI Agents*: `https://docs.databricks.com/en/machine-learning/model-serving/`
  2. Databricks Docs — *Unity Catalog Governance, Lineage & System Tables*: `https://docs.databricks.com/en/data-governance/unity-catalog/`
  3. MLflow Docs — *Model Registry, Aliases & GenAI Lifecycle*: `https://mlflow.org/docs/latest/`
  4. Michael Nygard — *Documenting Architecture Decisions (ADR Template)*.
]
