# Agentic Platform MLOps — Master Playbook & Sistema Operativo de Estudio

> **Propósito de este documento:** Fuente de verdad única para dominar el stack, la arquitectura y las prácticas de **Enterprise MLOps & Agentic Platform Engineering** exigidas para roles de plataforma agéntica en entornos corporativos regulados.
>
> **Punto de partida (Baseline):** Perfil Senior DevOps / SRE (CI/CD empresarial, promoción multi-entorno, gestión de secretos/identidad, IaC en AWS/Azure, Kubernetes, SLOs, respuesta a incidentes y Prometheus/Grafana). Este plan **no repite fundamentos básicos de DevOps**, sino que cierra al 100 % las brechas específicas del puesto: ecosistema **Databricks (MLflow, Mosaic AI, Unity Catalog)**, **gobernanza y auditoría por diseño**, **evaluación y observabilidad de agentes/RAG**, y **serving cloud-native (Seldon, Knative, Istio, ArgoCD, Crossplane, OpenSearch)**.
>
> **Restricción operativa:** Todo el temario se ejecuta en **micro-sprints de 20 minutos** combinando lectura técnica desde el móvil (`[MOB]`) y ejecución práctica en local (`[LOC]` con Docker/Kind/MLflow/Python) delegada a un **Agente de Código**, más laboratorios gratuitos en **Databricks Free Edition / Trial** (`[LAB]`).

---

## 0. Sistema Operativo del Cuaderno (Reglas de Higiene, Handover y Fork)

1. **Regla 1-a-1 (Máximo 4 chats activos):** Nunca existirá más de **1 chat activo por bloque** (`CHAT-B1` a `CHAT-B4`). Cuando un chat crezca demasiado, se cierra con un bloque **Handover** y se abre un **Fork** limpio.
2. **Separación de responsabilidades:**
   - **Este Cuaderno (LLM):** Tutor técnico, arquitecto de plataforma, revisor de ADRs (*Architecture Decision Records*), diseñador de rúbricas de evaluación agéntica y generador de especificaciones para tu agente de código.
   - **Tu Agente de Código (CLI/IDE):** Genera pipelines CI/CD, manifests de Kubernetes/ArgoCD/Crossplane/Istio/Seldon, scripts de evaluación en MLflow, configuraciones de OpenTelemetry/Prometheus y documentación técnica.
3. **Clasificación de Micro-Sprints (20 min):**
   - `[MOB]` **Móvil (Solo lectura/texto):** Lectura de documentación oficial, arquitecturas de referencia y preguntas de validación en el chat del bloque.
   - `[LOC]` **Local (Docker / Kind / Python):** Ejecución en tu máquina de MLflow Tracking/Evaluate, clúster Kubernetes local (Kind/Minikube), simuladores de RAG/agentes, Prometheus/Grafana y tests de políticas.
   - `[PREP]` **Pre-Flight (Código listo antes de probar):** Tu agente de código prepara los scripts, pipelines y tests reproducibles.
   - `[LAB]` **Sandbox Cloud (0 €):** Pruebas puntuales en Databricks Free Edition / Trial para validar Unity Catalog, MLflow Model Registry y Mosaic AI Agent Framework.

### Plantillas de Control (Copiar y Pegar)

#### A. Prompt de Inicio / Fork de Chat de Bloque

```text
[FORK CHAT-Bx]
Contexto: Lee `docs/temario/01-agentic-mlops-playbook.md` en las fuentes.
Bloque activo: [Bx] | Unidad actual: [Bx.y]
Estado previo (Handover): <pegar último bloque de handover o "Inicio de bloque">
Tiempo disponible ahora: 20 minutos | Modo: [MOB / LOC / PREP / LAB]
Objetivo de este micro-sprint: <ej. estudiar Unity Catalog lineage / generar pipeline MLflow Evaluate>
Regla: Respuestas concisas, legibles en móvil, sin relleno. Al final dame el bloque de Handover actualizado.
```

#### B. Bloque de Handover (Al cerrar cualquier micro-sesión de 20 min)

```yaml
HANDOVER:
  unidad_actual: "B1.1"
  estado: "en_progreso" # pendiente | en_progreso | done
  aprendido_o_verificado:
    - "Diferencia entre Databricks Mosaic AI Serverless y Seldon en K8s"
  artefacto_generado: "docs/adr/001-serving-topology.md"
  bloqueos_o_dudas: "Pendiente validar permisos de linaje en Unity Catalog"
  siguiente_micro_sprint_20m: "B1.2 [LOC] - Configurar MLflow Tracking + Evaluation Gate local"
```

#### C. Prompt de Delegación para tu Agente de Código

```text
Actúa como Staff MLOps & Platform Engineer. Vamos a construir el entregable para [Bx.y].
Stack objetivo: [MLflow / Kubernetes (Kind) / ArgoCD / Seldon / Istio / OpenTelemetry / Prometheus].
Requisitos:
1. Código declarativo, reproducible y versionable en Git.
2. Script de validación `verify.sh` (<60s) que demuestre el funcionamiento con salida clara por stdout.
3. Evidencias estructuradas (JSON/Markdown) listas para auditoría y portafolio.
```

---

## 1. Mapa de Dependencias y Proyecto Integrador de Portfolio

```text
[B1.1 Databricks & Mosaic AI] ──► [B1.2 MLflow Lifecycle & Gates] ──► [B1.3 Unity Catalog & Audit]
                                             │                                    │
                                             ▼                                    ▼
[B2.1 Agentic Evals (MLflow)] ──► [B2.2 RAG & Vector Health] ──► [B2.3 Telemetry, Drift & FinOps]
                                             │                                    │
                                             ▼                                    ▼
[B3.1 Serving: Seldon vs Mosaic] ─► [B3.2 Istio + ArgoCD Canary] ─► [B3.3 Crossplane Golden Paths]
                                                                                  │
                                                                                  ▼
                                                   [B4.1 Agentic SLOs & SRE] ──► [B4.2 Multi-Cloud & Latency]
                                                                                  │
                                                                                  ▼
                                                                           [B4.3 Cierre Portfolio]
```

### Proyecto Integrador: `agentic-platform-golden-path`

Cada bloque construye una pieza de un repositorio de referencia listo para mostrar en entrevista técnica:

- **Capa de Gobierno y Ciclo de Vida (B1):** Pipeline CI/CD de promoción `dev -> test -> prod` con artefactos inmutables, aprobaciones obligatorias, linaje de datos/modelos y paquete de evidencias *audit-ready*.
- **Capa de Evaluación y Observabilidad (B2):** Arnés automático con `mlflow.genai.evaluate()` que mide *task success*, fiabilidad de *tool calling* y salud de búsqueda vectorial (RAG), más paneles de telemetría de costes y latencia.
- **Capa de Serving y GitOps (B3):** Despliegue en Kubernetes con Seldon Core / Knative + Istio (*canary rollout* y *rollback* automático), manifiestos ArgoCD, abstracción Crossplane y un ADR comparando K8s/Seldon frente a Databricks Mosaic AI Serverless.
- **Capa de Operación SRE y Multi-Cloud (B4):** Definición formal de SLOs/SLIs para agentes, simulacro de incidente con post-mortem y estrategia de enrutamiento multi-región con control de latencia.

---

## 2. Bloque 1: Ecosistema Databricks, MLflow y Gobernanza por Diseño (`CHAT-B1`)

**Objetivo:** Dominar el núcleo corporativo de plataformas de IA gestionadas, trazabilidad de artefactos y cumplimiento normativo sin fricción (*governance-by-design*).

### `[B1.1]` Arquitectura Databricks Mosaic AI y Topologías de Serving
- **Modo:** `[MOB]` (3 sprints) + `[LAB]` (1 sprint)
- **Conceptos clave:**
  - Anatomía de Databricks Data Intelligence Platform: Control Plane vs. Compute Plane, Serverless Compute y Mosaic AI Model Serving (endpoints de modelos fundacionales, modelos personalizados y agentes compuestos).
  - Comparativa arquitectónica: Databricks Serverless (cero gestión de nodos, escalado automático, integración nativa con Unity Catalog) vs. Kubernetes / Seldon (control total de red, GPU/CPU custom, portabilidad multi-cloud).
  - Redacción de ADRs (*Architecture Decision Records*): cómo justificar criterios de migración desde Seldon en K8s hacia Databricks Mosaic AI según latencia, aislamiento, coste y cumplimiento.
- **Recursos en texto (móvil):**
  1. Databricks Docs — *Mosaic AI Model Serving Overview y Deploy AI Agents*: `https://docs.databricks.com/en/machine-learning/model-serving/`
  2. Documentación de patrones ADR (Michael Nygard ADR template).
- **Práctica:** Redactar `docs/adr/001-serving-topology-databricks-vs-seldon.md` definiendo matriz de decisión (SLOs, coste, gobernanza, portabilidad).
- **Done:** Puedes defender en 3 minutos cuándo mantener un modelo/agente en Kubernetes/Seldon y cuándo migrarlo a Databricks Mosaic AI Serverless.

### `[B1.2]` Ciclo de Vida de Modelos y Agentes con MLflow (Promoción `dev -> test -> prod`)
- **Modo:** `[MOB]` (2 sprints) + `[LOC]` (2 sprints)
- **Conceptos clave:**
  - MLflow 3 / GenAI Lifecycle: `LoggedModel` como ciudadano de primera clase, firmas (*signatures*), dependencias fijadas, *Artifact Provenance* (hash de código Git, dataset y parámetros) y *releases* inmutables.
  - Model Registry y Aliases: Gestión de estados mediante aliases (`@dev`, `@staging`, `@champion`, `@challenger`) en lugar de etapas rígidas.
  - Diseño de *Golden Paths* de CI/CD: pipelines automatizados con *approval gates* humanos y técnicos antes de promover a `prod`, más patrones de *rollback* y *canary*.
- **Recursos en texto (móvil):**
  1. MLflow Docs — *MLflow Model Registry & Aliases + MLflow for GenAI / Agents*: `https://mlflow.org/docs/latest/`
- **Práctica (Agente -> Local Docker/MLflow):** Levantar MLflow Tracking + Registry local con Docker Compose y crear un pipeline CI (GitHub Actions / script local) que empaquete un agente con herramientas, valide su firma, registre su procedencia Git y ejecute una promoción controlada entre entornos con aprobación.
- **Done:** Pipeline funcional que impide promover un artefacto a `prod` si no es inmutable o carece de metadatos de procedencia.

### `[B1.3]` Governance-by-Design: Unity Catalog Lineage, Políticas y Auditoría
- **Modo:** `[MOB]` (2 sprints) + `[LOC]` (1 sprint) + `[LAB]` (1 sprint)
- **Conceptos clave:**
  - Unity Catalog (UC): Jerarquía de 3 niveles (`catalog.schema.model/table/volume`), control de acceso basado en atributos/roles (RBAC/ABAC), tablas del sistema (`system.access.audit`, `system.billing.usage`).
  - Linaje automático (*End-to-End Lineage*): Trazabilidad desde las tablas/documentos fuente del índice vectorial hasta el modelo/agente desplegado en producción.
  - Evidencias listas para auditoría (*Audit-Ready Evidence*) y *Marketplace Readiness*: Cómo capturar automáticamente en el pipeline CI/CD los logs de aprobación, resultados de evaluación de seguridad/calidad, hash del artefacto y políticas aplicadas para superar auditorías internas o regulatorias (sector financiero/asegurador).
- **Recursos en texto (móvil):**
  1. Databricks Docs — *Unity Catalog Governance & Lineage y System Tables*: `https://docs.databricks.com/en/data-governance/unity-catalog/`
- **Práctica (Agente -> Local):** Crear un generador de *Evidence Bundles* (`audit-evidence.json` + reporte firmado en Markdown) integrado en el pipeline de B1.2 que recopile: quién aprobó, qué datos/índices alimentan el agente, resultados de tests y políticas de acceso verificadas.
- **Done:** Sabes explicar cómo automatizar el cumplimiento normativo dentro del pipeline sin frenar la velocidad de entrega de los equipos.

---

## 3. Bloque 2: Evaluación Agéntica, Observabilidad RAG y Telemetría de Costes (`CHAT-B2`)

**Objetivo:** Dominar la medición científica del comportamiento de agentes y sistemas RAG en offline y online, detectando regresiones, *drift* y sobrecostes.

### `[B2.1]` Evaluación Offline y Online de Agentes con MLflow (Task Success y Tool Reliability)
- **Modo:** `[MOB]` (2 sprints) + `[LOC]` (2 sprints)
- **Conceptos clave:**
  - Evaluación tradicional de LLMs vs. Evaluación Agéntica: medir la trayectoria completa (*trajectory evaluation*), no solo el texto final.
  - Métricas críticas del puesto:
    - **Task Success Rate:** ¿Completó el agente el objetivo de negocio de extremo a extremo?
    - **Tool Reliability:** Selección correcta de herramienta (*tool selection accuracy*), validez de argumentos JSON (*parameter extraction*), manejo de errores/timeouts de APIs y ausencia de bucles infinitos.
  - MLflow Evaluate & LLM-as-a-Judge: Creación de *scorers* personalizados deterministas (Python) + jueces semánticos, comparación A/B entre versiones del agente y *Evaluation Gates* en CI/CD.
- **Recursos en texto (móvil):**
  1. MLflow Docs — *Evaluating Agents & GenAI Apps (`mlflow.genai.evaluate`)*: `https://mlflow.org/docs/latest/genai/eval-monitor/`
- **Práctica (Agente -> Local):** Construir un agente de prueba con 3 herramientas (consulta de póliza, cálculo de cobertura, creación de ticket) y una suite `mlflow.genai.evaluate()` con 15 casos de prueba (incluyendo fallos simulados de herramienta) que bloquee el CI si *Tool Reliability* $< 95\%$ o *Task Success* $< 90\%$.
- **Done:** Tienes un *gate* automatizado en MLflow que mide fiabilidad de herramientas y éxito de tareas antes de cada despliegue.

### `[B2.2]` Observabilidad de RAG y Salud de Búsqueda Vectorial (Vector Search Health)
- **Modo:** `[MOB]` (2 sprints) + `[LOC]` (2 sprints)
- **Conceptos clave:**
  - Anatomía de fallos en RAG: Separar errores de *Retrieval* (no encontró el documento) frente a errores de *Generation* (alucinó teniendo el contexto).
  - Métricas de *Vector Search Health*: `Recall@K`, MRR/NDCG, latencia de consulta vectorial (p50/p95/p99), frescura del índice (*index lag* desde que cambia un documento hasta que es buscable), tasa de fragmentos vacíos/irrelevantes y *embedding drift*.
  - Mosaic AI Vector Search vs. motores vectoriales en Kubernetes (OpenSearch k-NN / pgvector): sincronización delta y control de permisos a nivel de documento.
- **Recursos en texto (móvil):**
  1. Databricks Docs — *Mosaic AI Vector Search & RAG Evaluation*: `https://docs.databricks.com/en/generative-ai/vector-search.html`
  2. OpenSearch Docs — *Vector Search & k-NN Monitoring*: `https://opensearch.org/docs/latest/search-plugins/knn/`
- **Práctica (Agente -> Local):** Implementar un colector de métricas de salud de RAG sobre una base vectorial local (OpenSearch o Qdrant/Chroma en Docker) que evalúe relevancia de contexto (*context precision/recall*), latencia de búsqueda y frescura de ingesta, exponiendo métricas a Prometheus.
- **Done:** Sabes diagnosticar si una caída en calidad de un agente se debe al LLM o a una degradación silenciosa del índice vectorial.

### `[B2.3]` Telemetría End-to-End, Drift/Quality Checks y FinOps Guardrails
- **Modo:** `[MOB]` (2 sprints) + `[LOC]` (2 sprints)
- **Conceptos clave:**
  - MLflow Tracing & OpenTelemetry (OTel): Instrumentación de *spans* anidados (entrada usuario -> búsqueda RAG -> llamada LLM -> ejecución de tool -> respuesta final) con correlación de `trace_id`.
  - Detección de *Drift* y Calidad Online: Monitorización continua en producción (muestreo de tráfico real evaluado asíncronamente, detección de *input/topic drift*, aumento de rechazos o alucinaciones).
  - *Cost & Performance Guardrails*: Tracking de tokens de entrada/salida por agente, usuario y caso de uso; cuotas por sesión, *circuit breakers* ante bucles agénticos y dashboards unificados en OpenSearch + Grafana.
- **Recursos en texto (móvil):**
  1. MLflow Tracing & OpenTelemetry Integration Docs: `https://mlflow.org/docs/latest/tracing/`
- **Práctica (Agente -> Local):** Desplegar con Docker Compose un stack OpenTelemetry Collector + Prometheus + Grafana + OpenSearch que capture trazas de ejecución agéntica, calcule el coste exacto por conversación y dispare una alerta si un agente supera el presupuesto de tokens o cae su puntuación de calidad online.
- **Done:** Dashboard operativo completo con trazabilidad *end-to-end*, detección de deriva y guardarraíles de coste activos.

---

## 4. Bloque 3: Serving Cloud-Native (Seldon, Knative, Istio) y GitOps (`CHAT-B3`)

**Objetivo:** Dominar el stack de plataforma en Kubernetes exigido para *serving* avanzado, despliegues progresivos y autoservicio de infraestructura.

### `[B3.1]` Serving de Modelos y Agentes en Kubernetes: Seldon Core v2 y Knative
- **Modo:** `[MOB]` (2 sprints) + `[LOC]` (2 sprints)
- **Conceptos clave:**
  - Seldon Core v2: Arquitectura orientada a datos (Kafka/gRPC/REST), `Model` y `Pipeline` CRDs (encadenamiento de preprocesado -> modelo -> postprocesado/guardrails), *multi-model serving* y explicación/monitorización integrada.
  - Knative Serving: Autoscaling basado en concurrencia/RPS (KPA/HPA), *scale-to-zero* para entornos dev/test o cargas intermitentes, gestión de *Revisions* y división de tráfico.
  - Cuándo usar Seldon Pipelines frente a orquestadores agénticos en Python (LangGraph/LlamaIndex) sobre Knative.
  - **Licencia:** desde enero de 2024 Seldon Core v1/v2 usan BSL 1.1 (producción requiere licencia comercial); valóralo en el ADR.
- **Recursos en texto (móvil):**
  1. Seldon Core v2 Architecture & Concepts Docs: `https://docs.seldon.io/projects/seldon-core/en/v2/`
  2. Knative Serving Docs — *Autoscaling & Traffic Management*: `https://knative.dev/docs/serving/`
- **Práctica (Agente -> Local Kind):** Desplegar en un clúster local Kind un servicio de inferencia/agente usando Knative Serving (o Seldon Core v2) con políticas de autoscaling por concurrencia y pruebas de carga básicas.
- **Done:** Tienes manifiestos funcionales y entiendes cómo escala y enruta peticiones cada pieza en Kubernetes.

### `[B3.2]` Despliegues Progresivos y Service Mesh: Istio + ArgoCD
- **Modo:** `[MOB]` (2 sprints) + `[LOC]` (2 sprints)
- **Conceptos clave:**
  - Istio Service Mesh: `VirtualService` y `DestinationRule` para *weighted traffic splitting* (95% `@champion` / 5% `@challenger`), *traffic mirroring / shadow testing* (evaluar un nuevo agente con tráfico real sin afectar al usuario), mTLS estricto y timeouts/retries seguros para *tool calls*.
  - (Elige una ruta: Knative reparte tráfico por *Revisions*; Argo Rollouts gestiona `Rollout`, no Services de Knative.) ArgoCD + Argo Rollouts: GitOps declarativo multi-entorno (`dev -> test -> prod`), sincronización automática con aprobación manual en producción, *Canary Rollouts* guiados por métricas de Prometheus/MLflow (si sube latencia p99 o baja *tool reliability*, *auto-rollback* inmediato).
- **Recursos en texto (móvil):**
  1. Istio Docs — *Traffic Management & Mirroring*: `https://istio.io/latest/docs/tasks/traffic-management/`
  2. Argo Rollouts Docs — *Canary & Prometheus Analysis*: `https://argo-rollouts.readthedocs.io/`
- **Práctica (Agente -> Local Kind):** Configurar en Kind un Rollout de Argo con análisis de métricas Prometheus que divida tráfico 90/10 entre dos versiones de un agente y ejecute un *rollback* automático cuando simules una degradación de latencia o tasa de error.
- **Done:** Demostración reproducible de despliegue *canary* con *auto-rollback* basado en SLOs reales.

### `[B3.3]` Infraestructura como Código y Golden Paths con Crossplane + Cloud Networking
- **Modo:** `[MOB]` (2 sprints) + `[LOC]` (1 sprint)
- **Conceptos clave:**
  - Crossplane (v2: composiciones solo mediante *funciones*, sin patch-and-transform nativo): *Control plane* de infraestructura nativo en Kubernetes; Providers (AWS/Azure), Managed Resources, `CompositeResourceDefinitions` (XRDs) y `Compositions` para ofrecer *Golden Paths* de autoservicio (ej. un equipo pide un `AgenticWorkspace` con un YAML sencillo y Crossplane aprovisiona bucket de artefactos, base vectorial, secretos e identidad IAM).
  - Cloud Networking & Security (Azure / AWS): Private Endpoints / PrivateLink hacia Databricks y servicios de modelos, Zero-Trust networking, Workload Identity / IRSA para eliminar credenciales estáticas en pipelines y pods.
- **Recursos en texto (móvil):**
  1. Crossplane Docs — *Compositions & Platform Engineering*: `https://docs.crossplane.io/`
- **Práctica (Agente -> Local Kind):** Instalar Crossplane en Kind y crear una XRD + Composition de ejemplo (`XAgenticEnvironment`) que abstraiga el aprovisionamiento de recursos de un equipo de producto.
- **Done:** Sabes explicar cómo convertir infraestructura compleja de nube en un *Golden Path* seguro y gobernado para equipos de IA.

---

## 5. Bloque 4: Enterprise SRE, Multi-Cloud Latency y Cierre de Portfolio (`CHAT-B4`)

**Objetivo:** Consolidar la mentalidad de operación crítica (SLOs, gestión de incidentes, capacidad y multi-región) y empaquetar todo el trabajo para entrevistas.

### `[B4.1]` Operación por SLOs, Capacity Planning e Incident Response para Agentes
- **Modo:** `[MOB]` (2 sprints) + `[LOC]` (1 sprint)
- **Conceptos clave:**
  - SLIs y SLOs específicos de plataformas agénticas:
    - Latencia extremo a extremo y TTFT (*Time To First Token*) p95/p99.
    - Disponibilidad del servicio y *Tool Execution Success Rate*.
    - *Quality SLO* (puntuación mínima en muestreo online).
  - *Capacity Planning*: Cálculo de concurrencia, cuotas de RPM/TPM (*Requests/Tokens Per Minute*) en endpoints serverless frente a réplicas provisionadas en K8s, gestión de *rate limits* de proveedores externos.
  - *Incident Response & Post-Mortems Blameless*: Runbooks operativos para fallos típicos de agentes (caída de API externa de herramienta, saturación de índice vectorial, bucle infinito de agente, deriva de modelo tras actualización de proveedor).
- **Práctica (Agente -> Docs):** Redactar 2 Runbooks operativos y 1 plantilla de Post-Mortem aplicada a un incidente real simulado (ej. fallo silencioso de *tool calling* tras cambio de esquema en una API interna).
- **Done:** Documentación operativa lista para defender en rondas de SRE / Platform Operations.

### `[B4.2]` Operaciones Multi-Cloud y Gestión de Latencia Cross-Region
- **Modo:** `[MOB]` (2 sprints)
- **Conceptos clave:**
  - Arquitecturas Multi-Cloud / Multi-Región (Azure + AWS): Enrutamiento basado en latencia y salud (DNS/Anycast/API Gateway), estrategias activo-pasivo vs. activo-activo para *serving* de agentes.
  - *Cross-Region Latency Management*: Impacto de la distancia geográfica entre el clúster de ejecución del agente, la base de datos vectorial y los endpoints de modelos fundacionales; *caching* semántico y colocación de datos para cumplir soberanía de datos en la UE (RGPD / regulación financiera).
- **Done:** Puedes diseñar en pizarra una topología multi-región con *failover* automático respetando residencia de datos europea y presupuestos de latencia.

### `[B4.3]` Ensamblaje Final del Repositorio y Preparación de Entrevista (Inglés)
- **Modo:** `[PREP]` (1 sprint) + `[LOC]` (2 sprints)
- **Conceptos clave:**
  - Consolidación de `agentic-platform-golden-path` con README ejecutivo en inglés, diagramas de arquitectura (Mermaid), evidencias de MLflow, manifests de Seldon/Istio/ArgoCD/Crossplane y ADRs.
  - Mapeo de discurso en inglés (*STAR stories*) conectando tu experiencia previa DevOps/SRE con cada requisito de la oferta.
- **Done:** Repositorio público impecable y guion técnico en inglés listo para aplicar y entrevistar.

---

## 6. Registro Maestro de Progreso

| ID | Unidad | Modo | Estado |
| :--- | :--- | :---: | :---: |
| `B1.1` | Databricks Mosaic AI & ADRs (Serverless vs Seldon) | `[MOB]` `[LAB]` | `[ ]` |
| `B1.2` | MLflow Lifecycle, Provenance & Promoción `dev->test->prod` | `[MOB]` `[LOC]` | `[ ]` |
| `B1.3` | Unity Catalog Lineage, Políticas y Audit Evidence | `[MOB]` `[LOC]` `[LAB]` | `[ ]` |
| `B2.1` | Evaluación Agéntica MLflow (Task Success & Tool Reliability) | `[MOB]` `[LOC]` | `[ ]` |
| `B2.2` | Observabilidad RAG & Vector Search Health | `[MOB]` `[LOC]` | `[ ]` |
| `B2.3` | Telemetría OTel/OpenSearch/Grafana, Drift & FinOps Guardrails | `[MOB]` `[LOC]` | `[ ]` |
| `B3.1` | Serving en K8s: Seldon Core v2 y Knative | `[MOB]` `[LOC]` | `[ ]` |
| `B3.2` | Progressive Delivery con Istio + ArgoCD Canary/Rollback | `[MOB]` `[LOC]` | `[ ]` |
| `B3.3` | Golden Paths con Crossplane + Cloud Networking (AWS/Azure) | `[MOB]` `[LOC]` | `[ ]` |
| `B4.1` | Operación por SLOs, Capacity Planning & Incident Runbooks | `[MOB]` `[LOC]` | `[ ]` |
| `B4.2` | Operaciones Multi-Cloud & Cross-Region Latency Management | `[MOB]` | `[ ]` |
| `B4.3` | Ensamblaje `agentic-platform-golden-path` & Entrevista EN | `[PREP]` `[LOC]` | `[ ]` |
