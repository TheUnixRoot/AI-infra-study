#import "../theme.typ": *

= Curso 2 · Bloque 2: Evaluación Agéntica, RAG y Telemetría FinOps (`CHAT-B2`)

#block-banner(
  "C2-B2",
  "Evaluación Agéntica (MLflow), Observabilidad RAG y Telemetría FinOps",
  "Core Enterprise MLOps: trajectory evals, salud de índices vectoriales, OpenTelemetry y control de costes",
  "~2 semanas · 12 sprints de 20m",
)[
  Sustituye la intuición manual ("vibes-based eval") por medición científica de *trayectorias agénticas*, separa fallos de *Retrieval* frente a *Generation* en sistemas RAG, e instrumenta trazas distribuidas *OpenTelemetry + MLflow Tracing* con alertas de *drift* y *circuit breakers* FinOps.
]

#infografia(
  "Pipeline de Evaluación Agéntica, Salud RAG y Telemetría OpenTelemetry + FinOps",
  "Integración de evaluación offline (CI Gate en MLflow) con observabilidad online (spans OTel, métricas Prometheus/OpenSearch y guardarraíles de coste).",
  height: 190pt,
)[
  // Izquierda: Evaluación Offline (Gate)
  #place(dx: 8pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[Evaluación Offline CI (`mlflow.genai.evaluate`)])
  #diag-node(8pt, 20pt, 146pt, 40pt, "Dataset Dorado (15+ casos)", sub: "Happy path + fallos de API\n+ timeouts + prompts ambiguos", fill: bg-light, col: slate)
  #arrow-d(81pt, 60pt, 74pt, col: slate)
  #diag-node(8pt, 74pt, 146pt, 54pt, "Trajectory & Scorers", sub: "• Task Success >= 90%\n• Tool Reliability >= 95%\n  (tool correcta + JSON válido)\n• Sin bucles (steps <= max_steps)", fill: bg-amber, col: amber)
  #arrow-d(81pt, 128pt, 142pt, col: amber)
  #diag-node(8pt, 142pt, 146pt, 38pt, "CI Evaluation Gate", sub: "Bloquea promoción si hay regresión\nvs @champion (A/B diff)", fill: bg-red, col: crimson)

  // Separador
  #place(dx: 164pt, dy: 6pt, line(length: 175pt, angle: 90deg, stroke: 0.5pt + border-gray))

  // Derecha: Spans OTel + RAG Health + FinOps
  #place(dx: 174pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[Trazabilidad Online (OTel Spans -> OpenSearch + Prometheus/Grafana)])
  #diag-node(174pt, 20pt, 294pt, 24pt, "Span Raíz: Agent Run (trace_id, user_id, cost_usd, session_tokens)", fill: bg-navy, col: navy)
  #arrow-d(232pt, 44pt, 58pt, col: teal)
  #arrow-d(321pt, 44pt, 58pt, col: blue)
  #arrow-d(410pt, 44pt, 58pt, col: purple)

  #diag-node(174pt, 58pt, 112pt, 58pt, "Span: RAG Retrieval", sub: "Recall@K · MRR\np95 vector latency\nIndex lag (min)\nEmpty chunk rate %", fill: bg-teal, col: teal)
  #diag-node(292pt, 58pt, 86pt, 58pt, "Span: LLM Call", sub: "TTFT / TPOT\nTokens in/out\nCache hit %\nHallucination judge", fill: bg-navy, col: navy)
  #diag-node(384pt, 58pt, 84pt, 58pt, "Span: Tool Call", sub: "Nombre + Schema\nHTTP status\nRetry count\nTimeout guard", fill: bg-purple, col: purple)

  #diag-node(174pt, 126pt, 294pt, 54pt, "FinOps Guardrails & Online Quality Monitor", sub: "• Cuotas por sesión y Circuit Breaker si steps > 8 o tokens > presupuesto (corta bucles infinitos)\n• Muestreo asíncrono de tráfico prod -> detección de Input/Topic Drift y caída de calidad\n• OpenSearch (logs/trazas) + Prometheus/Grafana (alertas SLO & coste por caso de uso)", fill: bg-light, col: slate)
]

== Unidad B2.1 · Evaluación Offline y Online de Agentes con MLflow

#unit-header(
  "B2.1",
  "Evaluación de Trayectorias: Task Success y Tool Reliability",
  "[MOB] 2s + [LOC] 2s",
  "B1.2",
  "B2.3, B3.2",
  "Local Python + MLflow Tracking",
  "Gate en MLflow que bloquea el CI si Tool Reliability < 95% o Task Success < 90%.",
)

#concepto-clave[
  1. *Por qué no basta evaluar solo el texto final:* Un agente puede dar una respuesta lingüísticamente plausible pero haber invocado dos veces una API mutante (`create_ticket`), inventado un parámetro `policy_id` o reintentado en bucle hasta agotar tokens. La *evaluación de trayectoria* inspecciona la secuencia completa `(pensamiento -> llamadas a herramientas -> observaciones -> respuesta final)`.
  2. *Descomposición de Métricas Agénticas:*
     - *Task Success Rate:* cumplimiento end-to-end del objetivo de negocio (validado por estado final en *mocks* deterministas o juez LLM-as-a-Judge con rúbrica estricta).
     - *Tool Reliability:* producto compuesto de (a) *Tool Selection Accuracy* (eligió la herramienta correcta), (b) *Argument Schema Validity* (JSON válido y valores extraídos sin alucinar), (c) *Error Resilience* (recuperación elegante ante HTTP 503/timeout de una herramienta sin inventar datos) y (d) *Loop Freedom* (sin ciclos repetitivos).
  3. *Scorers Deterministas + LLM-as-a-Judge:* Toda regla verificable por código (esquema JSON, número máximo de pasos, presencia de cita, no exposición de PII) debe ir en *scorers* Python deterministas rápidos y gratuitos; LLM-as-a-Judge se reserva para fidelidad semántica y tono.
]

#practica-box(title: "Práctica C2-B2.1 ([LOC]) · Suite mlflow.evaluate() con 15 Casos y Fallos Simulados")[
  - Construir un agente de soporte asegurador con 3 herramientas (`get_policy`, `calc_coverage`, `create_ticket`), incluyendo 4 casos de *chaos testing* (tool devuelve timeout o 404). Registrar en MLflow scorers deterministas (`tool_reliability`, `valid_json_args`, `no_infinite_loop`) y juez de `task_success`, devolviendo `exit 1` en CI si `tool_reliability < 0.95` o `task_success < 0.90`.
]

== Unidad B2.2 · Observabilidad de RAG y Salud de Búsqueda Vectorial

#unit-header(
  "B2.2",
  "Anatomía de Fallos RAG y Vector Search Health",
  "[MOB] 2s + [LOC] 2s",
  "B1.3",
  "B2.3, B4.1",
  "Docker local (OpenSearch o Qdrant) + Prometheus",
  "Diagnosticar si una degradación de calidad proviene del LLM o del índice vectorial.",
)

#concepto-clave[
  1. *Separación Retrieval vs Generation (Triada RAG):*
     - *Context Recall / Precision (Retrieval):* ¿Estaba el fragmento necesario dentro del `top-K` recuperado? Si no estaba, cambiar de LLM o ajustar el *temperature* *no arreglará el fallo*.
     - *Faithfulness / Groundedness (Generation):* Dado el contexto recuperado, ¿la respuesta se ciñe a él sin contradecirlo ni alucinar?
  2. *Métricas de Salud Operativa del Índice Vectorial:*
     - *Recall\@K, MRR y NDCG:* calidad de ranking frente a un conjunto dorado de pares `(query, doc_id)`.
     - *Index Lag (Frescura):* tiempo transcurrido desde que un documento se actualiza en la fuente (Unity Catalog / S3) hasta que sus embeddings están indexados y buscables.
     - *Empty / Low-Similarity Retrieval Rate:* porcentaje de consultas donde ningún chunk supera el umbral de similitud coseno (señal de *domain shift* o *embedding drift* tras cambiar el modelo de embeddings sin reindexar).
     - *Latencia k-NN (p50/p95/p99):* degradación al crecer el grafo HNSW o por falta de memoria RAM para mantener índices en caché.
  3. *Mosaic AI Vector Search vs OpenSearch k-NN / pgvector:* Mosaic AI ofrece sincronización *Delta Sync* automática e integración de permisos con Unity Catalog; OpenSearch k-NN aporta control completo en Kubernetes, búsqueda híbrida (BM25 + HNSW) y unificación de logs/vectores.
]

#practica-box(title: "Práctica C2-B2.2 ([LOC]) · Colector Prometheus de Vector Search Health")[
  - Levantar OpenSearch (o Qdrant) en Docker Compose, indexar un corpus versionado con timestamps y ejecutar un colector Python periódico que publique métricas Prometheus: `rag_vector_query_latency_ms` (histograma), `rag_index_lag_seconds`, `rag_empty_hits_ratio` y `rag_recall_at_5` sobre un lote sonda.
]

== Unidad B2.3 · Telemetría End-to-End, Drift/Quality Checks y FinOps Guardrails

#unit-header(
  "B2.3",
  "MLflow Tracing, OpenTelemetry, Drift Online y FinOps",
  "[MOB] 2s + [LOC] 2s",
  "B2.1, B2.2",
  "B3.2, B4.1",
  "Docker Compose: OTel Collector + Prometheus + Grafana + OpenSearch",
  "Dashboard end-to-end con trazas, detección de drift y alerta/corte ante sobrecoste.",
)

#concepto-clave[
  1. *Trazabilidad Distribuida (`trace_id` y Spans Anidados):* Con *MLflow Tracing* compatible con *OpenTelemetry*, cada petición genera un árbol de spans: `AGENT_RUN -> RETRIEVER (OpenSearch) -> CHAT_MODEL (vLLM/Mosaic) -> TOOL (API externa)`. Esto permite atribuir exactamente qué porcentaje de los 3.2 segundos totales fue búsqueda vectorial frente a espera de red de una herramienta frente a TTFT del LLM.
  2. *Monitorización Online y Drift:* En producción no existe *ground truth* instantáneo; se muestrea el 5%–10% del tráfico de forma asíncrona (fuera de ruta crítica) para detectar *input/topic drift* (longitud de consultas o clusters semánticos nuevos), picos de rechazos ("No puedo ayudar") y caída de *faithfulness*.
  3. *FinOps Guardrails y Circuit Breakers:* Contabilidad en tiempo real por `tenant_id` / `agent_id`:
     $ "Coste conversación" = sum_("turnos") ( frac(T_"in" dot P_"in" + T_"out" dot P_"out", 10^6) ) $
     y protecciones activas: límite estricto `max_iterations` (p. ej. 6 pasos), techo de tokens por sesión y *circuit breaker* que corta si un agente invoca la misma herramienta con idénticos argumentos $> 2$ veces seguidas.
]

#ejercicio-box[
  *Ejercicio C2-B2-A (Diagnóstico RAG vs LLM):* Tras el despliegue semanal, la métrica `task_success` cae del 93% al 71%. Al revisar tus paneles ves: `faithfulness = 0.96` (alto), `llm_ttft_p95 = 210 ms` (normal), pero `rag_empty_hits_ratio` subió del 3% al 28% y `rag_index_lag_seconds` pasó de `120 s` a `86,400 s`. ¿Dónde está la causa raíz y qué componente debes reparar?

  *Ejercicio C2-B2-B (FinOps Circuit Breaker):* Un agente entra en bucle llamando a `search_docs` con `2,500` tokens de contexto acumulado y `400` tokens de salida en cada turno. Si el precio es `$3.00 / 1M` input y `$15.00 / 1M` output, calcula cuánto cuesta una sola petición si el timeout HTTP permite 25 iteraciones sin límite de pasos frente a poner `max_steps = 5`.
]

#solucion-box[
  - *Sol. C2-B2-A:* La causa raíz está en el pipeline de ingesta/indexado vectorial (*Retrieval failure*, no LLM: la fidelidad al contexto disponible sigue siendo 96%, lo que confirma que el modelo no alucina sino que recibe contexto vacío o desactualizado tras 24h de `index_lag`). Acción: revisar el job de sincronización Delta/embedding hacia el índice vectorial y alertas de frescura antes de tocar prompts o cambiar de modelo.
  - *Sol. C2-B2-B:* Coste por iteración = $(2500 times 3 + 400 times 15) / 10^6 = (7500 + 6000) / 10^6 = \$0.0135$. Con 25 iteraciones: `$0.3375` *por una sola conversación* (con 10,000 usuarios = `$3,375` quemados en un incidente). Limitando a `max_steps = 5` + detección de argumentos duplicados (corte en paso 2): el coste máximo queda acotado a `$0.027 - $0.0675` (*5x–12x menos* impacto).
]

#fuentes-box[
  1. MLflow Docs — *Evaluating Agents & GenAI Apps (`mlflow.genai.evaluate`)*: `https://mlflow.org/docs/latest/genai/eval-monitor/`
  2. MLflow Tracing & OpenTelemetry Docs: `https://mlflow.org/docs/latest/tracing/`
  3. Databricks Docs — *Mosaic AI Vector Search & RAG Evaluation*: `https://docs.databricks.com/en/generative-ai/vector-search.html`
  4. OpenSearch Docs — *Vector Search & k-NN Monitoring*: `https://opensearch.org/docs/latest/search-plugins/knn/`
]
