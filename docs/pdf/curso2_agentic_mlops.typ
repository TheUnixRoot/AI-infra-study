#import "theme.typ": *

#show: master-setup.with(
  title: "Curso 2 · Enterprise MLOps & Agentic Platform Engineering",
  subtitle: "Databricks Mosaic AI, MLflow 3, Unity Catalog, Evals, K8s/Seldon/Istio y SRE",
  edition: "Curso 2 · Agentic MLOps",
)

#cover-page(
  "Itinerario 2 · Edición de Nivel Máster",
  "Enterprise MLOps & Agentic Platform Engineering",
  "Cierre completo de brechas desde Senior DevOps / SRE hacia plataformas corporativas de IA generativa y agentes en producción regulada",
  [
    Manual operativo diseñado específicamente para ingenieros con base *Senior DevOps / SRE* (Kubernetes, IaC, CI/CD, Prometheus/Grafana, gestión de incidentes). No repite fundamentos DevOps básicos: va directo al núcleo de una plataforma agéntica empresarial a través de *micro-sprints de 20 minutos* (`[MOB]`, `[LOC]`, `[PREP]`, `[LAB]`), construyendo el proyecto integrador `agentic-platform-golden-path`.
  ],
  (
    (
      title: "Sistema Operativo de Estudio (4 Chats)",
      bg: bg-navy,
      col: navy,
      body: [
        Regla 1-a-1 (`CHAT-B1` a `CHAT-B4`), bloques de *Handover* cada 20 minutos, y plantillas de delegación para tu Agente de Código con `verify.sh < 60s`.
      ],
    ),
    (
      title: "B1 · Databricks, MLflow 3 y Unity Catalog",
      bg: bg-teal,
      col: teal,
      body: [
        Control Plane vs Compute Plane, `LoggedModel` y aliases `@dev -> @staging -> @champion`, firmas estrictas, linaje end-to-end y paquete audit-ready.
      ],
    ),
    (
      title: "B2 · Evals Agénticas, RAG, OTel y FinOps",
      bg: bg-purple,
      col: purple,
      body: [
        Puertas automáticas con `mlflow.evaluate()`, jueces LLM, salud vectorial OpenSearch, trazas OTel GenAI, guardrails PII y atribución de coste por tenant.
      ],
    ),
    (
      title: "B3–B4 · K8s Serving, Crossplane y SRE",
      bg: bg-amber,
      col: amber,
      body: [
        Seldon Core v2, Knative KPA, Istio shadow/canary + Argo Rollouts, `XAgenticEnvironment` en Crossplane, error budgets agénticos y failover multi-región EU.
      ],
    ),
  ),
)

#outline(title: "Índice del Curso 2", depth: 2, indent: 1.4em)

#pagebreak(weak: true)
#include "sections/00_tronco_comun.typ"

#pagebreak(weak: true)
= Itinerario Técnico del Curso 2 (`B1`–`B4`)
#include "sections/c2_b1.typ"
#pagebreak(weak: true)
#include "sections/c2_b2.typ"
#pagebreak(weak: true)
#include "sections/c2_b3.typ"
#pagebreak(weak: true)
#include "sections/c2_b4.typ"
