#import "theme.typ": *

#show: master-setup.with(
  title: "Curso 3 · Edge-AI Sub-2-Bit, Kernels Portables & Réplica PrismML",
  subtitle: "De Qwen3.5-2B y Gemma 4 E2B (<1 GB) a CUDA sm_61 (__dp4a), Vulkan GLSL y < 8 W",
  edition: "Curso 3 · Edge-AI (B3-Edge)",
)

#cover-page(
  "Itinerario 3 · Edición de Nivel Máster (Presupuesto 0 EUR)",
  "Edge-AI Sub-2-Bit & Portable Kernels",
  "Replicación open-source de Ternary Bonsai (PrismML), PTQ Hadamard + GPTQ CPU, portabilidad a Gemma 4 E2B, kernels CUDA sm_61 / Vulkan Compute y despliegue < 8 W",
  [
    Extensión práctica (`B3-Edge`) totalmente compatible con el *Curso 1 (AI Infrastructure)* y diseñada como puente directo hacia *HPAC/UMA (Rafael Asenjo)* como evolución moderna de `LogFit`. Todo el itinerario (`E0`–`E5`) se ejecuta a *0 EUR* sobre hardware propio: *PC (CPU 16 GB+ + GTX 1050 `sm_61`)* y *móvil Android (`Termux` + ARM64 NEON + Vulkan)*, mediante micro-sprints de *20 minutos*.
  ],
  (
    (
      title: "0 · Tronco Común & Enganche B3-Edge",
      bg: bg-navy,
      col: navy,
      body: [
        Incluye el Capítulo 0 (*Tronco Común*: lo común vs. lo específico entre los 3 cursos) y la ruta de enganche incremental sobre `B2`/`B3` del Curso 1 sin quemarte ni gastar en cloud.
      ],
    ),
    (
      title: "E0–E1 · Telemetría mWh/token & Baseline",
      bg: bg-teal,
      col: teal,
      body: [
        `llama.cpp` *upstream* + `PrismML-llama.cpp`, calibración física `sysfs`/`NVML`/`RAPL` (`mWh/token` con varianza `< 5%`) y línea base `Qwen3.5-2B` vs `Bonsai 2`.
      ],
    ),
    (
      title: "E2–E3 · Hadamard + GPTQ CPU & Gemma 4",
      bg: bg-amber,
      col: amber,
      body: [
        Empaquetado `TQ1_0` base-3 sin divisiones, rotación `FHT` $O(N log N)$, `GPTQ` capa a capa (`g128`), protección `PLE`/`lm_head` en `Gemma 4 E2B` (`< 1 GB`) y evals `GSM8K`/`IFEval`.
      ],
    ),
    (
      title: "E4–E6 · CUDA __dp4a, Vulkan & Dossier UMA",
      bg: bg-purple,
      col: purple,
      body: [
        `GEMV` ternario `sm_61` + `FHT 1024`, portado a *Vulkan GLSL/SPIR-V* *zero-copy* móvil, API `< 8 W` 24/7, experimento *LogFit-Edge*, dossier 1 pág. UMA y bonus QAT `STE+KL`.
      ],
    ),
  ),
)

#outline(title: "Índice del Curso 3", depth: 2, indent: 1.4em)

#pagebreak(weak: true)
#include "sections/00_tronco_comun.typ"

#pagebreak(weak: true)
= Itinerario Técnico del Curso 3 (`E0`–`E6`)
#include "sections/c3_e0_e1.typ"
#pagebreak(weak: true)
#include "sections/c3_e2.typ"
#pagebreak(weak: true)
#include "sections/c3_e3.typ"
#pagebreak(weak: true)
#include "sections/c3_e4.typ"
#pagebreak(weak: true)
#include "sections/c3_e5_e6.typ"
