#import "theme.typ": *

#show: master-setup.with(
  title: "Curso 1 · AI Infrastructure, GPU Systems & Inference Engines",
  subtitle: "De memoria HBM y CUDA C++ a vLLM, FP8, NVLink y Benchmarking Multi-Nodo",
  edition: "Curso 1 · AI Infra",
)

#cover-page(
  "Itinerario 1 · Edición de Nivel Máster",
  "AI Infrastructure, GPU Systems & Inference Engines",
  "Arquitectura física de GPU, programación CUDA/Triton, serving de alto rendimiento, cuantización, entrenamiento distribuido e infraestructura multi-nodo",
  [
    Itinerario diseñado para dominar la pila técnica de un *AI Infrastructure / Inference Systems Engineer*: desde el comportamiento de caché y *coalescing* en CUDA C++ hasta la operación de motores modernos (vLLM, SGLang, TensorRT-LLM) y clústeres multi-GPU (`NVLink`, `NCCL`, `RoCEv2`). Todo el código local está acotado para ejecutarse sin OOM en una *GTX 1050 (2 GB VRAM)* y escalar mediante flujo `[PREP]` $arrow.r$ `[CLD]` a GPUs *L4 / L40S / H100*.
  ],
  (
    (
      title: "Tronco Común + Base Físico-Matemática",
      bg: bg-navy,
      col: navy,
      body: [
        Incluye el Capítulo 0 (*Tronco Común*) para unificar conceptos de Prefill/Decode, intensidad aritmética (`FLOP/byte`) y SLOs de latencia antes de bajar al hardware.
      ],
    ),
    (
      title: "Bloques B1–B2: Hardware, Kernels y Serving",
      bg: bg-teal,
      col: teal,
      body: [
        Mediciones reales en GTX 1050 (`b1_1_strides`, `b1_2_cuda_gemm`, `b1_3_profiling`, `b2_1_kv_cache`) y preparación completa de comparativa `vLLM` vs `SGLang`.
      ],
    ),
    (
      title: "Bloques B3–B5: FP8, 3D Parallelism y Clústeres",
      bg: bg-amber,
      col: amber,
      body: [
        Cuantización AWQ/FP8 y TensorRT-LLM, paralelismo `DP/TP/PP/EP`, arquitecturas MoE/DeepSeek, Kubernetes/MIG, NCCL AllReduce y diagnósticos con DCGM.
      ],
    ),
    (
      title: "Evaluación y Rigor Práctico",
      bg: bg-purple,
      col: purple,
      body: [
        Cada bloque incluye infografía vectorial, prácticas reproducibles (`run.sh`), ejercicios cuantitativos de entrevista con solucionario completo y fuentes primarias.
      ],
    ),
  ),
)

#outline(title: "Índice del Curso 1", depth: 2, indent: 1.4em)

#pagebreak(weak: true)
#include "sections/00_tronco_comun.typ"

#pagebreak(weak: true)
= Itinerario Técnico del Curso 1 (`B1`–`B5`)
#include "sections/c1_b1.typ"
#pagebreak(weak: true)
#include "sections/c1_b2.typ"
#pagebreak(weak: true)
#include "sections/c1_b3.typ"
#pagebreak(weak: true)
#include "sections/c1_b4.typ"
#pagebreak(weak: true)
#include "sections/c1_b5.typ"
