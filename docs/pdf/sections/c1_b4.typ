#import "../theme.typ": *

= Curso 1 · Bloque 4: Inferencia Distribuida, MoE, PDD y Redes (`CHAT-B4`)

#block-banner(
  "B4",
  "Paralelismo (TP/PP/DP/CP), MoE WideEP, PDD/LMCache e Interconexiones NCCL",
  "Oferta 1 (NCCL, interconexiones, escalabilidad) y Oferta 2 (PDD, WideEP, LMCache, routing)",
  "~4 semanas · 20–24 sprints de 20m",
)[
  Escala modelos de cientos de miles de millones de parámetros y largas ventanas de contexto sobre múltiples GPUs y nodos, diagnosticando cuellos de botella de red (`PCIe` vs `NVLink` vs `IB/RoCE`) y desbalanceos en Mixture-of-Experts.
]

#infografia(
  "Jerarquía de Interconexiones, Colectivas NCCL y Arquitectura PDD (Prefill-Decode Disaggregation)",
  "Izquierda: comparación de ancho de banda de interconexiones. Derecha: arquitectura desagregada Prefill->Decode con transferencia KV-Cache vía NIXL/LMCache y routing cache-aware.",
  height: 190pt,
)[
  // Izquierda: Barras de ancho de banda
  #place(dx: 8pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[Jerarquía de Interconexiones (Ancho de Banda Unidir./Bidir.)])
  #hbar(8pt, 20pt, "NVLink 5 (Blackwell)", "~900 GB/s bidir.", 1.0, max-w: 105pt, h: 11pt, col: teal)
  #hbar(8pt, 36pt, "NVLink 4 (H100)", "~450 GB/s unidir.", 0.5, max-w: 105pt, h: 11pt, col: navy)
  #hbar(8pt, 52pt, "PCIe Gen5 x16", "~64 GB/s dir.", 64.0 / 900.0, max-w: 105pt, h: 11pt, col: blue)
  #hbar(8pt, 68pt, "IB NDR / RoCEv2 (400G)", "50 GB/s por NIC", 50.0 / 900.0, max-w: 105pt, h: 11pt, col: amber)
  #hbar(8pt, 84pt, "PCIe Gen4 x16", "~32 GB/s dir.", 32.0 / 900.0, max-w: 105pt, h: 11pt, col: crimson)

  #place(dx: 8pt, dy: 106pt, box(width: 242pt, fill: bg-light, stroke: 0.4pt + border-gray, inset: 5pt, radius: 3pt)[
    #set text(size: 7.3pt, fill: slate)
    *Fórmula `nccl-tests` AllReduce ($N$ GPUs):*
    $ "busbw" = "algbw" times frac(2(N - 1), N) $
    Mide eficiencia física del enlace: si en `2x–8x H100` cae de ~370 GB/s a ~25 GB/s, el tráfico ha caído a PCIe/SHM (*P2P degradado*).
  ])

  // Derecha: PDD + WideEP
  #place(dx: 262pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[PDD & Cache-Aware Routing (B4.3)])
  #diag-node(262pt, 20pt, 206pt, 26pt, "Smart Router (KV-Cache Aware)", sub: "Alinea prefijo caliente sin saturar cola", fill: bg-purple, col: purple)
  #arrow-d(365pt, 46pt, 62pt, col: purple)
  #diag-node(262pt, 62pt, 96pt, 48pt, "Pool Prefill (P)", sub: "Compute-bound\nGenera KV inicial\n(ej. ratio 1P : 3D)", fill: bg-amber, col: amber)
  #arrow-r(358pt, 86pt, 376pt, col: teal)
  #diag-node(376pt, 62pt, 92pt, 48pt, "Pool Decode (D)", sub: "BW-bound\nRecibe KV vía\nNIXL / RDMA", fill: bg-teal, col: teal)
  #diag-node(262pt, 120pt, 206pt, 56pt, "WideEP en MoE (DP Atención + EP Expertos)", sub: "Atención en DP (no replica KV-Cache)\nExpertos en EP ancho -> All-to-All (Dispatch/Combine)\nOcultar latencia con Two-Batch Overlap (DeepEP)", fill: bg-navy, col: navy)
]

== Unidad B4.1 · Paralelismo Multi-GPU (TP, PP, DP, CP) y Colectivas NCCL

#unit-header(
  "B4.1",
  "Paralelismo 3D/4D y Primitivas Colectivas NCCL",
  "[MOB] 4s",
  "B1.3, B2.1",
  "B4.2, B4.3, B4.4",
  "Móvil (100% lectura técnica)",
  "Dibujar qué colectiva NCCL dispara TP vs DP vs EP y por qué TP no cruza de nodo.",
)

#concepto-clave[
  1. *Tensor Parallelism (TP):* Particiona las matrices MLP y cabezas de atención en columnas/filas dentro de cada capa. Exige *2 `AllReduce` bloqueantes por capa Transformer* (en atención y en MLP) dentro de la ruta crítica de cada token. Por su sensibilidad extrema a latencia (decenas de microsegundos), TP rara vez supera un único nodo de 8 GPUs conectadas por NVLink/NVSwitch.
  2. *Pipeline Parallelism (PP):* Asigna bloques contiguos de capas por GPU/nodo comunicándose punto-a-punto (`Send`/`Recv`). Reduce drásticamente el tráfico de red respecto a TP, pero introduce *pipeline bubbles* y *no reduce la latencia de un único token* (un usuario sigue esperando todas las etapas secuenciales).
  3. *Data Parallelism (DP) y Context Parallelism (CP):* DP escala throughput linealmente replicando pesos con un router delante; CP fragmenta la secuencia a lo largo de la dimensión de tokens (*Ring Attention*) para contextos $> 128"k"$ donde un solo nodo no puede alojar el KV-Cache ni activaciones.
  4. *Colectivas NCCL:* `AllReduce` (suma distribuida = `ReduceScatter` + `AllGather`, volumen $2 frac(N-1, N) dot M$), `AllGather` (reconstruir tensor particionado), `ReduceScatter` (reducir y repartir trozos) y `All-to-All` (enrutamiento personalizado de tokens hacia expertos en MoE).
]

== Unidad B4.2 · Mixture-of-Experts (MoE) y Wide Expert Parallelism (WideEP)

#unit-header(
  "B4.2",
  "Anatomía MoE, WideEP y Expert Load Imbalance",
  "[MOB] 4s",
  "B4.1",
  "B4.4",
  "Móvil (100% lectura técnica)",
  "Diagnosticar por qué un clúster MoE tiene buen p50 pero mal p99 (imbalance + All-to-All).",
)

#concepto-clave[
  1. *Capacidad vs Cómputo Activo:* Modelos como DeepSeek-V3 ($671"B"$ totales, $37"B"$ activos por token) o Qwen-MoE necesitan alojar cientos de expertos en VRAM aunque cada token solo active $K=8$.
  2. *WideEP (DP Atención + EP Ancho):* Aplicar TP=32 destruiría la eficiencia replicando pesos/KV innecesariamente. *WideEP* usa *Data Parallelism* en las capas de atención (evitando replicar KV-Cache) y *Expert Parallelism* ancho (p. ej. 32–72 GPUs) distribuyendo expertos, comunicados mediante `All-to-All` (*Dispatch* de tokens y *Combine* de resultados). Se solapa comunicación y cómputo dividiendo el batch en dos micro-lotes (*Two-Batch Overlap* / *DeepEP*).
  3. *Expert Load Imbalance (El enemigo del p99):* Si un tópico o formato (p. ej. código Python o JSON) dispara la ruta hacia 2 expertos "calientes" en una GPU concreta, esa GPU tarda mucho más y bloquea la barrera síncrona `All-to-All` de todo el clúster, arruinando el p99. Mitigación: replicación dinámica de expertos calientes (*EPLB*, *Expert Parallelism Load Balancer*).
]

== Unidad B4.3 · Prefill–Decode Disaggregation (PDD), LMCache y Cache-Aware Routing

#unit-header(
  "B4.3",
  "Desagregación Prefill-Decode y Transferencia KV-Cache",
  "[MOB] 3s + [PREP] 2s + [CLD] 1.5s",
  "B2.3, B4.1",
  "B4.4",
  "Cloud-4 parte 1: Nebius 2x H100 (o 2x L40S) (~1.5h)",
  "Evaluar cuándo PDD gana en cola p99 frente a chunked prefill colocado.",
)

#concepto-clave[
  - Mezclar *prefill* (compute-bound, tarda cientos de ms) con *decode* (memory-bound, SLO 20 ms/token) en la misma GPU causa *interferencia* y picos de ITL p99. *Chunked prefill* trocea el prefill pero sigue compartiendo contención en cada paso.
  - *PDD* separa físicamente el *pool Prefill* del *pool Decode*, transfiriendo el KV-Cache calculado mediante RDMA/NVLink (*NIXL*, *LMCache*) y ajustando el *ratio de instancias P:D* según el perfil de tráfico (cargas RAG/agénticas de entrada larga y salida corta exigen `2P:1D` o `1P:1D`; generación larga exige `1P:3D`).
]

#practica-box(title: "Práctica B4.3 ([PREP] -> [CLD] Cloud-4) · PDD 2x GPU vs DP=2 Colocated")[
  - Desplegar en `2x H100` un setup PDD (`GPU 0 = Prefill`, `GPU 1 = Decode` con transferencia KV) frente a `2 réplicas DP=2` con *chunked prefill*, inyectando tráfico mixto (prompts de 8k tokens + generación corta) y comparando TTFT p99 e ITL p99.
]

== Unidad B4.4 · Interconexiones (PCIe, NVLink, IB/RoCE) y `nccl-tests`

#unit-header(
  "B4.4",
  "Interconexiones Físicas, Topología y Validación con nccl-tests",
  "[MOB] 2s + [PREP] 1s + [CLD] 1.5s",
  "B4.1, B4.2",
  "B5.2, Proyecto B",
  "Cloud-4 parte 2: Nebius 2x–4x H100 (~1h)",
  "Detectar degradación a PCIe/SHM leyendo la salida de all_reduce_perf.",
)

#concepto-clave[
  - `nvidia-smi topo -m` revela la matriz física (`NV#` = NVLink, `PIX`/`PXB`/`SYS` = PCIe/NUMA, `NET` = GPUDirect RDMA).
  - *algbw vs busbw:* En `nccl-tests`, `algbw = tamaño_datos / tiempo`. Para reflejar el tráfico real que circula por los enlaces físicos en un anillo `AllReduce` de $N$ GPUs (donde cada GPU envía/recibe $2(N-1)/N$ del buffer), se usa *busbw*:
    $ "busbw" = "algbw" times frac(2(N - 1), N) $
  - Forzar `NCCL_P2P_DISABLE=1` simula una avería de enlace P2P/NVLink cayendo a memoria compartida de host/PCIe, sirviendo como prueba de calibración para el validador del Proyecto B.
]

#practica-box(title: "Práctica B4.4 ([PREP] -> [CLD] Cloud-4) · Runner nccl-tests")[
  - Script que captura `nvidia-smi topo -m`, ejecuta `all_reduce_perf` y `alltoall_perf` (de `8 MB` a `2 GB`) con NVLink activo vs `NCCL_P2P_DISABLE=1`, parsea `busbw` en GB/s y exporta JSON para el módulo de aceptación de Proyecto B.
]

#ejercicio-box[
  *Ejercicio B4.4-A:* En un nodo de $N=8$ H100 ejecuta `all_reduce_perf` con buffer de $1 "GB"$ y reporta `time = 4.75 ms`, `algbw = 210.5 GB/s`. Calcula el `busbw` real y determina si el nodo opera por NVLink 4 o si está degradado a PCIe Gen5.
]

#solucion-box[
  - *Sol. B4.4-A:* $"busbw" = 210.5 times frac(2(8 - 1), 8) = 210.5 times 1.75 = 368.4 "GB/s"$. Como supera ampliamente el tope de PCIe Gen5 ($\le 64 "GB/s"$) y alcanza ~82% de los $450 "GB/s"$ de NVLink 4 unidireccional, el tejido NVLink/NVSwitch está sano.
]

#fuentes-box[
  1. Hugging Face — *The Ultra-Scale Playbook (DP, TP, PP, CP & Collectives)*: `https://huggingface.co/spaces/nanotron/ultrascale-playbook`
  2. Google DeepMind — *How To Scale Your Model, Parts 2 & 3*: `https://jax-ml.github.io/scaling-book/`
  3. DeepSeek-V3 Technical Report (Section 3: Infrastructures): `https://arxiv.org/abs/2412.19437`
  4. LMCache & NVIDIA Dynamo Disaggregated Serving: `https://github.com/LMCache/LMCache` · `https://github.com/ai-dynamo/dynamo`
  5. `NVIDIA/nccl-tests` & `PERFORMANCE.md`: `https://github.com/NVIDIA/nccl-tests`
]
