#import "../theme.typ": *

= Curso 1 · Bloque 5: Benchmarking Científico y Cluster Acceptance (`CHAT-B5`)

#block-banner(
  "B5",
  "Metodología Científica, Cluster Acceptance (DCGM/Xid), Megatron MFU y K8s GPU",
  "Oferta 1 (Core: Cluster Acceptance Testing, Megatron-LM, DCGM, dashboards, regresiones)",
  "~3 semanas · 15–18 sprints de 20m",
)[
  Construye el *Proyecto B (`gpu-cluster-acceptance-profiler`)*: un harness estadístico riguroso que audita la salud de nodos multi-GPU (`dcgmi`, errores `Xid`, `nccl-tests`, GEMM burn-in), mide MFU de entrenamiento con Megatron-LM y se despliega en Kubernetes con DCGM/Prometheus/Grafana.
]

#infografia(
  "Pipeline de Cluster Acceptance Testing, MFU y Stack de Observabilidad K8s (Proyecto B)",
  "Flujo automatizado de validación de un nodo GPU recién aprovisionado antes de marcarlo como schedulable en Kubernetes / Slurm.",
  height: 185pt,
)[
  // Pipeline superior de Acceptance
  #place(dx: 8pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[Suite Automatizada de Cluster Acceptance (`PASS / FAIL`)])
  #diag-node(8pt, 18pt, 96pt, 46pt, "1. Inventario & ECC", sub: "Driver · CUDA · Topo\nErrores Xid dmesg\nContadores SRAM/DRAM", fill: bg-light, col: slate)
  #arrow-r(104pt, 41pt, 118pt, col: slate)
  #diag-node(118pt, 18pt, 100pt, 46pt, "2. dcgmi diag -r 3", sub: "Estrés térmico/power\nPCIe & NVLink bandwidth\nThrottle reasons", fill: bg-navy, col: navy)
  #arrow-r(218pt, 41pt, 232pt, col: navy)
  #diag-node(232pt, 18pt, 106pt, 46pt, "3. nccl-tests + Burn", sub: "all_reduce busbw\nalltoall + GEMM FP16/FP8\nSin thermal throttling", fill: bg-teal, col: teal)
  #arrow-r(338pt, 41pt, 352pt, col: teal)
  #diag-node(352pt, 18pt, 114pt, 46pt, "4. Veredicto & MFU", sub: "Regresión CI (<5%)\nMegatron MFU (45-55%)\nNode taint -> PASS/FAIL", fill: bg-purple, col: purple)

  // Separador
  #place(dx: 8pt, dy: 74pt, line(length: 458pt, stroke: 0.5pt + border-gray))

  // Parte inferior: Orquestación & Observabilidad
  #place(dx: 8pt, dy: 80pt, text(fill: navy, weight: "bold", size: 8pt)[Orquestación K8s, Aislamiento y Cold-Start Engineering (B5.4)])
  #diag-node(8pt, 96pt, 146pt, 78pt, "Particionado GPU en K8s", sub: "MIG (Multi-Instance GPU):\nAislamiento físico SM + VRAM + L2\n(Cero noisy neighbor)\nvs Time-Slicing:\nCompartición lógica temporal (sin SLA)", fill: bg-navy, col: navy)
  #diag-node(162pt, 96pt, 150pt, 78pt, "Observabilidad Real DCGM", sub: "No usar nvidia-smi GPU-Util!\nUsar DCGM_FI_PROF_*:\n- PIPE_TENSOR_ACTIVE\n- DRAM_ACTIVE & SM_OCCUPANCY\nExporter -> Prometheus -> Grafana", fill: bg-amber, col: amber)
  #diag-node(320pt, 96pt, 148pt, 78pt, "Multi-Nodo & Cold Starts", sub: "LeaderWorkerSet (LWS) / Slurm\nCold Starts rápidos:\n- safetensors streaming desde S3\n- Caché NVMe local\n- Lazy image (eStargz / SOCI)", fill: bg-teal, col: teal)
]

== Unidad B5.1 · Metodología de Benchmarking Reproducible (Harness Científico)

#unit-header(
  "B5.1",
  "Metodología de Benchmarking y Control de Regresiones",
  "[MOB] 2s + [LOC] 2s",
  "B1.3, B2.3",
  "B5.2, Proyecto B",
  "Móvil + CPU local",
  "Esqueleto CLI del Proyecto B testeado en local con aserciones estadísticas PASS/FAIL.",
)

#concepto-clave[
  1. *Higiene Experimental:* Todo benchmark serio impone *warmup* obligatorio (compilación JIT, asignación de caché allocator, estabilización térmica/frecuencias), semillas deterministas y bloqueo/monitorización de relojes GPU (`nvidia-smi -q -d CLOCK,PERFORMANCE`) para evitar confundir *thermal throttling* con un cambio algorítmico.
  2. *Coordinated Omission:* En pruebas de latencia de servidores, si el cliente síncrono espera a recibir respuesta antes de enviar la siguiente petición (bucle cerrado), cuando el servidor se congela 2 segundos el cliente deja de enviar carga y *oculta el desastre en los percentiles*. Para medir SLOs reales bajo carga abierta se usa un patrón de llegadas *Poisson* (open-loop) con timestamps ideales programados de antemano.
  3. *Gates Estadísticos:* Comparar medias oculta colas largas; se audita p95/p99 y se definen aserciones automáticas en CI (p. ej. fallo si p99 sube $> 10\%$ o *goodput* cae $> 5\%$).
]

#practica-box(title: "Práctica B5.1 ([LOC], inicio de gpu-cluster-acceptance-profiler)")[
  - Definir el esquema JSON estándar del Proyecto B y su motor de aserciones (`assert_regression.py`) verificado con tests unitarios locales sobre trazas sintéticas controladas antes de subir a Cloud-5.
]

== Unidad B5.2 · Cluster Acceptance Testing y Salud de Hardware (`dcgmi` & Xid)

#unit-header(
  "B5.2",
  "Diagnósticos Activos DCGM, Errores Xid y Burn-In",
  "[MOB] 2s + [PREP] 2s + [CLD] 1.5s",
  "B4.4, B5.1",
  "B5.4, Proyecto B",
  "Cloud-5 parte 1: Nebius 2x–8x H100 (~1.5h)",
  "Validador real de nodos GPU emitiendo veredicto PASS/FAIL tras auditar hardware nuevo.",
)

#concepto-clave[
  1. *Por qué `nvidia-smi` "GPU-Util" miente:* Solo indica el porcentaje de tiempo durante el último intervalo en que *al menos un kernel* estuvo activo en la GPU, aunque use un único SM al 2% de capacidad. Las métricas reales provienen de contadores hardware de *NVIDIA DCGM*: `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, `DCGM_FI_PROF_DRAM_ACTIVE` y `DCGM_FI_PROF_SM_OCCUPANCY`.
  2. *Errores Xid Críticos del Kernel (`dmesg`):*
     - *Xid 31:* GPU Memory Page Fault (MMU; bug de aplicación o corrupción de punteros).
     - *Xid 48:* Double Bit Error (DBE) en memoria ECC (fallo físico de HBM/VRAM, requiere retirada/RMA).
     - *Xid 79:* GPU fallen off the bus (caída de enlace PCIe/alimentación o fallo grave de placa).
     - *Xid 94 / 95:* Contained / Uncontained ECC Error (aislamiento dinámico de página o reinicio obligatorio).
  3. *Diagnóstico Activo (`dcgmi diag -r 3`):* Ejecuta pruebas de memoria, PCIe bandwidth, NVLink, estrés térmico y consumo eléctrico para descartar nodos degradados antes de meterlos en producción.
]

#practica-box(title: "Práctica B5.2 ([PREP] -> [CLD] Cloud-5) · Suite de Acceptance Testing")[
  - Integrar en la CLI del Proyecto B las 4 fases: (1) driver/CUDA/ECC/Xid check, (2) `dcgmi diag -r 2/3`, (3) `nccl-tests` (`all_reduce_perf` + `alltoall_perf`), (4) GEMM *burn-in* de potencia/temperatura, emitiendo reporte Markdown/JSON y código de salida `0 (PASS)` / `1 (FAIL)`.
]

== Unidad B5.3 · Benchmarking de Entrenamiento: Megatron-LM y MFU

#unit-header(
  "B5.3",
  "Megatron-LM, Paralelismo 3D y Model FLOPs Utilization (MFU)",
  "[MOB] 2s + [PREP] 1s + [CLD] 1.5s",
  "B4.1, B5.2",
  "Cobertura total Oferta 1",
  "Cloud-5 parte 2: Nebius 2x–8x H100 (~1h)",
  "Interpretar un MFU del 45-55% en H100 y diagnosticar con nsys comunicación vs data loading.",
)

#concepto-clave[
  1. *Cálculo de MFU (*Model FLOPs Utilization*):* Mide la fracción de la potencia teórica pico del hardware aprovechada en FLOPs *útiles* del modelo (aproximadamente $2 N$ en *forward* + $4 N$ en *backward* = $6 N_"params"$ por token, *sin sumar* el overhead de *activation checkpointing* / recomputación, que sí cuenta en HFU *Hardware FLOPs Utilization*):
     $ "MFU" = frac(6 dot N_"params" dot "tokens/s", "Peak FLOPs"_"hardware") $
  2. *Megatron-LM:* Combina TP intra-nodo, PP inter-nodo y DP (o ZeRO/FSDP), con *FlashAttention*, *Activation Checkpointing* selectivo y FP8 vía *Transformer Engine*. Un MFU saludable en clústeres H100 ronda el 45%–55%; cifras menores suelen deberse a comunicaciones `AllReduce`/`AllGather` sin solapar o a cuellos de I/O en *DataLoader* visibles en `nsys`.
]

#practica-box(title: "Práctica B5.3 ([PREP] -> [CLD] Cloud-5) · Mini-Megatron con nsys y MFU")[
  - Ejecutar 20 iteraciones con datos sintéticos sobre contenedor NGC PyTorch (`TP=2` o `torchtitan`), capturar `nsys` en las iteraciones 10–12 y calcular automáticamente MFU y TFLOPS/GPU efectivos en el módulo de entrenamiento del Proyecto B.
]

== Unidad B5.4 · Orquestación K8s, Observabilidad DCGM y Cold Starts

#unit-header(
  "B5.4",
  "GPU Operator, MIG vs Time-Slicing, LWS, DCGM y Cold Starts",
  "[MOB] 2s + [LOC]/[PREP] 3s + [CLD] opcional",
  "B5.2",
  "Proyecto B Completo",
  "Local (Kind/Docker Compose) + opcional Nebius K8s",
  "Proyecto B cerrado con manifests K8s, Job de Acceptance y dashboards DCGM+vLLM.",
)

#concepto-clave[
  1. *Aislamiento GPU:* *MIG* (Hopper/Ampere) particiona físicamente SMs, caché L2 y controladores HBM en instancias aisladas (ideal para cargas predecibles multi-tenant); *Time-Slicing* solo multiplexa tiempo de GPU sin aislamiento de memoria ni SLA.
  2. *Orquestación Multi-Nodo:* *LeaderWorkerSet (LWS)* gestiona grupos acoplados líder+trabajadores en Kubernetes (imprescindible para inferencia distribuida PP/WideEP), complementado con *Slurm* (`sbatch`, GRES, Pyxis/Enroot) en entornos HPC.
  3. *Ingeniería de Cold Starts:* Los pesos de decenas de GB dominan el arranque; se optimizan con `safetensors` + streaming directo desde S3/R2, caché NVMe local en nodo, *lazy image loading* (eStargz / SOCI) o *checkpoint/restore* (`ai-dynamo/snapshot`).
]

#ejercicio-box[
  *Ejercicio B5.3-A (MFU):* Entrenas un modelo denso de $70 times 10^9$ parámetros sobre un nodo de `8x H100 SXM` (pico BF16 = $989 "TFLOP/s"$ por GPU) logrando un throughput global de `9,000 tokens/s` sostenidos. Calcula el MFU del nodo.

  *Ejercicio B5.2-B (Diagnóstico Xid):* Tras un job de *burn-in*, `dmesg` muestra `NVRM: Xid (PCI:0000:18:00): 48, DBE` en la GPU 3 y `nvidia-smi` reporta 2 páginas pendientes de retiro (*Pending Page Blacklist*). ¿Qué acción automatizada debe tomar el Job de Cluster Acceptance en Kubernetes?
]

#solucion-box[
  - *Sol. B5.3-A:* FLOPs útiles por segundo = $6 times (70 times 10^9) times 9000 = 3.78 times 10^(15) "FLOP/s" = 3780 "TFLOP/s"$. Pico total del nodo = $8 times 989 = 7912 "TFLOP/s"$. $"MFU" = 3780 / 7912 = 47.8\%$, dentro del rango saludable (45%–55%) en H100. *(Sanity check: si una herramienta reportase MFU > 100%, indica error en el conteo de tokens o de parámetros activos en MoE).*
  - *Sol. B5.2-B:* Un `Xid 48 (DBE)` indica corrupción física incorregible de dos bits en HBM. El Job debe emitir veredicto `FAIL`, aplicar un `taint` `gpu-health=failed:NoSchedule` al nodo en Kubernetes, drenar cargas si las hubiera y abrir ticket automático de reinicio de GPU/reemplazo hardware en el proveedor.
]

#fuentes-box[
  1. NVIDIA DCGM User Guide & Xid Errors Reference: `https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/` · `https://docs.nvidia.com/deploy/xid-errors/`
  2. NVIDIA Megatron-LM & Ultra-Scale Playbook (MFU): `https://github.com/NVIDIA/Megatron-LM` · `https://huggingface.co/spaces/nanotron/ultrascale-playbook`
  3. NVIDIA GPU Operator & Kubernetes `LeaderWorkerSet` (LWS): `https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/` · `https://github.com/kubernetes-sigs/lws`
  4. NVIDIA `ai-dynamo/aiperf`, `modelexpress` & `snapshot`: `https://github.com/ai-dynamo`
]
