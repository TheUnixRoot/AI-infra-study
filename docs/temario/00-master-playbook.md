# AI Infra — Master Playbook & Sistema Operativo de Estudio

> **Propósito de este documento:** Fuente de verdad única (autorreferenciada) para el cuaderno **AI infra**. Diseñado para pivotar desde DevOps + agentes/vLLM hacia roles de **GPU Platform Benchmarking (Oferta 1)** y **Senior MLE Inference / Token Factory (Oferta 2)** en proveedores GPU como Nebius.
>
> **Restricción operativa (Modo Bebé):** Cero sesiones maratonianas. Todo el plan se ejecuta en **micro-sprints de 20 minutos** cada 1–2 horas: lectura en texto desde el móvil + ejecución delegada a un **Agente de Código** contra GPU local o ráfagas cortas y preparadas en **Nebius / Scaleway**.

---

## 0. Sistema Operativo del Cuaderno (Reglas de Higiene, Handover y Fork)

Para evitar chats infinitos que degraden el contexto del LLM y pierdan trazabilidad, este cuaderno opera bajo **4 reglas estrictas**:

1. **Regla 1-a-1 (Máximo 5 chats):** Nunca existirá más de **1 chat activo por bloque** (`CHAT-B1` a `CHAT-B5`). Si un chat de bloque se alarga demasiado, se cierra mediante **Handover** y se hace **Fork** (nuevo chat del mismo bloque borrando o archivando el anterior).
2. **Separación de responsabilidades:**
   - **Este Cuaderno (LLM):** Tutor socrático, diseño de experimentos, diagnóstico de trazas (`nsys`, `ncu`, logs de vLLM/NCCL), revisión de conceptos y generación de *prompts* de especificación para tu agente de código.
   - **Tu Agente de Código (CLI/IDE):** Escribe código boilerplate, Dockerfiles, scripts de benchmark, Terraform/scripts de aprovisionamiento y ejecuta pruebas contra tu GTX 1050 o instancias SSH en Nebius/Scaleway.
3. **Clasificación de Micro-Sprints (20 min):**
   - `[MOB]` **Móvil (Solo lectura/texto):** Cero hardware. Lectura de blogs técnicos, glosarios o papers + 2-3 preguntas de comprobación en el chat del bloque.
   - `[LOC]` **Local (GTX 1050 / CPU):** Compilación CUDA C++, `nsys`, Linux `perf`, PyTorch FP32 en tensores pequeños o simuladores offline (`aisim`).
   - `[PREP]` **Pre-Flight Cloud (0 EUR):** Con tu agente de código, preparas el script reproducible, `Dockerfile` y comandos exactos *antes* de alquilar la GPU. **Prohibido alquilar una GPU sin haber completado su `[PREP]`.**
   - `[CLD]` **Ráfaga Cloud (Nebius / Scaleway):** Alquiler por horas (objetivo: 1–2 micro-sprints consecutivos o ejecución desatendida con *auto-shutdown*). Arrancas, ejecutas el script de `[PREP]`, descargas trazas/CSVs/JSONs y apagas la VM.
4. **Regla de Auto-Kill en Cloud:** Todo script de aprovisionamiento en Nebius o Scaleway debe incluir un temporizador de apagado automático al inicio de la sesión:
   ```bash
   # Ejemplo obligatorio al entrar en cualquier VM GPU alquilada
   sudo shutdown -h +90
   ```

### Plantillas de Control (Copiar y Pegar)

#### A. Prompt de Inicio / Fork de Chat de Bloque
```text
[FORK CHAT-Bx]
Contexto: Lee `ai_infra_master_playbook.md` en las fuentes.
Bloque activo: [Bx] | Unidad actual: [Bx.y]
Estado previo (Handover): <pegar último bloque de handover o "Inicio de bloque">
Tiempo disponible ahora: 20 minutos | Modo: [MOB / LOC / PREP / CLD]
Objetivo de este micro-sprint: <ej. entender Roofline / generar script Pre-Flight para el agente>
Regla: Respuestas concisas, legibles en móvil, sin relleno. Al final dame el bloque de Handover actualizado.
```

#### B. Bloque de Handover (Al cerrar cualquier micro-sesión de 20 min)
```yaml
HANDOVER:
  unidad_actual: "B1.2"
  estado: "en_progreso" # pendiente | en_progreso | done
  aprendido_o_verificado:
    - "Diferencia entre memory-bound (decode) y compute-bound (prefill)"
  artefacto_generado: "repo/b1/tiled_gemm.cu (compila en sm_61)"
  bloqueos_o_dudas: "Pendiente interpretar stall de memoria global en nsys"
  siguiente_micro_sprint_20m: "B1.2 [LOC] - Ejecutar nsys sobre naive vs tiled GEMM"
```

#### C. Prompt de Delegación para tu Agente de Código (Pre-Flight)
```text
Actúa como ingeniero de sistemas GPU. Vamos a preparar un experimento reproducible para [Bx.y].
Hardware destino: [GTX 1050 sm_61 2GB | Nebius/Scaleway 1x L40S | 1x H100 | 2x H100].
Requisitos:
1. Un único script de entrada `run.sh` idempotente (instala/verifica deps, corre benchmark, guarda métricas en JSON/CSV y trazas en `./artifacts`).
2. Smoke-test rápido (<30s) para validar antes de la ejecución real.
3. Manejo de errores limpio y resumen por stdout listo para pegar en mi cuaderno de estudio.
```

---

## 1. Mapa de Dependencias y Matriz de Infraestructura

### 1.1 Grafo de Dependencias entre Bloques y Proyectos

```text
[B1.1 Roofline & PyTorch] ──► [B1.2 CUDA C++ Local] ──► [B1.3 Profiling nsys/perf]
         │                                                        │
         │                                                        ▼
         ├──────────────────────────────────────────────► [B1.4 Triton + ncu (Cloud)]
         │                                                        │
         ▼                                                        ▼
[B2.1 KV-Cache Math] ──► [B2.2 vLLM vs SGLang] ──► [B2.3 TRT-LLM & Dynamo] ──► [B2.4 Agentic Serving]
                                 │                                                     │
                                 ▼                                                     ▼
                    [B3.1 Quantization Theory] ──► [B3.2 FP8 & INT4/AWQ]       (Hito 1: Proyecto A)
                                                          │
                                                          ▼
                                             [B3.3 Speculative Decoding] ──► [B3.4 Evals & Pareto]
                                                                                       │
                                                                                       ▼
                                                                               (Proyecto A Completo)

[B1.3] + [B2.1] ──► [B4.1 Paralelismo TP/PP/DP] ──► [B4.2 MoE & WideEP] ──► [B4.3 PDD & LMCache]
                                                                                       │
                                                                                       ▼
                                                                           [B4.4 NCCL & Interconnects]
                                                                                       │
                                                                                       ▼
[B1.3] + [B4.4] ──► [B5.1 Harness Científico] ──► [B5.2 Cluster Acceptance] ──► [B5.3 Megatron & MFU]
                                                                                       │
                                                                                       ▼
                                                                           [B5.4 K8s + DCGM + CI]
                                                                                       │
                                                                                       ▼
                                                                               (Proyecto B Completo)
```

> **Nota sobre paralelización:** Si una semana no puedes alquilar GPU (`[CLD]`), puedes avanzar sin bloqueo todas las unidades `[MOB]` y `[PREP]` de bloques posteriores cuyas dependencias teóricas ya estén listas (por ejemplo, leer `B3.1` o `B4.1` en el móvil mientras esperas el fin de semana para lanzar la ráfaga `[CLD]` de `B2.2`).

### 1.2 Fronteras de tu Hardware Local (GTX 1050 2 GB, Pascal `sm_61`)

| Capacidad | GTX 1050 (`sm_61`) | Motivo técnico / Alternativa |
| :--- | :---: | :--- |
| CUDA C++ (`nvcc`), *warps*, *shared memory*, `streams` | **SÍ** | Base idéntica a arquitecturas modernas para fundamentos. |
| PyTorch internals (strides, `autograd`, FP32 mini-redes) | **SÍ** | Usar tensores pequeños (`< 512 MB`) para evitar OOM. |
| **Nsight Systems (`nsys`)** y **Linux `perf`** | **SÍ** | Excelente para detectar cuello de botella CPU/Python/syncs. |
| **OpenAI Triton** / **Nsight Compute (`ncu`)** moderno | **NO** | Requieren Volta (`sm_70`) / Turing (`sm_75`) o superior. Alquilar 1x L4/L40S. |
| **FlashAttention / FlashInfer / SGLang / TRT-LLM** | **NO** | Requieren Ampere (`sm_80+`). Alquilar 1x L40S o H100. |
| **FP8 nativo** (Transformer Engine / FP8 GEMM) | **NO** | Requiere Ada (`sm_89`, L40S) o Hopper (`sm_90`, H100). |
| **Multi-GPU (TP, EP, PDD, NCCL NVLink/IB)** | **NO** | Alquilar nodo 2x–8x L40S o H100 en Nebius/Scaleway. |

### 1.3 Matriz de Dependencias de Alquiler de Infraestructura (Nebius / Scaleway)

| Ráfaga Cloud | Unidades cubiertas | SKU Recomendada | Proveedor sugerido | Tiempo estimado | Prerrequisito `[PREP]` obligatorio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cloud-1** | `B1.4` | 1x L4 o 1x L40S | Scaleway (L4/L40S) | 1.5 – 2 h | Scripts Triton + comandos `ncu` listos en git. |
| **Cloud-2** | `B2.2`, `B2.3`, `B2.4` | 1x L40S (48 GB) o 1x H100 (80 GB) | Scaleway / Nebius | 3 – 4 h (en 2 sesiones) | `docker-compose` / scripts vLLM + SGLang + cliente carga agéntica. |
| **Cloud-3** | `B3.2`, `B3.3`, `B3.4` | 1x H100 (80 GB, FP8 nativo) | Nebius / Scaleway | 3 – 4 h (en 2 sesiones) | Scripts calibración FP8/AWQ + `lm-eval` + draft model probados en seco. |
| **Cloud-4** | `B4.3`, `B4.4` | 2x–4x H100 (NVLink) | Nebius | 2 – 3 h | Scripts `nccl-tests`, config PDD (LMCache/vLLM) y MoE TP/EP. |
| **Cloud-5** | `B5.2`, `B5.3`, `B5.4` | 2x–8x H100 (1 nodo, opcional IB) | Nebius | 2 – 3 h | Suite CLI de Proyecto B + manifests DCGM/Prometheus testeados en local. |

> **Ahorro de coste:** En Scaleway/Nebius, guarda los pesos de Hugging Face en un volumen persistente temporal durante la semana de la ráfaga o usa modelos de 7B–14B (ej. `Qwen2.5-Coder-7B-Instruct`) para iterar rápido; sube a modelos MoE más grandes (ej. `Qwen3-30B-A3B` o `Mixtral-8x7B`) solo en la validación final.

---

## 2. Bloque 1: Arquitectura GPU, CUDA C++ y Profiling (`CHAT-B1`)

**Alineación:** Oferta 1 (Core: profiling kernel/sistema) y Oferta 2 (Nice-to-have: CUDA/Triton, cuello de botella).
**Duración estimada:** ~3 semanas (~18–24 micro-sprints de 20 min).

### `[B1.1]` Modelo Mental de Hardware, Roofline y PyTorch Internals
- **Tipo:** `[MOB]` (4 sprints) + `[LOC]` (2 sprints)
- **Depende de:** Ninguna | **Desbloquea:** `B1.2`, `B1.3`, `B2.1`
- **Infraestructura:** Móvil + GTX 1050 (CPU/GPU)
- **Conceptos clave:**
  - Intensidad aritmética:
    $$I = \frac{\text{FLOPs}}{\text{Bytes transferidos de VRAM}}$$
  - Por qué *Prefill* es *compute-bound* y *Decode* (batch pequeño) es *memory-bandwidth bound*.
  - Representación interna de tensores en PyTorch (`storage`, `offset`, `shape`, `stride`), coste de `.contiguous()`, y fases de `torch.compile` (TorchDynamo -> AOTAutograd -> TorchInductor).
- **Recursos gratuitos en texto (móvil):**
  1. Horace He — *Making Deep Learning Go Brrrr From First Principles*: `https://horace.io/brrr_intro.html` (Leer en 2 sprints de 20m: Compute vs. Memory Bandwidth vs. Overhead).
  2. Modal — *GPU Glossary* (sección *Device Hardware* y *Memory*): `https://modal.com/gpu-glossary` (Lectura modular en fichas de 5 min).
  3. Edward Z. Yang — *PyTorch Internals*: `http://blog.ezyang.com/2019/05/pytorch-internals/` (Solo sección de Tensor/Strides).
- **Micro-práctica (Agente -> GTX 1050):** Pide al agente un script Python que cree una matriz FP32 de $2048 \times 2048$, mida el tiempo de `.transpose(0, 1)` vs `.transpose(0, 1).contiguous()` y calcule el ancho de banda efectivo en GB/s frente al teórico de tu GTX 1050 ($\approx 112\text{ GB/s}$).
- **Done:** Puedes explicar sin mirar notas por qué aumentar el *batch size* en decode sube la intensidad aritmética sin aumentar linealmente la latencia hasta llegar al punto de inflexión del Roofline.

### `[B1.2]` Fundamentos de CUDA C++ en Local (`sm_61`)
- **Tipo:** `[MOB]` (3 sprints) + `[LOC]` (4 sprints)
- **Depende de:** `B1.1` | **Desbloquea:** `B1.3`, `B1.4`
- **Infraestructura:** Móvil + GTX 1050 (`nvcc`)
- **Conceptos clave:** Jerarquía Grid / Block / Warp (32 hilos) / Thread; coalescencia de memoria global (GMEM); *Shared Memory* (SMEM) y *tiling*; *warp divergence*; reducciones intra-warp con `__shfl_down_sync`; *online softmax* en una pasada.
- **Recursos gratuitos en texto (móvil):**
  1. Simon Boehm — *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*: `https://siboehm.com/articles/22/CUDA-MMM` (Leer solo Kernels 1, 2 y 3: Naive, GMEM Coalescing y SMEM Caching; 3 sprints de 20m).
  2. NVIDIA — *An Even Easier Introduction to CUDA* (texto): `https://developer.nvidia.com/blog/even-easier-introduction-cuda/`
  3. Modal — *GPU Glossary* (sección *CUDA Programming Model*: Warps, Thread Blocks, Occupancy): `https://modal.com/gpu-glossary`
- **Micro-práctica (Agente -> GTX 1050):**
  1. Pide al agente que genere 3 versiones de GEMM FP32 en CUDA (`naive`, `coalesced`, `smem_tiled`) compilables con `-arch=sm_61` y un benchmark contra `cuBLAS` para matrices $1024 \times 1024$.
  2. Implementa (o revisa paso a paso con el agente) una reducción de suma/máximo usando `__shfl_down_sync` (base de *online softmax*).
- **Done:** Ves en tu GTX 1050 una mejora real de GFLOPs entre `naive`, `coalesced` y `smem_tiled`, y sabes explicar qué cambió en los accesos a GMEM/SMEM.

### `[B1.3]` Profiling Multinivel: `nsys` y Linux `perf`
- **Tipo:** `[MOB]` (2 sprints) + `[LOC]` (3 sprints)
- **Depende de:** `B1.1`, `B1.2` | **Desbloquea:** `B1.4`, `B4.1`, `B5.1`, Proyecto B
- **Infraestructura:** Móvil + GTX 1050 (`nsys`, `perf`)
- **Conceptos clave:** Diferenciar cuellos de botella en CPU (Python GIL, *kernel launch overhead*, sincronizaciones implícitas `.item()` / `.cpu()`) vs. GPU; usar `nvtx` para anotar rangos; leer *flamegraphs* de Linux `perf` en procesos Python/C++.
- **Recursos gratuitos en texto (móvil):**
  1. NVIDIA Nsight Systems User Guide (sección *CLI Profiling* y *CUDA Trace*): `https://docs.nvidia.com/nsight-systems/UserGuide/index.html`
  2. PyTorch Profiler & Nsight integration docs / blog de *CUDA streams & synchronization*.
- **Micro-práctica (Agente -> GTX 1050):** Pide al agente un bucle de entrenamiento/inferencia sintético en PyTorch con 3 antipatrones inyectados (un `.item()` dentro del bucle, transferencia síncrona CPU->GPU sin `pin_memory`, y muchos kernels diminutos sin fusionar). Captura la traza con `nsys profile --stats=true`, identifica los huecos muertos (*bubbles*) en la GPU y verifica la corrección.
- **Done:** Sabes generar un resumen CLI con `nsys stats` y distinguir en menos de 2 minutos si el problema es *CPU-bound* (overhead de lanzamiento) o *GPU-bound*.

### `[B1.4]` Salto a Cloud: OpenAI Triton y Nsight Compute (`ncu`)
- **Tipo:** `[MOB]` (2 sprints) + `[PREP]` (1 sprint) + `[CLD]` (1–2 sprints, **Cloud-1**)
- **Depende de:** `B1.2`, `B1.3` | **Desbloquea:** Base de kernels modernos para `B2`, `B3`
- **Infraestructura:** **Cloud-1** -> Scaleway `1x L4` o `1x L40S` (1.5–2h)
- **Conceptos clave:** Programación orientada a bloques en OpenAI Triton (`tl.program_id`, `tl.load`/`tl.store` con máscaras, *kernel fusion* de RMSNorm/Softmax); lectura de métricas de `ncu` (*Memory Throughput %*, *Compute (SM) Throughput %*, *Roofline chart* y *Warp Stall Reasons*).
- **Recursos gratuitos en texto (móvil):**
  1. OpenAI Triton Tutorials (texto oficial — leer *01 Vector Add* y *02 Fused Softmax*): `https://triton-lang.org/main/getting-started/tutorials/`
  2. NVIDIA Nsight Compute CLI Guide (sección *SpeedOfLight* y *Roofline*): `https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html`
- **Micro-práctica (Agente -> `[PREP]` -> `[CLD]`):**
  - `[PREP]`: El agente escribe un script con un kernel Triton de *Fused Softmax* y *Fused RMSNorm*, comparándolo contra PyTorch nativo y `torch.compile`, además de un comando `ncu --set roofline` que exporte el resumen a CSV/texto.
  - `[CLD]`: Arrancas `1x L4/L40S`, ejecutas `run.sh`, descargas los CSVs de `ncu` y apagas la máquina.
- **Done:** Tienes un informe comparativo (PyTorch eager vs. `torch.compile` vs. Triton custom) con métricas de `ncu` que demuestran la reducción de lecturas/escrituras a HBM.

---

## 3. Bloque 2: Motores de Inferencia, KV-Cache y Cargas Agénticas (`CHAT-B2`)

**Alineación:** Oferta 2 (Core: vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo, FlashInfer, cargas agénticas).
**Duración estimada:** ~4 semanas (~20–26 micro-sprints de 20 min).

### `[B2.1]` Aritmética del KV-Cache, Atención y Prefill vs. Decode
- **Tipo:** `[MOB]` (4 sprints) + `[LOC]` (1 sprint, calculadora Python en CPU)
- **Depende de:** `B1.1` | **Desbloquea:** `B2.2`, `B3.1`, `B4.1`, Proyecto A
- **Infraestructura:** Móvil + CPU local
- **Conceptos clave:**
  - Fórmula exacta de memoria del KV-Cache (en bytes) con GQA (*Grouped-Query Attention*):
    $$\text{KV}_{\text{bytes}} = 2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times S \times B \times \text{bytes\_per\_elem}$$
    donde $L$ es el número de capas, $n_{\text{kv}}$ cabezas KV, $d_{\text{head}}$ dimensión por cabeza, $S$ longitud de secuencia y $B$ *batch size*.
  - Diferencia entre MHA, GQA y MLA (*Multi-head Latent Attention*, estilo DeepSeek).
  - *PagedAttention*, *Continuous Batching*, *Chunked Prefill* y *Automatic Prefix Caching* (APC).
  - Métricas de SLO: **TTFT** (*Time To First Token*), **TPOT** (*Time Per Output Token*) / **ITL** (*Inter-Token Latency*) y **Goodput**.
- **Recursos gratuitos en texto (móvil):**
  1. Kipply — *Transformer Inference Arithmetic*: `https://kipp.ly/transformer-inference-arithmetic/` (Imprescindible; leer en 2 sprints).
  2. Google DeepMind — *How To Scale Your Model*, Parte 7 (*Inference*): `https://jax-ml.github.io/scaling-book/inference/` (Excelente texto con matemática limpia y diagramas claros en móvil).
  3. Blog oficial de vLLM (entradas sobre *PagedAttention*, *Automatic Prefix Caching* y *Chunked Prefill*): `https://blog.vllm.ai/`
- **Micro-práctica (Agente -> CPU local):** Pide al agente una CLI en Python (`kv_calc.py`) que lea el `config.json` de cualquier modelo de Hugging Face (Llama-3-8B, Qwen2.5-72B, DeepSeek-V3) y calcule: VRAM de pesos, VRAM del KV-Cache por petición y concurrencia máxima antes de *preemption* en L40S (48 GB) vs. H100 (80 GB).
- **Done:** Puedes calcular de cabeza (o en 30 segundos) cuántas peticiones concurrentes de 8k tokens caben en una H100 para un modelo 8B/70B en FP16 vs. FP8.

### `[B2.2]` Comparativa Práctica: vLLM vs. SGLang y FlashInfer
- **Tipo:** `[MOB]` (2 sprints) + `[PREP]` (2 sprints) + `[CLD]` (2 sprints, **Cloud-2 parte 1**)
- **Depende de:** `B2.1` | **Desbloquea:** `B2.3`, `B2.4`, `B3.2`
- **Infraestructura:** **Cloud-2** -> Scaleway/Nebius `1x L40S` o `1x H100` (~2h)
- **Conceptos clave:**
  - **vLLM (arquitectura V1):** separación de procesos CPU/GPU, *scheduler* sin bloqueos, PagedAttention + backends de atención (FlashAttention vs. **FlashInfer**).
  - **SGLang:** **RadixAttention** (árbol radix LRU para reutilización automática de KV-Cache multi-turno), *zero-overhead CPU scheduler*.
  - Por qué **FlashInfer** es el estándar para kernels de atención paginada/desigual (*ragged*).
- **Recursos gratuitos en texto (móvil):**
  1. SGLang Docs & Blog (*RadixAttention* y arquitectura del runtime): `https://docs.sglang.ai/`
  2. FlashInfer Docs / Blog (arquitectura de atención para serving): `https://docs.flashinfer.ai/`
  3. vLLM V1 Architecture Blog: `https://blog.vllm.ai/`
- **Micro-práctica (Agente -> `[PREP]` -> `[CLD]`):**
  - `[PREP]`: El agente crea un `run_compare.sh` que levanta primero **vLLM** y luego **SGLang** con `Qwen2.5-Coder-7B-Instruct` y lanza un benchmark sintético de concurrencia (1, 8, 32, 64 peticiones concurrentes) midiendo TTFT p50/p99, TPOT p50/p99 y tok/s/GPU.
  - `[CLD]`: Ejecutas en `1x L40S` o `1x H100`, guardas resultados JSON y trazas de logs de métricas (KV-cache usage %, preemption count).
- **Done:** Tienes tablas reales de latencia/throughput comparando vLLM vs. SGLang bajo la misma carga y entiendes por qué varía el *overhead* de CPU al subir la concurrencia.

### `[B2.3]` TensorRT-LLM y Arquitectura NVIDIA Dynamo (Offline + Cloud)
- **Tipo:** `[MOB]` (3 sprints) + `[LOC]` (1 sprint con `aisim` / `aiconfigurator`) + `[CLD]` (opcional dentro de Cloud-2)
- **Depende de:** `B2.2` | **Desbloquea:** `B4.3`, `B5.1`
- **Infraestructura:** Móvil + CPU local (simulador `aisim`) + opcional `1x H100`
- **Conceptos clave:**
  - **TensorRT-LLM:** Flujo moderno (PyTorch backend vs. engines compilados con `trtllm-build`), fusión de grafos, *in-flight batching* y *Triton Inference Server* como capa de serving empresarial.
  - **NVIDIA Dynamo (`ai-dynamo`):** Framework de inferencia distribuida en Rust/Python: *Smart Router* (KV-cache aware), *Disaggregated Serving*, **NIXL** (transferencia de KV-cache), **AIPerf** (benchmarking) y **AISim / AIConfigurator** (simulación offline sin gastar GPU!).
- **Recursos gratuitos en texto (móvil):**
  1. NVIDIA Dynamo GitHub README y docs de arquitectura: `https://github.com/ai-dynamo/dynamo`
  2. NVIDIA TensorRT-LLM Architecture Docs: `https://nvidia.github.io/TensorRT-LLM/`
  3. NVIDIA AIPerf & AISim repos: `https://github.com/ai-dynamo/aiperf` y `https://github.com/ai-dynamo/aisim`
- **Micro-práctica (Agente -> CPU local):** Instala/corre `ai-dynamo/aiconfigurator` o `aisim` (offline, 0 EUR en GPU) guiado por tu agente para modelar configuraciones óptimas de serving (TP, batching, SLA de TTFT/TPOT) y contrastar sus predicciones con tus números reales de `B2.2`.
- **Done:** Conoces las piezas de NVIDIA Dynamo (Router, Workers, NIXL, AIPerf) y sabes cuándo elegir TensorRT-LLM frente a vLLM/SGLang en producción.

### `[B2.4]` Optimización de Cargas Agénticas (Tu Ventaja Competitiva)
- **Tipo:** `[MOB]` (2 sprints) + `[PREP]` (2 sprints) + `[CLD]` (2 sprints, **Cloud-2 parte 2**)
- **Depende de:** `B2.2` | **Desbloquea:** **Hito 1 de Proyecto A**, `B3.3`
- **Infraestructura:** **Cloud-2** -> Scaleway/Nebius `1x L40S` o `1x H100` (~1.5–2h)
- **Conceptos clave:**
  - Perfil de tráfico agéntico: *system prompts* masivos repetidos (definiciones de *tools*), múltiples turnos cortos, *streaming* SSE, y **Structured Outputs** (FSM / gramáticas libres de contexto con **XGrammar** o *Outlines* / *llguidance*).
  - Impacto real del **Prefix Caching** (vLLM APC vs. SGLang RadixAttention) en TTFT cuando 50 agentes comparten un prefijo de 4k tokens.
  - Coste de enmascaramiento de logits en CPU al forzar JSON Schema estricto bajo alta concurrencia.
- **Recursos gratuitos en texto (móvil):**
  1. SGLang Docs — *Constrained Decoding (XGrammar / Compressed FSM)*: `https://docs.sglang.ai/`
  2. vLLM Docs — *Structured Outputs* y *Automatic Prefix Caching*: `https://docs.vllm.ai/`
- **Micro-práctica (Agente -> `[PREP]` -> `[CLD]` — Hito 1 de Proyecto A):**
  - `[PREP]`: El agente genera un generador de tráfico realista con 32–64 agentes concurrentes simulados (prefijo común de 3k tokens con 10 *tools* + respuestas con JSON Schema estricto en *streaming*).
  - `[CLD]`: Mide TTFT p95/p99 y throughput con Prefix Caching OFF vs. ON, y JSON libre vs. JSON Schema (XGrammar) en vLLM y SGLang.
- **Done:** Tienes el primer entregable funcional del **Proyecto A (`Agentic Inference Optimization Lab`)** con evidencia cuantitativa de cuánto ahorra el Prefix Caching y cuánto cuesta el *structured output*.

---

## 4. Bloque 3: Cuantización, Compresión y Aceleración de Decode (`CHAT-B3`)

**Alineación:** Oferta 2 (Core: FP8, INT4, AWQ, GPTQ, NVFP4/MXFP4, Speculative Decoding / EAGLE, quality regressions).
**Duración estimada:** ~4 semanas (~20–24 micro-sprints de 20 min).

### `[B3.1]` Fundamentos de Cuantización (PTQ, QAT, FP8, INT4, MXFP4/NVFP4)
- **Tipo:** `[MOB]` (4 sprints)
- **Depende de:** `B2.1` | **Desbloquea:** `B3.2`, `B3.4`
- **Infraestructura:** Móvil (100% texto)
- **Conceptos clave:**
  - Cuantización afín vs. simétrica; granularidad (*per-tensor*, *per-channel*, *per-token*, *per-group / block-wise*).
  - **W4A16 (AWQ, GPTQ):** comprime pesos para aliviar el *memory-bandwidth bound* del *decode*; activaciones siguen en FP16/BF16 (descuantización *on-the-fly* en kernel como Marlin).
  - **W8A8 (SmoothQuant INT8 y FP8 E4M3):** acelera también el *compute* usando Tensor Cores FP8 (Hopper/Ada); el problema de los *outliers* en activaciones.
  - **Microscaling FP4 (MXFP4 y NVFP4 en Blackwell):** factores de escala por bloque pequeño (16 o 32 elementos) en formato FP8 (`E4M3` en NVFP4 / `E8M0` en MXFP4) para mantener precisión a 4 bits.
  - Cuantización del **KV-Cache a FP8** (`--kv-cache-dtype fp8`) para duplicar la capacidad de contexto/concurrencia.
- **Recursos gratuitos en texto (móvil):**
  1. Hugging Face — *Quantization Overview & Conceptual Guide*: `https://huggingface.co/docs/transformers/quantization/overview`
  2. vLLM Docs — *Quantization (FP8, AWQ, GPTQ, KV Cache FP8)*: `https://docs.vllm.ai/en/latest/features/quantization/`
  3. NVIDIA Technical Blog — buscar en móvil: `"Introducing NVFP4 for Efficient and Accurate Low-Precision Inference"` (texto oficial NVIDIA Developer).
  4. *LLM Compressor* (`vllm-project/llm-compressor`) README y guías de SmoothQuant/AWQ/FP8: `https://github.com/vllm-project/llm-compressor`
- **Done:** Sabes justificar cuándo elegir W4A16 (decode-heavy, GPU pequeña) vs. FP8 W8A8 (alta concurrencia/prefill en H100/L40S) y explicar qué cambia estructuralmente en NVFP4/MXFP4.

### `[B3.2]` Despliegue y Calibración Práctica: FP8 vs. AWQ/GPTQ + KV-Cache FP8
- **Tipo:** `[PREP]` (2 sprints) + `[CLD]` (2 sprints, **Cloud-3 parte 1**)
- **Depende de:** `B2.2`, `B3.1` | **Desbloquea:** `B3.4`
- **Infraestructura:** **Cloud-3** -> Nebius/Scaleway `1x H100 80GB` (~2h)
- **Conceptos clave:** Calibración PTQ con `llm-compressor` (pesos FP8 estáticos/dinámicos y escalas de KV-Cache), kernels **Marlin** para INT4, comparación de memoria VRAM libre y throughput a alta concurrencia.
- **Micro-práctica (Agente -> `[PREP]` -> `[CLD]`):**
  - `[PREP]`: Script que ejecuta tu harness agéntico del Proyecto A sobre el mismo modelo en 4 configuraciones: (1) BF16 baseline, (2) FP8 W8A8, (3) FP8 W8A8 + KV-Cache FP8, (4) AWQ W4A16 (Marlin). Incluye cálculo automático de **coste por millón de tokens** usando el precio/hora real de la H100.
  - `[CLD]`: Ejecutar en `1x H100`, recolectar JSONs de latencia/throughput/VRAM.
- **Done:** Gráfica/tabla de speedup real y ahorro de $\$ / 1\text{M tokens}$ en H100 demostrando en qué régimen de concurrencia gana AWQ y en cuál gana FP8.

### `[B3.3]` Speculative Decoding (Draft Models, EAGLE, Medusa y MTP)
- **Tipo:** `[MOB]` (3 sprints) + `[PREP]` (1 sprint) + `[CLD]` (2 sprints, **Cloud-3 parte 2**)
- **Depende de:** `B2.4`, `B3.1` | **Desbloquea:** `B3.4`
- **Infraestructura:** **Cloud-3** -> `1x H100` (~1.5h)
- **Conceptos clave:**
  - Por qué el *Speculative Decoding* convierte un decode *memory-bound* en una verificación paralela *compute-bound* **sin degradar la distribución de salida** (verificación exacta).
  - Tasa de aceptación ($\alpha$) y longitud especulativa ($k$): por qué brilla en **código, JSON estructurado y RAG/edición** (donde muchos tokens son predecibles) y se degrada a *batch sizes* altos.
  - Arquitecturas: *Draft Model* clásico, **N-gram / Prompt Lookup** (gratis, ideal para edición de código/agentes), **EAGLE / EAGLE-3** (predicción a nivel de *features*), **Medusa** y **MTP** (*Multi-Token Prediction*, DeepSeek-V3).
- **Recursos gratuitos en texto (móvil):**
  1. vLLM Docs — *Speculative Decoding*: `https://docs.vllm.ai/en/latest/features/spec_decode.html`
  2. SGLang Docs — *Speculative Decoding (EAGLE / EAGLE-3)*: `https://docs.sglang.ai/`
  3. Blog técnico de Hugging Face — *Assisted Generation / Speculative Decoding* (texto): `https://huggingface.co/blog/assisted-generation`
- **Micro-práctica (Agente -> `[PREP]` -> `[CLD]`):**
  - `[PREP]`: Configura pruebas en vLLM/SGLang comparando: baseline vs. *N-gram matching* vs. *Draft model* (o *EAGLE* si hay checkpoint disponible para tu modelo base) en cargas de generación de código y JSON con concurrencia baja ($B=1, 4$) y alta ($B=32$).
  - `[CLD]`: Mide TPOT, tasa media de aceptación de tokens y el punto de cruce (*crossover batch size*) donde el speculative decoding deja de compensar.
- **Done:** Sabes explicar exactamente a partir de qué concurrencia el *speculative decoding* empieza a empeorar el throughput por saturación de cómputo.

### `[B3.4]` Control de Regresiones de Calidad y Frontera de Pareto
- **Tipo:** `[MOB]` (1 sprint) + `[PREP]` (1 sprint) + `[CLD]` (1 sprint, cierre **Cloud-3**)
- **Depende de:** `B3.2`, `B3.3` | **Desbloquea:** **Proyecto A Completo**
- **Infraestructura:** **Cloud-3** (misma `1x H100`, ~30–45 min) + análisis en CPU local
- **Conceptos clave:** Nunca optimizar velocidad a ciegas; evaluar caída de calidad tras cuantización con `lm-evaluation-harness` o **EvalPlus** (HumanEval+ / MBPP+ para código) y tests de validez de *tool calling* / JSON Schema; construir **curvas de Pareto** (Latencia TPOT vs. Throughput/GPU vs. Calidad vs. Coste/token).
- **Recursos gratuitos en texto (móvil):**
  1. EleutherAI `lm-evaluation-harness` README: `https://github.com/EleutherAI/lm-evaluation-harness`
  2. EvalPlus README: `https://github.com/evalplus/evalplus`
- **Micro-práctica (Agente -> Proyecto A):** El agente ejecuta un subset reproducible de calidad (ej. `gsm8k` o `humaneval` + tasa de *tool-call accuracy*) contra tus endpoints BF16, FP8 y AWQ, y genera un reporte Markdown + gráficas PNG de frontera de Pareto.
- **Done:** **Proyecto A (`Agentic Inference Optimization Lab`) cerrado y publicable en GitHub**, respondiendo de punta a punta a los requisitos de la Oferta 2 (Token Factory).

---

## 5. Bloque 4: Inferencia Distribuida, MoE, PDD y Redes (`CHAT-B4`)

**Alineación:** Oferta 1 (NCCL, interconexiones, escalabilidad) y Oferta 2 (PDD, WideEP, LMCache, cache-aware routing).
**Duración estimada:** ~4 semanas (~20–24 micro-sprints de 20 min).

### `[B4.1]` Paralelismo Multi-GPU (TP, PP, DP, CP) y Colectivas NCCL
- **Tipo:** `[MOB]` (4 sprints)
- **Depende de:** `B1.3`, `B2.1` | **Desbloquea:** `B4.2`, `B4.3`, `B4.4`
- **Infraestructura:** Móvil (100% texto)
- **Conceptos clave:**
  - **Tensor Parallelism (TP):** divide capas en el mismo nodo; requiere 2 `AllReduce` por capa Transformer (muy sensible a latencia -> exige NVLink).
  - **Pipeline Parallelism (PP):** divide capas entre nodos (`Send`/`Recv` P2P); menor tráfico de red, pero introduce *pipeline bubbles* y no reduce latencia de un solo token.
  - **Data Parallelism (DP)** en inferencia (réplicas independientes con router) y **Context Parallelism (CP)** (Ring Attention para contextos >128k).
  - Colectivas **NCCL**: `AllReduce`, `AllGather`, `ReduceScatter` y `All-to-All` (coste en volumen de bytes y latencia).
- **Recursos gratuitos en texto (móvil):**
  1. Hugging Face — *The Ultra-Scale Playbook* (secciones de DP, TP, PP, CP y Collectives): `https://huggingface.co/spaces/nanotron/ultrascale-playbook` (Oro puro en texto, diagramas claros en móvil).
  2. Google DeepMind — *How To Scale Your Model*, Partes 2 y 3 (*Sharding* y *Collectives*): `https://jax-ml.github.io/scaling-book/`
- **Done:** Puedes dibujar el flujo de activaciones y decir qué colectiva NCCL se dispara en TP vs. DP vs. EP, y por qué TP rara vez se cruza fuera de un nodo de 8 GPUs.

### `[B4.2]` Mixture-of-Experts (MoE) y Wide Expert Parallelism (WideEP)
- **Tipo:** `[MOB]` (4 sprints)
- **Depende de:** `B4.1` | **Desbloquea:** `B4.4`
- **Infraestructura:** Móvil (100% texto)
- **Conceptos clave:**
  - Anatomía MoE (DeepSeek-V3/R1, Qwen-MoE): parámetros totales enormes (ej. 671B) vs. activos bajos (ej. 37B) -> cuello de botella de memoria para alojar expertos + comunicaciones **`All-to-All`** (Dispatch / Combine).
  - **WideEP (Wide Expert Parallelism):** combinar DP para atención (evitando replicar KV-cache innecesariamente) con EP ancho (ej. 32–72 GPUs) para expertos, solapando comunicación y cómputo (*Two-Batch Overlap* / DeepEP).
  - El gran enemigo de producción: **Expert Load Imbalance** (expertos calientes degradando el **p99** de latencia) y estrategias de balanceo/replicación dinámica de expertos (*EPLB*).
- **Recursos gratuitos en texto (móvil):**
  1. DeepSeek-V3 Technical Report (solo Sección 3: *Infrastructures* y *Deployment* — 2 sprints de 20m): `https://arxiv.org/abs/2412.19437`
  2. vLLM / SGLang Blog sobre *Data Parallel Attention + Wide Expert Parallelism (DeepEP)* (buscar en `https://blog.vllm.ai/` o `https://lmsys.org/blog/`).
- **Done:** Sabes diagnosticar por qué un clúster MoE tiene buen p50 pero un p99 terrible (desbalanceo de tokens hacia un experto + sincronización `All-to-All`).

### `[B4.3]` Prefill–Decode Disaggregation (PDD), LMCache y Cache-Aware Routing
- **Tipo:** `[MOB]` (3 sprints) + `[PREP]` (2 sprints) + `[CLD]` (1.5 sprints, **Cloud-4 parte 1**)
- **Depende de:** `B2.3`, `B4.1` | **Desbloquea:** `B4.4`
- **Infraestructura:** **Cloud-4** -> Nebius `2x H100` (o `2x L40S`) (~1.5h)
- **Conceptos clave:**
  - Por qué mezclar Prefill y Decode en la misma GPU provoca interferencia (*chunked prefill* mitiga, pero no elimina la contención de SLA en contextos largos).
  - **PDD:** Pool de nodos *Prefill* (*compute-optimized*) + Pool de nodos *Decode* (*memory-bandwidth optimized*) conectados por transferencia rápida de KV-Cache (**NIXL** / RDMA / **LMCache**).
  - Criterio económico: calcular el **ratio de instancias P:D** (ej. 1P:3D vs. 2P:1D según relación longitud de entrada vs. salida) y verificar si la transferencia de KV-Cache compensa frente al coste de recalcularlo.
  - **Cache-Aware Routing:** enrutar peticiones al worker que ya tiene caliente el prefijo en su KV-Cache sin desbalancear la cola de espera.
- **Recursos gratuitos en texto (móvil):**
  1. LMCache GitHub & Docs (arquitectura de almacenamiento/transferencia KV-Cache y PDD): `https://github.com/LMCache/LMCache`
  2. Paper/Blog *DistServe* (concepto fundacional de PDD) + NVIDIA Dynamo Disaggregated Serving docs (`https://github.com/ai-dynamo/dynamo`).
- **Micro-práctica (Agente -> `[PREP]` -> `[CLD]`):**
  - `[PREP]`: Configura en un nodo de 2 GPUs un experimento PDD (GPU 0 = Prefill, GPU 1 = Decode con vLLM + LMCache/NIXL o configuración PDD nativa de vLLM/SGLang) frente a 2 réplicas colocadas (*colocated* DP=2).
  - `[CLD]`: Ejecuta tráfico mixto (prompts largos de 8k tokens + generación corta/media) y mide el impacto en TTFT p99 y TPOT p99.
- **Done:** Comprendes el trade-off real de PDD (cuándo gana latencia de cola y cuándo el overhead de transferencia KV-Cache lo hace peor que *chunked prefill*).

### `[B4.4]` Interconexiones (PCIe, NVLink, IB/RoCE) y `nccl-tests`
- **Tipo:** `[MOB]` (2 sprints) + `[PREP]` (1 sprint) + `[CLD]` (1.5 sprints, **Cloud-4 parte 2**)
- **Depende de:** `B4.1`, `B4.2` | **Desbloquea:** `B5.2`, **Proyecto B**
- **Infraestructura:** **Cloud-4** -> Nebius `2x–4x H100` (misma sesión o siguiente, ~1h)
- **Conceptos clave:**
  - Jerarquía de ancho de banda: PCIe Gen4/5 ($\approx 32\text{--}64\text{ GB/s}$ por dirección) vs. NVLink 4/5 ($\approx 450\text{--}900\text{ GB/s}$ bidireccional) vs. InfiniBand NDR / RoCEv2 ($400\text{ Gbps} = 50\text{ GB/s}$ por NIC).
  - Topología física (`nvidia-smi topo -m`), GPUDirect RDMA y P2P.
  - Métrica clave de `nccl-tests`: **algbw** (*algorithm bandwidth*) vs. **busbw** (*bus bandwidth*), definida para `AllReduce` de $N$ GPUs como:
    $$\text{busbw} = \text{algbw} \times \frac{2(N - 1)}{N}$$
- **Recursos gratuitos en texto (móvil):**
  1. `NVIDIA/nccl-tests` README y documento `PERFORMANCE.md` (lectura obligatoria de 15 min): `https://github.com/NVIDIA/nccl-tests`
  2. Modal — *GPU Glossary* (sección *Interconnects & Networking*): `https://modal.com/gpu-glossary`
- **Micro-práctica (Agente -> `[PREP]` -> `[CLD]`):**
  - `[PREP]`: Script que captura `nvidia-smi topo -m`, compila/ejecuta `all_reduce_perf` y `alltoall_perf` (de 8 MB a 2 GB), prueba con `NCCL_P2P_DISABLE=1` (simulando degradación a PCIe/host) vs. NVLink activo, y compara `busbw`.
  - `[CLD]`: Ejecuta en `2x–4x H100` en Nebius y guarda los logs para el módulo de validación del Proyecto B.
- **Done:** Sabes leer una salida de `all_reduce_perf` y detectar inmediatamente si un nodo está degradado (ej. cayendo a PCIe/SHM en vez de usar NVLink o IB).

---

## 6. Bloque 5: Benchmarking Científico y Cluster Acceptance (`CHAT-B5`)

**Alineación:** Oferta 1 (Core: Cluster Acceptance Testing, Megatron-LM, DCGM, dashboards, detección de regresiones) + Oferta 2 (Benchmarking riguroso).
**Duración estimada:** ~3 semanas (~15–18 micro-sprints de 20 min).

### `[B5.1]` Metodología de Benchmarking Reproducible (Harness Científico)
- **Tipo:** `[MOB]` (2 sprints) + `[LOC]` (2 sprints)
- **Depende de:** `B1.3`, `B2.3` | **Desbloquea:** `B5.2`, Proyecto B
- **Infraestructura:** Móvil + CPU local
- **Conceptos clave:** Warmup obligatorio, control de varianza (semillas, frecuencias de reloj bloqueadas / estado térmico), distribución de llegadas (Poisson vs. concurrencia fija), evitar *coordinated omission*, métricas de cola (**p95/p99**), y **AIPerf / GenAI-Perf** como referencia estándar.
- **Recursos gratuitos en texto (móvil):**
  1. NVIDIA `ai-dynamo/aiperf` Docs: `https://github.com/ai-dynamo/aiperf`
  2. vLLM Benchmarking Guide (`vllm bench serve`): `https://docs.vllm.ai/`
- **Micro-práctica (Agente -> CPU local):** Define el esquema JSON estándar y las aserciones de regresión estadística (ej. fallo automático si p99 sube $> 10\%$ o throughput cae $> 5\%$) que usará tu herramienta CLI del **Proyecto B**.
- **Done:** Tienes el esqueleto de la CLI del Proyecto B testeado en local con datos simulados.

### `[B5.2]` Cluster Acceptance Testing y Salud de Hardware (DCGM & Xid)
- **Tipo:** `[MOB]` (2 sprints) + `[PREP]` (2 sprints) + `[CLD]` (1.5 sprints, **Cloud-5 parte 1**)
- **Depende de:** `B4.4`, `B5.1` | **Desbloquea:** `B5.4`, Proyecto B
- **Infraestructura:** **Cloud-5** -> Nebius `2x–8x H100` (~1.5h)
- **Conceptos clave:**
  - Por qué `nvidia-smi` *"GPU-Util"* miente (solo mide si algún kernel estuvo activo en la ventana, no saturación de SMs ni Tensor Cores).
  - **NVIDIA DCGM:** diagnósticos activos (`dcgmi diag -r 3` para estrés térmico, memoria ECC, PCIe y NVLink) + métricas reales de profiling (`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, `DCGM_FI_PROF_DRAM_ACTIVE`, `DCGM_FI_PROF_SM_OCCUPANCY`).
  - Errores **Xid** críticos (ej. Xid 31 MMU fault, Xid 48 DBE ECC, Xid 79 GPU fallen off the bus, Xid 94/95 ECC containment) y *throttling* térmico/energético.
- **Recursos gratuitos en texto (móvil):**
  1. NVIDIA DCGM User Guide (secciones *Diagnostics* y *Field Identifiers*): `https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/`
  2. NVIDIA Xid Errors Reference (consulta rápida en móvil): `https://docs.nvidia.com/deploy/xid-errors/`
- **Micro-práctica (Agente -> `[PREP]` -> `[CLD]`):**
  - `[PREP]`: Integra en tu CLI del Proyecto B una suite de *Acceptance Testing* automatizada que ejecute: (1) inspección de driver/CUDA/topo/ECC, (2) `dcgmi diag`, (3) `nccl-tests` (`all_reduce_perf` + `alltoall_perf`), (4) GEMM burn-in (estrés térmico/potencia), y emita un veredicto `PASS / FAIL` con umbrales configurables.
  - `[CLD]`: Corre la suite completa contra un nodo multi-GPU recién aprovisionado en Nebius y guarda el informe HTML/Markdown.
- **Done:** Tienes un validador real de nodos GPU digno del equipo de *GPU Platform Benchmarking* de Nebius.

### `[B5.3]` Benchmarking de Entrenamiento: Megatron-LM y MFU
- **Tipo:** `[MOB]` (2 sprints) + `[PREP]` (1 sprint) + `[CLD]` (1.5 sprints, **Cloud-5 parte 2**)
- **Depende de:** `B4.1`, `B5.2` | **Desbloquea:** Cobertura total Oferta 1
- **Infraestructura:** **Cloud-5** -> Nebius `2x–8x H100` (~1h)
- **Conceptos clave:**
  - **MFU (*Model FLOPs Utilization*):** relación entre los FLOPs útiles de entrenamiento (aproximadamente $6 \times N_{\text{params}}$ por token en *forward + backward*, sin contar recomputación) y los TFLOPS pico teóricos de la GPU en BF16/FP8:
    $$\text{MFU} = \frac{6 \times N_{\text{params}} \times \text{tokens/s}}{\text{Peak FLOPs}_{\text{hardware}}}$$
  - **Megatron-LM:** paralelismo 3D (DP + TP + PP), *Activation Checkpointing*, *FlashAttention* y FP8 con *Transformer Engine*.
- **Recursos gratuitos en texto (móvil):**
  1. Hugging Face — *The Ultra-Scale Playbook* (sección *Benchmarking & MFU*): `https://huggingface.co/spaces/nanotron/ultrascale-playbook`
  2. NVIDIA Megatron-LM GitHub README (arquitectura y scripts de benchmark): `https://github.com/NVIDIA/Megatron-LM`
- **Micro-práctica (Agente -> `[PREP]` -> `[CLD]`):**
  - `[PREP]`: Contenedor oficial NGC PyTorch + script corto (20 iteraciones, datos sintéticos) ejecutando un entrenamiento pequeño en Megatron-LM (o `torchtitan` / mini-Megatron TP=2) con perfilado `nsys` en las iteraciones 10–12 y cálculo automático de MFU y TFLOPS/GPU.
  - `[CLD]`: Ejecuta en el mismo nodo multi-GPU de **Cloud-5**, extrae MFU y añade el módulo de entrenamiento a tu suite del Proyecto B.
- **Done:** Puedes interpretar un MFU del 45–55% en H100 y diagnosticar con `nsys` si el tiempo perdido se fue en comunicación NCCL sin solapar o en *data loading*.

### `[B5.4]` Orquestación K8s, Observabilidad DCGM y Arranque Rápido (Tu Base DevOps)
- **Tipo:** `[MOB]` (2 sprints) + `[LOC]` / `[PREP]` (3 sprints) + `[CLD]` (opcional validación en K8s)
- **Depende de:** `B5.2` | **Desbloquea:** **Proyecto B Completo**
- **Infraestructura:** Local (Kind/Minikube o Docker Compose para manifests/dashboards) + validación opcional en Nebius Managed K8s / Scaleway Kapsule GPU
- **Conceptos clave:**
  - **NVIDIA GPU Operator**, particionado **MIG** (*Multi-Instance GPU*, aislamiento físico) vs. **Time-Slicing** (compartición lógica).
  - Orquestación multi-nodo: **LeaderWorkerSet (LWS)** y nociones de **Slurm** (`sbatch`, GRES, Pyxis/Enroot) muy usado en Nebius.
  - Stack de observabilidad: **DCGM Exporter -> Prometheus -> Grafana** (paneles de Tensor Core active, HBM bandwidth, temperatura, errores Xid, TTFT/TPOT de vLLM).
  - Ingeniería de *Cold Starts*: *safetensors* + streaming directo desde S3, caché NVMe local, *image lazy loading* (eStargz / SOCI) o *checkpoint/restore* (`ai-dynamo/snapshot`).
- **Recursos gratuitos en texto (móvil):**
  1. NVIDIA GPU Operator & DCGM Exporter Docs: `https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/`
  2. Kubernetes SIG `LeaderWorkerSet` README: `https://github.com/kubernetes-sigs/lws`
  3. NVIDIA Dynamo `modelexpress` / `snapshot` READMEs: `https://github.com/ai-dynamo`
- **Micro-práctica (Agente -> Proyecto B):** El agente empaqueta tus dashboards JSON de Grafana (métricas DCGM + vLLM), Helm chart / manifests K8s (con Job de *Cluster Acceptance* antes de marcar el nodo como *schedulable*) y pipeline CI de detección de regresiones.
- **Done:** **Proyecto B (`GPU Cluster Acceptance & Bottleneck Profiler`) cerrado y publicable en GitHub**, respondiendo de punta a punta a la Oferta 1 (GPU Benchmarking) y potenciando tu perfil DevOps->MLOps.

---

## 7. Mapeo Incremental de los 2 Proyectos de Portfolio

No construyas los proyectos al final: cada unidad alimenta directamente un repositorio real en GitHub.

| Proyecto | Objetivo de Vacante | Unidades que lo construyen paso a paso | Entregable Final (README + Datos Reales) |
| :--- | :--- | :--- | :--- |
| **Proyecto A:** `agentic-inference-opt-lab` | **Oferta 2** (Nebius Token Factory — Senior MLE) | `B2.1` (calc KV-cache) -> `B2.2` (vLLM vs SGLang) -> `B2.4` (tráfico agéntico + XGrammar + Prefix Cache) -> `B3.2` (FP8/AWQ) -> `B3.3` (Speculative) -> `B3.4` (Evals + Pareto) | Harness reproducible + reporte con curvas de Pareto (TTFT, TPOT, p99, tok/s/GPU, $\$ / 1\text{M tokens}$ y precisión en evals). |
| **Proyecto B:** `gpu-cluster-acceptance-profiler` | **Oferta 1** (GPU Platform Benchmarking) | `B1.3` (parser `nsys`) -> `B4.4` (runner `nccl-tests`) -> `B5.1` (harness estadístico) -> `B5.2` (`dcgmi` + Xid + burn-in) -> `B5.3` (MFU Megatron) -> `B5.4` (K8s Job + DCGM/Prometheus/Grafana) | CLI contenedorizada + Helm/K8s + dashboards que auditan un nodo GPU nuevo con veredicto automático `PASS/FAIL` y detección de cuellos de botella. |

---

## 8. Registro Maestro de Progreso (Checkpoints de Handover)

> Actualiza esta tabla (o pega tu último estado al hacer Fork de un chat) para saber siempre cuál es tu próximo micro-sprint de 20 minutos.

| ID | Unidad | Modo principal | Cloud | Estado |
| :--- | :--- | :--- | :---: | :---: |
| `B1.1` | Roofline & PyTorch Internals | `[MOB]` `[LOC]` | No | `[ ]` |
| `B1.2` | CUDA C++ Local (`sm_61`) | `[MOB]` `[LOC]` | No | `[ ]` |
| `B1.3` | Profiling `nsys` y Linux `perf` | `[MOB]` `[LOC]` | No | `[ ]` |
| `B1.4` | OpenAI Triton y `ncu` | `[PREP]` `[CLD]` | Cloud-1 | `[ ]` |
| `B2.1` | Aritmética KV-Cache y Prefill/Decode | `[MOB]` `[LOC]` | No | `[ ]` |
| `B2.2` | vLLM vs. SGLang y FlashInfer | `[PREP]` `[CLD]` | Cloud-2 | `[ ]` |
| `B2.3` | TensorRT-LLM y NVIDIA Dynamo (`aisim`) | `[MOB]` `[LOC]` | Opcional | `[ ]` |
| `B2.4` | Cargas Agénticas (Prefix Cache + JSON) | `[PREP]` `[CLD]` | Cloud-2 | `[ ]` |
| `B3.1` | Teoría Cuantización (FP8/INT4/NVFP4) | `[MOB]` | No | `[ ]` |
| `B3.2` | FP8 vs. AWQ + KV-Cache FP8 | `[PREP]` `[CLD]` | Cloud-3 | `[ ]` |
| `B3.3` | Speculative Decoding (EAGLE/Draft/N-gram) | `[PREP]` `[CLD]` | Cloud-3 | `[ ]` |
| `B3.4` | Evals de Calidad y Frontera de Pareto | `[PREP]` `[CLD]` | Cloud-3 | `[ ]` |
| `B4.1` | Paralelismo (TP/PP/DP/CP) y NCCL | `[MOB]` | No | `[ ]` |
| `B4.2` | MoE y WideEP (All-to-All / Imbalance) | `[MOB]` | No | `[ ]` |
| `B4.3` | PDD, LMCache y Cache-Aware Routing | `[PREP]` `[CLD]` | Cloud-4 | `[ ]` |
| `B4.4` | Interconexiones y `nccl-tests` | `[PREP]` `[CLD]` | Cloud-4 | `[ ]` |
| `B5.1` | Metodología de Benchmarking Científico | `[MOB]` `[LOC]` | No | `[ ]` |
| `B5.2` | Cluster Acceptance (`dcgmi` + Xid) | `[PREP]` `[CLD]` | Cloud-5 | `[ ]` |
| `B5.3` | Entrenamiento Megatron-LM y MFU | `[PREP]` `[CLD]` | Cloud-5 | `[ ]` |
| `B5.4` | K8s GPU Operator, DCGM y Cold Starts | `[MOB]` `[LOC]` | Opcional | `[ ]` |