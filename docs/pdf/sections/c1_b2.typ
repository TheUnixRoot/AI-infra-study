#import "../theme.typ": *

= Curso 1 · Bloque 2: Motores de Inferencia, KV-Cache y Cargas Agénticas (`CHAT-B2`)

#block-banner(
  "B2",
  "Aritmética KV-Cache, vLLM vs SGLang, Dynamo/TRT-LLM y Cargas Agénticas",
  "Oferta 2 (Core: vLLM, SGLang, TensorRT-LLM, NVIDIA Dynamo, FlashInfer, cargas agénticas)",
  "~4 semanas · 20–26 sprints de 20m",
)[
  Cuantifica con precisión la memoria KV-Cache en arquitecturas MHA, GQA y MLA; compara motores de *serving* modernos (`vLLM V1`, `SGLang`, `FlashInfer`, `TRT-LLM`, `NVIDIA Dynamo`); y optimiza cargas agénticas con *Prefix Caching* y *Structured Outputs* (Hito 1 del Proyecto A).
]

#infografia(
  "B2.1 Medido · Consumo KV-Cache por Token (KiB/tok) y Concurrencia en H100 (80 GB, S=8192)",
  "Comparativa analítica exacta generada por b2_1_kv_cache/kv_calc.py (util=0.90, overhead=2 GB -> 70.0 GiB útiles en H100, 41.2 GiB en L40S).",
  height: 185pt,
)[
  #place(dx: 8pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[KV-Cache por Token (FP16 vs FP8) — Impacto de GQA vs MLA])
  #hbar(8pt, 20pt, "Qwen2.5-72B (GQA FP16)", "320.0 KiB/tok (2.50 GiB/seq)", 1.0, max-w: 185pt, h: 12pt, col: crimson)
  #hbar(8pt, 37pt, "Qwen2.5-72B (GQA FP8)", "160.0 KiB/tok (1.25 GiB/seq)", 0.5, max-w: 185pt, h: 12pt, col: amber)
  #hbar(8pt, 54pt, "Llama-3-8B (GQA FP16)", "128.0 KiB/tok (1.00 GiB/seq)", 0.4, max-w: 185pt, h: 12pt, col: blue)
  #hbar(8pt, 71pt, "DeepSeek-V3 (MLA FP16)", "68.6 KiB/tok (0.54 GiB/seq)", 68.6 / 320.0, max-w: 185pt, h: 12pt, col: teal)
  #hbar(8pt, 88pt, "Llama-3-8B (GQA FP8)", "64.0 KiB/tok (0.50 GiB/seq)", 0.2, max-w: 185pt, h: 12pt, col: navy)
  #hbar(8pt, 105pt, "DeepSeek-V3 (MLA FP8)", "34.3 KiB/tok (0.27 GiB/seq)", 34.3 / 320.0, max-w: 185pt, h: 12pt, col: purple)

  // Diagrama inferior: Flujo PagedAttention / RadixAttention / Prefix Caching
  #place(dx: 8pt, dy: 126pt, text(fill: navy, weight: "bold", size: 7.8pt)[Arquitectura de Serving Agéntico (vLLM V1 vs SGLang + FlashInfer):])
  #diag-node(8pt, 140pt, 106pt, 38pt, "Prefijo Común (3k tok)", sub: "System Prompt + 10 Tools\nReutilizado en Prefix Cache", fill: bg-teal, col: teal)
  #arrow-r(114pt, 159pt, 134pt, col: teal)
  #diag-node(134pt, 140pt, 114pt, 38pt, "Scheduler + KV Paged", sub: "vLLM APC / SGLang Radix\nChunked Prefill + Continuous Batch", fill: bg-navy, col: navy)
  #arrow-r(248pt, 159pt, 268pt, col: navy)
  #diag-node(268pt, 140pt, 100pt, 38pt, "Backend FlashInfer", sub: "Atención Ragged/Paginada\nSM80+ / SM89 / SM90", fill: bg-amber, col: amber)
  #arrow-r(368pt, 159pt, 386pt, col: amber)
  #diag-node(386pt, 140pt, 82pt, 38pt, "XGrammar / FSM", sub: "JSON Schema estricto\n(Masking logits CPU)", fill: bg-purple, col: purple)
]

== Unidad B2.1 · Aritmética del KV-Cache, Atención y Prefill vs. Decode

#unit-header(
  "B2.1",
  "Aritmética del KV-Cache y Presupuestado de VRAM (MHA, GQA, MLA)",
  "[MOB] 4s + [LOC] 1s",
  "B1.1",
  "B2.2, B3.1, B4.1, Proyecto A",
  "Móvil + CPU local (0 EUR)",
  "Calcular mentalmente cuántas secuencias de 8k tokens caben en H100 para 8B/70B en FP16 vs FP8.",
)

#concepto-clave[
  1. *Fórmulas Exactas del KV-Cache:* Para $L$ capas, $n_"kv"$ cabezas KV, dimensión $d_"head"$, longitud $S$, batch $B$ y precisión $b_"kv"$ bytes:
     $ "KV"_"MHA/GQA" = 2 dot L dot n_"kv" dot d_"head" dot S dot B dot b_"kv" $
     El factor $2$ corresponde a los tensores Key y Value. En *GQA* ($n_"kv" < n_q$), varias cabezas de consulta comparten el mismo grupo KV (p. ej. Llama-3-8B reduce 32 a 8 cabezas = *4x* ahorro). En *MLA* (*Multi-Head Latent Attention*, DeepSeek-V2/V3) se proyectan K y V a un vector latente conjunto $c_t^"KV"$ de rango `kv_lora_rank` ($512$) más una clave RoPE desacoplada $k_t^R$ de dimensión `qk_rope_head_dim` ($64$):
     $ "KV"_"MLA" = L dot ("kv_lora_rank" + "qk_rope_head_dim") dot S dot B dot b_"kv" $
     *Sin factor 2*, logrando en DeepSeek-V3 (61 capas, 128 cabezas) solo $68.625 "KiB/token"$ en FP16 (*4.66x menos* que Qwen2.5-72B).
  2. *Presupuesto de VRAM y Preemption:* La memoria utilizable en una GPU de capacidad $M_"total"$ con utilización $u$ y *overhead* $O$ es $M_"util" = M_"total" dot u - O$. Si $M_"pesos" + "KV"_"seq" > M_"util"$ no cabe ni una secuencia (`OOM: Available KV < 1 seq`). En producción, al llenarse los bloques KV de *PagedAttention*, el *scheduler* expulsa (*preempt-by-recompute* o *swap*) las peticiones más recientes, disparando el p99 de TTFT/TPOT.
  3. *SLOs de Serving:* *TTFT* (*Time To First Token*, dominado por *prefill* y cola), *TPOT* / *ITL* (*Time Per Output Token* / *Inter-Token Latency*, dominados por *decode memory-bound*) y *Goodput* (peticiones/s que cumplen simultáneamente TTFT y TPOT SLO).
]

#practica-box(title: "Práctica B2.1 (b2_1_kv_cache/) · Calculadora CLI kv_calc.py")[
  - *Ejecución offline:* `cd b2_1_kv_cache && bash ./run.sh` calcula la matriz para `Meta-Llama-3-8B`, `Qwen2.5-72B-Instruct` y `DeepSeek-V3` a $S=8192$, `gpu_mem_util=0.90`, `overhead=2.0 GB` ($41.2 "GiB"$ útiles en L40S; $70.0 "GiB"$ en H100).
  - *Hallazgos clave (`artifacts/b2_1_kv_report.json`):*
    - *Llama-3-8B:* pasa de `26` (L40S) / `55` (H100) streams en FP16+FP16 a *`67` / `125` streams* en FP8+FP8.
    - *Qwen2.5-72B:* con pesos FP8 (`67.7 GiB`) + KV FP16 (`2.50 GiB/seq`) quedan `2.29 GiB` libres en 1x H100 $arrow.r$ `0 streams (OOM: Available KV < 1 seq)`; al cuantizar el KV a FP8 (`1.25 GiB/seq`) entra *1 stream* sin Tensor Parallelism ($"TP" >= 2$ en producción).
    - *DeepSeek-V3:* 671B parámetros exigen mínimo *9x H100* en FP8 solo para pesos, pero su KV MLA de `34.3 KiB/tok` (FP8) permite alta concurrencia una vez distribuido.
]

#ejercicio-box[
  *Ejercicio B2.1-A:* Calcula a mano el KV-Cache en KiB/token (FP16) y GiB por secuencia ($S = 32768$) de un modelo con $L = 40$, $n_q = 40$, $n_"kv" = 8$, $d_"head" = 128$. Si los pesos ocupan $28 "GiB"$ en FP16, ¿cuántas secuencias completas de 32k caben en `1x H100` ($70 "GiB"$ útiles)?
]

== Unidad B2.2 · Comparativa Práctica: vLLM vs. SGLang y FlashInfer

#unit-header(
  "B2.2",
  "vLLM (V1) vs. SGLang y Backend FlashInfer",
  "[MOB] 2s + [PREP] 2s + [CLD] 2s",
  "B2.1",
  "B2.3, B2.4, B3.2",
  "Cloud-2: 1x L40S o 1x H100 (~2h)",
  "Tablas reales TTFT/TPOT p50/p99 y tok/s/GPU comparando vLLM vs SGLang bajo igual carga.",
)

#concepto-clave[
  1. *vLLM V1 vs SGLang:* vLLM V1 desacopla la ejecución de CPU (*API Server + Scheduler* sin bloqueos) del bucle de GPU (*EngineCore*), eliminando el overhead de Python del *critical path*, e integra *PagedAttention* + *Automatic Prefix Caching (APC)*. SGLang introduce *RadixAttention* (árbol radix con política LRU que reutiliza prefijos entre turnos multi-agente y llamadas paralelas *tree-structured*) y un *zero-overhead CPU scheduler* que solapa preparación de batch $t+1$ mientras GPU ejecuta paso $t$.
  2. *FlashInfer como Estándar de Serving:* En decode con *continuous batching*, cada petición tiene longitud de KV distinta y bloques no contiguos (*ragged tensors*). FlashInfer unifica kernels JIT especializados (FP16/FP8/paged/cascade attention) maximizando ocupación incluso con tamaños de página pequeños.
]

#practica-box(title: "Práctica B2.2 (b2_2_vllm_vs_sglang/) · Harness run_compare.sh + bench_client.py")[
  - *Diseño científico (`[PREP]` listo):* `bench_client.py` genera prompts como IDs aleatorios únicos (`--prefix-share 0.0`) para medir motores puros sin favorecer caché de prefijo (y `--prefix-share 0.5` por separado), fuerza salida exacta de 256 tokens (`ignore_eos` + `min_tokens`), mide TTFT, TPOT y huecos SSE (ITL) p50/p95/p99 a $B in {1, 8, 32, 64}$, y scrapea `/metrics` de Prometheus (`kv_cache_usage_perc`, `num_preemptions`, `sglang:token_usage`).
  - *Validación local (`0 EUR`):* `bash b2_2_vllm_vs_sglang/run_compare.sh --dry-run` verifica sintaxis, parser Prometheus y esquema JSON en `artifacts/b2_2_dryrun/` (marcado `synthetic: true`). En Cloud-2, `cloud_bootstrap.sh` instala venvs separadas `.venv-vllm` y `.venv-sglang` tras fijar `sudo shutdown -h +90`.
]

#ejercicio-box[
  *Ejercicio B2.2-A:* En tu ejecución en Cloud-2 observas que al pasar de $B=32$ a $B=64$, el ITL p50 apenas sube de `14 ms` a `17 ms`, pero el *TTFT p99* salta de `180 ms` a `2,400 ms` y `vllm:num_preemptions_total` sube de `0` a `19`. Explica la cadena causal física y qué tres flags ajustarías en vLLM/SGLang.
]

== Unidad B2.3 · TensorRT-LLM y Arquitectura NVIDIA Dynamo (Offline + Cloud)

#unit-header(
  "B2.3",
  "TensorRT-LLM y NVIDIA Dynamo (Router, NIXL, AIPerf, AIConfigurator)",
  "[MOB] 3s + [LOC] 1s + [CLD] opcional",
  "B2.2",
  "B4.3, B5.1",
  "CPU local (aiconfigurator) + opcional 1x H100",
  "Modelar configuraciones óptimas en aiconfigurator y conocer piezas de Dynamo vs vLLM/SGLang.",
)

#concepto-clave[
  1. *TensorRT-LLM:* Ofrece motores C++/CUDA compilados (`trtllm-build` con fusión agresiva de grafos, *in-flight batching* y plugins FP8/INT4) junto al *PyTorch backend* moderno para iteración rápida, integrándose en producción detrás de *Triton Inference Server*.
  2. *Ecosistema NVIDIA Dynamo (`ai-dynamo`, Rust + Python):*
     - *Smart Router (KV-Cache Aware):* Enruta peticiones hacia el worker que ya retiene los bloques del prefijo en su VRAM/LMCache.
     - *Disaggregated Serving + NIXL:* Separa nodos de *Prefill* y *Decode* transfiriendo el KV-Cache por RDMA/NVLink mediante la librería NIXL (*NVIDIA Inference Xfer Library*).
     - *AIPerf & AIConfigurator:* Permiten simular el espacio de búsqueda $(T P, D P, B, "SLA")$ en CPU local (`0 EUR`) antes de alquilar H100s.
]

#practica-box(title: "Práctica B2.3 ([LOC] 0 EUR) · Modelado Offline con aiconfigurator")[
  - Ejecutar el modelador offline sobre `Qwen2.5-7B` y `Llama-3-70B` fijando SLOs `TTFT <= 350 ms` y `TPOT <= 25 ms`, exportando la configuración óptima estimada de TP y *batch size* para contrastar con las mediciones reales de B2.2 en Cloud-2.
]

== Unidad B2.4 · Optimización de Cargas Agénticas (Hito 1 de Proyecto A)

#unit-header(
  "B2.4",
  "Cargas Agénticas: Prefix Caching Masivo y Structured Outputs",
  "[MOB] 2s + [PREP] 2s + [CLD] 2s",
  "B2.2",
  "Hito 1 Proyecto A, B3.3",
  "Cloud-2 parte 2: 1x L40S o 1x H100 (~1.5-2h)",
  "Evidencia cuantitativa del ahorro por Prefix Caching y coste de XGrammar/JSON Schema.",
)

#concepto-clave[
  1. *Perfil de Tráfico Agéntico:* A diferencia del chat genérico, los agentes envían *system prompts masivos repetidos* (definiciones de 10–30 herramientas + ejemplos *few-shot* = 3k–6k tokens idénticos), producen salidas cortas en *streaming* SSE y exigen *Structured Outputs* (JSON Schema válido sin errores de sintaxis).
  2. *Prefix Caching (APC / RadixAttention):* Evita recalcular el *prefill* de los primeros $S_"pref"$ tokens compartidos; convierte un coste $O(S_"pref"^2)$ por petición en una lectura directa de KV-Cache, reduciendo el TTFT (la magnitud depende de la carga y se mide en B2.4) cuando 32–64 agentes comparten contexto.
  3. *Coste de Constrained Decoding (XGrammar / Outlines / llguidance):* En cada token de salida, una máquina de estados finitos (FSM) o gramática libre de contexto precomputa una máscara binaria sobre el vocabulario ($|V| approx 151936$ en Qwen2.5) para poner a $-infinity$ los logits inválidos. Si el enmascaramiento CPU no está compilado/cacheado (*jump-forward* / *compressed FSM*), puede introducir burbujas en el loop de decode a alta concurrencia.
]

#practica-box(title: "Práctica B2.4 ([PREP] -> [CLD], Hito 1 de agentic-inference-opt-lab)")[
  - Construir un generador de tráfico agéntico con 32–64 agentes concurrentes (prefijo común de 3,000 tokens + 10 definiciones de *tools* + respuestas JSON Schema estricto en *streaming*), evaluando la matriz $2 times 2$: `(Prefix Cache OFF/ON) x (JSON libre vs XGrammar)` en vLLM y SGLang.
]

#ejercicio-box[
  *Ejercicio B2.4-A:* 50 agentes comparten un prefijo de `3072` tokens y envían `256` tokens nuevos de observación cada turno (total `3328` entrada, `128` salida). Calcula qué porcentaje de cómputo de *prefill* ahorra el Prefix Caching tras el primer turno y qué ocurre en un esquema *round-robin* ciego frente a *sticky/radix routing*.
]

#solucion-box[
  - *Sol. B2.1-A:* KV/token = $2 dot 40 dot 8 dot 128 dot 2 = 163840 "B" = 160.0 "KiB/tok"$. Por secuencia de `32,768` tokens: $32768 dot 160 "KiB" = 5242880 "KiB" = 5.00 "GiB/seq"$. VRAM disponible para KV en H100 = $70.0 - 28.0 = 42.0 "GiB"$. Concurrencia máxima = $floor(42.0 / 5.00) = 8$ secuencias simultáneas.
  - *Sol. B2.2-A:* Al pasar a $B=64$, el KV-Cache supera el presupuesto útil y activa *preemption*: las peticiones expulsadas vuelven a cola para recalcular su prefill desde cero, disparando la cola de espera (TTFT p99 = 2,400 ms) mientras los tokens en vuelo mantienen TPOT bajo. Ajustes: (1) activar `--kv-cache-dtype fp8` (duplica capacidad KV), (2) subir `--gpu-memory-utilization` si hay margen seguro (p. ej. `0.90 -> 0.93`) o activar *chunked prefill* (`--enable-chunked-prefill`) / limitar `max-num-seqs` para acotar la admisión sin expulsiones.
  - *Sol. B2.4-A:* Se ahorra $3072 / 3328 = 92.3\%$ de los tokens de prefill en cada turno posterior. Con un balanceador *round-robin* ciego entre 4 réplicas sin afinidad de prefijo, la tasa de acierto de caché cae al ~25% (o obliga a recalcular el prefijo en cada réplica la primera vez); con *cache-aware routing* se consolida el hit por encima del 90%.
]

#fuentes-box[
  1. Kipply — *Transformer Inference Arithmetic*: `https://kipp.ly/transformer-inference-arithmetic/`
  2. Google DeepMind — *How To Scale Your Model, Part 7 (Inference)*: `https://jax-ml.github.io/scaling-book/inference/`
  3. vLLM Architecture & APC/Chunked Prefill Blog: `https://blog.vllm.ai/` · `https://docs.vllm.ai/`
  4. SGLang Docs (*RadixAttention & XGrammar*) & FlashInfer Docs: `https://docs.sglang.ai/` · `https://docs.flashinfer.ai/`
  5. NVIDIA Dynamo, AIPerf, AIConfigurator & TensorRT-LLM: `https://github.com/ai-dynamo/dynamo` · `https://nvidia.github.io/TensorRT-LLM/`
]
