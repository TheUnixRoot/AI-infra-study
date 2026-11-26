#import "../theme.typ": *

= Curso 1 · Bloque 3: Cuantización, Speculative Decoding y Pareto (`CHAT-B3`)

#block-banner(
  "B3",
  "Cuantización (FP8/AWQ/MXFP4), Speculative Decoding y Frontera de Pareto",
  "Oferta 2 (Core: FP8, INT4, AWQ, GPTQ, NVFP4/MXFP4, EAGLE/Speculative, control de regresiones)",
  "~4 semanas · 20–24 sprints de 20m",
)[
  Reduce drásticamente los bytes leídos por token y el coste por millón de tokens mediante cuantización de pesos/activaciones/KV, acelera el decode *memory-bound* con decodificación especulativa exacta y valida todo con evaluaciones de calidad y curvas de Pareto (cierre del Proyecto A).
]

#infografia(
  "Taxonomía de Precisión (W16A16 -> FP8 -> W4A16 -> NVFP4) y Verificación Especulativa",
  "Arriba: anatomía de formatos numéricos y su impacto en memoria vs Tensor Cores. Abajo: mecanismo exacto de Speculative Decoding y punto de cruce por batch size.",
  height: 195pt,
)[
  // Fila superior: formatos de cuantización
  #place(dx: 8pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[Evolución de Formatos y Régimen de Aceleración])
  #diag-node(8pt, 18pt, 105pt, 54pt, "BF16 / FP16 (16-bit)", sub: "1 bit signo · 8/5 exp · 7/10 mant\nBaseline sin pérdida\n1x VRAM · H100 ~989 TF/s", fill: bg-light, col: slate)
  #arrow-r(113pt, 45pt, 127pt, col: slate)
  #diag-node(127pt, 18pt, 110pt, 54pt, "FP8 W8A8 (E4M3)", sub: "Tensor Cores FP8 nativos\n2x VRAM · 2x Compute (H100)\n+ KV FP8 = 2x concurrencia", fill: bg-navy, col: navy)
  #arrow-r(237pt, 45pt, 251pt, col: navy)
  #diag-node(251pt, 18pt, 108pt, 54pt, "AWQ / GPTQ W4A16", sub: "Pesos 4-bit (group=128)\nKernels Marlin -> FP16\n4x ahorro VRAM (decode B bajo)", fill: bg-teal, col: teal)
  #arrow-r(359pt, 45pt, 373pt, col: teal)
  #diag-node(373pt, 18pt, 96pt, 54pt, "MXFP4 / NVFP4", sub: "Microscaling (Blackwell)\nBloque 16 (E4M3) / 32 (E8M0)\n4 bits con alto rango dinámico", fill: bg-purple, col: purple)

  // Separador horizontal
  #place(dx: 8pt, dy: 82pt, line(length: 458pt, stroke: 0.5pt + border-gray))

  // Fila inferior: Speculative Decoding
  #place(dx: 8pt, dy: 88pt, text(fill: navy, weight: "bold", size: 8pt)[Speculative Decoding: De Decode Memory-Bound a Verificación Paralela Compute-Bound])
  #diag-node(8pt, 104pt, 112pt, 44pt, "1. Propuesta (k=4 tokens)", sub: "Draft / EAGLE-3 / N-gram / MTP\nCosto barato en latencia", fill: bg-amber, col: amber)
  #arrow-r(120pt, 126pt, 138pt, col: amber)
  #diag-node(138pt, 104pt, 128pt, 44pt, "2. Verificación 1 Forward Target", sub: "1 lectura de pesos Target para\nvalidar k+1 logits en paralelo", fill: bg-teal, col: teal)
  #arrow-r(266pt, 126pt, 284pt, col: teal)
  #diag-node(284pt, 104pt, 184pt, 44pt, "3. Rejection Sampling Exacto", sub: "Acepta prefix sin distorsión estadística.\nSpeedup alto con alpha alto (código/JSON) y B bajo.\nEn B alto satura cómputo -> empeora TPOT.", fill: bg-navy, col: navy)

  #place(dx: 8pt, dy: 155pt, box(width: 460pt, fill: bg-light, stroke: 0.4pt + border-gray, inset: 5pt, radius: 3pt)[
    #set text(size: 7.4pt, fill: slate)
    *Regla de Oro Pareto (B3.4):* Ninguna optimización se acepta sin medir calidad (`lm-eval` / `EvalPlus` + validez JSON/Tool-Call). El objetivo es minimizar `$/1M tokens` sujeto a cumplir SLO de p99 y caída de calidad `< 1%`.
  ])
]

== Unidad B3.1 · Fundamentos de Cuantización (PTQ, QAT, FP8, INT4, MXFP4/NVFP4)

#unit-header(
  "B3.1",
  "Fundamentos de Cuantización y Formatos de Baja Precisión",
  "[MOB] 4s",
  "B2.1",
  "B3.2, B3.4",
  "Móvil (100% lectura técnica)",
  "Justificar cuándo elegir W4A16 frente a FP8 W8A8 y explicar microscaling NVFP4/MXFP4.",
)

#concepto-clave[
  1. *Cuantización Afín vs Simétrica y Granularidad:*
     - *Simétrica:* $x_q = "clamp"("round"(x / s), -2^(b-1), 2^(b-1)-1)$ con *zero-point* $z=0$; menor sobrecarga aritmética en kernels.
     - *Afín (Asimétrica):* añade *zero-point* entero $z$, aprovechando todo el rango para distribuciones sesgadas (p. ej. tras ReLU).
     - *Granularidad:* *per-tensor* (1 escala), *per-channel* (por fila/columna en pesos), *per-token* (dinámica en activaciones) y *per-group / block-wise* (bloques de 32–128 elementos; esencial en 4 bits).
  2. *W4A16 (AWQ / GPTQ + Kernels Marlin) vs W8A8 (SmoothQuant / FP8 E4M3):*
     - *W4A16:* comprime pesos 4x para aliviar el cuello *memory-bandwidth bound* del decode a concurrencia baja/media; *AWQ* protege el ~1% de canales de pesos salientes según magnitud de activación (*Activation-aware*), mientras *GPTQ* usa información de segundo orden (Hessiano inverso). En ejecución, kernels como *Marlin* descuantizan al vuelo de INT4 a FP16 en registros sin acelerar los Tensor Cores FP16.
     - *W8A8 (FP8 E4M3):* En Ada (`sm_89`) y Hopper (`sm_90`), los pesos *y* las activaciones se multiplican nativamente en *Tensor Cores FP8* (*2x TFLOP/s*), ganando tanto en *prefill* largo como en *decode* de alta concurrencia donde el problema pasa a *compute-bound*. *SmoothQuant* migra matemáticamente la varianza de *outliers* de activación hacia los pesos.
  3. *Microscaling FP4 (MXFP4 y NVFP4 en Blackwell):* Al bajar a 4 bits, un único factor de escala por 128 elementos falla ante *outliers*. *MXFP4* (estándar OCP) usa bloques de 32 elementos con escala exponente `E8M0` (potencias de 2); *NVFP4* (NVIDIA Blackwell) reduce el bloque a 16 elementos con escala `FP8 E4M3` + escala global FP32, reteniendo casi toda la precisión de FP8 a la mitad de ancho de banda.
]

#practica-box(title: "Guía de Decisión Arquitectural B3.1")[
  - *GPU pequeña o batch pequeño ($B <= 8$, decode puro):* W4A16 (AWQ + Marlin) minimiza bytes leídos de VRAM por paso.
  - *Alta concurrencia o prompts largos en L40S/H100:* FP8 W8A8 + `--kv-cache-dtype fp8` acelera Tensor Cores *y* duplica la capacidad concurrente de KV-Cache.
]

== Unidad B3.2 · Despliegue y Calibración Práctica: FP8 vs AWQ + KV-Cache FP8

#unit-header(
  "B3.2",
  "Calibración PTQ (llm-compressor) y Benchmark en H100",
  "[PREP] 2s + [CLD] 2s",
  "B2.2, B3.1",
  "B3.4",
  "Cloud-3 parte 1: 1x H100 80GB (~2h)",
  "Tabla real de speedup y \$/1M tokens demostrando el punto de cruce AWQ vs FP8.",
)

#concepto-clave[
  - Con `vllm-project/llm-compressor`, la calibración PTQ estima factores de escala estáticos sobre 128–512 muestras representativas del dominio (cuidado: calibrar solo con texto inglés degrada herramientas/JSON en español si hay *outliers* específicos).
  - La cuantización del *KV-Cache a FP8* (`--kv-cache-dtype fp8` con factores `k_scale`, `v_scale` calibrados o `auto`) reduce a la mitad la memoria por token y duplica la concurrencia antes de *preemption* en `B2.1`.
]

#practica-box(title: "Práctica B3.2 ([PREP] -> [CLD] Cloud-3) · Matriz 4 Vías y Coste $/1M")[
  - Reutilizar el harness agéntico de B2.4 sobre `Qwen2.5-14B/32B` o `Llama-3.1-8B` comparando: (1) BF16, (2) FP8 W8A8, (3) FP8 W8A8 + KV FP8, (4) AWQ W4A16 (Marlin).
  - Calcular automáticamente el coste por millón de tokens a partir del throughput sostenido y el precio/hora $C_"GPU"$ de la H100:
    $ "Coste / 1M tokens" = frac(C_"GPU" ("USD/h"), "tok/s sostenidos" times 3600) times 10^6 $
]

== Unidad B3.3 · Speculative Decoding (Draft, EAGLE, Medusa y MTP)

#unit-header(
  "B3.3",
  "Speculative Decoding y Predicción Multi-Token",
  "[MOB] 3s + [PREP] 1s + [CLD] 2s",
  "B2.4, B3.1",
  "B3.4",
  "Cloud-3 parte 2: 1x H100 (~1.5h)",
  "Identificar empíricamente el batch size de cruce donde speculative deja de compensar.",
)

#concepto-clave[
  1. *Principio Fundamental:* En decode $B=1$, leer 140 GB de pesos para generar 1 token desperdicia el >95% de los Tensor Cores. Si un mecanismo barato propone $k$ tokens candidatos, el modelo *target* puede evaluarlos *todos en una sola pasada forward* (como un mini-prefill de longitud $k+1$) por prácticamente la misma latencia de memoria. Mediante *rejection sampling* modificado, la distribución matemática de salida es *idéntica* a la del modelo target sin especulación.
  2. *Speedup Teórico y Tasa de Aceptación $alpha$:* El número esperado de tokens aceptados por paso es $E["tokens"] = frac(1 - alpha^(k+1), 1 - alpha)$ (hasta $k+1$ si todos se aceptan). Brilla en *código, JSON estructurado, RAG y edición* ($alpha approx 0.7"–"0.9$) y pierde ventaja al subir $B$ porque verificar $B times (k+1)$ tokens empuja la GPU al régimen *compute-bound*.
  3. *Familias Modernas:* *Draft Model* clásico (p. ej. 0.5B guiando a 7B/72B), *N-gram / Prompt Lookup* (cero VRAM extra, ideal para edición de código/RAG copiando subcadenas del prompt), *EAGLE / EAGLE-3* (cabeza ligera autorregresiva sobre *features* ocultos de penúltima capa), *Medusa* (múltiples cabezas paralelas) y *MTP* (*Multi-Token Prediction* entrenado nativamente en DeepSeek-V3).
]

#practica-box(title: "Práctica B3.3 ([PREP] -> [CLD] Cloud-3) · Curva de Cruce vs Batch Size")[
  - Medir en vLLM/SGLang: `baseline` vs `ngram` vs `draft / EAGLE` bajo cargas JSON/código a $B in {1, 4, 16, 32}$, registrando `TPOT`, `acceptance_rate` y el punto exacto donde la saturación de cómputo anula la ganancia.
]

== Unidad B3.4 · Control de Regresiones de Calidad y Frontera de Pareto

#unit-header(
  "B3.4",
  "Evaluación de Calidad y Frontera de Pareto (Cierre Proyecto A)",
  "[MOB] 1s + [PREP] 1s + [CLD] 1s",
  "B3.2, B3.3",
  "Proyecto A Completo",
  "Cloud-3 cierre: 1x H100 (~30-45m) + análisis CPU",
  "Proyecto A publicable con curvas de Pareto (TPOT vs tok/s/GPU vs Calidad vs \$/1M).",
)

#concepto-clave[
  Optimizar velocidad sin medir degradación de tarea lleva a fallos silenciosos en producción (el modelo cuantizado genera rápido pero rompe esquemas JSON o ignora herramientas). Se combina `lm-evaluation-harness` / *EvalPlus* (`HumanEval+`, `MBPP+`, `gsm8k`) con un suite de validez de *tool-calling* y se construye la *frontera de Pareto* descartando configuraciones estrictamente dominadas.
]

#ejercicio-box[
  *Ejercicio B3.4-A (FinOps & Pareto):* Una H100 (`$3.00/h`) sirve Qwen-32B en BF16 a `420 tok/s` totales (`TPOT p99 = 38 ms`, `Tool Accuracy = 96.5%`), en FP8+KV-FP8 a `890 tok/s` (`TPOT p99 = 24 ms`, `Tool Accuracy = 96.0%`), y en AWQ 4-bit a `940 tok/s` (`TPOT p99 = 22 ms`, `Tool Accuracy = 91.2%`). Calcula el coste `$/1M tokens` de las tres opciones y razona cuál queda en la frontera de Pareto si el SLO exige `Tool Accuracy >= 95%`.
]

#solucion-box[
  - *Sol. B3.4-A:* Costes: BF16 = $3.00 / (420 times 3600) times 10^6 = \$1.98 / "1M"$; FP8+KV-FP8 = $3.00 / (890 times 3600) times 10^6 = \$0.94 / "1M"$; AWQ = $\$0.89 / "1M"$. Aunque AWQ es ligeramente más barato, incumple el umbral de calidad (`91.2% < 95%`), por lo que la solución óptima en la frontera factible de Pareto es *FP8 W8A8 + KV-FP8* (`$0.94 / 1M`, *2.1x* más barata que BF16 con caída de solo 0.5 pp).
]

#fuentes-box[
  1. Hugging Face Quantization Guide & `vllm-project/llm-compressor`: `https://huggingface.co/docs/transformers/quantization/overview` · `https://github.com/vllm-project/llm-compressor`
  2. vLLM Quantization & Speculative Decoding Docs: `https://docs.vllm.ai/en/latest/features/quantization/` · `https://docs.vllm.ai/en/latest/features/spec_decode.html`
  3. NVIDIA Technical Blog — *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference*.
  4. Hugging Face — *Assisted Generation / Speculative Decoding*: `https://huggingface.co/blog/assisted-generation`
  5. EleutherAI `lm-evaluation-harness` & EvalPlus: `https://github.com/EleutherAI/lm-evaluation-harness` · `https://github.com/evalplus/evalplus`
]
