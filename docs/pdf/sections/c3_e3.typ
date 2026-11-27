#import "../theme.typ": *

= Curso 3 · Módulo E3: Portabilidad a Gemma 4 E2B y Calidad Real

#block-banner(
  "E3",
  "Anatomía de Gemma 4 E2B (PLE, Visión/Audio), Evals GSM8K/IFEval y Pareto",
  "Extensión B3-Edge (0 EUR): cuantización selectiva < 1 GB y control estricto de regresiones",
  "5 sprints de 20m (H3)",
)[
  Porta el pipeline Hadamard + GPTQ de `Qwen3.5-2B` a `Gemma 4 E2B` protegiendo sus capas críticas (*Per-Layer Embeddings*, torres multimodales y `lm_head`), evalúa capacidad real (`GSM8K`, `IFEval`, JSON / *Tool Calling*, `<|think|>`) y despliega en Android con KV `q4_0`.
]

#infografia(
  "Anatomía Selectiva de Gemma 4 E2B y Presupuesto de Memoria en Móvil (< 1 GB)",
  "Separación estricta entre proyecciones densas del decoder (ternarizadas a ~1.73 bpp) y capas de vocabulario/multimodales protegidas (Q4_0/FP16) + KV-Cache de 4 bits.",
  height: 185pt,
)[
  // Izquierda: Mapa de tensores
  #place(dx: 8pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[Mapa de Cuantización Selectiva (`E3.1–E3.2`)])
  #diag-node(8pt, 20pt, 218pt, 48pt, "Ternarizar (FHT + GPTQ g128 ~1.73 bpp)", sub: "Decoder texto: q_proj, k_proj, v_proj, o_proj\ngate_proj, up_proj, down_proj", fill: bg-teal, col: teal)
  #diag-node(8pt, 76pt, 218pt, 54pt, "Proteger en Mayor Precisión (FP16 / Q4_0)", sub: "• Per-Layer Embeddings (PLE) & Token Embeds\n• vision_tower & audio_tower (encoders)\n• lm_head + RMSNorm / QK-Norm", fill: bg-red, col: crimson)
  #diag-node(8pt, 138pt, 218pt, 36pt, "KV-Cache 4-bit en Runtime (-ctk/-ctv q4_0)", sub: "Contexto 4096 tok en ~70-80 MB RAM móvil", fill: bg-navy, col: navy)

  // Separador
  #place(dx: 236pt, dy: 6pt, line(length: 172pt, angle: 90deg, stroke: 0.5pt + border-gray))

  // Derecha: Evaluación y Pareto
  #place(dx: 246pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[Suite Real E3.4 + Comparativa BF16 vs QAT])
  #diag-node(246pt, 20pt, 220pt, 46pt, "Doble Origen de Checkpoint (E3.3)", sub: "Comparar partida desde BF16 vs Gemma 4 QAT oficial\nExportar GGUF (< 1 GB) para llama.cpp", fill: bg-purple, col: purple)
  #diag-node(246pt, 74pt, 220pt, 52pt, "Más allá de PPL (Mini-Suite E3.4)", sub: "• 200 muestras GSM8K (razonamiento)\n• 200 muestras IFEval (seguimiento estricto)\n• 30 prompts JSON / Tool Calling + <|think|>", fill: bg-amber, col: amber)
  #diag-node(246pt, 134pt, 220pt, 40pt, "Frontera de Pareto H3", sub: "Eje X: MB + mWh/token · Eje Y: KLD + GSM8K/IFEval\nValidado en GTX 1050 y móvil Android", fill: bg-light, col: slate)
]

== Unidades E3.1–E3.5 · De `config.json` a Android `< 1 GB`

#unit-header(
  "E3",
  "Portabilidad Gemma 4 E2B, Reglas de Exclusión y Evals Funcionales",
  "[MOB] 1s + [LOC-PC] 3s + [LOC-AND] 1s",
  "E2",
  "E4, E6 (Bonus)",
  "PC CPU/GTX 1050 + Android Termux (-ctk q4_0 -ctv q4_0)",
  "Gemma-4-E2B-Ternary (< 1 GB) en frontera de Pareto real en GTX 1050 y Android.",
)

#concepto-clave[
  1. *Trampas Arquitectónicas de Gemma 4 E2B:* A diferencia de un Transformer plano tipo Qwen, Gemma 4 combina *Per-Layer Embeddings (PLE)*, normalizaciones `RMSNorm`/`QK-Norm` y ramas multimodales (`vision_tower`, `audio_tower`). Aplicar ternarización ciega sobre todo el `state_dict` destruye los *lookups* de embeddings e inutiliza la cabeza `lm_head`.
  2. *Origen BF16 vs Checkpoint QAT Oficial:* Partir del checkpoint `Gemma 4 QAT` oficial aporta pesos ya adaptados a tolerar ruido de cuantización antes de rotar con Hadamard y ejecutar `GPTQ`, reduciendo la brecha en tareas de seguimiento de instrucciones (`IFEval`).
  3. *Por qué PPL no basta (`<|think|>` y `<end_of_turn>`):* Un modelo puede subir apenas `+1.0` de perplejidad en WikiText-2 y aun así entrar en bucles infinitos al olvidar emitir `<end_of_turn>` o corromper llaves JSON en *tool calling*. La evaluación `E3.4` exige `GSM8K`, `IFEval` y 30 prompts propios de JSON/`<|think|>`.
]

#practica-box(title: "Sprints E3.1–E3.5 (Hito H3)")[
  - *E3.1–E3.2 `[MOB]`+`[LOC-PC]`:* Mapear tensores en `docs/03_gemma4_anatomy.md` y configurar exclusiones en `quant/ternary_gptq.py` (proteger `PLE`, `vision_tower`, `audio_tower`, `lm_head`).
  - *E3.3 `[LOC-PC]`:* Correr PTQ + FHT + GPTQ sobre Gemma 4 E2B (`BF16` vs `QAT`) y empaquetar GGUF (`< 1 GB`).
  - *E3.4 `[AGENT]`+`[LOC-PC]`:* Evaluar `Gemma-4-E2B-Ternary` vs `Gemma-4-E2B-Q4_0` vs `Bonsai 2 2B` en `GSM8K`, `IFEval` y JSON/`<|think|>`.
  - *E3.5 `[LOC-AND]`:* Desplegar en Android con `-ctk q4_0 -ctv q4_0` midiendo `tok/s`, RAM pico y `mWh/token`.
]

#ejercicio-box[
  *Ejercicio E3-A (Presupuesto en Móvil):* Si `2.0B` parámetros del *decoder* quedan en `1.725 bpp`, `0.45B` de PLE/embeddings/`lm_head` quedan en `Q4_0` (`4.5 bpp`), y el KV-Cache de `4096` tokens (`L=35, n_kv=2, d_head=256`) usa `q4_0` (`0.5 B/elem`), calcula el tamaño total y qué % ocupan las capas protegidas.

  *Ejercicio E3-B (Fallos Silenciosos):* Enumera 3 causas raíz cuando `PPL` apenas empeora (`< +1.5`) pero `IFEval` falla o hay bucle infinito al cerrar turno, y qué ablación confirma cada una.
]

#solucion-box[
  - *Sol. E3-A:* Decoder = $2.0 times 10^9 times 1.725 / 8 approx 431.3 "MB"$; PLE/`lm_head` = $0.45 times 10^9 times 4.5 / 8 approx 253.1 "MB"$; KV `q4_0` = $2 times 35 times 2 times 256 times 4096 times 0.5 approx 73.4 "MB"$. Total $approx 758 "MB"$ (`< 1 GB`). Las capas protegidas representan `~37%` del peso del GGUF; intentar ternarizarlas dispara la `KLD`.
  - *Sol. E3-B:* (1) Escala/cuantización agresiva de `lm_head` (ablación: subir `lm_head` a `Q8_0`/`FP16`); (2) calibración `X^T X` solo con prosa sin JSON ni delimitadores `<|think|>`/`<end_of_turn>` (ablación: mezclar 30% chat/JSON/código); (3) *outliers* en capas frontera `layers.0`/`layers.N-1` (ablación: dejar primera y última capa en `Q4_0`).
]

#fuentes-box[
  1. EleutherAI `lm-evaluation-harness`: `https://github.com/EleutherAI/lm-evaluation-harness`
  2. `llama.cpp` KV-Cache Quantization (`-ctk q4_0 -ctv q4_0`): `https://github.com/ggml-org/llama.cpp`
]
