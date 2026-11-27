#import "../theme.typ": *

= Curso 3 · Módulos E5–E6: Servidor `< 8 W`, *LogFit-Edge*, Dossier UMA y Bonus QAT

#block-banner(
  "E5–E6",
  "Despliegue Doméstico 24/7, Reparto Heterogéneo LogFit-Edge y Bonus QAT STE+KL",
  "Extensión B3-Edge + Puente HPAC/UMA: utilidad diaria real, portfolio público y dossier Rafael Asenjo",
  "4 + 3 sprints de 20m (H5 + H6)",
)[
  Convierte los avances del curso en un servidor local OpenAI `< 8 W` en casa, cuantifica el cruce de eficiencia CPU+GPU (*LogFit-Edge*), enlaza tu TFG (`TheUnixRoot/Logfit`) con `edge-ternary-lab` en un dossier de 1 página para Rafael Asenjo y explora opcionalmente la recuperación QAT mínima (`STE + KL`).
]

#infografia(
  "Evolución LogFit -> LogFit-Edge, Servidor Doméstico < 8 W y Bonus QAT STE+KL",
  "Consolidación final: puente directo desde el TFG oneTBB hacia kernels portables Vulkan/CUDA sobre LLMs sub-2-bit y despliegue doméstico de coste eléctrico mínimo.",
  height: 180pt,
)[
  #diag-node(8pt, 14pt, 136pt, 60pt, "TheUnixRoot/Logfit (TFG)", sub: "Reparto heterogéneo CPU+GPU\noriginal sobre oneTBB\n(Base académica HPAC/UMA)", fill: bg-light, col: slate)
  #arrow-r(144pt, 44pt, 162pt, col: slate)
  #diag-node(162pt, 14pt, 146pt, 60pt, "LogFit-Edge (E5.2)", sub: "Embeddings/PLE/Draft en CPU NEON\n+ GEMV ternario en Vulkan GPU\nCruce óptimo por long. prompt", fill: bg-teal, col: teal)
  #arrow-r(308pt, 44pt, 326pt, col: teal)
  #diag-node(326pt, 14pt, 142pt, 60pt, "Dossier UMA 1 pág. (E5.4)", sub: "Qwen3.5-2B & Gemma 4 E2B\nmWh/tok + Kernels Vulkan/CUDA\nListo para Rafael Asenjo", fill: bg-navy, col: navy)

  #place(dx: 8pt, dy: 86pt, line(length: 460pt, stroke: 0.5pt + border-gray))

  #diag-node(8pt, 98pt, 224pt, 72pt, "Servidor Doméstico 24/7 (E5.1 · H5)", sub: "• deploy/local_server.sh (llama-server + KV q4_0)\n• Móvil < 8 W (~3-6 EUR/año) vs PC torre (> 100 EUR/año)\n• Endpoint OpenAI local para terminal, código y resúmenes", fill: bg-amber, col: amber)
  #diag-node(242pt, 98pt, 226pt, 72pt, "Módulo E6 Bonus (Colab/Kaggle T4 0 EUR · H6)", sub: "• Pesos latentes obligatorios en FP32 (evitar underflow)\n• TernaryLinearG128 STE + pérdida KL vs profesor FP16\n• 1M–5M tokens -> salto JSON/Tools (tipo Bonsai 2.5)", fill: bg-purple, col: purple)
]

== Módulo E5 · Ahorro Local (`< 8 W`), *LogFit-Edge* y Dossier UMA (`H5`)

#unit-header(
  "E5",
  "Servidor 24/7, Reparto Heterogéneo y Dossier HPAC/UMA",
  "[LOC-AND] 1.5s + [LOC-PC] 1s + [MOB] 1.5s",
  "E4",
  "Portfolio + Dossier Rafael Asenjo",
  "Móvil Android (< 8 W) + PC local",
  "API local < 8 W funcionando, Logfit enlazado y docs/04_dossier_uma.md (1 pág.).",
)

#practica-box(title: "Sprints E5.1–E5.4 (Hito H5)")[
  - *E5.1 `[LOC-AND]`:* Configurar `deploy/local_server.sh` con `llama-server` + modelo ganador + KV `q4_0` en LAN como endpoint OpenAI; medir vatios sostenidos y calcular factura en `€/año`.
  - *E5.2 `[LOC-PC]`+`[LOC-AND]`:* Experimento *LogFit-Edge*: comparar `tok/s` y `mWh/token` ejecutando todo en GPU vs dejar embeddings/`PLE`/capas concretas o *draft* MTP en CPU NEON/`x86`, trazando la curva de cruce por longitud de prompt.
  - *E5.3 `[MOB]`+`[LOC-PC]`:* Actualizar el `README.md` de `TheUnixRoot/Logfit` (3 párrafos sobre `oneTBB` + contexto TFG + enlace a `edge-ternary-lab`).
  - *E5.4 `[MOB]`:* Redactar `docs/04_dossier_uma.md` (exactamente 1 página): cuello de botella de memoria/`mWh/tok`, resultados reales `Qwen3.5-2B` y `Gemma 4 E2B` y contribución en kernels Vulkan/CUDA.
]

== Módulo E6 (Bonus Opcional) · QAT Mínimo con STE + Destilación KL (`H6`)

#unit-header(
  "E6",
  "Recuperación QAT Ligera (1M–5M tokens) con STE y Pérdida KL",
  "[MOB] 1s + [AGENT] 1s + [LOC-PC] 1s",
  "E3",
  "Comparativa Bonsai 2 (PTQ) vs Bonsai 2.5 (QAT)",
  "Google Colab / Kaggle T4 gratuita (16 GB)",
  "Medir mejora real de KLD y JSON/Tool Calling tras destilación KL mínima.",
)

#concepto-clave[
  - *Pesos Latentes en FP32:* Durante QAT ternario con *Straight-Through Estimator (STE)*, los pasos de gradiente pequeños ($eta dot g approx 10^(-6)$) sufrirían *underflow* a cero en `BF16` (7 bits de mantisa) antes de cruzar los umbrales de redondeo hacia `{-1, 0, +1}`. Mantener los pesos maestros en `FP32` y vigilar el % de *code flips* por paso es obligatorio para estabilizar la recuperación.
  - *Destilación KL con Adafactor + Gradient Checkpointing:* Permite entrenar `1M–5M` tokens sobre `Gemma 4 E2B` dentro de los `16 GB` gratuitos de una `T4` minimizando $D_"KL"(P_"FP16" parallel P_"QAT")$ y midiendo cuánto sube la validez de JSON / *Tool Calling*.
]

#ejercicio-box[
  *Ejercicio E5-A (Factura eléctrica 24/7):* Compara un nodo móvil (`2.0 W` medios 24/7; `5–7 W` bajo carga) frente a un PC antiguo (`75 W` medios 24/7) durante 8,760 h/año a `0.18 €/kWh`.

  *Ejercicio E6-A (Por qué FP32 latente):* Con `BF16` ($epsilon approx 7.8 times 10^(-3)$) y actualización $Delta w = 10^(-6)$ sobre $w = 0.5$, explica qué ocurre al optimizar y qué indica la tasa de *code flips*.
]

#solucion-box[
  - *Sol. E5-A:* Móvil: $0.002 times 8760 times 0.18 = 3.15 "€/año"$ (`~0.26 €/mes`); PC torre: $0.075 times 8760 times 0.18 = 118.26 "€/año"$ (*37.5x* más caro en electricidad).
  - *Sol. E6-A:* En `BF16` el incremento $10^(-6)$ es menor que el medio ULP alrededor de $0.5$ (`0.0039/2`), por lo que $w + Delta w$ se redondea a $w$ y el peso nunca cruza el umbral ternario. En `FP32` se acumula sin pérdida; los *code flips* miden qué fracción de pesos cambia de símbolo `{-1, 0, +1}` en cada paso para verificar convergencia sin oscilación.
]

#fuentes-box[
  1. Bengio et al. — *Estimating or Propagating Gradients Through Stochastic Neurons (STE)*: `https://arxiv.org/abs/1308.3432`
  2. `llama.cpp` `llama-server` (API OpenAI local): `https://github.com/ggml-org/llama.cpp/tree/master/tools/server`
]
