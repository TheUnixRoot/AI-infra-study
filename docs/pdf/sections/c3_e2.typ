#import "../theme.typ": *

= Curso 3 · Módulo E2: Matemáticas y PTQ Sub-2-Bit (Base-3, Hadamard y GPTQ)

#block-banner(
  "E2",
  "Empaquetado TQ1_0/TQ2_0, Fast Hadamard Transform (FHT) y GPTQ en CPU",
  "Extensión B3-Edge (0 EUR): réplica matemática open-source del pipeline PTQ de Bonsai 2",
  "6 sprints de 20m (H2)",
)[
  Implementa en Python puro sobre la CPU/RAM local (`< 10 GB` RAM, capa a capa) las tres técnicas que evitan el colapso bajo 2 bits: rotación ortogonal *Fast Hadamard Transform*, compensación de segundo orden *GPTQ* por bloques `g128` y empaquetado ternario base-3 sin divisiones.
]

#infografia(
  "Pipeline PTQ Sub-2-Bit Capa a Capa (H2): Rotación Hadamard + GPTQ Hessiano + Base-3",
  "La rotación FHT dispersa outliers preservando el producto escalar; GPTQ actualiza pesos restantes vía Cholesky de (2 X^T X + lambda I) invirtiendo ~20-30% de signos respecto a RTN.",
  height: 185pt,
)[
  #place(dx: 8pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[Transformación de Pesos y Activaciones por Capa (`< 10 GB` RAM en CPU)])
  #diag-node(8pt, 20pt, 102pt, 58pt, "1. W y X Originales", sub: "Curtosis alta (outliers)\nRTN directo colapsa\nPPL/KLD bajo 2 bits", fill: bg-red, col: crimson)
  #arrow-r(110pt, 49pt, 126pt, col: crimson)
  #diag-node(126pt, 20pt, 112pt, 58pt, "2. FHT O(N log N)", sub: "W' = W H · x' = x H\nBloques B = 512 / 1024\nMax |w| cae drásticamente", fill: bg-teal, col: teal)
  #arrow-r(238pt, 49pt, 254pt, col: teal)
  #diag-node(254pt, 20pt, 108pt, 58pt, "3. GPTQ g128 (CPU)", sub: "H = (2/N) X^T X + damp*I\nCholesky -> propaga error\n{-1, 0, +1} * scale_fp16", fill: bg-navy, col: navy)
  #arrow-r(362pt, 49pt, 378pt, col: navy)
  #diag-node(378pt, 20pt, 90pt, 58pt, "4. GGUF TQ1_0/2_0", sub: "5 trits/byte base-3\n(xi * 3) >> 8\n0 divisiones GPU", fill: bg-purple, col: purple)

  #place(dx: 8pt, dy: 90pt, line(length: 460pt, stroke: 0.5pt + border-gray))

  // Parte inferior: Decodificación branchless y fenómeno sign flips
  #place(dx: 8pt, dy: 96pt, text(fill: navy, weight: "bold", size: 8pt)[Mecánica Interna: Empaquetado Base-3 (`3^5 = 243 <= 256`) y Sign Flips GPTQ])
  #diag-node(8pt, 112pt, 224pt, 64pt,
    "Decodificación TQ1_0 sin `% 3` ni `/ 3`",
    sub: "Codificar 5 trits q in {0,1,2} -> b_base3 in [0..242]\nEscalar: b = floor((b_base3 * 256 + 242) / 243)\nBucle 5x: q_j = (xi * 3) >> 8; xi = (xi * 3) & 0xFF",
    fill: bg-light, col: slate)
  #diag-node(242pt, 112pt, 226pt, 64pt,
    "Fenómeno Empírico Bonsai (~20–30% Sign Flips)",
    sub: "Al proyectar a {-s, 0, +s}, el error delta_j es grande.\nGPTQ compensa columnas correladas en H^-1:\n20–30% de pesos cambian de signo frente a RTN puro.",
    fill: bg-amber, col: amber)
]

== Unidades E2.1–E2.6 · Del Packing Base-3 a `ternary_gptq.py`

#unit-header(
  "E2",
  "Bit-Packing Base-3, Fast Hadamard Transform y GPTQ CPU",
  "[MOB] 2s + [LOC-PC] 2s + [AGENT] 2s",
  "E1",
  "E3",
  "Móvil + PC CPU (16 GB+ RAM, < 10 GB pico)",
  "Tests Hadamard en verde; Qwen3.5-2B cuantizado en CPU bajando KLD vs TQ1_0 naive.",
)

#concepto-clave[
  1. *Empaquetado Ternario en `llama.cpp` (`block_tq1_0` y `block_tq2_0`):* En `TQ2_0` se empaquetan 4 pesos por byte (`2 bits/peso`, `{0 -> -1, 1 -> 0, 2 -> +1}`). En `TQ1_0`, dado $3^5 = 243 <= 256$, caben *5 pesos ternarios en 1 byte* (`1.6 bits/peso`), reduciendo ~20% extra el tráfico de memoria respecto a 2 bits.
  2. *Por qué la Rotación de Hadamard salva el Sub-2-Bit:* Una fila de pesos con un único *outlier* grande ($|w_"max"| >> "media"$) dispara la escala de grupo $s$ y colapsa los otros 127 pesos a `0`. Multiplicar por una matriz de Hadamard normalizada $H$ ($H^T H = I$) redistribuye la energía entre los $B$ canales convirtiendo picos en una distribución casi gaussiana sin sesgos, manteniendo exacto el producto escalar gracias a que en inferencia se rota la activación $tilde(x) = x H$:
     $ tilde(x) tilde(W)^T = (x H)(W H)^T = x (H H^T) W^T = x W^T $
  3. *Compensación Hessiana GPTQ Capa a Capa (`< 10 GB` RAM):* Cargar `Qwen3.5-2B` completo en FP32 superaría 8–10 GB más activaciones; por ello `ternary_gptq.py` procesa un bloque Transformer cada vez en CPU (`FP32`/`BF16`), acumula $H = frac(2, N) X^T X + lambda "diag_mean" I$, calcula su descomposición de Cholesky y propaga el error de cuantización por bloques de `group_size=128` con escala `scale_fp16 = mean(|W_g|).half()`.
]

#practica-box(title: "Sprints E2.1–E2.6 (Hito H2)")[
  - *E2.1–E2.2 `[MOB]`:* Analizar `ggml-quants.c` (`block_tq1_0`, `block_tq2_0`), invarianza $W' = W H$ y actualización OBQ/GPTQ con $H^(-1)$.
  - *E2.3 `[LOC-PC]`:* Baseline rápido `llama-imatrix` con corpus mixto (Wikipedia + Python/C++ + JSON tools) cuantizando `Qwen3.5-2B` a `TQ1_0`, `TQ2_0` e `IQ1_M`.
  - *E2.4 `[AGENT]` + `[LOC-PC]`:* Crear `quant/hadamard.py` (`block_hadamard_transform` vectorizada $O(N log N)$) y `tests/test_hadamard.py` (involución `< 1e-5`, producto escalar y caída de outliers).
  - *E2.5–E2.6 `[AGENT]` + `[LOC-PC]`:* Implementar `quant/ternary_gptq.py` + `quant/export_gguf.py`, verificar MSE y ~20–30% *sign flips* vs RTN, exportar a GGUF y comparar PPL/KLD contra `Bonsai 2 2B`.
]

#ejercicio-box[
  *Ejercicio E2-A (Empaquetado Base-3 sin divisiones):* Codifica los 5 pesos `[-1, +1, 0, -1, +1]` en base 3 MSB-first ($b_"base3" = 81 q_0 + 27 q_1 + 9 q_2 + 3 q_3 + q_4$), escálalo a `[0, 255]` y decodifícalo usando solo `* 3` y `>> 8`.

  *Ejercicio E2-B (Coste de FHT):* Compara el número de operaciones de una `FHT` de tamaño $N = 1024$ ($N log_2 N$) frente a una `GEMV` de $1024 times 1024$.
]

#solucion-box[
  - *Sol. E2-A:* $q = [0, 2, 1, 0, 2] arrow.r b_"base3" = 27(2) + 9(1) + 2 = 65$. Escalado: $floor((65 times 256 + 242)/243) = 69$. Decodificación: `69*3=207 -> >>8 = 0 (-1)`, `207*3=621 -> 2 (+1) rem 109`, `109*3=327 -> 1 (0) rem 71`, `71*3=213 -> 0 (-1)`, `213*3=639 -> 2 (+1)`. Cero divisiones.
  - *Sol. E2-B:* FHT consume $1024 times 10 = 10240$ sumas/restas frente a $2 times 1024^2 = 2097152$ ops de la GEMV (`~0.49%` del cómputo), perfectamente amortizable en registros / *shared memory*.
]

#fuentes-box[
  1. Frantar et al. — *GPTQ: Accurate Post-Training Quantization*: `https://arxiv.org/abs/2210.17323`
  2. Ashkboos et al. — *QuaRot: Outlier-Free Inference via Hadamard*: `https://arxiv.org/abs/2404.00456`
  3. `llama.cpp` `ggml-quants.c` (`block_tq1_0` / `block_tq2_0`): `https://github.com/ggml-org/llama.cpp`
]
