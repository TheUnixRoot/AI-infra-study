#import "../theme.typ": *

= Curso 3 · Módulos E0–E1: Infraestructura Edge, Telemetría `mWh/token` y Línea Base

#block-banner(
  "E0–E1",
  "Compilación llama.cpp/PrismML, Medidor mWh/token y Baseline Qwen3.5-2B vs Bonsai 2",
  "Extensión B3-Edge (0 EUR): base empírica sobre GTX 1050 (sm_61) y móvil Android Termux",
  "8 sprints de 20m (H0 + H1)",
)[
  Construye un arnés reproducible de telemetría energética física en PC (`NVML` + `RAPL`) y Android (`sysfs` batería) y mide el suelo (`TQ1_0` *naive*) frente al techo real (`Ternary Bonsai 2 2B`) sobre la misma arquitectura base `Qwen3.5-2B`.
]

#infografia(
  "Flujo de Telemetría Física mWh/token (H0) y Techo Roofline GTX 1050 (H1)",
  "Izquierda: protocolo de calibración anti-ruido para aislar la potencia neta del modelo. Derecha: límite teórico decode B=1 en GTX 1050 (84 GB/s prácticos) según tamaño de pesos.",
  height: 185pt,
)[
  // Izquierda: flujo telemetría
  #place(dx: 8pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[Protocolo Energético H0 (Varianza < 5%)])
  #diag-node(8pt, 20pt, 195pt, 34pt, "1. Control de Entorno + Calibración 5s", sub: "Pantalla off · Modo avión · Auto-detect uA/mA y signo", fill: bg-light, col: slate)
  #arrow-d(105pt, 54pt, 66pt, col: slate)
  #diag-node(8pt, 66pt, 195pt, 34pt, "2. Línea Base Idle (10–15 s)", sub: "Registra P_idle (W) antes de lanzar llama-bench", fill: bg-navy, col: navy)
  #arrow-d(105pt, 100pt, 112pt, col: navy)
  #diag-node(8pt, 112pt, 195pt, 60pt, "3. Muestreo 100 ms & Potencia Neta", sub: "P_net = P_run - P_idle (W)\nmWh/token = P_net / (3.6 * TPS)\n3 repeticiones -> bench/results/test_h0.csv", fill: bg-teal, col: teal)

  // Separador
  #place(dx: 215pt, dy: 6pt, line(length: 172pt, angle: 90deg, stroke: 0.5pt + border-gray))

  // Derecha: barras Roofline GTX 1050
  #place(dx: 226pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[Techo Teórico Decode B=1 (GTX 1050 · ~84 GB/s)])
  #place(dx: 226pt, dy: 18pt, text(fill: muted, size: 7.2pt)[`TPS_max = BW_practica / Bytes_modelo` (`112 GB/s * 0.75 = 84 GB/s`):])
  #hbar(226pt, 34pt, "F16 (~3.80 GB)", "OOM VRAM (CPU)", 0.08, max-w: 112pt, h: 12pt, col: crimson)
  #hbar(226pt, 54pt, "Q4_K_M (~1.30 GB)", "~64.6 tok/s techo", 64.6 / 186.7, max-w: 112pt, h: 12pt, col: amber)
  #hbar(226pt, 74pt, "IQ2_XXS (~0.75 GB)", "~112.0 tok/s techo", 112.0 / 186.7, max-w: 112pt, h: 12pt, col: blue)
  #hbar(226pt, 94pt, "TQ1_0 (~0.55 GB)", "~152.7 tok/s techo", 152.7 / 186.7, max-w: 112pt, h: 12pt, col: teal)
  #hbar(226pt, 114pt, "PTQ1_0 Bonsai (~0.45 GB)", "~186.7 tok/s techo", 1.0, max-w: 112pt, h: 12pt, col: navy)

  #place(dx: 226pt, dy: 136pt, box(width: 240pt, fill: bg-amber, stroke: 0.5pt + amber, inset: 5pt, radius: 3pt)[
    #set text(size: 7.1pt, fill: ink)
    *Diagnóstico H1:* Si `tg128` real cae `< 60%` del techo, el freno no es GDDR5 sino desempaquetado en registros, *launch latency* o `FHT`.
  ])
]

== Módulo E0 · Infraestructura Base y Telemetría Energética (`H0`)

#unit-header(
  "E0",
  "Compilación llama.cpp / PrismML y Medidor mWh/token",
  "[LOC-PC] 2s + [LOC-AND] 1s + [AGENT] 1s",
  "B1.1–B2.1",
  "E1",
  "PC (GTX 1050 CUDA/Vulkan) + Android Termux",
  "Binarios upstream y prism listos; test_h0.csv con varianza < 5% entre 3 repeticiones.",
)

#concepto-clave[
  1. *Doble Binario (`upstream` vs `PrismML-llama.cpp`):* `llama.cpp` *upstream* (`b11556+`) aporta soporte CUDA (`sm_61`), Vulkan (`ggml-vulkan`), rotación Hadamard y cuantizaciones estándar (`Q4_K_M`, `IQ2_XXS`, `TQ1_0`); el fork `PrismML-llama.cpp` (rama `prism`) es necesario en `E0.2` para ejecutar los tensores empaquetados propietarios `PTQ1_0` y `PQ2_0` de `Ternary Bonsai 2`.
  2. *Calibración `sysfs` en Android (`E0.4`):* No todos los kernels Linux/Android exponen la misma polaridad ni unidad en `/sys/class/power_supply/battery/`. Algunos reportan `current_now` en microamperios ($mu"A"$) y otros en miliamperios ($"mA"$); unos usan signo negativo al descargar y otros positivo. El muestreador `measure_energy_and.sh` ejecuta 5 s de calibración inicial para detectar escala y signo antes de medir 10 s de *idle* y muestrear cada `100 ms`.
  3. *Conversión Física a `mWh/token`:* Dado $1 "Wh" = 3600 "J"$, resulta $1 "mWh" = 3.6 "J"$. Con potencia neta $P_"net" = P_"run" - P_"idle"$ en vatios durante $t$ segundos generando $N$ tokens:
     $ "TPS" = frac(N, t), quad E_"mWh" = frac(P_"net" dot t, 3.6), quad eta_E = frac(E_"mWh", N) = frac(P_"net", 3.6 dot "TPS") ["mWh/token"] $
]

#practica-box(title: "Sprints E0.1–E0.4 (Hito H0)")[
  - *E0.1 `[LOC-PC]`:* Inicializar `edge-ternary-lab`, compilar `llama.cpp` con `-DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_VULKAN=ON` y comprobar `CUDA0` + `Vulkan0` en `llama-bench`.
  - *E0.2 `[LOC-PC]`:* Clonar y compilar `PrismML-llama.cpp` (rama `prism`) en ruta separada.
  - *E0.3 `[LOC-AND]`:* Compilar `llama.cpp` en Termux (F-Droid) con backend CPU ARM64 NEON (y probar ICD Vulkan si `/dev/kgsl-3d0` o driver está accesible).
  - *E0.4 `[AGENT]` + `[LOC-AND]`:* Implementar `bench/measure_energy_and.sh` y `bench/measure_energy_pc.py` (`nvidia-smi` + `intel-rapl`), validar `--dry-run` unitario y capturar `bench/results/test_h0.csv` (varianza `< 5%`).
]

== Módulo E1 · Línea Base: `Qwen3.5-2B` vs. `Ternary Bonsai 2 2B` (`H1`)

#unit-header(
  "E1",
  "Suelo (llama.cpp estándar) vs Techo (PrismML Bonsai 2)",
  "[LOC-PC] 3s + [LOC-AND] 1s",
  "E0",
  "E2",
  "GTX 1050 (2 GB) + CPU + Android Termux",
  "docs/02_bonsai_vs_qwen.md con PPL, KLD vs F16, tok/s y mWh/token reales.",
)

#concepto-clave[
  1. *Suelo vs Techo sobre la misma arquitectura (`Qwen3.5-2B`):* Comparar `F16` (~3.8 GB en CPU/RAM), `Q4_K_M` (~1.3 GB), `IQ2_XXS` (~0.75 GB) y `TQ1_0` (~0.55 GB) frente a `Ternary-Bonsai-2-2B` (`Q2_0`, `PQ2_0`, `PTQ1_0`) aísla exactamente cuánta mejora aporta la rotación Hadamard + optimización de PrismML frente a una cuantización ternaria *naive* (`TQ1_0`).
  2. *Coste Real en Bits por Peso (`bpp_eff`):* Empaquetar 5 trits por byte en base 3 ($3^5 = 243 <= 256$) cuesta $8/5 = 1.60 "bits/peso"$ nominales (o `2.00` bits en `TQ2_0`), a los que se suma 1 escala FP16 (`16 bits`) cada $g$ pesos:
     $ "bpp"_"eff" = b_"peso" + frac(16, g) $
]

#ejercicio-box[
  *Ejercicio E0-A (Energía neta):* Un móvil descarga a $P_"net" = 4.2 "W"$ durante $t = 12.8 "s"$ generando $N = 256 "tokens"$. Calcula TPS, energía en J y mWh, y `mWh/token`. Explica por qué hay que medir con pantalla apagada, modo avión y línea base *idle* de 15 s.

  *Ejercicio E1-A (Roofline GTX 1050):* Con $"BW"_"peak" = 112 "GB/s"$ ($75\% approx 84 "GB/s"$ prácticos), calcula el techo decode ($B=1$) para `Q4_K_M` (`1.30 GB`) vs `PTQ1_0` (`0.45 GB`).

  *Ejercicio E1-B (Group Size `g=128` vs `g=32`):* Calcula `"bpp"_"eff"` para empaquetado base-3 (`1.6 bpp`) y `TQ2_0` (`2.0 bpp`) con `g=128` y `g=32`, y cuánto crece un modelo de `2B` parámetros al bajar a `g=32`.
]

#solucion-box[
  - *Sol. E0-A:* $"TPS" = 256 / 12.8 = 20.0 "tok/s"$. $E = 4.2 times 12.8 = 53.76 "J" = 14.933 "mWh"$. Eficiencia = $14.933 / 256 = 0.0583 "mWh/token"$. Sin pantalla fija/apagada ni modo avión, el consumo variable de OLED (`1.5–3.5 W`) y radios celulares contamina la medida superando la señal del LLM.
  - *Sol. E1-A:* `Q4_K_M`: $84 / 1.30 approx 64.6 "tok/s"$; `PTQ1_0`: $84 / 0.45 approx 186.7 "tok/s"$ (*2.89x*). Quedar `< 60%` indica cuello en desempaquetado, *launch overhead* o `FHT`.
  - *Sol. E1-B:* Base-3 `g=128`: $1.6 + 16/128 = 1.725 "bpp"$ (`~431 MB` en 2B; `1.75 bpp` alineado a 28 B); `g=32`: $1.6 + 16/32 = 2.10 "bpp"$ (`~525 MB`, `+94 MB` = *+21.7%*). `TQ2_0` `g=128`: `2.125 bpp` (`531 MB`) vs `g=32`: `2.500 bpp` (`625 MB`, `+94 MB`).
]

#fuentes-box[
  1. `llama.cpp` (`llama-bench`, `llama-perplexity --kl-divergence-base`): `https://github.com/ggml-org/llama.cpp`
  2. PrismML Ternary Bonsai (`PrismML-llama.cpp`): `https://github.com/PrismML`
  3. Linux Kernel `power_supply` sysfs ABI: `https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-class-power`
]
