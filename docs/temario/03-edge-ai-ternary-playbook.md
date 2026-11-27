# Edge-AI Sub-2-Bit & Portable Kernels — Master Playbook (`Curso 3 · B3-Edge`)

> **Propósito:** Replicar con herramientas 100 % *open-source* y presupuesto **0 €** el rendimiento, la compresión sub-2-bit y la eficiencia energética (`mWh/token`) de la familia **Ternary Bonsai (PrismML)**, portar la metodología a **Gemma 4 E2B**, construir *kernels* portables desde cero en **CUDA (`sm_61`)** y **Vulkan Compute (GLSL/SPIR-V)**, y desplegar inferencia local de ultrabajo consumo (`< 8 W`) en casa.
>
> **Alineación estratégica:** Extensión práctica (`B3-Edge`) totalmente compatible con [Curso 1 · AI Infrastructure (`00-master-playbook.md`)](00-master-playbook.md) —refuerza el perfil de **Senior Inference MLE**— y puente directo hacia un dossier técnico presentable a **Rafael Asenjo (HPAC/UMA)** como evolución natural de `LogFit` (de `oneTBB` a *kernels* portables CPU/GPU heterogéneos sobre `ggml-vulkan`).

---

## 0. Filosofía Operativa («Modo Bebé») y Entorno de Hardware

### Reglas de ejecución

- **Unidad atómica:** Micro-sprints de **20 minutos** con un único objetivo verificable y **1 commit o anotación cerrada** al terminar.
- **Regla de no bloqueo:** Si una tarea requiere más de 20 minutos de *boilerplate* (scripts de medición, parsing de logs, *harnesses* de test), se delega al agente de código mediante el **Prompt Pre-Flight** incluido en cada módulo.
- **Taxonomía de sprints:**
  - `[MOB]` — Lectura técnica, matemáticas en papel/notas o diseño de *kernels* desde el móvil (sin PC).
  - `[LOC-PC]` — Ejecución en PC local (CPU `x86_64` + NVIDIA GTX 1050 `2 GB` VRAM).
  - `[LOC-AND]` — Ejecución y telemetría real en móvil Android (`Termux` + ARM64 NEON + GPU Adreno/Mali Vulkan).
  - `[AGENT]` — Delegación controlada al agente de código con **tests unitarios obligatorios**.

### Perfil de hardware disponible (0 €)

| Dispositivo | Arquitectura / Cómputo | Memoria y Ancho de Banda | Rol en el Playbook |
| :--- | :--- | :--- | :--- |
| **PC Local (GPU)** | NVIDIA GTX 1050 (Pascal `sm_61`, 640 cores CUDA, soporta `__dp4a`, sin Tensor Cores) | `2 GB` GDDR5 (`~112 GB/s` teóricos) | Desarrollo y profiling (`nsys`) de kernels GEMV ternarios (`sm_61`) y Vulkan; ejecución completa de modelos 2B sub-2-bit (`< 1 GB` VRAM). |
| **PC Local (CPU)** | `x86_64` + RAM del sistema | `16 GB+` RAM | Ejecución de PTQ + rotación Hadamard + GPTQ capa a capa en Python sobre modelos de 2B–4B sin necesidad de GPU cloud. |
| **Móvil Android** | SoC ARM64 (`NEON` / `dotprod`) + GPU Adreno/Mali | Memoria unificada / compartida | Benchmarking real de `mWh/token`, despliegue de servidor local de ultrabajo consumo (`3–8 W`) y validación de *compute shaders* Vulkan *zero-copy*. |

### Prerrequisitos ya cubiertos en `AI-infra-study` (Curso 1)

- ✅ **B1.1:** Modelo Roofline, distinción *compute-bound* (*prefill*/GEMM) vs. *memory-bandwidth-bound* (*decode*/GEMV).
- ✅ **B1.2:** CUDA C++ en `sm_61`, accesos coalescentes, *shared memory tiling* y reducciones con `__shfl_down_sync`.
- ✅ **B2.1:** Cálculo exacto de tamaño de KV-cache y huella de memoria por secuencia.
- 🟡 **B1.3:** `nsys` y `perf` (se consolidan aquí sobre `GEMV` propio y `llama.cpp`).

### Plantillas de Control (Copiar y Pegar)

#### A. Prompt de Inicio / Fork de Chat de Módulo

```text
[FORK CURSO3-Ex]
Contexto: Lee `docs/temario/03-edge-ai-ternary-playbook.md` en las fuentes.
Módulo activo: [Ex] | Sprint actual: [Ex.y]
Estado previo (Handover): <pegar último bloque HANDOVER o "Inicio de módulo">
Tiempo disponible ahora: 20 minutos | Modo: [MOB / LOC-PC / LOC-AND / AGENT]
Objetivo de este micro-sprint: <ej. calibrar sysfs mWh/tok / implementar FHT bloque 512>
Regla: Respuestas concisas, legibles en móvil, sin relleno. Al final dame el bloque HANDOVER actualizado.
```

#### B. Bloque de Handover (Al cerrar cualquier micro-sprint de 20 min)

```yaml
HANDOVER:
  itinerario: "Curso 3 (Edge-AI Sub-2-Bit)"
  unidad_actual: "E0.4"
  modo: "LOC-AND" # MOB | LOC-PC | LOC-AND | AGENT
  estado: "en_progreso" # pendiente | en_progreso | done
  aprendido_o_verificado:
    - "Unidades sysfs verificadas (uA, descarga negativa) y varianza < 5% entre 3 corridas"
  artefacto_generado: "edge-ternary-lab/bench/results/test_h0.csv"
  bloqueos_o_dudas: "Pendiente validar ICD Vulkan en Termux"
  siguiente_micro_sprint_20m: "E1.1 [LOC-PC] - Descargar Qwen3.5-2B y Bonsai 2 2B GGUF"
```

---

## 1. Mapa General de Módulos, Sprints e Hitos

```text
[E0 Infra + Telemetría mWh/tok] ──► [E1 Baseline Qwen3.5-2B vs Bonsai 2]
                                              │
                                              ▼
                               [E2 Matemáticas + PTQ Hadamard/GPTQ]
                                              │
                                              ▼
                               [E3 Portabilidad Gemma 4 E2B + Evals]
                                      │                    │
                                      ▼                    ▼ (Opcional Colab T4)
                      [E4 Kernels CUDA sm_61 & Vulkan]   [E6 Bonus: QAT STE + KL]
                                      │
                                      ▼
                      [E5 API <8W + LogFit-Edge + Dossier UMA]
```

| Módulo | Foco principal | Sprints (20m) | Coste | Hito / Entregable verificable |
| :--- | :--- | :---: | :---: | :--- |
| **E0** | Infraestructura `llama.cpp`, Termux y telemetría `mWh/token` | 4 | 0 € | **H0:** Harness reproducible PC + Android midiendo `tok/s`, RAM/VRAM y `mWh/token`. |
| **E1** | Línea base: `Qwen3.5-2B` vs. `Ternary Bonsai 2 2B` | 4 | 0 € | **H1:** Informe Roofline + `KLD` + `PPL` + energía (suelo y techo real de PrismML). |
| **E2** | Teoría y PTQ sub-2-bit: *bit-packing*, Hadamard (`FHT`) y `GPTQ` | 6 | 0 € | **H2:** Cuantizador Python CPU (`PTQ` + Hadamard `g128`) sobre `Qwen3.5-2B` acercándose a Bonsai 2. |
| **E3** | Portabilidad a `Gemma 4 E2B` y evaluación de calidad real | 5 | 0 € | **H3:** `Gemma-4-E2B-Ternary` (`< 1 GB`) evaluado en frontera de Pareto (`KLD`, `IFEval`, `GSM8K`, `mWh/token`). |
| **E4** | *Kernels* desde cero: `GEMV` ternario + `FHT` en CUDA `sm_61` y Vulkan | 7 | 0 € | **H4:** Kernel `GEMV` ternario en CUDA y *compute shader* Vulkan corriendo en GTX 1050 + móvil (candidato a PR en `llama.cpp`). |
| **E5** | Ahorro local (`< 8 W`), reparto heterogéneo y dossier técnico | 4 | 0 € | **H5:** API local 24/7 en casa + repo público documentado + Dossier de 1 página para Rafael Asenjo. |
| **E6 (Bonus)** | Recuperación QAT mínima con STE + Destilación KL | 3 | 0 € | **H6:** Experimento QAT sobre un bloque/capa o 10M tokens en Colab/Kaggle gratis para medir salto tipo Bonsai 2.5. |

---

## 2. Estructura del Repositorio de Portfolio (`edge-ternary-lab`)

Crea este repositorio independiente desde el **Sprint E0.1** para mantener todo ordenado, reproducible y listo para portfolio (análogo a `agentic-platform-golden-path` en el Curso 2):

```text
edge-ternary-lab/
├── README.md                  # Tabla maestra de resultados (frontera de Pareto)
├── docs/
│   ├── 01_roofline_edge.md    # Cálculos teóricos GTX 1050 y móvil
│   ├── 02_bonsai_vs_qwen.md   # Análisis Hito H1
│   ├── 03_gemma4_anatomy.md   # Capas críticas de Gemma 4 E2B
│   └── 04_dossier_uma.md      # Resumen 1 pág. (evolución LogFit -> Edge Ternary)
├── bench/
│   ├── measure_energy_pc.py   # Wrapper NVML (GTX 1050) + Intel/AMD RAPL
│   ├── measure_energy_and.sh  # Muestreador sysfs Android (Termux) -> mWh/token
│   ├── run_pareto.py          # Orquesta llama-bench + llama-perplexity + KLD
│   └── results/               # CSVs/JSONs crudos por dispositivo y fecha
├── quant/
│   ├── hadamard.py            # Fast Hadamard Transform por bloques (512/1024)
│   ├── ternary_gptq.py        # PTQ ternario g128 con compensación Hessiana en CPU
│   ├── export_gguf.py         # Empaquetado a TQ1_0 / TQ2_0 o GGUF compatible
│   └── tests/                 # Tests de ortogonalidad (H*H^T = I) y error de reconstrucción
├── kernels/
│   ├── cuda_sm61/             # GEMV ternario g128 + FHT en CUDA Pascal
│   ├── vulkan_glsl/           # Compute shaders portables (GTX 1050 + Adreno/Mali)
│   └── bench_kernels.cpp      # Microbenchmark de ancho de banda efectivo (GB/s)
└── deploy/
    └── local_server.sh        # Arranque llama-server modo ahorro energético
```

---

## 3. Desarrollo Detallado por Módulos

### Módulo E0 — Infraestructura Base y Telemetría Energética (`mWh/token`)

**Objetivo:** Compilar `llama.cpp` *upstream* (`b11556+`, soporte Hadamard) y el *fork* `PrismML-llama.cpp` en PC y Android, y construir un medidor fiable de milivatios-hora por token (`mWh/token`).

#### Micro-sprints (`4 × 20 min`)

- `[ ]` **E0.1 `[LOC-PC]`:** Inicializar el repo `edge-ternary-lab`. Compilar `llama.cpp` *upstream* con soporte CUDA (`-DCMAKE_CUDA_ARCHITECTURES=61`) y Vulkan (`-DGGML_VULKAN=ON`) en tu PC. Verificar que `llama-bench` detecta ambos backends (`CUDA0` y `Vulkan0`).
- `[ ]` **E0.2 `[LOC-PC]`:** Clonar y compilar el fork `PrismML-llama.cpp` (rama `prism`) en un directorio separado para poder correr los formatos `PTQ1_0` y `PQ2_0` de Bonsai 2.
- `[ ]` **E0.3 `[LOC-AND]`:** En tu móvil Android con Termux (instalado desde F-Droid), compilar `llama.cpp` con `cmake` (primero backend CPU ARM NEON; después probar compilación con Vulkan/ICD de Adreno o Mali si el dispositivo expone `/dev/kgsl-3d0` o driver Vulkan en Termux).
- `[ ]` **E0.4 `[AGENT]` + `[LOC-AND]`:** Construir y calibrar los scripts `bench/measure_energy_pc.py` y `bench/measure_energy_and.sh`. Verificar las unidades reales de `/sys/class/power_supply/battery/current_now` y `voltage_now` en tu móvil (algunos fabricantes reportan en $\mu\text{A}$ y otros en $\text{mA}$, y el signo de descarga puede ser positivo o negativo).

#### Ejercicios de papel y boli (`[MOB]`)

1. **Fórmula de energía por token:** Si tu móvil descarga a una potencia media neta (restando *idle*) de $P_{\text{net}} = 4.2\text{ W}$ durante $t = 12.8\text{ s}$ generando $N = 256\text{ tokens}$, calcula:
   - *Throughput* ($\text{TPS} = N / t$).
   - Energía total en julios ($E_{\text{J}} = P_{\text{net}} \cdot t$) y en $\text{mWh}$ ($1\text{ mWh} = 3.6\text{ J}$).
   - Eficiencia energética en $\text{mWh/token}$:

   $$\eta_E = \frac{E_{\text{mWh}}}{N} = \frac{P_{\text{net}} \cdot t}{3.6 \cdot N} = \frac{P_{\text{net}}}{3.6 \cdot \text{TPS}}\quad [\text{mWh/token}]$$

2. **Error sistemático en móvil:** Explica por qué debes medir la potencia con la pantalla apagada o brillo mínimo fijo, en modo avión y restando una línea base *idle* de 15 segundos antes de cada corrida.

#### Prompt Pre-Flight (para Sprint E0.4)

```text
Actúa como ingeniero de sistemas Linux/Android. Crea dos scripts autocontenidos en `bench/`:
1. `measure_energy_and.sh` (POSIX sh/bash para Termux):
   - Auto-detecta las rutas en `/sys/class/power_supply/battery/{current_now,voltage_now}` y hace una lectura de calibración de 5s para detectar si las unidades están en uA/mA y el signo de descarga.
   - Mide 10s de potencia base (idle), lanza un comando pasado por argumentos (ej. `llama-bench ...`), muestrea potencia cada 100 ms en segundo plano mientras el proceso vive, calcula potencia media total y neta (W), y cruza con los tokens/s de salida para exportar una fila CSV con: timestamp, device, model, backend, pp_tps, tg_tps, power_idle_w, power_run_w, mwh_per_tok.
2. `measure_energy_pc.py` (Python 3):
   - Usa `nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits` (o pynvml) para GTX 1050 y `/sys/class/powercap/intel-rapl:0/energy_uj` (con fallback graceful si no hay permisos) para CPU.
   - Ejecuta el mismo flujo de calibración idle + muestreo + salida CSV idéntica.
Incluye un test en modo `--dry-run` que simule lecturas y verifique las conversiones de unidades (J -> mWh/tok).
```

#### Definición de Hecho — Hito H0

- Ambos binarios (`upstream` y `prism`) compilados en PC y al menos CPU NEON (idealmente + Vulkan) en Termux.
- `bench/results/test_h0.csv` contiene una medición real de potencia en *idle* y bajo carga sintética en PC y en móvil con **varianza $< 5\%$ entre 3 repeticiones**.

---

### Módulo E1 — Línea Base: `Qwen3.5-2B` vs. `Ternary Bonsai 2 2B`

**Objetivo:** Establecer el suelo (cuantizaciones estándar de `llama.cpp`) y el techo (modelo oficial de PrismML) sobre exactamente la misma arquitectura base (`Qwen3.5-2B`) en tu hardware.

#### Micro-sprints (`4 × 20 min`)

- `[ ]` **E1.1 `[LOC-PC]`:** Descargar los pesos GGUF de `Qwen3.5-2B` en `F16` (`~3.8 GB`, para referencia en CPU/RAM), `Q4_K_M` (`~1.3 GB`), `IQ2_XXS` (`~0.75 GB`) y `TQ1_0` (`~0.55 GB`), junto con `Ternary-Bonsai-2-2B-gguf` (`Q2_0`, `PQ2_0`, `PTQ1_0`).
- `[ ]` **E1.2 `[LOC-PC]`:** Ejecutar `llama-bench -p 512 -n 128` sobre la GTX 1050 (`-ngl 99` en CUDA y en Vulkan) para todos los modelos que quepan en `2 GB` VRAM. Registrar `pp512`, `tg128`, uso de VRAM y `mWh/token`.
- `[ ]` **E1.3 `[LOC-AND]`:** Pasar los 4 modelos más pequeños al móvil y correr el mismo benchmark en Termux (CPU NEON vs. Vulkan) midiendo temperatura de batería, `tok/s` y `mWh/token`.
- `[ ]` **E1.4 `[LOC-PC]`:** Medir Perplejidad (`PPL`) en `wikitext-2-raw` y Divergencia KL (`KLD`) contra `Qwen3.5-2B-F16` usando `llama-perplexity --kl-divergence-base`. Documentar los hallazgos en `docs/02_bonsai_vs_qwen.md`.

#### Ejercicios de papel y boli (`[MOB]`)

1. **Cálculo Roofline para GTX 1050 (`sm_61`):**
   - Ancho de banda teórico de la GTX 1050: $\text{BW}_{\text{peak}} = 112\text{ GB/s}$ (asume eficiencia práctica del $75\% \approx 84\text{ GB/s}$).
   - Para un modelo 2B en `Q4_K_M` (`~1.30 GB`) vs. `PTQ1_0` (`~0.45 GB`), calcula el techo teórico de tokens/s en fase *decode* ($B=1$):

   $$\text{TPS}_{\max} = \frac{\text{BW}_{\text{práctica}}}{\text{Bytes}_{\text{pesos}} + \text{Bytes}_{\text{KV/paso}}}$$

   - Compara ese techo teórico con el `tg128` que midas en `E1.2`. Si estás muy por debajo del $60\%$ de $\text{TPS}_{\max}$, identifica si el cuello de botella es el desempaquetado en registros, la latencia de lanzamiento de *kernels* o el cálculo de la transformada de Hadamard.

2. **Bits efectivos reales con `group_size = 128`:**
   - Si guardas 128 pesos en $1.58$ bits teóricos (empaquetados a $1.6\text{ bits/peso}$: 5 pesos por byte en base 3) más 1 escala FP16 (16 bits) por bloque, calcula el coste real en bits por peso:

   $$\text{bpp}_{\text{eff}} = \frac{g \cdot b_{\text{peso}} + b_{\text{escala}}}{g} = b_{\text{peso}} + \frac{16}{g}$$

   - Repite el cálculo para empaquetado de 2 bits por peso (`TQ2_0`, 4 pesos/byte) y para `group_size = 32`. ¿Cuánto crece el modelo de 2B parámetros si bajas el tamaño de grupo de 128 a 32?

#### Definición de Hecho — Hito H1

- Tabla completa en `docs/02_bonsai_vs_qwen.md` comparando `F16`, `Q4_K_M`, `IQ2_XXS`, `TQ1_0` (estándar) y `Bonsai 2 2B` en PC (CUDA + Vulkan) y Android (CPU + Vulkan).
- Sabes con números exactos cuánta ventaja real en `KLD` y en `mWh/token` tiene Bonsai 2 frente a un `TQ1_0` *naive*.

---

### Módulo E2 — Matemáticas y PTQ Sub-2-Bit: *Bit-Packing*, Hadamard (`FHT`) y `GPTQ`

**Objetivo:** Entender y programar en Python (ejecutable en la CPU/RAM de tu PC) las tres piezas matemáticas que evitan el colapso por debajo de 2 bits, y replicar la cuantización sobre `Qwen3.5-2B`.

#### Micro-sprints (`6 × 20 min`)

- `[ ]` **E2.1 `[MOB]`:** Estudiar el empaquetado ternario en `llama.cpp` (`ggml-quants.c`: estructuras `block_tq1_0` y `block_tq2_0`) y el mecanismo de rotación de Hadamard en el runtime de Bonsai 2 ($W' = W H$, y en inferencia $x' = x H$ de modo que $x' (W')^T = x H H^T W^T = x W^T$).
- `[ ]` **E2.2 `[MOB]`:** Estudiar la intuición matemática de GPTQ / OBQ (por qué cuantizar una columna de $W$ introduce un error $\delta_j$ que se compensa actualizando las columnas restantes mediante la inversa del Hessiano $H^{-1} = (2 X^T X + \lambda I)^{-1}$).
- `[ ]` **E2.3 `[LOC-PC]`:** *Baseline* rápido con `llama.cpp`: generar una matriz de importancia (`llama-imatrix`) usando un texto de calibración mixto (Wikipedia + código Python/C++ + JSON de *tool calling*) y cuantizar `Qwen3.5-2B-F16` a `TQ1_0`, `TQ2_0` e `IQ1_M`. Medir cuánto mejora la `KLD` respecto a E1 sin tocar aún Hadamard.
- `[ ]` **E2.4 `[AGENT]` + `[LOC-PC]`:** Implementar `quant/hadamard.py` (*Fast Hadamard Transform* por bloques de tamaño $B \in \{128, 512, 1024\}$ con normalización $1/\sqrt{B}$) y sus tests unitarios verificando que $H(H(x)) = x$ y que la curtosis (presencia de *outliers*) de las filas de pesos de `Qwen3.5-2B` cae drásticamente tras rotar.
- `[ ]` **E2.5 `[AGENT]` + `[LOC-PC]`:** Implementar `quant/ternary_gptq.py` (carga capa a capa de un modelo Hugging Face en CPU/FP32 para no superar los `16 GB` de RAM, aplica rotación Hadamard por bloques, acumula $X^T X$ con 128 secuencias de calibración y aplica GPTQ proyectando cada bloque de 128 pesos a $\{-s, 0, +s\}$ con escala FP16).
- `[ ]` **E2.6 `[LOC-PC]`:** Ejecutar `ternary_gptq.py` sobre `Qwen3.5-2B`, guardar el modelo resultante (descuantizado simulado para verificar PPL/KLD primero, y exportado a GGUF después vía `quant/export_gguf.py`) y compararlo cara a cara contra el `Ternary Bonsai 2 2B` oficial.

#### Ejercicios de papel y boli (`[MOB]`)

1. **Empaquetado en base 3 (`TQ1_0`):** Un byte tiene $2^8 = 256$ estados. Dado que $3^5 = 243 \le 256$, puedes empaquetar 5 pesos ternarios $q_0, q_1, q_2, q_3, q_4 \in \{0, 1, 2\}$ (representando $\{-1, 0, +1\}$ vía $q_k = w_k + 1$) en un solo byte mediante:

   $$b_{\text{base3}} = \sum_{k=0}^{4} q_k \cdot 3^{4-k} = 81 q_0 + 27 q_1 + 9 q_2 + 3 q_3 + q_4 \in [0, 242]$$

   - Codifica el vector de 5 pesos $[-1, +1, 0, -1, +1]$ en un único byte `uint8` (en representación pura $[0, 242]$ y escalada al rango $[0, 255]$).
   - Escribe el procedimiento inverso para decodificar los 5 pesos usando únicamente multiplicaciones por constante entera y desplazamientos de bits (sin división `% 3` ni `/ 3`, que son lentísimas en GPU/CPU). *Pista:* en `llama.cpp`, `q = (xi * 3) >> 8` tras escalar al rango `[0, 255]` con $\lceil b_{\text{base3}} \cdot 256 / 243 \rceil = \lfloor (b_{\text{base3}} \cdot 256 + 242)/243 \rfloor$ y actualizar `xi = (xi * 3) & 0xFF`.

2. **Invarianza de Hadamard:** Demuestra en 3 líneas que si $H \in \mathbb{R}^{d \times d}$ es una matriz de Hadamard normalizada ($H^T H = I$) y almacenas los pesos rotados $\tilde{W} = W H$, entonces para cualquier vector de activaciones $x \in \mathbb{R}^d$, multiplicar $\tilde{x} = x H$ por $\tilde{W}^T$ preserva exactamente el producto escalar original $x W^T$, y calcula cuántas sumas/restas cuesta aplicar FHT de tamaño $N=1024$ ($N \log_2 N$) frente al coste de una GEMV de $1024 \times 1024$.

#### Prompt Pre-Flight (para Sprints E2.4 y E2.5)

```text
Actúa como investigador de cuantización de LLMs en PyTorch. Trabajamos en un PC con 16 GB de RAM y una GTX 1050 (2 GB VRAM), así que toda la calibración debe correr en CPU (FP32/BF16) cargando el modelo capa por capa (layer-by-layer) para nunca exceder 10 GB de RAM.

Crea dos archivos en `quant/`:
1. `hadamard.py`:
   - Implementa `block_hadamard_transform(x, block_size=512)` vectorizada en PyTorch usando mariposas (butterfly) O(N log N) sin construir la matriz densa, normalizada por `1/sqrt(block_size)`.
   - Incluye `tests/test_hadamard.py` comprobando: (a) involución `fht(fht(x)) ≈ x` con error < 1e-5; (b) preservación del producto escalar `dot(fht(x), fht(w)) ≈ dot(x, w)`; (c) reducción del máximo valor absoluto (outliers) sobre un vector con spikes sintéticos.
2. `ternary_gptq.py`:
   - Toma un checkpoint de Hugging Face (ej. Qwen3.5-2B), pasa 64-128 muestras de calibración (WikiText-2 + código) capa por capa.
   - Para cada `nn.Linear` del decoder (excluyendo `embed_tokens` y `lm_head` por defecto, configurable por flag):
     a) Aplica rotación Hadamard por bloques a las columnas de W y a las activaciones X de entrada.
     b) Calcula el Hessiano regularizado `H = (2/N) X^T X + damp * diag_mean * I` y su descomposición de Cholesky.
     c) Ejecuta GPTQ por bloques de `group_size=128` proyectando cada peso a `{-1, 0, +1} * scale_fp16` (donde `scale_fp16` es la media absoluta del grupo casteada a float16) y propagando el error de cuantización al resto de columnas.
   - Imprime por capa: MSE antes/después de GPTQ, % de signos invertidos respecto a RTN (round-to-nearest) y tiempo por capa.
```

#### Definición de Hecho — Hito H2

- Tests unitarios de Hadamard en verde.
- `ternary_gptq.py` procesa `Qwen3.5-2B` completo en tu PC sin OOM y logra bajar drásticamente la `PPL`/`KLD` frente a RTN y frente al `TQ1_0` sin rotación del Hito H1.
- Verificas empíricamente el fenómeno observado en ingeniería inversa de Bonsai: el porcentaje de signos invertidos por GPTQ frente al redondeo directo ronda el **20–30 %**.

---

### Módulo E3 — Portabilidad a `Gemma 4 E2B` y Evaluación de Calidad Real

**Objetivo:** Trasladar tu *pipeline* validado en `Qwen3.5-2B` a `Gemma 4 E2B` (y opcionalmente `E4B`), respetando las trampas arquitectónicas propias de Gemma 4 y evaluando calidad real (más allá de la perplejidad).

#### Micro-sprints (`5 × 20 min`)

- `[ ]` **E3.1 `[MOB]` + `[LOC-PC]`:** Inspeccionar el `config.json` y los nombres de tensores de `Gemma 4 E2B` (versión BF16 y *checkpoint* QAT oficial). Mapear qué tensores pertenecen al *decoder* de texto, cuáles son **Per-Layer Embeddings (PLE)**, cuáles son *encoders* de visión/audio y cuáles son `RMSNorm` / `QK-Norm`. Documentar el mapa en `docs/03_gemma4_anatomy.md`.
- `[ ]` **E3.2 `[LOC-PC]`:** Adaptar la lista de exclusiones de `quant/ternary_gptq.py` para `Gemma 4 E2B`:
  - **Ternarizar (`g128` + Hadamard):** proyecciones `q/k/v/o_proj` y `gate/up/down_proj` del *decoder*.
  - **Dejar protegidos (`FP16` o `Q4_0`):** `vision_tower`, `audio_tower`, *Per-Layer Embeddings (PLE)* y `lm_head`.
- `[ ]` **E3.3 `[LOC-PC]`:** Ejecutar el *pipeline* PTQ + Hadamard + GPTQ sobre `Gemma 4 E2B` (comparando partir desde el *checkpoint* BF16 vs. partir desde el *checkpoint* Gemma 4 QAT oficial) y empaquetar el resultado en GGUF ejecutable en `llama.cpp`.
- `[ ]` **E3.4 `[AGENT]` + `[LOC-PC]`:** Configurar una mini-suite de evaluación rápida con `lm-evaluation-harness` (*subset* manejable en local: 200 muestras de `GSM8K`, `IFEval` y una batería propia de 30 prompts de JSON / *Tool Calling* y modo `<|think|>`). Evaluar tu `Gemma-4-E2B-Ternary` frente a `Gemma-4-E2B-Q4_0` y frente a `Bonsai 2 2B`.
- `[ ]` **E3.5 `[LOC-AND]`:** Transferir tu `Gemma-4-E2B-Ternary` (`< 1 GB`) al móvil Android, activar KV-cache de 4 bits (`-ctk q4_0 -ctv q4_0`) y medir `tok/s`, RAM pico y `mWh/token` con `bench/measure_energy_and.sh`.

#### Ejercicios de papel y boli (`[MOB]`)

1. **Presupuesto de memoria de `Gemma 4 E2B` en móvil:** Desglosa el tamaño total en MB del archivo final si el *decoder* ternario ocupa `~1.725 bits/peso`, los *Per-Layer Embeddings* y `lm_head` se quedan en `Q4_0` (`~4.5 bits/peso`) y el KV-cache de `4096` tokens usa `q4_0`. ¿Qué porcentaje del tamaño total se comen los *embeddings* y qué pasa con la `KLD` si pruebas a bajarlos también?
2. **Diagnóstico de fallos sub-2-bit:** Cuando un modelo ternario mantiene una perplejidad aceptable (`< +1.5` sobre FP16) pero falla en `IFEval` o entra en bucle infinito al cerrar un turno, enumera 3 causas raíz (escala de `lm_head`, pérdida de calibración del token `<end_of_turn>`, o *outliers* en capas tempranas/tardías) y qué experimento de ablación rápida confirma cada una.

#### Definición de Hecho — Hito H3

- Modelo `Gemma-4-E2B-Ternary` (`< 1 GB`) ejecutándose en tu GTX 1050 y en tu móvil Android.
- Gráfica/tabla de **frontera de Pareto** (Tamaño en MB y `mWh/token` en el eje X vs. `KLD` / puntuación `IFEval+GSM8K` en el eje Y) comparando `Gemma 4 E2B` (`Q4` vs. tu ternario) con `Qwen3.5-2B` y `Bonsai 2 2B`.

---

### Módulo E4 — Ingeniería de Kernels: `GEMV` Ternario y Hadamard en CUDA (`sm_61`) y Vulkan Compute (GLSL)

**Objetivo:** Aquí es donde brilla tu perfil del TFG (`LogFit`) y donde pasas de «usuario de herramientas» a «ingeniero de *kernels*». Escribirás los *kernels* de inferencia desde cero primero en el entorno que ya dominas (CUDA `sm_61`) y luego en un único lenguaje portable (Vulkan Compute GLSL) que corra tanto en tu GTX 1050 como en la GPU con memoria compartida de tu móvil.

#### Micro-sprints (`7 × 20 min`)

- `[ ]` **E4.1 `[MOB]`:** Mapear equivalencias exactas entre CUDA (B1.2) y Vulkan Compute (GLSL):
  - `blockIdx` $\leftrightarrow$ `gl_WorkGroupID`
  - `threadIdx` $\leftrightarrow$ `gl_LocalInvocationID`
  - `__shared__` $\leftrightarrow$ `shared`
  - *warp* (32 hilos) + `__shfl_down_sync` $\leftrightarrow$ *subgroup* (`GL_KHR_shader_subgroup_arithmetic`: `subgroupAdd` / `subgroupShuffleDown`).
- `[ ]` **E4.2 `[LOC-PC]`:** Implementar en `kernels/cuda_sm61/gemv_ternary.cu` un kernel `GEMV` ternario `g128` ($y = W_{\text{tq}} \cdot x$) para una matriz típica de Qwen/Gemma ($M = 2048, K = 2048$ y $M = 6144, K = 2048$):
  - Cada *warp* (32 hilos) procesa 1 fila de $W$ (4 pesos/hilo por bloque de 128).
  - Lectura coalescente de bytes empaquetados (base 3 o 2-bit), desempaquetado en registros sin ramas (*branchless*), producto-suma en FP32 (o entero con `__dp4a` sobre Pascal `sm_61` si cuantizas $x$ a `INT8` por grupo — `Q8_1`, igual que `llama.cpp`) y reducción con `__shfl_down_sync`.
- `[ ]` **E4.3 `[LOC-PC]`:** Perfilar tu `gemv_ternary.cu` con Nsight Systems (`nsys`) y medir el ancho de banda efectivo conseguido en `GB/s` frente al límite de tu GTX 1050 (`112 GB/s`). Optimizar lecturas vectorizadas (`uint4` / 128-bit loads) hasta superar el **65–75 % del ancho de banda pico**.
- `[ ]` **E4.4 `[LOC-PC]`:** Implementar el kernel de *Fast Hadamard Transform* (`FHT`) de ancho 512 y 1024 en `kernels/cuda_sm61/fht.cu` cargando el bloque entero en *shared memory* (sin *bank conflicts*) y ejecutando las $\log_2(1024) = 10$ etapas de mariposas usando registros/*shuffles* para las primeras 5 etapas y *shared memory* para las 5 siguientes.
- `[ ]` **E4.5 `[AGENT]` + `[LOC-PC]`:** Portar ambos kernels (`gemv_ternary.comp` y `fht_1024.comp`) a Vulkan Compute (GLSL) compilables con `glslc` a SPIR-V y crear un lanzador mínimo en C++ (`kernels/bench_kernels.cpp`) para validarlos numéricamente contra referencia FP32 en tu GTX 1050.
- `[ ]` **E4.6 `[LOC-AND]`:** Ejecutar los mismos *compute shaders* Vulkan en tu móvil Android (aprovechando la memoria compartida CPU/GPU con `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` para eliminar copias H2D/D2H innecesarias). Comparar ancho de banda efectivo y `mWh/operación` frente a CPU NEON.
- `[ ]` **E4.7 `[LOC-PC]`:** Inspeccionar los *issues* abiertos de `ggml-vulkan` en `llama.cpp` (por ejemplo, soporte de Hadamard de bloque grande como 1024 o integración de descuantización `TQ1_0`/`TQ2_0`) y probar la integración de tu *shader* directamente dentro de una rama local de `llama.cpp`.

#### Ejercicios de papel y boli (`[MOB]`)

1. **Aritmética sin multiplicaciones flotantes en Pascal (`sm_61`):** La GTX 1050 tiene bajo rendimiento en FP16 nativo, pero incluye la instrucción hardware `__dp4a(a, b, c)` (producto escalar de 4 enteros de 8 bits en 1 ciclo). Diseña cómo empaquetar 4 pesos desempaquetados $\{-1, 0, +1\}$ como `int8x4` dentro de un `int32`, cuantizar el vector de activación $x$ a `int8` por bloque de 128 (`Q8_1`), acumular 32 llamadas a `__dp4a` en `int32` y aplicar el producto de escalas $s_w \cdot s_x$ una sola vez al final del bloque de 128. Calcula cuántos ciclos de ALU ahorras por bloque.
2. ***Zero-copy* en memoria compartida móvil:** En un PC discreto (PCIe), mover activaciones entre CPU y GPU cuesta microsegundos por transferencia. En un SoC móvil con memoria unificada, explica cómo afecta el *buffer mapping* persistente en Vulkan al diseño de un *scheduler* híbrido (CPU para *router*/embeddings + GPU para `GEMV` ternario), trazando el paralelismo directo con tu trabajo en `LogFit`.

#### Prompt Pre-Flight (para Sprints E4.2 y E4.5)

```text
Actúa como ingeniero de kernels GPU especializado en CUDA Pascal (sm_61) y Vulkan Compute (GLSL/SPIR-V).
Necesito construir un microbenchmark autocontenido en `kernels/` para inferencia GEMV ternaria con group_size=128 (formato TQ2_0: 4 pesos ternarios en 1 byte, 2 bits por peso {0->-1, 1->0, 2->+1}, 1 escala fp16 por cada 128 pesos):

1. En `kernels/cuda_sm61/gemv_ternary.cu`:
   - Escribe un kernel `gemv_tq2_0_dp4a` optimizado para `sm_61` donde cada warp (32 hilos) calcula 1 fila de salida.
   - Usa cargas vectorizadas de 32/64 bits para los pesos empaquetados, desempaquetado branchless a int8x4, instrucción `__dp4a` contra activaciones cuantizadas `Q8_1` (int8 + escala fp16 por bloque de 128) y reducción final con `__shfl_down_sync`.
   - Incluye un harness en el mismo archivo que verifique error absoluto máximo < 1e-2 contra una referencia CPU FP32 para matrices (2048x2048) y (6144x2048), y reporte latencia (us) y ancho de banda efectivo (GB/s) sobre 200 iteraciones tras warmup.

2. En `kernels/vulkan_glsl/fht_1024.comp`:
   - Escribe un compute shader GLSL (Vulkan 1.2+, `GL_KHR_shader_subgroup_shuffle`) que aplique Fast Hadamard Transform in-place de longitud 1024 sobre un batch de vectores usando `shared float sdata[1024]` y minimizando `barrier()` mediante shuffles en subgrupo cuando sea posible.
   - Asegura compatibilidad con GPUs móviles (subgroups de tamaño 32 o 64 usando `gl_SubgroupSize`).
```

#### Definición de Hecho — Hito H4

- Kernel `gemv_tq2_0_dp4a` en CUDA `sm_61` validado numéricamente y alcanzando **$> 65\%$ del ancho de banda teórico** de tu GTX 1050 (`~70–85 GB/s` efectivos).
- *Compute shaders* Vulkan (`fht_1024.comp` y/o `gemv_tq2_0.comp`) ejecutándose con el mismo código en tu GTX 1050 y en tu móvil Android, con tabla comparativa de rendimiento y energía.
- *(Objetivo estrella)* Rama limpia de `llama.cpp` con tu mejora lista para abrir un Pull Request público.

---

### Módulo E5 — Ahorro Local (`< 8 W`), Reparto Heterogéneo y Dossier Técnico

**Objetivo:** Convertir el trabajo de investigación en utilidad diaria real (ahorro en casa) y empaquetar los resultados en dos activos de altísimo valor: tu *portfolio* de **Inference MLE** y un dossier de 1 página listo para cuando decidas hablar con **Rafael Asenjo**.

#### Micro-sprints (`4 × 20 min`)

- `[ ]` **E5.1 `[LOC-AND]`:** Configurar `deploy/local_server.sh` en tu móvil Android (o PC viejo) levantando `llama-server` con tu modelo ganador (`Bonsai 2 2B` o `Gemma 4 E2B Ternary` + KV-cache `q4_0`), expuesto en tu red local como *endpoint* compatible con OpenAI para tus herramientas diarias (resúmenes, asistente ligero de terminal/código). Medir consumo sostenido en vatios y coste eléctrico anual estimado en `€/año`.
- `[ ]` **E5.2 `[LOC-PC]` + `[LOC-AND]`:** Experimento **LogFit-Edge** (reparto heterogéneo): medir la diferencia de `tok/s` y `mWh/token` al ejecutar *prefill* y capas densas en GPU (Vulkan/CUDA) frente a dejar capas concretas (*embeddings*/PLE o *draft* MTP) en CPU NEON/`x86`, registrando en qué longitud de *prompt* se cruza el punto óptimo CPU vs. GPU.
- `[ ]` **E5.3 `[MOB]` + `[LOC-PC]`:** Actualizar el `README.md` de tu antiguo repo `TheUnixRoot/Logfit` (3 párrafos explicando qué hacía sobre `oneTBB`, el contexto del TFG y un enlace hacia tu nuevo repo `edge-ternary-lab` como su continuación moderna sobre Vulkan/CUDA para LLMs).
- `[ ]` **E5.4 `[MOB]`:** Redactar `docs/04_dossier_uma.md` (**exactamente 1 página**): problema (cuello de botella de memoria y `mWh/token` en Edge AI), resultados medidos en tus dispositivos con `Qwen3.5-2B` y `Gemma 4 E2B`, y tu aportación en *kernels* portables Vulkan/CUDA.

#### Definición de Hecho — Hito H5

- Servidor local funcionando en casa consumiendo `< 8 W` (móvil) o `< 35 W` (PC) con coste mensual prácticamente despreciable.
- Repositorios `Logfit` y `edge-ternary-lab` limpios, enlazados y con métricas reproducibles.
- Dossier de 1 página listo en el cajón para el día que decidas proponer una colaboración, publicación o proyecto a Rafael Asenjo (o para adjuntar en entrevistas de AI Infra / Inference MLE).

---

### Módulo E6 (Bonus Opcional) — Recuperación QAT Mínima con STE + Destilación KL

**Objetivo:** Solo si tras completar `E3` quieres comprobar empíricamente cómo PrismML cierra la brecha entre PTQ puro (Bonsai 2) y QAT agéntico (Bonsai 2.5), usando una GPU gratuita de Google Colab / Kaggle (`T4 16 GB`).

#### Micro-sprints (`3 × 20 min`)

- `[ ]` **E6.1 `[MOB]`:** Revisar por qué los pesos latentes en QAT ternario deben vivir siempre en FP32 (evitar *underflow* en actualizaciones pequeñas al cruzar los umbrales de redondeo) y cómo monitorizar la métrica de *code flips* (% de pesos que cambian de estado `{-1, 0, +1}` por paso).
- `[ ]` **E6.2 `[AGENT]`:** Adaptar la clase `TernaryLinearG128` con *Straight-Through Estimator* (`STE`) y pérdida de Divergencia KL respecto a los *logits* del profesor FP16 congelado, ejecutando una prueba corta (1M–5M tokens) sobre `Gemma 4 E2B` en Colab/Kaggle con Adafactor y *gradient checkpointing*.
- `[ ]` **E6.3 `[LOC-PC]`:** Comparar la `KLD` y la tasa de acierto en JSON/*tool calling* antes (solo PTQ+Hadamard de E3) y después de esos pocos millones de tokens de destilación KL.

---

## 4. Plantilla Maestra de Resultados (Frontera de Pareto)

Copia esta tabla al `README.md` de `edge-ternary-lab` y ve completándola al cerrar cada hito (`H1` → `H4`):

| Modelo | Método / Formato | Bits/peso eff. | Tamaño (MB) | Dispositivo / Backend | `pp512` (`tok/s`) | `tg128` (`tok/s`) | `PPL` (WT-2) | `KLD` vs F16 | `GSM8K` / `IFEval` | Potencia neta (W) | `mWh / token` |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Qwen3.5-2B` | `F16` (Ref.) | `16.00` | `~3800` | PC CPU | — | — | — | `0.000` | — | — | — |
| `Qwen3.5-2B` | `Q4_K_M` | `~4.80` | `~1300` | GTX 1050 CUDA | — | — | — | — | — | — | — |
| `Qwen3.5-2B` | `TQ1_0` (naive) | `~1.73` | `~550` | GTX 1050 CUDA | — | — | — | — | — | — | — |
| `Bonsai 2 2B` | `PTQ1_0` / `PQ2_0` | `~1.73` | `~450` | GTX 1050 / Android | — | — | — | — | — | — | — |
| `Qwen3.5-2B` (Tuyo) | Hadamard + GPTQ `g128` | `~1.73` | `~550` | GTX 1050 / Android | — | — | — | — | — | — | — |
| `Gemma 4 E2B` | `Q4_0` (Oficial) | `~4.50` | — | GTX 1050 / Android | — | — | — | — | — | — | — |
| `Gemma 4 E2B` (Tuyo) | Hadamard + GPTQ `g128` | `~1.9`* | `< 1000` | GTX 1050 / Android | — | — | — | — | — | — | — |
| `Gemma 4 E2B` (Tuyo) | Tu Kernel Vulkan/CUDA | `~1.9`* | `< 1000` | GTX 1050 / Android | — | — | — | — | — | — | — |

*\*Incluye capas protegidas (`PLE`/`lm_head`) en mayor precisión.*

---

## 5. Checklist de Enganche con `AI-infra-study` (Sin Quemarte)

- **Mientras estés cerrando el Bloque 2 de Curso 1 (`AI-infra-study`):** Ejecuta solo los Módulos **E0** y **E1** en ratos sueltos (fines de semana o micro-sprints de descanso); no escribas código complejo todavía, solo monta el banco de pruebas y observa los números.
- **Al arrancar el Bloque 3 (`B3.1 Cuantización`) de Curso 1:** Ejecuta **E2** y **E3** en paralelo a la teoría de `B3.1` y `B3.4`; te servirá como laboratorio práctico real a coste **0 €** antes de alquilar GPU para `Cloud-3`.
- **Tras cerrar el Bloque 3:** Ejecuta **E4** (tus kernels CUDA `sm_61` + Vulkan) como proyecto diferencial de sistemas y cierra con **E5** (servidor en casa + dossier para Rafael Asenjo).

---

## 6. Registro Maestro de Progreso (`E0`–`E6`)

| ID | Módulo | Sprint | Modo | Estado |
| :--- | :---: | :--- | :---: | :---: |
| `E0.1` | `E0` | Repo `edge-ternary-lab` + `llama.cpp` CUDA/Vulkan PC | `[LOC-PC]` | `[ ]` |
| `E0.2` | `E0` | Compilar `PrismML-llama.cpp` (`PTQ1_0` / `PQ2_0`) | `[LOC-PC]` | `[ ]` |
| `E0.3` | `E0` | Compilar `llama.cpp` ARM NEON (+ Vulkan) en Termux | `[LOC-AND]` | `[ ]` |
| `E0.4` | `E0` | Telemetría `measure_energy_pc.py` + `measure_energy_and.sh` | `[AGENT]` `[LOC-AND]` | `[ ]` |
| `E1.1` | `E1` | Descargar GGUFs `Qwen3.5-2B` y `Bonsai 2 2B` | `[LOC-PC]` | `[ ]` |
| `E1.2` | `E1` | `llama-bench` GTX 1050 (CUDA vs Vulkan) + `mWh/tok` | `[LOC-PC]` | `[ ]` |
| `E1.3` | `E1` | `llama-bench` Android Termux (NEON vs Vulkan) | `[LOC-AND]` | `[ ]` |
| `E1.4` | `E1` | `PPL` + `KLD` vs F16 (`docs/02_bonsai_vs_qwen.md`) | `[LOC-PC]` | `[ ]` |
| `E2.1` | `E2` | `block_tq1_0` / `block_tq2_0` y rotación Hadamard | `[MOB]` | `[ ]` |
| `E2.2` | `E2` | Matemática GPTQ / OBQ e inversa del Hessiano | `[MOB]` | `[ ]` |
| `E2.3` | `E2` | `llama-imatrix` + cuantización `TQ1_0`/`TQ2_0`/`IQ1_M` | `[LOC-PC]` | `[ ]` |
| `E2.4` | `E2` | `quant/hadamard.py` + tests ortogonalidad y outliers | `[AGENT]` `[LOC-PC]` | `[ ]` |
| `E2.5` | `E2` | `quant/ternary_gptq.py` layer-by-layer en CPU | `[AGENT]` `[LOC-PC]` | `[ ]` |
| `E2.6` | `E2` | Cuantizar `Qwen3.5-2B` y comparar vs Bonsai 2 | `[LOC-PC]` | `[ ]` |
| `E3.1` | `E3` | Anatomía `Gemma 4 E2B` (`PLE`, visión/audio, `QK-Norm`) | `[MOB]` `[LOC-PC]` | `[ ]` |
| `E3.2` | `E3` | Reglas de exclusión `Gemma 4 E2B` en `ternary_gptq.py` | `[LOC-PC]` | `[ ]` |
| `E3.3` | `E3` | PTQ+FHT+GPTQ sobre `Gemma 4 E2B` (BF16 vs QAT) → GGUF | `[LOC-PC]` | `[ ]` |
| `E3.4` | `E3` | Evals reales (`GSM8K`, `IFEval`, JSON/Tools, `<|think|>`) | `[AGENT]` `[LOC-PC]` | `[ ]` |
| `E3.5` | `E3` | Despliegue Android `< 1 GB` + KV `q4_0` + `mWh/token` | `[LOC-AND]` | `[ ]` |
| `E4.1` | `E4` | Equivalencias CUDA $\leftrightarrow$ Vulkan Compute (`subgroup`) | `[MOB]` | `[ ]` |
| `E4.2` | `E4` | Kernel CUDA `sm_61` `gemv_ternary.cu` (`__dp4a`) | `[LOC-PC]` | `[ ]` |
| `E4.3` | `E4` | Profiling `nsys` y vectorización (`> 65–75%` de 112 GB/s) | `[LOC-PC]` | `[ ]` |
| `E4.4` | `E4` | Kernel CUDA `fht.cu` (512/1024, shuffles + SMEM) | `[LOC-PC]` | `[ ]` |
| `E4.5` | `E4` | Shaders Vulkan GLSL (`gemv_ternary.comp`, `fht_1024.comp`) | `[AGENT]` `[LOC-PC]` | `[ ]` |
| `E4.6` | `E4` | Ejecución Vulkan *zero-copy* en Android vs NEON | `[LOC-AND]` | `[ ]` |
| `E4.7` | `E4` | Integración en rama local `llama.cpp` (`ggml-vulkan`) | `[LOC-PC]` | `[ ]` |
| `E5.1` | `E5` | `deploy/local_server.sh` 24/7 (`< 8 W`) + `€/año` | `[LOC-AND]` | `[ ]` |
| `E5.2` | `E5` | Experimento *LogFit-Edge* (reparto heterogéneo CPU+GPU) | `[LOC-PC]` `[LOC-AND]` | `[ ]` |
| `E5.3` | `E5` | Actualizar `TheUnixRoot/Logfit` enlazando `edge-ternary-lab` | `[MOB]` `[LOC-PC]` | `[ ]` |
| `E5.4` | `E5` | Redactar `docs/04_dossier_uma.md` (1 pág. Rafael Asenjo) | `[MOB]` | `[ ]` |
| `E6.1` | `E6` | Pesos latentes FP32 y métrica *code flips* en QAT | `[MOB]` | `[ ]` |
| `E6.2` | `E6` | `TernaryLinearG128` STE + KL en Colab/Kaggle T4 | `[AGENT]` | `[ ]` |
| `E6.3` | `E6` | Comparar `KLD` y JSON/*Tool Calling* PTQ vs QAT ligero | `[LOC-PC]` | `[ ]` |
