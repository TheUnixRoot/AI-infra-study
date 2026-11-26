#import "../theme.typ": *

= Curso 1 · Bloque 1: Arquitectura GPU, CUDA C++ y Profiling (`CHAT-B1`)

#block-banner(
  "B1",
  "Arquitectura GPU, CUDA C++ (`sm_61`), Profiling y Salto a Triton",
  "Oferta 1 (Core: profiling kernel/sistema) y Oferta 2 (Nice-to-have: CUDA/Triton, cuellos de botella)",
  "~3 semanas · 18–24 sprints de 20m",
)[
  Domina el modelo físico de ejecución en GPU: jerarquía de memoria (Registros, Shared Memory, Caché L2, DRAM GDDR5/HBM), modelo Roofline, coalescencia de warps, sincronización de bloques, profiling CPU-GPU con `nsys`/`perf` y fusión de kernels en Triton (`ncu`).
]

#infografia(
  "Jerarquía de Memoria Pascal GP107 (GTX 1050) y Resultados Medidos B1.1 / B1.2",
  "Izquierda: jerarquía y ancho de banda relativo en GTX 1050 (sm_61). Derecha: mediciones reales de B1.1 (clone vs contiguous) y B1.2 (progresión GEMM FP32 2048x2048).",
  height: 190pt,
)[
  // Pirámide izquierda
  #place(dx: 8pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[Jerarquía Física GTX 1050 (sm\_61)])
  #diag-node(36pt, 20pt, 140pt, 24pt, "Registros SM (~256 KB/SM)", sub: "Latencia 1 ciclo · Privados por hilo", fill: bg-teal, col: teal)
  #diag-node(24pt, 49pt, 164pt, 24pt, "Shared Memory (48 KB/blk · 96 KB/SM)", sub: "32 bancos · Sin conflictos = ~TB/s", fill: bg-navy, col: navy)
  #diag-node(14pt, 78pt, 184pt, 24pt, "Caché L2 Global (1.0 MB)", sub: "Líneas 128 B = 4 sectores de 32 B", fill: bg-amber, col: amber)
  #diag-node(4pt, 107pt, 204pt, 26pt, "VRAM GDDR5 (2.0 GB · 128-bit)", sub: "Pico teórico: 112 GB/s · Medido clone(): 91.8 GB/s", fill: bg-red, col: crimson)
  #diag-node(4pt, 139pt, 204pt, 36pt, "Host RAM <-> PCIe 3.0 x16", sub: "Pageable: 3.71 ms (doble hop) vs Pinned DMA: 1.78 ms (2.08x)\nRoofline ridge: ~16.6 FLOP/Byte (1860 GFLOP/s / 112 GB/s)", fill: bg-light, col: slate)

  // Separador vertical
  #place(dx: 222pt, dy: 6pt, line(length: 175pt, angle: 90deg, stroke: 0.6pt + border-gray))

  // Barras derechas: B1.1 y B1.2 medidos
  #place(dx: 232pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[Mediciones Reales GTX 1050 (`artifacts/`)])
  #place(dx: 232pt, dy: 18pt, text(fill: muted, size: 7.2pt)[B1.1 · Copia 2048x2048 FP32 (33.55 MB DRAM, pico 112 GB/s):])
  #hbar(232pt, 30pt, "x.clone() coalescente", "91.8 GB/s (82%)", 91.8 / 112.0, max-w: 110pt, h: 11pt, col: teal)
  #hbar(232pt, 46pt, "y.contiguous() strided", "33.4 GB/s (30%)", 33.4 / 112.0, max-w: 110pt, h: 11pt, col: amber)

  #place(dx: 232pt, dy: 68pt, text(fill: muted, size: 7.2pt)[B1.2 · GEMM FP32 2048x2048x2048 (escala relativa a cuBLAS 1627 GFLOP/s):])
  #hbar(232pt, 80pt, "gemm_naive (strided)", "24.0 GF/s (714.6 ms)", 24.0 / 1627.1, max-w: 110pt, h: 11pt, col: crimson)
  #hbar(232pt, 96pt, "gemm_coalesced", "115.6 GF/s (4.81x)", 115.6 / 1627.1, max-w: 110pt, h: 11pt, col: amber)
  #hbar(232pt, 112pt, "gemm_smem_tiled 32x32", "281.2 GF/s (11.7x)", 281.2 / 1627.1, max-w: 110pt, h: 11pt, col: blue)
  #hbar(232pt, 128pt, "cublasSgemm (Ref)", "1627.1 GF/s (87% pico)", 1.0, max-w: 110pt, h: 11pt, col: navy)

  #place(dx: 232pt, dy: 149pt, box(width: 236pt, fill: bg-light, stroke: 0.4pt + border-gray, inset: 4pt, radius: 3pt)[
    #set text(size: 7pt, fill: slate)
    *B1.2 Softmax 1024x1024:* 3-Pass `0.175 ms` (47.9 GB/s) vs Online `0.134 ms` (62.7 GB/s, *1.31x*).
    *B1.3:* `pin_memory` acelera H2D `2.08x`; 600 kernels unfused dominan (218 ms = 93 GB/s).
  ])
]

== Unidad B1.1 · Modelo Mental de Hardware, Roofline y PyTorch Internals

#unit-header(
  "B1.1",
  "Modelo Mental de Hardware, Roofline y PyTorch Internals",
  "[MOB] 4s + [LOC] 2s",
  "Ninguna",
  "B1.2, B1.3, B2.1",
  "Móvil + GTX 1050 (CPU/GPU)",
  "Explicar sin notas por qué subir batch size en decode eleva I hasta el codo Roofline.",
)

#concepto-clave[
  1. *Modelo Roofline e Intensidad Aritmética:* El rendimiento alcanzable de cualquier kernel está acotado por el mínimo entre el techo de cómputo $pi_"pico"$ (FLOP/s) y el producto del ancho de banda de memoria $beta_"mem"$ (Bytes/s) por la intensidad aritmética $I$:
     $ I = frac("FLOPs ejecutados", "Bytes transferidos DRAM"), quad P_"max" = min(pi_"pico", beta_"mem" dot I) $
     El *punto de inflexión (ridge point)* ocurre en $I^* = pi_"pico" / beta_"mem"$. En una GTX 1050 ($1860 "GFLOP/s"$ y $112 "GB/s"$), $I^* approx 16.6 "FLOP/B"$; en una H100 SXM ($989 "TFLOP/s"$ FP16 y $3.35 "TB/s"$ HBM3), $I^* approx 295 "FLOP/B"$. Por eso un *Prefill* de $S = 2048$ tokens ($I >> I^*$) es *compute-bound*, mientras un *Decode* con batch $B = 1$ ($I approx 1 "FLOP/B"$, al multiplicar un vector por matrices de pesos de GBs) es estrictamente *memory-bandwidth bound*.
  2. *PyTorch Internals (`Storage`, `shape`, `stride`, `offset`):* Un `torch.Tensor` separa el búfer lineal físico en VRAM (`UntypedStorage`) del descriptor ligero en CPU (`TensorImpl`). La posición en memoria del elemento $(i, j)$ es:
     $ "addr"(i, j) = "base" + ("offset" + i dot "stride"_0 + j dot "stride"_1) dot "sizeof"("dtype") $
     Al invocar `y = x.transpose(0, 1)` sobre una matriz $2048 times 2048$, PyTorch solo intercambia `(2048, 1)` por `(1, 2048)` en CPU (*zero-copy*): `y.data_ptr() == x.data_ptr()`, cero tráfico DRAM. Al invocar `.contiguous()`, asigna un nuevo búfer en VRAM y lanza un kernel de copia física que lee/escribe $2 times N^2 times 4$ bytes con patrón *strided* en una de las dimensiones.
  3. *Pipeline de `torch.compile`:* *TorchDynamo* (captura grafo FX vía PEP 523) $arrow.r$ *AOTAutograd* (traza forward+backward) $arrow.r$ *TorchInductor* (genera kernels Triton fusionados; requiere `sm_70+`).
]

#practica-box(title: "Práctica B1.1 (b1_1_strides/) · Medición de Strides y Baseline Coalescente")[
  - *Script:* `b1_1_strides/b1_1_strides_bench.py` + `b1_1_strides/run.sh` (por defecto $2048 times 2048$ FP32 = $16.0 "MiB"$, superando la L2 de $1 "MB"$ para saturar DRAM; `--smoke-test` usa $512 times 512$).
  - *Lectura empírica en GTX 1050 (`artifacts/b1_1_metrics.json`):*
    - `y = x.transpose(0, 1)`: `0.0036 ms` GPU, $0 "B"$ transferidos, `y.data_ptr() == x.data_ptr()`.
    - `x.clone()` (baseline secuencial coalescente): `0.366 ms` $arrow.r$ *91.77 GB/s (81.9% del pico teórico)*, probando que la GPU rinde sanamente.
    - `z = y.contiguous()` (copia física transpuesta): `1.004 ms` $arrow.r$ *33.44 GB/s (29.9%)*, penalización de *2.74x* frente a `clone()` (y *8x* a $8192 times 8192$: `8.2 GB/s` vs `92.7 GB/s`) causada por accesos no coalescentes.
]

#ejercicio-box[
  *Ejercicio B1.1-A (Cálculo Roofline):* Un modelo denso de $8 times 10^9$ parámetros en FP16 (2 B/param) ejecuta un paso de *decode* con $B=1$ ($2 "FLOP"$ por parámetro) sobre una GPU con $pi_"pico" = 312 "TFLOP/s"$ y $beta = 2000 "GB/s"$. Calcula $I$, compáralo con $I^*$, determina la latencia mínima teórica del paso y calcula qué batch $B$ hace falta para alcanzar el codo Roofline.

  *Ejercicio B1.1-B (Diagnóstico de Strides):* Tienes un tensor `t` contiguo de forma `(4, 8, 64)` en FP16. Tras `v = t.transpose(0, 1)[:, ::2, :]`, calcula su `shape`, `stride`, si `v.is_contiguous()` es verdadero y cuántos bytes moverá `v.contiguous()` en DRAM.
]

== Unidad B1.2 · Fundamentos de CUDA C++ en Local (`sm_61`)

#unit-header(
  "B1.2",
  "Fundamentos de CUDA C++: Coalescencia, SMEM Tiling y Warp Shuffles",
  "[MOB] 3s + [LOC] 4s",
  "B1.1",
  "B1.3, B1.4",
  "Móvil + GTX 1050 (nvcc)",
  "Medir mejora real naive -> coalesced -> smem_tiled y explicar los accesos GMEM/SMEM.",
)

#concepto-clave[
  1. *Topología Warp y Sectores de Memoria (32 B / 128 B):* Un *warp* de 32 hilos ejecuta en *lock-step* con `threadIdx.x` consecutivo (`0..31`). Las líneas de caché L2 miden 128 B compuestas por *4 sectores de 32 B*. En `gemm_naive` (`tx -> row`, `ty -> col`), hilos adyacentes leen `A[row * K + k]` a saltos de $K dot 4$ bytes: cada hilo solicita un sector de 32 B distinto para aprovechar solo 4 B (*87.5% de desperdicio*), mientras `B[k * N + col]` hace *broadcast*. Al permutar a `gemm_coalesced` (`tx -> col`, `ty -> row`), `A` hace *broadcast* de 1 ciclo y `B`/`C` empaquetan 32 floats contiguos en una línea L2 de 128 B (100% eficiencia).
  2. *Tiling Cooperativo en Shared Memory (`32x32` vs `16x16`):* Cargar tiles $T times T$ de `A` y `B` en `__shared__` reutiliza cada float $T$ veces y evita conflictos entre los 32 bancos de SMEM:
     - `gemm_smem_tiled` (`32x32`, 1024 hilos/bloque, 8 KB SMEM): reutilización *32x*, pero en Pascal (`max 2048 hilos/SM`, `65,536 regs/SM`) admite máximo 2 bloques/SM y si excede 32 regs/hilo cae de golpe al *50% de ocupación*, deteniendo los 32 warps juntos en cada `__syncthreads()`.
     - Variante `gemm_smem_tiled_16` (`16x16`, 256 hilos/bloque, 2 KB SMEM): reutilización *16x* (doble de cargas GMEM/barreras), pero admite hasta *8 bloques/SM* con degradación gradual de ocupación y ocultación de latencia entre bloques, manteniendo 0 *bank conflicts* entre ambos medios warps (`ty=0` y `ty=1`).
  3. *Online Softmax y Reducción Intra-Warp (`__shfl_down_sync`):* Softmax tradicional requiere 3 pasadas (máximo, suma, normalización = $4N$ accesos lógicos). Aplicando la ley asociativa de Milakov & Gimelshein sobre tuplas $(m, d)$:
     $ m_"AB" = max(m_A, m_B), quad d_"AB" = d_A dot e^(m_A - m_"AB") + d_B dot e^(m_B - m_"AB") $
     se reducen 32 hilos en 5 pasos de mariposa (`offset = 16, 8, 4, 2, 1`) en registros sin usar SMEM, bajando de $4N$ a $3N$ accesos lógicos (2 lecturas de `X` + 1 escritura de `Y`, base del *tiling* de FlashAttention).
]

#practica-box(title: "Práctica B1.2 (b1_2_cuda_gemm/) · GEMM 32x32 / 16x16 y Warp Online Softmax")[
  - *Ficheros:* `gemm_bench.cu` (5 kernels: `cublasSgemm`, `gemm_naive`, `gemm_coalesced`, `gemm_smem_tiled_16`, `gemm_smem_tiled`), `warp_online_softmax.cu`, `parse_results.py`, `run.sh` (compila vía `CUDA_ARCH` en Docker Linux/Windows).
  - *Mediciones reales GTX 1050 (2048×2048, `artifacts/b1_2_results.json`):* `naive` = `714.6 ms` (`24.0 GFLOP/s`) $arrow.r$ `coalesced` = `148.6 ms` (`115.6 GFLOP/s`, *4.81x*) $arrow.r$ `smem_tiled 32x32` = `61.1 ms` (`281.2 GFLOP/s`, *11.69x*) $arrow.r$ `cuBLAS` = `10.6 ms` (`1627.1 GFLOP/s`, ~87% del pico FP32). El gap `smem_tiled` vs `cuBLAS` (~17%) ilustra la necesidad de *register blocking* 2D y cargas vectorizadas (`float4`).
  - *Bug de borde corregido en `warp_online_softmax.cu`:* para `N <= 16`, los *lanes* inactivos tenían `m = -INFINITY, d = 0.0f` y producían `-INFINITY - (-INFINITY) = NaN` al combinarse entre sí; protegido comprobando `d == 0.0f` / `other_d == 0.0f`.
]

#ejercicio-box[
  *Ejercicio B1.2-A (Ocupación SM en Pascal):* Un kernel GEMM con bloques `32x32` (1024 hilos) compila usando 44 registros por hilo en `sm_61` (65,536 registros/SM, máx. 2048 hilos/SM). ¿Cuántos bloques caben por SM y qué % de ocupación alcanza? ¿Cuánto subiría si se reescribe con bloques `16x16` (256 hilos) manteniendo 44 registros/hilo?

  *Ejercicio B1.2-B (Tráfico Softmax):* Para $M = 1024, N = 1024$ FP32, calcula el volumen total en MB de accesos lógicos de `3-Pass` ($4N$) vs `Online` ($3N$) frente al tráfico mínimo teórico útil ($2N$).
]

== Unidad B1.3 · Profiling Multinivel: `nsys` y Linux `perf`

#unit-header(
  "B1.3",
  "Profiling Multinivel CPU-GPU con Nsight Systems y Linux perf",
  "[MOB] 2s + [LOC] 3s",
  "B1.1, B1.2",
  "B1.4, B4.1, B5.1, Proyecto B",
  "Móvil + GTX 1050 (nsys, perf)",
  "Distinguir con nsys stats en <2 min si un cuello de botella es CPU-bound o GPU-bound.",
)

#concepto-clave[
  1. *Tres Antipatrones Clásicos en Pipelines PyTorch:*
     - *Antipatrón A (`.item()` en bucle caliente):* Fuerza `cudaStreamSynchronize` + copia D2H en cada iteración, vaciando la cola de comandos y serializando CPU y GPU. Se corrige acumulando en un tensor residente en dispositivo y leyendo una sola vez fuera del bucle.
     - *Antipatrón B (H2D desde memoria paginable):* El motor DMA PCIe no puede acceder directamente a memoria virtual paginable del SO; el driver CUDA copia primero a un *staging buffer* *pinned* en CPU y bloquea el hilo. Con `pin_memory=True` y `non_blocking=True` el DMA va directo por PCIe.
     - *Antipatrón C (Cientos de ops element-wise sin fusionar):* Cada operación eager (`mul`, `add`, `relu`) despacha un kernel independiente con roundtrip completo a DRAM cuando el tensor excede la L2 (1 MB).
  2. *Ley de Amdahl en Diagnóstico Real (Lección de `nsys` en GTX 1050):* Al perfilar 200 pasos (`600 kernels`) sobre $2048 times 2048$ (`16 MiB`): `pin_memory` redujo H2D de `3.71 ms` a `1.78 ms` (*2.08x*), pero el cómputo sin fusionar tardó `218.2 ms` (cada kernel moviendo $33.55 "MB"$ a ~93 GB/s). En `nsys`, la aparente espera de `210 ms` en `.item()` de `BAD` *no era coste de sincronización*, sino la CPU esperando a que acabara el cómputo ya encolado (99% del paso). Por eso eliminar `.item()` solo mejora la latencia total cuando hay trabajo CPU solapable (`--cpu-work-ms`) o cuando el cómputo GPU ha sido previamente fusionado (`B1.4`).
]

#practica-box(title: "Práctica B1.3 (b1_3_profiling/) · Trazas NVTX y nsys Real")[
  - *Instrumentación:* `b1_3_antipatterns.py` delimita cada fase con `torch.cuda.nvtx.range_push/pop`, evita `torch.cuda.synchronize()` dentro del bucle en `mode="opt"` mediante arrays de `torch.cuda.Event`, e incluye `--cpu-work-ms` y `--check-only`.
  - *Nota de plataforma:* en WSL2 `nsys` falla al importar trazas Pascal (`Unrecognized GPU UUID`); se ejecuta `nsys` nativo en Windows (`--trace=cuda,nvtx`) o Linux nativo (`--trace=cuda,nvtx,osrt` + `perf stat`). Las trazas `.nsys-rep` embeben variables de entorno y se excluyen en `.gitignore`; las tablas resumidas se guardan en `artifacts/b1_3_nsys_stats.txt`.
]

#ejercicio-box[
  *Ejercicio B1.3-A (Diagnóstico `nsys`):* En `cuda_gpu_kern_sum` ves 6,000 llamadas a kernels `elementwise` de `362 us` cada uno para un tensor de $16.78 "MB"$ y `cudaLaunchKernel` promedio de `3.5 us`. Calcula si el cuello de botella es *launch-overhead bound* (CPU) o *DRAM bandwidth bound* (GPU) y qué ancho de banda efectivo logra cada kernel.
]

== Unidad B1.4 · Salto a Cloud: OpenAI Triton y Nsight Compute (`ncu`)

#unit-header(
  "B1.4",
  "OpenAI Triton y Nsight Compute (ncu) en Cloud-1",
  "[MOB] 2s + [PREP] 1s + [CLD] 1-2s",
  "B1.2, B1.3",
  "Kernels modernos para B2, B3",
  "Cloud-1: Scaleway 1x L4 o 1x L40S (sm_89)",
  "Informe comparando eager vs torch.compile vs Triton custom con métricas Roofline de ncu.",
)

#concepto-clave[
  1. *Modelo de Programación por Bloques en Triton:* Frente a CUDA C++ (orientado a hilos individuales y gestión manual de bancos SMEM), Triton programa a nivel de *bloques de programa* (`tl.program_id(axis=0)`), donde operaciones vectorizadas (`tl.load`, `tl.store` con `mask`, `tl.arange`) se compilan mediante MLIR/LLVM a código PTX optimizado gestionando automáticamente *coalescing*, *shared memory* y registros en `sm_70+`.
  2. *Fusión de Kernels (*Fused RMSNorm* y *Fused Softmax*):* Al fusionar múltiples lecturas/escrituras de activaciones en un único kernel Triton, los tensores intermedios viven en SRAM/registros dentro del bloque, reduciendo los viajes a HBM de $4"–"6$ pasadas a $1$ lectura + $1$ escritura.
  3. *Lectura de Nsight Compute (`ncu --set roofline`):*
     - *Speed of Light (SOL):* compara `SM Throughput %` frente a `Memory (DRAM/HBM) Throughput %` para clasificar el kernel sin ambigüedades.
     - *Warp Stall Reasons:* distingue `stall_long_scoreboard` (esperando respuesta de HBM/L2), `stall_wait` (barreras/dependencias cortas) y `stall_math_pipe_throttle` (saturación aritmética).
]

#practica-box(title: "Práctica B1.4 ([PREP] -> [CLD] Cloud-1) · Fused Softmax & RMSNorm en Triton")[
  - *Pre-Flight (`[PREP]`, 0 EUR):* Implementar `fused_softmax_triton.py` y `fused_rmsnorm_triton.py` con `@triton.jit`, benchmark `triton.testing.do_bench` frente a PyTorch eager y `torch.compile`, y wrapper `run.sh` con `sudo shutdown -h +90` y `ncu --set roofline --csv --log-file artifacts/b1_4_ncu_roofline.csv`.
  - *Validación en `1x L4` (`sm_89`):* Confirmar reducción de `dram__bytes.sum` en `ncu` frente al modo eager y cierre del Antipatrón C pendiente de B1.3 (bajando de `~218 ms` / 600 kernels a 1 kernel fusionado).
]

#solucion-box[
  - *Sol. B1.1-A:* FLOPs paso $B=1$: $2 times 8 times 10^9 = 16 "GFLOP"$. Bytes pesos FP16: $16 "GB"$. $I = 16 "GFLOP" / 16 "GB" = 1.0 "FLOP/B"$. El codo es $I^* = 312000 / 2000 = 156 "FLOP/B"$. Como $I << I^*$, es *memory-bound*; latencia mínima = $16 "GB" / 2000 "GB/s" = 8.0 "ms"$. Para alcanzar $I^* = 156$, se necesita un batch $B approx 156$ tokens concurrentes.
  - *Sol. B1.1-B:* `t` original tiene `stride = (512, 64, 1)`. Tras `transpose(0, 1)`: `(8, 4, 64)` con `stride = (64, 512, 1)`. Tras slice `[:, ::2, :]`: `shape = (8, 2, 64)`, `stride = (64, 1024, 1)`, `is_contiguous() == False`. Elementos finales = $8 times 2 times 64 = 1024$ (`2048 B`), tráfico DRAM de `v.contiguous()` = lectura + escritura = `4096 B`.
  - *Sol. B1.2-A:* Con `32x32` (1024 hilos) y 44 regs/hilo: $1024 times 44 = 45056$ regs/bloque. En 65,536 regs/SM solo cabe *1 bloque* = 1024/2048 hilos = *50% de ocupación*. Con `16x16` (256 hilos): $256 times 44 = 11264$ regs/bloque $arrow.r$ $floor(65536 / 11264) = 5$ bloques/SM = $1280 / 2048$ = *62.5% de ocupación* (con 5 bloques independientes ocultando latencia de barreras).
  - *Sol. B1.2-B:* $M N = 1048576$ floats = `4.0 MiB` = `4.194 MB`. Tráfico mínimo útil ($2N$) = `8.39 MB`; `Online` ($3N$) = `12.58 MB`; `3-Pass` ($4N$) = `16.78 MB`.
  - *Sol. B1.3-A:* Tiempo de GPU por kernel = `362 us` frente a `3.5 us` de lanzamiento (~1% del total): estrictamente *GPU DRAM bandwidth bound*. BW por kernel = $33.55 "MB" / 362 "us" approx 92.7 "GB/s"$ (~83% del pico teórico de 112 GB/s).
]

#fuentes-box[
  1. Horace He — *Making Deep Learning Go Brrrr From First Principles*: `https://horace.io/brrr_intro.html`
  2. Modal — *GPU Glossary (Device Hardware, Memory & CUDA Model)*: `https://modal.com/gpu-glossary`
  3. Edward Z. Yang — *PyTorch Internals*: `http://blog.ezyang.com/2019/05/pytorch-internals/`
  4. Simon Boehm — *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*: `https://siboehm.com/articles/22/CUDA-MMM`
  5. NVIDIA Nsight Systems & Nsight Compute Guides: `https://docs.nvidia.com/nsight-systems/` · `https://docs.nvidia.com/nsight-compute/`
  6. OpenAI Triton Official Tutorials: `https://triton-lang.org/main/getting-started/tutorials/`
]
