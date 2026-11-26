# Bloque B1.2: Fundamentos de CUDA C++ en Local
**Estudio y Benchmark de Sistemas GPU: Coalescencia, Shared Memory Tiling y Warp Shuffles**

---

## 1. Especificaciones del Hardware Destino

| Parámetro de Arquitectura | Valor Hardware (NVIDIA GeForce GTX 1050) |
| :--- | :--- |
| **Microarquitectura** | Pascal (GP107) |
| **Compute Capability** | `sm_61` |
| **Streaming Multiprocessors (SMs)** | 5 SMs (640 CUDA Cores FP32) |
| **VRAM Total** | 2048 MB (2.0 GB) GDDR5 |
| **Ancho de Banda Teórico** | **~112.0 GB/s** (128-bit @ ~7 Gbps) |
| **Caché L2** | 1024 KB (1.0 MB) |
| **Shared Memory** | 48 KB máx. por bloque (96 KB por SM en GP107) |
| **Warp Size** | 32 hilos |
| **Máximo de hilos por bloque** | 1024 hilos |
| **Throughput Teórico FP32** | ~1.86 TFLOP/s (Boost Clock ~1455 MHz) |

---

## 2. Fundamentos de Arquitectura de Sistemas GPU

### A. Anatomía de la Coalescencia en GEMM (`gemm_naive` vs `gemm_coalesced`)

En CUDA, las instrucciones de carga/escritura de memoria global DRAM son emitidas a nivel de **Warp** (32 hilos que comparten `threadIdx.x` consecutivos de 0 a 31).

1. **`gemm_naive` (`threadIdx.x -> row`, `threadIdx.y -> col`)**:
   - Cada hilo del warp tiene el mismo `col` y un `row` distinto:
     $$\text{address}(A[\text{row}, k]) = \text{base}_A + (\text{row} \times K + k) \times 4\text{ B}$$
   - A medida que `threadIdx.x` varía de 0 a 31, las direcciones de `A` y `C` avanzan en saltos de $K \times 4$ o $N \times 4\text{ bytes}$ (ej. $2048 \times 4 = 8192\text{ bytes}$), mientras que `B[k * N + col]` es idéntica para todo el warp (**broadcast**).
   - **Consecuencia**: En `A` y `C`, cada hilo solicita un sector L2 de 32 bytes distinto para usar solo 4 bytes (**87.5% de desperdicio** del ancho de banda GDDR5 por acceso strided no coalescente).

2. **`gemm_coalesced` (`threadIdx.x -> col`, `threadIdx.y -> row`)**:
   - Cada hilo del warp tiene el mismo `row` y un `col` consecutivo (`col, col+1, col+2, ...`):
     - **Matriz A**: Todos los 32 hilos leen exactamente la misma dirección $A[\text{row} \times K + k]$. El SM realiza un **broadcast en 1 ciclo** desde L1/L2 cache.
     - **Matriz B**: Los hilos leen direcciones contiguas $B[k \times N + \text{col}]$. Los 32 accesos contiguos de 4 bytes se empaquetan en una **línea L2 de 128 bytes (4 sectores de 32 bytes)** con 100% de eficiencia.
     - **Matriz C**: Escritura contigua y coalescente de 128 bytes.
   - **Ganancia típica**: Speedup de **~4–5x** sin usar memoria compartida, solo mapeando los índices correctamente a la topología del Warp.

---

### B. Tiling Cooperativo en Shared Memory (`gemm_smem_tiled` 32×32 y variante `gemm_smem_tiled_16` 16×16)

El benchmark incluye dos tamaños de tile cooperativo para contrastar reutilización de datos frente a granularidad de ocupación en Pascal (`sm_61`):

- **`gemm_smem_tiled` (`32x32`, 1024 hilos/bloque, 8 KB SMEM)**:
  ```cpp
  __shared__ float As[32][32]; // 4096 bytes (4 KB)
  __shared__ float Bs[32][32]; // 4096 bytes (4 KB)
  ```
  - **Reutilización 32x**: Cada float cargado desde DRAM se lee 32 veces desde SMEM.
  - **Límite de ocupación**: Con 1024 hilos por bloque y tope de 2048 hilos/SM (65,536 registros/SM), caben como máximo 2 bloques/SM; si el kernel supera **32 registros/hilo**, la ocupación cae bruscamente al **50%** (1 bloque/SM) y los 32 warps del bloque se detienen a la vez en cada `__syncthreads()`.

- **`gemm_smem_tiled_16` (`16x16`, 256 hilos/bloque, 2 KB SMEM)**:
  ```cpp
  __shared__ float As[16][16]; // 1024 bytes (1 KB)
  __shared__ float Bs[16][16]; // 1024 bytes (1 KB)
  ```
  - **Ocupación más fina**: Con 256 hilos (8 warps) por bloque caben hasta **8 bloques/SM** (degradación gradual, p. ej. 75% a 40 regs/hilo) y mientras un bloque espera en `__syncthreads()` el planificador puede ejecutar warps de otros bloques.
  - **Coste**: La reutilización en SMEM baja a **16x** (doble de cargas GMEM y el doble de barreras `__syncthreads()`).

- **Cero Bank Conflicts (32 Bancos de SMEM)**:
  - En `32x32`: `As[ty][k]` realiza broadcast por warp y `Bs[k][tx]` mapea el hilo $i$ al banco $i$.
  - En `16x16`: cada warp de 32 hilos cubre dos filas consecutivas (`ty=0` y `ty=1` con `tx=0..15`). En `As[ty][k]`, ambos medios warps golpean bancos separados por 16 (`k` vs `(k+16)%32`); en `Bs[k][tx]`, ambos leen las mismas 16 palabras (`multicast`). **0 bank conflicts** en ambos casos.

- **Sincronización con Doble Barrera**:
  - **Barrera 1 (`__syncthreads()`)**: Espera a que todos los hilos del bloque carguen colaborativamente el tile completo en `As` y `Bs` antes de multiplicar.
  - **Barrera 2 (`__syncthreads()`)**: Espera a que todos los hilos terminen de consumir el tile actual antes de sobrescribir `As` y `Bs` en el siguiente paso de $K$.

---

### C. Online Softmax en 1 Pasada de Reducción e Intra-Warp Shuffle (`warp_online_softmax.cu`)

El operador Softmax estándar sobre una fila de tamaño $N$ requiere 3 pasadas sobre la memoria global DRAM:
1. $m = \max_j x_j$
2. $d = \sum_j e^{x_j - m}$
3. $y_j = \frac{e^{x_j - m}}{d}$

**Accesos lógicos tradicionales**: Lectura 1 + Lectura 2 + Lectura 3 + Escritura = $4N$ palabras ($16N\text{ bytes}$).

#### Derivación de Online Softmax (Milakov & Gimelshein / Principio FlashAttention)
Al iterar secuencialmente o en paralelo sobre los elementos, mantenemos el par acumulativo $(m, d)$:
Al descubrir un nuevo elemento $x$:
$$m_{\text{new}} = \max(m_{\text{prev}}, x)$$
$$d_{\text{new}} = d_{\text{prev}} \cdot e^{m_{\text{prev}} - m_{\text{new}}} + e^{x - m_{\text{new}}}$$

Para combinar dos resultados parciales $(m_A, d_A)$ y $(m_B, d_B)$ (entre hilos del warp):
$$m_{AB} = \max(m_A, m_B)$$
$$d_{AB} = d_A \cdot e^{m_A - m_{AB}} + d_B \cdot e^{m_B - m_{AB}}$$

Este operador asociativo se ejecuta directamente en los registros del SM mediante `__shfl_down_sync(0xffffffff, val, offset)` en un árbol mariposa de 5 saltos ($\text{offset} = 16, 8, 4, 2, 1$):
- **Cero uso de Shared Memory**: Cero consumo de SMEM, eliminando latencia de búferes compartidos.
- **Accesos lógicos reducidos**: Fusiona máximo y suma en **1 sola pasada de lectura** para estadísticas + **1 pasada de lectura/escritura** para normalización ($3N$ palabras en lugar de $4N$). El `BW Efectivo` de la tabla toma como referencia el tráfico útil mínimo teórico ($2N$ palabras: 1 lectura de $X$ + 1 escritura de $Y$).

---

## 3. Instrucciones de Compilación y Ejecución

### Prerrequisitos en la Máquina Destino (GPU NVIDIA sm_61)
- NVIDIA CUDA Toolkit $\ge 11.0$ (`nvcc` disponible en `$PATH`).
- Driver NVIDIA compatible con Pascal (`nvidia-smi` funcionando).

### A. Ejecución Idempotente con `run.sh`

```bash
# Otorgar permisos de ejecución
chmod +x run.sh

# 1. Smoke Test rápido (< 30s): valida precisión numérica en 256x256
./run.sh --smoke

# 2. Benchmark Completo: evalúa 1024x1024 y 2048x2048 (GEMM) y 1024x1024 (Softmax)
./run.sh --full

# 3. Limpieza de binarios temporales
./run.sh --clean
```

### B. Compilación Manual con `nvcc`

```bash
# Compilar gemm_bench con cuBLAS
nvcc -O3 -arch=sm_61 gemm_bench.cu -lcublas -o gemm_bench

# Compilar warp_online_softmax
nvcc -O3 -arch=sm_61 warp_online_softmax.cu -o warp_online_softmax

# Ejecutar benchmarks directamente
./gemm_bench --full
./warp_online_softmax --rows 1024 --cols 1024 --warmup 20 --iters 100
```

---

## 4. Tabla de Resultados (Medidos en GTX 1050, `artifacts/b1_2_results.*`)

### GEMM FP32 ($2048 \times 2048$, 16 MiB por matriz)

| Kernel GEMM | Dimensión (MxNxK) | Latencia Media (ms) | Rendimiento (GFLOP/s) | Speedup vs Naive | Verificación vs cuBLAS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`gemm_naive`** | 2048x2048x2048 | 714.6 ms | 24.0 GFLOP/s | 1.00x | PASS (err: 0.00e+00) |
| **`gemm_coalesced`** | 2048x2048x2048 | 148.6 ms | 115.6 GFLOP/s | **4.81x** | PASS (err: 0.00e+00) |
| **`gemm_smem_tiled`** | 2048x2048x2048 | 61.1 ms | 281.2 GFLOP/s | **11.69x** | PASS (err: 0.00e+00) |
| **`cublasSgemm (Ref)`**| 2048x2048x2048 | 10.6 ms | 1627.1 GFLOP/s | **67.68x** | PASS (Referencia dorada) |

### Softmax FP32 ($1024 \times 1024$, 4.0 MiB)

| Kernel Softmax | Dimensión (MxN) | Latencia (ms) | BW Efectivo (GB/s) | Speedup vs 3-Pass | Max Error Abs vs CPU | Estado |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard 3-Pass (Warp Shfl)** | 1024x1024 | 0.175 ms | 47.9 GB/s | 1.00x | 1.86e-09 | PASS [OK] |
| **Online 1-Pass (Warp Shfl)** | 1024x1024 | 0.134 ms | 62.7 GB/s | **1.31x** | 2.79e-09 | PASS [OK] |
