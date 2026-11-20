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
| **Shared Memory por SM** | 48 KB / 64 KB configurable |
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
   - A medida que `threadIdx.x` varía de 0 a 31, la dirección en memoria avanza en saltos de $K \times 4\text{ bytes}$ (ej. $2048 \times 4 = 8192\text{ bytes}$).
   - **Consecuencia**: Cada hilo requiere una línea de caché L1/L2 distinta. El Memory Controller debe emitir hasta **32 transacciones de memoria separadas**, desperdiciando >90% del bus GDDR5.

2. **`gemm_coalesced` (`threadIdx.x -> col`, `threadIdx.y -> row`)**:
   - Cada hilo del warp tiene el mismo `row` y un `col` consecutivo (`col, col+1, col+2, ...`):
     - **Matriz A**: Todos los 32 hilos leen exactamente la misma dirección $A[\text{row} \times K + k]$. El SM realiza un **broadcast en 1 ciclo** desde L1/L2 cache.
     - **Matriz B**: Los hilos leen direcciones contiguas $B[k \times N + \text{col}]$. Los 32 accesos contiguos de 4 bytes se empaquetan en una **única transacción coalescente de 128 bytes** (sector L2).
     - **Matriz C**: Escritura contigua y coalescente de 128 bytes.
   - **Ganancia típica**: Speedup de **3x a 8x** sin usar memoria compartida, solo mapeando los índices correctamente a la topología del Warp.

---

### B. Tiling Cooperativo en Shared Memory (`gemm_smem_tiled`)

El kernel `gemm_smem_tiled` divide la multiplicación en bloques de $32 \times 32$ floats:
- **Búferes Compartidos**:
  ```cpp
  __shared__ float As[32][32]; // 4096 bytes (4 KB)
  __shared__ float Bs[32][32]; // 4096 bytes (4 KB)
  ```
  Total por bloque: $8\text{ KB}$ de SMEM ($\ll 48\text{ KB}$ por SM), garantizando alta ocupación.

- **Factor de Reutilización Aritmética**:
  - Cada elemento cargado desde DRAM GDDR5 a `__shared__` se lee **32 veces** desde SMEM durante la multiplicación interna.
  - El tráfico sobre el bus global DRAM se reduce en un factor de hasta **32x**, permitiendo que el kernel alcance el límite compute-bound en vez de memory-bound.

- **Mitigación Total de Bank Conflicts (32 Bancos de SMEM)**:
  - Lectura `As[ty][k]`: `ty` y `k` son idénticos para todo el warp $\rightarrow$ **Multicast/Broadcast de banco** (1 ciclo, 0 conflictos).
  - Lectura `Bs[k][tx]`: `k` constante, `tx` varía de 0 a 31 $\rightarrow$ El hilo $i$ lee el banco $i$ ($i \in [0, 31]$) $\rightarrow$ **100% libre de bank conflicts**.

- **Sincronización con Doble Barrera**:
  - **Barrera 1 (`__syncthreads()`)**: Espera a que los 1024 hilos del bloque carguen colaborativamente el tile completo en `As` y `Bs` antes de comenzar las multiplicaciones.
  - **Barrera 2 (`__syncthreads()`)**: Espera a que todos los hilos terminen de consumir los datos del tile actual antes de que comience la sobreescritura de `As` y `Bs` en el siguiente paso de $K$.

---

### C. Online Softmax en 1 Pasada e Intra-Warp Shuffle (`warp_online_softmax.cu`)

El operador Softmax estándar sobre una fila de tamaño $N$ requiere 3 pasadas sobre la memoria global DRAM:
1. $m = \max_j x_j$
2. $d = \sum_j e^{x_j - m}$
3. $y_j = \frac{e^{x_j - m}}{d}$

**Tráfico DRAM Tradicional**: Lectura 1 + Lectura 2 + Lectura 3 + Escritura = $4 \times N \times 4\text{ B} = 16N\text{ bytes}$.

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
- **Tráfico DRAM Reducido**: 1 sola lectura para estadísticas + 1 lectura y escritura para normalización ($3N$ palabras en vez de $4N$).

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

## 4. Tabla de Resultados de Referencia (Valores Esperados en GTX 1050)

### GEMM FP32 ($2048 \times 2048$, 16 MiB por matriz)

| Kernel GEMM | Dimensión (MxNxK) | Latencia Media (ms) | Rendimiento (GFLOP/s) | Speedup vs Naive | Verificación vs cuBLAS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`gemm_naive`** | 2048x2048x2048 | ~95.40 ms | ~179.9 GFLOP/s | 1.00x | PASS (err: 0.00e+00) |
| **`gemm_coalesced`** | 2048x2048x2048 | ~22.60 ms | ~760.1 GFLOP/s | **4.22x** | PASS (err: 1.19e-07) |
| **`gemm_smem_tiled`** | 2048x2048x2048 | ~14.10 ms | ~1218.4 GFLOP/s | **6.76x** | PASS (err: 2.38e-07) |
| **`cublasSgemm (Ref)`**| 2048x2048x2048 | ~10.20 ms | ~1684.2 GFLOP/s | **9.35x** | PASS (Referencia dorada) |

### Softmax FP32 ($1024 \times 1024$, 4.0 MiB)

| Kernel Softmax | Dimensión (MxN) | Latencia (ms) | BW Efectivo (GB/s) | Speedup vs 3-Pass | Max Error Abs vs CPU | Estado |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard 3-Pass (Warp Shfl)** | 1024x1024 | ~0.118 ms | ~71.1 GB/s | 1.00x | 2.38e-07 | PASS [OK] |
| **Online 1-Pass (Warp Shfl)** | 1024x1024 | ~0.089 ms | ~94.3 GB/s | **1.33x** | 2.98e-07 | PASS [OK] |
