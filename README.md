# AI Infrastructure Study: Benchmarking de Sistemas GPU

Repositorio reproducible de ingeniería de sistemas GPU enfocado en la jerarquía de memoria (VRAM GDDR5, Caché L2, Shared Memory y Registros de SM) y optimización de kernels para deep learning.

## Hardware Destino de Referencia

| Parámetro de Arquitectura | Especificación Hardware (NVIDIA GeForce GTX 1050) |
| :--- | :--- |
| **Microarquitectura** | Pascal (GP107) |
| **Compute Capability** | `sm_61` |
| **Streaming Multiprocessors (SMs)** | 5 SMs (640 CUDA Cores FP32) |
| **VRAM Total** | 2048 MB (2.0 GB) GDDR5 |
| **Bus de Memoria** | 128 bits @ ~7.008 Gbps |
| **Ancho de Banda Teórico** | **~112.0 GB/s** |
| **Caché L2 Global** | 1024 KB (1.0 MB) |
| **Shared Memory por SM** | 48 KB (hasta 64 KB configurable) |
| **Warp Size** | 32 hilos en lock-step |
| **Throughput Teórico FP32** | ~1.86 TFLOP/s (Boost Clock ~1455 MHz) |

---

## Bloques de Estudio

```
AI-infra-study/
├── b1_1_strides/              # B1.1: PyTorch Strides, VRAM Views vs Contiguous Copy
│   ├── b1_1_strides_bench.py
│   ├── run.sh
│   └── README.md
├── b1_2_cuda_gemm/            # B1.2: CUDA C++ Nativo (Coalescencia, Tiling y Warp Shuffles)
│   ├── gemm_bench.cu
│   ├── warp_online_softmax.cu
│   ├── Makefile
│   ├── run.sh
│   └── README.md
├── b1_3_profiling/            # B1.3: Profiling con nsys y Linux perf (Antipatrones y NVTX)
│   ├── b1_3_antipatterns.py
│   ├── run.sh
│   └── README.md
└── artifacts/                 # Salidas JSON/CSV y trazas generadas en runtime
    ├── b1_1_metrics.json
    ├── b1_2_results.json
    └── b1_2_results.csv
```

---

## Bloque B1.1: Strides y Jerarquía de Memoria en PyTorch

Estudio del impacto de la memoria discontinua y el coste de reordenamiento físico en VRAM frente a modificaciones de descriptores de metadatos.

Para teoría detallada y microarquitectura, consulta [`b1_1_strides/README.md`](file:///data/data/com.termux/files/home/raiz/AI-infra-study/b1_1_strides/README.md).

### Conceptos Clave
1. **OOM Safety**: Tensor FP32 de $2048 \times 2048 = 16.0\text{ MiB} \ll 64\text{ MB}$ (umbral de seguridad para 2GB VRAM).
2. **Traspaso de Caché L2**: Los $16\text{ MiB}$ saturan el $1\text{ MB}$ de L2 cache, forzando tráfico real contra el bus DRAM GDDR5.
3. **`y = x.transpose(0, 1)` (Zero-Copy)**: Modifica únicamente el descriptor en CPU (`stride: (1, 2048)`). Mantiene el puntero `y.data_ptr() == x.data_ptr()`. Tráfico DRAM = 0 Bytes, latencia GPU $\approx 0.000\text{ ms}$.
4. **`z = y.contiguous()` (Copia Física)**: Asigna un nuevo búfer row-major (`stride: (2048, 1)`). Requiere lectura ($16.78\text{ MB}$) + escritura ($16.78\text{ MB}$) = $33.55\text{ MB}$. Mide el ancho de banda efectivo del bus GDDR5.

### Instrucciones de Ejecución (B1.1)

```bash
cd b1_1_strides

# 1. Smoke Test rápido (< 30s)
./run.sh --smoke-test

# 2. Benchmark Completo (20 warmup + 50 iteraciones cronometradas con CUDA Events)
./run.sh

# 3. Opciones y ayuda
./run.sh --help
```

---

## Bloque B1.2: Fundamentos de CUDA C++ en Local

Estudio a nivel de ensamblador y kernel CUDA de los cuellos de botella fundamentales: coalescencia de memoria global, reutilización en Shared Memory y reducciones en registros mediante Warp Shuffles.

### Conceptos Clave
1. **Anatomía de la Coalescencia (`gemm_naive` vs `gemm_coalesced`)**:
   - `gemm_naive`: Asigna `threadIdx.x -> row`, provocando que hilos consecutivos de un warp salten $K \times 4\text{ bytes}$ en memoria. Emite hasta 32 transacciones por instrucción de carga DRAM.
   - `gemm_coalesced`: Asigna `threadIdx.x -> col`, permitiendo que los 32 hilos accedan a direcciones contiguas empaquetadas en transacciones óptimas de 128 bytes (sector L2) y un broadcast de 1 ciclo para la matriz A. Speedup medido: **4.8x** (2048³, GTX 1050).
2. **Tiling Cooperativo en Shared Memory (`gemm_smem_tiled`)**:
   - Búferes compartidos `__shared__ float As[32][32]` y `Bs[32][32]` (8 KB por bloque).
   - Factor de reutilización: Cada float cargado de DRAM se reutiliza **32 veces** en SMEM, reduciendo el tráfico DRAM hasta en 32x.
   - Cero conflictos de banco (Bank Conflicts) gracias al patrón de acceso alineado. Speedup medido: **11.7x** (~281 GFLOP/s, ~17 % de cuBLAS).
3. **Warp Online Softmax en 1 Pasada (`warp_online_softmax.cu`)**:
   - Implementa la formulación de Milakov & Gimelshein (fundamento de FlashAttention).
   - Mantiene estadísticas acumulativas de máximo y suma escalada $(m, d)$ en registros privados.
   - Reducción intra-warp mediante `__shfl_down_sync` en 5 ciclos de mariposa.
   - Cero uso de Shared Memory y reducción de pasadas sobre DRAM de 3 a 1. Speedup medido: **1.31x** vs 3-pass tradicional.

### Instrucciones de Ejecución (B1.2)

```bash
# Cambiar al directorio del experimento B1.2
cd b1_2_cuda_gemm

# 1. Otorgar permisos de ejecución (si no los tiene)
chmod +x run.sh

# 2. Smoke Test rápido (< 30s, dimensión 256x256 con verificación numérica vs cuBLAS y CPU)
./run.sh --smoke

# 3. Benchmark Completo (dimensiones 1024x1024 y 2048x2048)
./run.sh --full

# 4. Limpieza de binarios generados
./run.sh --clean
```

### Compilación Alternativa con Makefile o NVCC

```bash
cd b1_2_cuda_gemm

# Con Makefile:
make smoke     # Compila y ejecuta smoke test
make full      # Compila y ejecuta benchmark completo
make clean     # Elimina binarios

# Manual con NVCC:
nvcc -O3 -arch=sm_61 gemm_bench.cu -lcublas -o gemm_bench
nvcc -O3 -arch=sm_61 warp_online_softmax.cu -o warp_online_softmax

./gemm_bench --full
./warp_online_softmax --rows 1024 --cols 1024 --warmup 20 --iters 100
```

---

## Tabla Resumen de Rendimiento Medido (GTX 1050)

Valores medidos en la GTX 1050 (`artifacts/b1_2_results.*`, `artifacts/b1_3_summary.json`). Las tablas "esperadas" anteriores eran proyecciones no reproducidas.

### GEMM FP32 ($2048 \times 2048$, 16 MiB por matriz)
| Kernel | Mapeo / Técnica | Latencia Media | Throughput | Speedup vs Naive | Verificación |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `gemm_naive` | `tx->row, ty->col` (No Coalescente) | 714.6 ms | 24.0 GFLOP/s | 1.00x | PASS |
| `gemm_coalesced` | `tx->col, ty->row` (Coalescente 128B) | 148.6 ms | 115.6 GFLOP/s | **4.81x** | PASS |
| `gemm_smem_tiled` | Tile 32x32 SMEM + Doble `__syncthreads` | 61.1 ms | 281.2 GFLOP/s | **11.69x** | PASS |
| `cublasSgemm` | Librería optimizada NVIDIA (Límite Teórico) | 10.6 ms | 1627.1 GFLOP/s | **67.68x** | Golden Ref |

### Softmax FP32 ($1024 \times 1024$, 4 MiB)
| Kernel | Técnica de Reducción | Latencia Media | Ancho de Banda Efectivo | Speedup | Estado |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Standard 3-Pass | 3 lecturas DRAM + Warp Shuffles | 0.175 ms | 47.9 GB/s | 1.00x | PASS [OK] |
| Online 1-Pass | 1 lectura DRAM + Registros + Warp Shuffles | 0.134 ms | 62.7 GB/s | **1.31x** | PASS [OK] |

---

## Bloque B1.3: Profiling con NVIDIA Nsight Systems (nsys) y Linux perf

Estudio empírico de cuellos de botella en el pipeline CPU-GPU: serialización por `.item()`, transferencias H2D síncronas paginables vs pinned DMA, y sobrecarga de kernel launches vs fusión con `torch.compile`.

Para teoría detallada, microarquitectura y análisis NVTX, consulta la documentación dedicada del módulo en [`b1_3_profiling/README.md`](file:///data/data/com.termux/files/home/raiz/AI-infra-study/b1_3_profiling/README.md).

### Instrucciones de Ejecución (B1.3)

```bash
cd b1_3_profiling

# 1. Smoke test rápido (< 30s)
./run.sh --smoke-test

# 2. Benchmark completo con nsys y Linux perf
./run.sh

# 3. Opciones configurables
./run.sh --help
```

### Resumen de Resultados Medidos (GTX 1050 / sm_61)

La fusión (Opt C) no está disponible en sm_61 (Triton exige CC >= 7.0): se difiere a B1.4 (cloud).

| Métrica / Fase | Modo BAD (Antipatrones) | Modo OPT (Optimizado) | Factor de Mejora | Mecanismo |
| :--- | :---: | :---: | :---: | :--- |
| **H2D Transfer** | 3.71 ms | 1.78 ms | **2.08x** | Pinned DMA directo PCIe |
| **Element-wise Compute** | 218.2 ms | 218.1 ms | **1.00x** | Sin fusión en sm_61 (pendiente B1.4) |
| **Sync (.item)** | 210.2 ms (espera al cómputo encolado) | 0.12 ms | n/a | Acumulación asíncrona en GPU; no reduce el tiempo total |
| **Latencia por Paso** | **223.1 ms** | **220.7 ms** | **1.01x** | Domina el cómputo sin fusionar |
| **Kernel Launches** | 600 launches | 600 launches | 1.0x | Sin fusión (nsys: 3 kernels x 200 ops) |

