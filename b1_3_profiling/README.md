# Bloque B1.3: Profiling con NVIDIA Nsight Systems (nsys) y Linux perf
**Estudio de Sistemas GPU: Pipeline CPU-GPU, Memoria Pinned, Fusión de Kernels y NVTX**

---

## 1. Especificaciones del Hardware Destino

| Parámetro de Arquitectura | Valor Hardware (NVIDIA GeForce GTX 1050) |
| :--- | :--- |
| **Microarquitectura** | Pascal (GP107) |
| **Compute Capability** | `sm_61` |
| **Streaming Multiprocessors (SMs)** | 5 SMs (640 CUDA Cores FP32) |
| **VRAM Total** | 2048 MB (2.0 GB) GDDR5 |
| **Ancho de Banda Teórico VRAM** | **~112.0 GB/s** (128-bit @ ~7.008 Gbps) |
| **Caché L2 Global** | 1024 KB (1.0 MB) |
| **Bus Host-Device** | PCIe 3.0 x16 (Límite teórico ~15.75 GB/s, efectivo ~11.5 - 12.5 GB/s) |
| **Umbral de Seguridad VRAM** | < 256 MB por tensor para evitar OOM |

---

## 2. Anatomía de los 3 Antipatrones y Soluciones de Arquitectura

```
            +-------------------------------------------------------------+
            |                    MODO BAD (Antipatrones)                  |
            |                                                             |
   Host     | [Pageable Mem] --(Memcpy)--> [Pinned Buffer] --(DMA Sync)-> |
            |                              ^                              |
   CPU-GPU  | CPU bloqueada esperando DMA  |                              |
            |                                                             |
   GPU Kern | [Kernel 1] -> [Kernel 2] -> ... -> [Kernel 600]             |
            | 200 ops x 3 kernels/op, DRAM roundtrips (No L2 reuse)      |
            |                                                             |
   Sync     | [ .item() ] ===> cudaStreamSynchronize (Pipeline Stall)     |
            | CPU y GPU completamente serializadas (Gaps gigantes en timeline)
            +-------------------------------------------------------------+

            +-------------------------------------------------------------+
            |                     MODO OPT (Optimizado)                   |
            |                                                             |
   Host     | [Pinned Memory (Page-locked)] ----(Async DMA non_blocking)->|
            |                              ^                              |
   CPU-GPU  | CPU encola trabajo asíncrono sin bloquearse                |
            |                                                             |
   GPU Kern | [CC >= 7.0 (B1.4): Triton fused] / [sm_61: 600 kernels]     |
            | Fusión con torch.compile/Inductor requiere CC >= 7.0        |
            |                                                             |
   Sync     | [ loss += step_loss ] (GPU Tensor, ZERO Sync en bucle)      |
            | Sincronización diferida fuera del bucle de entrenamiento    |
            +-------------------------------------------------------------+
```

### A. Antipatrón A: `.item()` en el Bucle Caliente (Pipeline Bubble)
- **Causa Raíz**: PyTorch despacha operaciones a una cola de comandos asíncrona (`cudaStream_t`). Invocar `.item()`, `float(t)` o `t.tolist()` en cada iteración fuerza un `cudaStreamSynchronize` / `cudaMemcpy(D2H)` síncrono.
- **Impacto**: La CPU se bloquea hasta que la GPU termine todo el trabajo en cola y transfiera el escalar de vuelta. Esto destruye el solapamiento (*pipelining*) CPU-GPU, generando burbujas donde los SMs permanecen ociosos esperando que el hilo de Python despache el siguiente lote.
- **Solución**: Mantener las métricas y pérdidas como tensores en el dispositivo (`total_loss += step_loss.detach()`). La llamada a `.item()` se realiza una sola vez al final del experimento o periódicamente cada $N$ pasos.

### B. Antipatrón B: Transferencia H2D sin Memoria Pinned ni Asíncrona
- **Causa Raíz**: La memoria reservada por defecto en CPU es **paginable** (*pageable virtual memory*). El motor DMA de la GPU no puede leer con seguridad desde memoria paginable porque el kernel del sistema operativo podría reubicar o paginar esas páginas a disco en cualquier instante.
- **Impacto**: El driver CUDA debe realizar una doble copia:
  1. Copia física de la memoria paginable a un búfer intermedio *pinned* (staging buffer) en CPU mediante `memcpy`.
  2. Emisión de la transferencia DMA por el bus PCIe de forma síncrona (`cudaMemcpy`), bloqueando el hilo de ejecución de Python.
- **Solución**: Asignar memoria de host bloqueada en páginas físicas (`torch.empty(..., pin_memory=True)` o `.pin_memory()`) y transferir con `.to(device, non_blocking=True)`. Permite transferencias directas sobre PCIe vía `cudaMemcpyAsync` solapadas con la ejecución de kernels.

### C. Antipatrón C: 200 Operaciones Element-wise Diminutas sin Fusionar (600 Kernels)
- **Causa Raíz**: Encadenar 200 pasos de `x = torch.relu(x * 1.0001 + 0.0001)` en modo eager genera **3 kernels por paso** (`mul`, `add`, `clamp`/`relu`) = **600 llamadas `cudaLaunchKernel`** consecutivas.
- **Impacto**:
  1. *Driver Launch Overhead*: ~3.5 µs de sobrecarga por kernel $\times 600 \approx 2.1\text{ ms}$ de despacho CPU por paso.
  2. *Saturación de DRAM GDDR5*: Para una matriz de $16.0\text{ MiB}$ ($16.78\text{ MB}$), cada kernel lee $16.78\text{ MB}$ y escribe $16.78\text{ MB}$ ($33.55\text{ MB}$). 600 kernels generan $600 \times 33.55\text{ MB} \approx 20.1\text{ GB}$ de tráfico DRAM por paso. Como la caché L2 de la GTX 1050 es de 1024 KB, se desborda de inmediato y el cómputo queda estrictamente *memory-bandwidth bound* (~93 GB/s según `nsys`).
- **Solución**: Fusión de operadores mediante `torch.compile` (Inductor/Triton, abordado en **B1.4** sobre GPUs con `CC >= 7.0`), manteniendo los valores intermedios en registros y realizando 1 sola lectura y 1 escritura a DRAM. En Pascal `sm_61` (`GTX 1050`), Triton no está soportado; el módulo diagnostica el cuello de botella sin simular fusión falsa.

*(Nota: puedes pasar `--cpu-work-ms 5` para simular trabajo CPU por paso y observar directamente la pérdida de solapamiento causada por `.item()` en el Antipatrón A).*

---

## 3. Instrumentación NVTX y Profiling de Sistema

### A. Anotación con NVTX (`torch.cuda.nvtx`)
El script [`b1_3_antipatterns.py`](b1_3_antipatterns.py) anota cada etapa:
```python
# Delimitación de transferencia PCIe
torch.cuda.nvtx.range_push("H2D_Transfer")
gpu_tensor = host_tensor.to(device, non_blocking=True)
torch.cuda.nvtx.range_pop()

# Delimitación de cómputo
torch.cuda.nvtx.range_push("Compute_Fused_Kernel")
gpu_tensor = fused_fn(gpu_tensor)
torch.cuda.nvtx.range_pop()
```

### B. Comando de Perfilado con `nsys`
```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite=true \
  -o ../artifacts/b1_3_profile \
  python3 b1_3_antipatterns.py --mode bad
```
- `--trace=cuda,nvtx,osrt`: Captura llamadas CUDA API, eventos NVTX y tiempos de runtime del SO.
- `--stats=true`: Extrae automáticamente al finalizar las tablas estadísticas `cuda_api_sum` y `cuda_gpu_kern_sum`.

---

## 4. Instrucciones de Ejecución (en Host con GTX 1050)

```bash
cd b1_3_profiling

# 1. Smoke test rápido (< 30s)
./run.sh --smoke-test

# 2. Benchmark completo (Tensor 2048x2048, 20 iteraciones, 200 ops)
./run.sh

# 3. Ayuda y parámetros configurables
./run.sh --help
```

### Inspección Gráfica de Trazas
```bash
nsys-ui ../artifacts/b1_3_bad.nsys-rep &
nsys-ui ../artifacts/b1_3_opt.nsys-rep &
```

---

## 5. Rendimiento Medido (GTX 1050 sm_61)

Opt C (fusión) no está disponible en sm_61: Triton exige CC >= 7.0 (se aborda en B1.4, cloud). Trazas reales con nsys 2024.5.1 nativo de Windows (`nsys` en WSL2 no identifica la GPU).

Tensor FP32 de $2048 \times 2048$ ($16.0\text{ MiB}$), 200 operaciones element-wise:

| Métrica / Fase | Modo BAD (Antipatrones) | Modo OPT (Optimizado) | Factor de Mejora (Speedup) | Mecanismo de Aceleración |
| :--- | :---: | :---: | :---: | :--- |
| **Transferencia H2D** | 3.71 ms | 1.78 ms | **2.08x** | Pinned memory + DMA directo PCIe 3.0 x16 |
| **Element-wise Compute** | 218.2 ms | 218.1 ms | **1.00x** | Sin fusión en sm_61 (200 ops = 600 kernels, ~362 us cada uno según nsys) |
| **Sincronización (.item)** | 210.2 ms (espera al cómputo encolado) | 0.12 ms | n/a | `.item()` bloquea la CPU hasta que acaba la cola; no añade cómputo |
| **Latencia Total por Paso** | **223.1 ms** | **220.7 ms** | **1.01x** | Pipelining continuo CPU-GPU |
| **Kernel Launches por Paso** | 600 launches | 600 launches | 1.0x | Sin fusión: pendiente de B1.4 |
