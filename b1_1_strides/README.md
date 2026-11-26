# Bloque B1.1: Strides y Jerarquía de Memoria en PyTorch
**Estudio de Sistemas GPU: Descriptores de Metadatos, Copia Física en VRAM y Ancho de Banda GDDR5**

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
| **Umbral de Seguridad OOM** | < 64.0 MiB por tensor para evitar fragmentación |

---

## 2. Fundamentos de Arquitectura de Sistemas GPU

### A. Anatomía de `x.transpose(0, 1)` (Zero-Copy View)
1. **Manipulación de Metadatos**:
   - En PyTorch, un tensor en CUDA consta de un bloque de memoria contiguo en VRAM (`Storage`) y un descriptor ligero en CPU (`TensorImpl` con `shape`, `stride` y `storage_offset`).
   - Al ejecutar `y = x.transpose(0, 1)`, PyTorch **no invoca ningún kernel de GPU ni transfiere datos en VRAM**.
   - Solo intercambia los valores de `stride`:
     $$\text{Original } x: \text{shape } (2048, 2048), \text{stride } (2048, 1)$$
     $$\text{Transpuesto } y: \text{shape } (2048, 2048), \text{stride } (1, 2048)$$
   - Los punteros físicos son idénticos: `y.data_ptr() == x.data_ptr()`.
   - **Coste**: Tráfico DRAM = $0\text{ Bytes}$, latencia GPU $\approx 0.0036\text{ ms}$ (cero kernels lanzados), sobrecarga puramente de mutación de descriptor en CPU (~2-5 µs; si se cronometra junto con `cudaDeviceSynchronize()`, el overhead de sincronización sube a ~130 µs).

### B. Anatomía de `z = y.contiguous()` (Copia Física y Reordenamiento en DRAM)
1. **Reordenamiento Físico en VRAM**:
   - Como `y` es discontinua en memoria (`y.is_contiguous() == False`), llamar a `.contiguous()` obliga al driver de PyTorch a asignar un nuevo búfer en VRAM y lanzar un kernel CUDA de copia con lectura discontinua.
2. **Volumen de Tráfico en el Bus GDDR5**:
   - Para una matriz FP32 de $2048 \times 2048$ ($16.0\text{ MiB} = 16.78\text{ MB}$):
     - **Lectura**: $16.78\text{ MB}$ (lectura/escritura no coalescente en una dimensión).
     - **Escritura**: $16.78\text{ MB}$ (contigua fila por fila).
     - **Tráfico Total**: $2 \times 16.78\text{ MB} = 33.55\text{ MB}$ ($32.0\text{ MiB}$).
3. **Saturación de Caché L2**:
   - Como el tensor mide $16.0\text{ MiB}$ y la caché L2 de la GTX 1050 es de solo $1\text{ MB}$, el tráfico desborda la caché L2 y fuerza la saturación real del bus DRAM GDDR5.
4. **Cálculo de Ancho de Banda Efectivo y Baseline `x.clone()`**:
   $$\text{BW}_{\text{efectivo}} (\text{GB/s}) = \frac{2 \times \text{tensor.nbytes}}{\text{tiempo}_{\text{GPU}} (\text{s}) \times 10^9}$$
   $$\text{Saturación de Bus (\%)} = \frac{\text{BW}_{\text{efectivo}}}{112.0\text{ GB/s}} \times 100$$
   - En la GTX 1050 real, la copia secuencial coalescente (`x.clone()`) alcanza **91.8 GB/s (81.9%)**, mientras que `y.contiguous()` tras transponer cae a **33.4 GB/s (29.9%)**: una penalización de **2.74x** debida íntegramente al patrón de acceso strided no coalescente.

---

## 3. Instrucciones de Ejecución (en Host con GTX 1050)

```bash
cd b1_1_strides

# 1. Smoke test rápido (< 30s)
./run.sh --smoke-test

# 2. Benchmark Completo (20 warmup + 50 iteraciones cronometradas con CUDA Events)
./run.sh

# 3. Opciones y ayuda
./run.sh --help
```

### Opciones Disponibles
- `--smoke-test`: Ejecuta validación rápida con matriz reducida ($512 \times 512$).
- `--size N`: Dimensión de la matriz (por defecto: 2048 -> 16 MiB FP32).
- `--iterations N`: Iteraciones de medición (por defecto: 50).
- `--warmup N`: Iteraciones de calentamiento de caché/relojes GPU (por defecto: 20).
- `--output PATH`: Ruta personalizada para el JSON de métricas (por defecto: `../artifacts/b1_1_metrics.json`).

---

## 4. Rendimiento Medido (GTX 1050 sm_61, `artifacts/b1_1_metrics.json`)

Matriz FP32 de $2048 \times 2048$ ($16.0\text{ MiB}$ / $16.78\text{ MB}$, 4,194,304 elementos):

| Operación | Tipo | Tráfico DRAM | Latencia GPU (Eventos) | Ancho de Banda Efectivo | Saturación Bus GDDR5 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `y = x.transpose(0, 1)` | Zero-Copy (Metadatos) | 0 Bytes | 0.0036 ms | N/A (Descriptor CPU) | 0.0% |
| `z = y.contiguous()` | Copia Física Strided en VRAM | 33.55 MB | 1.004 ms | 33.44 GB/s | **29.9%** |
| `x.clone()` (baseline) | Copia Física Coalescente en VRAM | 33.55 MB | 0.366 ms | 91.77 GB/s | **81.9%** |
