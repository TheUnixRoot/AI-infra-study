# AI Infrastructure Study - Experiment B1.1: Strides & Memory Layout

Estudio y benchmark reproducible de sistemas GPU para caracterizar el comportamiento de la jerarquía de memoria (VRAM GDDR5 y caché L2) frente a manipulaciones de layout y stride en tensores de PyTorch.

## Hardware Destino
- **GPU**: NVIDIA GeForce GTX 1050 (Pascal GP107, `sm_61`)
- **VRAM Total**: 2048 MB (2 GB) GDDR5
- **Bus de Memoria**: 128 bits @ ~7.008 Gbps
- **Ancho de Banda Teórico**: **~112.0 GB/s**
- **Caché L2**: 1024 KB (1.0 MB)

---

## Fundamentos de Arquitectura de Sistemas GPU

### 1. Dimensionamiento del Tensor y OOM Safety
- **Forma (`shape`)**: `(2048, 2048)` en FP32 (`torch.float32`, 4 bytes por elemento).
- **Número de elementos**: $2048 \times 2048 = 4{,}194{,}304$ elementos.
- **Tamaño en memoria**: $4{,}194{,}304 \times 4\text{ B} = 16{,}777{,}216\text{ B} = 16.0\text{ MiB} \approx 16.78\text{ MB}$.
- **Seguridad**: $16.0\text{ MiB} \ll 64\text{ MB}$ (umbral de seguridad) y $\ll 2048\text{ MB}$ de VRAM, eliminando riesgo de Out-Of-Memory (OOM).
- **Traspaso de Caché L2**: Dado que la GTX 1050 posee 1 MB de L2 cache, los 16 MiB garantizan que la operación mide tráfico real del bus DRAM GDDR5 y no aciertos en L2 cache.

### 2. Operación (a): `y = x.transpose(0, 1)` (Zero-Copy)
- Modifica únicamente los metadatos del descriptor del tensor en CPU (`shape: (2048, 2048)`, `stride: (1, 2048)`).
- **Puntero de VRAM**: `y.data_ptr() == x.data_ptr()`.
- **Tráfico en DRAM**: **0 Bytes**.
- **Latencia GPU**: $\approx 0.000\text{ ms}$ (no lanza kernels CUDA).

### 3. Operación (b): `z = x.transpose(0, 1).contiguous()` (Copia Física)
- Almacena físicamente los elementos en un nuevo búfer contiguo row-major (`stride: (2048, 1)`).
- **Puntero de VRAM**: `z.data_ptr() != x.data_ptr()`.
- **Tráfico en DRAM**: Lectura del tensor no contiguo ($16.78\text{ MB}$) + Escritura en el nuevo tensor contiguo ($16.78\text{ MB}$) = **$33.55\text{ MB}$ ($32.0\text{ MiB}$)**.
- **Cálculo de Ancho de Banda Efectivo**:
  $$\text{BW}_{\text{efectivo}} = \frac{2 \times \text{tensor.nbytes}}{t_{\text{segundos}} \times 10^9} \quad [\text{GB/s}]$$
- **Saturación del Bus de Memoria**:
  $$\text{Saturación} = \frac{\text{BW}_{\text{efectivo}}}{112.0\text{ GB/s}} \times 100\%$$

---

## Ejecución Reproducible

### Smoke Test (< 30s)
```bash
./run.sh --smoke-test
```

### Benchmark Completo
```bash
./run.sh
```

### Opciones Disponibles
```bash
./run.sh --help
```

### Artefactos Generados
- **JSON de Métricas**: `./artifacts/b1_1_metrics.json`
- **Consola**: Tabla con formato ANSI/ASCII detallando latencias, punteros, transferencias y saturación de ancho de banda.
