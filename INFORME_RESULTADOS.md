# Informe de resultados: prácticas B1.1, B1.2, B1.3 y B2.1

Learning path MLOps / ML Engineer (AI Infra). Repositorio: `TheUnixRoot/AI-infra-study`.
Todas las cifras proceden de los artefactos de `artifacts/` generados por los runners del repo
(commit `2e29363`). Ninguna es simulada salvo que se indique.

## 1. Entorno de ejecución

| Elemento | Valor |
| :--- | :--- |
| GPU | NVIDIA GeForce GTX 1050, Pascal GP107, `sm_61`, 2 GB GDDR5 (~112 GB/s teóricos) |
| Host | Windows 10, driver NVIDIA 581.57 |
| B1.1, B2.1 | Python 3.11 + PyTorch 2.14.1+cu126 en una venv del repo (Windows) |
| B1.2, B1.3 | Contenedor Docker Linux (`Dockerfile` del repo): CUDA 12.6 (nvcc 12.6.85), Python 3.10, PyTorch 2.14.1+cu126, GPU por WSL2 |

## 2. B1.1: Strides y coste de `.contiguous()` (PyTorch)

Matriz FP32 2048×2048 (16 MiB), 20 warmup + 50 iteraciones con CUDA events.

| Operación | Latencia GPU | Tráfico DRAM | Ancho de banda efectivo |
| :--- | :---: | :---: | :---: |
| `y = x.transpose(0,1)` | 0,0035 ms | 0 B (misma `data_ptr`, solo cambian los strides) | no aplica |
| `z = y.contiguous()` | 1,004 ms | 33,55 MB (16,78 MB leídos + 16,78 MB escritos) | **33,4 GB/s = 29,8 % de 112 GB/s** |

Lectura: el transpose es solo metadatos; la copia física es la que cuesta y alcanza ~30 % del
ancho de banda teórico. Un acceso transpuesto (strides no unitarios) es poco amigable con la
coalescencia, lo que explica que quede lejos del pico.

## 3. B1.2: CUDA C++ (coalescencia, tiling y warp shuffles)

GEMM FP32 comparado con cuBLAS (50 iteraciones medidas, todas las verificaciones numéricas PASS).

| Kernel | 1024³: ms / GFLOP/s | 2048³: ms / GFLOP/s | Speedup vs naive (2048³) |
| :--- | :---: | :---: | :---: |
| `gemm_naive` | 73,5 / 29,2 | 714,6 / 24,0 | 1,00× |
| `gemm_coalesced` | 18,0 / 119,6 | 148,6 / 115,6 | 4,81× |
| `gemm_smem_tiled` | 7,6 / 283,0 | 61,1 / 281,2 | 11,69× |
| `cublasSgemm` (referencia) | 1,4 / 1563,7 | 10,6 / 1627,1 | 67,68× |

Softmax 1024×1024 (4 MiB):

| Kernel | Latencia | BW efectivo | Error máx. vs CPU |
| :--- | :---: | :---: | :---: |
| 3 pasadas (warp shuffle) | 0,1752 ms | 47,9 GB/s | 1,9e-9 |
| Online 1 pasada (warp shuffle) | 0,1339 ms | 62,7 GB/s | 2,8e-9 |

Lectura: cada mejora (coalescencia, tiling en shared memory) aumenta GFLOP/s de forma medible,
que es el objetivo del "Done" del temario. Aun así el mejor kernel propio alcanza ~17 % de cuBLAS.
Softmax online: 1,31× más rápido y numéricamente equivalente.

## 4. B1.3: Profiling de antipatrones (CUDA events; sin traza nsys)

Tensor 2048×2048, 200 operaciones element-wise, 20 iteraciones (+5 warmup).

| Métrica | BAD | OPT | Mejora |
| :--- | :---: | :---: | :---: |
| Transferencia H2D | 4,04 ms (pageable, síncrona) | 1,93 ms (`pin_memory` + `non_blocking`) | **2,09×** |
| Cómputo element-wise (200 ops, 600 kernels) | 218,0 ms | 218,1 ms | 1,00× |
| Tiempo de paso total | 223,4 ms | 221,0 ms | **1,01×** |

Notas de interpretación (importante):
- El único antipatrón con mejora clara en el tiempo de paso es B (`pin_memory`): −2,1 ms.
- La "sincronización" de BAD (210,6 ms) **no es coste de `.item()`**: es el tiempo que la CPU espera
  al cómputo ya encolado. Quitar el `.item()` del bucle no reduce el tiempo total aquí, porque el
  cómputo domina (99 %). El problema de `.item()` es perder el solapamiento CPU-GPU, que con tan poca
  CPU por paso no se nota en esta configuración.
- Antipatrón C (muchos kernels diminutos) **no se corrige**: la fusión con `torch.compile` exige
  Triton (compute capability ≥ 7.0) y la GTX 1050 es 6.1. El temario deja Triton para B1.4 (cloud).
- Estas cifras salen de CUDA events, no de nsys. Los `.nsys-rep` y las tablas `nsys stats` no existen
  (ver sección 6).

## 5. B2.1: Aritmética del KV-cache (CPU, analítico)

Secuencia 8192, `gpu_mem_util=0.90`, overhead 2 GB. Fórmula GQA: 2·L·n_kv·d_head·bytes; MLA (DeepSeek):
L·(kv_lora_rank + qk_rope_head_dim)·bytes. Peticiones concurrentes máximas (8k tokens):

| Modelo | Precisión (pesos + KV) | KV/token | KV/secuencia | L40S 48 GB | H100 80 GB |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Llama-3-8B (GQA) | FP16 + FP16 | 128 KiB | 1,00 GiB | 26 | 55 |
| Llama-3-8B | FP8 + FP16 | 128 KiB | 1,00 GiB | 33 | 62 |
| Llama-3-8B | FP8 + FP8 | 64 KiB | 0,50 GiB | 67 | 125 |
| Qwen2.5-72B (GQA) | FP16 + FP16 | 320 KiB | 2,50 GiB | 0 (OOM pesos) | 0 (OOM pesos) |
| Qwen2.5-72B | FP8 + FP16 | 320 KiB | 2,50 GiB | 0 (OOM pesos) | 0 (< 1 secuencia) |
| Qwen2.5-72B | FP8 + FP8 | 160 KiB | 1,25 GiB | 0 (OOM pesos) | 1 |
| DeepSeek-V3 (MLA) | cualquiera | 68,6 / 34,3 KiB | 0,54 / 0,27 GiB | 0 (pesos 625–1250 GiB) | 0 (necesita 9–18 H100 solo para pesos) |

Lectura: FP8 en el KV-cache duplica la concurrencia; MLA (DeepSeek-V3) usa ~4,7× menos KV/token que Qwen2.5-72B (GQA).
Los valores de Llama-3-8B coinciden con el cálculo manual (128 KiB/token).

## 6. Limitaciones y problemas encontrados

1. **nsys no puede trazar CUDA** en este equipo: Nsight Systems 2024.5 dentro de WSL2 con GPU Pascal
   falla con `No GPU associated to the given UUID` (solo funciona `--trace=nvtx,osrt`, sin GPU). Por
   tanto B1.3 no cumple todavía el "Done" (resumen `nsys stats`). `perf` tampoco funciona en el kernel WSL2.
2. **Triton/ncu no disponibles** en sm_61 (previsto por el temario: B1.4 en cloud).
3. **Cifras del README raíz no reproducidas.** Las tablas "esperadas" difieren de lo medido (p. ej.
   GEMM naive 2048³: README ~95 ms, medido ~715 ms; tiled: ~14 ms frente a ~61 ms; B1.3: 51× frente a
   1,01×). El origen de las cifras del README no está verificado; los artefactos previos de B1.1 y B1.3
   eran simulados (`is_simulation: true`). Causa de la diferencia sin determinar (posibles: reloj de la
   GPU bajo carga, GPU compartida con el display en Windows, otra GPU).
4. **Defectos del código corregidos durante la revisión:** `kv_calc.py` marcaba "OK" con 0 secuencias;
   `run.sh` de B1.2 pasaba mal los flags a nvcc; los CSV/JSON de B1.2 no contenían medidas; B1.3
   generaba trazas `.nsys-rep` y tablas nsys falsas y rellenaba con datos simulados el modo ausente.
   Todo eliminado: los runners ya no fabrican resultados.

## 7. Propuestas para trabajar a partir de aquí

- Obtener trazas nsys reales: instalar Nsight Systems en Windows, o hacerlo en la ráfaga cloud
  (L4/L40S con Linux nativo) reutilizando el `Dockerfile` del repo (`--build-arg CUDA_ARCH=sm_89`).
- B1.4 (cloud): fusión con Triton para cerrar el antipatrón C y comparar contra los 218 ms medidos.
- Investigar por qué GEMM naive/tiled son 4–7× más lentos que el README (relojes con `nvidia-smi -q -d CLOCK`).
- Decidir con el profesor si B1.3 debe seguir usando `.item()`+H2D+kernels como único experimento, o
  añadir un bucle con CPU suficiente para que el solapamiento CPU-GPU sea visible.
