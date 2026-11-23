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
| B1.2 | Contenedor Docker Linux (`Dockerfile` del repo): CUDA 12.6 (nvcc 12.6.85), GPU por WSL2 |
| B1.3 | Python 3.11 + PyTorch 2.14.1+cu126 (Windows) con Nsight Systems 2024.5.1 nativo de Windows |

## 2. B1.1: Strides y coste de `.contiguous()` (PyTorch)

Matriz FP32 2048×2048 (16 MiB), 20 warmup + 50 iteraciones con CUDA events.

| Operación | Latencia GPU | Tráfico DRAM | Ancho de banda efectivo |
| :--- | :---: | :---: | :---: |
| `y = x.transpose(0,1)` | 0,0035 ms | 0 B (misma `data_ptr`, solo cambian los strides) | no aplica |
| `z = y.contiguous()` | 1,004 ms | 33,55 MB (16,78 MB leídos + 16,78 MB escritos) | **33,4 GB/s = 29,8 % de 112 GB/s** |
| `x.clone()` (baseline secuencial) | 0,366 ms | 33,55 MB | **91,8 GB/s = 81,9 %** |

Prueba adicional con 8192×8192 (256 MiB): `clone()` 92,7 GB/s (83 %) frente a `t().contiguous()` 8,2 GB/s (**7 %**).

Lectura: el transpose es solo metadatos. La GPU **sí** sostiene ~92 GB/s en una copia coalescida (el ~82 % del
pico, coherente con cuBLAS a ~87 % del pico FP32 y con los kernels element-wise de B1.3, ~93 GB/s según nsys).
La copia transpuesta es 2,7× más lenta con 2048² y 8× con 8192² porque lee por filas y escribe por columnas
(accesos no coalescidos), no por un problema de hardware ni de overhead de lanzamiento (el ideal a 112 GB/s
serían 0,30 ms y `clone` mide 0,37 ms). El runner de B1.1 incluye ahora el baseline `clone()`.

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

Lectura: cada mejora (coalescencia, tiling en shared memory) aumenta GFLOP/s de forma medible, que es el objetivo
del "Done" del temario. cuBLAS alcanza 1627 GFLOP/s (~87 % del pico FP32 de ~1862), así que la GPU rinde con
normalidad. El mejor kernel propio llega a ~17 % de cuBLAS. Softmax online: 1,31× más rápido y numéricamente equivalente.

Discrepancia con el README original (naive 2048³ ~95 ms / 180 GFLOP/s; tiled ~14 ms / 1218 GFLOP/s): **no se explica por
el tamaño de matriz**. Se mide en 1024³ (73,5 ms, 29 GFLOP/s) y en 2048³ (714,6 ms, 24 GFLOP/s); ninguno se acerca a
180 GFLOP/s. Además un tile 32×32 sin register blocking que alcanzase 1218 GFLOP/s (72 % de cuBLAS) es inverosímil en
Pascal. Esas cifras del README eran proyecciones no medidas y se han sustituido por las medidas.

## 4. B1.3: Profiling de antipatrones (CUDA events + trazas nsys reales)

Tensor 2048×2048, 200 operaciones element-wise, 20 iteraciones (+5 warmup). Trazas con Nsight Systems 2024.5.1
nativo de Windows (`artifacts/b1_3_bad.nsys-rep`, `artifacts/b1_3_opt.nsys-rep`).

| Métrica | BAD | OPT | Mejora |
| :--- | :---: | :---: | :---: |
| Transferencia H2D | 3,71 ms (pageable, síncrona) | 1,78 ms (`pin_memory` + `non_blocking`) | **2,08×** |
| Cómputo element-wise (200 ops, 600 kernels) | 218,2 ms | 218,1 ms | 1,00× |
| Tiempo de paso total | 223,1 ms | 220,7 ms | **1,01×** |

Datos de `nsys stats` (`cuda_gpu_kern_sum`): 3 tipos de kernel (`mul`, `add`, `clamp`) × 5000 instancias
(200 ops × 25 pasos), ~362 µs cada uno. Cada kernel mueve 33,5 MB, es decir ~93 GB/s: el cómputo está limitado por
ancho de banda DRAM, no por el lanzamiento (coste de lanzamiento estimado ~2 ms frente a 217 ms de GPU).

Notas de interpretación:
- El único antipatrón con mejora clara en el paso es B (`pin_memory`): −1,9 ms.
- La "sincronización" de BAD (210 ms) **no es coste de `.item()`**: es el tiempo que la CPU espera al cómputo ya
  encolado. Quitar `.item()` no reduce el tiempo total aquí porque el cómputo domina (99 %, Ley de Amdahl).
- Antipatrón C (muchos kernels diminutos) **no se corrige**: la fusión con `torch.compile` exige Triton (CC ≥ 7.0) y
  la GTX 1050 es 6.1. El temario deja Triton para B1.4 (cloud).

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

1. **nsys en WSL2 no sirve con esta GPU.** Con `CuptiUseRawGpuTimestamps=false` la captura ya no aborta, pero el
   importador falla con `Unrecognized GPU UUID` (WSL2 no expone la identidad de la GPU). La versión 2024.5.1 sí soporta
   Pascal; el bloqueo es WSL2. Solución aplicada: Nsight Systems 2024.5.1 **nativo de Windows**, que traza CUDA
   correctamente. `perf` no existe en Windows ni funciona en el kernel de WSL2 (queda para Linux nativo / cloud).
2. **Triton/ncu no disponibles** en sm_61 (previsto por el temario: B1.4 en cloud).
3. **Cifras del README original no reproducidas** (proyecciones, no medidas): ya sustituidas por valores medidos.
   Causa descartada: throttling (cuBLAS a ~87 % del pico) y tamaño de matriz (ver sección 3).
4. **Defectos del código corregidos durante la revisión:** `kv_calc.py` marcaba "OK" con 0 secuencias; `run.sh` de B1.2
   pasaba mal los flags a nvcc; los CSV/JSON de B1.2 no contenían medidas; B1.3 generaba trazas `.nsys-rep` y tablas
   nsys falsas, rellenaba con datos simulados el modo ausente, y su runner usaba `osrt` (solo Linux) con nsys de Windows.

## 7. Propuestas para trabajar a partir de aquí

- B1.4 (cloud, L4/L40S): fusión con Triton para cerrar el antipatrón C y comparar contra los 218 ms medidos; también
  `perf` y nsys en Linux nativo con el `Dockerfile` del repo (`--build-arg CUDA_ARCH=sm_89`).
- Ampliar B1.3 con un bucle con más trabajo de CPU por paso, para que la pérdida de solapamiento CPU-GPU causada por
  `.item()` sea visible en el tiempo total.
- Estudiar en B1.2 por qué `smem_tiled` se queda en ~17 % de cuBLAS (register blocking, vectorización).
