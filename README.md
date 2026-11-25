# AI Infra Study

Learning path de **AI Infrastructure** (GPU, inferencia y benchmarking de clústeres) con prácticas reproducibles.
Este README es el **índice del temario**: cada unidad enlaza a su práctica, a sus resultados y, cuando exista, a su documento.

- 📘 **Temario completo:** [Master Playbook](docs/temario/00-master-playbook.md)
- 📊 **Resultados medidos de las prácticas:** [Informe de resultados](INFORME_RESULTADOS.md)
- 🐳 **Entorno reproducible:** [`Dockerfile`](Dockerfile) (CUDA, nvcc, nsys, ncu, PyTorch; reutilizable en cloud)
- 🤖 **Cómo ejecutar en Windows, Linux y WSL2, y herramientas disponibles por sistema:** [`AGENTS.md`](AGENTS.md) (hardware local GTX 1050 frente a GPUs cloud L4/L40S/H100)

> Todas las cifras de este repositorio proceden de ejecuciones reales (`artifacts/`). Ninguna práctica genera
> resultados simulados.

## Hardware de referencia (local)

| Parámetro | NVIDIA GeForce GTX 1050 |
| :--- | :--- |
| Microarquitectura / Compute Capability | Pascal GP107 / `sm_61` |
| SMs / CUDA cores | 5 SMs / 640 |
| VRAM | 2 GB GDDR5, bus 128 bit, ~112 GB/s teóricos |
| Caché L2 / Shared memory por SM | 1 MB / 48 KB |
| FP32 teórico | ~1,86 TFLOP/s |

Límites de esta GPU (por qué hay unidades en cloud): sin Triton ni `ncu` moderno (necesitan `sm_70+`), sin FP8, sin multi-GPU.

## Temario e índice de prácticas

Estados: ✅ hecha y medida · 🟡 parcial · ⬜ pendiente. Modos: `[MOB]` lectura, `[LOC]` GPU local, `[PREP]`/`[CLD]` preparación y ráfaga cloud.

### Bloque 1: Arquitectura GPU, CUDA C++ y Profiling

| ID | Unidad | Modo | Estado | Práctica |
| :--- | :--- | :---: | :---: | :--- |
| B1.1 | Roofline, strides y `.contiguous()` en PyTorch | `[MOB]` `[LOC]` | ✅ | [`b1_1_strides/`](b1_1_strides/README.md) |
| B1.2 | CUDA C++: coalescencia, tiling y warp shuffles | `[MOB]` `[LOC]` | ✅ | [`b1_2_cuda_gemm/`](b1_2_cuda_gemm/README.md) |
| B1.3 | Profiling con `nsys` y `perf` | `[MOB]` `[LOC]` | 🟡 `nsys` real hecho; `perf` pendiente (Linux nativo/cloud) | [`b1_3_profiling/`](b1_3_profiling/README.md) |
| B1.4 | Triton y Nsight Compute (`ncu`) | `[PREP]` `[CLD]` | ⬜ Cloud-1 | — |

### Bloque 2: Motores de inferencia, KV-Cache y cargas agénticas

| ID | Unidad | Modo | Estado | Práctica |
| :--- | :--- | :---: | :---: | :--- |
| B2.1 | Aritmética del KV-Cache | `[MOB]` `[LOC]` | ✅ | [`b2_1_kv_cache/`](b2_1_kv_cache/README.md) |
| B2.2 | [vLLM vs SGLang y FlashInfer](b2_2_vllm_vs_sglang/README.md) | `[PREP]` ✅ `[CLD]` | 🟡 PREP hecho, Cloud-2 pendiente | — |
| B2.3 | TensorRT-LLM y NVIDIA Dynamo | `[MOB]` `[LOC]` | ⬜ | — |
| B2.4 | Cargas agénticas (prefix cache + JSON) | `[PREP]` `[CLD]` | ⬜ Cloud-2 | — |

### Bloque 3: Cuantización y aceleración de decode

| ID | Unidad | Modo | Estado | Práctica |
| :--- | :--- | :---: | :---: | :--- |
| B3.1 | Teoría de cuantización (FP8/INT4/NVFP4) | `[MOB]` | ⬜ | — |
| B3.2 | FP8 vs AWQ + KV-Cache FP8 | `[PREP]` `[CLD]` | ⬜ Cloud-3 | — |
| B3.3 | Speculative decoding | `[PREP]` `[CLD]` | ⬜ Cloud-3 | — |
| B3.4 | Evals de calidad y frontera de Pareto | `[PREP]` `[CLD]` | ⬜ Cloud-3 | — |

### Bloque 4: Inferencia distribuida, MoE y redes

| ID | Unidad | Modo | Estado | Práctica |
| :--- | :--- | :---: | :---: | :--- |
| B4.1 | Paralelismo (TP/PP/DP/CP) y NCCL | `[MOB]` | ⬜ | — |
| B4.2 | MoE y Wide Expert Parallelism | `[MOB]` | ⬜ | — |
| B4.3 | PDD, LMCache y routing cache-aware | `[PREP]` `[CLD]` | ⬜ Cloud-4 | — |
| B4.4 | Interconexiones y `nccl-tests` | `[PREP]` `[CLD]` | ⬜ Cloud-4 | — |

### Bloque 5: Benchmarking científico y cluster acceptance

| ID | Unidad | Modo | Estado | Práctica |
| :--- | :--- | :---: | :---: | :--- |
| B5.1 | Metodología de benchmarking reproducible | `[MOB]` `[LOC]` | ⬜ | — |
| B5.2 | Cluster acceptance (`dcgmi` + Xid) | `[PREP]` `[CLD]` | ⬜ Cloud-5 | — |
| B5.3 | Entrenamiento con Megatron-LM y MFU | `[PREP]` `[CLD]` | ⬜ Cloud-5 | — |
| B5.4 | K8s, DCGM y cold starts | `[MOB]` `[LOC]` | ⬜ | — |

### Documentos del temario

Los documentos de cada unidad se añaden en [`docs/temario/`](docs/temario/) y se enlazan aquí a medida que se publican.

| Documento | Unidad |
| :--- | :--- |
| [Master Playbook y sistema operativo de estudio](docs/temario/00-master-playbook.md) | Todo el temario |

## Resultados medidos (GTX 1050)

| Práctica | Resultado principal |
| :--- | :--- |
| B1.1 | `transpose` no copia datos; `contiguous()` 33,4 GB/s (30 % del pico) frente a `clone()` a 91,8 GB/s (82 %) |
| B1.2 | `gemm_smem_tiled` 11,7× más rápido que `naive` (281 GFLOP/s); cuBLAS 1627 GFLOP/s; softmax online 1,31× |
| B1.3 | `pin_memory` 2,1× en H2D; paso total 1,01× (domina el cómputo sin fusionar: fusión pendiente en B1.4) |
| B2.1 | Llama-3-8B FP16: 26 peticiones de 8k tokens en L40S, 55 en H100 |

Detalle, tablas y limitaciones: [`INFORME_RESULTADOS.md`](INFORME_RESULTADOS.md).

## Cómo ejecutar las prácticas

Requisitos: GPU NVIDIA, Python 3.10+ y PyTorch con CUDA. Cada práctica tiene un `run.sh` con `--smoke-test` (< 30 s).

```bash
# Python (B1.1, B1.3, B2.1): venv del repositorio
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch numpy huggingface_hub --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
(cd b1_1_strides && ./run.sh --smoke-test)
(cd b2_1_kv_cache && ./run.sh)

# CUDA C++ y profiling (B1.2, B1.3) en Linux: imagen Docker con nvcc, nsys y ncu
docker build -t ai-infra-study:cuda12.6 --build-arg INSTALL_TORCH=0 .
docker run --rm --gpus all -v "$PWD":/work ai-infra-study:cuda12.6 bash -c "cd b1_2_cuda_gemm && ./run.sh --smoke"
```

Notas: `nsys` en WSL2 no identifica la GPU (usa Nsight Systems nativo de Windows o Linux); las trazas `.nsys-rep` no se versionan
porque embeben las variables de entorno del proceso. Cloud: `--build-arg CUDA_ARCH=sm_90` (H100) o `sm_89` (L4/L40S); apaga la VM
con `sudo shutdown -h +90`.

## Estructura

```
├── docs/temario/          # Documentos del temario (Master Playbook y siguientes)
├── b1_1_strides/          # B1.1 strides y .contiguous()
├── b1_2_cuda_gemm/        # B1.2 CUDA C++ (GEMM, warp online softmax)
├── b1_3_profiling/        # B1.3 antipatrones y nsys
├── b2_1_kv_cache/         # B2.1 calculadora de KV-cache
├── artifacts/             # Salidas JSON/CSV/TXT generadas por los runners
├── INFORME_RESULTADOS.md  # Informe de resultados
└── Dockerfile             # Entorno CUDA reproducible (local y cloud)
```
