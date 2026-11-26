# AI Infra & Agentic MLOps · Campus Virtual

> **Sitio Web / Campus Virtual (GitHub Pages):** [`https://theunixroot.github.io/AI-infra-study/`](https://theunixroot.github.io/AI-infra-study/) · [Entrar al Tour del Campus](docs/campus/index.md) · [Mi Progreso y Handover (20 min)](docs/campus/progreso.md)

Plataforma de aprendizaje técnico de nivel Máster con **dos itinerarios sobre un Tronco Común** de ingeniería de IA, prácticas reproducibles en GPU/local y manuales editados en **Typst → PDF**:

- 🧭 **Tronco Común (6 Pilares + Micro-sprints 20m):** [Portal interactivo](docs/campus/tronco-comun.md) · [Matriz comparativa](docs/temario/02-tronco-comun-y-itinerarios.md)
- ⚙️ **Curso 1 · AI Infrastructure & GPU Systems:** [Portal Curso 1](docs/campus/curso1/index.md) · [Playbook maestro](docs/temario/00-master-playbook.md) · [PDF Curso 1 (21 pág.)](docs/pdf/Curso_1_AI_Infrastructure_GPU.pdf)
- 🤖 **Curso 2 · Enterprise MLOps & Agentic Platform:** [Portal Curso 2](docs/campus/curso2/index.md) · [Playbook maestro](docs/temario/01-agentic-mlops-playbook.md) · [PDF Curso 2 (16 pág.)](docs/pdf/Curso_2_Agentic_Platform_MLOps.pdf)
- 📚 **Biblioteca PDF Unificada (35 pág.):** [Volumen Máster Unificado](docs/pdf/Master_AI_Infra_y_Agentic_MLOps.pdf) · [Centro PDF y fuentes Typst](docs/campus/biblioteca-pdf.md)
- 📊 **Resultados medidos de las prácticas:** [Informe de resultados](INFORME_RESULTADOS.md) · 🐳 [`Dockerfile`](Dockerfile) · 🤖 [`AGENTS.md`](AGENTS.md)

> Todas las cifras de este repositorio proceden de ejecuciones reales (`artifacts/`). Ninguna práctica genera
> resultados simulados.

## Hardware de referencia (local)

| Parámetro | NVIDIA GeForce GTX 1050 |
| :--- | :--- |
| Microarquitectura / Compute Capability | Pascal GP107 / `sm_61` |
| SMs / CUDA cores | 5 SMs / 640 |
| VRAM | 2 GB GDDR5, bus 128 bit, ~112 GB/s teóricos |
| Caché L2 / Shared memory | 1 MB / 48 KB máx. por bloque (96 KB por SM en GP107) |
| FP32 teórico | ~1,86 TFLOP/s |

Límites de esta GPU (por qué hay unidades en cloud): sin Triton ni `ncu` moderno (necesitan `sm_70+`), sin FP8, sin multi-GPU.

## Temario del Campus e Índice de Prácticas

Estados: ✅ hecha y medida · 🟡 parcial · ⬜ pendiente. Modos: `[MOB]` lectura, `[LOC]` local, `[PREP]`/`[CLD]` preparación y ráfaga cloud, `[LAB]` sandbox 0 €.

### Curso 1 · AI Infrastructure & GPU Systems ([Portal del Curso 1](docs/campus/curso1/index.md))

#### [Bloque 1: Arquitectura GPU, CUDA C++ y Profiling](docs/campus/curso1/b1-arquitectura-cuda.md)

| ID | Unidad | Modo | Estado | Práctica |
| :--- | :--- | :---: | :---: | :--- |
| B1.1 | Roofline, strides y `.contiguous()` en PyTorch | `[MOB]` `[LOC]` | ✅ | [`b1_1_strides/`](b1_1_strides/README.md) |
| B1.2 | CUDA C++: coalescencia, tiling y warp shuffles | `[MOB]` `[LOC]` | ✅ | [`b1_2_cuda_gemm/`](b1_2_cuda_gemm/README.md) |
| B1.3 | Profiling con `nsys` y `perf` | `[MOB]` `[LOC]` | 🟡 `nsys` real hecho; `perf` pendiente (Linux nativo/cloud) | [`b1_3_profiling/`](b1_3_profiling/README.md) |
| B1.4 | Triton y Nsight Compute (`ncu`) | `[PREP]` `[CLD]` | ⬜ Cloud-1 | [Unidad B1.4](docs/campus/curso1/b1-arquitectura-cuda.md) |

#### [Bloque 2: Motores de inferencia, KV-Cache y cargas agénticas](docs/campus/curso1/b2-motores-kv-cache.md)

| ID | Unidad | Modo | Estado | Práctica |
| :--- | :--- | :---: | :---: | :--- |
| B2.1 | Aritmética del KV-Cache | `[MOB]` `[LOC]` | ✅ | [`b2_1_kv_cache/`](b2_1_kv_cache/README.md) |
| B2.2 | vLLM vs SGLang y FlashInfer | `[PREP]` ✅ `[CLD]` | 🟡 PREP hecho, Cloud-2 pendiente | [`b2_2_vllm_vs_sglang/`](b2_2_vllm_vs_sglang/README.md) |
| B2.3 | TensorRT-LLM y NVIDIA Dynamo | `[MOB]` `[LOC]` | ⬜ | [Unidad B2.3](docs/campus/curso1/b2-motores-kv-cache.md) |
| B2.4 | Cargas agénticas (prefix cache + JSON) | `[PREP]` `[CLD]` | ⬜ Cloud-2 | [Unidad B2.4](docs/campus/curso1/b2-motores-kv-cache.md) |

#### [Bloque 3: Cuantización y aceleración de decode](docs/campus/curso1/b3-cuantizacion-pareto.md)

| ID | Unidad | Modo | Estado | Práctica |
| :--- | :--- | :---: | :---: | :--- |
| B3.1 | Teoría de cuantización (FP8/INT4/NVFP4) | `[MOB]` | ⬜ | [Bloque 3](docs/campus/curso1/b3-cuantizacion-pareto.md) |
| B3.2 | FP8 vs AWQ + KV-Cache FP8 | `[PREP]` `[CLD]` | ⬜ Cloud-3 | [Bloque 3](docs/campus/curso1/b3-cuantizacion-pareto.md) |
| B3.3 | Speculative decoding | `[PREP]` `[CLD]` | ⬜ Cloud-3 | [Bloque 3](docs/campus/curso1/b3-cuantizacion-pareto.md) |
| B3.4 | Evals de calidad y frontera de Pareto | `[PREP]` `[CLD]` | ⬜ Cloud-3 | [Bloque 3](docs/campus/curso1/b3-cuantizacion-pareto.md) |

#### [Bloque 4: Inferencia distribuida, MoE y redes](docs/campus/curso1/b4-paralelismo-moe-nccl.md)

| ID | Unidad | Modo | Estado | Práctica |
| :--- | :--- | :---: | :---: | :--- |
| B4.1 | Paralelismo (TP/PP/DP/CP) y NCCL | `[MOB]` | ⬜ | [Bloque 4](docs/campus/curso1/b4-paralelismo-moe-nccl.md) |
| B4.2 | MoE y Wide Expert Parallelism | `[MOB]` | ⬜ | [Bloque 4](docs/campus/curso1/b4-paralelismo-moe-nccl.md) |
| B4.3 | PDD, LMCache y routing cache-aware | `[PREP]` `[CLD]` | ⬜ Cloud-4 | [Bloque 4](docs/campus/curso1/b4-paralelismo-moe-nccl.md) |
| B4.4 | Interconexiones y `nccl-tests` | `[PREP]` `[CLD]` | ⬜ Cloud-4 | [Bloque 4](docs/campus/curso1/b4-paralelismo-moe-nccl.md) |

#### [Bloque 5: Benchmarking científico y cluster acceptance](docs/campus/curso1/b5-cluster-benchmarking.md)

| ID | Unidad | Modo | Estado | Práctica |
| :--- | :--- | :---: | :---: | :--- |
| B5.1 | Metodología de benchmarking reproducible | `[MOB]` `[LOC]` | ⬜ | [Bloque 5](docs/campus/curso1/b5-cluster-benchmarking.md) |
| B5.2 | Cluster acceptance (`dcgmi` + Xid) | `[PREP]` `[CLD]` | ⬜ Cloud-5 | [Bloque 5](docs/campus/curso1/b5-cluster-benchmarking.md) |
| B5.3 | Entrenamiento con Megatron-LM y MFU | `[PREP]` `[CLD]` | ⬜ Cloud-5 | [Bloque 5](docs/campus/curso1/b5-cluster-benchmarking.md) |
| B5.4 | K8s, DCGM y cold starts | `[MOB]` `[LOC]` | ⬜ | [Bloque 5](docs/campus/curso1/b5-cluster-benchmarking.md) |

### Curso 2 · Enterprise MLOps & Agentic Platform ([Portal del Curso 2](docs/campus/curso2/index.md))

| Chat | Bloque y Unidades | Modo | Portal Campus |
| :--- | :--- | :---: | :--- |
| `CHAT-B1` | **B1 (`B1.1`–`B1.3`) · Databricks Mosaic AI, MLflow 3 y Unity Catalog** | `[MOB]` `[LOC]` `[LAB]` | [Bloque 1 →](docs/campus/curso2/b1-databricks-mlflow-uc.md) |
| `CHAT-B2` | **B2 (`B2.1`–`B2.3`) · Agentic Evals (`mlflow.evaluate`), RAG Health y OTel/FinOps** | `[MOB]` `[LOC]` | [Bloque 2 →](docs/campus/curso2/b2-evals-rag-observabilidad.md) |
| `CHAT-B3` | **B3 (`B3.1`–`B3.3`) · Serving K8s (Seldon/Knative), Istio + ArgoCD y Crossplane** | `[MOB]` `[LOC]` | [Bloque 3 →](docs/campus/curso2/b3-serving-gitops-crossplane.md) |
| `CHAT-B4` | **B4 (`B4.1`–`B4.3`) · SRE SLOs, GameDay Blameless y Multi-Región UE** | `[MOB]` `[LOC]` | [Bloque 4 →](docs/campus/curso2/b4-sre-slos-multiregion.md) |

### Documentos del temario

Los documentos completos del temario se mantienen en Markdown (ver [Playbook Curso 1](docs/temario/00-master-playbook.md)):

| Documento | Unidad |
| :--- | :--- |
| [Curso 1 · Master Playbook: AI Infrastructure & GPU Systems](docs/temario/00-master-playbook.md) | Curso 1 (`B1`–`B5`) |
| [Curso 2 · Master Playbook: Agentic Platform MLOps](docs/temario/01-agentic-mlops-playbook.md) | Curso 2 (`B1`–`B4`) |
| [Tronco común y mapa comparativo de los dos itinerarios](docs/temario/02-tronco-comun-y-itinerarios.md) | Base compartida (6 pilares) |

### Manuales de nivel Máster (Typst → PDF)

Ediciones autocontenidas con infografías vectoriales, temario extendido, ejercicios resueltos y fuentes primarias en la [Biblioteca PDF](docs/campus/biblioteca-pdf.md):

| Edición | Fuente Typst | PDF Compilado |
| :--- | :--- | :--- |
| **Volumen Máster Unificado** (Tronco Común + Curso 1 + Curso 2) | [`master_ai_infra_y_mlops.typ`](docs/pdf/master_ai_infra_y_mlops.typ) | [`Master_AI_Infra_y_Agentic_MLOps.pdf`](docs/pdf/Master_AI_Infra_y_Agentic_MLOps.pdf) |
| **Curso 1:** AI Infrastructure, GPU Systems & Inference Engines | [`curso1_ai_infra.typ`](docs/pdf/curso1_ai_infra.typ) | [`Curso_1_AI_Infrastructure_GPU.pdf`](docs/pdf/Curso_1_AI_Infrastructure_GPU.pdf) |
| **Curso 2:** Enterprise MLOps & Agentic Platform Engineering | [`curso2_agentic_mlops.typ`](docs/pdf/curso2_agentic_mlops.typ) | [`Curso_2_Agentic_Platform_MLOps.pdf`](docs/pdf/Curso_2_Agentic_Platform_MLOps.pdf) |

## Resultados medidos (GTX 1050)

| Práctica | Resultado principal |
| :--- | :--- |
| B1.1 | `transpose` no copia datos; `contiguous()` 33,4 GB/s (30 % del pico) frente a `clone()` a 91,8 GB/s (82 %) |
| B1.2 | `gemm_smem_tiled` (32×32) 11,7× más rápido que `naive` (281 GFLOP/s; incluye variante `16×16` pendiente de captura); cuBLAS 1627 GFLOP/s; softmax online 1,31× |
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
├── docs/campus/           # Campus Virtual web (Tronco Común, Curso 1, Curso 2, Biblioteca PDF, Progreso)
├── docs/temario/          # Temario Markdown (Master Playbooks y Tronco Común)
├── docs/pdf/              # Manuales Máster en Typst (.typ) y PDF compilados (.pdf)
├── b1_1_strides/          # B1.1 strides, .contiguous() y baseline clone()
├── b1_2_cuda_gemm/        # B1.2 CUDA C++ (GEMM 32x32/16x16, warp online softmax)
├── b1_3_profiling/        # B1.3 antipatrones y nsys
├── b2_1_kv_cache/         # B2.1 calculadora de KV-cache
├── b2_2_vllm_vs_sglang/   # B2.2 comparativa vLLM vs SGLang + FlashInfer ([PREP])
├── artifacts/             # Salidas JSON/CSV/TXT reales generadas por los runners
├── tests/                 # Suite unitaria offline (sin GPU)
├── tools/                 # Generador del sitio estático (MkDocs + KaTeX)
├── AGENTS.md              # Guía única para agentes y entornos (Windows/Linux/WSL2/Docker/Termux)
├── INFORME_RESULTADOS.md  # Informe de resultados medidos
├── mkdocs.yml             # Configuración de GitHub Pages (MkDocs Material)
└── Dockerfile             # Entorno CUDA reproducible (local y cloud)
```
