# Tronco Común y Tres Itinerarios Profesionales (AI Infra, Agentic MLOps y Edge-AI)

Este documento extrae la intersección técnica entre los **tres playbooks** del repositorio (**lo común**) y delimita claramente qué aporta cada especialización (**lo específico**), organizando el estudio en un **Tronco Común de 6 pilares** seguido de **tres itinerarios profesionales**, editados como manuales didácticos en Typst/PDF dentro de la [Biblioteca PDF](../campus/biblioteca-pdf.md).

## 1. Mapa de Tres Itinerarios desde un Baseline DevOps/SRE + C++/GPU

| Dimensión | Curso 1: AI Infrastructure & GPU Systems | Curso 2: Enterprise MLOps & Agentic Platform | Curso 3: Edge-AI Sub-2-Bit & Portable Kernels |
| :--- | :--- | :--- | :--- |
| **Fuente maestra** | [`00-master-playbook.md`](00-master-playbook.md) | [`01-agentic-mlops-playbook.md`](01-agentic-mlops-playbook.md) | [`03-edge-ai-ternary-playbook.md`](03-edge-ai-ternary-playbook.md) |
| **Roles objetivo** | GPU Platform Benchmarking · Senior MLE Inference (Token Factory) | Staff/Senior MLOps Engineer · Agentic Platform Engineer | Senior Edge Inference MLE · Portable GPU/CPU Kernel Engineer (puente HPAC/UMA `LogFit`) |
| **Hardware / Entorno** | GTX 1050 `sm_61` (`B1`–`B2.1`) + Cloud GPU (`L4`, `L40S`, `2x–8x H100`) | CPU local (Docker, Kind, MLflow, OpenSearch) + Databricks Sandbox (`[LAB]`) | **0 € obligatorio:** GTX 1050 `sm_61` (2 GB) + CPU `x86_64` (16 GB+ RAM) + Android Termux (`ARM64 NEON` + Vulkan) |
| **Proyectos Portfolio** | `agentic-inference-opt-lab` (A) y `gpu-cluster-acceptance-profiler` (B) | `agentic-platform-golden-path` | `edge-ternary-lab` + actualización `TheUnixRoot/Logfit` + dossier UMA (`04_dossier_uma.md`) |
| **Métrica Reina** | `TTFT`, `TPOT`, `tok/s/GPU`, `busbw`, `MFU`, frontera Pareto `$/1M tokens` | `Task Success`, `Tool Reliability`, `Recall@K`, Availability/Quality SLOs, `Audit Readiness` | `mWh/token` (potencia neta W), `% BW pico` (`GB/s`), `PPL`, `KLD vs F16`, `GSM8K`/`IFEval`, coste `€/año` |
| **Foco diferencial** | CUDA/Triton, `nsys`/`ncu`, vLLM/SGLang/TRT-LLM, FP8/AWQ, MoE/NCCL, DCGM/K8s | Databricks Mosaic AI, MLflow 3, Unity Catalog, RAG Health, Seldon/Istio/Argo/Crossplane, SRE | Compresión sub-2-bit (`TQ1_0`/`TQ2_0`), Hadamard (`FHT`) + `GPTQ` CPU, `Gemma 4 E2B`, `__dp4a` CUDA + Vulkan GLSL, *LogFit-Edge* |

---

## 2. Análisis: Lo Común frente a Lo Específico

### 2.1 Lo Común (El Tronco Unificado — ~35–40 % compartido)

Estudiar estos tres itinerarios de forma aislada duplicaría cerca de un 40 % del esfuerzo conceptual y de ingeniería experimental. Todos comparten **6 pilares invariantes**:

1. **Economía Física de Inferencia (Prefill vs. Decode) y KV-Cache:**
   - El *Prefill* procesa la secuencia completa en paralelo (*compute-bound*), mientras el *Decode* con batch bajo es estrictamente *memory-bandwidth-bound* ($I \approx 1\ \text{FLOP/B} \ll I^*$).
   - La huella del KV-Cache crece linealmente con capas, cabezas KV, dimensión de cabeza, longitud de contexto y precisión ($2 \cdot L \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot S \cdot B \cdot b_{\text{kv}}$), gobernando qué cabe en memoria y cuánta concurrencia se alcanza.
2. **Anatomía de Cargas Agénticas, Razonamiento y Salidas Estructuradas:**
   - La inferencia moderna ya no es solo chat libre: exige *system prompts* compartidos, llamadas a herramientas (*Tool Calling*), generación JSON Schema válida y trazas de razonamiento (`<|think|>`).
   - Las optimizaciones de infraestructura (cuantización, caché o gramáticas) pueden romper silenciosamente esquemas o secuencias de cierre (`<end_of_turn>`) sin que la perplejidad lo delate.
3. **Rigor Científico, Control de Regresiones y Frontera de Pareto:**
   - Ninguna métrica es válida sin *warmup*, repeticiones controladas, medidas de varianza/cola y prohibición absoluta de cifras inventadas.
   - Ninguna compresión o aceleración se acepta por velocidad pura: se evalúa calidad real (`lm-evaluation-harness`, `GSM8K`, `IFEval`, JSON validity, `KLD`) y se decide sobre la **frontera de Pareto** multi-objetivo.
4. **Observabilidad Causal End-to-End y Economía (FinOps / Energía):**
   - Medir causalmente dónde se consume cada milisegundo y cada vatio/dólar: desde `nsys`/NVTX y contadores hardware hasta OpenTelemetry GenAI y muestreo `sysfs`/`NVML`/`RAPL`.
5. **Despliegue Reproducible y Portabilidad Operativa:**
   - Automatizar de punta a punta el ciclo de vida del servicio de inferencia, exponiendo APIs estándar compatibles con OpenAI bajo restricciones estrictas de latencia, aislamiento, coste o consumo eléctrico.
6. **Sistema Operativo de Micro-Sprints (20 min) y Delegación Controlada:**
   - Unidades atómicas de 20 minutos, regla de no bloqueo delegando *boilerplate* al agente de código con **tests unitarios obligatorios** (`run.sh --smoke-test`, `verify.sh`, `--dry-run`) y bloques `HANDOVER:` YAML para mantener contexto limpio.

### 2.2 Lo Específico de Cada Itinerario (y Puente Directo Curso 1 $\leftrightarrow$ Curso 3)

| Eje técnico | Específico Curso 1 (Datacenter GPU) | Específico Curso 2 (Enterprise Platform) | Específico Curso 3 (Edge-AI & Portable Kernels) |
| :--- | :--- | :--- | :--- |
| **Aritmética y Cuantización** | Formatos Tensor Core (`FP8 E4M3`, `AWQ INT4 Marlin`, `MXFP4`/`NVFP4`), calibración con `llm-compressor` para `L40S`/`H100` (`sm_89`/`sm_90`). | Evaluación de modelos desplegados por calidad agéntica y coste `$/1M` sin modificar pesos ni escribir *kernels*. | **Sub-2-bit (`1.725–2.125 bpp`):** empaquetado base-3 `TQ1_0` (5 trits/byte sin divisiones via `(xi*3)>>8`), `TQ2_0`, **Fast Hadamard Transform** ($O(N \log N)$), `GPTQ` Hessiano en CPU capa a capa (`<10 GB` RAM) y protección de `PLE`/`lm_head`/`vision_tower` en `Gemma 4 E2B`. |
| **Programación GPU y Portabilidad** | CUDA C++ educativo en `sm_61` (`B1.2`) como puente hacia **OpenAI Triton** y **Nsight Compute (`ncu`)** en `L4`/`H100`. | No programa GPU; orquesta contenedores y malla de servicio sobre Kubernetes y Databricks. | **Kernels de inferencia desde cero:** `GEMV` ternario `g128` con **`__dp4a` (`int8x4`)** en Pascal `sm_61` + `FHT 1024` en *shared memory* $\rightarrow$ portado a **Vulkan Compute (GLSL/SPIR-V)** con `GL_KHR_shader_subgroup_*` para GTX 1050 y Adreno/Mali *zero-copy*, más ejecución heterogénea CPU+GPU (*LogFit-Edge*). |
| **Motor y Serving** | `vLLM V1`, `SGLang` (`RadixAttention`), `FlashInfer`, `TensorRT-LLM`, `NVIDIA Dynamo` (`PDD` + `NIXL`), `Speculative Decoding` (`EAGLE-3`, `MTP`). | `Databricks Mosaic AI`, `MLflow 3` (`LoggedModel` + aliases), `Unity Catalog` (`lineage`/`RBAC`), `Seldon Core v2`, `Knative KPA`, `Istio`, `ArgoCD`, `Crossplane`. | `llama.cpp` *upstream* (`b11556+`, Hadamard + Vulkan) vs. fork `PrismML-llama.cpp` (`PTQ1_0`/`PQ2_0`), KV-cache de 4 bits (`q4_0`), servidor OpenAI 24/7 en Termux (`< 8 W`) y contribución potencial a `ggml-vulkan`. |
| **Economía y Métrica Física** | Coste cloud `$/1M tokens` en `L40S`/`H100`, `MFU` en `Megatron-LM`, `busbw` en `nccl-tests`, *cluster acceptance* con `dcgmi`/`Xid`. | Cuotas `TPM`/`RPM`, *circuit breakers* por *tenant*, *error budgets* SRE `99.9 %`, soberanía RGPD multi-región UE. | Potencia física neta (W sobre línea base *idle* via `sysfs` Android y `NVML`/`RAPL` PC), **`mWh/token`** ($P_{\text{net}} / (3.6 \cdot \text{TPS})$), degradación térmica móvil y factura eléctrica doméstica `€/año`. |
| **Estrategia de Hardware y Coste** | Ráfagas cloud preparadas (`[PREP]` $\rightarrow$ `[CLD]` con `sudo shutdown -h +90`). | Sandbox gratuito Databricks (`[LAB]`) + Kubernetes local (`Kind`). | **100 % 0 € en hardware propio** (`[LOC-PC]` + `[LOC-AND]`), salvo experimento bonus opcional `E6` (QAT STE + KL en Colab/Kaggle T4 gratuita). |

### 2.3 Sinergia Temporal: Cómo Enganchar Curso 3 con Curso 1 (Sin Quemarte)

El Curso 3 actúa como extensión `B3-Edge` de los bloques del Curso 1:

1. **Durante Bloque 2 de Curso 1 (`B2.1`–`B2.4`):** Ejecuta **E0 y E1** en ratos libres (`0 €`), montando `llama.cpp`/PrismML y el arnés de energía `mWh/token` sobre GTX 1050 y Android.
2. **Al estudiar Bloque 3 de Curso 1 (`B3.1` Cuantización / `B3.4` Pareto):** Ejecuta **E2 y E3** en paralelo; contrastarás cuantización datacenter (`FP8`/`AWQ`) con cuantización extrema sub-2-bit (`Hadamard + GPTQ`) en CPU local antes de gastar presupuesto en `Cloud-3`.
3. **Tras cerrar Bloque 3 de Curso 1:** Ejecuta **E4** (kernels `__dp4a` + Vulkan Compute GLSL) y **E5** (servidor `< 8 W` + *LogFit-Edge* + dossier para Rafael Asenjo en HPAC/UMA).

---

## 3. Edición Didáctica en Typst y PDF (`docs/pdf/`)

Los archivos fuente modulares en Typst y sus PDFs compilados cubren cada bloque/módulo con **infografías vectoriales**, **desarrollo teórico nivel máster**, **prácticas guiadas**, **ejercicios con solucionario** y **fuentes primarias**:

- **Volumen Maestro Unificado (Tronco Común + Curso 1 + Curso 2 + Curso 3):** [`master_ai_infra_y_mlops.typ`](../pdf/master_ai_infra_y_mlops.typ) → [`Master_AI_Infra_y_Agentic_MLOps.pdf`](../pdf/Master_AI_Infra_y_Agentic_MLOps.pdf)
- **Curso 1 (AI Infrastructure & GPU Systems):** [`curso1_ai_infra.typ`](../pdf/curso1_ai_infra.typ) → [`Curso_1_AI_Infrastructure_GPU.pdf`](../pdf/Curso_1_AI_Infrastructure_GPU.pdf)
- **Curso 2 (Enterprise MLOps & Agentic Platform):** [`curso2_agentic_mlops.typ`](../pdf/curso2_agentic_mlops.typ) → [`Curso_2_Agentic_Platform_MLOps.pdf`](../pdf/Curso_2_Agentic_Platform_MLOps.pdf)
- **Curso 3 (Edge-AI Sub-2-Bit & Portable Kernels):** [`curso3_edge_ai.typ`](../pdf/curso3_edge_ai.typ) → [`Curso_3_Edge_AI_Sub2Bit_Kernels.pdf`](../pdf/Curso_3_Edge_AI_Sub2Bit_Kernels.pdf)
