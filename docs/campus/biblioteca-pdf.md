# Biblioteca de Manuales PDF y Fuentes Typst

Todo el temario extendido de los **tres cursos**, junto con **infografías vectoriales**, **ejercicios cuantitativos y de arquitectura con solucionario** y **bibliografía primaria**, está editado en **Typst `0.15.1`** y compilado a **PDF** dentro del propio repositorio.

## Descarga Directa de los PDFs Compilados

| Edición | Contenido incluido | PDF Compilado | Fuente Principal (`.typ`) |
| :--- | :--- | :--- | :--- |
| **Volumen Máster Unificado** | Tronco Común + Curso 1 (`B1`–`B5`) + Curso 2 (`B1`–`B4`) + Curso 3 (`E0`–`E6`) + Anexo Consolidado | [Descargar PDF Unificado](../pdf/Master_AI_Infra_y_Agentic_MLOps.pdf) | [`master_ai_infra_y_mlops.typ`](../pdf/master_ai_infra_y_mlops.typ) |
| **Curso 1 · AI Infrastructure & GPU Systems** | Tronco Común + Hardware CUDA, `nsys`/`ncu`, KV-Cache, vLLM/SGLang, FP8, MoE, NCCL, DCGM | [Descargar PDF Curso 1](../pdf/Curso_1_AI_Infrastructure_GPU.pdf) | [`curso1_ai_infra.typ`](../pdf/curso1_ai_infra.typ) |
| **Curso 2 · Enterprise Agentic MLOps** | Tronco Común + Databricks Mosaic AI, MLflow 3, Unity Catalog, Evals, RAG, Seldon/Istio/Argo, Crossplane, SRE | [Descargar PDF Curso 2](../pdf/Curso_2_Agentic_Platform_MLOps.pdf) | [`curso2_agentic_mlops.typ`](../pdf/curso2_agentic_mlops.typ) |
| **Curso 3 · Edge-AI Sub-2-Bit & Portable Kernels** | Tronco Común + `mWh/tok`, `Qwen3.5-2B` vs `Bonsai 2`, Hadamard + GPTQ CPU, `Gemma 4 E2B`, CUDA `__dp4a` + Vulkan GLSL, `< 8 W` + UMA | [Descargar PDF Curso 3](../pdf/Curso_3_Edge_AI_Sub2Bit_Kernels.pdf) | [`curso3_edge_ai.typ`](../pdf/curso3_edge_ai.typ) |

---

## Arquitectura Modular de Fuentes Typst

Si quieres editar bloques, añadir notas personales o recompilar los manuales en local:

- Sistema de diseño e infografías vectoriales (sin dependencias externas): [`theme.typ`](../pdf/theme.typ)
- Capítulo 0 · Tronco Común (Lo Común vs. Lo Específico): [`00_tronco_comun.typ`](../pdf/sections/00_tronco_comun.typ)
- Curso 1 (`B1`–`B5`):
  - [`c1_b1.typ`](../pdf/sections/c1_b1.typ) · [`c1_b2.typ`](../pdf/sections/c1_b2.typ) · [`c1_b3.typ`](../pdf/sections/c1_b3.typ) · [`c1_b4.typ`](../pdf/sections/c1_b4.typ) · [`c1_b5.typ`](../pdf/sections/c1_b5.typ)
- Curso 2 (`B1`–`B4`):
  - [`c2_b1.typ`](../pdf/sections/c2_b1.typ) · [`c2_b2.typ`](../pdf/sections/c2_b2.typ) · [`c2_b3.typ`](../pdf/sections/c2_b3.typ) · [`c2_b4.typ`](../pdf/sections/c2_b4.typ)
- Curso 3 (`E0`–`E6`):
  - [`c3_e0_e1.typ`](../pdf/sections/c3_e0_e1.typ) · [`c3_e2.typ`](../pdf/sections/c3_e2.typ) · [`c3_e3.typ`](../pdf/sections/c3_e3.typ) · [`c3_e4.typ`](../pdf/sections/c3_e4.typ) · [`c3_e5_e6.typ`](../pdf/sections/c3_e5_e6.typ)

### Recompilar en Local

```bash
cd docs/pdf
typst compile master_ai_infra_y_mlops.typ Master_AI_Infra_y_Agentic_MLOps.pdf
typst compile curso1_ai_infra.typ Curso_1_AI_Infrastructure_GPU.pdf
typst compile curso2_agentic_mlops.typ Curso_2_Agentic_Platform_MLOps.pdf
typst compile curso3_edge_ai.typ Curso_3_Edge_AI_Sub2Bit_Kernels.pdf
```

!!! tip "Requisitos de fuentes tipográficas"
    Los `.typ` solo utilizan `Libertinus Serif` y `DejaVu Sans Mono` para que compilen en Linux, macOS, Windows y Termux sin descargas externas ni paquetes remotos de Typst.
