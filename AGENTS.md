# AGENTS.md

Guía para agentes de código (Claude Code, Gemini CLI, Codex, Copilot…) y para personas que trabajen en este repositorio.
`CLAUDE.md` y `GEMINI.md` apuntan a este archivo: **esta es la única fuente de instrucciones**.

## 1. Qué es este repositorio

Learning path de **AI Infrastructure** (GPU, inferencia y benchmarking de clústeres) con prácticas reproducibles.
El temario está en [`docs/temario/00-master-playbook.md`](docs/temario/00-master-playbook.md) y el [`README.md`](README.md) es su índice,
con enlaces a cada práctica. Los resultados medidos están en [`INFORME_RESULTADOS.md`](INFORME_RESULTADOS.md).
Se publica como sitio de GitHub Pages (Jekyll, ver `_config.yml`).

## 2. Hardware: local frente a cloud

| Entorno | GPU | Arquitectura | VRAM | Se usa para |
| :--- | :--- | :--- | :---: | :--- |
| **Local (portátil)** | **NVIDIA GeForce GTX 1050** | Pascal, `sm_61` | **2 GB** | Prácticas `[LOC]`: B1.1, B1.2, B1.3, B2.1, B2.3 (simulador), B5.1 |
| **Cloud** | **L4** | Ada, `sm_89` | 24 GB | B1.4 (Triton + `ncu`), pruebas ligeras |
| **Cloud** | **L40S** | Ada, `sm_89` | 48 GB | B2.2, B2.4, B3.x (FP8, un solo nodo) |
| **Cloud** | **H100** (1x, 2x–8x) | Hopper, `sm_90` | 80 GB | B3.x, B4.3, B4.4, B5.2, B5.3 (NVLink, multi-GPU) |

Reglas que se derivan de esto:

- **Todo lo local debe caber en 2 GB de VRAM.** Tensores < 256 MB (por defecto 2048×2048 FP32 = 16 MiB). No lances nada que haga OOM.
- La GTX 1050 **no** soporta Triton, `ncu` moderno, FlashAttention/FlashInfer/TensorRT-LLM, FP8 ni multi-GPU (exigen `sm_70`–`sm_90`).
  Esas unidades son `[PREP]` (se prepara en local, 0 €) + `[CLD]` (ráfaga en cloud). **No alquiles una GPU sin haber completado su `[PREP]`.**
- En cloud, lo primero en la VM: `sudo shutdown -h +90` (auto-kill). Nunca dejes una GPU encendida.
- Para compilar para cloud, el runner de B1.2 y el `Dockerfile` aceptan `CUDA_ARCH` (`sm_89` L4/L40S, `sm_90` H100, `sm_61` por defecto).

## 3. Cómo ejecutar las prácticas

Todas las prácticas tienen `run.sh` (bash, idempotente) con `--smoke-test` (< 30 s) y escriben resultados reales en `artifacts/`.
Usa **una sola venv en la raíz del repo**: `.venv` (Windows) o `.venv-linux` (Docker/Linux). No crees otras.

### 3.1 Windows (nativo)

Requisitos: driver NVIDIA, Python 3.11, **Git Bash** (los `run.sh` son bash) y, para `nvcc`, CUDA Toolkit 12.6 + Visual Studio Build Tools (MSVC).

```bash
python -m venv .venv && .venv/Scripts/activate
pip install torch numpy huggingface_hub --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
mkdir -p .venv/bin && printf '#!/usr/bin/env bash\nexec "$(dirname "$0")/../Scripts/python.exe" "$@"\n' > .venv/bin/python3 && chmod +x .venv/bin/python3
export PATH="$PWD/.venv/bin:$PATH"            # los run.sh buscan `python3`
(cd b1_1_strides && ./run.sh --smoke-test)    # B1.1
(cd b2_1_kv_cache && ./run.sh)                # B2.1
```

- **B1.3 con nsys real:** instala Nsight Systems nativo de Windows (2024.5.1 o similar; tiene soporte Pascal) y ejecuta con el
  intérprete real, porque `nsys.exe` no puede lanzar un script bash:
  `PYTHON_BIN="$PWD/.venv/Scripts/python.exe" PATH="<ruta a nsys>:$PATH" ./run.sh` (en `b1_3_profiling/`). El runner usa `--trace=cuda,nvtx` en Windows.
- **B1.2 (CUDA C++):** requiere `nvcc` y `cl.exe` en el entorno (abre una consola "x64 Native Tools" o carga `vcvars64.bat`). El `run.sh` no está
  probado en Windows nativo; el camino verificado es Docker (3.4). Los `*.cu` compilan con MSVC.
- Python nativo de Windows no entiende rutas estilo `/d/...` dentro de `python -c "…"`: pasa rutas por argumentos.

### 3.2 Linux (nativo)

Requisitos: driver NVIDIA, CUDA Toolkit 12.6 (`nvcc`), Nsight Systems, `linux-tools` (`perf`) y Python 3.10+.

```bash
python3 -m venv .venv-linux && source .venv-linux/bin/activate
pip install torch numpy huggingface_hub --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
(cd b1_1_strides && ./run.sh --smoke-test)
(cd b1_2_cuda_gemm && ./run.sh --smoke)       # --full para el benchmark completo
(cd b1_3_profiling && ./run.sh)               # nsys (cuda,nvtx,osrt) y perf si están en el PATH
(cd b2_1_kv_cache && ./run.sh)
```

Es el entorno de referencia para `nsys`, `ncu` y `perf` completos. No se ha probado en un Linux nativo con la GTX 1050 (el portátil es Windows);
sí en contenedor Linux (ver 3.4) y se espera en las VM cloud.

### 3.3 Windows Subsystem for Linux 2 (WSL2)

Requisitos: driver NVIDIA **de Windows** (no instales driver dentro de WSL), CUDA Toolkit para WSL (sin driver) o la imagen Docker, Python 3.10+.
Los comandos son los de Linux (3.2). Es el entorno donde `nvcc` y PyTorch funcionan con la GPU y donde corre Docker Desktop.

- ✅ `nvcc`, PyTorch con CUDA, `run.sh` de B1.1, B1.2, B2.1.
- ❌ **`nsys`**: captura pero no puede importar la traza con esta GPU (`Unrecognized GPU UUID`); usa Nsight Systems nativo de Windows.
- ❌ **`perf`**: no hay paquete `linux-tools` para el kernel de WSL2.

### 3.4 Docker (recomendado para CUDA C++ y para cloud)

Imagen con `nvcc`, cuBLAS, `nsys`, `ncu`, `perf` y Python (`Dockerfile`; requiere NVIDIA Container Toolkit, ya incluido en Docker Desktop/WSL2).

```bash
# Local (GTX 1050): imagen ligera, torch en la venv montada (.venv-linux)
docker build -t ai-infra-study:cuda12.6 --build-arg INSTALL_TORCH=0 .
docker run --rm --gpus all -v "$PWD":/work ai-infra-study:cuda12.6 bash -c "cd b1_2_cuda_gemm && ./run.sh --full"
# Cloud: autocontenida (torch dentro) y arquitectura de la GPU
docker build -t ai-infra-study:cloud --build-arg CUDA_ARCH=sm_90 .      # sm_89 para L4/L40S
```

## 4. Qué herramienta está disponible en cada sistema

✅ funciona · ❌ no disponible o no funciona · ⚠️ con limitaciones · ➖ no aplica. Columnas de la GTX 1050 local; "Cloud" es L4/L40S/H100 con Linux nativo.

| Herramienta | Windows | Linux nativo | WSL2 | Docker (WSL2) | Cloud |
| :--- | :---: | :---: | :---: | :---: | :---: |
| PyTorch + CUDA (B1.1, B2.1) | ✅ | ✅ | ✅ | ✅ | ✅ |
| `nvcc` / CUDA C++ (B1.2) | ⚠️ necesita MSVC; `run.sh` sin probar | ✅ | ✅ | ✅ (verificado) | ✅ |
| `nsys` con traza CUDA (B1.3) | ✅ nativo (`cuda,nvtx`) | ✅ (`cuda,nvtx,osrt`) | ❌ no importa la traza | ❌ (igual que WSL2) | ✅ |
| `perf` (flamegraphs CPU) | ❌ solo Linux | ✅ | ❌ sin `linux-tools` | ❌ | ✅ |
| `torch.compile` / Triton (B1.4) | ❌ en la 1050 (`sm_61 < 7.0`) | ❌ en la 1050 | ❌ en la 1050 | ❌ en la 1050 | ✅ |
| Nsight Compute `ncu` moderno | ❌ en la 1050 | ❌ en la 1050 | ❌ | ❌ | ✅ |
| FlashAttention / FlashInfer / TRT-LLM / SGLang | ❌ (`sm_80+`) | ❌ en la 1050 | ❌ | ❌ | ✅ |
| FP8 nativo (Transformer Engine) | ❌ (`sm_89/90`) | ❌ | ❌ | ❌ | ✅ L40S/H100 |
| Multi-GPU, NCCL, NVLink (B4.x, B5.x) | ➖ 1 GPU | ➖ 1 GPU | ➖ 1 GPU | ➖ 1 GPU | ✅ 2–8x H100 |
| `bash` para los `run.sh` | ⚠️ solo Git Bash/WSL | ✅ | ✅ | ✅ | ✅ |

## 5. Convenciones obligatorias

1. **Nunca generes resultados simulados.** Las cifras de `artifacts/` y de los README deben salir de ejecuciones reales. Si una herramienta no está
   disponible, el runner avisa y no inventa trazas ni tablas (ya se eliminaron los modos que fabricaban datos).
2. **Secretos:** nunca subas claves, tokens ni `.env`. Los `.nsys-rep` están en `.gitignore` porque **embeben las variables de entorno** del
   proceso trazado. Antes de publicar, revisa con `gitleaks detect --source . --log-opts="--all"`.
3. **Sin gasto implícito:** no añadas workflows de CI (salvo `pages.yml`) ni scripts que llamen a servicios cloud de pago sin que se pida expresamente.
4. **Finales de línea LF** en `*.sh`, `*.py`, `*.cu`, `Makefile`, `Dockerfile` (`.gitattributes`). Los `.sh` se ejecutan en Linux.
5. **Cuidado con el disco:** Docker Desktop guarda sus datos en `C:`. Descarga y compila en `D:`, usa `pip --no-cache-dir`, borra zips y no
   dejes contenedores corriendo. No ejecutes `docker system prune` ni borres imágenes que no sean tuyas.
6. **Una venv por entorno en la raíz** (`.venv`, `.venv-linux`), fuera del control de versiones.
7. **Commits:** mensajes convencionales (`fix(b1.3): …`, `docs: …`), un commit por tema. No hagas push sin que se pida.

## 6. Cómo añadir una práctica nueva

1. Carpeta `bX_Y_nombre/` con `run.sh` (idempotente, `--smoke-test` < 30 s, errores claros, salida pegable), `README.md` y código.
2. Salidas JSON/CSV/TXT reales en `artifacts/`. Sin rutas absolutas locales en los artefactos.
3. Añade la fila en el índice del [`README.md`](README.md) (estado ✅/🟡/⬜ y enlace) y los resultados en [`INFORME_RESULTADOS.md`](INFORME_RESULTADOS.md).
4. Documentos del temario en `docs/temario/` y enlazados desde el índice del README.
5. Contenido pensado para Pages: enlaces relativos, fórmulas LaTeX con `$...$` / `$$...$$` (KaTeX), sin rutas `file:///`, y la página añadida a `nav` de `mkdocs.yml`.

## 7. Sitio web (GitHub Pages)

Se genera con **MkDocs Material** (`mkdocs.yml`). `tools/build_site.py` copia los ficheros versionados a `build_src/` (misma estructura, para que
los enlaces relativos de los README funcionen) y compila en `site/`. Las fórmulas LaTeX (`$...$`, `$$...$$`) las renderiza KaTeX mediante
`pymdownx.arithmatex`; los `$$...$$` de una sola línea (p. ej. dentro de listas) se aíslan en la copia de `build_src/`, sin tocar el original.

```bash
pip install -r tools/requirements-site.txt
python tools/build_site.py --strict     # prueba local, resultado en site/
```

Despliegue: workflow `.github/workflows/pages.yml` en cada push a `main` (único workflow permitido; sin coste en repos públicos).
Activar una vez en *Settings → Pages → Build and deployment → Source: GitHub Actions*. Para que una página salga en el menú, añádela a `nav` en `mkdocs.yml`.
