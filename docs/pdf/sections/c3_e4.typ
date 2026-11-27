#import "../theme.typ": *

= Curso 3 · Módulo E4: Kernels Portables CUDA (`sm_61`) y Vulkan Compute (GLSL)

#block-banner(
  "E4",
  "GEMV Ternario __dp4a (sm_61), Fast Hadamard 1024 y Compute Shaders Vulkan",
  "Extensión B3-Edge + Puente HPAC/UMA: de CUDA Pascal a Vulkan portátil (GTX 1050 + Android)",
  "7 sprints de 20m (H4)",
)[
  Escribe los *kernels* críticos de inferencia desde cero: primero en CUDA `sm_61` aprovechando la instrucción entera *`__dp4a`* de Pascal y optimizando con `nsys` hasta superar el `65–75%` de los `112 GB/s`; después en *Vulkan Compute (GLSL/SPIR-V)* ejecutándose sin cambios en PC y móvil *zero-copy*.
]

#infografia(
  "Aceleración __dp4a en Pascal sm_61 y Portabilidad Zero-Copy en Vulkan Compute",
  "Arriba: reducción de presión ALU mediante 32 llamadas __dp4a por bloque de 128 pesos. Abajo: eliminación de copias PCIe H2D/D2H en memoria unificada móvil.",
  height: 188pt,
)[
  // Arriba: Kernel __dp4a
  #place(dx: 8pt, dy: 4pt, text(fill: navy, weight: "bold", size: 8pt)[Kernel GEMV Ternario g128 por Warp (1 Warp = 32 hilos = 1 Fila · 4 pesos/hilo)])
  #diag-node(8pt, 18pt, 108pt, 46pt, "1. Carga Vectorizada", sub: "TQ2_0 coalescente\nuint32 / uint4 (128-bit)\n>65–75% de 112 GB/s", fill: bg-light, col: slate)
  #arrow-r(116pt, 41pt, 130pt, col: slate)
  #diag-node(130pt, 18pt, 114pt, 46pt, "2. Unpack Branchless", sub: "{0,1,2} -> {-1,0,+1}\nEmpaquetado int8x4\nActivación x en Q8_1", fill: bg-teal, col: teal)
  #arrow-r(244pt, 41pt, 258pt, col: teal)
  #diag-node(258pt, 18pt, 106pt, 46pt, "3. 32x __dp4a (1 ciclo)", sub: "acc_i32 += dp4a(w,x)\n1 FMA (sw * sx) final\npor bloque de 128", fill: bg-navy, col: navy)
  #arrow-r(364pt, 41pt, 378pt, col: navy)
  #diag-node(378pt, 18pt, 90pt, 46pt, "4. Reducción Warp", sub: "__shfl_down_sync\n(CUDA) o subgroupAdd\n(Vulkan GLSL)", fill: bg-purple, col: purple)

  #place(dx: 8pt, dy: 74pt, line(length: 460pt, stroke: 0.5pt + border-gray))

  // Abajo: FHT 1024 y Zero-copy
  #place(dx: 8pt, dy: 80pt, text(fill: navy, weight: "bold", size: 8pt)[FHT 1024 Híbrida y Arquitectura de Memoria PC (PCIe) vs Móvil (Unificada)])
  #diag-node(8pt, 96pt, 146pt, 82pt, "FHT 1024 (10 Etapas Mariposa)", sub: "• Etapas 1–5: registros / warp\n  (subgroupShuffleXor en GLSL)\n• Etapas 6–10: shared sdata[1024]\n  sin bank conflicts\n• Compatible subgroup 32 / 64", fill: bg-amber, col: amber)
  #diag-node(162pt, 96pt, 146pt, 82pt, "PC Discreto (GTX 1050)", sub: "CPU RAM <-- PCIe 3.0 --> VRAM\nCopiar activaciones cuesta us.\nConviene mantener todo el\nmodelo residente en 2 GB VRAM.", fill: bg-light, col: slate)
  #diag-node(316pt, 96pt, 152pt, 82pt, "Móvil ARM + Adreno/Mali", sub: "Vulkan HOST_VISIBLE |\nDEVICE_LOCAL (Zero-Copy!)\nCPU NEON y GPU comparten\npáginas físicas -> ideal para\nscheduler híbrido LogFit-Edge.", fill: bg-teal, col: teal)
]

== Equivalencias CUDA `sm_61` $arrow.r.l$ Vulkan Compute (GLSL)

#styled-table(
  columns: (1.2fr, 1.4fr, 1.8fr),
  headers: ("Concepto", "CUDA C++ (B1.2 / sm_61)", "Vulkan Compute (GLSL 4.60 / SPIR-V)"),
  rows: (
    ([*Workgroup / Block ID*], [`blockIdx.{x,y,z}`], [`gl_WorkGroupID.{x,y,z}`]),
    ([*Local Thread ID*], [`threadIdx.{x,y,z}`], [`gl_LocalInvocationID.{x,y,z}`]),
    ([*Shared Memory*], [`__shared__ float sdata[N];`], [`shared float sdata[N];`]),
    ([*Barrera de Bloque*], [`__syncthreads();`], [`barrier();` (+ `memoryBarrierShared();`)]),
    ([*Warp / Subgroup Ops*], [`__shfl_down_sync` (`32` hilos)], [`subgroupAdd` / `subgroupShuffleDown` (`gl_SubgroupSize` 32/64)]),
  ),
)

== Sprints E4.1–E4.7 (`Hito H4`)

#unit-header(
  "E4",
  "Kernels GEMV Ternario y FHT en CUDA sm_61 y Vulkan GLSL",
  "[MOB] 1s + [LOC-PC] 4s + [AGENT] 1s + [LOC-AND] 1s",
  "B1.2, E3",
  "E5, PR llama.cpp",
  "GTX 1050 (nvcc/nsys/glslc) + Android Vulkan",
  "gemv_tq2_0_dp4a > 65-75% de 112 GB/s; mismos shaders .spv en GTX 1050 y móvil.",
)

#practica-box(title: "Ruta de Implementación E4.1–E4.7")[
  1. *E4.1 `[MOB]`:* Consolidar la tabla CUDA $arrow.r.l$ Vulkan y manejo dinámico de `gl_SubgroupSize` (`32` NVIDIA/Adreno vs `64` Mali/AMD).
  2. *E4.2–E4.3 `[LOC-PC]`:* Escribir `kernels/cuda_sm61/gemv_ternary.cu` (`gemv_tq2_0_dp4a` para $2048 times 2048$ y $6144 times 2048$, error `< 1e-2`), perfilar con `nsys` y vectorizar cargas (`uint4`) superando `65–75%` del pico (`~73–84 GB/s`).
  3. *E4.4 `[LOC-PC]`:* Implementar `kernels/cuda_sm61/fht.cu` (`512`/`1024`) combinando shuffles intra-warp y *shared memory* sin conflictos de banco.
  4. *E4.5–E4.6 `[AGENT]`+`[LOC-PC]`+`[LOC-AND]`:* Portar a `gemv_ternary.comp` y `fht_1024.comp` (compilados con `glslc`), validar en `bench_kernels.cpp` sobre GTX 1050 y correr idéntico SPIR-V en móvil *zero-copy* midiendo `GB/s` y `mWh/op` frente a NEON.
  5. *E4.7 `[LOC-PC]`:* Revisar *issues* de `ggml-vulkan` en `llama.cpp` (FHT 1024 o `TQ1_0`/`TQ2_0`) e integrar tu shader en rama local como candidato a PR público.
]

#ejercicio-box[
  *Ejercicio E4-A (`__dp4a` en Pascal):* Diseña el ciclo de 128 pesos `TQ2_0` contra activaciones `Q8_1` (`int8` + escala $s_x$) usando `__dp4a(a, b, c)` y calcula cuántas instrucciones flotantes ahorras por bloque.

  *Ejercicio E4-B (*Zero-Copy* Móvil y `LogFit`):* Explica cómo el mapeo persistente `HOST_VISIBLE | DEVICE_LOCAL` en Vulkan habilita un *scheduler* híbrido CPU+GPU sin copias PCIe, conectando con `TheUnixRoot/Logfit`.
]

#solucion-box[
  - *Sol. E4-A:* Desempaquetas 4 pesos a `int8x4` en un `int32` y ejecutas 32 llamadas `__dp4a` en `int32` (1 ciclo/inst), aplicando una única multiplicación flotante por $(s_w dot s_x)$ al cerrar el bloque de 128 (`~8x–10x` menos instrucciones ALU/conversión que 128 FMAs en FP32).
  - *Sol. E4-B:* Sin bus PCIe de por medio, CPU ARM NEON y GPU Vulkan ven el mismo búfer físico: CPU ejecuta *embeddings* / PLE o *router* mientras Vulkan lanza GEMV ternario denso sin copias H2D/D2H, replicando el reparto adaptativo de `LogFit` (`oneTBB`) en silicio móvil moderno.
]

#fuentes-box[
  1. NVIDIA CUDA Math API (`__dp4a`): `https://docs.nvidia.com/cuda/cuda-math-api/`
  2. Khronos GLSL Subgroup Extensions (`GL_KHR_shader_subgroup`): `https://github.com/KhronosGroup/GLSL`
  3. `llama.cpp` `ggml-vulkan`: `https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-vulkan`
]
