# Mi Progreso y Generador de Handover (20 min)

Marca tus unidades completadas en ambos cursos (**Tronco Común**, **Curso 1** y **Curso 2**). El progreso se guarda localmente en tu navegador (`localStorage`) sin necesidad de servidor externo.

## Panel de Avance por Itinerario

<div id="campus-progress-app">Cargando panel interactivo de progreso (disponible en el sitio web GitHub Pages)...</div>

---

## Generador Rápido de Bloque `HANDOVER`

Rellena los 5 campos al terminar cualquier micro-sprint de **20 minutos** para pegar el bloque en tu siguiente chat `[FORK]`:

<div class="campus-handover-box">
  <p>
    <label><strong>Unidad actual:</strong><br/>
    <input id="ho-unit" type="text" value="C2-B1.2" style="width:100%;padding:0.4rem;margin-top:0.2rem;" /></label>
  </p>
  <p>
    <label><strong>Modo:</strong><br/>
    <select id="ho-mode" style="width:100%;padding:0.4rem;margin-top:0.2rem;">
      <option value="LOC">[LOC] Local</option>
      <option value="MOB">[MOB] Móvil / Lectura</option>
      <option value="PREP">[PREP] Pre-Flight</option>
      <option value="CLD">[CLD] Cloud GPU</option>
      <option value="LAB">[LAB] Databricks Sandbox</option>
    </select></label>
  </p>
  <p>
    <label><strong>Aprendido o verificado hoy:</strong><br/>
    <input id="ho-learned" type="text" value="Gate MLflow local validado con alias @staging y firma" style="width:100%;padding:0.4rem;margin-top:0.2rem;" /></label>
  </p>
  <p>
    <label><strong>Artefacto generado:</strong><br/>
    <input id="ho-artifact" type="text" value="docs/adr/001-serving-topology.md" style="width:100%;padding:0.4rem;margin-top:0.2rem;" /></label>
  </p>
  <p>
    <label><strong>Siguiente micro-sprint (20m):</strong><br/>
    <input id="ho-next" type="text" value="C2-B1.3 [LOC] - Generar audit-evidence.json con SHA-256" style="width:100%;padding:0.4rem;margin-top:0.2rem;" /></label>
  </p>
  <pre><code id="handover-output">HANDOVER: ...</code></pre>
  <button id="handover-copy" class="md-button md-button--primary" type="button">Copiar YAML Handover</button>
</div>

---

## Resumen Oficial de Entregas Verificadas en el Repo

Las métricas de hardware reales registradas actualmente en el repositorio están documentadas en [`INFORME_RESULTADOS.md`](../../INFORME_RESULTADOS.md):

- **B1.1 (GTX 1050):** `transpose` zero-copy (`0.0036 ms`), `.contiguous()` `33.44 GB/s` vs `clone()` `91.77 GB/s` (`2.74x` en `2048x2048`).
- **B1.2 (GTX 1050):** `gemm_smem_tiled` (`32x32`) `281.4 GFLOP/s` (`11.7x` vs `naive`; variante `16x16` añadida para mitigar ocupación Pascal); `warp_online_softmax` `1.31x` vs `3-pass`.
- **B1.3 (GTX 1050 + `nsys` real):** `pin_memory` `2.08x` en H2D; paso total `1.01x` dominado por `~600` kernels eager (~218 ms GPU).
- **B2.1 (`kv_calc.py`):** cálculo GQA/MLA con capacidades reales (`Llama-3-8B` FP16: `26` streams L40S / `55` H100; `Qwen2.5-72B` FP8/FP16 requiere TP o KV-FP8).
- **B2.2 (`[PREP]`):** comparador `vLLM` vs `SGLang` + cliente de métricas Prometheus listo para `Cloud-2`.
