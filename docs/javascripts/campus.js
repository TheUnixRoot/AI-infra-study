// Interactive Campus Progress Tracker & 20m Handover Generator (localStorage-only, zero backend)
(() => {
  const STORAGE_KEY = "ai_infra_campus_progress_v1";

  const CURRICULUM = [
    { id: "TC-1", track: "tronco", label: "TC.1 Física Prefill vs Decode y Roofline" },
    { id: "TC-2", track: "tronco", label: "TC.2 Cargas agénticas, Prefix Caching y RAG" },
    { id: "TC-3", track: "tronco", label: "TC.3 Metodología científica y CI Evaluation Gates" },
    { id: "TC-4", track: "tronco", label: "TC.4 Trazabilidad (nsys/OTel) y FinOps $/1M" },
    { id: "TC-5", track: "tronco", label: "TC.5 Kubernetes Serving, SLOs y Cold Starts" },
    { id: "TC-6", track: "tronco", label: "TC.6 Sistema Operativo 20m y Regla 1-a-1" },

    { id: "C1-B1.1", track: "c1", label: "C1 B1.1 Roofline, strides y .contiguous() (GTX 1050)", defaultDone: true },
    { id: "C1-B1.2", track: "c1", label: "C1 B1.2 CUDA C++ coalescencia, tiling 32x32/16x16 y warp softmax", defaultDone: true },
    { id: "C1-B1.3", track: "c1", label: "C1 B1.3 Profiling con nsys y antipatrones H2D/CPU sync", defaultDone: false },
    { id: "C1-B1.4", track: "c1", label: "C1 B1.4 Triton y Nsight Compute (ncu) [Cloud-1]", defaultDone: false },
    { id: "C1-B2.1", track: "c1", label: "C1 B2.1 Aritmética del KV-Cache y calculadora MLA/GQA", defaultDone: true },
    { id: "C1-B2.2", track: "c1", label: "C1 B2.2 vLLM vs SGLang y FlashInfer ([PREP] listo)", defaultDone: false },
    { id: "C1-B2.3", track: "c1", label: "C1 B2.3 TensorRT-LLM, Dynamo y PDD (NIXL)", defaultDone: false },
    { id: "C1-B2.4", track: "c1", label: "C1 B2.4 Cargas agénticas (Prefix Caching + XGrammar)", defaultDone: false },
    { id: "C1-B3.1", track: "c1", label: "C1 B3.1 Teoría de cuantización (FP8/INT4/AWQ/NVFP4)", defaultDone: false },
    { id: "C1-B3.2", track: "c1", label: "C1 B3.2 FP8 vs AWQ + KV-Cache FP8 en L40S/H100", defaultDone: false },
    { id: "C1-B3.3", track: "c1", label: "C1 B3.3 Speculative Decoding (EAGLE-3 / N-gram / MTP)", defaultDone: false },
    { id: "C1-B3.4", track: "c1", label: "C1 B3.4 Evals de calidad y frontera de Pareto", defaultDone: false },
    { id: "C1-B4.1", track: "c1", label: "C1 B4.1 Paralelismo DP/TP/PP/CP y NCCL", defaultDone: false },
    { id: "C1-B4.2", track: "c1", label: "C1 B4.2 MoE y Wide Expert Parallelism (All-to-All)", defaultDone: false },
    { id: "C1-B4.3", track: "c1", label: "C1 B4.3 PDD, LMCache y routing cache-aware", defaultDone: false },
    { id: "C1-B4.4", track: "c1", label: "C1 B4.4 Interconexiones NVLink/IB/RoCEv2 y nccl-tests", defaultDone: false },
    { id: "C1-B5.1", track: "c1", label: "C1 B5.1 Metodología de benchmarking reproducible", defaultDone: false },
    { id: "C1-B5.2", track: "c1", label: "C1 B5.2 Cluster acceptance (dcgmi + Xid + burn-in)", defaultDone: false },
    { id: "C1-B5.3", track: "c1", label: "C1 B5.3 Entrenamiento Megatron-LM, MFU y stragglers", defaultDone: false },
    { id: "C1-B5.4", track: "c1", label: "C1 B5.4 Kubernetes GPU, MIG/Time-Slicing y cold starts", defaultDone: false },

    { id: "C2-B1.1", track: "c2", label: "C2 B1.1 Databricks Mosaic AI vs Seldon en K8s (ADR-001)", defaultDone: false },
    { id: "C2-B1.2", track: "c2", label: "C2 B1.2 MLflow 3 LoggedModel, aliases y gates de promoción", defaultDone: false },
    { id: "C2-B1.3", track: "c2", label: "C2 B1.3 Unity Catalog lineage, RBAC/ABAC y audit-evidence.json", defaultDone: false },
    { id: "C2-B2.1", track: "c2", label: "C2 B2.1 Agentic Evals y CI Evaluation Gate (mlflow.evaluate)", defaultDone: false },
    { id: "C2-B2.2", track: "c2", label: "C2 B2.2 Salud vectorial RAG (OpenSearch, Recall@K, index lag)", defaultDone: false },
    { id: "C2-B2.3", track: "c2", label: "C2 B2.3 OpenTelemetry GenAI, drift, PII guardrails y FinOps", defaultDone: false },
    { id: "C2-B3.1", track: "c2", label: "C2 B3.1 Seldon Core v2 y Knative Serving (KPA concurrencia)", defaultDone: false },
    { id: "C2-B3.2", track: "c2", label: "C2 B3.2 Istio shadow/canary + ArgoCD Rollouts auto-rollback", defaultDone: false },
    { id: "C2-B3.3", track: "c2", label: "C2 B3.3 Crossplane XRDs, PrivateLink e IRSA/Workload Identity", defaultDone: false },
    { id: "C2-B4.1", track: "c2", label: "C2 B4.1 SLIs/SLOs agénticos y error budgets", defaultDone: false },
    { id: "C2-B4.2", track: "c2", label: "C2 B4.2 GameDay simulacro de incidente y post-mortem blameless", defaultDone: false },
    { id: "C2-B4.3", track: "c2", label: "C2 B4.3 Multi-región EU, soberanía y degradación elegante", defaultDone: false },
  ];

  function loadState() {
    const base = {};
    CURRICULUM.forEach((item) => {
      base[item.id] = Boolean(item.defaultDone);
    });
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
      return { ...base, ...saved };
    } catch {
      return base;
    }
  }

  function saveState(state) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      // Ignore storage quota errors in restricted browsers.
    }
  }

  function buildHandoverYaml(unitId, mode, learned, artifact, nextStep) {
    return [
      "HANDOVER:",
      `  unidad_actual: "${unitId}"`,
      `  modo: "${mode}"`,
      '  estado: "en_progreso"',
      "  aprendido_o_verificado:",
      `    - "${learned.replace(/"/g, "'")}"`,
      `  artefacto_generado: "${artifact.replace(/"/g, "'")}"`,
      `  siguiente_micro_sprint_20m: "${nextStep.replace(/"/g, "'")}"`,
    ].join("\n");
  }

  function renderApp(container) {
    let state = loadState();
    let filter = "all";

    function update() {
      const count = (track) => {
        const items = CURRICULUM.filter((x) => !track || x.track === track);
        const done = items.filter((x) => state[x.id]).length;
        return { done, total: items.length, pct: Math.round((done / items.length) * 100) };
      };

      const tc = count("tronco");
      const c1 = count("c1");
      const c2 = count("c2");
      const total = count(null);
      const visible = CURRICULUM.filter((x) => filter === "all" || x.track === filter);

      container.innerHTML = `
        <div class="campus-progress-grid">
          <div class="campus-stat-card">
            <h4>Global Campus</h4>
            <strong>${total.done} / ${total.total}</strong> (${total.pct}%)
            <div class="campus-meter"><span style="width:${total.pct}%"></span></div>
          </div>
          <div class="campus-stat-card">
            <h4>Tronco Común</h4>
            <strong>${tc.done} / ${tc.total}</strong> (${tc.pct}%)
            <div class="campus-meter"><span style="width:${tc.pct}%"></span></div>
          </div>
          <div class="campus-stat-card">
            <h4>Curso 1 · AI Infra</h4>
            <strong>${c1.done} / ${c1.total}</strong> (${c1.pct}%)
            <div class="campus-meter"><span style="width:${c1.pct}%"></span></div>
          </div>
          <div class="campus-stat-card">
            <h4>Curso 2 · Agentic MLOps</h4>
            <strong>${c2.done} / ${c2.total}</strong> (${c2.pct}%)
            <div class="campus-meter"><span style="width:${c2.pct}%"></span></div>
          </div>
        </div>

        <p style="display:flex;flex-wrap:wrap;gap:0.5rem;align-items:center;">
          <strong>Filtrar por itinerario:</strong>
          <button class="md-button ${filter === "all" ? "md-button--primary" : ""}" data-filter="all">Todos (${total.total})</button>
          <button class="md-button ${filter === "tronco" ? "md-button--primary" : ""}" data-filter="tronco">Tronco (${tc.total})</button>
          <button class="md-button ${filter === "c1" ? "md-button--primary" : ""}" data-filter="c1">Curso 1 (${c1.total})</button>
          <button class="md-button ${filter === "c2" ? "md-button--primary" : ""}" data-filter="c2">Curso 2 (${c2.total})</button>
          <button class="md-button" id="campus-reset-btn" style="margin-left:auto;">Restablecer por defecto</button>
        </p>

        <div class="campus-checklist">
          ${visible
            .map(
              (item) => `
              <div class="campus-checklist-item">
                <input type="checkbox" id="chk-${item.id}" data-unit="${item.id}" ${state[item.id] ? "checked" : ""} />
                <label for="chk-${item.id}"><code>${item.id}</code> · ${item.label}</label>
              </div>`
            )
            .join("")}
        </div>
      `;

      container.querySelectorAll("button[data-filter]").forEach((btn) => {
        btn.addEventListener("click", () => {
          filter = btn.getAttribute("data-filter");
          update();
        });
      });

      const resetBtn = container.querySelector("#campus-reset-btn");
      if (resetBtn) {
        resetBtn.addEventListener("click", () => {
          localStorage.removeItem(STORAGE_KEY);
          state = loadState();
          update();
        });
      }

      container.querySelectorAll('input[type="checkbox"][data-unit]').forEach((cb) => {
        cb.addEventListener("change", () => {
          state[cb.getAttribute("data-unit")] = cb.checked;
          saveState(state);
          update();
        });
      });
    }

    update();
  }

  function bindHandoverForm() {
    const out = document.getElementById("handover-output");
    const btn = document.getElementById("handover-copy");
    const inputs = ["ho-unit", "ho-mode", "ho-learned", "ho-artifact", "ho-next"]
      .map((id) => document.getElementById(id))
      .filter(Boolean);
    if (!out || inputs.length !== 5) return;

    const refresh = () => {
      out.textContent = buildHandoverYaml(
        inputs[0].value || "C2-B1.2",
        inputs[1].value || "LOC",
        inputs[2].value || "Gate MLflow local validado",
        inputs[3].value || "docs/adr/001-serving-topology.md",
        inputs[4].value || "C2-B1.3 [LOC] - Generar audit-evidence.json"
      );
    };
    inputs.forEach((el) => el.addEventListener("input", refresh));
    if (btn) {
      btn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(out.textContent);
          btn.textContent = "Copiado ✓";
          setTimeout(() => (btn.textContent = "Copiar YAML Handover"), 1600);
        } catch {
          btn.textContent = "Selecciona y copia manualmente";
        }
      });
    }
    refresh();
  }

  const init = () => {
    const app = document.getElementById("campus-progress-app");
    if (app) renderApp(app);
    bindHandoverForm();
  };

  if (typeof document$ !== "undefined") {
    document$.subscribe(init);
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
