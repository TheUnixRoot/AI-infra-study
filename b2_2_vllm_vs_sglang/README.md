# B2.2 — vLLM (V1) vs SGLang con FlashInfer

Comparativa reproducible de `Qwen/Qwen2.5-Coder-7B-Instruct` (BF16) en **1× L40S (sm_89)** o **1× H100 (sm_90)**.
Estado: `[PREP]` completo (0 €); `[CLD]` pendiente. La GTX 1050 local no puede ejecutarlo (AGENTS.md §2).

| Fichero | Función |
| :--- | :--- |
| `run_compare.sh` | `--dry-run` (CPU, ~2 s) o modo real: arranca un motor, espera `/health`, mide B = 1, 8, 32, 64 (1024 in / 256 out), lo para y libera VRAM antes del siguiente |
| `bench_client.py` | Cliente asyncio (aiohttp): warmup, TTFT/TPOT/ITL p50/p95/p99, tok/s, scrapeo de `/metrics` (KV-cache %, preemptions) |
| `cloud_bootstrap.sh` | Plantilla de la VM: `sudo shutdown -h +90`, `nvidia-smi`, dos venvs (vLLM, SGLang), descarga del modelo |

```bash
bash b2_2_vllm_vs_sglang/run_compare.sh --dry-run          # local, sin GPU (alias: --smoke-test)
# En Cloud-2:
bash b2_2_vllm_vs_sglang/cloud_bootstrap.sh
bash b2_2_vllm_vs_sglang/run_compare.sh [--engines vllm] [--force]
```

Salidas reales: `artifacts/b2_2_raw_<motor>.json`, `artifacts/b2_2_results.json`, `artifacts/b2_2_summary.md`.
El dry-run escribe **solo** en `artifacts/b2_2_dryrun/`, marcado `synthetic: true` (AGENTS.md §5.1); nunca se citan esas cifras.

## Decisiones de diseño (revisar en cloud)
- Prompts como **IDs de token aleatorios únicos** por petición: sin reutilización de prefijo, para no favorecer a RadixAttention ni al APC de vLLM. `--prefix-share 0.5` mide el caso favorable por separado.
- `ignore_eos` + `min_tokens` fuerzan 256 tokens de salida. TPOT por petición = (e2e − TTFT)/(n − 1); ITL = huecos entre chunks SSE.
- Nombres de métricas Prometheus probados: `vllm:kv_cache_usage_perc`/`gpu_cache_usage_perc`, `vllm:num_preemptions(_total)`, `sglang:token_usage`. SGLang puede no exponer preemptions (`n/d`). **Verifica los nombres con `curl :8000/metrics` en la primera ejecución**; no se han podido comprobar sin GPU.
- Flags (`VLLM_ATTENTION_BACKEND=FLASHINFER`, `--attention-backend flashinfer`, `--enable-metrics`, `--disable-log-requests`) dependen de la versión instalada; si un motor rechaza alguno, el log queda en `artifacts/b2_2_logs/`.
