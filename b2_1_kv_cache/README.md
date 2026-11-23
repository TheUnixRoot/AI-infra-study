# B2.1: Aritmética del KV-Cache y presupuesto de VRAM

Calculadora en Python (CPU, sin GPU) que, a partir del `config.json` de un modelo de Hugging Face, calcula la VRAM de
pesos, el KV-cache por token y por secuencia, y cuántas peticiones concurrentes caben en una L40S (48 GB) y una H100 (80 GB).

- Temario: [`docs/temario/00-master-playbook.md`](../docs/temario/00-master-playbook.md) (unidad `B2.1`).
- Resultados y análisis: [`INFORME_RESULTADOS.md`](../INFORME_RESULTADOS.md#5-b21-aritmética-del-kv-cache-cpu-analítico).

## Fórmulas

- **MHA / GQA:** `KV_bytes = 2 · L · n_kv · d_head · S · B · bytes_por_elemento`
- **MLA (DeepSeek-V2/V3):** `KV_bytes = L · (kv_lora_rank + qk_rope_head_dim) · S · B · bytes_por_elemento`
- **VRAM útil:** `total · gpu_mem_util − overhead`; concurrencia = `⌊(útil − pesos) / KV_por_secuencia⌋`.
  Si no cabe ni una secuencia el estado es `OOM: Available KV < 1 seq`.

## Ejecución

```bash
./run.sh                      # smoke test + matriz completa (Llama-3-8B, Qwen2.5-72B, DeepSeek-V3)
python3 kv_calc.py --help     # CLI: --models, --seq-len, --gpu-mem-util, --overhead-gb, --output-json/--output-csv
```

Si no hay acceso a Hugging Face (o el modelo es *gated*), usa configuraciones embebidas verificadas. Salidas en
`../artifacts/b2_1_kv_report.json` y `.csv`.

## Resultado (8192 tokens, util 0.90, overhead 2 GB)

| Modelo | Pesos + KV | KV/token | L40S | H100 |
| :--- | :--- | :---: | :---: | :---: |
| Llama-3-8B | FP16 + FP16 | 128 KiB | 26 | 55 |
| Llama-3-8B | FP8 + FP8 | 64 KiB | 67 | 125 |
| Qwen2.5-72B | FP8 + FP8 | 160 KiB | 0 | 1 |
| DeepSeek-V3 (MLA) | FP8 + FP8 | 34,3 KiB | 0 | 0 |
