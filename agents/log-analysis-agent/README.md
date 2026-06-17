# Log Analysis Agent (AGENTS_030)

Post-CI failure triage for TheRock and other build logs. Deterministic **tool-only** analysis by default (no LLM); optional LangGraph ReAct agent with `--agent`.

Complements [change-impact-agent](../change-impact-agent/) (pre-merge blast radius) — together they cover the full triage loop.

## Quick start

```powershell
pip install -r agents/log-analysis-agent/requirements.txt

python agents/log-analysis-agent/analyze_log.py `
  --log agents/log-analysis-agent/tests/fixtures/example.log `
  --output-dir agents/log-analysis-agent/out/example
```

## TheRock presets

```powershell
python agents/log-analysis-agent/analyze_log.py `
  --log path/to/multi_arch_job.log `
  --preset therock_multi_arch `
  --output-dir out/run-123
```

Presets: `therock_multi_arch`, `therock_install`, `therock_pytorch`, `therock_unit_tests`, `custom`.

Each preset adds workflow-specific grep patterns (ninja/cmake/gfx, apt/dpkg, pytorch/hip, ctest, etc.).

## Outputs

| File | Description |
|------|-------------|
| `report.json` | Structured errors, KB lookups, stats, stack traces |
| `report.html` | Human-readable triage report |
| `executive_summary.md` | Template summary for PR/issue discussion |

## GitHub Actions integration

### Analyze a failed run by ID

```powershell
# Fork or upstream (needs GITHUB_TOKEN / PAT with Actions read)
python agents/log-analysis-agent/analyze_log.py `
  --github-run-id 12345678 `
  --repo ROCm/TheRock `
  --output-dir agents/log-analysis-agent/out/run-12345678
```

Preset is inferred from workflow name (`--preset auto`, default) or override with `--preset therock_multi_arch`.

### Poll recent upstream failures

```powershell
python agents/log-analysis-agent/fetch_failed_runs.py `
  --repo ROCm/TheRock `
  --preset therock_multi_arch `
  --max-runs 3 `
  --analyze
```

Dedup state: `out/.ingested_run_ids.json`.

### Automated workflows (fork)

| Workflow | Trigger | Scope |
|----------|---------|-------|
| `log-analysis-agent.yml` | `workflow_run` when monitored CI fails | Same repo (fork) — reactive |
| `log-analysis-upstream.yml` | `workflow_dispatch` + schedule (6h) | Upstream `ROCm/TheRock` via PAT |

**Fork reactive** fires when Multi-Arch CI, Unit Tests, PyTorch Wheels, or install tests fail. Downloads job logs, uploads report artifact, optional PR comment.

**Upstream poll** requires a PAT with **Actions read** on public `ROCm/TheRock` stored as `GITHUB_TOKEN` secret (default fork token cannot read upstream Actions).

## Optional agent mode

Requires `requirements-agent.txt` and an LLM backend: **OpenAI**, **NVIDIA API**, or **vLLM on MI300** (same pattern as [change-impact-agent](../change-impact-agent/)).

### vLLM on MI300

```bash
VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve Qwen/Qwen3-30B-A3B \
  --served-model-name Qwen3-30B-A3B \
  --api-key abc-123 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code
```

```powershell
copy agents\log-analysis-agent\.env.example agents\log-analysis-agent\.env
# Set USE_VLLM=1, VLLM_BASE_URL, VLLM_MODEL, OPENAI_API_KEY=abc-123

pip install -r agents/log-analysis-agent/requirements-agent.txt

python agents/log-analysis-agent/analyze_log.py `
  --log job.log --agent --model Qwen3-30B-A3B
```

Tool-calling flags are required for LangGraph ReAct agent mode with Qwen3 on vLLM.

### Executive summary with vLLM

```powershell
python agents/log-analysis-agent/analyze_log.py `
  --log agents/log-analysis-agent/tests/fixtures/example.log `
  --output-dir out/example --summary-backend vllm

# Or standalone:
python agents/log-analysis-agent/summarize_log.py `
  --backend vllm --input out/example/report.json
```

Set `LOG_SUMMARY_BACKEND=vllm` in `.env` to default summaries to vLLM after every `analyze_log.py` run.

### OpenAI cloud

```powershell
pip install -r agents/log-analysis-agent/requirements-agent.txt

python agents/log-analysis-agent/analyze_log.py `
  --log job.log --agent --model gpt-4o-mini
```

## Knowledge base

Built-in patterns in `knowledge/patterns.json` plus learned resolutions in `knowledge/resolutions.jsonl`.

```powershell
python agents/log-analysis-agent/analyze_log.py `
  --record-resolution "Connection timeout" "Check DB firewall and pool size"
```

## Tests

```powershell
python -m pytest agents/log-analysis-agent/tests/ -q
```

## Sample run

See [sample-runs/log-example/](sample-runs/log-example/) for a tool-only report on the bundled example log.
