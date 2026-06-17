#!/usr/bin/env python3
"""Generate agents/notebook/multi_agent_triage_demo.ipynb."""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebook" / "multi_agent_triage_demo.ipynb"

NB = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    cells = [
        md(
            """# AGENTS_030 — Multi-Agent Infrastructure Triage (Pydantic AI)

Orchestrates **change-impact-agent** (pre-merge) and **log-analysis-agent** (post-CI) via Pydantic AI tools on vLLM / MI300.

| Agent | When | Tool |
|-------|------|------|
| change-impact-agent | Before merge | `analyze_upstream_pr` |
| log-analysis-agent | After CI fails | `triage_ci_log` |
| Orchestrator | User chat | picks tools at runtime |

## Steps
1. Launch vLLM (terminal)
2. Config + verify
3. Install deps
4. Direct tool smoke test (no LLM)
5. Pydantic AI orchestrator demo
"""
        ),
        md(
            """## Step 1: Launch vLLM

```bash
VLLM_USE_TRITON_FLASH_ATTN=0 \\
vllm serve Qwen/Qwen3-30B-A3B \\
  --served-model-name Qwen3-30B-A3B \\
  --api-key abc-123 \\
  --port 8000 \\
  --enable-auto-tool-choice \\
  --tool-call-parser hermes \\
  --trust-remote-code
```

Monitor: `watch rocm-smi`
"""
        ),
        code(
            """import os
import sys
from pathlib import Path

def resolve_agents_dir() -> Path:
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "multi_agent_tools.py").is_file():
            return candidate
        if candidate.name == "notebook" and (candidate.parent / "multi_agent_tools.py").is_file():
            return candidate.parent
        if (candidate / "agents" / "multi_agent_tools.py").is_file():
            return candidate / "agents"
    raise RuntimeError("Start Jupyter from TheRock/agents/ or agents/notebook/")

AGENTS_DIR = resolve_agents_dir()
THEROCK_ROOT = AGENTS_DIR.parent if AGENTS_DIR.name == "agents" else AGENTS_DIR
NOTEBOOK_OUT = AGENTS_DIR / "notebook" / "out"

if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

from multi_agent_tools import DEFAULT_DEMO_LOG, DEFAULT_DEMO_PR, DEMO_PRS

BASE_URL = os.environ.get("VLLM_BASE_URL", os.environ.get("BASE_URL", "http://localhost:8000/v1"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "abc-123")
LLM_MODEL = os.environ.get("VLLM_MODEL", os.environ.get("LLM_MODEL", "Qwen3-30B-A3B"))

os.environ["BASE_URL"] = BASE_URL
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LLM_MODEL"] = LLM_MODEL

print("AGENTS_DIR   =", AGENTS_DIR)
print("THEROCK_ROOT =", THEROCK_ROOT)
print("BASE_URL     =", BASE_URL)
print("LLM_MODEL    =", LLM_MODEL)
print("DEMO_PRS     =", DEMO_PRS)
print("DEFAULT_LOG  =", DEFAULT_DEMO_LOG)
"""
        ),
        code(
            """import httpx

VLLM_REACHABLE = False
VLLM_TOOL_SUPPORT = False
headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

try:
    r = httpx.get(f"{BASE_URL}/models", headers=headers, timeout=15.0)
    VLLM_REACHABLE = r.status_code < 400
    if VLLM_REACHABLE:
        print("vLLM reachable:", [m.get("id") for m in r.json().get("data", [])[:3]])
    else:
        print("vLLM models endpoint:", r.status_code, r.text[:200])
except Exception as exc:
    print("vLLM not reachable:", exc)
    print("Start vLLM in a terminal (Step 1) — direct tool tests still work below.")

if VLLM_REACHABLE:
    probe = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}}],
            "tool_choice": "auto",
            "max_tokens": 16,
        },
        timeout=60.0,
    )
    VLLM_TOOL_SUPPORT = probe.status_code < 400
    print("Tool calling:", "OK" if VLLM_TOOL_SUPPORT else probe.text[:300])
"""
        ),
        code(
            """import sys
!{sys.executable} -m pip install -q -r "{AGENTS_DIR / 'requirements-notebook.txt'}"
"""
        ),
        md("## Step 4: Direct tool smoke test (deterministic, no LLM)"),
        code(
            """from multi_agent_tools import (
    list_demo_assets,
    run_change_impact_for_pr,
    run_log_analysis_for_path,
    run_infrastructure_triage_loop,
)

print(list_demo_assets())
print()
print(run_change_impact_for_pr(DEFAULT_DEMO_PR))
print()
print(run_log_analysis_for_path(str(DEFAULT_DEMO_LOG)))
print()
print(run_infrastructure_triage_loop(DEFAULT_DEMO_PR, str(DEFAULT_DEMO_LOG))[:1200], "...")
"""
        ),
        md(
            """## Step 5: Pydantic AI orchestrator

The orchestrator LLM chooses which specialist tool to call based on your prompt.

**Requires vLLM** with `--enable-auto-tool-choice --tool-call-parser hermes`.
"""
        ),
        code(
            """from multi_agent_tools import build_agent_model, build_orchestrator_agent

orchestrator = None
if globals().get("VLLM_TOOL_SUPPORT"):
    model = build_agent_model(BASE_URL, OPENAI_API_KEY, LLM_MODEL)
    orchestrator = build_orchestrator_agent(model)
    print("Orchestrator ready (Pydantic AI + multi-agent tools).")
else:
    print(
        "Skipping orchestrator — vLLM needs tool calling. "
        "Re-run Step 1 with --enable-auto-tool-choice --tool-call-parser hermes"
    )
"""
        ),
        code(
            """from pydantic_ai.exceptions import ModelHTTPError

async def ask_orchestrator(prompt: str):
    if orchestrator is None:
        print("Orchestrator not available — use Step 4 direct tools.")
        return None
    try:
        result = await orchestrator.run(prompt)
        print(result.output)
        return result
    except ModelHTTPError as exc:
        print("ModelHTTPError:", exc)
        return None

# Scenario A: pre-merge only
# await ask_orchestrator("Analyze upstream TheRock PR 5572 for blast radius and rollout labels.")

# Scenario B: post-CI only
# await ask_orchestrator(f"Triage this CI log for errors: {DEFAULT_DEMO_LOG}")

# Scenario C: full infrastructure loop (hackathon use case)
await ask_orchestrator(
    f"We planned infrastructure change PR {DEFAULT_DEMO_PR}. CI failed afterward. "
    f"Run full infrastructure triage using log {DEFAULT_DEMO_LOG}. "
    "Summarize impact, historical-style fixes, and safer rollout."
)
"""
        ),
        md(
            """## Outputs

Artifacts land in `agents/notebook/out/`:

- `pr-<N>/report.json` — change-impact briefing
- `log-<name>/report.json` — log triage

Open HTML reports from change-impact `sample-runs/` or re-run with live `GITHUB_TOKEN`.
"""
        ),
    ]

    NB["cells"] = cells
    OUT.parent.mkdir(parents=True, exist_ok=True)

    import json

    OUT.write_text(json.dumps(NB, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
