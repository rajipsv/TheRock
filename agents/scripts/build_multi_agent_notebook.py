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
            """# AGENTS_030 — Multi-Agent Infrastructure Triage (MI300 / vLLM)

Runs **change-impact-agent** (pre-merge) and **log-analysis-agent** (post-CI) as separate notebook steps.

| Step | Agent | Demo asset |
|------|-------|------------|
| 5 | change-impact-agent | PR **5572** (sample briefing) |
| 6 | log-analysis-agent | GHA run **[27710372755](https://github.com/ROCm/TheRock/actions/runs/27710372755)** job `81992436725` (rocSPARSE OOM) |

## Steps
1. Launch vLLM (terminal)
2. Config + verify vLLM
3. Install deps
4. *(commented out)* combined direct smoke test — use Steps 5–6 instead
5. **change-impact-agent** — pre-merge blast radius / rollout
6. **log-analysis-agent** — post-CI log triage + vLLM executive summary
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
  --trust-remote-code
```

For optional Pydantic orchestrator later, add: `--enable-auto-tool-choice --tool-call-parser hermes`

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

from multi_agent_tools import (
    DEFAULT_DEMO_LOG,
    DEFAULT_DEMO_LOG_URL,
    DEFAULT_DEMO_JOB_ID,
    DEFAULT_DEMO_PR,
    DEFAULT_DEMO_RUN_ID,
    DEMO_PRS,
)

BASE_URL = os.environ.get("VLLM_BASE_URL", os.environ.get("BASE_URL", "http://localhost:8000/v1"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "abc-123")
LLM_MODEL = os.environ.get("VLLM_MODEL", os.environ.get("LLM_MODEL", "Qwen3-30B-A3B"))

# MI300 notebook: vLLM executive summaries for log-analysis-agent (errors stay tool-only)
USE_VLLM = True

os.environ["BASE_URL"] = BASE_URL
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LLM_MODEL"] = LLM_MODEL

if USE_VLLM:
    os.environ["USE_VLLM"] = "1"
    os.environ["USE_VLLM_SUMMARY"] = "1"
    os.environ["LOG_SUMMARY_BACKEND"] = "vllm"
    os.environ.setdefault("VLLM_BASE_URL", BASE_URL)
    os.environ.setdefault("VLLM_MODEL", LLM_MODEL)

print("AGENTS_DIR   =", AGENTS_DIR)
print("THEROCK_ROOT =", THEROCK_ROOT)
print("USE_VLLM     =", USE_VLLM)
print("BASE_URL     =", BASE_URL)
print("LLM_MODEL    =", LLM_MODEL)
print("DEMO_PRS     =", DEMO_PRS)
print("DEMO_PR      =", DEFAULT_DEMO_PR)
print("DEMO_RUN_ID  =", DEFAULT_DEMO_RUN_ID)
print("DEMO_JOB_ID  =", DEFAULT_DEMO_JOB_ID)
print("DEFAULT_LOG  =", DEFAULT_DEMO_LOG)
print("RUN URL      =", DEFAULT_DEMO_LOG_URL)
"""
        ),
        code(
            """import httpx

VLLM_REACHABLE = False
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
    print("Start vLLM in Step 1. Step 5 still works; Step 6 needs vLLM for executive summary.")
"""
        ),
        code(
            """import sys
!{sys.executable} -m pip install -q -r "{AGENTS_DIR / 'requirements-notebook.txt'}"
"""
        ),
        md(
            """## Step 4: Direct combined smoke test *(commented out)*

Skipped in the MI300 / vLLM workflow — run **Step 5** (change-impact) and **Step 6** (log-analysis) instead.
"""
        ),
        code(
            """# # Step 4 — combined direct smoke test (offline / no vLLM orchestrator)
# from multi_agent_tools import (
#     list_demo_assets,
#     run_change_impact_for_pr,
#     run_log_analysis_for_path,
#     run_infrastructure_triage_loop,
# )
#
# print(list_demo_assets())
# print(run_change_impact_for_pr(DEFAULT_DEMO_PR))
# print(run_log_analysis_for_path(str(DEFAULT_DEMO_LOG), use_vllm_summary=USE_VLLM))
# print(run_infrastructure_triage_loop(DEFAULT_DEMO_PR, str(DEFAULT_DEMO_LOG), use_vllm_summary=USE_VLLM))
"""
        ),
        md(
            """## Step 5: change-impact-agent (pre-merge)

Loads committed sample `sample-runs/pr-5572/report.json` — blast radius, CI labels, rollout strategy.

**Output:** `agents/notebook/out/pr-5572/report.json`
"""
        ),
        code(
            """from multi_agent_tools import run_change_impact_for_pr

change_impact_summary = run_change_impact_for_pr(DEFAULT_DEMO_PR, use_sample=True)
print(change_impact_summary)
"""
        ),
        md(
            """## Step 6: log-analysis-agent (post-CI)

Analyzes bundled log for GHA run **27710372755** / job **81992436725** (rocSPARSE `hipErrorOutOfMemory`).

- **Errors:** deterministic tool-only pass (grep + KB)
- **Executive summary:** vLLM when `USE_VLLM=True` (Step 2)

**Output:** `agents/notebook/out/log-job-81992436725/report.json` + `executive_summary.md`
"""
        ),
        code(
            """from multi_agent_tools import run_log_analysis_for_path

if not DEFAULT_DEMO_LOG.is_file():
    raise FileNotFoundError(
        f"Demo log missing: {DEFAULT_DEMO_LOG}\\n"
        f"git pull and ensure sample-runs/run-{DEFAULT_DEMO_RUN_ID}/ is present."
    )

log_analysis_summary = run_log_analysis_for_path(
    str(DEFAULT_DEMO_LOG),
    preset="therock_multi_arch",
    use_vllm_summary=USE_VLLM,
)
print(log_analysis_summary)
"""
        ),
        md(
            """## Outputs

| Step | Artifact |
|------|----------|
| 5 | `agents/notebook/out/pr-<N>/report.json` |
| 6 | `agents/notebook/out/log-job-81992436725/report.json` |
| 6 | `agents/notebook/out/log-job-81992436725/executive_summary.md` |

Bundled samples also live under `change-impact-agent/sample-runs/` and `log-analysis-agent/sample-runs/run-27710372755/`.
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
