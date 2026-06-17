#!/usr/bin/env python3
"""Generate agents/log-analysis-agent/notebook/log_analysis_demo.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebook" / "log_analysis_demo.ipynb"

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
            """# AGENTS_030 — Log Analysis Agent (MI300 / vLLM)

Deterministic error extraction (grep + KB) plus **vLLM executive summary** on AMD MI300.

| Step | What |
|------|------|
| 1 | Launch vLLM |
| 2 | Config — `USE_VLLM = True` (default) |
| 3 | Analyze bundled run **27710372755** (rocSPARSE OOM) |
| 4 | Optional: live GitHub fetch (`GITHUB_TOKEN` in `.env`) |

**Note:** Errors in `report.json` are always tool-only; vLLM writes `executive_summary.md` only.
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

Monitor: `watch rocm-smi`
"""
        ),
        code(
            """import os
import sys
from pathlib import Path

# --- MI300 notebook flag (set False for template-only summaries) ---
USE_VLLM = True

def resolve_agent_dir() -> Path:
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "analyze_log.py").is_file():
            return candidate
        if (candidate / "log-analysis-agent" / "analyze_log.py").is_file():
            return candidate / "log-analysis-agent"
        if (candidate / "agents" / "log-analysis-agent" / "analyze_log.py").is_file():
            return candidate / "agents" / "log-analysis-agent"
    raise RuntimeError("Start Jupyter from log-analysis-agent/ or TheRock/agents/")

AGENT_DIR = resolve_agent_dir()
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from env_loader import load_agent_env
from llm import configure_vllm_env, default_summary_backend

load_agent_env()

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen3-30B-A3B")
API_KEY = os.environ.get("OPENAI_API_KEY", "abc-123")

cfg = configure_vllm_env(
    use_vllm=USE_VLLM,
    base_url=BASE_URL,
    model=LLM_MODEL,
    api_key=API_KEY,
)

DEMO_RUN_ID = 27710372755
DEMO_JOB_ID = 81992436725
DEMO_LOG = AGENT_DIR / "sample-runs" / f"run-{DEMO_RUN_ID}" / f"job-{DEMO_JOB_ID}" / f"job-{DEMO_JOB_ID}.log"
OUT_DIR = AGENT_DIR / "notebook" / "out" / f"run-{DEMO_RUN_ID}-demo"

print("AGENT_DIR     =", AGENT_DIR)
print("USE_VLLM      =", USE_VLLM)
print("summary backend =", default_summary_backend())
print("VLLM_BASE_URL =", cfg["base_url"])
print("VLLM_MODEL    =", cfg["model"])
print("DEMO_LOG      =", DEMO_LOG)
print("OUT_DIR       =", OUT_DIR)
"""
        ),
        code(
            """import httpx

if USE_VLLM:
    try:
        r = httpx.get(
            f"{cfg['base_url']}/models",
            headers={"Authorization": f"Bearer {cfg['api_key']}"},
            timeout=15.0,
        )
        print("vLLM:", "OK" if r.status_code < 400 else r.text[:200])
    except Exception as exc:
        print("vLLM not reachable:", exc)
        print("Summaries will fall back to template if LLM call fails.")
else:
    print("USE_VLLM=False — template summaries only.")
"""
        ),
        code(
            """import sys
!{sys.executable} -m pip install -q -r "{AGENT_DIR / 'requirements.txt'}"
"""
        ),
        md("## Step 3: Analyze rocSPARSE failure (bundled sample)"),
        code(
            """from analyze_log import build_report, write_outputs

if not DEMO_LOG.is_file():
    raise FileNotFoundError(f"Missing demo log: {DEMO_LOG}")

report = build_report(DEMO_LOG, preset_name="therock_multi_arch", use_agent=False)
write_outputs(
    report,
    OUT_DIR,
    write_summary=True,
    summary_backend=default_summary_backend(),
)

print("Mode:", report.get("mode"))
print("Errors:", report.get("errors_count"))
print("Report:", OUT_DIR / "report.json")
print("Summary:", OUT_DIR / "executive_summary.md")
if report.get("executive_summary"):
    print()
    print(report["executive_summary"][:2500])
"""
        ),
        md(
            """## Step 4 (optional): Live GitHub — all failed jobs

Requires `GITHUB_TOKEN` in `agents/log-analysis-agent/.env`.
"""
        ),
        code(
            """# Uncomment to fetch live run (needs network + token):
# from analyze_log import analyze_github_run
# reports = analyze_github_run(
#     DEMO_RUN_ID,
#     output_dir=AGENT_DIR / "sample-runs" / f"run-{DEMO_RUN_ID}-live",
#     all_failed=True,
#     write_summary=True,
#     summary_backend=default_summary_backend(),
# )
# print(f"Analyzed {len(reports)} jobs")
"""
        ),
    ]

    NB["cells"] = cells
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(NB, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
