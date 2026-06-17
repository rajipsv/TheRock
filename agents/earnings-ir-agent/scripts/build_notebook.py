"""Generate notebooks/earnings_ir_demo.ipynb with valid JSON."""

import json
from pathlib import Path


def md(text: str) -> dict:
    parts = text.split("\n")
    src = [p + "\n" for p in parts[:-1]]
    if parts:
        src.append(parts[-1])
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(text: str) -> dict:
    parts = text.split("\n")
    src = [p + "\n" for p in parts[:-1]]
    if parts:
        src.append(parts[-1])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


cells = [
    md(
        """# Autonomous Earnings Call Script & Presentation (IR Demo)

Multi-agent workflow for **Investor Relations** using:

- **Data source:** [Rogersurf/earnings-call-transcripts](https://huggingface.co/datasets/Rogersurf/earnings-call-transcripts) (research/education)
- **LLM:** Qwen3 via vLLM on AMD GPU
- **Agents:** Extract -> Predict hard questions -> Draft script, deck bullets, Q&A cheat sheet

> **Disclaimer:** Demo only — not for investor distribution."""
    ),
    md(
        """## Step 1: Start vLLM (separate terminal)

```bash
VLLM_USE_TRITON_FLASH_ATTN=0 \\
vllm serve Qwen/Qwen3-30B-A3B \\
    --served-model-name Qwen3-30B-A3B \\
    --api-key abc-123 \\
    --port 8000 \\
    --enable-auto-tool-choice \\
    --tool-call-parser hermes \\
    --trust-remote-code
```"""
    ),
    code(
        """import os
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["BASE_URL"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "abc-123"
os.environ["LLM_MODEL"] = "Qwen3-30B-A3B"
os.environ["USE_LLM"] = "true"

TICKER = "AMD"
print("Project root:", ROOT)
print("Ticker:", TICKER)"""
    ),
    code(
        """import httpx

response = httpx.get(
    f"{os.environ['BASE_URL']}/models",
    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    timeout=30.0,
)
response.raise_for_status()
print("Models:", [m["id"] for m in response.json().get("data", [])])"""
    ),
    md("## Step 2: Install dependencies"),
    code('!pip install -q -r "{ROOT / \'requirements.txt\'}"'),
    md("## Step 3: Load earnings transcripts (HF or fallback cache)"),
    code(
        """from earnings_ir.dataset import load_transcripts

records = load_transcripts(TICKER, limit=4)
print(f"Loaded {len(records)} transcript(s)")
for r in records:
    print(f"  - {r.quarter} {r.earnings_year}: {r.title[:70]}...")"""
    ),
    md("## Step 4: Data Extraction Agent (Pydantic AI + tools)"),
    code(
        """from earnings_ir.agents import build_agent_model, build_extraction_agent

model = build_agent_model()
extractor = build_extraction_agent(model)"""
    ),
    code(
        """extraction = await extractor.run(
    f"List transcripts and show demo financials for {TICKER}"
)
print(extraction.output)"""
    ),
    md("## Step 5: Full IR pipeline (Predictive Analyst + Drafting)"),
    code(
        """from earnings_ir.pipeline import run_earnings_ir_pipeline

result = await run_earnings_ir_pipeline(TICKER)
print("LLM used:", result.llm_used)
print("Data source:", result.data_source)
print("Snippets used:", result.transcript_snippets_used)"""
    ),
    code(
        """print("=== Predicted investor questions ===")
for i, q in enumerate(result.predicted_questions, 1):
    print(f"{i}. [{q.severity}] {q.question}")
    print(f"   Why: {q.rationale}\\n")"""
    ),
    code(
        """print("=== Earnings call script (opening) ===\\n")
print(result.earnings_script)

print("\\n=== Investor presentation bullets ===")
for b in result.presentation_bullets:
    print(" •", b)

print("\\n=== Q&A cheat sheet (CEO/CFO) ===")
for item in result.qa_cheat_sheet[:5]:
    print(f"\\nQ: {item.question}")
    print(f"A: {item.suggested_answer}")
    if item.talking_points:
        print("  Talking points:", "; ".join(item.talking_points))"""
    ),
    md(
        """## Step 6: Orchestrator agent (optional one-shot)

Ask the orchestrator agent to run the full pipeline via a single tool call."""
    ),
    code(
        """from earnings_ir.agents import build_orchestrator_agent

orchestrator = build_orchestrator_agent(model)
orch = await orchestrator.run(f"Prepare earnings IR materials for {TICKER}")
print(orch.output)"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parents[1] / "notebooks" / "earnings_ir_demo.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {out}")
