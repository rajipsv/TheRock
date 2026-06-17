# Earnings IR Agent

Autonomous **earnings call script & presentation** workflow for Investor Relations demos.

Originally built for the AMD TCS Hackathon; lives under TheRock `agents/` alongside change-impact, log-analysis, and document-comparison.

## Agents

| Agent | Role |
|-------|------|
| **Data extraction** | Demo quarterly financials (CSV) + earnings transcripts (Hugging Face or bundled fallback) |
| **Predictive analyst** | Predicts difficult institutional investor questions from history + metrics |
| **Drafting** | Earnings call script, presentation bullets, Q&A cheat sheet for CEO/CFO |

Pipeline: `extract → predict → draft` in `earnings_ir/pipeline.py`.

## Setup

```powershell
cd TheRock
pip install -r agents/earnings-ir-agent/requirements-notebook.txt
copy agents\earnings-ir-agent\.env.example agents\earnings-ir-agent\.env
```

## vLLM (MI300)

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

Set `USE_LLM=false` for rule-based fallback without vLLM.

## CLI

```powershell
python agents/earnings-ir-agent/scripts/run_demo.py --ticker AMD --pretty -o agents/earnings-ir-agent/out/result.json
```

## Notebook

`notebook/earnings_ir_demo.ipynb` — vLLM verify, Pydantic AI tools, full pipeline.

First Hugging Face load streams transcripts for **AMD** and caches to `data/cache/` (gitignored).

## Data

| Source | Purpose |
|--------|---------|
| [Rogersurf/earnings-call-transcripts](https://huggingface.co/datasets/Rogersurf/earnings-call-transcripts) | Earnings call text (research/education) |
| `data/sample_financials.csv` | Synthetic demo metrics (AMD, NVDA) |
| `data/fallback_amd_transcripts.json` | Offline fallback when HF unavailable |

**Not included:** SEC EDGAR ingestion, live market sentiment, official IR slide decks.

## Layout

```
agents/earnings-ir-agent/
  earnings_ir/       # pipeline, agents, dataset, LLM
  scripts/run_demo.py
  data/
  notebook/
  out/               # CLI JSON output (gitignored)
```

## Disclaimer

Research and education demo only. Do not distribute generated materials to investors. Verify all figures against official SEC filings and company releases.
