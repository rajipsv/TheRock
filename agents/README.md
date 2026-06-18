# TheRock Agents

Hackathon and experimental agents built on top of TheRock (not part of upstream ROCm core).

| Agent | Path | Description |
|-------|------|-------------|
| AGENTS_030 Change Impact | [change-impact-agent/](change-impact-agent/) | Pre-merge: manifest + topology blast radius and CI label recommendations |
| AGENTS_030 Log Analysis | [log-analysis-agent/](log-analysis-agent/) | Post-CI failure: log triage, KB-backed fixes; reactive GHA on fork + upstream poll |
| AGENTS_003 Document Comparison | [document-comparison-agent/](document-comparison-agent/) | Legacy vs modernized PDF diff, semantic analysis, regulatory impact |
| Earnings IR | [earnings-ir-agent/](earnings-ir-agent/) | Earnings call script, deck bullets, Q&A cheat sheet from transcripts + demo financials |

## Multi-agent orchestrator

| Resource | Path |
|----------|------|
| Pydantic AI notebook | [notebook/multi_agent_triage_demo.ipynb](notebook/multi_agent_triage_demo.ipynb) |
| Shared tools | [multi_agent_tools.py](multi_agent_tools.py) |

Coordinates **change-impact-agent** (Step 5) + **log-analysis-agent** (Step 6) on MI300 with vLLM executive summaries for log triage.

Notebook flow: launch vLLM → config → **Step 5** re-analyze PR 5572 (manifest/topology/CI) + vLLM → **Step 6** re-analyze all 8 failed jobs from GHA run `27710372755` (tool_only + vLLM per job). Step 4 combined smoke test is commented out.

```powershell
pip install -r agents/requirements-notebook.txt
jupyter notebook agents/notebook/multi_agent_triage_demo.ipynb
```
