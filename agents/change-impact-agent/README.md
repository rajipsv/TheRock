# Change Impact Agent (AGENTS_030)

Autonomous **change briefing** for [TheRock](https://github.com/ROCm/TheRock): manifest diff + `BUILD_TOPOLOGY.toml` → blast radius, rollout guidance, and **recommended** CI labels.

Does **not** build ROCm or trigger CI automatically.

## Quick start (local)

```powershell
cd TheRock
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r agents/change-impact-agent/requirements.txt

python agents/change-impact-agent/analyze.py --start main~30 --end main
python agents/change-impact-agent/summarize.py --backend template
```

Open `agents/change-impact-agent/out/report.html`.

## Hackathon notebook (MI300 + vLLM)

```bash
git clone --depth 100 https://github.com/rajipsv/TheRock.git
cd TheRock
git checkout feature/change-impact-agent
pip install -r agents/change-impact-agent/requirements.txt
python agents/change-impact-agent/analyze.py --start main~20 --end main
python agents/change-impact-agent/summarize.py --backend vllm --base-url http://localhost:8000/v1 --model <model>
```

## CLI

### analyze.py

| Flag | Description |
|------|-------------|
| `--end` | End ref (required) |
| `--start` | Start ref |
| `--pr-base-ref` | Use merge-base with branch as start |
| `--output-dir` | Default: `agents/change-impact-agent/out` |

### summarize.py

| Flag | Description |
|------|-------------|
| `--backend` | `template`, `ollama`, `openai`, `vllm` |
| `--input` | `out/report.json` |
| `--model` | LLM model name |
| `--base-url` | Ollama or vLLM OpenAI-compatible URL |

## Outputs

- `out/report.json` — structured impact data
- `out/report.html` — visual report
- `out/executive_summary.md` — after summarize

## Tests

```powershell
python -m unittest agents.change-impact-agent.tests.test_impact_graph
```

Or from agent directory:

```powershell
cd agents/change-impact-agent
python -m unittest tests.test_impact_graph
```

## Fork

https://github.com/rajipsv/TheRock (branch: `feature/change-impact-agent`)
