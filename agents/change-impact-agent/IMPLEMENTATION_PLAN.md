# AGENTS_030 Implementation Plan

See [README.md](README.md) for usage. This document summarizes scope, inputs, and dependencies.

## What we built

A **change briefing agent** on TheRock:

- Manifest diff (what changed)
- `BUILD_TOPOLOGY.toml` (blast radius, build stages)
- CI label **recommendations** (not auto-CI)
- Executive summary (`template` | `ollama` | `openai` | `vllm`)

## Inputs

| Input | Required |
|-------|----------|
| `--end` ref | Yes |
| `--start` or `--pr-base-ref` | One required |
| `GITHUB_TOKEN` | Optional (GitHub API for superrepo components) |
| `OPENAI_API_KEY` | Optional (openai backend) |

## Dependencies

- Python 3.10+, Git
- `pip install -r agents/change-impact-agent/requirements.txt`
- **Not required:** full ROCm build, `fetch_sources.py`, GPU (except vLLM summary)

## Hackathon (MI300)

- Clone fork, run `analyze.py` (CPU)
- Run `summarize.py --backend vllm` on pre-installed vLLM + ROCm 7
- See [notebook/demo.ipynb](notebook/demo.ipynb)

## Differentiation

TheRock CI already runs tests after build with rules + labels. This agent **recommends** labels and explains impact **before** merge — it does not replace `multi_arch_ci`.

## Files

| File | Role |
|------|------|
| `analyze.py` | Main CLI |
| `manifest_bridge.py` | TheRock manifest diff |
| `impact_graph.py` | Topology traversal |
| `ci_mapping.py` | PR label suggestions |
| `summarize.py` | LLM / template summary |

Fork: https://github.com/rajipsv/TheRock branch `feature/change-impact-agent`
