# AGENTS_030 — Implementation status

**Status:** Feature complete for hackathon / fork PR scope. See [SCOPE.md](SCOPE.md) for boundaries.

## Implemented

- [x] `analyze.py` — manifest, topology, CI mapping, content parsers, `--pr`, `--full-manifest`
- [x] `manifest_bridge.py`, `path_bridge.py`, `content_diff.py`, `component_diff_bridge.py`
- [x] `impact_graph.py`, `ci_mapping.py` — component-scoped `test:*` labels (not whole math-libs default)
- [x] `summarize.py` — template / ollama / openai / vllm
- [x] `upstream_pr_scan.py`, `github_pr.py`
- [x] `.github/workflows/change-impact-upstream-scan.yml`
- [x] `.github/workflows/change-impact-agent.yml`
- [x] Tests (14 passing)
- [x] README, demo notebook, `run_demo.ps1`

## Deferred / out of scope

- [ ] ARVIL dashboard integration
- [ ] Cross-repo “similar bug” pattern scanner
- [ ] Comments on upstream `ROCm/TheRock` PRs (fork only today)
- [ ] Per-ctest label mapping

## Quick commands

```powershell
pip install -r agents/change-impact-agent/requirements.txt
python agents/change-impact-agent/analyze.py --pr 5688 --output-dir agents/change-impact-agent/out/pr-5688
python agents/change-impact-agent/summarize.py --backend template --input agents/change-impact-agent/out/pr-5688/report.json
python -m pytest agents/change-impact-agent/tests/ -q
```

Fork: https://github.com/rajipsv/TheRock branch `feature/change-impact-agent`
