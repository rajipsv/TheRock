# AGENTS_030 — Final scope

## In scope (implemented)

**Change Impact Agent** on TheRock fork `feature/change-impact-agent`:

| Capability | Description |
|------------|-------------|
| Manifest diff | Local git pin diff; optional `--full-manifest` (GitHub API drill-down) |
| Path + content diff | TheRock file paths; `fetch_test_configurations.py` timeouts/disabled jobs; artifact TOML |
| Superrepo component scope | GitHub compare + per-directory commit detection on submodule SHA bumps |
| Topology blast radius | `BUILD_TOPOLOGY.toml` → stages, severity, rollout text |
| CI label recommendations | `test:*` suites + `test_filter:*` depth (not individual ctest cases) |
| Executive summary | Template / Ollama / OpenAI / vLLM via `summarize.py` |
| Upstream PR workflow | `--pr N`, `upstream_pr_scan.py`, fork GHA workflows |
| Tests | 14 unit tests (`impact_graph`, `content_diff`, `ci_mapping`) |

### Deliverables

- CLI: `analyze.py`, `summarize.py`, `upstream_pr_scan.py`
- Reports: `report.json`, `report.html`, `executive_summary.md`
- Workflows: `change-impact-upstream-scan.yml`, `change-impact-agent.yml` (fork PR comments)
- Sample upstream PR analyses: #5572, #5688, #5718, #5480, #5629
- Demo: `run_demo.ps1`, `notebook/hackathon_demo.ipynb` (clone fork + full test run)

### Positioning

- **Before merge:** impact briefing + recommended CI labels (assistant — human applies labels)
- **Complements:** `manifest-diff.yml` CI, `assistant-librarian` (creates bump PRs; this agent **analyzes** any PR)
- **Does not replace:** `multi_arch_ci`, full ROCm build, or per-ctest mapping

## Out of scope (explicitly not in this branch)

| Item | Reason |
|------|--------|
| **ARVIL integration** | Separate Vercel/Neon app; no shared implementation in this PR |
| **Pattern / similar-bug scanner** | Future idea (PR-derived rules across component repos); deferred |
| **Full clone of all ROCm component repos** | Heavy; use sparse API/commit detection instead |
| **Auto-apply PR labels or trigger CI** | Recommendations only |
| **Per-ctest / UT name mapping** | TheRock uses `test_matrix` + `TEST_TYPE` |
| **Upstream PR comments on ROCm/TheRock** | Fork workflows only unless merged upstream |

## Requirements

- Python 3.10+, Git, TheRock repo clone
- `pip install -r agents/change-impact-agent/requirements.txt`
- `GITHUB_TOKEN` recommended for upstream PR fetch, superrepo component detection, `--full-manifest`
- GPU optional (vLLM summary on MI300 hackathon path)

## Fork

https://github.com/rajipsv/TheRock — branch `feature/change-impact-agent`
