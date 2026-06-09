# Change Impact Agent (AGENTS_030)

A **pre-merge change briefing** tool for [TheRock](https://github.com/ROCm/TheRock). Given a git range or an upstream pull request, it answers:

- **What changed?** — submodule pins, superrepo components, CI scripts, packaging manifests
- **What breaks downstream?** — blast radius from `BUILD_TOPOLOGY.toml` (stages, severity score)
- **What should CI run?** — suggested `test:*` labels and `test_filter:*` depth (`quick`, `standard`, `full`)

It produces `report.json`, `report.html`, and an optional executive summary. It **recommends** labels for humans to apply; it does not build ROCm, trigger CI, or apply labels automatically.

Recommendations target **test_matrix jobs** (e.g. `test:miopen`, `test:hipdnn`), not individual ctest cases.

## How it works

1. **Manifest diff** — compares submodule / superrepo SHAs between two refs (local git or GitHub API with `--full-manifest`).
2. **Path diff** — lists changed files in the TheRock repo (`git diff --name-only`).
3. **Content diff** — parses CI test-matrix scripts and artifact TOML for timeout changes, disabled jobs, and packaging edits.
4. **Superrepo drill-down** — on SHA bumps (e.g. `rocm-libraries`), uses GitHub compare and per-directory commit detection to name changed components (`miopen`, `hipblaslt`, …).
5. **Topology impact** — maps changes onto `BUILD_TOPOLOGY.toml` for affected build stages and a severity score.
6. **CI mapping** — turns the above into `test:*` suite labels and `test_filter:*` depth plus rollout guidance text.

## Features

- Analyze any upstream `ROCm/TheRock` PR with `--pr N` (auto-fetches `pull/N/head`)
- Works from fork clones (auto-fetches `upstream-main` when local `main` is missing)
- Component-scoped `test:*` labels for superrepo bumps (file compare + commit-by-directory)
- List or batch-analyze open upstream PRs via `upstream_pr_scan.py`
- Template or LLM summaries (`summarize.py`: Ollama, OpenAI, vLLM)
- Pre-generated demo reports in [`sample-runs/`](sample-runs/) for reviewers
- 15 unit tests; hackathon walkthrough in `notebook/hackathon_demo.ipynb`

## Secrets (`.env`)

```powershell
copy agents\change-impact-agent\.env.example agents\change-impact-agent\.env
# Edit .env: GITHUB_TOKEN=<your PAT>  (file is gitignored)
```

`analyze.py`, `upstream_pr_scan.py`, and `summarize.py` load `.env` via `env_loader.py`. A shell `GITHUB_TOKEN` already in the environment wins.

**Never commit `.env`, hardcode tokens in notebooks, or paste tokens in chat.**

`GITHUB_TOKEN` is needed for:

- Listing upstream PRs (anonymous API rate limits)
- Full superrepo component drill-down (`--full-manifest`)
- Reliable GitHub compare / commit-by-directory on `rocm-libraries`

## Quick start

```powershell
cd TheRock
pip install -r agents/change-impact-agent/requirements.txt

# Analyze an upstream PR
python agents/change-impact-agent/analyze.py --pr 5688 --output-dir agents/change-impact-agent/out/pr-5688
python agents/change-impact-agent/summarize.py --backend template --input agents/change-impact-agent/out/pr-5688/report.json

# Superrepo bump with full component lists
python agents/change-impact-agent/analyze.py --pr 5718 --full-manifest --output-dir agents/change-impact-agent/out/pr-5718

# List or batch-analyze open upstream PRs
python agents/change-impact-agent/upstream_pr_scan.py --max 10
python agents/change-impact-agent/upstream_pr_scan.py --analyze --max 3 --full-manifest

# Local git range (fork clones: use HEAD, not main)
python agents/change-impact-agent/analyze.py --start HEAD~6 --end HEAD --output-dir agents/change-impact-agent/out/demo
```

Open `out/report.html`, or browse committed samples under `sample-runs/`.

**PowerShell:** `run_demo.ps1` runs `HEAD~6..HEAD` on the current branch.

## Sample runs

Pre-generated reports in [`sample-runs/`](sample-runs/):

| Folder | PR | What it shows |
|--------|-----|----------------|
| `pr-5572/` | [ROCm/TheRock#5572](https://github.com/ROCm/TheRock/pull/5572) | CI timeout change → `test:miopen`, `test_filter:quick` |
| `pr-5688/` | [#5688](https://github.com/ROCm/TheRock/pull/5688) | hipDNN CI + artifact TOML → `test:hipdnn`, `test_filter:quick` |
| `pr-5480/` | [#5480](https://github.com/ROCm/TheRock/pull/5480) | Third-party packaging → `test_filter:quick` |
| `pr-5718/` | [#5718](https://github.com/ROCm/TheRock/pull/5718) | rocm-libraries bump → component-scoped `test:*` |

Each folder has `report.json`, `report.html`, and `executive_summary.md`. Local runs write to `out/` (gitignored).

## Analysis layers

| Layer | Source | Example |
|-------|--------|---------|
| Submodule pins | Manifest diff | `rocm-libraries` SHA bump |
| Superrepo components | GitHub compare + commits | `miopen`, `hipblaslt` |
| Superrepo paths | Compare on old/new SHAs | `projects/miopen/...` |
| TheRock paths | `git diff --name-only` | CI scripts, third-party CMake |
| File content | Matrix + TOML parsers | timeout 60→120 min |
| Topology | `BUILD_TOPOLOGY.toml` | stages, severity |

## Project layout

```
agents/change-impact-agent/
├── analyze.py               # Main entry — build report from git range or --pr
├── summarize.py             # Executive summary (template or LLM)
├── upstream_pr_scan.py      # List / analyze open upstream PRs
├── manifest_bridge.py       # Submodule pin diff
├── path_bridge.py           # TheRock file path diff
├── content_diff.py          # CI matrix + artifact TOML
├── component_diff_bridge.py # Superrepo compare + commit-by-directory
├── impact_graph.py          # Topology blast radius
├── ci_mapping.py            # test:* + test_filter:* labels
├── github_pr.py             # Upstream PR fetch + GitHub API
├── sample-runs/              # Committed demo outputs
├── notebook/hackathon_demo.ipynb
└── tests/
```

## CLI reference

### analyze.py

| Flag | Description |
|------|-------------|
| `--end` | End ref (required unless `--pr`) |
| `--start` | Start ref |
| `--pr` | Upstream PR — fetch `pull/N/head` to `pr-N` |
| `--upstream-repo` | Default `ROCm/TheRock` |
| `--pr-base-ref` | Merge-base start; fetches upstream branch if missing locally |
| `--refetch` | Force git fetch for `--pr` |
| `--output-dir` | Default `agents/change-impact-agent/out` |
| `--full-manifest` | GitHub superrepo drill-down |
| `--therock-root` | TheRock repo root path |

### upstream_pr_scan.py

| Flag | Description |
|------|-------------|
| `--max` | Max open PRs to list or analyze (default 10) |
| `--pr` | Single PR number |
| `--analyze` | Run `analyze.py` for each PR |
| `--full-manifest` | Pass through to `analyze.py` |
| `--refetch` | Force git fetch per PR |

### summarize.py

| Flag | Description |
|------|-------------|
| `--backend` | `template`, `ollama`, `openai`, `vllm` |
| `--input` / `--output` | `report.json` → `executive_summary.md` |
| `--model` | LLM model name |
| `--base-url` | Ollama or vLLM URL |

## Outputs

| File | Contents |
|------|----------|
| `report.json` | Impact data, `ci_recommendations`, `content_insights`, `rollout_strategy` |
| `report.html` | HTML report |
| `executive_summary.md` | Human-readable summary from `summarize.py` |

## Upstream vs fork PRs

Open PRs on **ROCm/TheRock** are upstream. Your fork's Pull requests tab only shows PRs on the fork. Use `upstream_pr_scan.py` or the **Change Impact Upstream PR Scan** GitHub Action to analyze upstream PRs.

| Workflow | Purpose |
|----------|---------|
| `change-impact-upstream-scan.yml` | Dispatch: analyze upstream PR(s); upload reports |
| `change-impact-agent.yml` | Comment on fork PRs with severity + label suggestions |

## Demo notebook

```powershell
jupyter notebook agents/change-impact-agent/notebook/hackathon_demo.ipynb
```

Clones the fork, runs `pytest`, analyzes PRs #5572, #5688, #5480, #5718. Copy `.env` into the clone if `GITHUB_TOKEN` is not in the environment. Shallow clones lack `main` — the notebook fetches `upstream-main` and uses `HEAD~6..HEAD` for the local range demo.

## Tests

```powershell
python -m pytest agents/change-impact-agent/tests/ -q
```

## Fork

https://github.com/rajipsv/TheRock (`feature/change-impact-agent`) — sample reports at `agents/change-impact-agent/sample-runs/`
