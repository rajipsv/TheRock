# Change Impact Agent (AGENTS_030)

Autonomous **change briefing** for [TheRock](https://github.com/ROCm/TheRock): manifest diff + `BUILD_TOPOLOGY.toml` → blast radius, rollout guidance, and **recommended** CI labels.

Does **not** build ROCm or trigger CI automatically.

**Scope:** Recommends `test_matrix` jobs (e.g. `test:miopen`, `test:hipdnn`) and `test_filter:*` depth (`quick`, `standard`, `full`). Does **not** enumerate individual ctest / UT cases.

See [SCOPE.md](SCOPE.md) for final in-scope / out-of-scope boundaries (ARVIL and cross-repo pattern scanning are **out of scope** for this branch).

See [SCOPE.md](SCOPE.md) for final in-scope / out-of-scope boundaries (ARVIL and cross-repo pattern scanning are **out of scope** for this branch).

## Secrets (`.env`)

```powershell
copy agents\change-impact-agent\.env.example agents\change-impact-agent\.env
# Edit .env and paste GITHUB_TOKEN=ghp_... (file is gitignored)
```

`analyze.py`, `upstream_pr_scan.py`, and `summarize.py` load `agents/change-impact-agent/.env` automatically. Shell `GITHUB_TOKEN` wins if already set.

**Never commit `.env` or paste tokens in chat.**

## Quick start (local)

```powershell
cd TheRock
pip install -r agents/change-impact-agent/requirements.txt

# Analyze a real upstream PR (--pr fetches pull/N/head automatically)
python agents/change-impact-agent/analyze.py --pr 5688 --output-dir agents/change-impact-agent/out/pr-5688
python agents/change-impact-agent/summarize.py --backend template --input agents/change-impact-agent/out/pr-5688/report.json

# List open upstream PRs (fork PR list is separate — see below)
python agents/change-impact-agent/upstream_pr_scan.py --max 10
python agents/change-impact-agent/upstream_pr_scan.py --analyze --max 3

# Submodule bump demo (needs GITHUB_TOKEN for component file lists)
$env:GITHUB_TOKEN = "<PAT>"
python agents/change-impact-agent/analyze.py --start main~6 --end main --full-manifest
```

Open `agents/change-impact-agent/out/report.html`.

## What the agent analyzes

| Layer | Source | Example output |
|-------|--------|----------------|
| Submodule pins | Manifest diff | `rocm-libraries` SHA bump |
| Superrepo components | `--full-manifest` + GitHub API | `miopen`, `hipblaslt` changed |
| Superrepo file paths | GitHub compare on submodule SHAs | `projects/miopen/src/...` sample paths |
| TheRock file paths | `git diff --name-only` | CI scripts, artifact TOML |
| File content | Parsers for test matrix + artifact TOML | `hipdnn_python_bindings` disabled |

## CLI

### analyze.py

| Flag | Description |
|------|-------------|
| `--end` | End ref (required unless `--pr`) |
| `--start` | Start ref |
| `--pr` | Upstream PR number — auto-fetch `pull/N/head` to `pr-N` |
| `--upstream-repo` | Default `ROCm/TheRock` |
| `--pr-base-ref` | Use merge-base with branch as start (default: PR base from API) |
| `--refetch` | Force git fetch for `--pr` |
| `--output-dir` | Default: `agents/change-impact-agent/out` |
| `--full-manifest` | GitHub API superrepo drill-down (`GITHUB_TOKEN`) |

### upstream_pr_scan.py

| Flag | Description |
|------|-------------|
| `--max` | Max open upstream PRs to list or analyze (default 10) |
| `--pr` | Single PR number |
| `--analyze` | Run `analyze.py` for each listed PR |
| `--upstream-repo` | Default `ROCm/TheRock` |
| `--refetch` | Force git fetch per PR |

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

## Upstream PRs vs fork PR list

Open PRs on **ROCm/TheRock** live on upstream. Your fork’s GitHub **Pull requests** tab only shows PRs **on the fork**, so it will look empty unless you open a PR from your branch.

Use `upstream_pr_scan.py` or fork Actions **Change Impact Upstream PR Scan** to analyze upstream PRs without opening a fork PR.

### GitHub Actions (fork)

| Workflow | Purpose |
|----------|---------|
| `change-impact-upstream-scan.yml` | Dispatch: analyze one upstream PR or batch open PRs; uploads `out/**` artifact |
| `change-impact-agent.yml` | Comment on PRs opened **on your fork** with severity + label suggestions |

## Hackathon demo notebook

```powershell
jupyter notebook agents/change-impact-agent/notebook/hackathon_demo.ipynb
```

Clones `rajipsv/TheRock` (`feature/change-impact-agent`), runs `pytest`, analyzes demo PRs #5572, #5688, #5480, #5718.

## Tests

```powershell
python -m pytest agents/change-impact-agent/tests/ -q
```

## Fork

https://github.com/rajipsv/TheRock (branch: `feature/change-impact-agent`)
