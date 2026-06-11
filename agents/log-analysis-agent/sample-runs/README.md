# Sample log analysis runs

| Run | Input | Mode |
|-----|-------|------|
| [log-example/](log-example/) | `example.log` (synthetic app/CI log) | tool-only |
| [run-27270735424/](run-27270735424/) | [Multi-Arch CI #7812](https://github.com/ROCm/TheRock/actions/runs/27270735424) — math-libs job `80542543540`, exit 130 runner shutdown | tool-only |

### Regenerate log-example

```powershell
python agents/log-analysis-agent/analyze_log.py `
  --log agents/log-analysis-agent/tests/fixtures/example.log `
  --output-dir agents/log-analysis-agent/sample-runs/log-example
```

### Regenerate run-27270735424

With a PAT (`Actions: read` on public repos):

```powershell
python agents/log-analysis-agent/analyze_log.py `
  --github-run-id 27270735424 `
  --github-job-id 80542543540 `
  --repo ROCm/TheRock `
  --preset therock_multi_arch `
  --output-dir agents/log-analysis-agent/sample-runs/run-27270735424
```

Without PAT (fixture from public annotations):

```powershell
python agents/log-analysis-agent/analyze_log.py `
  --log agents/log-analysis-agent/tests/fixtures/run-27270735424-math-libs.log `
  --preset therock_multi_arch `
  --output-dir agents/log-analysis-agent/sample-runs/run-27270735424
```
