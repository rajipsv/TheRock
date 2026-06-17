# Sample log analysis runs

| Run | Input | Mode |
|-----|-------|------|
| **[run-27697860238/](run-27697860238/)** | [Multi-Arch CI #8864 — Adding kfdtest](https://github.com/ROCm/TheRock/actions/runs/27697860238) — compiler-runtime job `81925995968`, exit 1 | tool-only |
| [run-27270735424/](run-27270735424/) | [Multi-Arch CI #7812](https://github.com/ROCm/TheRock/actions/runs/27270735424) — math-libs job `80542543540`, exit 130 runner shutdown | tool-only |
| [log-example/](log-example/) | `example.log` (synthetic app/CI log — legacy quick demo) | tool-only |

### Regenerate run-27697860238 (primary demo)

Fixture from saved run page annotations (no PAT required):

```powershell
python agents/log-analysis-agent/scripts/build_fixture_from_run_page.py
python agents/log-analysis-agent/analyze_log.py `
  --log agents/log-analysis-agent/tests/fixtures/run-27697860238-compiler-runtime.log `
  --preset therock_multi_arch `
  --output-dir agents/log-analysis-agent/sample-runs/run-27697860238
python agents/log-analysis-agent/scripts/patch_run_27697860238_sample.py
```

With a PAT (`Actions: read`), fetch full job log:

```powershell
python agents/log-analysis-agent/analyze_log.py `
  --github-run-id 27697860238 `
  --github-job-id 81925995968 `
  --repo ROCm/TheRock `
  --preset therock_multi_arch `
  --output-dir agents/log-analysis-agent/sample-runs/run-27697860238
```

Kfdtest test job (warnings only, job succeeded):

```powershell
python agents/log-analysis-agent/scripts/build_fixture_from_run_page.py `
  --job-id 81974468002 `
  --job-name "Linux::release / Test gfx94X-dcgpu / Test kfdtest / Test kfdtest (shard 1/1) (gfx94X-dcgpu)" `
  --fixture-name run-27697860238-kfdtest.log
```

### Regenerate log-example (legacy)

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
