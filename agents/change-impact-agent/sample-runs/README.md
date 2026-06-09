# Sample change-impact reports

Pre-generated outputs from analyzing upstream `ROCm/TheRock` pull requests (hackathon demo set).

| Folder | PR | Description |
|--------|-----|-------------|
| `pr-5572/` | [#5572](https://github.com/ROCm/TheRock/pull/5572) | MIOpen GHA timeout 60→120 min |
| `pr-5688/` | [#5688](https://github.com/ROCm/TheRock/pull/5688) | hipDNN CI + artifact TOML |
| `pr-5480/` | [#5480](https://github.com/ROCm/TheRock/pull/5480) | OpenMPI CMake version file naming |
| `pr-5718/` | [#5718](https://github.com/ROCm/TheRock/pull/5718) | rocm-libraries superrepo bump |

Each folder contains:

- `report.json` — structured impact + CI recommendations
- `report.html` — HTML report (open in browser)
- `executive_summary.md` — template executive summary

Regenerate locally:

```bash
python agents/change-impact-agent/analyze.py --pr 5572 --output-dir agents/change-impact-agent/out/pr-5572
python agents/change-impact-agent/summarize.py --backend template \
  --input agents/change-impact-agent/out/pr-5572/report.json \
  --output agents/change-impact-agent/out/pr-5572/executive_summary.md
```

Local runs write to `out/` (gitignored). Copy new samples here when updating the demo set.
