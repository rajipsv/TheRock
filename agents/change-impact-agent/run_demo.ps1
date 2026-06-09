# Quick demo script for AGENTS_030 Change Impact Agent
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
Set-Location $RepoRoot

$AgentDir = Join-Path $RepoRoot "agents\change-impact-agent"
$OutDir = Join-Path $AgentDir "out"

Write-Host "=== Change Impact Agent Demo ===" -ForegroundColor Cyan
python "$AgentDir\analyze.py" --start main~15 --end main --output-dir $OutDir
python "$AgentDir\summarize.py" --backend template --input (Join-Path $OutDir "report.json")
Write-Host ""
Write-Host "Open: $OutDir\report.html" -ForegroundColor Green
