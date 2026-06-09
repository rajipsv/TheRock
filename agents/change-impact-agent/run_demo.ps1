# Quick demo script for AGENTS_030 Change Impact Agent
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
Set-Location $RepoRoot

$AgentDir = Join-Path $RepoRoot "agents\change-impact-agent"
$OutDir = Join-Path $AgentDir "out"

Write-Host "=== Change Impact Agent Demo ===" -ForegroundColor Cyan
# Shallow fork clones may lack main — use HEAD on the checked-out branch
python "$AgentDir\analyze.py" --start HEAD~6 --end HEAD --output-dir $OutDir
python "$AgentDir\summarize.py" --backend template --input (Join-Path $OutDir "report.json")
Write-Host ""
Write-Host "Open: $OutDir\report.html" -ForegroundColor Green
