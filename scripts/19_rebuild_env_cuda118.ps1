$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Write-Host "[info] project_root=$projectRoot"

if (Test-Path ".venv") {
    Write-Host "[step] removing .venv"
    Remove-Item -LiteralPath ".venv" -Recurse -Force
}

if (Test-Path "uv.lock") {
    Write-Host "[step] removing uv.lock"
    Remove-Item -LiteralPath "uv.lock" -Force
}

Write-Host "[step] creating venv"
uv venv .venv --python 3.11

Write-Host "[step] syncing dependencies with CUDA 11.8 torch source"
uv sync --extra nlp-providers

Write-Host "[step] exporting fully pinned requirements"
uv export --extra nlp-providers --format requirements-txt -o requirements-cu118.txt

Write-Host "[step] downloading provider assets"
.\.venv\Scripts\python.exe .\scripts\17_download_provider_assets.py

Write-Host "[step] verifying runtime environment"
.\.venv\Scripts\python.exe .\scripts\18_verify_cuda118_env.py

Write-Host "[done] CUDA 11.8 environment rebuilt."
