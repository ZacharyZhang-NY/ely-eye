$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

.\.venv\Scripts\python.exe .\scripts\download_models.py --model Qwen/Qwen3.5-9B
.\.venv\Scripts\python.exe .\scripts\download_models.py --model Qwen/Qwen3.5-0.8B
.\.venv\Scripts\python.exe .\scripts\download_models.py --model Qwen/Qwen3-VL-Embedding-8B
