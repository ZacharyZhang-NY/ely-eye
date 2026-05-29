$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path ".venv")) {
  python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e .\backend
.\.venv\Scripts\python.exe -m pip install --upgrade --no-cache-dir unsloth unsloth_zoo
.\.venv\Scripts\python.exe -m pip install --upgrade --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 torch==2.12.0+cu130 torchvision==0.27.0+cu130
.\.venv\Scripts\python.exe -m pip install --upgrade --no-cache-dir transformers==5.9.0 datasets==4.8.5 dill==0.4.1 multiprocess==0.70.19 fsspec==2026.2.0 safetensors==0.8.0rc0

Push-Location .\frontend
npm install
Pop-Location

Copy-Item .env.example .env -ErrorAction SilentlyContinue

.\.venv\Scripts\ely-eye.exe status
