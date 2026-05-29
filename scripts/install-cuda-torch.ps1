$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
.\.venv\Scripts\python.exe -m pip install --upgrade --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 torch==2.12.0+cu130 torchvision==0.27.0+cu130
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
