$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
  throw "WSL is required for SGLang, vLLM, and KTransformers runtime profiles."
}

wsl.exe -d Ubuntu -- bash -lc "command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null"
wsl.exe -d Ubuntu -- bash -lc "command -v curl >/dev/null || (apt-get update && apt-get install -y curl)"
wsl.exe -d Ubuntu -- bash -lc "command -v /root/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh"
$wslRoot = (wsl.exe -d Ubuntu wslpath -a (Get-Location).Path).Trim()
wsl.exe -d Ubuntu -- bash -lc "cd '$wslRoot' && /root/.local/bin/uv venv .venv-linux --python 3.12"
wsl.exe -d Ubuntu -- bash -lc "cd '$wslRoot' && /root/.local/bin/uv venv .venv-linux-vllm --python 3.12"
wsl.exe -d Ubuntu -- bash -lc "cd '$wslRoot' && source .venv-linux/bin/activate && /root/.local/bin/uv pip install --prerelease=allow -r backend/requirements-sglang-linux.txt"
wsl.exe -d Ubuntu -- bash -lc "cd '$wslRoot' && source .venv-linux-vllm/bin/activate && /root/.local/bin/uv pip install --prerelease=allow -r backend/requirements-vllm-linux.txt"
wsl.exe -d Ubuntu -- bash -lc "cd '$wslRoot' && source .venv-linux/bin/activate && /root/.local/bin/uv pip install --prerelease=allow -r backend/requirements-ktransformers-linux.txt"

if (-not (Get-Command llama-server -ErrorAction SilentlyContinue)) {
  winget install --id ggml.llamacpp -e --accept-package-agreements --accept-source-agreements
}

.\.venv\Scripts\ely-eye.exe runtime-profiles
