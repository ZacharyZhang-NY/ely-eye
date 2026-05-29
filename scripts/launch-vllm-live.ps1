$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
  throw "WSL is required for the vLLM Linux runtime."
}

$wslRoot = (wsl.exe -d Ubuntu wslpath -a (Get-Location).Path).Trim()
$command = @"
set -euo pipefail
cd "$wslRoot"
source .venv-linux-vllm/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-9B \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 262144 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.82 \
  --trust-remote-code
"@

wsl.exe -d Ubuntu -- bash -lc $command
