$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
  throw "WSL is required for the vLLM Linux runtime."
}

$wslRoot = (wsl.exe -d Ubuntu wslpath -a (Get-Location).Path).Trim()
$ropeScaling = '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}'
$command = @"
set -euo pipefail
cd "$wslRoot"
source .venv-linux-vllm/bin/activate
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-9B \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 1010000 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.88 \
  --trust-remote-code \
  --rope-scaling '$ropeScaling'
"@

wsl.exe -d Ubuntu -- bash -lc $command
