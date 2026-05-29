$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
  throw "WSL is required for the SGLang Linux runtime."
}

$wslRoot = (wsl.exe -d Ubuntu wslpath -a (Get-Location).Path).Trim()
$command = @"
set -euo pipefail
cd "$wslRoot"
source .venv-linux/bin/activate
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-9B \
  --port 8000 \
  --tp-size 1 \
  --mem-fraction-static 0.82 \
  --context-length 262144 \
  --reasoning-parser qwen3 \
  --speculative-algo NEXTN \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4
"@

wsl.exe -d Ubuntu -- bash -lc $command
