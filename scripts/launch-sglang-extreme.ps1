$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
  throw "WSL is required for the SGLang Linux runtime."
}

$override = '{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'
$wslRoot = (wsl.exe -d Ubuntu wslpath -a (Get-Location).Path).Trim()
$command = @"
set -euo pipefail
cd "$wslRoot"
source .venv-linux/bin/activate
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-9B \
  --port 8000 \
  --tp-size 1 \
  --mem-fraction-static 0.88 \
  --context-length 1010000 \
  --json-model-override-args '$override'
"@

wsl.exe -d Ubuntu -- bash -lc $command
