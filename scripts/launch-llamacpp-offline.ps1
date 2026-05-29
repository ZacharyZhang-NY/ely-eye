$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not $env:ELY_EYE_GGUF_MODEL) {
  throw "Set ELY_EYE_GGUF_MODEL to the absolute GGUF model path."
}

$server = $env:LLAMA_CPP_SERVER
if (-not $server) {
  $server = "llama-server"
}

if (-not (Get-Command $server -ErrorAction SilentlyContinue)) {
  $wingetPackageRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
  $installed = Get-ChildItem $wingetPackageRoot -Recurse -Filter llama-server.exe -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -like "*ggml.llamacpp*" } |
    Select-Object -First 1
  if ($installed) {
    $server = $installed.FullName
  }
}

if (-not (Get-Command $server -ErrorAction SilentlyContinue)) {
  throw "llama-server executable is missing. Install ggml.llamacpp or set LLAMA_CPP_SERVER."
}

& $server `
  --model $env:ELY_EYE_GGUF_MODEL `
  --host 127.0.0.1 `
  --port 8003 `
  --ctx-size 262144 `
  --flash-attn `
  --cache-type-k q4_0 `
  --cache-type-v q4_0
