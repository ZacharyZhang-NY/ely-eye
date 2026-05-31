#requires -version 5.1
[CmdletBinding()]
param(
  [ValidateSet("codex", "claude", "both")]
  [string]$Client = "both",

  [ValidateSet("project", "user")]
  [string]$Scope = "project",

  [string]$ServerName = "ely-eye",

  [ValidateSet("auto", "download", "source")]
  [string]$Method = "auto",

  [string]$Version = "",
  [string]$ProjectRoot = "",
  [string]$ElyEyeHome = "",
  [string]$Repo = "ZacharyZhang-NY/ely-eye"
)

# Ely-Eye MCP installer for Windows. By default it downloads a prebuilt,
# checksum-verified binary for this platform and needs no toolchain. Use
# -Method source from a repository checkout to build with Go instead.

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

if (-not $ProjectRoot) {
  if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "go.mod"))) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
  }
  else {
    $ProjectRoot = (Get-Location).Path
  }
}
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
if (-not $ElyEyeHome) { $ElyEyeHome = Join-Path $ProjectRoot ".ely_eye" }
$BinDir = Join-Path $ElyEyeHome "bin"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
$Binary = Join-Path $BinDir "ely-eye-mcp.exe"

function Get-Platform {
  # PROCESSOR_ARCHITEW6432 holds the true machine architecture when PowerShell
  # runs as an x64-emulated process on ARM64 Windows; fall back to the process
  # architecture otherwise.
  $arch = $env:PROCESSOR_ARCHITEW6432
  if (-not $arch) { $arch = $env:PROCESSOR_ARCHITECTURE }
  switch ($arch) {
    "AMD64" { return "windows_amd64" }
    "ARM64" { return "windows_arm64" }
    default { throw "unsupported architecture: $arch" }
  }
}

function Resolve-Version {
  if ($Version) { return $Version }
  try {
    $releases = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases" -Headers @{ "User-Agent" = "ely-eye-mcp-installer" }
  }
  catch { return "" }
  $match = $releases | Where-Object { $_.tag_name -like "mcp-*" } | Select-Object -First 1
  if ($match) { return $match.tag_name }
  return ""
}

function Build-FromSource {
  if (-not (Get-Command go -ErrorAction SilentlyContinue)) { throw "Go 1.25 or newer is required to build from source." }
  if (-not ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "go.mod")))) { throw "source builds must run from a repository checkout." }
  Write-Host "Building ely-eye-mcp from source with Go."
  $env:CGO_ENABLED = "0"
  Push-Location $PSScriptRoot
  try {
    go build -trimpath -ldflags="-s -w" -o $Binary .\cmd\ely-eye-mcp
    if ($LASTEXITCODE -ne 0) { throw "go build failed with exit code $LASTEXITCODE." }
  }
  finally { Pop-Location }
}

function Download-Binary([string]$Tag) {
  $platform = Get-Platform
  $asset = "ely-eye-mcp_$platform.zip"
  $url = "https://github.com/$Repo/releases/download/$Tag/$asset"
  $sumsUrl = "https://github.com/$Repo/releases/download/$Tag/ely-eye-mcp_SHA256SUMS.txt"
  $tmp = Join-Path ([IO.Path]::GetTempPath()) ("ely-eye-mcp-" + [Guid]::NewGuid().ToString("N"))
  New-Item -ItemType Directory -Force -Path $tmp | Out-Null
  try {
    Write-Host "Downloading $asset from release $Tag."
    $archive = Join-Path $tmp $asset
    Invoke-WebRequest -Uri $url -OutFile $archive
    $sumsFile = Join-Path $tmp "SHA256SUMS"
    Invoke-WebRequest -Uri $sumsUrl -OutFile $sumsFile
    $line = Select-String -Path $sumsFile -Pattern ("\*?" + [Regex]::Escape($asset) + "$") | Select-Object -First 1
    if (-not $line) { throw "checksum for $asset not found in release." }
    $expected = ($line.Line -split "\s+")[0]
    $actual = (Get-FileHash -Algorithm SHA256 -Path $archive).Hash
    if ($expected.ToLower() -ne $actual.ToLower()) { throw "checksum mismatch for $asset." }
    Expand-Archive -Force -Path $archive -DestinationPath $tmp
    Move-Item -Force -Path (Join-Path $tmp "ely-eye-mcp.exe") -Destination $Binary
  }
  finally { Remove-Item -Recurse -Force $tmp }
}

switch ($Method) {
  "source" { Build-FromSource }
  "download" {
    $tag = Resolve-Version
    if (-not $tag) { throw "no prebuilt release found for $Repo." }
    Download-Binary $tag
  }
  "auto" {
    $tag = Resolve-Version
    if ($tag) { Download-Binary $tag }
    else {
      Write-Host "No prebuilt release available; building from source."
      Build-FromSource
    }
  }
}

& $Binary setup `
  --client $Client `
  --scope $Scope `
  --server-name $ServerName `
  --project-root $ProjectRoot `
  --ely-eye-home $ElyEyeHome `
  --binary $Binary
