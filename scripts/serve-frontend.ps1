$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
Push-Location .\frontend
npm run dev
Pop-Location
