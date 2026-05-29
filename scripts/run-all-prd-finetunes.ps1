param(
  [string]$CartridgeId = "ely_eye_prd_7e5cd27b8284",
  [int]$MaxLength = 8192
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$proofImage = Join-Path $root ".ely_eye\proofs\ely-eye-dashboard.png"
$coreDataset = Join-Path $root ".ely_eye\training\hme_prd_sft.jsonl"
$visualDataset = Join-Path $root ".ely_eye\training\hme_visual_sft.jsonl"
$routerDataset = Join-Path $root ".ely_eye\training\hme_router_sft.jsonl"
$retrievalDataset = Join-Path $root ".ely_eye\training\hme_retrieval_pairs.jsonl"

.\.venv\Scripts\ely-eye.exe build-prd-training-data `
  --prd-path .\PRD.md `
  --output-path $coreDataset `
  --max-sections 48

.\.venv\Scripts\ely-eye.exe build-visual-training-data `
  --prd-path .\PRD.md `
  --output-path $visualDataset `
  --cartridge-id $CartridgeId `
  --image-path $proofImage

.\.venv\Scripts\ely-eye.exe build-router-training-data `
  --prd-path .\PRD.md `
  --output-path $routerDataset `
  --cartridge-id $CartridgeId

.\.venv\Scripts\ely-eye.exe build-retrieval-training-data `
  --prd-path .\PRD.md `
  --output-path $retrievalDataset `
  --cartridge-id $CartridgeId `
  --max-sections 32

.\.venv\Scripts\ely-eye.exe train-lora `
  $coreDataset `
  .\.ely_eye\adapters\hme_core_lora_qwen35_9b `
  --max-steps 37 `
  --kind hme_core_lora `
  --rank 16 `
  --alpha 32 `
  --max-length $MaxLength `
  --gradient-accumulation-steps 4 `
  --cartridge-id $CartridgeId

.\.venv\Scripts\ely-eye.exe train-lora `
  $visualDataset `
  .\.ely_eye\adapters\hme_vision_lora_qwen35_9b `
  --max-steps 20 `
  --kind hme_vision_lora `
  --rank 4 `
  --alpha 8 `
  --max-length $MaxLength `
  --gradient-accumulation-steps 4 `
  --cartridge-id $CartridgeId

.\.venv\Scripts\ely-eye.exe train-lora `
  $visualDataset `
  .\.ely_eye\adapters\hme_ttt_vl_qwen35_9b `
  --max-steps 20 `
  --kind hme_ttt_vl `
  --rank 4 `
  --alpha 8 `
  --max-length $MaxLength `
  --gradient-accumulation-steps 4 `
  --cartridge-id $CartridgeId

.\.venv\Scripts\ely-eye.exe train-lora `
  $visualDataset `
  .\.ely_eye\adapters\hme_visual_mtp_qwen35_9b `
  --max-steps 20 `
  --kind hme_visual_mtp `
  --rank 4 `
  --alpha 8 `
  --max-length $MaxLength `
  --gradient-accumulation-steps 4 `
  --cartridge-id $CartridgeId

.\.venv\Scripts\ely-eye.exe train-lora `
  $routerDataset `
  .\.ely_eye\adapters\hme_router_qwen35_08b `
  --max-steps 16 `
  --kind hme_router `
  --rank 8 `
  --alpha 16 `
  --max-length $MaxLength `
  --gradient-accumulation-steps 4 `
  --cartridge-id $CartridgeId

.\.venv\Scripts\ely-eye.exe train-retrieval-lora `
  $retrievalDataset `
  .\.ely_eye\adapters\hme_retrieval_qwen3_vl_embedding_8b `
  --max-steps 32 `
  --rank 8 `
  --alpha 16 `
  --max-length $MaxLength `
  --cartridge-id $CartridgeId
