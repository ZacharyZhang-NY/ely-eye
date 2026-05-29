# Ely-Eye

Ely-Eye is a local multimodal context operating system for Qwen3.5-9B. The implementation follows `PRD.md`: ingestion, Evidence Atoms, Context Cartridge, sparse and optional Qwen embedding retrieval, Cache Fabric, Qwen runtime adapters, verifier contracts, Visual HashHop proof generation, BF16 LoRA fine-tuning, adapter cartridge binding, and a local dashboard.

## Local Run

```powershell
Set-Location C:\Users\zacharyzhang\Documents\Github\ELY-EYE
.\scripts\install.ps1
.\scripts\setup-linux-runtimes.ps1
.\scripts\serve-backend.ps1
.\scripts\serve-frontend.ps1
```

Backend: `http://127.0.0.1:8765`  
Frontend: `http://127.0.0.1:5173`

## Model Runtime

SGLang is supported as an OpenAI-compatible runtime target. The Windows local runtime uses Transformers.

```powershell
.\scripts\download-models.ps1
.\scripts\launch-sglang-live.ps1
```

The Transformers backend is available by setting:

```powershell
$env:ELY_EYE_RUNTIME_BACKEND = "transformers"
```

When `ELY_EYE_ADAPTER_DIR` is empty, the Transformers runtime auto-selects the
latest trained `HME-Core-LoRA` adapter whose base model matches
`Qwen/Qwen3.5-9B`. `ely-eye status` reports the resolved adapter path, kind,
adapter id, and adapter hash.

The runtime generation proof loads `Qwen/Qwen3.5-9B`, injects the trained
`HME-Core-LoRA` adapter, generates a PRD-cited answer, and writes the raw model
output plus verifier report:

```powershell
.\.venv\Scripts\ely-eye.exe runtime-generation-proof
```

## Fine-Tuning

The PRD-derived datasets are built from real `PRD.md` sections, the ingested PRD Evidence Atom, and local UI proof images. Training runs real BF16 LoRA optimizer updates on RTX 4090 through Unsloth-patched Transformers, PEFT, and bitsandbytes, then writes PEFT adapters, `adapter_manifest.json`, `training_trace.jsonl`, `training_summary.json`, `training_proof.json`, dataset hashes, adapter hashes, trainable parameter hashes, optimizer update norms, per-step loss, CUDA/BF16 environment metadata, target modules, adapter weight SHA-256, LoRA tensor statistics, and cartridge binding metadata. The training path requires CUDA BF16 and bitsandbytes `PagedAdamW8bit`; every proof records proof version, precision, optimizer family, checkpointing mode, max length, framework, and target-module coverage.

```powershell
.\scripts\run-prd-finetune.ps1 -MaxSteps 37 -CartridgeId ely_eye_prd_7e5cd27b8284
.\scripts\run-all-prd-finetunes.ps1 -CartridgeId ely_eye_prd_7e5cd27b8284
$env:ELY_EYE_ADAPTER_DIR = ".ely_eye\adapters\hme_core_lora_qwen35_9b"
.\.venv\Scripts\ely-eye.exe status
```

The PRD adapter set is:

```text
HME-Core-LoRA      Qwen3.5-9B attention/MLP
HME-Vision-LoRA    Qwen3.5-9B visual projector and selected vision layers
HME-TTT-VL         Qwen3.5-9B mutable MLP and visual adapter
HME-Visual-MTP     Qwen3.5-9B draft-head LoRA
HME-Router         Qwen3.5-0.8B context planner
HME-Retrieval      Qwen3-VL-Embedding-8B contrastive retrieval adapter
```

## Core Commands

```powershell
.\.venv\Scripts\ely-eye.exe status
.\.venv\Scripts\ely-eye.exe ingest C:\path\to\package --cartridge-name project-cartridge
.\.venv\Scripts\ely-eye.exe hashhop --kind visual --hops 2 --token-equivalent 262144
.\.venv\Scripts\ely-eye.exe validate-training-data C:\path\to\train.jsonl
.\.venv\Scripts\ely-eye.exe train-lora C:\path\to\train.jsonl C:\path\to\adapter
```

Adapter proof state is available through `GET /api/adapters` and the dashboard Adapter Matrix, including training method, precision, optimizer family, checkpointing state, optimizer update counts, and loss delta.

## PRD Proof Suite

The proof suite verifies the PRD adapter set with CUDA BF16 optimizer updates,
changed trainable parameter hashes, matching manifest/summary/proof hashes,
proof-versioned BF16 LoRA contracts, 8192-token max length, paged 8-bit AdamW,
checkpointing, target-module coverage, nonzero LoRA safetensors, runtime adapter binding, runtime adapter generation,
cartridge replay, Visual HashHop assets, model-solved text HashHop, model-solved
Visual HashHop 1-hop and 2-hop pass-rate gates, learned memory adapter binding,
100M memory capsule commitments, Memory DNA, cache trace evidence, and Visual
Contradiction Lens cross-version drift detection (>=80% recall with zero false
positives across the PRD 11.15 taxonomy: design-token, layout, typography, copy,
visual-code, and temporal drift).
It also verifies code repository input coverage through a local structure index,
call graph extraction, and git issue/PR history association.
It writes proof JSON, generated HashHop visual blocks, raw model HashHop
outputs, Visual HashHop arena images, and official benchmark sample predictions under `.ely_eye/data/eval_proofs/`, then
attaches the suite to the active Context Cartridge under `eval_proofs/`.
MMMU-Pro and OCRBench sample predictions must include finite local scores before
the standard benchmark check passes. Every standard benchmark sample must include
a sample SHA-256 and source dataset SHA. LongVideoBench uses the official
`longvideobench/LongVideoBench-Meta` public validation metadata source while the
canonical gated dataset remains tracked in the benchmark registry.

Latest local PRD proof suite: `prd_proof_suite_6bc3fa744c60a0e6` (15 checks passed).

```powershell
.\.venv\Scripts\ely-eye.exe proof-suite --cartridge-id ely_eye_prd_7e5cd27b8284
.\.venv\Scripts\ely-eye.exe latest-proof-suite
.\.venv\Scripts\ely-eye.exe runtime-generation-proof
.\.venv\Scripts\ely-eye.exe visual-contradiction-proof
.\.venv\Scripts\ely-eye.exe refresh-cartridge ely_eye_prd_7e5cd27b8284
.\.venv\Scripts\ely-eye.exe finalize-cartridge ely_eye_prd_7e5cd27b8284
.\.venv\Scripts\ely-eye.exe benchmark-registry
.\.venv\Scripts\ely-eye.exe benchmark-sources
.\.venv\Scripts\ely-eye.exe benchmark-samples
.\.venv\Scripts\ely-eye.exe benchmark-predictions
.\.venv\Scripts\ely-eye.exe runtime-profiles
```

The same surface is available through `POST /api/proof-suite`, `GET /api/proof-suite/latest`, `POST /api/visual-contradiction-proof`, `POST /api/cartridges/{cartridge_id}/refresh`, `POST /api/cartridges/{cartridge_id}/finalize`, `GET /api/cartridges/{cartridge_id}/assets`, `GET /api/benchmarks/registry`, `GET /api/benchmarks/sources`, `GET /api/benchmarks/samples`, `GET /api/benchmarks/predictions`, `GET /api/runtime/profiles`, and the dashboard PRD Proof Suite panel.

Runtime launchers:
`ely-eye runtime-profiles` validates the WSL GPU, Linux runtime packages,
SGLang/vLLM launcher modules, KTransformers import path, and llama.cpp help
output before the proof suite marks runtime delivery ready.

```powershell
.\scripts\setup-linux-runtimes.ps1
.\scripts\launch-sglang-live.ps1
.\scripts\launch-sglang-extreme.ps1
.\scripts\launch-vllm-live.ps1
.\scripts\launch-vllm-extreme.ps1
.\scripts\launch-ktransformers-extreme.ps1
.\scripts\launch-llamacpp-offline.ps1
```

## Data Layout

Runtime state is stored under `.ely_eye/` by default.

```text
.ely_eye/
├─ adapters/
├─ data/ely_eye.sqlite
├─ training/
├─ objects/
├─ cache/
└─ cartridges/
```

Context Cartridge artifacts are materialized as `manifest.json`, `atoms.parquet`, `sources.json`, `sparse_index.bm25.zst`, `temporal_graph.sqlite`, `text_vectors.fp16.zstd`, `visual_vectors.int8.zstd`, `kv_snapshots.kivi`, `memory_capsule.json.zst`, `memory_capsule_index.json`, `ttt_vl_adapter.safetensors`, `visual_mtp_head.safetensors`, `eval_proofs/`, and attached adapter folders.
