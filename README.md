# Ely-Eye

Ely-Eye is a local multimodal context operating system built on Qwen3.5-9B. It
runs on a single consumer GPU and gives a small model three tiers of context: a
262K native window, a 1.01M extended window, and a 100M-token memory capsule it
can retrieve from, cite, and recall across sessions.

Under the hood it ingests PDFs, screenshots, video, and code into Evidence
Atoms, packs them into a Context Cartridge, retrieves with sparse and Qwen
embedding search, runs Qwen3.5-9B with trained LoRA adapters, and checks every
answer with a verifier. A fifteen-check proof suite exercises the whole thing
against the live model, so a real run can be told apart from a fabricated one.

The landing site lives at https://ely-eye-web.zhangyanghaha0407.workers.dev
(its source is under `ely-eye-web/`, which this repository does not track).

## Requirements

- Windows with an NVIDIA GPU and CUDA. Training and the proof suite were built
  on an RTX 4090; deployment targets a 16GB card such as an RTX 4080.
- The extended-context runtimes (SGLang, vLLM, KTransformers, llama.cpp) are
  optional and run under WSL. The default Windows runtime uses Transformers and
  needs none of them.

## Run it locally

```powershell
.\scripts\install.ps1            # virtual env, PyTorch + CUDA, Transformers, PEFT, bitsandbytes
.\scripts\download-models.ps1    # Qwen3.5-9B, the 0.8B planner, Qwen3-VL-Embedding-8B
.\scripts\serve-backend.ps1      # API on http://127.0.0.1:8765
.\scripts\serve-frontend.ps1     # dashboard on http://127.0.0.1:5173
```

When `ELY_EYE_ADAPTER_DIR` is empty, the runtime picks the latest trained
HME-Core-LoRA adapter for `Qwen/Qwen3.5-9B` on its own; `ely-eye status` prints
which adapter it resolved.

For the 262K and 1.01M extended-context profiles, set up the Linux runtimes once
and launch one:

```powershell
.\scripts\setup-linux-runtimes.ps1
.\scripts\launch-sglang-live.ps1       # 262K
.\scripts\launch-sglang-extreme.ps1    # 1.01M via YaRN
```

## Core commands

```powershell
.\.venv\Scripts\ely-eye.exe status
.\.venv\Scripts\ely-eye.exe ingest C:\path\to\package --cartridge-name project
.\.venv\Scripts\ely-eye.exe hashhop --kind visual --hops 2 --token-equivalent 262144
.\.venv\Scripts\ely-eye.exe proof-suite
```

Every command has an HTTP equivalent under `/api/...`, and the dashboard shows
the same state: Adapter Matrix, Proof Suite, Memory Map, Cache Trace.

## Fine-tuning

Training runs real BF16 LoRA updates on the GPU through Unsloth-patched
Transformers, PEFT, and bitsandbytes with paged 8-bit AdamW. Each run writes the
PEFT adapter alongside a manifest, a per-step training trace, a summary, and a
proof file that records dataset and weight hashes, per-step loss, optimizer-update
norms, and the CUDA/BF16 environment.

The adapter set:

```text
HME-Core-LoRA      Qwen3.5-9B attention/MLP
HME-Vision-LoRA    Qwen3.5-9B visual projector and selected vision layers
HME-TTT-VL         Qwen3.5-9B mutable MLP and visual adapter
HME-Visual-MTP     Qwen3.5-9B draft-head LoRA
HME-Router         Qwen3.5-0.8B context planner
HME-Retrieval      Qwen3-VL-Embedding-8B contrastive retrieval adapter
```

## Proof suite

`ely-eye proof-suite` runs fifteen checks against the live model and writes the
evidence under `.ely_eye/data/eval_proofs/`. Among them:

- real CUDA BF16 LoRA training — changed weight hashes, nonzero optimizer
  updates, per-step loss
- the runtime loading the model and adapter and generating a cited answer
- Visual HashHop: multi-hop visual addressing the model has to actually solve,
  with no OCR shortcut
- the 100M-token memory capsule committing to a reproducible Memory DNA
- the Visual Contradiction Lens localizing cross-version drift (design-token,
  layout, typography, copy, visual-code, temporal) at >=80% recall with no false
  positives
- standard benchmark samples (MMMU-Pro, OCRBench, and others) scored against
  ground truth

```powershell
.\.venv\Scripts\ely-eye.exe proof-suite
.\.venv\Scripts\ely-eye.exe latest-proof-suite
.\.venv\Scripts\ely-eye.exe runtime-generation-proof
.\.venv\Scripts\ely-eye.exe visual-contradiction-proof
```

## Data layout

Runtime state lives under `.ely_eye/` (git-ignored):

```text
.ely_eye/
├─ adapters/     trained LoRA adapters
├─ data/         sqlite database and eval proofs
├─ training/     datasets and training traces
├─ objects/      ingested source files
├─ cache/        cache fabric
└─ cartridges/   materialized Context Cartridges
```

A Context Cartridge is a self-contained memory package: the Evidence Atoms, a
sparse index and vector indexes, a temporal graph, KV snapshots, the memory
capsule, any attached adapters, and the eval proofs.

## Project layout

```text
backend/    Python: ingestion, retrieval, runtime, training, proof suite, FastAPI
frontend/   the local dashboard (Vite + React)
scripts/    PowerShell: install, model download, serving, runtimes, fine-tuning
```
