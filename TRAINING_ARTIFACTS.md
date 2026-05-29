# Ely-Eye Training Artifacts

Generated on 2026-05-28 from the local `PRD.md`, ingested Evidence Atom
`PRD_0b88b12114634a56:text`, visual Evidence Atom
`ely_eye_dashboard_0180927ffdd3f8b0:image`, local UI proof images, and Qwen
model snapshots. Every adapter has `training_trace.jsonl`, `training_summary.json`,
and `training_proof.json` with RTX 4090 CUDA BF16 execution metadata, proof
version 2, BF16 LoRA method, 8192-token max length, paged 8-bit AdamW,
Unsloth-patched checkpointing mode, per-step loss, trainable parameter hashes
before and after training, optimizer update norms, dataset hashes, environment
versions, adapter weight SHA-256, LoRA tensor statistics, runtime adapter binding
proof, runtime adapter generation proof, and proof-suite verified nonzero LoRA
safetensors.

| Adapter | Base model | Samples | Steps | Updates | Max length | Trainable params | Final loss | Loss delta | Max update L2 | Seconds | SHA-256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HME-Core-LoRA | Qwen/Qwen3.5-9B | 37 | 37 | 10 | 8,192 | 29,097,984 | 0.0001596613001311198 | 0.22773139642958995 | 0.8183563251107077 | 104.4 | 8f97890e802f9adfa3bcb2c02b33d7b39dd8954654d7057d4058e1331e7b5df4 |
| HME-Vision-LoRA | Qwen/Qwen3.5-9B | 10 | 20 | 5 | 8,192 | 1,996,672 | 2.0484652519226074 | -0.11261439323425293 | 0.6595521616294707 | 49.3 | 593f7295f466c5fba07bdae26d1d8798c5affbad8f3863cda9f1881d06453dec |
| HME-TTT-VL | Qwen/Qwen3.5-9B | 10 | 20 | 5 | 8,192 | 8,288,128 | 1.5277884006500244 | 0.4080624580383301 | 1.2924649382409095 | 51.2 | 030b37480155a8eaa6ef4c2dfa819d5f2564838935654f0d826235eb66cc9de5 |
| HME-Visual-MTP | Qwen/Qwen3.5-9B | 10 | 20 | 5 | 8,192 | 2,019,328 | 2.066615343093872 | -0.13076448440551758 | 0.1419292917069751 | 75.8 | 489da9460f51693fa8775ad4975a18f7400dd3fc41eac401d00305617c627717 |
| HME-Router | Qwen/Qwen3.5-0.8B | 8 | 16 | 4 | 8,192 | 3,194,880 | 1.2781919240951538 | 1.0438536405563354 | 0.2871291982199057 | 28.8 | 9ef884a1078df71ba73376a29737a27e22e2ff25854352bfc2c7d960102d886e |
| HME-Retrieval | Qwen/Qwen3-VL-Embedding-8B | 32 | 32 | 32 | 8,192 | 21,823,488 | 0.5665192604064941 | 0.91921067237854 | 1.3181151117874825 | 55.3 | 6490f03e445eb70da7020be992c6eccd3ca2b0e4c48e73a80a819ddac5faa8f3 |

Adapter weight proof:

| Adapter | LoRA tensors | Nonzero elements | Weight SHA-256 |
|---|---:|---:|---|
| HME-Core-LoRA | 256 | 29,097,984 | 4673ac90abdd8ef952dad9c2292aecadba161669c3b1fc3d1af5c075cf0ae62d |
| HME-Vision-LoRA | 220 | 1,992,340 | dde2eb9051866634f9173366a6ffa3fbf9219e5b6d640eafb7f99d0258c52422 |
| HME-TTT-VL | 412 | 8,283,800 | 9a572ac51c6df08e38898c6c30acd8ed6530c7e852611552baaec6a67862e6c5 |
| HME-Visual-MTP | 4 | 1,028,316 | 58d058b862a542c042d27234b5b22344ba02ee6e016be6c8838b329ffea1c6f9 |
| HME-Router | 192 | 3,194,880 | 90079b8f2c244efb00c2285816b98f51ff03cf06a9ba2335ee5bbdf24a1aff8f |
| HME-Retrieval | 504 | 21,823,488 | bcb1381888ae84e82ab18256e152191e56a0892c46cdac81874a9c1fb1af14ee |

Local adapter roots:

```text
.ely_eye/adapters/hme_core_lora_qwen35_9b
.ely_eye/adapters/hme_vision_lora_qwen35_9b
.ely_eye/adapters/hme_ttt_vl_qwen35_9b
.ely_eye/adapters/hme_visual_mtp_qwen35_9b
.ely_eye/adapters/hme_router_qwen35_08b
.ely_eye/adapters/hme_retrieval_qwen3_vl_embedding_8b
```

Cartridge binding:

```text
.ely_eye/cartridges/ely_eye_prd_7e5cd27b8284/adapters/hme_core_lora
.ely_eye/cartridges/ely_eye_prd_7e5cd27b8284/adapters/hme_vision_lora
.ely_eye/cartridges/ely_eye_prd_7e5cd27b8284/adapters/hme_ttt_vl
.ely_eye/cartridges/ely_eye_prd_7e5cd27b8284/adapters/hme_visual_mtp
.ely_eye/cartridges/ely_eye_prd_7e5cd27b8284/adapters/hme_router
.ely_eye/cartridges/ely_eye_prd_7e5cd27b8284/adapters/hme_retrieval
```

PRD proof suite:

```text
.ely_eye/data/eval_proofs/prd_proof_suite_b03b88d7cfe5ba96/proof_suite.json
.ely_eye/cartridges/ely_eye_prd_7e5cd27b8284/eval_proofs/prd_proof_suite_b03b88d7cfe5ba96/proof_suite.json
```

Runtime generation proof:

```text
.ely_eye/data/eval_proofs/prd_proof_suite_b03b88d7cfe5ba96/runtime_generation.json
transformers generated 218 tokens with hme_core_lora citation accuracy 1.00
```

100M memory capsule proof:

```text
.ely_eye/cartridges/ely_eye_prd_7e5cd27b8284/memory_capsule.json.zst
.ely_eye/cartridges/ely_eye_prd_7e5cd27b8284/memory_capsule_index.json
.ely_eye/data/eval_proofs/prd_proof_suite_b03b88d7cfe5ba96/memory_capsule.json
The Context Cartridge materialized a 100,000,000 token-equivalent memory capsule with 4,096 committed segments and a verified segment Merkle root.
```

Model-solved HashHop proofs:

```text
.ely_eye/data/eval_proofs/prd_proof_suite_b03b88d7cfe5ba96/hashhop/0a10e6394ada0da7/model_text_hashhop.json
.ely_eye/data/eval_proofs/prd_proof_suite_b03b88d7cfe5ba96/hashhop/60c9d5f5f1648ea0/model_visual_hashhop.json
.ely_eye/data/eval_proofs/prd_proof_suite_b03b88d7cfe5ba96/hashhop/a29a310b31f9cef5/model_visual_hashhop.json
.ely_eye/data/eval_proofs/prd_proof_suite_b03b88d7cfe5ba96/hashhop/7f7b8b1f54ec3d75/model_visual_hashhop.json
.ely_eye/data/eval_proofs/prd_proof_suite_b03b88d7cfe5ba96/hashhop/4f24efd13342694c/model_visual_hashhop.json
.ely_eye/data/eval_proofs/prd_proof_suite_b03b88d7cfe5ba96/hashhop/a3f49402ef2710a3/model_visual_hashhop.json
.ely_eye/data/eval_proofs/prd_proof_suite_b03b88d7cfe5ba96/hashhop/0bd17526ab51392b/model_visual_hashhop.json
Qwen/Qwen3.5-9B with HME-Core-LoRA solved a 1.01M token-equivalent 2-hop text HashHop and 6 Visual HashHop arenas at 262K token-equivalent. The Visual HashHop gate passed 7/7 model-solved proofs, with 262K 1-hop=1.00 and 2-hop=1.00.
```

Benchmark source and sample proof:

```text
.ely_eye/data/eval_proofs/prd_proof_suite_b03b88d7cfe5ba96/standard_benchmark_sources.json
.ely_eye/data/eval_proofs/prd_proof_suite_b03b88d7cfe5ba96/standard_benchmark_samples.json
.ely_eye/data/eval_proofs/prd_proof_suite_b03b88d7cfe5ba96/standard_benchmark_predictions.json
MMMU-Pro, OCRBench, MMLongBench-Doc, VideoMME, LongVideoBench, RULER, and LongBench v2 loaded official streaming samples or official public metadata samples with sample hashes and source dataset SHA values. LongVideoBench uses `longvideobench/LongVideoBench-Meta` validation metadata with canonical `longvideobench/LongVideoBench` registry tracking. MMMU-Pro and OCRBench generated local Qwen/Qwen3.5-9B + HME-Core-LoRA predictions from official image samples, with local sample scores MMMU-Pro=1.00 and OCRBench=1.00.
```

Repository input proof:

```text
.ely_eye/data/eval_proofs/prd_proof_suite_b03b88d7cfe5ba96/repository_input.json
The repository input proof indexed 94 files, 49 code files, 11,977 call graph edges, and 26 local git commits for PRD 10.1 code repository coverage.
```

Runtime profile readiness:

```text
.ely_eye/data/eval_proofs/prd_proof_suite_b03b88d7cfe5ba96/runtime_profiles.json
SGLang 0.5.12.post1, vLLM 0.21.0, KTransformers 0.6.2.post4, WSL RTX 4090 GPU visibility, launcher module probes, and winget llama-server help output are verified.
```

Checks:

```text
Adapter Training Proof    passed  # proof-versioned BF16 LoRA + paged 8-bit AdamW + nonzero LoRA safetensors
Runtime Adapter Binding   passed
Runtime Adapter Generation passed
Cartridge Physical Assets passed
100M Memory Capsule      passed
Context Cartridge Replay  passed
Visual HashHop            passed  # 7/7 model-solved proofs
Standard Benchmark Registry passed  # MMMU-Pro=1.00, OCRBench=1.00 local samples
Repository Input Coverage passed  # 94 files + 11977 call graph edges + local git history
Runtime Profile Delivery  passed
Research Module Delivery  passed  # executable module gates, evidence graph edge, probability sum
Learned Memory Recall     passed
Memory DNA                passed
Cache Trace               passed
```

Cartridge physical assets:

```text
.ely_eye/cartridges/ely_eye_prd_7e5cd27b8284/text_vectors.fp16.zstd
.ely_eye/cartridges/ely_eye_prd_7e5cd27b8284/visual_vectors.int8.zstd  # item_count=1
.ely_eye/cartridges/ely_eye_prd_7e5cd27b8284/kv_snapshots.kivi
.ely_eye/cartridges/ely_eye_prd_7e5cd27b8284/memory_capsule.json.zst   # token_equivalent=100000000, segment_count=4096
.ely_eye/cartridges/ely_eye_prd_7e5cd27b8284/ttt_vl_adapter.safetensors
.ely_eye/cartridges/ely_eye_prd_7e5cd27b8284/visual_mtp_head.safetensors
```
