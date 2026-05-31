# Ely-Eye MCP Research Notes

Date: 2026-05-30

## Current protocol baseline

The MCP specification latest version observed during implementation is `2025-11-25`. It defines a JSON-RPC 2.0 lifecycle with initialization, capability negotiation, normal operation, and transport-level shutdown. Server features are resources, prompts, and tools; client features include roots, sampling, and elicitation.

Local Ely-Eye deployment uses stdio transport because Codex and Claude Code both support local stdio MCP servers. HTTP remains the better fit for remote shared services with OAuth; Ely-Eye memory is local project state, so stdio keeps the trust boundary on the user's machine.

The official Go SDK is Tier 1 in the MCP SDK list and supports the latest 2025-11-25 spec in v1.4.0 and newer. This project uses `github.com/modelcontextprotocol/go-sdk` v1.6.1.

## Product mapping

Ely-Eye already materializes Context Cartridges, Evidence Atoms, memory capsule indexes, proof suites, and cache events. The MCP layer maps those into agent-native capabilities:

| Ely-Eye object | MCP surface |
|---|---|
| Evidence Atom | `search_atoms`, `fetch_atom` |
| Context Cartridge | `list_cartridges`, `get_cartridge` |
| Context Compiler | `compile_context` |
| PRD and docs | MCP resources |
| Proof Suite | `list_proof_suites`, `get_proof_suite` |
| HashHop / Visual HashHop | `list_hashhop_proofs`, `get_proof_suite` |
| Cache Fabric | `ely_eye_status` |

## Long-context research anchors

Magic's LTM-2-mini announcement establishes the design target: 100M-token context and HashHop as a stricter long-context addressing test than needle-in-a-haystack. Ely-Eye keeps 100M as a logical memory capsule and exposes the capsule through MCP as cartridge scope. Single-request full-attention execution belongs to the physical-model profile.

Qwen3.5-9B supplies the base local VLM target: 262K native context and a 1.01M YaRN profile. Qwen3-VL-Embedding-8B supplies multimodal retrieval coverage across text, images, screenshots, video, and multimodal combinations.

Native Sparse Attention, VL-Cache, SparseVLM, DuoAttention, and the TTT-E2E/Titans line shape the memory system: sparse long-context compute, modality-aware KV compression, visual token sparsification, retrieval/streaming head separation, and test-time memory formation. The MCP server exposes these as evidence and proof artifacts so coding agents can inspect what exists locally before proposing changes.

## Security anchors

Recent MCP security papers and threat taxonomies identify tool poisoning, prompt injection, over-broad tool permissions, and hidden metadata attacks as central risks. This implementation limits the first MCP milestone to read-only local tools, concise tool descriptions, structured output schemas, and project-scoped Claude configuration. Write operations can be added as separate audited tools with explicit names and narrow inputs.

## Sources

- Model Context Protocol specification 2025-11-25: https://modelcontextprotocol.io/specification/2025-11-25
- MCP Go SDK: https://github.com/modelcontextprotocol/go-sdk
- Claude Code MCP docs: https://code.claude.com/docs/en/mcp
- OpenAI Docs MCP and Codex config docs: https://developers.openai.com/learn/docs-mcp
- Magic 100M Token Context Windows: https://magic.dev/blog/100m-token-context-windows
- Qwen3.5-9B model card: https://huggingface.co/Qwen/Qwen3.5-9B
- Qwen3-VL-Embedding-8B model card: https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B
- Native Sparse Attention: https://arxiv.org/abs/2502.11089
- End-to-End Test-Time Training for Long Context: https://arxiv.org/abs/2512.23675
- VL-Cache: https://arxiv.org/abs/2410.23317
- SparseVLM: https://arxiv.org/abs/2410.04417
- DuoAttention: https://arxiv.org/abs/2410.10819
- Titans: https://arxiv.org/abs/2501.00663
- Model Context Protocol security landscape: https://arxiv.org/abs/2503.23278
- MCP threat modeling and tool poisoning: https://arxiv.org/abs/2603.22489
