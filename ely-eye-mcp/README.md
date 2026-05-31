# Ely-Eye MCP

Ely-Eye MCP exposes a local Context Cartridge memory store as a stdio Model Context Protocol server for coding agents such as Codex and Claude Code.

The server is local-first and read-only against `<ely-eye-home>/data/ely_eye.sqlite`. It reads Evidence Atoms, cartridge manifests, Memory DNA, proof-suite artifacts, HashHop proofs, and compiled evidence packs. The production database is never modified.

It can be deployed for any `.ely_eye` store. Identity is anchored on the database file, not on the Ely-Eye repository layout, so the server walks up from the working directory to find `.ely_eye/data/ely_eye.sqlite`, or takes an explicit `--ely-eye-home`.

## One-command setup

The installer downloads a checksum-verified prebuilt binary for the platform when a tagged release is available, otherwise it builds from source with Go from a repository checkout. Either way it installs the binary into `<ely-eye-home>/bin` and registers the server with Codex and Claude Code. No Docker.

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\ely-eye-mcp\install.ps1 -Client both
```

Linux and macOS:

```bash
bash ./ely-eye-mcp/install.sh --client both
```

Force a source build (requires Go 1.25 or newer):

```bash
bash ./ely-eye-mcp/install.sh --method source
```

Codex registration uses `codex mcp add`. Claude Code project registration writes `.mcp.json` at the project root; `--scope user` registers through the Claude CLI instead.

## Manual run

From inside a project that owns an `.ely_eye` store:

```bash
go run ./cmd/ely-eye-mcp server
```

Or point at an explicit store:

```bash
go run ./cmd/ely-eye-mcp server --ely-eye-home /path/to/.ely_eye
```

## Tools

`ely_eye_status` returns atom, source, cartridge, token, and cache-layer counts.

`list_cartridges` lists local Context Cartridges with manifest JSON and Memory DNA.

`get_cartridge` reads one cartridge plus artifact metadata, memory capsule index, and asset report.

`search_atoms` queries the local SQLite FTS index and returns ranked Evidence Atoms.

`fetch_atom` reads one Evidence Atom by exact id.

`compile_context` compiles a question into a local evidence pack with the same profile names used by the app (`live_demo`, `extreme_context`, `library_100m`, `research_theater`).

`list_proof_suites` lists PRD proof suites under `<ely-eye-home>/data/eval_proofs`.

`list_hashhop_proofs` aggregates HashHop and Visual HashHop long-context addressing proofs by kind, hop count, and token budget, with pass counts and example proofs. Proofs are deduplicated by content-derived id.

`get_proof_suite` reads one PRD proof suite by id, including all checks and a summary of its HashHop proofs.

## Resources

`ely-eye://project/prd`, `ely-eye://project/readme`, and `ely-eye://project/mcp-research` expose `PRD.md`, the project README, and these research notes when those files are present next to the store.

## Releases

Pushing a `mcp-v*` tag runs `.github/workflows/mcp-release.yml`, which cross-compiles the pure-Go server (CGO disabled) for Linux, macOS, and Windows on amd64 and arm64, then publishes the archives with a `SHA256SUMS` manifest to a GitHub Release. The installers consume those assets.

## Security posture

The server opens SQLite read-only with `mode=ro` and `PRAGMA query_only(1)`, and uses `busy_timeout` so reads tolerate concurrent writes by the Ely-Eye app. Tools expose structured outputs and return tool-level errors for invalid inputs so agents self-correct through normal tool calls. Proof ids are validated as single path elements before any filesystem read. The server uses stdio transport for local clients and takes credentials only from the local environment.
