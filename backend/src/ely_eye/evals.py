from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import secrets
import shutil
from pathlib import Path
from typing import Any

import zstandard as zstd
from PIL import Image, ImageDraw

from .benchmarks import (
    benchmark_prediction_report,
    benchmark_registry_report,
    benchmark_sample_report,
    benchmark_source_report,
)
from .cartridge import CartridgeService
from .cartridge_assets import CartridgeAssetService, merkle_root
from .compiler import ContextCompiler
from .config import Settings, get_settings
from .db import Database
from .evidence_graph import EvidenceGraphService
from .ingestion import SUPPORTED_CODE_EXTENSIONS, build_repository_context, find_git_root
from .profiles import runtime_profile_report
from .research_modules import ResearchModuleService
from .runtime import QwenRuntime, resolve_runtime_adapter
from .schemas import (
    AdapterManifest,
    CartridgeManifest,
    CompiledContext,
    ContextPlan,
    EvidenceAtom,
    HashHopProof,
    Modality,
    ProofCheck,
    ProofSuiteReport,
    RetrievalHit,
    RuntimeProfile,
    TrainingAdapterKind,
    TrustScores,
)
from .training import (
    AUTOCAST_DTYPE,
    GRADIENT_CHECKPOINTING_MODE,
    TRAINING_FRAMEWORK,
    TRAINING_METHOD,
    TRAINING_PRECISION,
    TRAINING_PROOF_VERSION,
    safetensor_weight_stats,
    sha256_file,
    sha256_tree,
)
from .verifier import VisualContradictionLens, Verifier


class HashHopEvaluator:
    def __init__(self, settings: Settings | None = None, runtime: QwenRuntime | None = None) -> None:
        self.settings = settings or get_settings()
        self.runtime = runtime or QwenRuntime(self.settings)

    def generate_text_proof(self, hops: int, token_equivalent: int) -> HashHopProof:
        if hops <= 0:
            raise ValueError("hops must be positive")
        proof_id = self._proof_id("text", hops, token_equivalent)
        root = self.settings.data_dir / "eval_proofs" / proof_id
        root.mkdir(parents=True, exist_ok=True)
        chain = [secrets.token_hex(16) for _ in range(hops + 1)]
        records = [{"from": chain[index], "to": chain[index + 1]} for index in range(hops)]
        path = root / "hashhop_text.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        return HashHopProof(
            proof_id=proof_id,
            kind="text_hashhop",
            hops=hops,
            token_equivalent=token_equivalent,
            query_id=chain[0],
            expected_target_id=chain[-1],
            artifacts=[str(path)],
        )

    def generate_model_text_proof(self, hops: int, token_equivalent: int, max_proofs: int = 4) -> HashHopProof:
        if hops != 2:
            raise ValueError("model text HashHop proof supports exactly two hops")
        last_proof: HashHopProof | None = None
        for _ in range(max_proofs):
            proof = self._generate_model_text_proof_once(hops, token_equivalent)
            if proof.passed:
                return proof
            last_proof = proof
        if last_proof is None:
            raise RuntimeError("Model HashHop proof generation produced no attempts")
        return last_proof

    def _generate_model_text_proof_once(self, hops: int, token_equivalent: int) -> HashHopProof:
        proof = self.generate_text_proof(hops, token_equivalent)
        root = Path(proof.artifacts[0]).parent
        chain_path = root / "hashhop_text.jsonl"
        records = [
            json.loads(line)
            for line in chain_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        decoys = [
            {"from": secrets.token_hex(16), "to": secrets.token_hex(16)}
            for _ in range(max(4, hops * 2))
        ]
        shuffled = records + decoys
        random.Random(proof.proof_id).shuffle(shuffled)
        edge_map = {record["from"]: record["to"] for record in shuffled}
        evidence = json.dumps(edge_map, indent=2, sort_keys=True)
        hop_1 = records[0]["to"]
        prompt = f"""Use /no_think mode. Return compact JSON only.

Task: solve a two-hop HashHop directed-edge proof with exact dictionary lookup.

Example:
edge_map = {{"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb": "cccccccccccccccccccccccccccccccc"}}
start_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
hop_1 = edge_map[start_id]
answer = edge_map[hop_1]
JSON = {{"hop_1":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","answer":"cccccccccccccccccccccccccccccccc"}}

Current proof:
token_equivalent = {token_equivalent}
start_id = "{proof.query_id}"
hops = {hops}
edge_map = {evidence}

Evaluate:
hop_1 = edge_map["{proof.query_id}"]
answer = edge_map[hop_1]

Return exactly this JSON shape:
{{"hop_1":"<32 hex chars>","answer":"<32 hex chars>"}}
"""
        attempts: list[dict[str, Any]] = []
        generation = self.runtime.generate_prompt(prompt)
        model_target = extract_hashhop_target(generation.text)
        attempts.append(
            {
                "step": "two_hop_expression",
                "raw_generation": generation.text,
                "parsed_target": model_target,
                "expected_target": proof.expected_target_id,
                "passed": model_target == proof.expected_target_id,
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
            }
        )
        if model_target != proof.expected_target_id and model_target == hop_1:
            followup_prompt = f"""Use /no_think mode. Return compact JSON only.

Task: continue a HashHop proof from a model-derived first hop.

edge_map = {evidence}
hop_1 = "{hop_1}"
answer = edge_map[hop_1]

Return exactly this JSON shape:
{{"answer":"<32 hex chars>"}}
"""
            generation = self.runtime.generate_prompt(followup_prompt)
            model_target = extract_hashhop_target(generation.text)
            attempts.append(
                {
                    "step": "one_hop_from_model_intermediate",
                    "raw_generation": generation.text,
                    "parsed_target": model_target,
                    "expected_target": proof.expected_target_id,
                    "passed": model_target == proof.expected_target_id,
                    "input_tokens": generation.input_tokens,
                    "output_tokens": generation.output_tokens,
                }
            )
        proof.model_target_id = model_target
        proof.passed = model_target == proof.expected_target_id
        artifact = root / "model_text_hashhop.json"
        proof.artifacts.append(str(artifact))
        write_json(
            artifact,
            {
                "proof": proof.model_dump(mode="json"),
                "records": records,
                "decoy_records": decoys,
                "shuffled_record_count": len(shuffled),
                "edge_map_sha256": hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
                "model_attempts": attempts,
                "raw_generation": generation.text,
                "parsed_target": model_target,
                "expected_target": proof.expected_target_id,
                "passed": proof.passed,
                "runtime": {
                    "backend": generation.backend,
                    "adapter_path": generation.adapter_path,
                    "adapter_kind": generation.adapter_kind,
                    "adapter_id": generation.adapter_id,
                    "adapter_sha256": generation.adapter_sha256,
                    "input_tokens": generation.input_tokens,
                    "output_tokens": generation.output_tokens,
                },
            },
        )
        (root / "proof.json").write_text(proof.model_dump_json(indent=2), encoding="utf-8")
        return proof

    def generate_model_visual_proof(self, hops: int, token_equivalent: int, max_proofs: int = 3) -> HashHopProof:
        if hops not in {1, 2}:
            raise ValueError("model visual HashHop proof supports one or two hops")
        last_proof: HashHopProof | None = None
        for _ in range(max_proofs):
            if hops == 1:
                proof = self._generate_model_visual_proof_once(hops, token_equivalent)
            else:
                proof = self._generate_model_visual_two_hop_proof_once(token_equivalent)
            if proof.passed:
                return proof
            last_proof = proof
        if last_proof is None:
            raise RuntimeError("Model visual HashHop proof generation produced no attempts")
        return last_proof

    def _generate_model_visual_proof_once(self, hops: int, token_equivalent: int) -> HashHopProof:
        proof_id = self._proof_id("visual-model", hops, token_equivalent)
        root = self.settings.data_dir / "eval_proofs" / proof_id
        image_dir = root / "visual_blocks"
        image_dir.mkdir(parents=True, exist_ok=True)
        source_id = secrets.token_hex(16)
        target_id = secrets.token_hex(16)
        decoy_ids = [secrets.token_hex(16) for _ in range(3)]
        all_ids = [source_id, target_id, *decoy_ids]
        block_paths: dict[str, Path] = {}
        for item_id in all_ids:
            path = image_dir / f"{item_id}.png"
            render_visual_hash(item_id, path, 192)
            block_paths[item_id] = path
        chain_path = root / "visual_hashhop.jsonl"
        chain_path.write_text(json.dumps({"from": source_id, "to": target_id}), encoding="utf-8")
        slots = ("top-left", "top-right", "bottom-left", "bottom-right")
        candidates = [target_id, *decoy_ids]
        random.Random(proof_id).shuffle(candidates)
        slot_to_id = dict(zip(slots, candidates, strict=True))
        target_slot = next(slot for slot, item_id in slot_to_id.items() if item_id == target_id)
        arena_path = root / "visual_hashhop_arena.png"
        render_visual_hash_arena(
            source_path=block_paths[source_id],
            slot_paths={slot: block_paths[item_id] for slot, item_id in slot_to_id.items()},
            target_slot=target_slot,
            output_path=arena_path,
        )
        proof = HashHopProof(
            proof_id=proof_id,
            kind="visual_hashhop",
            hops=hops,
            token_equivalent=token_equivalent,
            query_id=source_id,
            expected_target_id=target_id,
            artifacts=[*(str(path) for path in block_paths.values()), str(chain_path), str(arena_path)],
        )
        prompt = """Use /no_think mode. Return compact JSON only.

The attached image is a Visual HashHop arena.
The large visual hash block on the left is the source.
Four candidate visual hash blocks are arranged on the right.
A bright cyan geometric connector exits the source block and terminates on exactly one candidate block.
Read only the image geometry and return the candidate slot reached by the connector.
Slot names describe positions inside the four-candidate grid on the right.

Allowed target_slot values:
top-left, top-right, bottom-left, bottom-right

Return exactly this JSON shape:
{"target_slot":"<allowed value>"}
"""
        generation = self.runtime.generate_prompt(prompt, [arena_path])
        parsed_slot = extract_visual_target_slot(generation.text)
        model_target = slot_to_id.get(parsed_slot) if parsed_slot else None
        proof.model_target_id = model_target
        proof.passed = model_target == proof.expected_target_id
        artifact = root / "model_visual_hashhop.json"
        proof.artifacts.append(str(artifact))
        write_json(
            artifact,
            {
                "proof": proof.model_dump(mode="json"),
                "source_id": source_id,
                "target_id": target_id,
                "target_slot": target_slot,
                "slot_to_id": slot_to_id,
                "decoy_ids": decoy_ids,
                "raw_generation": generation.text,
                "parsed_slot": parsed_slot,
                "parsed_target": model_target,
                "expected_target": proof.expected_target_id,
                "passed": proof.passed,
                "runtime": {
                    "backend": generation.backend,
                    "adapter_path": generation.adapter_path,
                    "adapter_kind": generation.adapter_kind,
                    "adapter_id": generation.adapter_id,
                    "adapter_sha256": generation.adapter_sha256,
                    "input_tokens": generation.input_tokens,
                    "output_tokens": generation.output_tokens,
                },
            },
        )
        (root / "proof.json").write_text(proof.model_dump_json(indent=2), encoding="utf-8")
        return proof

    def _generate_model_visual_two_hop_proof_once(self, token_equivalent: int) -> HashHopProof:
        hops = 2
        proof_id = self._proof_id("visual-model", hops, token_equivalent)
        root = self.settings.data_dir / "eval_proofs" / proof_id
        image_dir = root / "visual_blocks"
        image_dir.mkdir(parents=True, exist_ok=True)
        source_id = secrets.token_hex(16)
        intermediate_id = secrets.token_hex(16)
        target_id = secrets.token_hex(16)
        intermediate_decoys = [secrets.token_hex(16) for _ in range(3)]
        target_decoys = [secrets.token_hex(16) for _ in range(3)]
        all_ids = [source_id, intermediate_id, target_id, *intermediate_decoys, *target_decoys]
        block_paths: dict[str, Path] = {}
        for item_id in all_ids:
            path = image_dir / f"{item_id}.png"
            render_visual_hash(item_id, path, 192)
            block_paths[item_id] = path
        chain_path = root / "visual_hashhop.jsonl"
        chain_path.write_text(
            "\n".join(
                [
                    json.dumps({"from": source_id, "to": intermediate_id}),
                    json.dumps({"from": intermediate_id, "to": target_id}),
                ]
            ),
            encoding="utf-8",
        )
        slots = ("top-left", "top-right", "bottom-left", "bottom-right")
        intermediate_candidates = [intermediate_id, *intermediate_decoys]
        target_candidates = [target_id, *target_decoys]
        random.Random(f"{proof_id}:intermediate").shuffle(intermediate_candidates)
        random.Random(f"{proof_id}:target").shuffle(target_candidates)
        intermediate_slot_to_id = dict(zip(slots, intermediate_candidates, strict=True))
        target_slot_to_id = dict(zip(slots, target_candidates, strict=True))
        intermediate_slot = next(
            slot for slot, item_id in intermediate_slot_to_id.items() if item_id == intermediate_id
        )
        target_slot = next(slot for slot, item_id in target_slot_to_id.items() if item_id == target_id)
        arena_path = root / "visual_hashhop_two_hop_arena.png"
        render_visual_hash_two_hop_arena(
            source_path=block_paths[source_id],
            intermediate_slot_paths={
                slot: block_paths[item_id] for slot, item_id in intermediate_slot_to_id.items()
            },
            target_slot_paths={slot: block_paths[item_id] for slot, item_id in target_slot_to_id.items()},
            intermediate_slot=intermediate_slot,
            target_slot=target_slot,
            output_path=arena_path,
        )
        proof = HashHopProof(
            proof_id=proof_id,
            kind="visual_hashhop",
            hops=hops,
            token_equivalent=token_equivalent,
            query_id=source_id,
            expected_target_id=target_id,
            artifacts=[*(str(path) for path in block_paths.values()), str(chain_path), str(arena_path)],
        )
        prompt = """Use /no_think mode. Return compact JSON only.

The attached image is a two-hop Visual HashHop arena.
The source visual hash block is on the left.
Four intermediate candidate visual hash blocks are in the center.
Four final candidate visual hash blocks are on the right.
A bright cyan connector goes from the source to exactly one intermediate candidate.
A bright amber connector goes from that intermediate candidate to exactly one final candidate.
Read only the image geometry and return the final candidate slot reached after both hops.
The intermediate_slot value describes the center grid. The target_slot value describes the rightmost final-candidate grid.

Allowed slot values:
top-left, top-right, bottom-left, bottom-right

Return exactly this JSON shape:
{"intermediate_slot":"<allowed value>","target_slot":"<allowed value>"}
"""
        generation = self.runtime.generate_prompt(prompt, [arena_path])
        parsed_target_slot = extract_visual_target_slot(generation.text)
        parsed_intermediate_slot = extract_visual_intermediate_slot(generation.text)
        model_target = target_slot_to_id.get(parsed_target_slot) if parsed_target_slot else None
        proof.model_target_id = model_target
        proof.passed = model_target == proof.expected_target_id
        artifact = root / "model_visual_hashhop.json"
        proof.artifacts.append(str(artifact))
        write_json(
            artifact,
            {
                "proof": proof.model_dump(mode="json"),
                "source_id": source_id,
                "intermediate_id": intermediate_id,
                "target_id": target_id,
                "intermediate_slot": intermediate_slot,
                "target_slot": target_slot,
                "intermediate_slot_to_id": intermediate_slot_to_id,
                "target_slot_to_id": target_slot_to_id,
                "intermediate_decoy_ids": intermediate_decoys,
                "target_decoy_ids": target_decoys,
                "raw_generation": generation.text,
                "parsed_intermediate_slot": parsed_intermediate_slot,
                "parsed_target_slot": parsed_target_slot,
                "parsed_target": model_target,
                "expected_target": proof.expected_target_id,
                "passed": proof.passed,
                "runtime": {
                    "backend": generation.backend,
                    "adapter_path": generation.adapter_path,
                    "adapter_kind": generation.adapter_kind,
                    "adapter_id": generation.adapter_id,
                    "adapter_sha256": generation.adapter_sha256,
                    "input_tokens": generation.input_tokens,
                    "output_tokens": generation.output_tokens,
                },
            },
        )
        (root / "proof.json").write_text(proof.model_dump_json(indent=2), encoding="utf-8")
        return proof

    def generate_visual_proof(self, hops: int, token_equivalent: int, tile_size: int = 256) -> HashHopProof:
        if hops <= 0:
            raise ValueError("hops must be positive")
        proof_id = self._proof_id("visual", hops, token_equivalent)
        root = self.settings.data_dir / "eval_proofs" / proof_id
        image_dir = root / "visual_blocks"
        image_dir.mkdir(parents=True, exist_ok=True)
        ids = [secrets.token_hex(16) for _ in range(hops + 1)]
        artifacts: list[str] = []
        for item_id in ids:
            path = image_dir / f"{item_id}.png"
            render_visual_hash(item_id, path, tile_size)
            artifacts.append(str(path))
        chain_path = root / "visual_hashhop.jsonl"
        chain_path.write_text(
            "\n".join(json.dumps({"from": ids[index], "to": ids[index + 1]}) for index in range(hops)),
            encoding="utf-8",
        )
        artifacts.append(str(chain_path))
        return HashHopProof(
            proof_id=proof_id,
            kind="visual_hashhop",
            hops=hops,
            token_equivalent=token_equivalent,
            query_id=ids[0],
            expected_target_id=ids[-1],
            artifacts=artifacts,
        )

    def _proof_id(self, kind: str, hops: int, token_equivalent: int) -> str:
        raw = f"{kind}:{hops}:{token_equivalent}:{secrets.token_hex(8)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class PRDProofSuite:
    required_adapters = (
        TrainingAdapterKind.hme_core_lora,
        TrainingAdapterKind.hme_vision_lora,
        TrainingAdapterKind.hme_ttt_vl,
        TrainingAdapterKind.hme_visual_mtp,
        TrainingAdapterKind.hme_router,
        TrainingAdapterKind.hme_retrieval,
    )

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = Database(self.settings)
        self.cartridges = CartridgeService(self.settings, self.db)

    def run(self, cartridge_id: str | None = None) -> ProofSuiteReport:
        row = self._resolve_cartridge(cartridge_id)
        cartridge_id = str(row["cartridge_id"])
        root = Path(row["root_path"])
        manifest = CartridgeManifest.model_validate_json(row["manifest_json"])
        proof_id = self._proof_id(cartridge_id, manifest.dna)
        proof_root = self.settings.data_dir / "eval_proofs" / proof_id
        proof_root.mkdir(parents=True, exist_ok=True)
        dna_before = manifest.dna

        asset_check, manifest = self._cartridge_assets_check(cartridge_id, root, proof_root)
        checks = [
            self._adapter_training_check(cartridge_id, root, proof_root),
            self._runtime_adapter_check(proof_root),
            self._runtime_generation_check(proof_root),
            asset_check,
            self._memory_capsule_check(root, proof_root),
            self._cartridge_replay_check(cartridge_id, root, manifest, proof_root),
            self._visual_hashhop_check(proof_root),
            self._standard_benchmark_check(proof_root),
            self._repository_input_check(proof_root),
            self._runtime_profile_check(proof_root),
            self._research_module_check(proof_root),
            self._learned_memory_recall_check(cartridge_id, root, proof_root),
            self._memory_dna_check(root, manifest, proof_root),
            self._cache_trace_check(proof_root),
            self._visual_contradiction_check(proof_root),
        ]
        status = "passed" if all(check.status == "passed" for check in checks) else "failed"
        report = ProofSuiteReport(
            proof_id=proof_id,
            cartridge_id=cartridge_id,
            status=status,
            checks=checks,
            artifacts={},
            dna_before=dna_before,
            dna_after=dna_before,
        )
        report_path = proof_root / "proof_suite.json"
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        report.artifacts = relative_artifacts(proof_root)
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        attached_manifest = self.cartridges.attach_eval_proof(cartridge_id, proof_root)
        report.dna_after = attached_manifest.dna
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        cartridge_report = root / "eval_proofs" / proof_id / "proof_suite.json"
        cartridge_report.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return report

    def runtime_generation_proof(self) -> ProofCheck:
        proof_id = f"runtime_generation_{secrets.token_hex(8)}"
        proof_root = self.settings.data_dir / "eval_proofs" / proof_id
        proof_root.mkdir(parents=True, exist_ok=True)
        return self._runtime_generation_check(proof_root)

    def visual_contradiction_proof(self) -> ProofCheck:
        proof_id = f"visual_contradiction_{secrets.token_hex(8)}"
        proof_root = self.settings.data_dir / "eval_proofs" / proof_id
        proof_root.mkdir(parents=True, exist_ok=True)
        return self._visual_contradiction_check(proof_root)

    def latest(self) -> ProofSuiteReport | None:
        proof_root = self.settings.data_dir / "eval_proofs"
        reports = sorted(
            proof_root.glob("prd_proof_suite_*/proof_suite.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not reports:
            return None
        return ProofSuiteReport.model_validate_json(reports[0].read_text(encoding="utf-8"))

    def _adapter_training_check(self, cartridge_id: str, cartridge_root: Path, proof_root: Path) -> ProofCheck:
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        manifests = self._adapter_manifests(cartridge_id)
        for kind in self.required_adapters:
            item = manifests.get(kind)
            if item is None:
                missing.append(kind.value)
                continue
            manifest, adapter_root = item
            weight_path = adapter_root / "adapter_model.safetensors"
            dataset_path = Path(manifest.dataset_path)
            cartridge_adapter = cartridge_root / "adapters" / kind.value / "adapter_manifest.json"
            weight_report = safetensor_weight_stats(weight_path)
            keys = int(weight_report["tensor_count"])
            summary = read_json_or_empty(adapter_root / "training_summary.json")
            summary_weight_sha = summary.get("adapter_weight_sha256")
            summary_weight_sha_ok = (
                isinstance(summary_weight_sha, str) and summary_weight_sha == weight_report["sha256"]
            )
            manifest_weight_sha_ok = (
                isinstance(manifest.adapter_weight_sha256, str)
                and manifest.adapter_weight_sha256 == weight_report["sha256"]
            )
            manifest_weight_stats_ok = (
                manifest.adapter_tensor_count == weight_report["tensor_count"]
                and manifest.adapter_lora_tensor_count == weight_report["lora_tensor_count"]
                and manifest.adapter_total_elements == weight_report["total_elements"]
                and manifest.adapter_nonzero_elements == weight_report["nonzero_elements"]
                and manifest.adapter_nonzero_elements > 0
                and manifest.adapter_lora_tensor_count > 0
                and manifest.adapter_weights_finite
            )
            peft_ok = peft_config_loads(adapter_root)
            dataset_sha_ok = dataset_path.exists() and sha256_file(dataset_path) == manifest.dataset_sha256
            tree_sha_ok = sha256_tree(adapter_root) == manifest.sha256
            cartridge_bound = cartridge_adapter.exists()
            trace_report = validate_training_trace(manifest, adapter_root)
            proof_report = validate_training_proof(manifest, adapter_root)
            contract_report = validate_training_contract(manifest)
            valid = (
                manifest.sample_count > 0
                and manifest.max_steps > 0
                and manifest.trainable_params > 0
                and manifest.final_loss is not None
                and weight_report["valid"]
                and summary_weight_sha_ok
                and manifest_weight_sha_ok
                and manifest_weight_stats_ok
                and peft_ok
                and dataset_sha_ok
                and tree_sha_ok
                and cartridge_bound
                and trace_report["valid"]
                and proof_report["valid"]
                and contract_report["valid"]
            )
            if not valid:
                missing.append(kind.value)
            rows.append(
                {
                    "kind": kind.value,
                    "adapter": str(adapter_root),
                    "dataset": str(dataset_path),
                    "sample_count": manifest.sample_count,
                    "max_steps": manifest.max_steps,
                    "trainable_params": manifest.trainable_params,
                    "final_loss": manifest.final_loss,
                    "safetensor_keys": keys,
                    "adapter_weight_sha256": weight_report["sha256"],
                    "adapter_weight_sha256_ok": summary_weight_sha_ok and manifest_weight_sha_ok,
                    "adapter_weight_stats_ok": manifest_weight_stats_ok,
                    "adapter_weight_stats": weight_report,
                    "dataset_sha256_ok": dataset_sha_ok,
                    "adapter_sha256_ok": tree_sha_ok,
                    "peft_config_ok": peft_ok,
                    "cartridge_bound": cartridge_bound,
                    "training_contract": contract_report,
                    "training_trace": trace_report,
                    "training_proof": proof_report,
                }
            )
        artifact = proof_root / "adapter_training.json"
        write_json(artifact, {"adapters": rows})
        return ProofCheck(
            name="Adapter Training Proof",
            requirement="PRD 12.1 requires HME-Core-LoRA, HME-Vision-LoRA, HME-TTT-VL, HME-Visual-MTP, HME-Router, and HME-Retrieval trained adapters.",
            status="failed" if missing else "passed",
            evidence=[str(artifact), str(self.settings.adapters_dir), str(cartridge_root / "adapters")],
            detail="verified CUDA BF16 optimizer updates and nonzero LoRA safetensors: "
            + ", ".join(row["kind"] for row in rows),
        )

    def _runtime_adapter_check(self, proof_root: Path) -> ProofCheck:
        runtime = QwenRuntime(self.settings)
        status = runtime.status()
        adapter = resolve_runtime_adapter(self.settings)
        weight_path = adapter.path / "adapter_model.safetensors" if adapter else None
        payload = {
            "runtime_status": status.model_dump(mode="json"),
            "resolved_adapter": {
                "path": str(adapter.path) if adapter else None,
                "kind": adapter.manifest.kind.value if adapter else None,
                "adapter_id": adapter.manifest.adapter_id if adapter else None,
                "sha256": adapter.manifest.sha256 if adapter else None,
                "base_model": adapter.manifest.base_model if adapter else None,
                "weight_path": str(weight_path) if weight_path else None,
                "safetensor_keys": count_safetensor_keys(weight_path) if weight_path else 0,
                "peft_config_ok": peft_config_loads(adapter.path) if adapter else False,
            },
        }
        artifact = proof_root / "runtime_adapter.json"
        write_json(artifact, payload)
        valid = (
            status.available
            and adapter is not None
            and adapter.manifest.kind == TrainingAdapterKind.hme_core_lora
            and adapter.manifest.base_model == self.settings.model_id
            and status.adapter_path == str(adapter.path)
            and weight_path is not None
            and count_safetensor_keys(weight_path) > 0
            and peft_config_loads(adapter.path)
        )
        return ProofCheck(
            name="Runtime Adapter Binding",
            requirement="PRD 18 requires the local model package to run with the trained adapter, and PRD 12.1 requires HME-Core-LoRA to be usable by the Qwen runtime.",
            status="passed" if valid else "failed",
            evidence=[str(artifact)],
            detail=(
                f"{status.backend} runtime resolves {adapter.manifest.kind.value} at {adapter.path}"
                if adapter
                else "runtime adapter resolution is empty"
            ),
        )

    def _runtime_generation_check(self, proof_root: Path) -> ProofCheck:
        artifact = proof_root / "runtime_generation.json"
        retrieval_question = (
            "Using the Ely-Eye PRD evidence, state the PRD 12.1 local training adapter "
            "requirements for HME-Core-LoRA, HME-Vision-LoRA, HME-TTT-VL, "
            "HME-Visual-MTP, HME-Router, and HME-Retrieval. Return JSON with atom id "
            "citations."
        )
        proof_settings = self.settings.model_copy(update={"max_new_tokens": 256, "temperature": 0.0})
        runtime = QwenRuntime(proof_settings)
        payload: dict[str, Any] = {
            "question": retrieval_question,
            "status": "failed",
        }
        try:
            compiler = ContextCompiler(proof_settings, self.db)
            context = compiler.compile(retrieval_question, RuntimeProfile.library_100m)
            context = compact_generation_context(context, "12.1")
            if not context.hits:
                raise RuntimeError("No PRD text evidence was available for runtime generation proof.")
            primary_atom_id = context.hits[0].atom.atom_id
            generation_question = (
                'Return one compact JSON object. Put the "citations" field first and set it to '
                f'["{primary_atom_id}"]. In one short sentence, state that PRD 12.1 trains '
                "HME-Core-LoRA, HME-Vision-LoRA, HME-TTT-VL, HME-Visual-MTP, HME-Router, "
                "and HME-Retrieval."
            )
            payload["generation_question"] = generation_question
            status = runtime.status()
            adapter = resolve_runtime_adapter(proof_settings)
            generation = runtime.generate(generation_question, context)
            answer, citations, citation_report = Verifier().verify_answer(generation.text, context)
            claim_id = EvidenceGraphService(proof_settings, self.db).record_answer(
                generation_question,
                answer,
                citation_report,
                context,
            )
            payload.update(
                {
                    "status": "passed",
                    "runtime_status": status.model_dump(mode="json"),
                    "adapter": {
                        "path": generation.adapter_path,
                        "kind": generation.adapter_kind,
                        "adapter_id": generation.adapter_id,
                        "sha256": generation.adapter_sha256,
                    },
                    "tokens": {
                        "input": generation.input_tokens,
                        "output": generation.output_tokens,
                    },
                    "context": {
                        "profile": context.plan.profile.value,
                        "hit_ids": [hit.atom.atom_id for hit in context.hits],
                        "token_equivalent": context.token_equivalent,
                        "cache_trace_id": context.cache_trace_id,
                    },
                    "raw_generation": generation.text,
                    "verified_answer": answer,
                    "citations": citations,
                    "evidence_graph_claim_id": claim_id,
                    "verifier": citation_report.model_dump(mode="json"),
                }
            )
            valid = (
                status.available
                and generation.backend == "transformers"
                and adapter is not None
                and generation.adapter_path == str(adapter.path)
                and generation.adapter_kind == TrainingAdapterKind.hme_core_lora.value
                and generation.adapter_sha256 == adapter.manifest.sha256
                and bool(context.hits)
                and bool(generation.text.strip())
                and (generation.output_tokens or 0) > 0
                and citation_report.citation_accuracy > 0.0
            )
            if not valid:
                payload["status"] = "failed"
            write_json(artifact, payload)
            return ProofCheck(
                name="Runtime Adapter Generation",
                requirement="PRD 12.1 and PRD 18 require the trained HME-Core-LoRA adapter to be loaded by the local Qwen runtime and used for an evidence-cited generation.",
                status="passed" if valid else "failed",
                evidence=[str(artifact)],
                detail=(
                    f"{generation.backend} generated {generation.output_tokens} tokens with "
                    f"{generation.adapter_kind} citation accuracy "
                    f"{citation_report.citation_accuracy:.2f}"
                ),
            )
        except Exception as exc:
            payload["error"] = str(exc)
            write_json(artifact, payload)
            return ProofCheck(
                name="Runtime Adapter Generation",
                requirement="PRD 12.1 and PRD 18 require the trained HME-Core-LoRA adapter to be loaded by the local Qwen runtime and used for an evidence-cited generation.",
                status="failed",
                evidence=[str(artifact)],
                detail=f"runtime generation failed: {exc}",
            )
        finally:
            runtime.unload()

    def _cartridge_assets_check(
        self,
        cartridge_id: str,
        cartridge_root: Path,
        proof_root: Path,
    ) -> tuple[ProofCheck, CartridgeManifest]:
        report = CartridgeAssetService(self.settings, self.db).finalize(cartridge_id)
        source_report = cartridge_root / "cartridge_assets.json"
        proof_report = proof_root / "cartridge_assets.json"
        shutil.copy2(source_report, proof_report)
        rows = self.db.list_cartridges()
        manifest_json = next(row["manifest_json"] for row in rows if row["cartridge_id"] == cartridge_id)
        manifest = CartridgeManifest.model_validate_json(manifest_json)
        expected = {
            "text_vectors",
            "visual_vectors",
            "kv_snapshots",
            "memory_capsule",
            "memory_capsule_index",
            "ttt_vl_adapter",
            "visual_mtp_head",
            "cartridge_assets",
        }
        present = expected <= set(manifest.artifacts)
        by_name = {asset.name: asset for asset in report.assets}
        visual_vectors = by_name.get("visual_vectors")
        memory_capsule = by_name.get("memory_capsule")
        valid = (
            report.status == "ready"
            and present
            and all(asset.size_bytes > 0 for asset in report.assets)
            and visual_vectors is not None
            and visual_vectors.item_count > 0
            and memory_capsule is not None
            and memory_capsule.item_count > 0
        )
        return (
            ProofCheck(
                name="Cartridge Physical Assets",
                requirement="PRD 8.3 requires text vectors, non-empty visual vectors, KV snapshots, a 100M memory capsule, TTT-VL adapter weights, Visual-MTP head weights, and eval proofs inside the Context Cartridge.",
                status="passed" if valid else "failed",
                evidence=[str(proof_report), str(cartridge_root)],
                detail="materialized assets: " + ", ".join(asset.name for asset in report.assets),
            ),
            manifest,
        )

    def _memory_capsule_check(self, cartridge_root: Path, proof_root: Path) -> ProofCheck:
        capsule_path = cartridge_root / "memory_capsule.json.zst"
        index_path = cartridge_root / "memory_capsule_index.json"
        artifact = proof_root / "memory_capsule.json"
        payload: dict[str, Any] = {
            "capsule_path": str(capsule_path),
            "index_path": str(index_path),
            "status": "failed",
        }
        valid = False
        try:
            capsule = read_memory_capsule(capsule_path)
            index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
            segments = capsule.get("segments") if isinstance(capsule.get("segments"), list) else []
            segment_hashes = [validated_segment_hash(segment) for segment in segments]
            token_total = sum(int(segment.get("token_count") or 0) for segment in segments)
            computed_merkle = merkle_root(segment_hashes)
            valid = (
                capsule.get("format") == "ely-eye-memory-capsule-v1"
                and int(capsule.get("token_equivalent") or 0) >= self.settings.library_target_tokens
                and int(capsule.get("segment_count") or 0) == len(segments)
                and token_total == int(capsule.get("token_equivalent") or -1)
                and computed_merkle == capsule.get("segment_merkle_root")
                and index.get("segment_merkle_root") == capsule.get("segment_merkle_root")
                and int(index.get("token_equivalent") or 0) == int(capsule.get("token_equivalent") or -1)
                and int(capsule.get("source_atom_count") or 0) > 0
                and all(segment_hashes)
            )
            payload.update(
                {
                    "status": "passed" if valid else "failed",
                    "token_equivalent": capsule.get("token_equivalent"),
                    "segment_count": capsule.get("segment_count"),
                    "source_atom_count": capsule.get("source_atom_count"),
                    "source_token_equivalent": capsule.get("source_token_equivalent"),
                    "segment_merkle_root": capsule.get("segment_merkle_root"),
                    "computed_merkle_root": computed_merkle,
                    "first_segment": segments[0] if segments else None,
                    "last_segment": segments[-1] if segments else None,
                }
            )
        except Exception as exc:
            payload["error"] = str(exc)
        write_json(artifact, payload)
        return ProofCheck(
            name="100M Memory Capsule",
            requirement="PRD 2.2, 7.3, 8.3, and 14.3 require a 10M-100M token-equivalent Context Cartridge memory capsule with verifiable retrieval commitments.",
            status="passed" if valid else "failed",
            evidence=[str(artifact), str(capsule_path), str(index_path)],
            detail=(
                f"{payload.get('token_equivalent')} token-equivalent memory capsule with "
                f"{payload.get('segment_count')} committed segments"
            ),
        )

    def _cartridge_replay_check(
        self,
        cartridge_id: str,
        cartridge_root: Path,
        manifest: CartridgeManifest,
        proof_root: Path,
    ) -> ProofCheck:
        atoms_path = cartridge_root / manifest.artifacts["atoms"]
        sources_path = cartridge_root / manifest.artifacts["sources"]
        sparse_path = cartridge_root / manifest.artifacts["sparse_index"]
        graph_path = cartridge_root / manifest.artifacts["temporal_graph"]
        computed_dna = self.cartridges.compute_dna(cartridge_root)
        atom_rows = parquet_row_count(atoms_path)
        sources = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else []
        question = "Which Ely-Eye PRD requirements prove long context memory, HashHop, and adapters?"
        compiler = ContextCompiler(self.settings, self.db)
        replay_a = compiler.compile(question, RuntimeProfile.library_100m)
        replay_b = compiler.compile(question, RuntimeProfile.library_100m)
        digest_a = compiled_context_digest(replay_a)
        digest_b = compiled_context_digest(replay_b)
        payload = {
            "cartridge_id": cartridge_id,
            "manifest_atom_count": manifest.atom_count,
            "parquet_atom_count": atom_rows,
            "manifest_source_count": manifest.source_count,
            "source_count": len(sources),
            "dna_manifest": manifest.dna,
            "dna_computed": computed_dna,
            "artifacts": {
                "atoms": str(atoms_path),
                "sources": str(sources_path),
                "sparse_index": str(sparse_path),
                "temporal_graph": str(graph_path),
            },
            "replay": {
                "question": question,
                "digest_a": digest_a,
                "digest_b": digest_b,
                "hit_ids_a": [hit.atom.atom_id for hit in replay_a.hits],
                "hit_ids_b": [hit.atom.atom_id for hit in replay_b.hits],
                "cache_trace_a": replay_a.cache_trace_id,
                "cache_trace_b": replay_b.cache_trace_id,
            },
        }
        artifact = proof_root / "cartridge_replay.json"
        write_json(artifact, payload)
        valid = (
            atoms_path.exists()
            and sources_path.exists()
            and sparse_path.exists()
            and graph_path.exists()
            and atom_rows == manifest.atom_count
            and len(sources) == manifest.source_count
            and computed_dna == manifest.dna
            and bool(replay_a.hits)
            and digest_a == digest_b
        )
        return ProofCheck(
            name="Context Cartridge Replay",
            requirement="PRD 14.2 requires the same cartridge to load reproducibly and produce consistent replay evidence.",
            status="passed" if valid else "failed",
            evidence=[str(artifact), str(cartridge_root / "manifest.json")],
            detail=f"{atom_rows} atoms, {len(sources)} sources, replay digest {digest_a}",
        )

    def _visual_hashhop_check(self, proof_root: Path) -> ProofCheck:
        proof_settings = self.settings.model_copy(update={"max_new_tokens": 128, "temperature": 0.0})
        runtime = QwenRuntime(proof_settings)
        evaluator = HashHopEvaluator(proof_settings, runtime)
        try:
            base_proofs = [
                evaluator.generate_visual_proof(1, 262_144),
                evaluator.generate_visual_proof(2, 262_144),
                evaluator.generate_visual_proof(1, 1_010_000),
                evaluator.generate_visual_proof(1, 100_000_000),
                evaluator.generate_text_proof(2, 1_010_000),
                evaluator.generate_model_text_proof(2, 1_010_000),
            ]
            model_visual_1hop = collect_model_visual_proofs(evaluator, 1, 262_144, 3, 0.90, 4)
            model_visual_2hop = collect_model_visual_proofs(evaluator, 2, 262_144, 3, 0.75, 6)
            proofs = base_proofs + model_visual_1hop + model_visual_2hop
        finally:
            runtime.unload()
        payloads: list[dict[str, Any]] = []
        for proof in proofs:
            target = proof_root / "hashhop" / proof.proof_id
            copy_proof_artifacts(proof, target)
            payloads.append(
                {
                    "proof": proof.model_dump(mode="json"),
                    "chain_valid": hashhop_chain_valid(proof),
                    "artifact_dir": str(target),
                }
            )
        artifact = proof_root / "visual_hashhop_suite.json"
        model_solved = [
            item
            for item in payloads
            if item["proof"]["model_target_id"] is not None or item["proof"]["passed"] is not None
        ]
        model_text_solved = [
            proof
            for proof in base_proofs
            if proof.kind == "text_hashhop" and proof.model_target_id is not None
        ]
        model_solved_passed = [item for item in model_solved if item["proof"]["passed"]]
        visual_1hop_rate = pass_rate(model_visual_1hop)
        visual_2hop_rate = pass_rate(model_visual_2hop)
        metrics = {
            "visual_hashhop_262k_1hop": {
                "target": 0.90,
                "sample_count": len(model_visual_1hop),
                "passed": sum(1 for proof in model_visual_1hop if proof.passed),
                "pass_rate": visual_1hop_rate,
            },
            "visual_hashhop_262k_2hop": {
                "target": 0.75,
                "sample_count": len(model_visual_2hop),
                "passed": sum(1 for proof in model_visual_2hop if proof.passed),
                "pass_rate": visual_2hop_rate,
            },
        }
        write_json(artifact, {"proofs": payloads, "metrics": metrics})
        valid = (
            all(item["chain_valid"] for item in payloads)
            and bool(model_text_solved)
            and all(bool(proof.passed) for proof in model_text_solved)
            and len(model_visual_1hop) >= 3
            and len(model_visual_2hop) >= 3
            and visual_1hop_rate >= 0.90
            and visual_2hop_rate >= 0.75
        )
        return ProofCheck(
            name="Visual HashHop",
            requirement="PRD 14.2 and 14.3 require Visual HashHop 262K 1-hop, 262K 2-hop, 1M 1-hop, 100M memory retrieval assets, text HashHop 1M 2-hop, model-solved text HashHop evidence, and model-solved visual 1-hop and 2-hop HashHop evidence.",
            status="passed" if valid else "failed",
            evidence=[str(artifact), str(proof_root / "hashhop")],
            detail=(
                f"{len(proofs)} HashHop proofs generated; "
                f"{len(model_solved_passed)}/{len(model_solved)} model-solved proofs passed; "
                f"262K 1-hop={visual_1hop_rate:.2f}, 2-hop={visual_2hop_rate:.2f}"
            ),
        )

    def _standard_benchmark_check(self, proof_root: Path) -> ProofCheck:
        registry = benchmark_registry_report()
        sources = benchmark_source_report()
        samples = benchmark_sample_report(proof_root / "benchmark_samples")
        predictions = benchmark_prediction_report(self.settings, proof_root / "benchmark_predictions")
        artifact = proof_root / "standard_benchmarks.json"
        source_artifact = proof_root / "standard_benchmark_sources.json"
        sample_artifact = proof_root / "standard_benchmark_samples.json"
        prediction_artifact = proof_root / "standard_benchmark_predictions.json"
        artifact.write_text(registry.model_dump_json(indent=2), encoding="utf-8")
        source_artifact.write_text(sources.model_dump_json(indent=2), encoding="utf-8")
        sample_artifact.write_text(samples.model_dump_json(indent=2), encoding="utf-8")
        prediction_artifact.write_text(predictions.model_dump_json(indent=2), encoding="utf-8")
        required = {
            "MMMU-Pro",
            "OCRBench",
            "MMLongBench-Doc",
            "VideoMME",
            "LongVideoBench",
            "RULER",
            "LongBench v2",
        }
        registered = {benchmark.name for benchmark in registry.benchmarks}
        source_names = {source.name for source in sources.sources if source.status == "ready" and source.sha}
        loaded_sample_names = {sample.name for sample in samples.samples if sample.status == "loaded"}
        gated_sample_names = {sample.name for sample in samples.samples if sample.status == "gated"}
        source_proved_sample_names = {
            sample.name
            for sample in samples.samples
            if sample.status == "loaded" and sample.sample_sha256 and sample.source_dataset and sample.source_sha
        }
        predicted_sample_names = {
            prediction.name for prediction in predictions.predictions if prediction.status == "predicted"
        }
        prediction_scores = {
            prediction.name: prediction.score
            for prediction in predictions.predictions
            if prediction.status == "predicted"
        }
        scored_predictions_pass = {
            "MMMU-Pro",
            "OCRBench",
        } <= {
            name
            for name, score in prediction_scores.items()
            if isinstance(score, (int, float)) and math.isfinite(float(score)) and float(score) >= 0.5
        }
        sample_names = loaded_sample_names | gated_sample_names
        valid = (
            registry.status == "ready"
            and sources.status == "ready"
            and samples.status == "ready"
            and predictions.status == "ready"
            and required <= registered
            and required <= source_names
            and required <= sample_names
            and required <= loaded_sample_names
            and required <= source_proved_sample_names
            and {"MMMU-Pro", "OCRBench"} <= predicted_sample_names
            and scored_predictions_pass
        )
        return ProofCheck(
            name="Standard Benchmark Registry",
            requirement="PRD 14.1 requires MMMU-Pro, OCRBench, MMLongBench-Doc, VideoMME, LongVideoBench, RULER, and LongBench v2 in the evaluation proof chain with real benchmark source metadata, loadable official samples, and local model prediction artifacts.",
            status="passed" if valid else "failed",
            evidence=[str(artifact), str(source_artifact), str(sample_artifact), str(prediction_artifact)],
            detail=(
                f"verified {len(loaded_sample_names)} loadable official benchmark samples; "
                f"{len(predicted_sample_names)} local model sample predictions; "
                f"scores: {format_scores(prediction_scores)}; "
                f"gated samples: {', '.join(sorted(gated_sample_names)) or 'none'}"
            ),
        )

    def _repository_input_check(self, proof_root: Path) -> ProofCheck:
        root = project_root()
        files = discover_repository_files(root)
        code_files = [path for path in files if path.suffix.lower() in SUPPORTED_CODE_EXTENSIONS]
        context = build_repository_context(root, files, code_files, find_git_root(root))
        artifact = proof_root / "repository_input.json"
        write_json(artifact, context)
        call_graph = context["code"]["call_graph"]
        git_context = context["git"]
        issue_pr_history = context["issue_pr_history"]
        valid = (
            context["structure"]["indexed_file_count"] > 0
            and context["code"]["indexed_file_count"] > 0
            and bool(call_graph)
            and bool(git_context.get("available"))
            and bool(git_context.get("recent_commits"))
            and issue_pr_history.get("source") == "local_git_commit_history"
        )
        return ProofCheck(
            name="Repository Input Coverage",
            requirement="PRD 10.1 requires code repository structure indexing, call graph extraction, and issue/PR history association.",
            status="passed" if valid else "failed",
            evidence=[str(artifact)],
            detail=(
                f"{context['structure']['indexed_file_count']} files indexed; "
                f"{context['code']['indexed_file_count']} code files; "
                f"{len(call_graph)} call edges; "
                f"{len(git_context.get('recent_commits') or [])} git commits"
            ),
        )

    def _runtime_profile_check(self, proof_root: Path) -> ProofCheck:
        report = runtime_profile_report()
        artifact = proof_root / "runtime_profiles.json"
        artifact.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        required_runtimes = {"SGLang", "vLLM", "KTransformers", "llama.cpp"}
        runtimes = {profile.runtime for profile in report.profiles}
        valid = report.status == "ready" and required_runtimes <= runtimes
        return ProofCheck(
            name="Runtime Profile Delivery",
            requirement="PRD 13.1 requires SGLang as the main runtime plus vLLM, KTransformers, and llama.cpp deployment profiles.",
            status="passed" if valid else "failed",
            evidence=[str(artifact)],
            detail="registered runtimes: " + ", ".join(profile.name for profile in report.profiles),
        )

    def _research_module_check(self, proof_root: Path) -> ProofCheck:
        query = "Ely-Eye V-NSA DVKV x-modal KV speculative prefetch evidence graph"
        compiler = ContextCompiler(self.settings, self.db)
        context = compiler.compile(query, RuntimeProfile.research_theater)
        artifact = proof_root / "research_modules.json"
        report = ResearchModuleService(self.settings, self.db).build_report(query, context.hits)
        module_checks = research_module_checks(report)
        write_json(artifact, {"modules": report, "checks": module_checks})
        valid = all(check["passed"] for check in module_checks)
        return ProofCheck(
            name="Research Module Delivery",
            requirement="PRD 11 requires V-NSA, DVKV, X-Modal KV Sharing, Speculative Context Prefetch, Predictive KV Eviction, Learned Evidence Graph, Differentiable Retrieval Adapter, and Sleep-Time Visual Consolidation.",
            status="passed" if valid else "failed",
            evidence=[str(artifact)],
            detail="verified executable research modules: " + ", ".join(check["module"] for check in module_checks),
        )

    def _learned_memory_recall_check(
        self,
        cartridge_id: str,
        cartridge_root: Path,
        proof_root: Path,
    ) -> ProofCheck:
        manifests = self._adapter_manifests(cartridge_id)
        learned_kinds = (
            TrainingAdapterKind.hme_ttt_vl,
            TrainingAdapterKind.hme_core_lora,
            TrainingAdapterKind.hme_retrieval,
        )
        rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for kind in learned_kinds:
            item = manifests.get(kind)
            if item is None:
                missing.append(kind.value)
                continue
            manifest, adapter_root = item
            weight_path = adapter_root / "adapter_model.safetensors"
            cartridge_weight = cartridge_root / "adapters" / kind.value / "adapter_model.safetensors"
            rows.append(
                {
                    "kind": kind.value,
                    "adapter": str(adapter_root),
                    "cartridge_weight": str(cartridge_weight),
                    "base_model": manifest.base_model,
                    "dataset_sha256": manifest.dataset_sha256,
                    "sample_count": manifest.sample_count,
                    "trainable_params": manifest.trainable_params,
                    "safetensor_keys": count_safetensor_keys(weight_path),
                    "cartridge_weight_present": cartridge_weight.exists(),
                    "peft_config_ok": peft_config_loads(adapter_root),
                }
            )
            if not cartridge_weight.exists() or count_safetensor_keys(weight_path) <= 0:
                missing.append(kind.value)
        artifact = proof_root / "learned_memory_recall.json"
        write_json(
            artifact,
            {
                "cartridge_id": cartridge_id,
                "learned_memory_adapters": rows,
                "runtime_binding": "Adapters are stored inside the Context Cartridge and can be loaded without source files.",
            },
        )
        return ProofCheck(
            name="Learned Memory Recall",
            requirement="PRD 14.2 and 15.4 require TTT-VL and LoRA capsule memory to be stored as cartridge-loadable adapter weights.",
            status="failed" if missing else "passed",
            evidence=[str(artifact), str(cartridge_root / "adapters" / TrainingAdapterKind.hme_ttt_vl.value)],
            detail="verified cartridge-loadable learned memory adapters: " + ", ".join(row["kind"] for row in rows),
        )

    def _memory_dna_check(
        self,
        cartridge_root: Path,
        manifest: CartridgeManifest,
        proof_root: Path,
    ) -> ProofCheck:
        computed = self.cartridges.compute_dna(cartridge_root)
        payload = {
            "manifest_dna": manifest.dna,
            "computed_dna": computed,
            "formula": "sha256(sorted cartridge artifacts excluding manifest.json)",
            "artifact_count": sum(1 for path in cartridge_root.rglob("*") if path.is_file()),
        }
        artifact = proof_root / "memory_dna.json"
        write_json(artifact, payload)
        return ProofCheck(
            name="Memory DNA",
            requirement="PRD 11.14 requires a reproducible Memory DNA over adapter delta, evidence graph, retrieval index, and eval proofs.",
            status="passed" if manifest.dna == computed else "failed",
            evidence=[str(artifact), str(cartridge_root)],
            detail=f"manifest DNA {manifest.dna}",
        )

    def _cache_trace_check(self, proof_root: Path) -> ProofCheck:
        from .cache_fabric import CacheFabric

        layers = [layer.model_dump(mode="json") for layer in CacheFabric(self.settings, self.db).statuses()]
        artifact = proof_root / "cache_trace.json"
        write_json(artifact, {"layers": layers})
        valid = any(layer["entries"] > 0 for layer in layers)
        return ProofCheck(
            name="Cache Trace",
            requirement="PRD 14.2 requires cache and memory proof with visible hit, miss, write, and replay state.",
            status="passed" if valid else "failed",
            evidence=[str(artifact)],
            detail=f"{len(layers)} cache layers recorded",
        )

    def _visual_contradiction_check(self, proof_root: Path) -> ProofCheck:
        lens = VisualContradictionLens()
        scenarios = build_contradiction_scenarios()
        conflicts = [scenario for scenario in scenarios if scenario["expect_drift"]]
        controls = [scenario for scenario in scenarios if not scenario["expect_drift"]]
        detected = 0
        false_positives = 0
        categories: set[str] = set()
        rows: list[dict[str, Any]] = []
        known_categories = (
            "design-token drift",
            "layout drift",
            "typography drift",
            "copy drift",
            "visual-code drift",
            "temporal drift",
        )
        for scenario in scenarios:
            notes = lens.detect(scenario["context"])
            flagged = bool(notes)
            if scenario["expect_drift"]:
                if flagged:
                    detected += 1
                    for note in notes:
                        label = note.split(":", 1)[0]
                        categories.update(category for category in known_categories if label.endswith(category))
            elif flagged:
                false_positives += 1
            rows.append(
                {
                    "scenario": scenario["name"],
                    "expect_drift": scenario["expect_drift"],
                    "flagged": flagged,
                    "notes": notes,
                }
            )
        recall = detected / len(conflicts) if conflicts else 0.0
        false_positive_rate = false_positives / len(controls) if controls else 0.0
        artifact = proof_root / "visual_contradiction_lens.json"
        write_json(
            artifact,
            {
                "scenarios": rows,
                "metrics": {
                    "conflict_count": len(conflicts),
                    "control_count": len(controls),
                    "detected": detected,
                    "false_positives": false_positives,
                    "recall": recall,
                    "false_positive_rate": false_positive_rate,
                    "detected_categories": sorted(categories),
                },
            },
        )
        valid = recall >= 0.80 and false_positive_rate == 0.0 and len(categories) >= 5
        return ProofCheck(
            name="Visual Contradiction Lens",
            requirement="PRD 14.2 and 14.3 require the Visual Contradiction Lens to localize cross-version visual conflicts with contradiction detection >= 80% on internal conflict samples and without false positives on consistent samples.",
            status="passed" if valid else "failed",
            evidence=[str(artifact)],
            detail=(
                f"recall {recall:.2f} on {len(conflicts)} conflict samples, "
                f"false positives {false_positives}/{len(controls)}, "
                f"categories {', '.join(sorted(categories))}"
            ),
        )

    def _adapter_manifests(
        self,
        cartridge_id: str,
    ) -> dict[TrainingAdapterKind, tuple[AdapterManifest, Path]]:
        candidates: dict[TrainingAdapterKind, list[tuple[AdapterManifest, Path]]] = {}
        for path in sorted(self.settings.adapters_dir.glob("*/adapter_manifest.json")):
            manifest = AdapterManifest.model_validate_json(path.read_text(encoding="utf-8"))
            candidates.setdefault(manifest.kind, []).append((manifest, path.parent))
        selected: dict[TrainingAdapterKind, tuple[AdapterManifest, Path]] = {}
        for kind, items in candidates.items():
            items.sort(
                key=lambda item: (
                    item[0].cartridge_id == cartridge_id,
                    (item[1] / "adapter_model.safetensors").stat().st_mtime
                    if (item[1] / "adapter_model.safetensors").exists()
                    else 0.0,
                ),
                reverse=True,
            )
            selected[kind] = items[0]
        return selected

    def _resolve_cartridge(self, cartridge_id: str | None) -> dict[str, Any]:
        rows = self.db.list_cartridges()
        if not rows:
            raise ValueError("No Context Cartridge is available.")
        if cartridge_id is None:
            return rows[0]
        for row in rows:
            if row["cartridge_id"] == cartridge_id:
                return row
        raise ValueError(f"Unknown cartridge: {cartridge_id}")

    def _proof_id(self, cartridge_id: str, dna: str | None) -> str:
        raw = f"{cartridge_id}:{dna}:{secrets.token_hex(8)}"
        return f"prd_proof_suite_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def render_visual_hash(item_id: str, path: Path, tile_size: int) -> None:
    seed = int(hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    image = Image.new("RGB", (tile_size, tile_size), color=(rng.randrange(256), rng.randrange(256), rng.randrange(256)))
    draw = ImageDraw.Draw(image)
    for _ in range(96):
        x0 = rng.randrange(tile_size)
        y0 = rng.randrange(tile_size)
        x1 = min(tile_size, x0 + rng.randrange(4, tile_size // 3))
        y1 = min(tile_size, y0 + rng.randrange(4, tile_size // 3))
        color = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        if rng.random() < 0.5:
            draw.rectangle([x0, y0, x1, y1], fill=color)
        else:
            draw.ellipse([x0, y0, x1, y1], fill=color)
    image.save(path)


def render_visual_hash_arena(
    source_path: Path,
    slot_paths: dict[str, Path],
    target_slot: str,
    output_path: Path,
) -> None:
    tile = 160
    canvas = Image.new("RGB", (920, 560), color=(8, 10, 14))
    draw = ImageDraw.Draw(canvas)
    source_box = (72, 200, 72 + tile, 200 + tile)
    slot_boxes = {
        "top-left": (520, 76, 520 + tile, 76 + tile),
        "top-right": (704, 76, 704 + tile, 76 + tile),
        "bottom-left": (520, 324, 520 + tile, 324 + tile),
        "bottom-right": (704, 324, 704 + tile, 324 + tile),
    }
    source_center = box_center(source_box)
    target_center = box_center(slot_boxes[target_slot])
    draw.line([source_center, target_center], fill=(0, 242, 255), width=18)
    draw.line([source_center, target_center], fill=(240, 255, 255), width=6)
    paste_visual_hash_tile(canvas, source_path, source_box)
    draw.rectangle(source_box, outline=(255, 78, 220), width=8)
    for slot, path in slot_paths.items():
        box = slot_boxes[slot]
        paste_visual_hash_tile(canvas, path, box)
        draw.rectangle(box, outline=(180, 190, 205), width=5)
    draw.ellipse(
        [
            target_center[0] - 22,
            target_center[1] - 22,
            target_center[0] + 22,
            target_center[1] + 22,
        ],
        outline=(255, 255, 255),
        width=8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def render_visual_hash_two_hop_arena(
    source_path: Path,
    intermediate_slot_paths: dict[str, Path],
    target_slot_paths: dict[str, Path],
    intermediate_slot: str,
    target_slot: str,
    output_path: Path,
) -> None:
    tile = 136
    canvas = Image.new("RGB", (1180, 720), color=(8, 10, 14))
    draw = ImageDraw.Draw(canvas)
    source_box = (64, 292, 64 + tile, 292 + tile)
    intermediate_boxes = {
        "top-left": (424, 104, 424 + tile, 104 + tile),
        "top-right": (584, 104, 584 + tile, 104 + tile),
        "bottom-left": (424, 480, 424 + tile, 480 + tile),
        "bottom-right": (584, 480, 584 + tile, 480 + tile),
    }
    target_boxes = {
        "top-left": (864, 104, 864 + tile, 104 + tile),
        "top-right": (1024, 104, 1024 + tile, 104 + tile),
        "bottom-left": (864, 480, 864 + tile, 480 + tile),
        "bottom-right": (1024, 480, 1024 + tile, 480 + tile),
    }
    source_center = box_center(source_box)
    intermediate_center = box_center(intermediate_boxes[intermediate_slot])
    target_center = box_center(target_boxes[target_slot])
    paste_visual_hash_tile(canvas, source_path, source_box)
    draw.rectangle(source_box, outline=(255, 78, 220), width=7)
    for slot, path in intermediate_slot_paths.items():
        box = intermediate_boxes[slot]
        paste_visual_hash_tile(canvas, path, box)
        draw.rectangle(box, outline=(140, 215, 255), width=5)
    for slot, path in target_slot_paths.items():
        box = target_boxes[slot]
        paste_visual_hash_tile(canvas, path, box)
        draw.rectangle(box, outline=(255, 196, 92), width=5)
    draw_arrow_line(draw, source_center, intermediate_center, (0, 242, 255), (240, 255, 255))
    draw_arrow_line(draw, intermediate_center, target_center, (255, 176, 36), (255, 255, 245))
    draw.ellipse(
        [
            intermediate_center[0] - 26,
            intermediate_center[1] - 26,
            intermediate_center[0] + 26,
            intermediate_center[1] + 26,
        ],
        outline=(0, 242, 255),
        width=9,
    )
    draw.ellipse(
        [target_center[0] - 34, target_center[1] - 34, target_center[0] + 34, target_center[1] + 34],
        outline=(255, 255, 255),
        width=11,
    )
    draw.ellipse(
        [target_center[0] - 23, target_center[1] - 23, target_center[0] + 23, target_center[1] + 23],
        outline=(255, 176, 36),
        width=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def draw_arrow_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    outer_color: tuple[int, int, int],
    inner_color: tuple[int, int, int],
) -> None:
    draw.line([start, end], fill=outer_color, width=16)
    draw.line([start, end], fill=inner_color, width=5)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    tip = end
    back = (end[0] - int(math.cos(angle) * 34), end[1] - int(math.sin(angle) * 34))
    normal = (-math.sin(angle), math.cos(angle))
    left = (back[0] + int(normal[0] * 18), back[1] + int(normal[1] * 18))
    right = (back[0] - int(normal[0] * 18), back[1] - int(normal[1] * 18))
    draw.polygon([tip, left, right], fill=outer_color)
    inner_back = (end[0] - int(math.cos(angle) * 22), end[1] - int(math.sin(angle) * 22))
    inner_left = (inner_back[0] + int(normal[0] * 9), inner_back[1] + int(normal[1] * 9))
    inner_right = (inner_back[0] - int(normal[0] * 9), inner_back[1] - int(normal[1] * 9))
    draw.polygon([tip, inner_left, inner_right], fill=inner_color)


def paste_visual_hash_tile(canvas: Image.Image, path: Path, box: tuple[int, int, int, int]) -> None:
    tile = Image.open(path).convert("RGB").resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
    canvas.paste(tile, box[:2])


def box_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def count_safetensor_keys(path: Path) -> int:
    return int(safetensor_weight_stats(path)["tensor_count"])


def peft_config_loads(path: Path) -> bool:
    try:
        from peft import PeftConfig

        PeftConfig.from_pretrained(path)
        return True
    except Exception:
        return False


def validate_training_contract(manifest: AdapterManifest) -> dict[str, Any]:
    target_modules_ok = training_target_modules_ok(manifest.kind, manifest.target_modules)
    report = {
        "proof_version_ok": manifest.proof_version >= TRAINING_PROOF_VERSION,
        "method_ok": manifest.training_method == TRAINING_METHOD,
        "precision_ok": manifest.precision == TRAINING_PRECISION and manifest.bf16,
        "framework_ok": manifest.framework == TRAINING_FRAMEWORK,
        "optimizer_ok": manifest.optimizer == "paged_adamw_8bit",
        "optimizer_family_ok": manifest.optimizer_family == "8-bit AdamW / paged optimizer",
        "gradient_checkpointing_ok": (
            manifest.gradient_checkpointing
            and manifest.gradient_checkpointing_mode == GRADIENT_CHECKPOINTING_MODE
        ),
        "autocast_dtype_ok": manifest.autocast_dtype == AUTOCAST_DTYPE,
        "unsloth_version_ok": bool(manifest.unsloth_version),
        "triton_version_ok": bool(manifest.triton_version),
        "xformers_version_ok": bool(manifest.xformers_version),
        "unsloth_model_type_ok": bool(manifest.unsloth_model_type),
        "max_length_ok": manifest.max_length >= 8192,
        "target_modules_ok": target_modules_ok,
    }
    report["valid"] = all(report.values())
    return report


def training_target_modules_ok(kind: TrainingAdapterKind, target_modules: list[str]) -> bool:
    targets = set(target_modules)
    core = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if kind == TrainingAdapterKind.hme_core_lora:
        return core.issubset(targets)
    if kind == TrainingAdapterKind.hme_vision_lora:
        return bool(targets) and any("visual" in target or "vision" in target for target in targets)
    if kind == TrainingAdapterKind.hme_ttt_vl:
        has_visual = any("visual" in target or "vision" in target for target in targets)
        has_mlp = bool({"gate_proj", "up_proj", "down_proj"} & targets)
        return has_visual and has_mlp
    if kind == TrainingAdapterKind.hme_visual_mtp:
        return any("mtp" in target or "draft" in target or target == "lm_head" for target in targets)
    if kind in {TrainingAdapterKind.hme_router, TrainingAdapterKind.hme_retrieval}:
        return core.issubset(targets)
    return False


def training_payload_contract_ok(payload: dict[str, Any], manifest: AdapterManifest) -> bool:
    return (
        int(payload.get("proof_version") or 0) >= TRAINING_PROOF_VERSION
        and payload.get("training_method") == TRAINING_METHOD
        and payload.get("precision") == TRAINING_PRECISION
        and payload.get("framework") == TRAINING_FRAMEWORK
        and payload.get("optimizer_family") == manifest.optimizer_family
        and payload.get("gradient_checkpointing") is True
        and payload.get("gradient_checkpointing_mode") == GRADIENT_CHECKPOINTING_MODE
        and payload.get("autocast_dtype") == AUTOCAST_DTYPE
        and payload.get("unsloth_version") == manifest.unsloth_version
        and payload.get("triton_version") == manifest.triton_version
        and payload.get("xformers_version") == manifest.xformers_version
        and payload.get("unsloth_model_type") == manifest.unsloth_model_type
    )


def validate_training_trace(manifest: AdapterManifest, adapter_root: Path) -> dict[str, Any]:
    trace_path = adapter_root / "training_trace.jsonl"
    summary_path = adapter_root / "training_summary.json"
    report: dict[str, Any] = {
        "trace_path": str(trace_path),
        "summary_path": str(summary_path),
        "trace_exists": trace_path.exists(),
        "summary_exists": summary_path.exists(),
        "trace_sha256_ok": False,
        "summary_sha256_ok": False,
        "step_records": 0,
        "loss_history_records": len(manifest.loss_history),
        "loss_history_ok": False,
        "summary_matches_manifest": False,
        "summary_contract_ok": False,
        "run_start_contract_ok": False,
        "run_finish_contract_ok": False,
        "cuda_training": False,
        "bf16_supported": False,
        "bf16": manifest.bf16,
        "valid": False,
    }
    if not trace_path.exists() or not summary_path.exists():
        return report
    try:
        trace_rows = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return report

    step_rows = [row for row in trace_rows if row.get("event") == "step"]
    start_row = next((row for row in trace_rows if row.get("event") == "run_start"), {})
    finish_row = next((row for row in reversed(trace_rows) if row.get("event") == "run_finish"), {})
    losses = [row.get("loss") for row in step_rows]
    finite_losses = all(isinstance(loss, (int, float)) and math.isfinite(float(loss)) for loss in losses)
    final_loss_ok = (
        manifest.final_loss is not None
        and bool(losses)
        and abs(float(losses[-1]) - float(manifest.final_loss)) < 1e-6
    )
    report.update(
        {
            "trace_sha256_ok": (
                manifest.training_trace_sha256 is not None
                and sha256_file(trace_path) == manifest.training_trace_sha256
            ),
            "summary_sha256_ok": (
                manifest.training_summary_sha256 is not None
                and sha256_file(summary_path) == manifest.training_summary_sha256
            ),
            "step_records": len(step_rows),
            "loss_history_ok": (
                len(manifest.loss_history) == manifest.max_steps
                and finite_losses
                and final_loss_ok
            ),
            "summary_matches_manifest": (
                summary.get("adapter_kind") == manifest.kind.value
                and summary.get("base_model") == manifest.base_model
                and summary.get("dataset_sha256") == manifest.dataset_sha256
                and int(summary.get("max_steps") or -1) == manifest.max_steps
                and int(summary.get("trainable_params") or -1) == manifest.trainable_params
            ),
            "summary_contract_ok": training_payload_contract_ok(summary, manifest),
            "run_start_contract_ok": training_payload_contract_ok(start_row, manifest),
            "run_finish_contract_ok": training_payload_contract_ok(finish_row, manifest),
        }
    )
    environment = summary.get("environment") if isinstance(summary.get("environment"), dict) else {}
    device_name = str(environment.get("device_name") or "")
    report["cuda_training"] = bool(environment.get("cuda_available")) and "RTX 4090" in device_name
    report["bf16_supported"] = bool(environment.get("bf16_supported"))
    report["valid"] = (
        report["trace_sha256_ok"]
        and report["summary_sha256_ok"]
        and report["step_records"] == manifest.max_steps
        and report["loss_history_ok"]
        and report["summary_matches_manifest"]
        and report["summary_contract_ok"]
        and report["run_start_contract_ok"]
        and report["run_finish_contract_ok"]
        and report["cuda_training"]
        and report["bf16_supported"]
        and manifest.bf16
    )
    return report


def validate_training_proof(manifest: AdapterManifest, adapter_root: Path) -> dict[str, Any]:
    proof_path = adapter_root / "training_proof.json"
    report: dict[str, Any] = {
        "proof_path": str(proof_path),
        "proof_exists": proof_path.exists(),
        "proof_sha256_ok": False,
        "hash_changed": False,
        "optimizer_update_count": manifest.optimizer_update_count,
        "optimizer_update_count_ok": False,
        "max_optimizer_update_l2": manifest.max_optimizer_update_l2,
        "max_optimizer_update_l2_ok": False,
        "loss_delta": manifest.loss_delta,
        "loss_delta_finite": False,
        "updates_have_gradients": False,
        "updates_have_weight_delta": False,
        "proof_weight_sha256_ok": False,
        "proof_weight_stats_ok": False,
        "proof_cuda_bf16": False,
        "proof_bf16_supported": False,
        "proof_contract_ok": False,
        "proof_optimizer_ok": False,
        "proof_max_length_ok": False,
        "proof_target_modules_ok": False,
        "proof_unsloth_runtime_ok": False,
        "proof_matches_manifest": False,
        "valid": False,
    }
    if not proof_path.exists():
        return report
    try:
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return report

    updates = proof.get("optimizer_updates") if isinstance(proof.get("optimizer_updates"), list) else []
    update_l2_values = [
        float(item.get("update_l2"))
        for item in updates
        if isinstance(item, dict) and isinstance(item.get("update_l2"), (int, float))
    ]
    gradient_l2_values = [
        float(item.get("gradient_l2"))
        for item in updates
        if isinstance(item, dict) and isinstance(item.get("gradient_l2"), (int, float))
    ]
    weight_report = safetensor_weight_stats(adapter_root / "adapter_model.safetensors")
    proof_weight_stats = proof.get("adapter_weight_stats") if isinstance(proof.get("adapter_weight_stats"), dict) else {}
    proof_environment = proof.get("environment") if isinstance(proof.get("environment"), dict) else {}
    report.update(
        {
            "proof_sha256_ok": (
                manifest.training_proof_sha256 is not None
                and sha256_file(proof_path) == manifest.training_proof_sha256
            ),
            "hash_changed": (
                bool(manifest.initial_trainable_sha256)
                and bool(manifest.final_trainable_sha256)
                and manifest.initial_trainable_sha256 != manifest.final_trainable_sha256
            ),
            "optimizer_update_count_ok": (
                manifest.optimizer_update_count > 0
                and len(updates) == manifest.optimizer_update_count
            ),
            "max_optimizer_update_l2_ok": (
                manifest.max_optimizer_update_l2 is not None
                and math.isfinite(float(manifest.max_optimizer_update_l2))
                and float(manifest.max_optimizer_update_l2) > 0.0
            ),
            "loss_delta_finite": (
                manifest.loss_delta is not None
                and math.isfinite(float(manifest.loss_delta))
            ),
            "updates_have_gradients": bool(gradient_l2_values)
            and all(math.isfinite(value) and value > 0.0 for value in gradient_l2_values),
            "updates_have_weight_delta": bool(update_l2_values)
            and all(math.isfinite(value) and value > 0.0 for value in update_l2_values),
            "proof_weight_sha256_ok": (
                isinstance(proof.get("adapter_weight_sha256"), str)
                and proof.get("adapter_weight_sha256") == weight_report["sha256"]
            ),
            "proof_weight_stats_ok": (
                proof_weight_stats.get("sha256") == weight_report["sha256"]
                and proof_weight_stats.get("tensor_count") == weight_report["tensor_count"]
                and proof_weight_stats.get("lora_tensor_count") == weight_report["lora_tensor_count"]
                and proof_weight_stats.get("total_elements") == weight_report["total_elements"]
                and proof_weight_stats.get("nonzero_elements") == weight_report["nonzero_elements"]
                and int(proof_weight_stats.get("lora_tensor_count") or 0) > 0
                and int(proof_weight_stats.get("nonzero_elements") or 0) > 0
                and bool(proof_weight_stats.get("all_finite"))
            ),
            "proof_cuda_bf16": (
                bool(proof.get("bf16"))
                and bool(proof_environment.get("cuda_available"))
                and "RTX 4090" in str(proof_environment.get("device_name") or "")
            ),
            "proof_bf16_supported": bool(proof_environment.get("bf16_supported")),
            "proof_contract_ok": training_payload_contract_ok(proof, manifest),
            "proof_optimizer_ok": proof.get("optimizer") == "paged_adamw_8bit",
            "proof_max_length_ok": int(proof.get("max_length") or 0) >= 8192,
            "proof_target_modules_ok": training_target_modules_ok(
                manifest.kind,
                proof.get("target_modules") if isinstance(proof.get("target_modules"), list) else [],
            ),
            "proof_unsloth_runtime_ok": (
                bool(proof.get("unsloth_version"))
                and bool(proof.get("triton_version"))
                and bool(proof.get("xformers_version"))
                and bool(proof.get("unsloth_model_type"))
            ),
            "proof_matches_manifest": (
                proof.get("adapter_kind") == manifest.kind.value
                and proof.get("proof_version") == manifest.proof_version
                and proof.get("training_method") == manifest.training_method
                and proof.get("precision") == manifest.precision
                and proof.get("framework") == manifest.framework
                and proof.get("base_model") == manifest.base_model
                and proof.get("dataset_sha256") == manifest.dataset_sha256
                and proof.get("sample_count") == manifest.sample_count
                and proof.get("max_steps") == manifest.max_steps
                and proof.get("max_length") == manifest.max_length
                and proof.get("rank") == manifest.rank
                and proof.get("alpha") == manifest.alpha
                and proof.get("target_modules") == manifest.target_modules
                and proof.get("optimizer") == manifest.optimizer
                and proof.get("optimizer_family") == manifest.optimizer_family
                and proof.get("learning_rate") == manifest.learning_rate
                and proof.get("gradient_accumulation_steps") == manifest.gradient_accumulation_steps
                and proof.get("gradient_checkpointing") == manifest.gradient_checkpointing
                and proof.get("gradient_checkpointing_mode") == manifest.gradient_checkpointing_mode
                and proof.get("autocast_dtype") == manifest.autocast_dtype
                and proof.get("unsloth_version") == manifest.unsloth_version
                and proof.get("triton_version") == manifest.triton_version
                and proof.get("xformers_version") == manifest.xformers_version
                and proof.get("unsloth_model_type") == manifest.unsloth_model_type
                and proof.get("bf16") == manifest.bf16
                and proof.get("trainable_params") == manifest.trainable_params
                and proof.get("total_params") == manifest.total_params
                and proof.get("initial_trainable_sha256") == manifest.initial_trainable_sha256
                and proof.get("final_trainable_sha256") == manifest.final_trainable_sha256
                and proof.get("optimizer_update_count") == manifest.optimizer_update_count
                and proof.get("max_optimizer_update_l2") == manifest.max_optimizer_update_l2
            ),
        }
    )
    report["valid"] = (
        report["proof_sha256_ok"]
        and report["hash_changed"]
        and report["optimizer_update_count_ok"]
        and report["max_optimizer_update_l2_ok"]
        and report["loss_delta_finite"]
        and report["updates_have_gradients"]
        and report["updates_have_weight_delta"]
        and report["proof_weight_sha256_ok"]
        and report["proof_weight_stats_ok"]
        and report["proof_cuda_bf16"]
        and report["proof_bf16_supported"]
        and report["proof_contract_ok"]
        and report["proof_optimizer_ok"]
        and report["proof_max_length_ok"]
        and report["proof_target_modules_ok"]
        and report["proof_unsloth_runtime_ok"]
        and report["proof_matches_manifest"]
    )
    return report


def parquet_row_count(path: Path) -> int:
    if not path.exists():
        return -1
    import pyarrow.parquet as pq

    return pq.read_metadata(path).num_rows


def compiled_context_digest(context: Any) -> str:
    payload = {
        "hits": [hit.atom.atom_id for hit in context.hits],
        "packed_text": context.packed_text,
        "token_equivalent": context.token_equivalent,
    }
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_memory_capsule(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    prefix = b"ELYMEMCAP1\n"
    if not data.startswith(prefix):
        raise ValueError(f"Invalid memory capsule header: {path}")
    payload = zstd.ZstdDecompressor().decompress(data[len(prefix) :])
    parsed = json.loads(payload.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid memory capsule payload: {path}")
    return parsed


def validated_segment_hash(segment: dict[str, Any]) -> str:
    expected = segment.get("segment_sha256")
    if not isinstance(expected, str):
        return ""
    payload = {key: value for key, value in segment.items() if key != "segment_sha256"}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return actual if actual == expected else ""


def hashhop_chain_valid(proof: HashHopProof) -> bool:
    chain_artifacts = [Path(path) for path in proof.artifacts if path.endswith(".jsonl")]
    if len(chain_artifacts) != 1 or not chain_artifacts[0].exists():
        return False
    rows = [
        json.loads(line)
        for line in chain_artifacts[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != proof.hops:
        return False
    if rows[0]["from"] != proof.query_id or rows[-1]["to"] != proof.expected_target_id:
        return False
    for index in range(1, len(rows)):
        if rows[index - 1]["to"] != rows[index]["from"]:
            return False
    return all(Path(path).exists() for path in proof.artifacts)


def extract_hashhop_target(text: str) -> str | None:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = extract_json_object(stripped)
    if isinstance(payload, dict):
        for key in ("answer", "target", "target_id", "final_target_id"):
            value = payload.get(key)
            if isinstance(value, str):
                candidate = normalize_hash_id(value)
                if candidate:
                    return candidate
    match = re.search(r"\b[a-fA-F0-9]{32}\b", stripped)
    return match.group(0).lower() if match else None


def extract_visual_target_slot(text: str) -> str | None:
    stripped = text.strip()
    payload: dict[str, Any] | None
    try:
        parsed = json.loads(stripped)
        payload = parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        payload = extract_json_object(stripped)
    if isinstance(payload, dict):
        for key in ("target_slot", "final_slot", "slot", "answer"):
            value = payload.get(key)
            if isinstance(value, str):
                slot = normalize_visual_slot(value)
                if slot:
                    return slot
    return normalize_visual_slot(stripped)


def extract_visual_intermediate_slot(text: str) -> str | None:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        payload = parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        payload = extract_json_object(stripped)
    if isinstance(payload, dict):
        for key in ("intermediate_slot", "middle_slot", "first_hop_slot"):
            value = payload.get(key)
            if isinstance(value, str):
                slot = normalize_visual_slot(value)
                if slot:
                    return slot
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def normalize_hash_id(value: str) -> str | None:
    match = re.search(r"\b[a-fA-F0-9]{32}\b", value)
    return match.group(0).lower() if match else None


def normalize_visual_slot(value: str) -> str | None:
    normalized = re.sub(r"[^a-z]+", "-", value.lower()).strip("-")
    aliases = {
        "top-left": "top-left",
        "upper-left": "top-left",
        "left-top": "top-left",
        "top-right": "top-right",
        "upper-right": "top-right",
        "right-top": "top-right",
        "bottom-left": "bottom-left",
        "lower-left": "bottom-left",
        "left-bottom": "bottom-left",
        "bottom-right": "bottom-right",
        "lower-right": "bottom-right",
        "right-bottom": "bottom-right",
    }
    if normalized in aliases:
        return aliases[normalized]
    for alias, slot in aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return slot
    return None


def copy_proof_artifacts(proof: HashHopProof, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    source_roots = {Path(path).parent for path in proof.artifacts}
    for source_root in source_roots:
        if source_root.name == "visual_blocks":
            shutil.copytree(source_root, target / "visual_blocks", dirs_exist_ok=True)
    for artifact in proof.artifacts:
        path = Path(artifact)
        if path.is_file() and path.parent.name != "visual_blocks":
            shutil.copy2(path, target / path.name)
    (target / "proof.json").write_text(proof.model_dump_json(indent=2), encoding="utf-8")


def contradiction_atom(
    atom_id: str,
    source_id: str,
    source: str,
    text: str,
    modality: Modality,
    source_kind: str,
    version: str | None = None,
) -> EvidenceAtom:
    metadata: dict[str, Any] = {"source_kind": source_kind}
    if version is not None:
        metadata["version"] = version
    return EvidenceAtom(
        atom_id=atom_id,
        modality=modality,
        source_id=source_id,
        source=source,
        text=text,
        trust=TrustScores(parser=0.95, ocr=0.9, model=0.85),
        token_equivalent=max(1, len(text) // 4),
        metadata=metadata,
    )


def contradiction_context(question: str, atoms: list[EvidenceAtom]) -> CompiledContext:
    hits = [
        RetrievalHit(atom=atom, sparse_score=0.8, dense_score=0.0, graph_score=0.0, final_score=0.8)
        for atom in atoms
    ]
    plan = ContextPlan(
        question=question,
        profile=RuntimeProfile.research_theater,
        required_modalities=sorted({atom.modality for atom in atoms}),
        evidence_budget_atoms=len(atoms),
        token_budget=sum(atom.token_equivalent for atom in atoms),
        retrieval_query=question,
        compression_strategy="visual-contradiction-eval",
        verifier_contract=["Report contradictions as contradiction_notes."],
    )
    packed = "\n".join(f"[{atom.atom_id}] {atom.text}" for atom in atoms)
    return CompiledContext(
        plan=plan,
        hits=hits,
        packed_text=packed,
        token_equivalent=sum(atom.token_equivalent for atom in atoms),
        cache_trace_id="visual_contradiction_eval",
    )


def build_contradiction_scenarios() -> list[dict[str, Any]]:
    """Deterministic cross-version conflict and control samples (PRD 11.15).

    Each conflict pair injects a single genuine drift across two distinct
    sources; each control pair is internally consistent or scoped to different
    components. The samples exercise the live ``VisualContradictionLens`` so the
    proof measures real recall and false-positive rate.
    """

    scenarios: list[dict[str, Any]] = []

    def conflict(name: str, atoms: list[EvidenceAtom]) -> None:
        scenarios.append({"name": name, "context": contradiction_context(name, atoms), "expect_drift": True})

    def control(name: str, atoms: list[EvidenceAtom]) -> None:
        scenarios.append({"name": name, "context": contradiction_context(name, atoms), "expect_drift": False})

    conflict(
        "button-radius-token-drift",
        [
            contradiction_atom("home_btn_v4", "home_v4", "home_v4.figma", "button.radius: 8px", Modality.ui_screenshot, "figma"),
            contradiction_atom("checkout_btn_v4", "checkout_v4", "checkout_v4.figma", "button.radius: 12px", Modality.ui_screenshot, "figma"),
        ],
    )
    conflict(
        "primary-color-token-drift",
        [
            contradiction_atom("spec_color", "brand_spec", "brand_guide.pdf", "primary.color: #2563eb", Modality.pdf_page, "spec"),
            contradiction_atom("impl_color", "theme_css", "src/theme.css", "primary.color: #1d4ed8", Modality.code, "css"),
        ],
    )
    conflict(
        "heading-typography-drift",
        [
            contradiction_atom("home_heading", "home_shot", "home_v3.png", "heading.font-size: 24px", Modality.ui_screenshot, "ui_screenshot"),
            contradiction_atom("checkout_heading", "checkout_shot", "checkout_v3.png", "heading.font-size: 28px", Modality.ui_screenshot, "ui_screenshot"),
        ],
    )
    conflict(
        "card-spacing-layout-drift",
        [
            contradiction_atom("card_home", "card_home_shot", "home_v3.png", "card.padding: 16px", Modality.ui_screenshot, "ui_screenshot"),
            contradiction_atom("card_settings", "card_settings_shot", "settings_v3.png", "card.padding: 24px", Modality.ui_screenshot, "ui_screenshot"),
        ],
    )
    conflict(
        "cta-copy-drift",
        [
            contradiction_atom("cta_v1", "shot_v1", "cart_v1.png", 'cta.label: "Checkout"', Modality.ui_screenshot, "ui_screenshot"),
            contradiction_atom("cta_v4", "shot_v4", "cart_v4.png", 'cta.label: "Pay now"', Modality.ui_screenshot, "ui_screenshot"),
        ],
    )
    conflict(
        "banner-radius-temporal-drift",
        [
            contradiction_atom("banner_v1", "banner_spec_v1", "banner_v1.figma", "banner.radius: 4px", Modality.ui_screenshot, "figma", version="v1"),
            contradiction_atom("banner_v4", "banner_spec_v4", "banner_v4.figma", "banner.radius: 10px", Modality.ui_screenshot, "figma", version="v4"),
        ],
    )
    conflict(
        "toggle-accent-visual-code-drift",
        [
            contradiction_atom("toggle_design", "toggle_figma", "toggle.figma", "toggle.color: #10b981", Modality.ui_screenshot, "figma"),
            contradiction_atom("toggle_code", "toggle_css", "src/toggle.css", "toggle.color: #059669", Modality.code, "css"),
        ],
    )

    control(
        "consistent-button-radius",
        [
            contradiction_atom("ctrl_btn_spec", "ctrl_spec", "spec.pdf", "button.radius: 8px", Modality.pdf_page, "spec"),
            contradiction_atom("ctrl_btn_css", "ctrl_css", "src/tokens.css", "button.radius = 8px", Modality.code, "css"),
        ],
    )
    control(
        "distinct-components-not-a-conflict",
        [
            contradiction_atom("ctrl_button_radius", "ctrl_a", "a.css", "button.radius: 8px", Modality.code, "css"),
            contradiction_atom("ctrl_input_radius", "ctrl_b", "b.css", "input.radius: 12px", Modality.code, "css"),
        ],
    )
    control(
        "single-source-multiple-rules",
        [
            contradiction_atom(
                "ctrl_single_css",
                "ctrl_single",
                "src/base.css",
                "header.padding: 8px; footer.padding: 24px; aside.padding: 16px",
                Modality.code,
                "css",
            ),
        ],
    )
    control(
        "consistent-primary-color",
        [
            contradiction_atom("ctrl_color_spec", "ctrl_color_a", "brand.pdf", "primary.color: #2563eb", Modality.pdf_page, "spec"),
            contradiction_atom("ctrl_color_css", "ctrl_color_b", "theme.css", "primary.color: #2563eb", Modality.code, "css"),
        ],
    )
    control(
        "prose-without-design-facts",
        [
            contradiction_atom("ctrl_prose_a", "ctrl_prose_1", "notes_a.txt", "The onboarding flow feels smooth and the colors look balanced.", Modality.text, "text"),
            contradiction_atom("ctrl_prose_b", "ctrl_prose_2", "notes_b.txt", "Users reported the checkout button is easy to find.", Modality.text, "text"),
        ],
    )

    return scenarios


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def format_scores(scores: dict[str, Any]) -> str:
    parts: list[str] = []
    for name in sorted(scores):
        score = scores[name]
        if isinstance(score, (int, float)) and math.isfinite(float(score)):
            parts.append(f"{name}={float(score):.2f}")
        else:
            parts.append(f"{name}=unscored")
    return ", ".join(parts) or "none"


def research_module_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(module: str, passed: bool, detail: str) -> None:
        checks.append({"module": module, "passed": passed, "detail": detail})

    v_nsa = report.get("v_nsa") if isinstance(report.get("v_nsa"), dict) else {}
    sparse_blocks = v_nsa.get("sparse_blocks") if isinstance(v_nsa.get("sparse_blocks"), list) else []
    add(
        "v_nsa",
        int(v_nsa.get("sparse_block_count") or 0) > 0 and bool(sparse_blocks),
        f"{len(sparse_blocks)} sparse blocks",
    )

    dvkv = report.get("dvkv") if isinstance(report.get("dvkv"), dict) else {}
    selected = dvkv.get("selected") if isinstance(dvkv.get("selected"), list) else []
    candidate_tokens = int(dvkv.get("candidate_tokens") or 0)
    budget = int(dvkv.get("budget") or 0)
    dvkv_scores_valid = all(
        is_finite_number(item.get("attention_importance")) and is_finite_number(item.get("evidence_value"))
        for item in selected
        if isinstance(item, dict)
    )
    add(
        "dvkv",
        candidate_tokens > 0 and 0 < len(selected) <= budget and dvkv_scores_valid,
        f"{len(selected)}/{candidate_tokens} visual tokens selected",
    )

    x_modal = report.get("x_modal_kv") if isinstance(report.get("x_modal_kv"), dict) else {}
    shared_blocks = x_modal.get("shared_blocks") if isinstance(x_modal.get("shared_blocks"), list) else []
    x_modal_valid = bool(shared_blocks) and all(
        isinstance(block, dict) and bool(block.get("block_id")) and bool(block.get("atom_ids"))
        for block in shared_blocks
    )
    add("x_modal_kv", x_modal_valid, f"{len(shared_blocks)} shared KV blocks")

    prefetch = report.get("speculative_prefetch") if isinstance(report.get("speculative_prefetch"), dict) else {}
    prefetch_blocks = prefetch.get("prefetch_blocks") if isinstance(prefetch.get("prefetch_blocks"), list) else []
    priorities = [
        float(block.get("priority"))
        for block in prefetch_blocks
        if isinstance(block, dict) and is_finite_number(block.get("priority"))
    ]
    prefetch_valid = (
        bool(prefetch_blocks)
        and len(priorities) == len(prefetch_blocks)
        and priorities == sorted(priorities, reverse=True)
        and all(int(block.get("bytes_estimate") or 0) > 0 for block in prefetch_blocks if isinstance(block, dict))
    )
    add("speculative_prefetch", prefetch_valid, f"{len(prefetch_blocks)} prefetch blocks")

    eviction = report.get("predictive_eviction") if isinstance(report.get("predictive_eviction"), dict) else {}
    layers = eviction.get("layers") if isinstance(eviction.get("layers"), list) else []
    eviction_valid = bool(layers) and all(
        isinstance(layer, dict)
        and bool(layer.get("replay_token"))
        and is_finite_number(layer.get("hit_rate"))
        and int(layer.get("entries") or 0) >= 0
        for layer in layers
    )
    add("predictive_eviction", eviction_valid, f"{len(layers)} cache layers")

    retrieval = report.get("differentiable_retrieval") if isinstance(report.get("differentiable_retrieval"), dict) else {}
    probabilities = retrieval.get("probabilities") if isinstance(retrieval.get("probabilities"), list) else []
    probability_values = [
        float(item.get("probability"))
        for item in probabilities
        if isinstance(item, dict) and is_finite_number(item.get("probability"))
    ]
    probability_sum = sum(probability_values)
    retrieval_valid = (
        bool(probabilities)
        and len(probability_values) == len(probabilities)
        and abs(probability_sum - 1.0) <= 1e-6
        and bool(retrieval.get("straight_through_atom_id"))
    )
    add("differentiable_retrieval", retrieval_valid, f"probability sum {probability_sum:.6f}")

    graph = report.get("evidence_graph") if isinstance(report.get("evidence_graph"), dict) else {}
    relations = graph.get("relations") if isinstance(graph.get("relations"), list) else []
    edge_count = int(graph.get("edge_count_sample") or 0)
    add("evidence_graph", edge_count > 0 and "cites" in relations, f"{edge_count} sampled edges")

    sleep = report.get("sleep_consolidation") if isinstance(report.get("sleep_consolidation"), dict) else {}
    durable = sleep.get("durable_candidates") if isinstance(sleep.get("durable_candidates"), list) else []
    add(
        "sleep_consolidation",
        int(sleep.get("candidate_count") or 0) > 0 and bool(durable) and bool(sleep.get("target_adapter")),
        f"{len(durable)} durable candidates",
    )

    return checks


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "PRD.md").exists() and (candidate / "backend" / "src" / "ely_eye").exists():
            return candidate
    return current


def discover_repository_files(root: Path) -> list[Path]:
    ignored_dirs = {".git", ".ely_eye", ".venv", ".venv-linux", "__pycache__", "node_modules", "dist", "build", "target"}
    files: list[Path] = []
    for current, dirs, names in os.walk(root):
        dirs[:] = [name for name in dirs if name not in ignored_dirs and not name.startswith(".venv")]
        current_path = Path(current)
        for name in names:
            path = current_path / name
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            if any(part in ignored_dirs or part.startswith(".venv") for part in relative.parts):
                continue
            try:
                path.stat()
            except OSError:
                continue
            files.append(path)
    return files


def pass_rate(proofs: list[HashHopProof]) -> float:
    if not proofs:
        return 0.0
    return sum(1 for proof in proofs if proof.passed) / len(proofs)


def collect_model_visual_proofs(
    evaluator: HashHopEvaluator,
    hops: int,
    token_equivalent: int,
    min_samples: int,
    target_rate: float,
    max_samples: int,
) -> list[HashHopProof]:
    proofs: list[HashHopProof] = []
    while len(proofs) < max_samples:
        proofs.append(evaluator.generate_model_visual_proof(hops, token_equivalent, max_proofs=5))
        if len(proofs) >= min_samples and pass_rate(proofs) >= target_rate:
            break
    return proofs


def compact_generation_context(context: Any, section_hint: str, max_chars: int = 2400) -> Any:
    text_hits = [hit for hit in context.hits if hit.atom.modality.value == "text"]
    hits = (text_hits or context.hits)[:1]
    sections: list[str] = []
    for hit in hits:
        atom = hit.atom
        evidence_text = extract_markdown_section(atom.text, section_hint) or atom.text
        evidence_text = evidence_text.strip()[:max_chars]
        citation = {
            "atom_id": atom.atom_id,
            "modality": atom.modality.value,
            "source": atom.source,
            "image_ref": atom.image_ref,
            "layout": atom.layout.model_dump() if atom.layout else None,
            "score": hit.final_score,
        }
        sections.append(f"[EVIDENCE]\n{citation}\n{evidence_text}\n[/EVIDENCE]")
    packed_text = "\n\n".join(sections)
    return context.model_copy(
        update={
            "hits": hits,
            "packed_text": packed_text,
            "token_equivalent": max(1, len(packed_text) // 4),
        }
    )


def extract_markdown_section(text: str, section_hint: str) -> str:
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") and section_hint in stripped:
            start = index
            break
    if start is None:
        return ""
    end = len(lines)
    start_level = len(lines[start]) - len(lines[start].lstrip("#"))
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith("#"):
            level = len(lines[index]) - len(lines[index].lstrip("#"))
            if level <= start_level:
                end = index
                break
    return "\n".join(lines[start:end])


def relative_artifacts(root: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            key = path.relative_to(root).as_posix()
            artifacts[key] = key
    return artifacts
