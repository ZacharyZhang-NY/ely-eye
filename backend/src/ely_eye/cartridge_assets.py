from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import zstandard as zstd
from PIL import Image

from .cartridge import CartridgeService
from .config import Settings, get_settings
from .db import Database
from .schemas import CartridgeAssetReport, CartridgeAssetStatus
from .storage import ObjectStore, sha256_file


TEXT_VECTOR_DIMS = 384
VISUAL_VECTOR_DIMS = 256
MEMORY_CAPSULE_SEGMENTS = 4096


class CartridgeAssetService:
    def __init__(self, settings: Settings | None = None, database: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = database or Database(self.settings)
        self.cartridges = CartridgeService(self.settings, self.db)
        self.objects = ObjectStore(self.settings)

    def finalize(self, cartridge_id: str) -> CartridgeAssetReport:
        row = self._cartridge_row(cartridge_id)
        root = Path(row["root_path"])
        atoms = load_cartridge_atoms(root / "atoms.parquet")

        text_status = self._write_text_vectors(root, atoms)
        visual_status = self._write_visual_vectors(root, atoms)
        kv_status = self._write_kv_snapshot(root, atoms)
        memory_status = self._write_memory_capsule(root, atoms)
        ttt_status = self._link_adapter_weight(
            root,
            "ttt_vl_adapter.safetensors",
            root / "adapters" / "hme_ttt_vl" / "adapter_model.safetensors",
        )
        mtp_status = self._link_adapter_weight(
            root,
            "visual_mtp_head.safetensors",
            root / "adapters" / "hme_visual_mtp" / "adapter_model.safetensors",
        )
        report = CartridgeAssetReport(
            cartridge_id=cartridge_id,
            status="ready",
            assets=[text_status, visual_status, kv_status, memory_status, ttt_status, mtp_status],
        )
        if any(asset.status != "ready" for asset in report.assets):
            report.status = "incomplete"
        report_path = root / "cartridge_assets.json"
        report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        self.cartridges.update_artifacts(
            cartridge_id,
            {
                "text_vectors": "text_vectors.fp16.zstd",
                "visual_vectors": "visual_vectors.int8.zstd",
                "kv_snapshots": "kv_snapshots.kivi",
                "memory_capsule": "memory_capsule.json.zst",
                "memory_capsule_index": "memory_capsule_index.json",
                "ttt_vl_adapter": "ttt_vl_adapter.safetensors",
                "visual_mtp_head": "visual_mtp_head.safetensors",
                "cartridge_assets": "cartridge_assets.json",
            },
            token_equivalent=max(int(row["token_equivalent"]), self.settings.library_target_tokens),
        )
        return report

    def status(self, cartridge_id: str) -> CartridgeAssetReport | None:
        row = self._cartridge_row(cartridge_id)
        path = Path(row["root_path"]) / "cartridge_assets.json"
        if not path.exists():
            return None
        return CartridgeAssetReport.model_validate_json(path.read_text(encoding="utf-8"))

    def _write_text_vectors(self, root: Path, atoms: list[dict[str, Any]]) -> CartridgeAssetStatus:
        text_atoms = [atom for atom in atoms if str(atom.get("text") or "").strip()]
        vectors = np.zeros((len(text_atoms), TEXT_VECTOR_DIMS), dtype=np.float16)
        for index, atom in enumerate(text_atoms):
            vectors[index] = hashed_text_vector(str(atom.get("text") or ""), TEXT_VECTOR_DIMS).astype(np.float16)
        path = root / "text_vectors.fp16.zstd"
        write_compressed_npz(
            path,
            {
                "atom_ids": np.array([atom["atom_id"] for atom in text_atoms]),
                "vectors": vectors,
                "metadata": np.array(
                    json.dumps(
                        {
                            "format": "ely-eye-text-vectors-v1",
                            "dtype": "float16",
                            "dims": TEXT_VECTOR_DIMS,
                            "vectorizer": "signed feature hashing over parsed cartridge text",
                        },
                        separators=(",", ":"),
                    )
                ),
            },
        )
        return asset_status("text_vectors", path, len(text_atoms), "float16")

    def _write_visual_vectors(self, root: Path, atoms: list[dict[str, Any]]) -> CartridgeAssetStatus:
        visual_atoms = [atom for atom in atoms if atom.get("image_ref")]
        vectors = np.zeros((len(visual_atoms), VISUAL_VECTOR_DIMS), dtype=np.int8)
        for index, atom in enumerate(visual_atoms):
            image_path = self.objects.resolve(str(atom["image_ref"]))
            vectors[index] = visual_image_vector(image_path, VISUAL_VECTOR_DIMS)
        path = root / "visual_vectors.int8.zstd"
        write_compressed_npz(
            path,
            {
                "atom_ids": np.array([atom["atom_id"] for atom in visual_atoms]),
                "vectors": vectors,
                "metadata": np.array(
                    json.dumps(
                        {
                            "format": "ely-eye-visual-vectors-v1",
                            "dtype": "int8",
                            "dims": VISUAL_VECTOR_DIMS,
                            "vectorizer": "quantized luminance patch statistics over cartridge images",
                        },
                        separators=(",", ":"),
                    )
                ),
            },
        )
        return asset_status("visual_vectors", path, len(visual_atoms), "int8")

    def _write_kv_snapshot(self, root: Path, atoms: list[dict[str, Any]]) -> CartridgeAssetStatus:
        token_start = 0
        blocks: list[dict[str, Any]] = []
        for atom in atoms:
            token_count = int(atom.get("token_equivalent") or 0)
            text = str(atom.get("text") or "")
            blocks.append(
                {
                    "atom_id": atom["atom_id"],
                    "source": atom.get("source"),
                    "modality": atom.get("modality"),
                    "token_start": token_start,
                    "token_count": token_count,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "image_ref": atom.get("image_ref"),
                }
            )
            token_start += token_count
        payload = {
            "format": "ely-eye-kivi-replay-v1",
            "block_count": len(blocks),
            "token_equivalent": token_start,
            "blocks": blocks,
        }
        path = root / "kv_snapshots.kivi"
        compressed = zstd.ZstdCompressor(level=9).compress(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        path.write_bytes(b"ELYKIVI1\n" + compressed)
        return asset_status("kv_snapshots", path, len(blocks), "zstd-json")

    def _write_memory_capsule(self, root: Path, atoms: list[dict[str, Any]]) -> CartridgeAssetStatus:
        if not atoms:
            raise ValueError("Memory capsule materialization requires at least one cartridge atom")
        target_tokens = self.settings.library_target_tokens
        source_payload = json.dumps(
            [
                {
                    "atom_id": atom["atom_id"],
                    "source": atom.get("source"),
                    "modality": atom.get("modality"),
                    "token_equivalent": atom.get("token_equivalent"),
                    "text_sha256": hashlib.sha256(str(atom.get("text") or "").encode("utf-8")).hexdigest(),
                    "image_ref": atom.get("image_ref"),
                }
                for atom in atoms
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        seed = hashlib.sha256(source_payload.encode("utf-8")).hexdigest()
        base_tokens = target_tokens // MEMORY_CAPSULE_SEGMENTS
        remainder = target_tokens % MEMORY_CAPSULE_SEGMENTS
        token_start = 0
        segments: list[dict[str, Any]] = []
        segment_hashes: list[str] = []
        for index in range(MEMORY_CAPSULE_SEGMENTS):
            atom = atoms[index % len(atoms)]
            token_count = base_tokens + (1 if index < remainder else 0)
            segment = {
                "segment_id": capsule_hash(seed, index, "segment"),
                "source_atom_id": atom["atom_id"],
                "token_start": token_start,
                "token_count": token_count,
                "hashhop_query_id": capsule_hash(seed, index, "query")[:32],
                "hashhop_target_id": capsule_hash(seed, index, "target")[:32],
            }
            segment_hash = hashlib.sha256(
                json.dumps(segment, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            segment["segment_sha256"] = segment_hash
            segments.append(segment)
            segment_hashes.append(segment_hash)
            token_start += token_count
        capsule = {
            "format": "ely-eye-memory-capsule-v1",
            "token_equivalent": target_tokens,
            "segment_count": MEMORY_CAPSULE_SEGMENTS,
            "source_atom_count": len(atoms),
            "source_token_equivalent": sum(int(atom.get("token_equivalent") or 0) for atom in atoms),
            "source_payload_sha256": seed,
            "segment_merkle_root": merkle_root(segment_hashes),
            "segments": segments,
        }
        data = json.dumps(capsule, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        path = root / "memory_capsule.json.zst"
        path.write_bytes(b"ELYMEMCAP1\n" + zstd.ZstdCompressor(level=9).compress(data))
        index_path = root / "memory_capsule_index.json"
        index_path.write_text(
            json.dumps(
                {
                    key: capsule[key]
                    for key in (
                        "format",
                        "token_equivalent",
                        "segment_count",
                        "source_atom_count",
                        "source_token_equivalent",
                        "source_payload_sha256",
                        "segment_merkle_root",
                    )
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return asset_status("memory_capsule", path, MEMORY_CAPSULE_SEGMENTS, "zstd-json")

    def _link_adapter_weight(self, root: Path, name: str, source: Path) -> CartridgeAssetStatus:
        target = root / name
        if source.exists():
            if target.exists() and sha256_file(target) != sha256_file(source):
                target.unlink()
            if not target.exists():
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
        return asset_status(name.removesuffix(".safetensors"), target, 1 if target.exists() else 0, "safetensors")

    def _cartridge_row(self, cartridge_id: str) -> dict[str, Any]:
        for row in self.db.list_cartridges():
            if row["cartridge_id"] == cartridge_id:
                return row
        raise ValueError(f"Unknown cartridge: {cartridge_id}")


def load_cartridge_atoms(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pylist()


def hashed_text_vector(text: str, dims: int) -> np.ndarray:
    vector = np.zeros(dims, dtype=np.float32)
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dims
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def visual_image_vector(path: Path, dims: int) -> np.ndarray:
    side = int(np.sqrt(dims))
    if side * side != dims:
        raise ValueError("visual vector dimensions must be a square number")
    with Image.open(path) as image:
        gray = image.convert("L").resize((side, side))
    values = np.asarray(gray, dtype=np.float32).reshape(-1)
    values = (values - 127.5) / 127.5
    return np.clip(np.rint(values * 127), -127, 127).astype(np.int8)


def capsule_hash(seed: str, index: int, label: str) -> str:
    return hashlib.sha256(f"{seed}:{index}:{label}".encode("utf-8")).hexdigest()


def merkle_root(hashes: list[str]) -> str:
    if not hashes:
        return hashlib.sha256(b"").hexdigest()
    level = hashes[:]
    while len(level) > 1:
        next_level: list[str] = []
        for index in range(0, len(level), 2):
            left = level[index]
            right = level[index + 1] if index + 1 < len(level) else left
            next_level.append(hashlib.sha256(f"{left}{right}".encode("ascii")).hexdigest())
        level = next_level
    return level[0]


def write_compressed_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    compressed = zstd.ZstdCompressor(level=9).compress(buffer.getvalue())
    path.write_bytes(compressed)


def asset_status(name: str, path: Path, item_count: int, dtype: str) -> CartridgeAssetStatus:
    exists = path.exists()
    return CartridgeAssetStatus(
        name=name,
        path=str(path),
        sha256=sha256_file(path) if exists else "",
        size_bytes=path.stat().st_size if exists else 0,
        item_count=item_count,
        dtype=dtype,
        status="ready" if exists else "missing",
    )
