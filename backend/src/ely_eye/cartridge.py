from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

from .config import Settings, get_settings
from .db import Database
from .schemas import AdapterManifest, CartridgeManifest, EvidenceAtom, SourceRecord


class CartridgeService:
    def __init__(self, settings: Settings | None = None, database: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = database or Database(self.settings)

    def materialize(
        self,
        name: str,
        atoms: list[EvidenceAtom],
        sources: list[SourceRecord],
    ) -> CartridgeManifest:
        created_at = datetime.now(timezone.utc)
        cartridge_id = self._cartridge_id(name, created_at)
        root = self.settings.cartridge_dir / cartridge_id
        eval_dir = root / "eval_proofs"
        root.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        atoms_path = root / "atoms.parquet"
        sources_path = root / "sources.json"
        sparse_path = root / "sparse_index.bm25.zst"
        graph_path = root / "temporal_graph.sqlite"

        self._write_atoms_parquet(atoms_path, atoms)
        sources_path.write_text(
            json.dumps([source.model_dump(mode="json") for source in sources], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_sparse_index(sparse_path, atoms)
        self._write_graph(graph_path, atoms)

        manifest = CartridgeManifest(
            cartridge_id=cartridge_id,
            name=name,
            created_at=created_at,
            model_base=self.settings.model_id,
            source_count=len(sources),
            atom_count=len(atoms),
            token_equivalent=sum(atom.token_equivalent for atom in atoms),
            artifacts={
                "manifest": "manifest.json",
                "atoms": "atoms.parquet",
                "sources": "sources.json",
                "sparse_index": "sparse_index.bm25.zst",
                "temporal_graph": "temporal_graph.sqlite",
                "eval_proofs": "eval_proofs/",
            },
        )
        manifest_path = root / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        dna = self.compute_dna(root)
        manifest.dna = dna
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        self.db.write_cartridge(
            cartridge_id=cartridge_id,
            name=name,
            root_path=root,
            manifest_json=manifest.model_dump_json(),
            dna=dna,
            token_equivalent=manifest.token_equivalent,
            created_at=created_at,
        )
        return manifest

    def compute_dna(self, root: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if path.is_dir() or path.name in {"manifest.json", "proof_suite.json", "cartridge_assets.json"}:
                continue
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, _DigestWriter(digest))
        return digest.hexdigest()

    def list_manifests(self) -> list[CartridgeManifest]:
        manifests: list[CartridgeManifest] = []
        for row in self.db.list_cartridges():
            manifests.append(CartridgeManifest.model_validate_json(row["manifest_json"]))
        return manifests

    def rebuild(
        self,
        cartridge_id: str,
        atoms: list[EvidenceAtom],
        sources: list[SourceRecord],
    ) -> CartridgeManifest:
        rows = [row for row in self.db.list_cartridges() if row["cartridge_id"] == cartridge_id]
        if not rows:
            raise ValueError(f"Unknown cartridge: {cartridge_id}")
        row = rows[0]
        root = Path(row["root_path"])
        manifest = CartridgeManifest.model_validate_json(row["manifest_json"])
        root.mkdir(parents=True, exist_ok=True)

        self._write_atoms_parquet(root / "atoms.parquet", atoms)
        (root / "sources.json").write_text(
            json.dumps([source.model_dump(mode="json") for source in sources], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_sparse_index(root / "sparse_index.bm25.zst", atoms)
        self._write_graph(root / "temporal_graph.sqlite", atoms)

        manifest.source_count = len(sources)
        manifest.atom_count = len(atoms)
        manifest.token_equivalent = sum(atom.token_equivalent for atom in atoms)
        manifest.artifacts.update(
            {
                "manifest": "manifest.json",
                "atoms": "atoms.parquet",
                "sources": "sources.json",
                "sparse_index": "sparse_index.bm25.zst",
                "temporal_graph": "temporal_graph.sqlite",
                "eval_proofs": "eval_proofs/",
            }
        )
        manifest.dna = self.compute_dna(root)
        (root / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        self.db.write_cartridge(
            cartridge_id=manifest.cartridge_id,
            name=manifest.name,
            root_path=root,
            manifest_json=manifest.model_dump_json(),
            dna=manifest.dna,
            token_equivalent=manifest.token_equivalent,
            created_at=manifest.created_at,
        )
        return manifest

    def attach_adapter(self, cartridge_id: str, adapter_dir: Path) -> CartridgeManifest:
        rows = [row for row in self.db.list_cartridges() if row["cartridge_id"] == cartridge_id]
        if not rows:
            raise ValueError(f"Unknown cartridge: {cartridge_id}")
        row = rows[0]
        root = Path(row["root_path"])
        manifest = CartridgeManifest.model_validate_json(row["manifest_json"])
        adapter_manifest_path = adapter_dir / "adapter_manifest.json"
        if not adapter_manifest_path.exists():
            raise ValueError(f"Missing adapter manifest: {adapter_manifest_path}")
        adapter_manifest = AdapterManifest.model_validate_json(
            adapter_manifest_path.read_text(encoding="utf-8")
        )
        adapter_root = root / "adapters" / adapter_manifest.kind.value
        adapter_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(adapter_dir, adapter_root, dirs_exist_ok=True)
        manifest.artifacts[f"adapter_{adapter_manifest.kind.value}"] = (
            adapter_root.relative_to(root).as_posix() + "/"
        )
        manifest_path = root / "manifest.json"
        manifest.dna = self.compute_dna(root)
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        self.db.write_cartridge(
            cartridge_id=manifest.cartridge_id,
            name=manifest.name,
            root_path=root,
            manifest_json=manifest.model_dump_json(),
            dna=manifest.dna,
            token_equivalent=manifest.token_equivalent,
            created_at=manifest.created_at,
        )
        return manifest

    def update_artifacts(
        self,
        cartridge_id: str,
        artifacts: dict[str, str],
        token_equivalent: int | None = None,
    ) -> CartridgeManifest:
        rows = [row for row in self.db.list_cartridges() if row["cartridge_id"] == cartridge_id]
        if not rows:
            raise ValueError(f"Unknown cartridge: {cartridge_id}")
        row = rows[0]
        root = Path(row["root_path"])
        manifest = CartridgeManifest.model_validate_json(row["manifest_json"])
        manifest.artifacts.update(artifacts)
        if token_equivalent is not None:
            manifest.token_equivalent = token_equivalent
        manifest_path = root / "manifest.json"
        manifest.dna = self.compute_dna(root)
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        self.db.write_cartridge(
            cartridge_id=manifest.cartridge_id,
            name=manifest.name,
            root_path=root,
            manifest_json=manifest.model_dump_json(),
            dna=manifest.dna,
            token_equivalent=manifest.token_equivalent,
            created_at=manifest.created_at,
        )
        return manifest

    def attach_eval_proof(self, cartridge_id: str, proof_dir: Path) -> CartridgeManifest:
        rows = [row for row in self.db.list_cartridges() if row["cartridge_id"] == cartridge_id]
        if not rows:
            raise ValueError(f"Unknown cartridge: {cartridge_id}")
        if not proof_dir.exists() or not proof_dir.is_dir():
            raise ValueError(f"Missing proof directory: {proof_dir}")
        row = rows[0]
        root = Path(row["root_path"])
        manifest = CartridgeManifest.model_validate_json(row["manifest_json"])
        target = root / "eval_proofs" / proof_dir.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(proof_dir, target, dirs_exist_ok=True)
        manifest.artifacts[f"eval_proof_{proof_dir.name}"] = target.relative_to(root).as_posix() + "/"
        manifest_path = root / "manifest.json"
        manifest.dna = self.compute_dna(root)
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        self.db.write_cartridge(
            cartridge_id=manifest.cartridge_id,
            name=manifest.name,
            root_path=root,
            manifest_json=manifest.model_dump_json(),
            dna=manifest.dna,
            token_equivalent=manifest.token_equivalent,
            created_at=manifest.created_at,
        )
        return manifest

    def _write_atoms_parquet(self, path: Path, atoms: list[EvidenceAtom]) -> None:
        rows = [atom.model_dump(mode="json") for atom in atoms]
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path, compression="zstd")

    def _write_sparse_index(self, path: Path, atoms: list[EvidenceAtom]) -> None:
        payload = [
            {"atom_id": atom.atom_id, "source": atom.source, "text": atom.text, "relations": atom.relations}
            for atom in atoms
        ]
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        compressor = zstd.ZstdCompressor(level=9)
        path.write_bytes(compressor.compress(data))

    def _write_graph(self, path: Path, atoms: list[EvidenceAtom]) -> None:
        connection = sqlite3.connect(path)
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    atom_id TEXT PRIMARY KEY,
                    modality TEXT NOT NULL,
                    source TEXT NOT NULL,
                    token_equivalent INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS edges (
                    from_atom_id TEXT NOT NULL,
                    to_atom_id TEXT NOT NULL,
                    relation TEXT NOT NULL
                );
                """
            )
            connection.execute("DELETE FROM edges")
            connection.execute("DELETE FROM nodes")
            connection.executemany(
                "INSERT OR REPLACE INTO nodes(atom_id, modality, source, token_equivalent) VALUES (?, ?, ?, ?)",
                [(atom.atom_id, atom.modality.value, atom.source, atom.token_equivalent) for atom in atoms],
            )
            edge_rows: list[tuple[str, str, str]] = []
            known_ids = {atom.atom_id for atom in atoms}
            for atom in atoms:
                for relation in atom.relations:
                    if ":" not in relation:
                        continue
                    relation_name, target = relation.split(":", 1)
                    if target in known_ids:
                        edge_rows.append((atom.atom_id, target, relation_name))
            connection.executemany(
                "INSERT INTO edges(from_atom_id, to_atom_id, relation) VALUES (?, ?, ?)",
                edge_rows,
            )
            connection.commit()
        finally:
            connection.close()

    def _cartridge_id(self, name: str, created_at: datetime) -> str:
        raw = f"{name}:{created_at.isoformat()}".encode("utf-8")
        safe_name = "".join(char if char.isalnum() else "_" for char in name.lower()).strip("_")[:40]
        return f"{safe_name or 'cartridge'}_{hashlib.sha256(raw).hexdigest()[:12]}"


class _DigestWriter:
    def __init__(self, digest: "hashlib._Hash") -> None:
        self.digest = digest

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        return len(data)
