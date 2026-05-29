from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .schemas import EvidenceAtom, SourceRecord


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _loads(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


class Database:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.path = self.settings.db_path
        self.init()

    @contextmanager
    def connect(self) -> Iterable[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def init(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    mime TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS atoms (
                    atom_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
                    modality TEXT NOT NULL,
                    source TEXT NOT NULL,
                    time TEXT NOT NULL,
                    text TEXT NOT NULL,
                    image_ref TEXT,
                    embedding_refs_json TEXT NOT NULL,
                    layout_json TEXT,
                    relations_json TEXT NOT NULL,
                    trust_json TEXT NOT NULL,
                    token_equivalent INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_atoms_source_id ON atoms(source_id);
                CREATE INDEX IF NOT EXISTS idx_atoms_modality ON atoms(modality);
                CREATE VIRTUAL TABLE IF NOT EXISTS atom_fts USING fts5(
                    atom_id UNINDEXED,
                    source,
                    text,
                    tokenize = 'unicode61'
                );

                CREATE TABLE IF NOT EXISTS cartridges (
                    cartridge_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    root_path TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    dna TEXT,
                    token_equivalent INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cache_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    event TEXT NOT NULL,
                    bytes_estimate INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evidence_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_atom_id TEXT NOT NULL,
                    to_atom_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                """
            )

    def upsert_sources(self, sources: list[SourceRecord]) -> None:
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO sources (
                    source_id, path, kind, mime, sha256, size_bytes, created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    path=excluded.path,
                    kind=excluded.kind,
                    mime=excluded.mime,
                    sha256=excluded.sha256,
                    size_bytes=excluded.size_bytes,
                    metadata_json=excluded.metadata_json
                """,
                [
                    (
                        source.source_id,
                        source.path,
                        source.kind,
                        source.mime,
                        source.sha256,
                        source.size_bytes,
                        source.created_at.isoformat(),
                        _json(source.metadata),
                    )
                    for source in sources
                ],
            )

    def upsert_atoms(self, atoms: list[EvidenceAtom]) -> None:
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO atoms (
                    atom_id, source_id, modality, source, time, text, image_ref,
                    embedding_refs_json, layout_json, relations_json, trust_json,
                    token_equivalent, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(atom_id) DO UPDATE SET
                    modality=excluded.modality,
                    source=excluded.source,
                    time=excluded.time,
                    text=excluded.text,
                    image_ref=excluded.image_ref,
                    embedding_refs_json=excluded.embedding_refs_json,
                    layout_json=excluded.layout_json,
                    relations_json=excluded.relations_json,
                    trust_json=excluded.trust_json,
                    token_equivalent=excluded.token_equivalent,
                    metadata_json=excluded.metadata_json
                """,
                [self._atom_params(atom) for atom in atoms],
            )
            connection.executemany("DELETE FROM atom_fts WHERE atom_id = ?", [(atom.atom_id,) for atom in atoms])
            connection.executemany(
                "INSERT INTO atom_fts(atom_id, source, text) VALUES (?, ?, ?)",
                [(atom.atom_id, atom.source, atom.text) for atom in atoms],
            )

    def _atom_params(self, atom: EvidenceAtom) -> tuple[Any, ...]:
        return (
            atom.atom_id,
            atom.source_id,
            atom.modality.value,
            atom.source,
            atom.time.isoformat(),
            atom.text,
            atom.image_ref,
            _json(atom.embedding_refs),
            _json(atom.layout.model_dump() if atom.layout else None),
            _json(atom.relations),
            _json(atom.trust.model_dump()),
            atom.token_equivalent,
            _json(atom.metadata),
        )

    def list_atoms(self, limit: int | None = None) -> list[EvidenceAtom]:
        query = "SELECT * FROM atoms ORDER BY time DESC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_atom(row) for row in rows]

    def get_atoms(self, atom_ids: list[str]) -> list[EvidenceAtom]:
        if not atom_ids:
            return []
        placeholders = ",".join("?" for _ in atom_ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM atoms WHERE atom_id IN ({placeholders})", atom_ids
            ).fetchall()
        by_id = {row["atom_id"]: self._row_to_atom(row) for row in rows}
        return [by_id[atom_id] for atom_id in atom_ids if atom_id in by_id]

    def search_fts(self, query: str, limit: int) -> list[tuple[str, float]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT atom_id, bm25(atom_fts) AS score
                FROM atom_fts
                WHERE atom_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        return [(row["atom_id"], float(-row["score"])) for row in rows]

    def atom_count(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM atoms").fetchone()[0])

    def token_equivalent_sum(self) -> int:
        with self.connect() as connection:
            value = connection.execute("SELECT COALESCE(SUM(token_equivalent), 0) FROM atoms").fetchone()[0]
        return int(value)

    def cartridge_token_equivalent_sum(self) -> int:
        with self.connect() as connection:
            value = connection.execute(
                "SELECT COALESCE(SUM(token_equivalent), 0) FROM cartridges"
            ).fetchone()[0]
        return int(value)

    def source_count(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0])

    def list_sources(self) -> list[SourceRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT source_id, path, kind, mime, sha256, size_bytes, created_at, metadata_json "
                "FROM sources ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_source(row) for row in rows]

    def cartridge_count(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM cartridges").fetchone()[0])

    def write_cartridge(
        self,
        cartridge_id: str,
        name: str,
        root_path: Path,
        manifest_json: str,
        dna: str | None,
        token_equivalent: int,
        created_at: datetime,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO cartridges (
                    cartridge_id, name, root_path, manifest_json, dna, token_equivalent, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cartridge_id) DO UPDATE SET
                    name=excluded.name,
                    root_path=excluded.root_path,
                    manifest_json=excluded.manifest_json,
                    dna=excluded.dna,
                    token_equivalent=excluded.token_equivalent
                """,
                (
                    cartridge_id,
                    name,
                    str(root_path),
                    manifest_json,
                    dna,
                    token_equivalent,
                    created_at.isoformat(),
                ),
            )

    def list_cartridges(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT cartridge_id, name, root_path, manifest_json, dna, token_equivalent, created_at "
                "FROM cartridges ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def log_cache_event(
        self,
        layer: str,
        cache_key: str,
        event: str,
        bytes_estimate: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO cache_events(ts, layer, cache_key, event, bytes_estimate, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    layer,
                    cache_key,
                    event,
                    bytes_estimate,
                    _json(metadata or {}),
                ),
            )

    def write_evidence_edges(self, edges: list[dict[str, Any]]) -> None:
        if not edges:
            return
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO evidence_edges(from_atom_id, to_atom_id, relation, confidence, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(edge["from_atom_id"]),
                        str(edge["to_atom_id"]),
                        str(edge["relation"]),
                        float(edge["confidence"]),
                        _json(edge.get("metadata") or {}),
                    )
                    for edge in edges
                ],
            )

    def list_evidence_edges(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = (
            "SELECT from_atom_id, to_atom_id, relation, confidence, metadata_json "
            "FROM evidence_edges ORDER BY id DESC"
        )
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                "from_atom_id": row["from_atom_id"],
                "to_atom_id": row["to_atom_id"],
                "relation": row["relation"],
                "confidence": float(row["confidence"]),
                "metadata": _loads(row["metadata_json"], {}),
            }
            for row in rows
        ]

    def cache_layer_rows(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT
                  layer,
                  COUNT(DISTINCT cache_key) AS entries,
                  COALESCE(SUM(bytes_estimate), 0) AS bytes_estimate,
                  SUM(CASE WHEN event = 'hit' THEN 1 ELSE 0 END) AS hit_count,
                  SUM(CASE WHEN event = 'miss' THEN 1 ELSE 0 END) AS miss_count,
                  MAX(ts) AS last_event_at
                FROM cache_events
                GROUP BY layer
                """
            ).fetchall()

    def _row_to_atom(self, row: sqlite3.Row) -> EvidenceAtom:
        from .schemas import LayoutBox, Modality, TrustScores

        layout_data = _loads(row["layout_json"], None)
        return EvidenceAtom(
            atom_id=row["atom_id"],
            modality=Modality(row["modality"]),
            source_id=row["source_id"],
            source=row["source"],
            time=datetime.fromisoformat(row["time"]),
            text=row["text"],
            image_ref=row["image_ref"],
            embedding_refs=_loads(row["embedding_refs_json"], []),
            layout=LayoutBox(**layout_data) if layout_data else None,
            relations=_loads(row["relations_json"], []),
            trust=TrustScores(**_loads(row["trust_json"], {"parser": 0.0})),
            token_equivalent=int(row["token_equivalent"]),
            metadata=_loads(row["metadata_json"], {}),
        )

    def _row_to_source(self, row: sqlite3.Row) -> SourceRecord:
        return SourceRecord(
            source_id=row["source_id"],
            path=row["path"],
            kind=row["kind"],
            mime=row["mime"],
            sha256=row["sha256"],
            size_bytes=int(row["size_bytes"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=_loads(row["metadata_json"], {}),
        )
