from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings, get_settings
from .db import Database
from .schemas import EvidenceAtom


@dataclass
class DenseIndex:
    atom_ids: list[str]
    vectors: np.ndarray

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self.vectors.size == 0:
            return []
        query = normalize(query_vector.reshape(1, -1))[0]
        vectors = normalize(self.vectors)
        scores = vectors @ query
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.atom_ids[index], float(scores[index])) for index in order]


class EmbeddingService:
    def __init__(self, settings: Settings | None = None, database: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = database or Database(self.settings)
        self.index_path = self.settings.cache_dir / "dense_index.npy"
        self.map_path = self.settings.cache_dir / "dense_index_atoms.json"
        self._model: Any | None = None

    def enabled(self) -> bool:
        return self.settings.dense_embeddings_enabled

    def build_index(self, atoms: list[EvidenceAtom] | None = None) -> DenseIndex:
        atoms = atoms or self.db.list_atoms()
        if not atoms:
            index = DenseIndex(atom_ids=[], vectors=np.zeros((0, 0), dtype=np.float32))
            self._write_index(index)
            return index

        model = self._load_model()
        payloads = [self._atom_payload(atom) for atom in atoms]
        vectors = model.encode(payloads, normalize_embeddings=True, show_progress_bar=True)
        index = DenseIndex(atom_ids=[atom.atom_id for atom in atoms], vectors=np.asarray(vectors, dtype=np.float32))
        self._write_index(index)
        return index

    def load_index(self) -> DenseIndex | None:
        if not self.index_path.exists() or not self.map_path.exists():
            return None
        atom_ids = json.loads(self.map_path.read_text(encoding="utf-8"))
        vectors = np.load(self.index_path)
        return DenseIndex(atom_ids=atom_ids, vectors=vectors)

    def embed_query(self, query: str) -> np.ndarray:
        model = self._load_model()
        vector = model.encode([{"text": query}], normalize_embeddings=True)
        return np.asarray(vector[0], dtype=np.float32)

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.settings.embedding_model_id, trust_remote_code=True)
        return self._model

    def _write_index(self, index: DenseIndex) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.index_path, index.vectors)
        self.map_path.write_text(json.dumps(index.atom_ids, ensure_ascii=False, indent=2), encoding="utf-8")

    def _atom_payload(self, atom: EvidenceAtom) -> dict[str, Any]:
        if atom.image_ref:
            from .storage import ObjectStore

            image_path = ObjectStore(self.settings).resolve(atom.image_ref)
            payload: dict[str, Any] = {"image": str(image_path)}
            if atom.text:
                payload["text"] = atom.text
            return payload
        return {"text": atom.text or atom.source}


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
