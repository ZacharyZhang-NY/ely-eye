from __future__ import annotations

import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from .config import Settings, get_settings
from .db import Database
from .embeddings import EmbeddingService
from .schemas import EvidenceAtom, RetrievalHit


class RetrievalService:
    def __init__(self, settings: Settings | None = None, database: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = database or Database(self.settings)
        self.embeddings = EmbeddingService(self.settings, self.db)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        limit = top_k or self.settings.retrieval_top_k
        atoms = self.db.list_atoms()
        if not atoms:
            return []

        sparse_scores = self._sparse_search(query, atoms, limit * 3)
        dense_scores: dict[str, float] = {}
        if self.embeddings.enabled():
            dense_scores = self._dense_search(query, limit * 3)

        all_ids = set(sparse_scores) | set(dense_scores)
        atom_map = {atom.atom_id: atom for atom in atoms}
        hits: list[RetrievalHit] = []
        for atom_id in all_ids:
            atom = atom_map.get(atom_id)
            if atom is None:
                continue
            sparse_score = sparse_scores.get(atom_id, 0.0)
            dense_score = dense_scores.get(atom_id, 0.0)
            graph_score = self._graph_score(atom, query)
            final_score = sparse_score * 0.55 + dense_score * 0.35 + graph_score * 0.10
            hits.append(
                RetrievalHit(
                    atom=atom,
                    sparse_score=sparse_score,
                    dense_score=dense_score,
                    graph_score=graph_score,
                    final_score=final_score,
                )
            )
        hits.sort(key=lambda hit: hit.final_score, reverse=True)
        return hits[:limit]

    def _sparse_search(self, query: str, atoms: list[EvidenceAtom], limit: int) -> dict[str, float]:
        tokenized = [tokenize(atom.text + " " + atom.source) for atom in atoms]
        bm25 = BM25Okapi(tokenized)
        query_tokens = tokenize(query)
        scores = bm25.get_scores(query_tokens)
        lexical_scores = [lexical_score(query_tokens, tokens, atoms[index]) for index, tokens in enumerate(tokenized)]
        combined = [max(float(scores[index]), lexical_scores[index]) for index in range(len(atoms))]
        order = sorted(range(len(combined)), key=lambda index: combined[index], reverse=True)[:limit]
        if not order:
            return {}
        max_score = max(float(combined[index]) for index in order) or 1.0
        return {atoms[index].atom_id: float(combined[index]) / max_score for index in order if combined[index] > 0}

    def _dense_search(self, query: str, limit: int) -> dict[str, float]:
        index = self.embeddings.load_index()
        if index is None:
            index = self.embeddings.build_index()
        query_vector = self.embeddings.embed_query(query)
        return dict(index.search(query_vector, limit))

    def _graph_score(self, atom: EvidenceAtom, query: str) -> float:
        query_tokens = set(tokenize(query))
        relation_tokens = set()
        for relation in atom.relations:
            relation_tokens.update(tokenize(relation))
        if not query_tokens or not relation_tokens:
            return 0.0
        return len(query_tokens & relation_tokens) / len(query_tokens)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[\w\u4e00-\u9fff]+", text)]


def lexical_score(query_tokens: list[str], document_tokens: list[str], atom: EvidenceAtom) -> float:
    if not query_tokens or not document_tokens:
        return 0.0
    document_set = set(document_tokens)
    overlap = sum(1 for token in query_tokens if token in document_set)
    phrase_hits = sum(1 for token in query_tokens if token and token in atom.text.lower())
    return (overlap + 0.25 * phrase_hits) / len(query_tokens)


def group_hits_by_source(hits: list[RetrievalHit]) -> dict[str, list[RetrievalHit]]:
    grouped: dict[str, list[RetrievalHit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.atom.source].append(hit)
    return dict(grouped)
