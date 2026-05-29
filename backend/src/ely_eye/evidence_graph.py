from __future__ import annotations

import hashlib

from .config import Settings, get_settings
from .db import Database
from .schemas import CitationReport, CompiledContext


class EvidenceGraphService:
    def __init__(self, settings: Settings | None = None, database: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = database or Database(self.settings)

    def record_answer(
        self,
        question: str,
        answer: str,
        report: CitationReport,
        context: CompiledContext,
    ) -> str:
        claim_id = claim_node_id(question, answer)
        hit_by_id = {hit.atom.atom_id: hit for hit in context.hits}
        edges: list[dict[str, object]] = []
        for atom_id in report.cited_atom_ids:
            hit = hit_by_id.get(atom_id)
            edges.append(
                {
                    "from_atom_id": claim_id,
                    "to_atom_id": atom_id,
                    "relation": "cites",
                    "confidence": report.confidence,
                    "metadata": {
                        "question": question,
                        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                        "cache_trace_id": context.cache_trace_id,
                        "final_score": hit.final_score if hit else None,
                    },
                }
            )
        for atom_id in report.missing_atom_ids:
            edges.append(
                {
                    "from_atom_id": claim_id,
                    "to_atom_id": atom_id,
                    "relation": "missing_citation",
                    "confidence": 0.0,
                    "metadata": {"question": question, "cache_trace_id": context.cache_trace_id},
                }
            )
        for note in report.contradiction_notes:
            note_id = "contradiction_" + hashlib.sha256(note.encode("utf-8")).hexdigest()[:16]
            edges.append(
                {
                    "from_atom_id": claim_id,
                    "to_atom_id": note_id,
                    "relation": "contradiction_note",
                    "confidence": report.confidence,
                    "metadata": {"note": note, "question": question},
                }
            )
        self.db.write_evidence_edges(edges)
        return claim_id


def claim_node_id(question: str, answer: str) -> str:
    digest = hashlib.sha256(f"{question}\n{answer}".encode("utf-8")).hexdigest()[:20]
    return f"claim_{digest}"
