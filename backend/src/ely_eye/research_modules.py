from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .cache_fabric import CacheFabric
from .config import Settings, get_settings
from .db import Database
from .dvkv import VisualToken, select_dvkv_tokens
from .schemas import RetrievalHit


class ResearchModuleService:
    def __init__(self, settings: Settings | None = None, database: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = database or Database(self.settings)

    def build_report(self, query: str, hits: list[RetrievalHit], output_path: Path | None = None) -> dict[str, Any]:
        report = {
            "v_nsa": self.v_nsa_plan(hits),
            "dvkv": self.dvkv_plan(hits),
            "x_modal_kv": self.x_modal_kv_plan(hits),
            "speculative_prefetch": self.speculative_prefetch_plan(query, hits),
            "predictive_eviction": self.predictive_eviction_plan(),
            "differentiable_retrieval": self.differentiable_retrieval_distribution(hits),
            "evidence_graph": self.evidence_graph_status(),
            "sleep_consolidation": self.sleep_consolidation_plan(hits),
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def v_nsa_plan(self, hits: list[RetrievalHit]) -> dict[str, Any]:
        heads = {
            "visual_anchor_heads": [],
            "text_retrieval_heads": [],
            "streaming_heads": [],
        }
        for index, hit in enumerate(hits):
            target = {
                "atom_id": hit.atom.atom_id,
                "modality": hit.atom.modality.value,
                "score": hit.final_score,
                "layout": hit.atom.layout.model_dump() if hit.atom.layout else None,
            }
            if hit.atom.image_ref or hit.atom.modality.value in {"image", "pdf_page", "pdf_region", "video_frame", "ui_screenshot"}:
                heads["visual_anchor_heads"].append(target)
            elif hit.final_score >= 0.45:
                heads["text_retrieval_heads"].append(target)
            else:
                heads["streaming_heads"].append(target)
        sparse_blocks = sorted(
            heads["visual_anchor_heads"] + heads["text_retrieval_heads"],
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        return {
            "head_roles": heads,
            "sparse_block_count": len(sparse_blocks),
            "sparse_blocks": sparse_blocks[:32],
        }

    def dvkv_plan(self, hits: list[RetrievalHit], budget: int = 16) -> dict[str, Any]:
        tokens = [
            VisualToken(
                token_id=hit.atom.atom_id,
                vector=stable_vector(hit.atom.atom_id + hit.atom.text, 64),
                attention_importance=max(0.0, hit.final_score),
                evidence_value=hit.atom.trust.parser * 0.45 + hit.atom.trust.ocr * 0.35 + hit.atom.trust.model * 0.20,
            )
            for hit in hits
            if hit.atom.image_ref or hit.atom.layout is not None
        ]
        if not tokens:
            visual_atoms = [atom for atom in self.db.list_atoms() if atom.image_ref]
            tokens = [
                VisualToken(
                    token_id=atom.atom_id,
                    vector=stable_vector(atom.atom_id + atom.source, 64),
                    attention_importance=1.0 / (index + 1),
                    evidence_value=atom.trust.parser * 0.45 + atom.trust.ocr * 0.35 + atom.trust.model * 0.20,
                )
                for index, atom in enumerate(visual_atoms[:budget])
            ]
        selected = select_dvkv_tokens(tokens, min(budget, max(1, len(tokens)))) if tokens else []
        return {
            "candidate_tokens": len(tokens),
            "budget": budget,
            "selected": [
                {
                    "token_id": token.token_id,
                    "attention_importance": token.attention_importance,
                    "evidence_value": token.evidence_value,
                }
                for token in selected
            ],
        }

    def x_modal_kv_plan(self, hits: list[RetrievalHit]) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for hit in hits:
            source_key = hashlib.sha256(hit.atom.source_id.encode("utf-8")).hexdigest()[:16]
            groups.setdefault(source_key, []).append(
                {
                    "atom_id": hit.atom.atom_id,
                    "modality": hit.atom.modality.value,
                    "has_image": bool(hit.atom.image_ref),
                    "score": hit.final_score,
                }
            )
        shared_blocks = [
            {
                "block_id": f"xkv_{source_key}",
                "source_key": source_key,
                "modalities": sorted({item["modality"] for item in items}),
                "atom_ids": [item["atom_id"] for item in sorted(items, key=lambda value: value["score"], reverse=True)],
            }
            for source_key, items in groups.items()
        ]
        return {"shared_block_count": len(shared_blocks), "shared_blocks": shared_blocks}

    def speculative_prefetch_plan(self, query: str, hits: list[RetrievalHit]) -> dict[str, Any]:
        query_terms = {term for term in query.lower().split() if len(term) > 2}
        candidates = []
        for hit in hits:
            text_terms = set(hit.atom.text.lower().split())
            lexical_overlap = len(query_terms & text_terms) / max(1, len(query_terms))
            priority = hit.final_score * 0.75 + lexical_overlap * 0.25
            candidates.append(
                {
                    "cache_key": hashlib.sha256(hit.atom.atom_id.encode("utf-8")).hexdigest()[:16],
                    "atom_id": hit.atom.atom_id,
                    "priority": priority,
                    "bytes_estimate": max(128, hit.atom.token_equivalent * 16),
                }
            )
        candidates.sort(key=lambda item: item["priority"], reverse=True)
        return {"prefetch_blocks": candidates[:8], "planner_model": self.settings.planner_model_id}

    def predictive_eviction_plan(self) -> dict[str, Any]:
        layers = CacheFabric(self.settings, self.db).statuses()
        rows = []
        for layer in layers:
            total = max(1, layer.hit_count + layer.miss_count)
            hit_rate = layer.hit_count / total
            rows.append(
                {
                    "layer": layer.layer,
                    "entries": layer.entries,
                    "hit_rate": hit_rate,
                    "evict": hit_rate < 0.25 and layer.entries > 0,
                    "replay_token": hashlib.sha256(f"{layer.layer}:{layer.entries}".encode("utf-8")).hexdigest()[:16],
                }
            )
        return {"policy": "hit-rate weighted replay buffer", "layers": rows}

    def differentiable_retrieval_distribution(self, hits: list[RetrievalHit], temperature: float = 0.05) -> dict[str, Any]:
        if not hits:
            return {"temperature": temperature, "probabilities": []}
        scores = np.array([hit.final_score for hit in hits], dtype=np.float64)
        scaled = scores / max(temperature, 1e-6)
        scaled -= float(np.max(scaled))
        probs = np.exp(scaled)
        probs /= probs.sum()
        top_index = int(np.argmax(probs))
        return {
            "temperature": temperature,
            "estimator": "softmax distribution with straight-through top-1 selection",
            "straight_through_atom_id": hits[top_index].atom.atom_id,
            "probabilities": [
                {"atom_id": hit.atom.atom_id, "probability": float(prob)}
                for hit, prob in zip(hits, probs, strict=True)
            ],
        }

    def evidence_graph_status(self) -> dict[str, Any]:
        edges = self.db.list_evidence_edges(limit=64)
        return {
            "edge_count_sample": len(edges),
            "relations": sorted({edge["relation"] for edge in edges}),
            "recent_edges": edges[:16],
        }

    def sleep_consolidation_plan(self, hits: list[RetrievalHit]) -> dict[str, Any]:
        durable = [
            {
                "atom_id": hit.atom.atom_id,
                "reason": "high score and cited-ready evidence",
                "score": hit.final_score,
            }
            for hit in hits
            if hit.final_score >= 0.25
        ]
        return {
            "target_adapter": "hme_ttt_vl",
            "target_cartridge": "ely_eye_prd",
            "candidate_count": len(durable),
            "durable_candidates": durable[:32],
        }


def stable_vector(seed: str, dims: int) -> np.ndarray:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    values = []
    while len(values) < dims:
        for byte in digest:
            values.append((byte / 255.0) * 2.0 - 1.0)
            if len(values) == dims:
                break
        digest = hashlib.sha256(digest).digest()
    vector = np.array(values, dtype=np.float32)
    norm = float(math.sqrt(float(vector @ vector)))
    if norm > 0:
        vector /= norm
    return vector
