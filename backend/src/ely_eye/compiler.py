from __future__ import annotations

import hashlib
from typing import Iterable

from .cache_fabric import CacheFabric
from .config import Settings, get_settings
from .db import Database
from .retrieval import RetrievalService
from .schemas import CompiledContext, ContextPlan, Modality, RetrievalHit, RuntimeProfile


PROFILE_TOKEN_BUDGETS = {
    RuntimeProfile.live_demo: 128_000,
    RuntimeProfile.extreme_context: 1_010_000,
    RuntimeProfile.library_100m: 262_144,
    RuntimeProfile.research_theater: 128_000,
}


class ContextCompiler:
    def __init__(self, settings: Settings | None = None, database: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = database or Database(self.settings)
        self.retrieval = RetrievalService(self.settings, self.db)
        self.cache = CacheFabric(self.settings, self.db)

    def compile(self, question: str, profile: RuntimeProfile) -> CompiledContext:
        cache_key = self.cache.semantic_key(question, profile.value)
        cached = self.cache.read_semantic_cache(cache_key)
        if cached:
            compiled = CompiledContext.model_validate(cached)
            if compiled.hits:
                return compiled

        plan = self.plan(question, profile)
        hits = self.retrieval.search(plan.retrieval_query, plan.evidence_budget_atoms)
        packed_text, token_total = self._pack_hits(hits, plan.token_budget, plan.retrieval_query)
        trace_id = hashlib.sha256(f"{cache_key}:{token_total}".encode("utf-8")).hexdigest()[:16]
        compiled = CompiledContext(
            plan=plan,
            hits=hits,
            packed_text=packed_text,
            token_equivalent=token_total,
            cache_trace_id=trace_id,
        )
        self.cache.write_semantic_cache(cache_key, compiled.model_dump(mode="json"))
        return compiled

    def plan(self, question: str, profile: RuntimeProfile) -> ContextPlan:
        modalities = infer_modalities(question)
        token_budget = PROFILE_TOKEN_BUDGETS[profile]
        evidence_budget = 48 if profile in {RuntimeProfile.library_100m, RuntimeProfile.extreme_context} else 24
        return ContextPlan(
            question=question,
            profile=profile,
            required_modalities=modalities,
            evidence_budget_atoms=evidence_budget,
            token_budget=token_budget,
            retrieval_query=question,
            compression_strategy=self._compression_strategy(profile, modalities),
            verifier_contract=[
                "Every factual claim must cite atom ids.",
                "Citations must include page, image region, frame, or code path when available.",
                "Contradictions must be reported as contradiction_notes.",
            ],
        )

    def _compression_strategy(self, profile: RuntimeProfile, modalities: Iterable[Modality]) -> str:
        has_visual = any(modality in {Modality.image, Modality.pdf_page, Modality.video_frame, Modality.ui_screenshot} for modality in modalities)
        if profile == RuntimeProfile.extreme_context:
            return "YaRN + V-NSA sparse blocks + DVKV visual KV compression + CPU/NVMe parked context"
        if profile == RuntimeProfile.library_100m:
            return "Context Cartridge retrieval + ColQwen-compatible dense vectors + BM25 + evidence graph pack"
        if has_visual:
            return "Visual patch cache + DVKV evidence-biased region retention"
        return "Prefix cache + sparse evidence packing"

    def _pack_hits(self, hits: list[RetrievalHit], token_budget: int, query: str) -> tuple[str, int]:
        sections: list[str] = []
        token_total = 0
        for hit in hits:
            atom = hit.atom
            evidence_text = focused_evidence_text(atom.text, query)
            atom_tokens = max(1, len(evidence_text) // 4)
            if token_total + atom_tokens > token_budget:
                break
            citation = {
                "atom_id": atom.atom_id,
                "modality": atom.modality.value,
                "source": atom.source,
                "image_ref": atom.image_ref,
                "layout": atom.layout.model_dump() if atom.layout else None,
                "score": hit.final_score,
            }
            sections.append(f"[EVIDENCE]\n{citation}\n{evidence_text}\n[/EVIDENCE]")
            token_total += atom_tokens
        return "\n\n".join(sections), token_total


def infer_modalities(question: str) -> list[Modality]:
    lowered = question.lower()
    modalities: list[Modality] = []
    if any(word in lowered for word in ("pdf", "page", "paper", "论文", "页面", "页码")):
        modalities.append(Modality.pdf_page)
    if any(word in lowered for word in ("image", "screenshot", "figma", "ui", "图", "截图", "设计")):
        modalities.extend([Modality.image, Modality.ui_screenshot])
    if any(word in lowered for word in ("video", "frame", "clip", "视频", "帧")):
        modalities.append(Modality.video_frame)
    if any(word in lowered for word in ("code", "function", "class", "代码", "函数")):
        modalities.append(Modality.code)
    if not modalities:
        modalities.append(Modality.mixed)
    return list(dict.fromkeys(modalities))


def focused_evidence_text(text: str, query: str, window: int = 1400, max_windows: int = 4) -> str:
    if len(text) <= window * max_windows:
        return text
    lowered = text.lower()
    query_terms = [term for term in query.lower().replace("/", " ").replace("-", " ").split() if len(term) > 2]
    anchors: list[int] = []
    for term in query_terms:
        index = lowered.find(term)
        if index >= 0:
            anchors.append(index)
    anchors.extend([0, len(text) // 3, (len(text) * 2) // 3])
    windows: list[tuple[int, int]] = []
    for anchor in anchors:
        start = max(0, anchor - window // 2)
        end = min(len(text), start + window)
        start = max(0, end - window)
        if all(abs(start - existing_start) > window // 2 for existing_start, _ in windows):
            windows.append((start, end))
        if len(windows) >= max_windows:
            break
    return "\n\n[...]\n\n".join(text[start:end].strip() for start, end in windows if text[start:end].strip())
