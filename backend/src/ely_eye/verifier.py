from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .schemas import CitationReport, CompiledContext


ATOM_PATTERN = re.compile(r"[A-Za-z0-9_./\\:-]+:(?:text|image|page|frame|region)[A-Za-z0-9_./\\:#=-]*")


class Verifier:
    def verify_answer(self, raw_answer: str, context: CompiledContext) -> tuple[str, list[str], CitationReport]:
        parsed = self._parse_answer(raw_answer)
        answer = parsed.get("answer", raw_answer)
        citations = parsed.get("citations") or self._extract_citations(raw_answer)
        citations = normalize_citations(citations)
        available = {hit.atom.atom_id for hit in context.hits}
        cited_available, missing = resolve_citations(citations, available)
        accuracy = len(cited_available) / len(citations) if citations else 0.0
        contradiction_notes = parsed.get("contradiction_notes") or []
        if isinstance(contradiction_notes, str):
            contradiction_notes = [contradiction_notes]
        confidence = float(parsed.get("confidence", accuracy))
        return (
            str(answer),
            citations,
            CitationReport(
                cited_atom_ids=cited_available,
                missing_atom_ids=missing,
                citation_accuracy=accuracy,
                contradiction_notes=[str(note) for note in contradiction_notes],
                confidence=max(0.0, min(1.0, confidence)),
            ),
        )

    def _parse_answer(self, raw_answer: str) -> dict[str, Any]:
        stripped = raw_answer.strip()
        if not stripped:
            return {"answer": ""}
        candidates = [stripped]
        if "```" in stripped:
            parts = stripped.split("```")
            candidates.extend(part.removeprefix("json").strip() for part in parts)
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            candidates.append(stripped[start : end + 1])
        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return {"answer": raw_answer, "citations": self._extract_citations(raw_answer)}

    def _extract_citations(self, raw_answer: str) -> list[str]:
        return sorted(set(ATOM_PATTERN.findall(raw_answer)))


def normalize_citations(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    normalized: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict) and "atom_id" in item:
                normalized.append(str(item["atom_id"]))
    return normalized


def resolve_citations(citations: list[str], available: set[str]) -> tuple[list[str], list[str]]:
    cited_available: list[str] = []
    missing: list[str] = []
    for citation in citations:
        atom_id = citation.split("#", 1)[0]
        if citation in available:
            cited_available.append(citation)
        elif atom_id in available:
            cited_available.append(atom_id)
        else:
            missing.append(citation)
    return list(dict.fromkeys(cited_available)), missing


@dataclass(frozen=True)
class DriftRule:
    """One declarative extractor for a design fact (PRD 11.15 taxonomy).

    ``pattern`` exposes an optional ``scope`` group (component qualifier) and a
    mandatory ``value`` group. The same multi-value-across-sources mechanism
    surfaces every drift kind; ``category`` is the PRD baseline label.
    """

    name: str
    category: str
    pattern: re.Pattern[str]
    value_kind: str


class VisualContradictionLens:
    """Detect cross-version visual contradictions (PRD 11.15).

    A single mechanism backs the whole taxonomy: extract scoped ``(attribute,
    value)`` design facts from each Evidence Atom, group identical attributes
    across atoms, and flag any attribute carrying more than one distinct value
    across at least two distinct sources. Requiring two sources prevents a
    single CSS file with many rules from reading as a contradiction. Source
    kind and version metadata refine each note into design-token, layout,
    typography, copy, visual-code, or temporal drift.
    """

    RULES: tuple[DriftRule, ...] = (
        DriftRule(
            "radius",
            "design-token drift",
            re.compile(
                r"(?P<scope>[A-Za-z][\w .-]*?)?\.?(?:border-)?radius\s*[:=]\s*(?P<value>\d+(?:\.\d+)?)\s*px",
                re.IGNORECASE,
            ),
            "length",
        ),
        DriftRule(
            "color",
            "design-token drift",
            re.compile(
                r"(?P<scope>[A-Za-z][\w .-]*?)?\.?(?:colou?r|background|fill)\s*[:=]\s*(?P<value>#[0-9a-fA-F]{3,8})",
                re.IGNORECASE,
            ),
            "color",
        ),
        DriftRule(
            "font-size",
            "typography drift",
            re.compile(
                r"(?P<scope>[A-Za-z][\w .-]*?)?\.?font-size\s*[:=]\s*(?P<value>\d+(?:\.\d+)?)\s*(?:px|rem|pt)",
                re.IGNORECASE,
            ),
            "length",
        ),
        DriftRule(
            "spacing",
            "layout drift",
            re.compile(
                r"(?P<scope>[A-Za-z][\w .-]*?)?\.?(?:padding|margin|gap|spacing)\s*[:=]\s*(?P<value>\d+(?:\.\d+)?)\s*px",
                re.IGNORECASE,
            ),
            "length",
        ),
        DriftRule(
            "label",
            "copy drift",
            re.compile(
                r"(?P<scope>[A-Za-z][\w .-]*?)?\.?(?:label|copy|cta|caption)\s*[:=]\s*[\"'](?P<value>[^\"']+)[\"']",
                re.IGNORECASE,
            ),
            "text",
        ),
    )

    def detect(self, context: CompiledContext) -> list[str]:
        observations: dict[str, dict[str, set[str]]] = {}
        raw_display: dict[str, dict[str, str]] = {}
        category_of: dict[str, str] = {}
        source_kind: dict[str, str] = {}
        source_version: dict[str, str] = {}
        for hit in context.hits:
            atom = hit.atom
            source_kind[atom.source_id] = str(atom.metadata.get("source_kind") or atom.modality.value)
            version = atom.metadata.get("version")
            if version is not None:
                source_version[atom.source_id] = str(version)
            for rule in self.RULES:
                for match in rule.pattern.finditer(atom.text):
                    scope = (match.groupdict().get("scope") or "").strip().lower().rstrip(".")
                    key = f"{scope}.{rule.name}" if scope else rule.name
                    normalized = normalize_design_value(match.group("value"), rule.value_kind)
                    if not normalized:
                        continue
                    observations.setdefault(key, {}).setdefault(normalized, set()).add(atom.source_id)
                    raw_display.setdefault(key, {})[normalized] = match.group("value").strip()
                    category_of[key] = rule.category
        notes: list[str] = []
        for key in sorted(observations):
            value_sources = observations[key]
            sources = set().union(*value_sources.values())
            if len(value_sources) > 1 and len(sources) > 1:
                category = self._classify(category_of[key], sources, source_kind, source_version)
                values = ", ".join(raw_display[key][value] for value in sorted(value_sources))
                notes.append(f"{key} {category}: {values}")
        return notes

    @staticmethod
    def _classify(
        baseline: str,
        sources: set[str],
        source_kind: dict[str, str],
        source_version: dict[str, str],
    ) -> str:
        kinds = {source_kind.get(source, "") for source in sources}
        design_kinds = {"figma", "spec", "design", "ui_screenshot", "image", "pdf_page", "pdf_region"}
        code_kinds = {"code", "css"}
        if kinds & design_kinds and kinds & code_kinds:
            return "visual-code drift"
        versions = {source_version[source] for source in sources if source in source_version}
        if len(versions) > 1:
            return "temporal drift"
        return baseline


def normalize_design_value(value: str, value_kind: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    if value_kind == "length":
        try:
            return f"{float(cleaned):g}"
        except ValueError:
            return ""
    if value_kind == "color":
        hex_value = cleaned.lower().lstrip("#")
        if len(hex_value) in {3, 4}:
            hex_value = "".join(component * 2 for component in hex_value)
        return f"#{hex_value}"
    return " ".join(cleaned.lower().split())
