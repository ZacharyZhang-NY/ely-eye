from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VisualToken:
    token_id: str
    vector: np.ndarray
    attention_importance: float
    evidence_value: float


def select_dvkv_tokens(
    tokens: list[VisualToken],
    budget: int,
    diversity_weight: float = 0.35,
    evidence_weight: float = 0.30,
) -> list[VisualToken]:
    if budget <= 0:
        raise ValueError("DVKV budget must be positive")
    if len(tokens) <= budget:
        return tokens

    selected: list[VisualToken] = []
    remaining = tokens.copy()
    first = max(remaining, key=lambda token: token.attention_importance + evidence_weight * token.evidence_value)
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < budget:
        scores = []
        selected_vectors = np.stack([normalize(token.vector) for token in selected])
        for token in remaining:
            vector = normalize(token.vector)
            nearest_similarity = float(np.max(selected_vectors @ vector))
            diversity = 1.0 - nearest_similarity
            score = (
                token.attention_importance
                + diversity_weight * diversity
                + evidence_weight * token.evidence_value
            )
            scores.append(score)
        best_index = int(np.argmax(scores))
        selected.append(remaining.pop(best_index))
    return selected


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm
