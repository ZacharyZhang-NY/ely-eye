from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


class OcrResult:
    def __init__(self, text: str, confidence: float, regions: list[dict[str, Any]]) -> None:
        self.text = text
        self.confidence = confidence
        self.regions = regions


@lru_cache(maxsize=1)
def _rapid_ocr() -> Any:
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def read_text_from_image(path: Path) -> OcrResult:
    engine = _rapid_ocr()
    result = engine(str(path))
    raw = result[0] if isinstance(result, tuple) else result
    if raw is None:
        return OcrResult(text="", confidence=0.0, regions=[])

    texts: list[str] = []
    confidences: list[float] = []
    regions: list[dict[str, Any]] = []
    for item in raw:
        if len(item) < 3:
            continue
        box, text, confidence = item[0], str(item[1]), float(item[2])
        texts.append(text)
        confidences.append(confidence)
        regions.append({"box": box, "text": text, "confidence": confidence})

    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return OcrResult(text="\n".join(texts), confidence=average_confidence, regions=regions)
