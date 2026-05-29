from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from .config import Settings, get_settings
from .db import Database
from .schemas import CacheLayerStatus


class CacheFabric:
    def __init__(self, settings: Settings | None = None, database: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = database or Database(self.settings)

    def semantic_key(self, question: str, profile: str) -> str:
        return hashlib.sha256(f"compiler-v2\n{profile}\n{question}".encode("utf-8")).hexdigest()

    def record_hit(self, layer: str, cache_key: str, bytes_estimate: int = 0, metadata: dict[str, Any] | None = None) -> None:
        self.db.log_cache_event(layer, cache_key, "hit", bytes_estimate, metadata)

    def record_miss(self, layer: str, cache_key: str, bytes_estimate: int = 0, metadata: dict[str, Any] | None = None) -> None:
        self.db.log_cache_event(layer, cache_key, "miss", bytes_estimate, metadata)

    def write_semantic_cache(self, cache_key: str, payload: dict[str, Any]) -> None:
        path = self.settings.cache_dir / "semantic" / f"{cache_key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        path.write_bytes(data)
        self.db.log_cache_event("semantic", cache_key, "write", len(data), {"path": str(path)})

    def read_semantic_cache(self, cache_key: str) -> dict[str, Any] | None:
        path = self.settings.cache_dir / "semantic" / f"{cache_key}.json"
        if not path.exists():
            self.record_miss("semantic", cache_key)
            return None
        self.record_hit("semantic", cache_key, path.stat().st_size)
        return json.loads(path.read_text(encoding="utf-8"))

    def statuses(self) -> list[CacheLayerStatus]:
        rows = self.db.cache_layer_rows()
        statuses: list[CacheLayerStatus] = []
        for row in rows:
            last = row["last_event_at"]
            statuses.append(
                CacheLayerStatus(
                    layer=row["layer"],
                    entries=int(row["entries"] or 0),
                    bytes_estimate=int(row["bytes_estimate"] or 0),
                    hit_count=int(row["hit_count"] or 0),
                    miss_count=int(row["miss_count"] or 0),
                    last_event_at=datetime.fromisoformat(last) if last else None,
                )
            )
        return statuses
