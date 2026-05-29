from __future__ import annotations

import hashlib
import mimetypes
import shutil
from pathlib import Path

from .config import Settings, get_settings


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def safe_source_id(path: Path, content_hash: str) -> str:
    stem = "".join(char if char.isalnum() else "_" for char in path.stem).strip("_")
    prefix = stem[:48] or "source"
    return f"{prefix}_{content_hash[:16]}"


def token_equivalent(text: str) -> int:
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, ascii_chars // 4 + non_ascii_chars // 2)


class ObjectStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()

    def put_file(self, path: Path, namespace: str) -> str:
        content_hash = sha256_file(path)
        suffix = path.suffix.lower()
        target_dir = self.settings.object_dir / namespace / content_hash[:2]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{content_hash}{suffix}"
        if not target.exists():
            shutil.copy2(path, target)
        return self.to_object_uri(target)

    def put_bytes(self, data: bytes, suffix: str, namespace: str) -> str:
        content_hash = sha256_bytes(data)
        target_dir = self.settings.object_dir / namespace / content_hash[:2]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{content_hash}{suffix}"
        if not target.exists():
            target.write_bytes(data)
        return self.to_object_uri(target)

    def resolve(self, object_uri: str) -> Path:
        prefix = "object://"
        if not object_uri.startswith(prefix):
            raise ValueError(f"Invalid object URI: {object_uri}")
        relative = object_uri[len(prefix) :]
        path = (self.settings.object_dir / relative).resolve()
        root = self.settings.object_dir.resolve()
        if root not in path.parents and path != root:
            raise ValueError(f"Object URI escapes object store: {object_uri}")
        return path

    def to_object_uri(self, path: Path) -> str:
        return "object://" + path.resolve().relative_to(self.settings.object_dir.resolve()).as_posix()
