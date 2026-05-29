from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import cv2
from PIL import Image

from .config import Settings, get_settings
from .db import Database
from .ocr import read_text_from_image
from .schemas import (
    EvidenceAtom,
    IngestResponse,
    LayoutBox,
    Modality,
    SourceRecord,
    TrustScores,
)
from .storage import ObjectStore, guess_mime, safe_source_id, sha256_bytes, sha256_file, token_equivalent


SUPPORTED_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".html",
    ".css",
    ".xml",
}

SUPPORTED_CODE_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".rs",
    ".go",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".swift",
    ".kt",
    ".sql",
}

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
REPOSITORY_INDEX_LIMIT = 5000
REPOSITORY_COMMIT_LIMIT = 50


@dataclass
class IngestBatch:
    sources: list[SourceRecord]
    atoms: list[EvidenceAtom]

    @property
    def token_equivalent(self) -> int:
        return sum(atom.token_equivalent for atom in self.atoms)


class IngestionService:
    def __init__(
        self,
        settings: Settings | None = None,
        database: Database | None = None,
        object_store: ObjectStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = database or Database(self.settings)
        self.objects = object_store or ObjectStore(self.settings)

    def ingest_path(self, path: Path, cartridge_name: str | None = None) -> IngestResponse:
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        files = self._discover_files(path)
        batch = IngestBatch(sources=[], atoms=[])
        for file_path in files:
            file_batch = self._ingest_file(file_path)
            batch.sources.extend(file_batch.sources)
            batch.atoms.extend(file_batch.atoms)
        if path.is_dir():
            repository_batch = self._ingest_repository_context(path, files)
            batch.sources.extend(repository_batch.sources)
            batch.atoms.extend(repository_batch.atoms)

        self.db.upsert_sources(batch.sources)
        self.db.upsert_atoms(batch.atoms)

        cartridge_id = None
        if cartridge_name:
            from .cartridge import CartridgeService

            cartridge = CartridgeService(self.settings, self.db).materialize(
                name=cartridge_name,
                atoms=batch.atoms,
                sources=batch.sources,
            )
            cartridge_id = cartridge.cartridge_id

        return IngestResponse(
            source_count=len(batch.sources),
            atom_count=len(batch.atoms),
            cartridge_id=cartridge_id,
            token_equivalent=batch.token_equivalent,
            sources=batch.sources,
        )

    def _discover_files(self, path: Path) -> list[Path]:
        if path.is_file():
            return [path]
        ignored_dirs = {".git", ".ely_eye", "node_modules", ".venv", ".venv-linux", "__pycache__", "dist", "build", "target"}
        files: list[Path] = []
        for root, dirs, names in os.walk(path):
            dirs[:] = [name for name in dirs if name not in ignored_dirs and not name.startswith(".venv")]
            root_path = Path(root)
            for name in names:
                child = root_path / name
                if any(part in ignored_dirs or part.startswith(".venv") for part in child.parts):
                    continue
                try:
                    if child.is_dir():
                        continue
                    child.stat()
                except OSError:
                    continue
                files.append(child)
        return files

    def _ingest_file(self, path: Path) -> IngestBatch:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._ingest_pdf(path)
        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            return self._ingest_image(path)
        if suffix in SUPPORTED_VIDEO_EXTENSIONS:
            return self._ingest_video(path)
        if suffix in SUPPORTED_CODE_EXTENSIONS:
            return self._ingest_code(path)
        if suffix in SUPPORTED_TEXT_EXTENSIONS:
            return self._ingest_text(path, Modality.text)
        return IngestBatch(sources=[], atoms=[])

    def _source_record(self, path: Path, kind: str, metadata: dict[str, Any] | None = None) -> SourceRecord:
        digest = sha256_file(path)
        return SourceRecord(
            source_id=safe_source_id(path, digest),
            path=str(path),
            kind=kind,
            mime=guess_mime(path),
            sha256=digest,
            size_bytes=path.stat().st_size,
            metadata=metadata or {},
        )

    def _ingest_text(self, path: Path, modality: Modality) -> IngestBatch:
        if path.stat().st_size > self.settings.max_text_file_bytes:
            return IngestBatch(sources=[], atoms=[])
        source = self._source_record(path, modality.value)
        text = path.read_text(encoding="utf-8", errors="replace")
        atom = EvidenceAtom(
            atom_id=f"{source.source_id}:text",
            modality=modality,
            source_id=source.source_id,
            source=str(path),
            text=text,
            trust=TrustScores(parser=1.0, ocr=0.0, model=0.0),
            token_equivalent=token_equivalent(text),
            metadata={"extension": path.suffix.lower()},
        )
        return IngestBatch(sources=[source], atoms=[atom])

    def _ingest_code(self, path: Path) -> IngestBatch:
        batch = self._ingest_text(path, Modality.code)
        if not batch.atoms:
            return batch
        source_text = batch.atoms[0].text
        batch.atoms[0].metadata["symbols"] = extract_symbols(path, source_text)
        batch.atoms[0].metadata["line_count"] = source_text.count("\n") + 1
        return batch

    def _ingest_image(self, path: Path) -> IngestBatch:
        source = self._source_record(path, self._image_kind(path), metadata=image_metadata(path))
        image_ref = self.objects.put_file(path, "images")
        ocr = read_text_from_image(path)
        width, height = image_size(path)
        atom = EvidenceAtom(
            atom_id=f"{source.source_id}:image",
            modality=Modality.ui_screenshot if source.kind == "ui_screenshot" else Modality.image,
            source_id=source.source_id,
            source=str(path),
            text=ocr.text,
            image_ref=image_ref,
            layout=LayoutBox(x=0, y=0, w=width, h=height),
            trust=TrustScores(parser=1.0, ocr=ocr.confidence, model=0.0),
            token_equivalent=token_equivalent(ocr.text) + max(1, (width * height) // (224 * 224)),
            metadata={"ocr_regions": ocr.regions, **image_metadata(path)},
        )
        return IngestBatch(sources=[source], atoms=[atom])

    def _ingest_pdf(self, path: Path) -> IngestBatch:
        source = self._source_record(path, "pdf")
        atoms: list[EvidenceAtom] = []
        with fitz.open(path) as document:
            source.metadata["page_count"] = document.page_count
            for page_index, page in enumerate(document, start=1):
                rect = page.rect
                text = page.get_text("text").strip()
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                image_bytes = pixmap.tobytes("png")
                image_ref = self.objects.put_bytes(image_bytes, ".png", "pdf_pages")
                if not text:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                        temp_file.write(image_bytes)
                        temp_path = Path(temp_file.name)
                    try:
                        ocr = read_text_from_image(temp_path)
                        text = ocr.text
                        ocr_confidence = ocr.confidence
                    finally:
                        temp_path.unlink(missing_ok=True)
                else:
                    ocr_confidence = 0.0
                atoms.append(
                    EvidenceAtom(
                        atom_id=f"{source.source_id}:page:{page_index:05d}",
                        modality=Modality.pdf_page,
                        source_id=source.source_id,
                        source=f"{path}#page={page_index}",
                        text=text,
                        image_ref=image_ref,
                        layout=LayoutBox(x=0, y=0, w=float(rect.width), h=float(rect.height), page=page_index),
                        trust=TrustScores(parser=1.0, ocr=ocr_confidence, model=0.0),
                        token_equivalent=token_equivalent(text) + max(1, int(rect.width * rect.height) // 50_000),
                        metadata={"page": page_index, "width": rect.width, "height": rect.height},
                    )
                )
                atoms.extend(self._pdf_region_atoms(path, source.source_id, page_index, page))
        return IngestBatch(sources=[source], atoms=atoms)

    def _pdf_region_atoms(
        self, path: Path, source_id: str, page_index: int, page: fitz.Page
    ) -> list[EvidenceAtom]:
        regions: list[EvidenceAtom] = []
        blocks = page.get_text("blocks")
        for block_index, block in enumerate(blocks):
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            text = str(text).strip()
            if not text:
                continue
            regions.append(
                EvidenceAtom(
                    atom_id=f"{source_id}:page:{page_index:05d}:region:{block_index:04d}",
                    modality=Modality.pdf_region,
                    source_id=source_id,
                    source=f"{path}#page={page_index}&region={block_index}",
                    text=text,
                    layout=LayoutBox(x=float(x0), y=float(y0), w=float(x1 - x0), h=float(y1 - y0), page=page_index),
                    trust=TrustScores(parser=1.0, ocr=0.0, model=0.0),
                    token_equivalent=token_equivalent(text),
                    metadata={"page": page_index, "block": block_index},
                )
            )
        return regions

    def _ingest_video(self, path: Path) -> IngestBatch:
        source = self._source_record(path, "video")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video file: {path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        source.metadata.update({"fps": fps, "frame_count": frame_count, "duration_seconds": duration})

        atoms: list[EvidenceAtom] = []
        step = max(1, int(fps * self.settings.video_sample_seconds)) if fps > 0 else 1
        frame_index = 0
        while frame_index < frame_count:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                break
            second = frame_index / fps if fps > 0 else 0.0
            success, encoded = cv2.imencode(".png", frame)
            if not success:
                frame_index += step
                continue
            image_bytes = encoded.tobytes()
            image_ref = self.objects.put_bytes(image_bytes, ".png", "video_frames")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_file.write(image_bytes)
                temp_path = Path(temp_file.name)
            try:
                ocr = read_text_from_image(temp_path)
            finally:
                temp_path.unlink(missing_ok=True)
            height, width = frame.shape[:2]
            atoms.append(
                EvidenceAtom(
                    atom_id=f"{source.source_id}:frame:{frame_index:08d}",
                    modality=Modality.video_frame,
                    source_id=source.source_id,
                    source=f"{path}#t={second:.3f}",
                    text=ocr.text,
                    image_ref=image_ref,
                    layout=LayoutBox(x=0, y=0, w=float(width), h=float(height), frame_second=second),
                    trust=TrustScores(parser=1.0, ocr=ocr.confidence, model=0.0),
                    token_equivalent=token_equivalent(ocr.text) + max(1, (width * height) // (224 * 224)),
                    metadata={"frame_index": frame_index, "second": second, "ocr_regions": ocr.regions},
                )
            )
            frame_index += step
        capture.release()
        return IngestBatch(sources=[source], atoms=atoms)

    def _image_kind(self, path: Path) -> str:
        name = path.name.lower()
        ui_markers = ("screenshot", "screen", "figma", "ui", "wireframe", "mockup")
        return "ui_screenshot" if any(marker in name for marker in ui_markers) else "image"

    def _ingest_repository_context(self, root: Path, files: list[Path]) -> IngestBatch:
        code_files = [path for path in files if path.suffix.lower() in SUPPORTED_CODE_EXTENSIONS]
        git_root = find_git_root(root)
        if not code_files and git_root is None:
            return IngestBatch(sources=[], atoms=[])

        context = build_repository_context(root, files, code_files, git_root)
        payload = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = sha256_bytes(payload)
        source = SourceRecord(
            source_id=safe_source_id(root, digest),
            path=str(root),
            kind="code_repository",
            mime="application/vnd.ely-eye.repository+json",
            sha256=digest,
            size_bytes=len(payload),
            metadata={
                "file_count": context["structure"]["file_count"],
                "indexed_file_count": context["structure"]["indexed_file_count"],
                "code_file_count": context["code"]["file_count"],
                "git_available": context["git"]["available"],
            },
        )
        atom = EvidenceAtom(
            atom_id=f"{source.source_id}:text",
            modality=Modality.code,
            source_id=source.source_id,
            source=str(root),
            text=json.dumps(context, ensure_ascii=False, indent=2),
            trust=TrustScores(parser=1.0, ocr=0.0, model=0.0),
            token_equivalent=token_equivalent(json.dumps(context, ensure_ascii=False)),
            metadata=context,
        )
        return IngestBatch(sources=[source], atoms=[atom])


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def image_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        return {"width": image.width, "height": image.height, "mode": image.mode, "format": image.format}


def build_repository_context(
    root: Path,
    files: list[Path],
    code_files: list[Path],
    git_root: Path | None,
) -> dict[str, Any]:
    structure_entries = [file_structure_entry(root, path) for path in sorted(files)[:REPOSITORY_INDEX_LIMIT]]
    code_entries = [code_context_entry(root, path) for path in sorted(code_files)[:REPOSITORY_INDEX_LIMIT]]
    git_context = collect_git_context(git_root) if git_root is not None else {"available": False}
    return {
        "format": "ely-eye-repository-context-v1",
        "root": str(root),
        "structure": {
            "file_count": len(files),
            "indexed_file_count": len(structure_entries),
            "limit": REPOSITORY_INDEX_LIMIT,
            "files": structure_entries,
        },
        "code": {
            "file_count": len(code_files),
            "indexed_file_count": len(code_entries),
            "files": code_entries,
            "call_graph": repository_call_graph(code_entries),
        },
        "git": git_context,
        "issue_pr_history": extract_issue_pr_history(git_context),
    }


def file_structure_entry(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(root, path),
        "suffix": path.suffix.lower(),
        "kind": file_kind(path),
        "size_bytes": path.stat().st_size,
    }


def code_context_entry(root: Path, path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    symbols = extract_symbols(path, text)
    return {
        "path": relative_path(root, path),
        "language": path.suffix.lower().lstrip("."),
        "line_count": text.count("\n") + 1,
        "symbols": symbols,
    }


def repository_call_graph(code_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for entry in code_entries:
        symbols = entry.get("symbols") if isinstance(entry.get("symbols"), dict) else {}
        for call in symbols.get("calls", []):
            if not isinstance(call, dict):
                continue
            caller = call.get("caller")
            callee = call.get("callee")
            if caller and callee:
                edges.append({"path": entry["path"], "caller": caller, "callee": callee})
    return edges


def collect_git_context(root: Path) -> dict[str, Any]:
    commits = [
        parse_git_log_line(line)
        for line in run_git(root, ["log", f"--max-count={REPOSITORY_COMMIT_LIMIT}", "--pretty=format:%H%x09%an%x09%aI%x09%s"]).splitlines()
        if line.strip()
    ]
    commits = [commit for commit in commits if commit]
    return {
        "available": True,
        "root": str(root),
        "head": run_git(root, ["rev-parse", "HEAD"]).strip() or None,
        "branch": run_git(root, ["branch", "--show-current"]).strip() or None,
        "remotes": parse_git_remotes(run_git(root, ["remote", "-v"])),
        "recent_commits": commits,
    }


def parse_git_log_line(line: str) -> dict[str, str] | None:
    parts = line.split("\t", 3)
    if len(parts) != 4:
        return None
    commit_hash, author, authored_at, subject = parts
    return {"hash": commit_hash, "author": author, "authored_at": authored_at, "subject": subject}


def parse_git_remotes(output: str) -> list[dict[str, str]]:
    remotes: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            remotes.append({"name": parts[0], "url": parts[1], "kind": parts[2].strip("()")})
    return remotes


def extract_issue_pr_history(git_context: dict[str, Any]) -> dict[str, Any]:
    references: dict[str, set[int]] = {"issues": set(), "pull_requests": set()}
    for commit in git_context.get("recent_commits", []):
        if not isinstance(commit, dict):
            continue
        subject = str(commit.get("subject") or "")
        for number in re.findall(r"(?:fixes|closes|resolves)\s+#(\d+)", subject, flags=re.IGNORECASE):
            references["issues"].add(int(number))
        for number in re.findall(r"(?:pull request|pr)\s*#?(\d+)", subject, flags=re.IGNORECASE):
            references["pull_requests"].add(int(number))
        for number in re.findall(r"\(#(\d+)\)", subject):
            references["pull_requests"].add(int(number))
    return {
        "source": "local_git_commit_history",
        "issues": sorted(references["issues"]),
        "pull_requests": sorted(references["pull_requests"]),
    }


def run_git(root: Path, args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout if completed.returncode == 0 else ""


def find_git_root(path: Path) -> Path | None:
    current = path if path.is_dir() else path.parent
    while True:
        if (current / ".git").exists():
            return current
        if current.parent == current:
            return None
        current = current.parent


def file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_CODE_EXTENSIONS:
        return "code"
    if suffix in SUPPORTED_TEXT_EXTENSIONS:
        return "text"
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if suffix in SUPPORTED_VIDEO_EXTENSIONS:
        return "video"
    if suffix == ".pdf":
        return "pdf"
    return "binary"


def relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def extract_symbols(path: Path, text: str) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return extract_python_symbols(text)
    symbols = re.findall(
        r"\b(?:class|function|interface|type|enum|struct|fn|func)\s+([A-Za-z_][A-Za-z0-9_]*)",
        text,
    )
    constants = re.findall(r"\b(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", text)
    calls = extract_generic_calls(text)
    return {"symbols": sorted(set(symbols)), "bindings": sorted(set(constants)), "calls": calls}


def extract_python_symbols(text: str) -> dict[str, Any]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {
            "classes": [],
            "functions": [],
            "imports": [],
            "calls": extract_generic_calls(text),
            "parse_error": f"{exc.msg} at line {exc.lineno}",
        }
    classes: list[str] = []
    functions: list[str] = []
    imports: list[str] = []
    calls: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    call_visitor = PythonCallGraphVisitor()
    call_visitor.visit(tree)
    calls.extend(call_visitor.calls)
    return {
        "classes": sorted(set(classes)),
        "functions": sorted(set(functions)),
        "imports": sorted(set(imports)),
        "calls": calls,
    }


class PythonCallGraphVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.current: str | None = None
        self.calls: list[dict[str, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        previous = self.current
        self.current = node.name
        self.generic_visit(node)
        self.current = previous

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        previous = self.current
        self.current = node.name
        self.generic_visit(node)
        self.current = previous

    def visit_Call(self, node: ast.Call) -> Any:
        if self.current:
            callee = python_call_name(node.func)
            if callee:
                self.calls.append({"caller": self.current, "callee": callee})
        self.generic_visit(node)


def python_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = python_call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def extract_generic_calls(text: str) -> list[dict[str, str]]:
    calls: list[dict[str, str]] = []
    current = "<module>"
    function_pattern = re.compile(r"\b(?:function|fn|func)\s+([A-Za-z_][A-Za-z0-9_]*)|([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\([^)]*\)\s*=>")
    call_pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_.]*)\s*\(")
    keywords = {"if", "for", "while", "switch", "catch", "return", "function"}
    for line in text.splitlines():
        function_match = function_pattern.search(line)
        if function_match:
            current = next(group for group in function_match.groups() if group)
        for callee in call_pattern.findall(line):
            root = callee.split(".", 1)[0]
            if root not in keywords and callee != current:
                calls.append({"caller": current, "callee": callee})
    return calls


def write_sources_json(path: Path, sources: list[SourceRecord]) -> None:
    path.write_text(
        json.dumps([source.model_dump(mode="json") for source in sources], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
