from __future__ import annotations

import json
import math
import re
import tempfile
from ast import literal_eval
from pathlib import Path
from typing import Any

from .schemas import (
    BenchmarkPredictionReport,
    BenchmarkPredictionStatus,
    BenchmarkRegistryReport,
    BenchmarkSampleReport,
    BenchmarkSampleStatus,
    BenchmarkSourceReport,
    BenchmarkSourceStatus,
    BenchmarkSpec,
)


STANDARD_BENCHMARKS = (
    BenchmarkSpec(
        name="MMMU-Pro",
        target="Multimodal reasoning within Qwen3.5-9B baseline +/-2%.",
        dataset="MMMU/MMMU_Pro",
        task_type="image_text_multiple_choice",
        runner="ely-eye run-benchmark --benchmark mmmu-pro --dataset-path <local-jsonl-or-parquet>",
        metric="accuracy",
    ),
    BenchmarkSpec(
        name="OCRBench",
        target="OCR and document understanding at or above base model behavior.",
        dataset="echo840/OCRBench",
        task_type="image_text_qa",
        runner="ely-eye run-benchmark --benchmark ocrbench --dataset-path <local-jsonl-or-parquet>",
        metric="exact_or_contains_accuracy",
    ),
    BenchmarkSpec(
        name="MMLongBench-Doc",
        target="Long document QA above a RAG-only baseline.",
        dataset="VLM2Vec/MMLongBench-doc",
        task_type="long_document_qa",
        runner="ely-eye run-benchmark --benchmark mmlongbench-doc --dataset-path <local-jsonl-or-parquet>",
        metric="answer_f1_with_evidence",
    ),
    BenchmarkSpec(
        name="VideoMME",
        target="Long video understanding with sampled visual timeline evidence.",
        dataset="lmms-lab/Video-MME",
        task_type="video_multiple_choice",
        runner="ely-eye run-benchmark --benchmark videomme --dataset-path <local-jsonl-or-parquet>",
        metric="accuracy",
    ),
    BenchmarkSpec(
        name="LongVideoBench",
        target="Long-context interleaved video-language understanding.",
        dataset="longvideobench/LongVideoBench",
        task_type="video_qa",
        runner="ely-eye run-benchmark --benchmark longvideobench --dataset-path <local-jsonl-or-parquet>",
        metric="accuracy",
    ),
    BenchmarkSpec(
        name="RULER",
        target="Long-context retrieval, aggregation, and multi-hop reasoning.",
        dataset="RULER-dataset/RULER",
        task_type="long_context_text",
        runner="ely-eye run-benchmark --benchmark ruler --dataset-path <local-jsonl-or-parquet>",
        metric="task_accuracy",
    ),
    BenchmarkSpec(
        name="LongBench v2",
        target="Long text, multi-document, code, and structured-data reasoning.",
        dataset="THUDM/LongBench-v2",
        task_type="long_context_multiple_choice",
        runner="ely-eye run-benchmark --benchmark longbench-v2 --dataset-path <local-jsonl-or-parquet>",
        metric="accuracy",
    ),
)

BENCHMARK_PREDICTION_NAMES = ("MMMU-Pro", "OCRBench")

BENCHMARK_SAMPLE_LOADS: dict[str, tuple[str | None, str]] = {
    "MMMU-Pro": ("standard (10 options)", "test"),
    "OCRBench": ("default", "test"),
    "MMLongBench-Doc": ("queries", "test"),
    "VideoMME": ("videomme", "test"),
    "LongVideoBench": (None, "test"),
    "RULER": ("default", "test"),
    "LongBench v2": ("default", "train"),
}

BENCHMARK_SAMPLE_SOURCES: dict[str, tuple[str, str | None, str, str | None]] = {
    "LongVideoBench": (
        "longvideobench/LongVideoBench-Meta",
        "default",
        "validation",
        "Official public metadata release for the gated LongVideoBench media dataset.",
    ),
}


def benchmark_registry_report() -> BenchmarkRegistryReport:
    return BenchmarkRegistryReport(status="ready", benchmarks=list(STANDARD_BENCHMARKS))


def benchmark_source_report(timeout: int = 20) -> BenchmarkSourceReport:
    from huggingface_hub import HfApi

    api = HfApi()
    sources: list[BenchmarkSourceStatus] = []
    for benchmark in STANDARD_BENCHMARKS:
        try:
            info = api.dataset_info(benchmark.dataset, timeout=timeout)
            siblings = getattr(info, "siblings", None) or []
            sources.append(
                BenchmarkSourceStatus(
                    name=benchmark.name,
                    dataset=benchmark.dataset,
                    status="ready" if info.sha else "unavailable",
                    sha=info.sha,
                    gated=getattr(info, "gated", None),
                    private=getattr(info, "private", None),
                    sibling_count=len(siblings),
                )
            )
        except Exception as exc:
            sources.append(
                BenchmarkSourceStatus(
                    name=benchmark.name,
                    dataset=benchmark.dataset,
                    status="unavailable",
                    error=str(exc).splitlines()[0],
                )
            )
    status = "ready" if all(source.status == "ready" and source.sha for source in sources) else "incomplete"
    return BenchmarkSourceReport(status=status, sources=sources)


def benchmark_sample_report(output_dir: Path | None = None) -> BenchmarkSampleReport:
    output_root = output_dir or Path(tempfile.mkdtemp(prefix="ely_eye_benchmark_samples_"))
    output_root.mkdir(parents=True, exist_ok=True)
    samples = [load_benchmark_sample(benchmark, output_root) for benchmark in STANDARD_BENCHMARKS]
    loaded = [sample for sample in samples if sample.status == "loaded"]
    gated = [sample for sample in samples if sample.status == "gated"]
    failed = [sample for sample in samples if sample.status == "failed"]
    if len(loaded) == len(samples):
        status = "ready"
    elif loaded and gated and not failed:
        status = "partial"
    else:
        status = "failed"
    return BenchmarkSampleReport(status=status, samples=samples)


def benchmark_prediction_report(
    settings: Any,
    output_dir: Path | None = None,
    names: tuple[str, ...] = BENCHMARK_PREDICTION_NAMES,
) -> BenchmarkPredictionReport:
    from .runtime import QwenRuntime

    output_root = output_dir or Path(tempfile.mkdtemp(prefix="ely_eye_benchmark_predictions_"))
    output_root.mkdir(parents=True, exist_ok=True)
    selected = {name.lower() for name in names}
    benchmarks = [benchmark for benchmark in STANDARD_BENCHMARKS if benchmark.name.lower() in selected]
    prediction_settings = settings.model_copy(update={"max_new_tokens": 96, "temperature": 0.0})
    runtime = QwenRuntime(prediction_settings)
    try:
        predictions = [
            predict_benchmark_sample(benchmark, output_root, runtime)
            for benchmark in benchmarks
        ]
    finally:
        runtime.unload()
    predicted = [prediction for prediction in predictions if prediction.status == "predicted"]
    gated = [prediction for prediction in predictions if prediction.status == "gated"]
    failed = [prediction for prediction in predictions if prediction.status == "failed"]
    if len(predicted) == len(predictions) and predictions:
        status = "ready"
    elif predicted and gated and not failed:
        status = "partial"
    else:
        status = "failed"
    return BenchmarkPredictionReport(status=status, predictions=predictions)


def load_benchmark_sample(benchmark: BenchmarkSpec, output_root: Path) -> BenchmarkSampleStatus:
    source_dataset, config, split, source_note = benchmark_sample_source(benchmark)
    try:
        from datasets import load_dataset

        dataset_kwargs: dict[str, Any] = {"split": split, "streaming": True}
        stream = (
            load_dataset(source_dataset, config, **dataset_kwargs)
            if config
            else load_dataset(source_dataset, **dataset_kwargs)
        )
        row = next(iter(stream))
        sample_root = output_root / slugify(benchmark.name)
        sample_root.mkdir(parents=True, exist_ok=True)
        normalized, media_artifacts = normalize_sample(row, sample_root)
        return BenchmarkSampleStatus(
            name=benchmark.name,
            dataset=benchmark.dataset,
            source_dataset=source_dataset,
            source_sha=dataset_sha(source_dataset),
            source_note=source_note,
            status="loaded",
            config=config,
            split=split,
            sample_id=sample_identifier(row),
            fields=list(row.keys()),
            question_preview=first_present(row, ("question", "query", "text")),
            answer_preview=first_present(row, ("answer", "answers", "la", "label", "correct_choice")),
            sample_sha256=sha256_json(normalized),
            media_artifacts=media_artifacts,
        )
    except Exception as exc:
        message = str(exc).splitlines()[0]
        status = "gated" if "gated dataset" in message.lower() or "authenticated" in message.lower() else "failed"
        return BenchmarkSampleStatus(
            name=benchmark.name,
            dataset=benchmark.dataset,
            source_dataset=source_dataset,
            source_note=source_note,
            status=status,
            config=config,
            split=split,
            error=message,
        )


def predict_benchmark_sample(benchmark: BenchmarkSpec, output_root: Path, runtime: Any) -> BenchmarkPredictionStatus:
    try:
        from datasets import load_dataset

        config, split = BENCHMARK_SAMPLE_LOADS.get(benchmark.name, (None, "test"))
        dataset_kwargs: dict[str, Any] = {"split": split, "streaming": True}
        stream = (
            load_dataset(benchmark.dataset, config, **dataset_kwargs)
            if config
            else load_dataset(benchmark.dataset, **dataset_kwargs)
        )
        row = next(iter(stream))
        sample_root = output_root / slugify(benchmark.name)
        sample_root.mkdir(parents=True, exist_ok=True)
        normalized, media_artifacts = normalize_sample(row, sample_root)
        prediction_media = prepare_prediction_media(benchmark, media_artifacts)
        prompt = benchmark_prompt(benchmark, row)
        generation = runtime.generate_prompt(prompt, [Path(path) for path in prediction_media])
        prediction = extract_answer(generation.text)
        answer = canonical_answer(first_present_raw(row, ("answer", "answers", "la", "label")))
        score = benchmark_score(benchmark, row, prediction, answer) if answer else None
        return BenchmarkPredictionStatus(
            name=benchmark.name,
            dataset=benchmark.dataset,
            status="predicted",
            config=config,
            split=split,
            sample_id=sample_identifier(row),
            task_type=benchmark.task_type,
            metric=benchmark.metric,
            prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
            sample_sha256=sha256_json(normalized),
            prediction=prediction,
            answer=answer,
            score=score,
            runtime_backend=generation.backend,
            adapter_id=generation.adapter_id,
            adapter_sha256=generation.adapter_sha256,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            media_artifacts=prediction_media,
        )
    except Exception as exc:
        message = str(exc).splitlines()[0]
        status = "gated" if "gated dataset" in message.lower() or "authenticated" in message.lower() else "failed"
        config, split = BENCHMARK_SAMPLE_LOADS.get(benchmark.name, (None, None))
        return BenchmarkPredictionStatus(
            name=benchmark.name,
            dataset=benchmark.dataset,
            status=status,
            config=config,
            split=split,
            task_type=benchmark.task_type,
            metric=benchmark.metric,
            error=message,
        )


def normalize_benchmark_name(name: str) -> str:
    return name.lower().replace("_", "-")


def benchmark_sample_source(benchmark: BenchmarkSpec) -> tuple[str, str | None, str, str | None]:
    override = BENCHMARK_SAMPLE_SOURCES.get(benchmark.name)
    if override is not None:
        return override
    config, split = BENCHMARK_SAMPLE_LOADS.get(benchmark.name, (None, "test"))
    return benchmark.dataset, config, split, None


def dataset_sha(dataset: str) -> str | None:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(dataset)
    return str(info.sha) if info.sha else None


def run_prediction_benchmark(
    benchmark: str,
    dataset_path: Path,
    prediction_field: str = "prediction",
    answer_field: str = "answer",
    limit: int | None = None,
) -> dict[str, Any]:
    records = load_records(dataset_path)
    if limit is not None:
        records = records[:limit]
    scored: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        prediction = get_nested(record, prediction_field)
        answer = get_nested(record, answer_field)
        if prediction is None or answer is None:
            continue
        score = exact_or_contains_score(str(prediction), str(answer))
        scored.append(
            {
                "index": index,
                "score": score,
                "prediction": str(prediction),
                "answer": str(answer),
            }
        )
    accuracy = sum(item["score"] for item in scored) / len(scored) if scored else 0.0
    return {
        "benchmark": normalize_benchmark_name(benchmark),
        "dataset_path": str(dataset_path),
        "records": len(records),
        "scored": len(scored),
        "metric": "exact_or_contains_accuracy",
        "score": accuracy,
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(item) for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            for key in ("data", "records", "examples"):
                value = data.get(key)
                if isinstance(value, list):
                    return [dict(item) for item in value if isinstance(item, dict)]
            return [data]
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    raise ValueError(f"Unsupported benchmark dataset format: {path}")


def get_nested(record: dict[str, Any], path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def exact_or_contains_score(prediction: str, answer: str) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_answer = normalize_answer(answer)
    if not normalized_prediction or not normalized_answer:
        return 0.0
    if normalized_prediction == normalized_answer or normalized_answer in normalized_prediction:
        return 1.0
    prediction_tokens = normalized_prediction.split()
    answer_tokens = normalized_answer.split()
    if not prediction_tokens or not answer_tokens:
        return 0.0
    overlap = len(set(prediction_tokens) & set(answer_tokens))
    precision = overlap / len(set(prediction_tokens))
    recall = overlap / len(set(answer_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def benchmark_score(
    benchmark: BenchmarkSpec,
    row: dict[str, Any],
    prediction: str,
    answer: str | None,
) -> float | None:
    if answer is None:
        return None
    if "multiple_choice" in benchmark.task_type:
        option_text = answer_option_text(row, answer)
        if exact_or_contains_score(prediction, answer):
            return 1.0
        if option_text and exact_or_contains_score(prediction, option_text):
            return 1.0
        return 0.0
    return exact_or_contains_score(prediction, answer)


def answer_option_text(row: dict[str, Any], answer: str) -> str | None:
    normalized_answer = answer.strip().upper()
    if len(normalized_answer) != 1 or not normalized_answer.isalpha():
        return None
    options = row.get("options")
    if isinstance(options, str):
        try:
            options = literal_eval(options)
        except (SyntaxError, ValueError):
            options = None
    index = ord(normalized_answer) - ord("A")
    if isinstance(options, (list, tuple)) and 0 <= index < len(options):
        return str(options[index])
    if isinstance(options, dict):
        return canonical_answer(options.get(normalized_answer))
    choice_key = f"choice_{normalized_answer}"
    if choice_key in row:
        return canonical_answer(row[choice_key])
    return None


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\u4e00-\u9fff]+", " ", value.lower())).strip()


def benchmark_prompt(benchmark: BenchmarkSpec, row: dict[str, Any]) -> str:
    question = first_present_raw(row, ("question", "query", "text")) or ""
    options = first_present_raw(row, ("options", "choices")) or ""
    context = first_present_raw(row, ("context", "document", "passage")) or ""
    context = compact_preview(context, 6000)
    media_note = "Use the attached image evidence." if any(is_pil_image(value) for value in row.values()) else ""
    if benchmark.name == "OCRBench":
        return f"""Use /no_think mode. Return compact JSON only.

Benchmark: OCRBench
Task type: image text recognition
Metric: exact_or_contains_accuracy
Use the attached image evidence.

Question:
{question}

Read the exact text in the image. For handwritten English words, return the standard spelling with the same letters.

Return exactly this JSON shape:
{{"answer":"<recognized text>"}}
"""
    return f"""Use /no_think mode. Return compact JSON only.

Benchmark: {benchmark.name}
Task type: {benchmark.task_type}
Metric: {benchmark.metric}
{media_note}

Question:
{question}

Options:
{options}

Context:
{context}

Return exactly this JSON shape:
{{"answer":"<prediction>"}}
"""


def extract_answer(text: str) -> str:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        payload = None
        if start >= 0 and end > start:
            try:
                payload = json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                payload = None
    if isinstance(payload, dict) and payload.get("answer") is not None:
        return str(payload["answer"]).strip()
    return compact_preview(stripped, 500)


def canonical_answer(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return " | ".join(str(item) for item in value)
    return str(value)


def first_present_raw(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def normalize_sample(row: dict[str, Any], output_root: Path) -> tuple[dict[str, Any], list[str]]:
    media_artifacts: list[str] = []
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        normalized_value, artifact = normalize_sample_value(key, value, output_root)
        normalized[key] = normalized_value
        if artifact:
            media_artifacts.append(artifact)
    return normalized, media_artifacts


def prepare_prediction_media(benchmark: BenchmarkSpec, media_artifacts: list[str]) -> list[str]:
    if benchmark.name != "OCRBench":
        return media_artifacts
    return [str(enhance_ocr_benchmark_image(Path(path))) for path in media_artifacts]


def enhance_ocr_benchmark_image(path: Path) -> Path:
    from PIL import Image, ImageOps

    image = Image.open(path).convert("RGB")
    max_side = max(image.size)
    scale = max(1, min(10, math.ceil(768 / max_side)))
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.BICUBIC)
    border = max(8, scale * 2)
    image = ImageOps.expand(image, border=border, fill="white")
    target = path.with_name(f"{path.stem}_ocr_x{scale}{path.suffix}")
    image.save(target)
    return target


def normalize_sample_value(key: str, value: Any, output_root: Path) -> tuple[Any, str | None]:
    if is_pil_image(value):
        digest = image_sha256(value)
        artifact = output_root / f"{slugify(key)}_{digest[:12]}.png"
        value.save(artifact)
        return (
            {
                "type": "image",
                "mode": getattr(value, "mode", None),
                "size": list(getattr(value, "size", ())),
                "sha256": digest,
            },
            str(artifact),
        )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value, None
    if isinstance(value, (list, tuple)):
        return [normalize_sample_value(f"{key}_{index}", item, output_root)[0] for index, item in enumerate(value)], None
    if isinstance(value, dict):
        return {str(item_key): normalize_sample_value(str(item_key), item_value, output_root)[0] for item_key, item_value in value.items()}, None
    return str(value), None


def is_pil_image(value: Any) -> bool:
    return hasattr(value, "save") and hasattr(value, "mode") and hasattr(value, "size")


def image_sha256(image: Any) -> str:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return sha256_bytes(buffer.getvalue())


def sample_identifier(row: dict[str, Any]) -> str | None:
    for key in ("id", "_id", "question_id", "query-id", "text_id", "video_id", "corpus-id"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return None


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return compact_preview(value)
    return None


def compact_preview(value: Any, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:limit]


def sha256_json(value: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower() or "item"
