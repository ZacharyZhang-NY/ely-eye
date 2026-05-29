from __future__ import annotations

from pathlib import Path

import typer
import uvicorn

from .api import create_app
from .benchmarks import (
    benchmark_prediction_report,
    benchmark_registry_report,
    benchmark_sample_report,
    benchmark_source_report,
    run_prediction_benchmark,
)
from .cartridge import CartridgeService
from .cartridge_assets import CartridgeAssetService
from .config import get_settings
from .db import Database
from .embeddings import EmbeddingService
from .evals import HashHopEvaluator, PRDProofSuite
from .ingestion import IngestionService
from .profiles import runtime_profile_report
from .research_modules import ResearchModuleService
from .runtime import QwenRuntime
from .schemas import TrainingAdapterKind
from .training import TrainingService

app = typer.Typer(no_args_is_help=True)


def json_dump(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    uvicorn.run(create_app(), host=host, port=port)


@app.command()
def status() -> None:
    settings = get_settings()
    runtime = QwenRuntime(settings).status()
    db = Database(settings)
    typer.echo(
        {
            "runtime": runtime.model_dump(mode="json"),
            "atoms": db.atom_count(),
            "token_equivalent": db.token_equivalent_sum(),
            "library_token_equivalent": max(
                db.token_equivalent_sum(),
                db.cartridge_token_equivalent_sum(),
            ),
            "cartridges": db.cartridge_count(),
            "data_home": str(settings.home),
        }
    )


@app.command()
def ingest(path: Path, cartridge_name: str | None = None) -> None:
    result = IngestionService(get_settings()).ingest_path(path, cartridge_name)
    typer.echo(result.model_dump_json(indent=2))


@app.command("build-dense-index")
def build_dense_index() -> None:
    index = EmbeddingService(get_settings()).build_index()
    dimensions = int(index.vectors.shape[1]) if index.vectors.size else 0
    typer.echo({"atoms": len(index.atom_ids), "dimensions": dimensions})


@app.command("list-cartridges")
def list_cartridges() -> None:
    for manifest in CartridgeService(get_settings()).list_manifests():
        typer.echo(manifest.model_dump_json(indent=2))


@app.command("hashhop")
def hashhop(kind: str = "visual", hops: int = 2, token_equivalent: int = 262_144) -> None:
    evaluator = HashHopEvaluator(get_settings())
    if kind == "text":
        proof = evaluator.generate_text_proof(hops, token_equivalent)
    elif kind == "visual":
        proof = evaluator.generate_visual_proof(hops, token_equivalent)
    else:
        raise typer.BadParameter("kind must be text or visual")
    typer.echo(proof.model_dump_json(indent=2))


@app.command("proof-suite")
def proof_suite(cartridge_id: str | None = None) -> None:
    report = PRDProofSuite(get_settings()).run(cartridge_id)
    typer.echo(report.model_dump_json(indent=2))


@app.command("latest-proof-suite")
def latest_proof_suite() -> None:
    report = PRDProofSuite(get_settings()).latest()
    if report is None:
        raise typer.Exit(code=1)
    typer.echo(report.model_dump_json(indent=2))


@app.command("runtime-generation-proof")
def runtime_generation_proof() -> None:
    check = PRDProofSuite(get_settings()).runtime_generation_proof()
    typer.echo(check.model_dump_json(indent=2))


@app.command("visual-contradiction-proof")
def visual_contradiction_proof() -> None:
    check = PRDProofSuite(get_settings()).visual_contradiction_proof()
    typer.echo(check.model_dump_json(indent=2))


@app.command("finalize-cartridge")
def finalize_cartridge(cartridge_id: str) -> None:
    report = CartridgeAssetService(get_settings()).finalize(cartridge_id)
    typer.echo(report.model_dump_json(indent=2))


@app.command("refresh-cartridge")
def refresh_cartridge(cartridge_id: str) -> None:
    settings = get_settings()
    db = Database(settings)
    manifest = CartridgeService(settings, db).rebuild(
        cartridge_id=cartridge_id,
        atoms=db.list_atoms(),
        sources=db.list_sources(),
    )
    typer.echo(manifest.model_dump_json(indent=2))


@app.command("benchmark-registry")
def benchmark_registry() -> None:
    typer.echo(benchmark_registry_report().model_dump_json(indent=2))


@app.command("benchmark-sources")
def benchmark_sources() -> None:
    typer.echo(benchmark_source_report().model_dump_json(indent=2))


@app.command("benchmark-samples")
def benchmark_samples() -> None:
    settings = get_settings()
    output_dir = settings.data_dir / "benchmark_samples"
    typer.echo(benchmark_sample_report(output_dir).model_dump_json(indent=2))


@app.command("benchmark-predictions")
def benchmark_predictions() -> None:
    settings = get_settings()
    output_dir = settings.data_dir / "benchmark_predictions"
    typer.echo(benchmark_prediction_report(settings, output_dir).model_dump_json(indent=2))


@app.command("run-benchmark")
def run_benchmark(
    benchmark: str,
    dataset_path: Path,
    prediction_field: str = "prediction",
    answer_field: str = "answer",
    limit: int | None = None,
) -> None:
    report = run_prediction_benchmark(
        benchmark=benchmark,
        dataset_path=dataset_path,
        prediction_field=prediction_field,
        answer_field=answer_field,
        limit=limit,
    )
    typer.echo(report)


@app.command("runtime-profiles")
def runtime_profiles() -> None:
    typer.echo(runtime_profile_report().model_dump_json(indent=2))


@app.command("research-modules")
def research_modules(query: str = "Ely-Eye V-NSA DVKV x-modal KV speculative prefetch evidence graph") -> None:
    settings = get_settings()
    db = Database(settings)
    from .compiler import ContextCompiler
    from .schemas import RuntimeProfile

    context = ContextCompiler(settings, db).compile(query, RuntimeProfile.research_theater)
    report = ResearchModuleService(settings, db).build_report(query, context.hits)
    typer.echo(json_dump(report))


@app.command("validate-training-data")
def validate_training_data(path: Path) -> None:
    samples = TrainingService(get_settings()).validate_dataset(path)
    typer.echo({"samples": len(samples)})


@app.command("build-prd-training-data")
def build_prd_training_data(
    prd_path: Path = Path("PRD.md"),
    output_path: Path | None = None,
    cartridge_id: str | None = None,
    max_sections: int = 48,
) -> None:
    path = TrainingService(get_settings()).build_prd_dataset(
        prd_path=prd_path,
        output_path=output_path,
        cartridge_id=cartridge_id,
        max_sections=max_sections,
    )
    samples = TrainingService(get_settings()).validate_dataset(path)
    typer.echo({"dataset": str(path), "samples": len(samples)})


@app.command("build-visual-training-data")
def build_visual_training_data(
    prd_path: Path = Path("PRD.md"),
    output_path: Path | None = None,
    cartridge_id: str | None = None,
    image_path: list[Path] | None = typer.Option(None, "--image-path"),
) -> None:
    path = TrainingService(get_settings()).build_visual_dataset(
        prd_path=prd_path,
        output_path=output_path,
        image_paths=image_path,
        cartridge_id=cartridge_id,
    )
    samples = TrainingService(get_settings()).validate_dataset(path)
    typer.echo({"dataset": str(path), "samples": len(samples)})


@app.command("build-router-training-data")
def build_router_training_data(
    prd_path: Path = Path("PRD.md"),
    output_path: Path | None = None,
    cartridge_id: str | None = None,
) -> None:
    path = TrainingService(get_settings()).build_router_dataset(
        prd_path=prd_path,
        output_path=output_path,
        cartridge_id=cartridge_id,
    )
    samples = TrainingService(get_settings()).validate_dataset(path)
    typer.echo({"dataset": str(path), "samples": len(samples)})


@app.command("build-retrieval-training-data")
def build_retrieval_training_data(
    prd_path: Path = Path("PRD.md"),
    output_path: Path | None = None,
    cartridge_id: str | None = None,
    max_sections: int = 32,
) -> None:
    path = TrainingService(get_settings()).build_retrieval_dataset(
        prd_path=prd_path,
        output_path=output_path,
        cartridge_id=cartridge_id,
        max_sections=max_sections,
    )
    samples = TrainingService(get_settings()).validate_dataset(path)
    typer.echo({"dataset": str(path), "samples": len(samples)})


@app.command("train-lora")
def train_lora(
    dataset_path: Path,
    output_dir: Path,
    max_steps: int = 8,
    kind: TrainingAdapterKind = TrainingAdapterKind.hme_core_lora,
    model_id: str | None = None,
    rank: int = 16,
    alpha: int = 32,
    max_length: int | None = None,
    gradient_accumulation_steps: int | None = None,
    learning_rate: float | None = None,
    cartridge_id: str | None = None,
) -> None:
    manifest = TrainingService(get_settings()).train_lora(
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_steps=max_steps,
        kind=kind,
        model_id=model_id,
        rank=rank,
        alpha=alpha,
        max_length=max_length,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        cartridge_id=cartridge_id,
    )
    if cartridge_id:
        CartridgeService(get_settings()).attach_adapter(cartridge_id, output_dir)
    typer.echo(manifest.model_dump_json(indent=2))


@app.command("train-retrieval-lora")
def train_retrieval_lora(
    dataset_path: Path,
    output_dir: Path,
    max_steps: int = 8,
    rank: int = 8,
    alpha: int = 16,
    max_length: int | None = None,
    learning_rate: float | None = None,
    cartridge_id: str | None = None,
) -> None:
    manifest = TrainingService(get_settings()).train_retrieval_lora(
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_steps=max_steps,
        rank=rank,
        alpha=alpha,
        max_length=max_length,
        learning_rate=learning_rate,
        cartridge_id=cartridge_id,
    )
    if cartridge_id:
        CartridgeService(get_settings()).attach_adapter(cartridge_id, output_dir)
    typer.echo(manifest.model_dump_json(indent=2))


@app.command("attach-adapter")
def attach_adapter(cartridge_id: str, adapter_dir: Path) -> None:
    manifest = CartridgeService(get_settings()).attach_adapter(cartridge_id, adapter_dir)
    typer.echo(manifest.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
