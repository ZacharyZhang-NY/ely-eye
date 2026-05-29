from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import __version__
from .benchmarks import (
    benchmark_prediction_report,
    benchmark_registry_report,
    benchmark_sample_report,
    benchmark_source_report,
)
from .cache_fabric import CacheFabric
from .cartridge import CartridgeService
from .cartridge_assets import CartridgeAssetService
from .compiler import ContextCompiler
from .config import Settings, get_settings
from .db import Database
from .embeddings import EmbeddingService
from .evals import HashHopEvaluator, PRDProofSuite
from .evidence_graph import EvidenceGraphService
from .ingestion import IngestionService
from .profiles import runtime_profile_report
from .research_modules import ResearchModuleService
from .runtime import QwenRuntime, RuntimeUnavailableError
from .schemas import (
    AdapterManifest,
    AdapterStatus,
    AnswerResponse,
    BenchmarkRegistryReport,
    BenchmarkPredictionReport,
    BenchmarkSampleReport,
    BenchmarkSourceReport,
    CartridgeAssetReport,
    HashHopProof,
    MemoryStatus,
    PathIngestRequest,
    ProofCheck,
    ProofSuiteReport,
    RuntimeProfile,
    RuntimeProfileReport,
    SystemStatus,
    UploadIngestRequest,
)
from .training import safetensor_weight_stats
from .verifier import Verifier, VisualContradictionLens


class ChatRequest(BaseModel):
    question: str
    profile: RuntimeProfile = RuntimeProfile.library_100m


class HashHopRequest(BaseModel):
    kind: str = "visual"
    hops: int = 2
    token_equivalent: int = 262_144


class ProofSuiteRequest(BaseModel):
    cartridge_id: str | None = None


def settings_dep() -> Settings:
    return get_settings()


def create_app() -> FastAPI:
    app = FastAPI(title="Ely-Eye", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/status", response_model=SystemStatus)
    def status(settings: Annotated[Settings, Depends(settings_dep)]) -> SystemStatus:
        db = Database(settings)
        cache = CacheFabric(settings, db)
        runtime = QwenRuntime(settings).status()
        layers = cache.statuses()
        active_token_equivalent = db.token_equivalent_sum()
        memory = MemoryStatus(
            active_atoms=db.atom_count(),
            active_token_equivalent=active_token_equivalent,
            cartridges=db.cartridge_count(),
            library_token_equivalent=max(active_token_equivalent, db.cartridge_token_equivalent_sum()),
            layers=layers,
        )
        return SystemStatus(
            version=__version__,
            runtime=runtime,
            memory=memory,
            data_home=str(settings.home),
            profiles={
                "Live Demo": {"context_tokens": settings.live_context_tokens, "purpose": "stable 262K workflow"},
                "Extreme Context": {"context_tokens": settings.extreme_context_tokens, "purpose": "YaRN 1.01M workflow"},
                "100M Library": {"context_tokens": settings.library_target_tokens, "purpose": "Context Cartridge memory"},
                "Research Theater": {"context_tokens": settings.live_context_tokens, "purpose": "V-NSA, DVKV, TTT-VL proofs"},
            },
        )

    @app.post("/api/ingest/path")
    def ingest_path(
        request: PathIngestRequest,
        settings: Annotated[Settings, Depends(settings_dep)],
    ):
        try:
            return IngestionService(settings).ingest_path(request.path, request.cartridge_name)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/ingest/upload")
    def ingest_upload(
        metadata: Annotated[UploadIngestRequest, Depends()],
        settings: Annotated[Settings, Depends(settings_dep)],
        files: Annotated[list[UploadFile], File()],
    ):
        upload_root = settings.data_dir / "uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []
        total_size = 0
        for uploaded in files:
            target = upload_root / sanitize_filename(uploaded.filename or "upload.bin")
            with target.open("wb") as handle:
                shutil.copyfileobj(uploaded.file, handle)
            total_size += target.stat().st_size
            if total_size > settings.upload_max_bytes:
                raise HTTPException(status_code=413, detail="Upload exceeds configured maximum size.")
            saved_paths.append(target)

        service = IngestionService(settings)
        aggregate = None
        for path in saved_paths:
            result = service.ingest_path(path, None)
            if aggregate is None:
                aggregate = result
            else:
                aggregate.source_count += result.source_count
                aggregate.atom_count += result.atom_count
                aggregate.token_equivalent += result.token_equivalent
                aggregate.sources.extend(result.sources)
        if aggregate is None:
            raise HTTPException(status_code=400, detail="No files were uploaded.")
        if metadata.cartridge_name:
            atoms = Database(settings).list_atoms()
            sources = aggregate.sources
            cartridge = CartridgeService(settings).materialize(metadata.cartridge_name, atoms, sources)
            aggregate.cartridge_id = cartridge.cartridge_id
        return aggregate

    @app.post("/api/chat", response_model=AnswerResponse)
    def chat(request: ChatRequest, settings: Annotated[Settings, Depends(settings_dep)]) -> AnswerResponse:
        db = Database(settings)
        compiler = ContextCompiler(settings, db)
        context = compiler.compile(request.question, request.profile)
        if not context.hits:
            raise HTTPException(status_code=404, detail="No evidence atoms are available for this question.")
        runtime = QwenRuntime(settings)
        try:
            generation = runtime.generate(request.question, context)
        except RuntimeUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        answer, citations, report = Verifier().verify_answer(generation.text, context)
        contradiction_notes = VisualContradictionLens().detect(context)
        report.contradiction_notes.extend(note for note in contradiction_notes if note not in report.contradiction_notes)
        EvidenceGraphService(settings, db).record_answer(request.question, answer, report, context)
        return AnswerResponse(
            answer=answer,
            citations=citations,
            context=context,
            verifier=report,
            runtime_backend=generation.backend,
        )

    @app.get("/api/cartridges")
    def cartridges(settings: Annotated[Settings, Depends(settings_dep)]):
        return CartridgeService(settings).list_manifests()

    @app.post("/api/cartridges/{cartridge_id}/finalize", response_model=CartridgeAssetReport)
    def finalize_cartridge(
        cartridge_id: str,
        settings: Annotated[Settings, Depends(settings_dep)],
    ) -> CartridgeAssetReport:
        try:
            return CartridgeAssetService(settings).finalize(cartridge_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/cartridges/{cartridge_id}/refresh")
    def refresh_cartridge(
        cartridge_id: str,
        settings: Annotated[Settings, Depends(settings_dep)],
    ):
        db = Database(settings)
        try:
            return CartridgeService(settings, db).rebuild(
                cartridge_id=cartridge_id,
                atoms=db.list_atoms(),
                sources=db.list_sources(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/cartridges/{cartridge_id}/assets", response_model=CartridgeAssetReport)
    def cartridge_assets(
        cartridge_id: str,
        settings: Annotated[Settings, Depends(settings_dep)],
    ) -> CartridgeAssetReport:
        try:
            report = CartridgeAssetService(settings).status(cartridge_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if report is None:
            raise HTTPException(status_code=404, detail="Cartridge assets have not been finalized.")
        return report

    @app.get("/api/adapters", response_model=list[AdapterStatus])
    def adapters(settings: Annotated[Settings, Depends(settings_dep)]) -> list[AdapterStatus]:
        return list_adapter_statuses(settings)

    @app.get("/api/benchmarks/registry", response_model=BenchmarkRegistryReport)
    def benchmarks_registry() -> BenchmarkRegistryReport:
        return benchmark_registry_report()

    @app.get("/api/benchmarks/sources", response_model=BenchmarkSourceReport)
    def benchmarks_sources() -> BenchmarkSourceReport:
        return benchmark_source_report()

    @app.get("/api/benchmarks/samples", response_model=BenchmarkSampleReport)
    def benchmarks_samples(settings: Annotated[Settings, Depends(settings_dep)]) -> BenchmarkSampleReport:
        return benchmark_sample_report(settings.data_dir / "benchmark_samples")

    @app.get("/api/benchmarks/predictions", response_model=BenchmarkPredictionReport)
    def benchmarks_predictions(settings: Annotated[Settings, Depends(settings_dep)]) -> BenchmarkPredictionReport:
        return benchmark_prediction_report(settings, settings.data_dir / "benchmark_predictions")

    @app.get("/api/runtime/profiles", response_model=RuntimeProfileReport)
    def runtime_profiles() -> RuntimeProfileReport:
        return runtime_profile_report()

    @app.get("/api/research/modules")
    def research_modules(settings: Annotated[Settings, Depends(settings_dep)]):
        db = Database(settings)
        compiler = ContextCompiler(settings, db)
        context = compiler.compile(
            "Ely-Eye V-NSA DVKV x-modal KV speculative prefetch evidence graph",
            RuntimeProfile.research_theater,
        )
        return ResearchModuleService(settings, db).build_report(context.plan.question, context.hits)

    @app.post("/api/index/dense")
    def build_dense_index(settings: Annotated[Settings, Depends(settings_dep)]):
        service = EmbeddingService(settings)
        index = service.build_index()
        return {"atoms": len(index.atom_ids), "dimensions": int(index.vectors.shape[1]) if index.vectors.size else 0}

    @app.post("/api/hashhop", response_model=HashHopProof)
    def hashhop(request: HashHopRequest, settings: Annotated[Settings, Depends(settings_dep)]) -> HashHopProof:
        evaluator = HashHopEvaluator(settings)
        if request.kind == "text":
            return evaluator.generate_text_proof(request.hops, request.token_equivalent)
        if request.kind == "visual":
            return evaluator.generate_visual_proof(request.hops, request.token_equivalent)
        raise HTTPException(status_code=400, detail="kind must be text or visual")

    @app.post("/api/proof-suite", response_model=ProofSuiteReport)
    def proof_suite(
        request: ProofSuiteRequest,
        settings: Annotated[Settings, Depends(settings_dep)],
    ) -> ProofSuiteReport:
        try:
            return PRDProofSuite(settings).run(request.cartridge_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/proof-suite/latest", response_model=ProofSuiteReport)
    def latest_proof_suite(settings: Annotated[Settings, Depends(settings_dep)]) -> ProofSuiteReport:
        report = PRDProofSuite(settings).latest()
        if report is None:
            raise HTTPException(status_code=404, detail="No PRD proof suite has been generated.")
        return report

    @app.post("/api/visual-contradiction-proof", response_model=ProofCheck)
    def visual_contradiction_proof(settings: Annotated[Settings, Depends(settings_dep)]) -> ProofCheck:
        return PRDProofSuite(settings).visual_contradiction_proof()

    @app.get("/api/memory-map")
    def memory_map(settings: Annotated[Settings, Depends(settings_dep)]):
        db = Database(settings)
        atoms = db.list_atoms(limit=200)
        return {
            "atoms": [atom.model_dump(mode="json") for atom in atoms],
            "layers": CacheFabric(settings, db).statuses(),
            "cartridges": CartridgeService(settings, db).list_manifests(),
        }

    return app


def sanitize_filename(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in {".", "-", "_"} else "_" for char in name)
    return safe.strip("._") or "upload.bin"


def list_adapter_statuses(settings: Settings) -> list[AdapterStatus]:
    cartridge_bindings = {
        manifest.cartridge_id: manifest.artifacts
        for manifest in CartridgeService(settings).list_manifests()
    }
    statuses: list[AdapterStatus] = []
    for manifest_path in sorted(settings.adapters_dir.glob("*/adapter_manifest.json")):
        root = manifest_path.parent
        manifest = AdapterManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        weights = root / "adapter_model.safetensors"
        weight_bytes = weights.stat().st_size if weights.exists() else 0
        weight_stats = safetensor_weight_stats(weights)
        artifacts = cartridge_bindings.get(manifest.cartridge_id or "", {})
        cartridge_bound = f"adapter_{manifest.kind.value}" in artifacts
        statuses.append(
            AdapterStatus(
                name=root.name,
                kind=manifest.kind,
                proof_version=manifest.proof_version,
                training_method=manifest.training_method,
                precision=manifest.precision,
                framework=manifest.framework,
                base_model=manifest.base_model,
                path=str(root),
                adapter_id=manifest.adapter_id,
                sample_count=manifest.sample_count,
                max_steps=manifest.max_steps,
                trainable_params=manifest.trainable_params,
                final_loss=manifest.final_loss,
                sha256=manifest.sha256,
                weight_bytes=weight_bytes,
                safetensor_keys=int(weight_stats["tensor_count"]),
                cartridge_id=manifest.cartridge_id,
                cartridge_bound=cartridge_bound,
                training_trace_path=manifest.training_trace_path,
                training_trace_sha256=manifest.training_trace_sha256,
                training_trace_steps=count_training_steps(root / "training_trace.jsonl"),
                training_proof_path=manifest.training_proof_path,
                training_proof_sha256=manifest.training_proof_sha256,
                optimizer=manifest.optimizer,
                optimizer_family=manifest.optimizer_family,
                gradient_checkpointing=manifest.gradient_checkpointing,
                gradient_checkpointing_mode=manifest.gradient_checkpointing_mode,
                autocast_dtype=manifest.autocast_dtype,
                unsloth_version=manifest.unsloth_version,
                triton_version=manifest.triton_version,
                xformers_version=manifest.xformers_version,
                unsloth_model_type=manifest.unsloth_model_type,
                optimizer_update_count=manifest.optimizer_update_count,
                max_optimizer_update_l2=manifest.max_optimizer_update_l2,
                loss_delta=manifest.loss_delta,
                training_summary_path=manifest.training_summary_path,
                training_wall_seconds=manifest.training_wall_seconds,
                training_device=manifest.training_device,
                bf16=manifest.bf16,
                torch_version=manifest.torch_version,
                cuda_version=manifest.cuda_version,
                adapter_weight_sha256=str(weight_stats["sha256"] or manifest.adapter_weight_sha256 or ""),
                adapter_tensor_count=int(weight_stats["tensor_count"]),
                adapter_lora_tensor_count=int(weight_stats["lora_tensor_count"]),
                adapter_total_elements=int(weight_stats["total_elements"]),
                adapter_nonzero_elements=int(weight_stats["nonzero_elements"]),
                adapter_max_abs=weight_stats["max_abs"],
                adapter_weights_finite=bool(weight_stats["all_finite"]),
            )
        )
    return statuses


def count_training_steps(path: Path) -> int:
    if not path.exists():
        return 0
    steps = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            if json.loads(line).get("event") == "step":
                steps += 1
        except json.JSONDecodeError:
            return 0
    return steps


app = create_app()
