from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RuntimeProfile(StrEnum):
    live_demo = "live_demo"
    extreme_context = "extreme_context"
    library_100m = "library_100m"
    research_theater = "research_theater"


class TrainingAdapterKind(StrEnum):
    hme_core_lora = "hme_core_lora"
    hme_vision_lora = "hme_vision_lora"
    hme_ttt_vl = "hme_ttt_vl"
    hme_visual_mtp = "hme_visual_mtp"
    hme_router = "hme_router"
    hme_retrieval = "hme_retrieval"


class Modality(StrEnum):
    text = "text"
    code = "code"
    image = "image"
    pdf_page = "pdf_page"
    pdf_region = "pdf_region"
    video_frame = "video_frame"
    ui_screenshot = "ui_screenshot"
    mixed = "mixed"


class LayoutBox(BaseModel):
    x: float
    y: float
    w: float
    h: float
    page: int | None = None
    frame_second: float | None = None


class TrustScores(BaseModel):
    parser: float = Field(ge=0.0, le=1.0)
    ocr: float = Field(default=0.0, ge=0.0, le=1.0)
    model: float = Field(default=0.0, ge=0.0, le=1.0)


class EvidenceAtom(BaseModel):
    atom_id: str
    modality: Modality
    source_id: str
    source: str
    time: datetime = Field(default_factory=utc_now)
    text: str = ""
    image_ref: str | None = None
    embedding_refs: list[str] = Field(default_factory=list)
    layout: LayoutBox | None = None
    relations: list[str] = Field(default_factory=list)
    trust: TrustScores
    token_equivalent: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceRecord(BaseModel):
    source_id: str
    path: str
    kind: str
    mime: str
    sha256: str
    size_bytes: int
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    source_count: int
    atom_count: int
    cartridge_id: str | None
    token_equivalent: int
    sources: list[SourceRecord]


class RetrievalHit(BaseModel):
    atom: EvidenceAtom
    sparse_score: float
    dense_score: float = 0.0
    graph_score: float = 0.0
    final_score: float


class ContextPlan(BaseModel):
    question: str
    profile: RuntimeProfile
    required_modalities: list[Modality]
    evidence_budget_atoms: int
    token_budget: int
    retrieval_query: str
    compression_strategy: str
    verifier_contract: list[str]


class CompiledContext(BaseModel):
    plan: ContextPlan
    hits: list[RetrievalHit]
    packed_text: str
    token_equivalent: int
    cache_trace_id: str


class CitationReport(BaseModel):
    cited_atom_ids: list[str]
    missing_atom_ids: list[str]
    citation_accuracy: float
    contradiction_notes: list[str]
    confidence: float


class AnswerResponse(BaseModel):
    answer: str
    citations: list[str]
    context: CompiledContext
    verifier: CitationReport
    runtime_backend: str


class CacheLayerStatus(BaseModel):
    layer: str
    entries: int
    bytes_estimate: int
    hit_count: int
    miss_count: int
    last_event_at: datetime | None


class MemoryStatus(BaseModel):
    active_atoms: int
    active_token_equivalent: int
    cartridges: int
    library_token_equivalent: int
    layers: list[CacheLayerStatus]


class RuntimeStatus(BaseModel):
    backend: str
    model_id: str
    available: bool
    detail: str
    adapter_path: str | None = None
    adapter_kind: TrainingAdapterKind | None = None
    adapter_id: str | None = None
    adapter_sha256: str | None = None
    gpu_name: str | None = None
    cuda: str | None = None
    vram_total_mb: int | None = None
    vram_used_mb: int | None = None


class SystemStatus(BaseModel):
    app: str = "Ely-Eye"
    version: str
    runtime: RuntimeStatus
    memory: MemoryStatus
    profiles: dict[str, dict[str, Any]]
    data_home: str


class CartridgeManifest(BaseModel):
    cartridge_id: str
    name: str
    created_at: datetime
    model_base: str
    source_count: int
    atom_count: int
    token_equivalent: int
    artifacts: dict[str, str]
    dna: str | None = None


class AdapterManifest(BaseModel):
    adapter_id: str
    kind: TrainingAdapterKind
    created_at: datetime
    proof_version: int = 1
    training_method: str = "BF16 LoRA"
    precision: str = "bf16"
    framework: str = "unsloth_patched_transformers_peft"
    base_model: str
    dataset_path: str
    dataset_sha256: str
    sample_count: int
    max_steps: int
    max_length: int
    rank: int
    alpha: int
    target_modules: list[str]
    trainable_params: int
    total_params: int
    optimizer: str
    optimizer_family: str | None = None
    learning_rate: float
    gradient_accumulation_steps: int
    gradient_checkpointing: bool = False
    gradient_checkpointing_mode: str | None = None
    autocast_dtype: str | None = None
    unsloth_version: str | None = None
    triton_version: str | None = None
    xformers_version: str | None = None
    unsloth_model_type: str | None = None
    bf16: bool
    final_loss: float | None
    loss_history: list[float] = Field(default_factory=list)
    initial_trainable_sha256: str | None = None
    final_trainable_sha256: str | None = None
    initial_trainable_l2: float | None = None
    final_trainable_l2: float | None = None
    optimizer_update_count: int = 0
    max_optimizer_update_l2: float | None = None
    loss_delta: float | None = None
    training_proof_path: str | None = None
    training_proof_sha256: str | None = None
    training_trace_path: str | None = None
    training_trace_sha256: str | None = None
    training_summary_path: str | None = None
    training_summary_sha256: str | None = None
    training_started_at: datetime | None = None
    training_finished_at: datetime | None = None
    training_wall_seconds: float | None = None
    training_device: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    adapter_weight_sha256: str | None = None
    adapter_tensor_count: int = 0
    adapter_lora_tensor_count: int = 0
    adapter_total_elements: int = 0
    adapter_nonzero_elements: int = 0
    adapter_max_abs: float | None = None
    adapter_weights_finite: bool = False
    artifacts: dict[str, str]
    sha256: str
    cartridge_id: str | None = None


class AdapterStatus(BaseModel):
    name: str
    kind: TrainingAdapterKind
    proof_version: int = 1
    training_method: str = "BF16 LoRA"
    precision: str = "bf16"
    framework: str = "unsloth_patched_transformers_peft"
    base_model: str
    path: str
    adapter_id: str
    sample_count: int
    max_steps: int
    trainable_params: int
    final_loss: float | None
    sha256: str
    weight_bytes: int
    safetensor_keys: int
    cartridge_id: str | None = None
    cartridge_bound: bool
    training_trace_path: str | None = None
    training_trace_sha256: str | None = None
    training_trace_steps: int = 0
    training_proof_path: str | None = None
    training_proof_sha256: str | None = None
    optimizer: str | None = None
    optimizer_family: str | None = None
    gradient_checkpointing: bool = False
    gradient_checkpointing_mode: str | None = None
    autocast_dtype: str | None = None
    unsloth_version: str | None = None
    triton_version: str | None = None
    xformers_version: str | None = None
    unsloth_model_type: str | None = None
    optimizer_update_count: int = 0
    max_optimizer_update_l2: float | None = None
    loss_delta: float | None = None
    training_summary_path: str | None = None
    training_wall_seconds: float | None = None
    training_device: str | None = None
    bf16: bool = False
    torch_version: str | None = None
    cuda_version: str | None = None
    adapter_weight_sha256: str | None = None
    adapter_tensor_count: int = 0
    adapter_lora_tensor_count: int = 0
    adapter_total_elements: int = 0
    adapter_nonzero_elements: int = 0
    adapter_max_abs: float | None = None
    adapter_weights_finite: bool = False


class CartridgeAssetStatus(BaseModel):
    name: str
    path: str
    sha256: str
    size_bytes: int
    item_count: int
    dtype: str
    status: Literal["ready", "missing"]


class CartridgeAssetReport(BaseModel):
    cartridge_id: str
    status: Literal["ready", "incomplete"]
    created_at: datetime = Field(default_factory=utc_now)
    assets: list[CartridgeAssetStatus]


class BenchmarkSpec(BaseModel):
    name: str
    target: str
    dataset: str
    task_type: str
    runner: str
    metric: str


class BenchmarkRegistryReport(BaseModel):
    status: Literal["ready", "incomplete"]
    created_at: datetime = Field(default_factory=utc_now)
    benchmarks: list[BenchmarkSpec]


class BenchmarkSourceStatus(BaseModel):
    name: str
    dataset: str
    status: Literal["ready", "unavailable"]
    sha: str | None = None
    gated: bool | str | None = None
    private: bool | None = None
    sibling_count: int = 0
    error: str | None = None


class BenchmarkSourceReport(BaseModel):
    status: Literal["ready", "incomplete"]
    created_at: datetime = Field(default_factory=utc_now)
    sources: list[BenchmarkSourceStatus]


class BenchmarkSampleStatus(BaseModel):
    name: str
    dataset: str
    source_dataset: str | None = None
    source_sha: str | None = None
    source_note: str | None = None
    status: Literal["loaded", "gated", "failed"]
    config: str | None = None
    split: str | None = None
    sample_id: str | None = None
    fields: list[str] = Field(default_factory=list)
    question_preview: str | None = None
    answer_preview: str | None = None
    sample_sha256: str | None = None
    media_artifacts: list[str] = Field(default_factory=list)
    error: str | None = None


class BenchmarkSampleReport(BaseModel):
    status: Literal["ready", "partial", "failed"]
    created_at: datetime = Field(default_factory=utc_now)
    samples: list[BenchmarkSampleStatus]


class BenchmarkPredictionStatus(BaseModel):
    name: str
    dataset: str
    status: Literal["predicted", "gated", "failed"]
    config: str | None = None
    split: str | None = None
    sample_id: str | None = None
    task_type: str
    metric: str
    prompt_sha256: str | None = None
    sample_sha256: str | None = None
    prediction: str | None = None
    answer: str | None = None
    score: float | None = None
    runtime_backend: str | None = None
    adapter_id: str | None = None
    adapter_sha256: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    media_artifacts: list[str] = Field(default_factory=list)
    error: str | None = None


class BenchmarkPredictionReport(BaseModel):
    status: Literal["ready", "partial", "failed"]
    created_at: datetime = Field(default_factory=utc_now)
    predictions: list[BenchmarkPredictionStatus]


class RuntimeLaunchProfile(BaseModel):
    name: str
    runtime: str
    role: str
    context_tokens: int
    script: str
    command: list[str]
    environment: dict[str, str] = Field(default_factory=dict)
    required_features: list[str] = Field(default_factory=list)


class RuntimeProfileValidation(BaseModel):
    name: str
    runtime: str
    status: Literal["ready", "missing"]
    script_exists: bool
    package: str | None = None
    package_version: str | None = None
    executable: str | None = None
    wsl_gpu: bool = False
    probe_command: list[str] = Field(default_factory=list)
    probe_status: Literal["passed", "failed", "skipped"] = "skipped"
    probe_exit_code: int | None = None
    probe_output: str | None = None
    detail: str


class RuntimeProfileReport(BaseModel):
    status: Literal["ready", "incomplete"]
    created_at: datetime = Field(default_factory=utc_now)
    profiles: list[RuntimeLaunchProfile]
    validations: list[RuntimeProfileValidation] = Field(default_factory=list)


class HashHopProof(BaseModel):
    proof_id: str
    kind: Literal["text_hashhop", "visual_hashhop"]
    hops: int
    token_equivalent: int
    query_id: str
    expected_target_id: str
    model_target_id: str | None = None
    passed: bool | None = None
    artifacts: list[str]
    created_at: datetime = Field(default_factory=utc_now)


class ProofCheck(BaseModel):
    name: str
    requirement: str
    status: Literal["passed", "failed"]
    evidence: list[str]
    detail: str


class ProofSuiteReport(BaseModel):
    proof_id: str
    cartridge_id: str
    status: Literal["passed", "failed"]
    created_at: datetime = Field(default_factory=utc_now)
    checks: list[ProofCheck]
    artifacts: dict[str, str]
    dna_before: str | None = None
    dna_after: str | None = None


class UploadIngestRequest(BaseModel):
    cartridge_name: str | None = None
    dense_embeddings: bool | None = None


class PathIngestRequest(BaseModel):
    path: Path
    cartridge_name: str | None = None
    dense_embeddings: bool | None = None
