from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .db import Database
from .schemas import AdapterManifest, EvidenceAtom, TrainingAdapterKind


SYSTEM_PROMPT = (
    "You are Ely-Eye. Answer from supplied evidence, return compact JSON, "
    "and cite atom ids exactly."
)
TRAINING_PROOF_VERSION = 2
TRAINING_METHOD = "BF16 LoRA"
TRAINING_PRECISION = "bf16"
TRAINING_FRAMEWORK = "unsloth_patched_transformers_peft"
GRADIENT_CHECKPOINTING_MODE = "unsloth_patched_torch_non_reentrant"
AUTOCAST_DTYPE = "bf16"


class TrainingSample(BaseModel):
    id: str
    task: str
    inputs: list[dict[str, Any]]
    memory: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any]


class TrainingService:
    def __init__(self, settings: Settings | None = None, database: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = database or Database(self.settings)

    def validate_dataset(self, path: Path) -> list[TrainingSample]:
        samples: list[TrainingSample] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    samples.append(TrainingSample.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"Invalid training sample at line {line_number}: {exc}") from exc
        if not samples:
            raise ValueError(f"Training dataset is empty: {path}")
        return samples

    def build_prd_dataset(
        self,
        prd_path: Path,
        output_path: Path | None = None,
        cartridge_id: str | None = None,
        max_sections: int = 48,
    ) -> Path:
        prd_path = prd_path.resolve()
        text = prd_path.read_text(encoding="utf-8")
        atoms = self._atoms_for_path(prd_path)
        atom_id = atoms[0].atom_id if atoms else f"{prd_path.stem}:{sha256_bytes(text.encode('utf-8'))[:12]}"
        cartridge_id = cartridge_id or self._latest_cartridge_id()
        sections = split_markdown_sections(text)
        if not sections:
            raise ValueError(f"No markdown sections found in {prd_path}")
        output_path = output_path or (self.settings.training_dir / "hme_prd_sft.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        samples: list[TrainingSample] = []
        for index, section in enumerate(sections[:max_sections], start=1):
            title = section["title"]
            body = compact_text(section["body"], limit=5200)
            if len(body) < 160:
                continue
            citation = f"{atom_id}#section={slugify(title)}"
            answer = compact_text(body, limit=900)
            samples.append(
                TrainingSample(
                    id=f"hme_prd_{index:04d}",
                    task="hme_core_lora_evidence_json",
                    inputs=[
                        {
                            "type": "text",
                            "text": (
                                f"Section title: {title}\n"
                                f"Evidence atom: {atom_id}\n"
                                f"Evidence:\n{body}\n\n"
                                "Question: State the Ely-Eye requirement in this section. "
                                "Return JSON with answer, citations, contradiction_notes, confidence."
                            ),
                        }
                    ],
                    memory={
                        "cartridge": cartridge_id,
                        "required_atoms": [atom_id],
                        "source": str(prd_path),
                    },
                    output={
                        "answer": answer,
                        "citations": [citation],
                        "contradiction_notes": [],
                        "confidence": 1.0,
                        "reasoning_policy": "evidence-first",
                    },
                )
            )

        if not samples:
            raise ValueError(f"No usable PRD training sections found in {prd_path}")
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for sample in samples:
                handle.write(sample.model_dump_json() + "\n")
        return output_path

    def build_visual_dataset(
        self,
        prd_path: Path,
        output_path: Path | None = None,
        image_paths: list[Path] | None = None,
        cartridge_id: str | None = None,
    ) -> Path:
        prd_path = prd_path.resolve()
        text = prd_path.read_text(encoding="utf-8")
        atom_id = self._atom_id_for_path(prd_path, text)
        cartridge_id = cartridge_id or self._latest_cartridge_id()
        media_dir = self.settings.training_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)

        prepared_images: list[Path] = []
        for image_path in image_paths or []:
            if image_path.exists():
                prepared_images.append(prepare_training_image(image_path, media_dir))
        prd_card = render_prd_training_card(text, atom_id, media_dir / "prd_training_card.png")
        prepared_images.append(prd_card)
        if not prepared_images:
            raise ValueError("No visual training images are available")

        output_path = output_path or (self.settings.training_dir / "hme_visual_sft.jsonl")
        samples: list[TrainingSample] = []
        visual_tasks = [
            (
                "hme_vision_lora_visual_grounding",
                "12.1 训练目标",
                (
                    "Ely-Eye trains HME-Vision-LoRA on Qwen3.5-9B vision projector "
                    "and selected vision layers so UI, PDF, OCR, and chart evidence can be "
                    "grounded into cited multimodal memory."
                ),
            ),
            (
                "hme_ttt_vl_visual_memory",
                "11.2 X — TTT-VL：视觉语言测试时学习",
                (
                    "TTT-VL keeps Qwen3.5-9B frozen while mutable MLP and visual LoRA "
                    "adapters learn low-step visual memory updates that can be consolidated "
                    "into cartridge-local adapter deltas."
                ),
            ),
            (
                "hme_visual_hashhop_grounding",
                "11.3 X — Visual HashHop",
                (
                    "Visual HashHop converts random chain ids into non-text geometric image "
                    "blocks, then evaluates long-range visual addressing without OCR shortcuts."
                ),
            ),
            (
                "hme_dvkv_visual_cache",
                "11.4 S/H — DVKV：Diversity-Aware Visual KV Compression",
                (
                    "DVKV keeps visual KV tokens by combining attention importance, visual "
                    "diversity, and evidence value so OCR, layout, chart, and coordinate details "
                    "survive compression."
                ),
            ),
            (
                "hme_visual_contradiction_lens",
                "11.15 X — Visual Contradiction Lens",
                (
                    "The Visual Contradiction Lens compares visual versions and cites layout, "
                    "design-token, copy, temporal, and visual-code drift as evidence graph facts."
                ),
            ),
        ]
        for index, image_path in enumerate(prepared_images, start=1):
            for task_index, (task, section_title, answer) in enumerate(visual_tasks, start=1):
                requirement = section_by_title(text, section_title)
                samples.append(
                    TrainingSample(
                        id=f"hme_visual_{index:02d}_{task_index:02d}",
                        task=task,
                        inputs=[
                            {
                                "type": "ui_screenshot",
                                "path": str(image_path.resolve()),
                                "max_pixels": 262144,
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"Evidence atom: {atom_id}\n"
                                    f"Visual source: {image_path.name}\n"
                                    f"PRD visual requirement:\n"
                                    f"{compact_text(requirement, 2200)}\n\n"
                                    "Question: Map this visual evidence to the Ely-Eye PRD module. "
                                    "Return JSON with answer, citations, contradiction_notes, confidence."
                                ),
                            },
                        ],
                        memory={
                            "cartridge": cartridge_id,
                            "required_atoms": [atom_id],
                            "source": str(prd_path),
                        },
                        output={
                            "answer": answer,
                            "citations": [f"{atom_id}#section={slugify(section_title)}"],
                            "contradiction_notes": [],
                            "confidence": 1.0,
                            "reasoning_policy": "evidence-first",
                        },
                    )
                )
        write_jsonl(output_path, samples)
        return output_path

    def build_router_dataset(
        self,
        prd_path: Path,
        output_path: Path | None = None,
        cartridge_id: str | None = None,
    ) -> Path:
        prd_path = prd_path.resolve()
        text = prd_path.read_text(encoding="utf-8")
        atom_id = self._atom_id_for_path(prd_path, text)
        cartridge_id = cartridge_id or self._latest_cartridge_id()
        output_path = output_path or (self.settings.training_dir / "hme_router_sft.jsonl")
        scenarios = [
            (
                "Summarize a stable live-demo answer from local PRD evidence.",
                "live_demo",
                262144,
                ["mixed"],
                "Prefix cache + sparse evidence packing",
            ),
            (
                "Plan a 1.01M extreme context run with YaRN and parked context.",
                "extreme_context",
                1010000,
                ["mixed"],
                "YaRN + V-NSA sparse blocks + DVKV visual KV compression + CPU/NVMe parked context",
            ),
            (
                "Route a 100M library question through Context Cartridge memory.",
                "library_100m",
                100000000,
                ["mixed"],
                "Context Cartridge retrieval + ColQwen-compatible dense vectors + BM25 + evidence graph pack",
            ),
            (
                "Audit UI screenshot and PRD differences for the research demo.",
                "research_theater",
                262144,
                ["image", "ui_screenshot"],
                "Visual patch cache + DVKV evidence-biased region retention",
            ),
            (
                "Find visual token compression methods across a 100M library cartridge.",
                "library_100m",
                100000000,
                ["pdf_page", "image", "video_frame"],
                "ColQwen-compatible retrieval + graph grouping + cited evidence pack",
            ),
            (
                "Compare UI versions and cite coordinates plus code evidence.",
                "research_theater",
                262144,
                ["ui_screenshot", "code", "text"],
                "Visual Contradiction Lens + evidence graph writeback",
            ),
            (
                "Run Visual HashHop at 262K with a two-hop random image chain.",
                "research_theater",
                262144,
                ["image"],
                "Visual HashHop arena + verifier contract + no OCR-dependent evidence",
            ),
            (
                "Answer from a TTT-VL cartridge after original images are absent.",
                "library_100m",
                100000000,
                ["mixed"],
                "Load cartridge adapters + learned memory recall + cited adapter proof",
            ),
        ]
        samples = [
            TrainingSample(
                id=f"hme_router_{index:04d}",
                task="hme_router_context_plan",
                inputs=[
                    {
                        "type": "text",
                        "text": (
                            f"Evidence atom: {atom_id}\n"
                            f"PRD routing source:\n{compact_text(section_by_title(text, '1. 产品定义'), 1800)}\n\n"
                            f"User request: {request}\n"
                            "Return the Ely-Eye routing plan as compact JSON."
                        ),
                    }
                ],
                memory={
                    "cartridge": cartridge_id,
                    "required_atoms": [atom_id],
                    "source": str(prd_path),
                },
                output={
                    "profile": profile,
                    "token_budget": token_budget,
                    "required_modalities": modalities,
                    "retrieval_query": request,
                    "compression_strategy": compression,
                    "citations": [f"{atom_id}#section=1-产品定义"],
                    "confidence": 1.0,
                },
            )
            for index, (request, profile, token_budget, modalities, compression) in enumerate(scenarios, start=1)
        ]
        write_jsonl(output_path, samples)
        return output_path

    def build_retrieval_dataset(
        self,
        prd_path: Path,
        output_path: Path | None = None,
        cartridge_id: str | None = None,
        max_sections: int = 32,
    ) -> Path:
        prd_path = prd_path.resolve()
        text = prd_path.read_text(encoding="utf-8")
        atom_id = self._atom_id_for_path(prd_path, text)
        cartridge_id = cartridge_id or self._latest_cartridge_id()
        sections = [section for section in split_markdown_sections(text) if len(section["body"]) >= 160]
        output_path = output_path or (self.settings.training_dir / "hme_retrieval_pairs.jsonl")
        samples: list[TrainingSample] = []
        for index, section in enumerate(sections[:max_sections], start=1):
            title = section["title"]
            positive = compact_text(section["body"], 1600)
            samples.append(
                TrainingSample(
                    id=f"hme_retrieval_{index:04d}",
                    task="hme_retrieval_contrastive_pair",
                    inputs=[
                        {
                            "type": "text",
                            "text": f"Find the PRD section that answers: {title}",
                        }
                    ],
                    memory={
                        "cartridge": cartridge_id,
                        "required_atoms": [atom_id],
                        "source": str(prd_path),
                    },
                    output={
                        "positive": positive,
                        "positive_title": title,
                        "citations": [f"{atom_id}#section={slugify(title)}"],
                        "confidence": 1.0,
                    },
                )
            )
        if len(samples) < 2:
            raise ValueError("Retrieval contrastive training needs at least two PRD sections")
        write_jsonl(output_path, samples)
        return output_path

    def write_lora_config(
        self,
        output_dir: Path,
        kind: TrainingAdapterKind = TrainingAdapterKind.hme_core_lora,
        rank: int = 16,
        alpha: int = 32,
        model_id: str | None = None,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "base_model": model_id or self.settings.model_id,
            "adapter_kind": kind.value,
            "method": "BF16 LoRA",
            "framework": TRAINING_FRAMEWORK,
            "rank": rank,
            "alpha": alpha,
            "gradient_checkpointing": True,
            "gradient_checkpointing_mode": GRADIENT_CHECKPOINTING_MODE,
            "precision": "bf16",
            "optimizer": "paged_adamw_8bit",
            "ensure_weight_tying": kind == TrainingAdapterKind.hme_visual_mtp,
        }
        path = output_dir / "lora_config.json"
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def train_lora(
        self,
        dataset_path: Path,
        output_dir: Path,
        max_steps: int,
        kind: TrainingAdapterKind = TrainingAdapterKind.hme_core_lora,
        model_id: str | None = None,
        rank: int = 16,
        alpha: int = 32,
        max_length: int | None = None,
        gradient_accumulation_steps: int | None = None,
        learning_rate: float | None = None,
        cartridge_id: str | None = None,
    ) -> AdapterManifest:
        samples = self.validate_dataset(dataset_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_id = model_id or self._model_for_kind(kind)
        max_length = max_length or self.settings.train_max_length
        gradient_accumulation_steps = (
            gradient_accumulation_steps or self.settings.train_gradient_accumulation_steps
        )
        learning_rate = learning_rate or self.settings.train_learning_rate
        started_at = datetime.now(timezone.utc)
        wall_start = time.perf_counter()
        trace_path = output_dir / "training_trace.jsonl"

        import torch

        activate_unsloth_training()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForCausalLM

            model_cls = AutoModelForCausalLM

        require_cuda_bf16(torch)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        dtype = torch.bfloat16
        device_map: dict[str, int] = {"": 0}
        model = model_cls.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model.config.use_cache = False
        input_require_grads = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            input_require_grads = True

        target_modules = infer_lora_targets(model, kind)
        gradient_checkpointing = enable_gradient_checkpointing(model)
        peft_config_options: dict[str, Any] = {
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": target_modules,
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        if kind == TrainingAdapterKind.hme_visual_mtp:
            peft_config_options["ensure_weight_tying"] = True
        with bitsandbytes_gaudi_probe_guard():
            model = get_peft_model(model, LoraConfig(**peft_config_options))
        unsloth_model_type = model_type_for_proof(model)
        trainable_params, total_params = model.get_nb_trainable_parameters()
        optimizer, optimizer_name = make_optimizer(model, learning_rate)
        model.train()

        losses: list[float] = []
        optimizer_updates: list[dict[str, Any]] = []
        trainable_tensors = trainable_named_parameters(model)
        initial_trainable_sha = trainable_state_sha256(trainable_tensors)
        initial_trainable_l2 = trainable_l2_norm(trainable_tensors)
        optimizer.zero_grad(set_to_none=True)
        environment = training_environment(torch)
        contract = training_contract(
            gradient_checkpointing=gradient_checkpointing,
            input_require_grads=input_require_grads,
            optimizer_name=optimizer_name,
            unsloth_model_type=unsloth_model_type,
        )
        with trace_path.open("w", encoding="utf-8", newline="\n") as trace:
            write_trace(
                trace,
                {
                    "event": "run_start",
                    **contract,
                    "started_at": started_at.isoformat(),
                    "adapter_kind": kind.value,
                    "base_model": model_id,
                    "dataset_path": str(dataset_path.resolve()),
                    "dataset_sha256": sha256_file(dataset_path),
                    "sample_count": len(samples),
                    "max_steps": max_steps,
                    "max_length": max_length,
                    "rank": rank,
                    "alpha": alpha,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "learning_rate": learning_rate,
                    "target_modules": target_modules,
                    "trainable_params": int(trainable_params),
                    "total_params": int(total_params),
                    "optimizer": optimizer_name,
                    "environment": environment,
                },
            )
            for step in range(max_steps):
                sample = samples[step % len(samples)]
                batch = encode_training_sample(sample, processor, model.device, max_length)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=torch.cuda.is_available(),
                ):
                    outputs = model(**batch)
                    loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                optimizer_step = (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == max_steps
                grad_l2 = gradient_l2_norm(trainable_tensors)
                update_l2 = None
                if optimizer_step:
                    before_update = trainable_state_cpu(trainable_tensors)
                    optimizer.step()
                    update_l2 = trainable_update_l2(before_update, trainable_tensors)
                    optimizer_updates.append(
                        {
                            "step": step + 1,
                            "gradient_l2": grad_l2,
                            "update_l2": update_l2,
                        }
                    )
                    optimizer.zero_grad(set_to_none=True)
                actual_loss = float(loss.detach().cpu()) * gradient_accumulation_steps
                losses.append(actual_loss)
                write_trace(
                    trace,
                    {
                        "event": "step",
                        "step": step + 1,
                        "sample_id": sample.id,
                        "task": sample.task,
                        "loss": actual_loss,
                        "optimizer_step": optimizer_step,
                        "gradient_l2": grad_l2,
                        "update_l2": update_l2,
                    },
                )

        final_trainable_sha = trainable_state_sha256(trainable_tensors)
        final_trainable_l2 = trainable_l2_norm(trainable_tensors)
        max_update_l2 = max((item["update_l2"] for item in optimizer_updates), default=None)
        loss_delta = losses[0] - losses[-1] if len(losses) >= 2 else None
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        self.write_lora_config(output_dir, kind, rank, alpha, model_id)
        weight_stats = safetensor_weight_stats(output_dir / "adapter_model.safetensors")
        finished_at = datetime.now(timezone.utc)
        wall_seconds = time.perf_counter() - wall_start
        summary_path = output_dir / "training_summary.json"
        proof_path = output_dir / "training_proof.json"
        training_proof = {
            **contract,
            "adapter_kind": kind.value,
            "base_model": model_id,
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": sha256_file(dataset_path),
            "sample_count": len(samples),
            "max_steps": max_steps,
            "max_length": max_length,
            "rank": rank,
            "alpha": alpha,
            "target_modules": target_modules,
            "trainable_params": int(trainable_params),
            "total_params": int(total_params),
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "bf16": bool(torch.cuda.is_available()),
            "environment": environment,
            "initial_trainable_sha256": initial_trainable_sha,
            "final_trainable_sha256": final_trainable_sha,
            "initial_trainable_l2": initial_trainable_l2,
            "final_trainable_l2": final_trainable_l2,
            "optimizer_update_count": len(optimizer_updates),
            "max_optimizer_update_l2": max_update_l2,
            "loss_delta": loss_delta,
            "loss_history": losses,
            "optimizer_updates": optimizer_updates,
            "adapter_weight_sha256": weight_stats["sha256"],
            "adapter_weight_stats": weight_stats,
        }
        write_training_summary(proof_path, training_proof)
        write_training_summary(
            summary_path,
            {
                **contract,
                "adapter_kind": kind.value,
                "base_model": model_id,
                "dataset_path": str(dataset_path.resolve()),
                "dataset_sha256": sha256_file(dataset_path),
                "sample_count": len(samples),
                "max_steps": max_steps,
                "max_length": max_length,
                "rank": rank,
                "alpha": alpha,
                "target_modules": target_modules,
                "trainable_params": int(trainable_params),
                "total_params": int(total_params),
                "optimizer": optimizer_name,
                "learning_rate": learning_rate,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "bf16": bool(torch.cuda.is_available()),
                "loss_history": losses,
                "final_loss": losses[-1] if losses else None,
                "initial_trainable_sha256": initial_trainable_sha,
                "final_trainable_sha256": final_trainable_sha,
                "initial_trainable_l2": initial_trainable_l2,
                "final_trainable_l2": final_trainable_l2,
                "optimizer_update_count": len(optimizer_updates),
                "max_optimizer_update_l2": max_update_l2,
                "loss_delta": loss_delta,
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "wall_seconds": wall_seconds,
                "environment": environment,
                "peak_cuda_memory_mb": cuda_peak_memory_mb(torch),
                "adapter_weight_sha256": weight_stats["sha256"],
                "adapter_weight_stats": weight_stats,
            },
        )
        with trace_path.open("a", encoding="utf-8", newline="\n") as trace:
            write_trace(
                trace,
                {
                    "event": "run_finish",
                    **contract,
                    "finished_at": finished_at.isoformat(),
                    "wall_seconds": wall_seconds,
                    "final_loss": losses[-1] if losses else None,
                    "initial_trainable_sha256": initial_trainable_sha,
                    "final_trainable_sha256": final_trainable_sha,
                    "optimizer_update_count": len(optimizer_updates),
                    "max_optimizer_update_l2": max_update_l2,
                    "loss_delta": loss_delta,
                    "peak_cuda_memory_mb": cuda_peak_memory_mb(torch),
                    "adapter_weight_sha256": weight_stats["sha256"],
                },
            )
        adapter_sha = sha256_tree(output_dir)
        manifest = AdapterManifest(
            adapter_id=adapter_id(kind, dataset_path, model_id),
            kind=kind,
            created_at=datetime.now(timezone.utc),
            proof_version=TRAINING_PROOF_VERSION,
            training_method=TRAINING_METHOD,
            precision=TRAINING_PRECISION,
            framework=TRAINING_FRAMEWORK,
            base_model=model_id,
            dataset_path=str(dataset_path.resolve()),
            dataset_sha256=sha256_file(dataset_path),
            sample_count=len(samples),
            max_steps=max_steps,
            max_length=max_length,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            trainable_params=int(trainable_params),
            total_params=int(total_params),
            optimizer=optimizer_name,
            optimizer_family=optimizer_family(optimizer_name),
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_mode=GRADIENT_CHECKPOINTING_MODE if gradient_checkpointing else None,
            autocast_dtype=AUTOCAST_DTYPE,
            bf16=torch.cuda.is_available(),
            unsloth_version=package_version("unsloth"),
            triton_version=package_version("triton-windows") or package_version("triton"),
            xformers_version=package_version("xformers"),
            unsloth_model_type=unsloth_model_type,
            final_loss=losses[-1] if losses else None,
            loss_history=losses,
            initial_trainable_sha256=initial_trainable_sha,
            final_trainable_sha256=final_trainable_sha,
            initial_trainable_l2=initial_trainable_l2,
            final_trainable_l2=final_trainable_l2,
            optimizer_update_count=len(optimizer_updates),
            max_optimizer_update_l2=max_update_l2,
            loss_delta=loss_delta,
            training_proof_path=str(proof_path.resolve()),
            training_proof_sha256=sha256_file(proof_path),
            training_trace_path=str(trace_path.resolve()),
            training_trace_sha256=sha256_file(trace_path),
            training_summary_path=str(summary_path.resolve()),
            training_summary_sha256=sha256_file(summary_path),
            training_started_at=started_at,
            training_finished_at=finished_at,
            training_wall_seconds=wall_seconds,
            training_device=str(environment.get("device_name") or environment.get("device") or ""),
            torch_version=str(environment.get("torch_version") or ""),
            cuda_version=(
                str(environment["torch_cuda_version"])
                if environment.get("torch_cuda_version") is not None
                else None
            ),
            adapter_weight_sha256=str(weight_stats["sha256"] or ""),
            adapter_tensor_count=int(weight_stats["tensor_count"]),
            adapter_lora_tensor_count=int(weight_stats["lora_tensor_count"]),
            adapter_total_elements=int(weight_stats["total_elements"]),
            adapter_nonzero_elements=int(weight_stats["nonzero_elements"]),
            adapter_max_abs=weight_stats["max_abs"],
            adapter_weights_finite=bool(weight_stats["all_finite"]),
            artifacts=relative_artifacts(output_dir),
            sha256=adapter_sha,
            cartridge_id=cartridge_id,
        )
        (output_dir / "adapter_manifest.json").write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return manifest

    def train_retrieval_lora(
        self,
        dataset_path: Path,
        output_dir: Path,
        max_steps: int,
        rank: int = 8,
        alpha: int = 16,
        max_length: int | None = None,
        learning_rate: float | None = None,
        cartridge_id: str | None = None,
    ) -> AdapterManifest:
        samples = self.validate_dataset(dataset_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        max_length = max_length or self.settings.train_max_length
        learning_rate = learning_rate or self.settings.train_learning_rate
        started_at = datetime.now(timezone.utc)
        wall_start = time.perf_counter()
        trace_path = output_dir / "training_trace.jsonl"

        import torch
        import torch.nn.functional as F

        activate_unsloth_training()
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModel, AutoProcessor

        model_id = self.settings.embedding_model_id
        require_cuda_bf16(torch)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
        )
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        target_modules = infer_lora_targets(model, TrainingAdapterKind.hme_retrieval)
        gradient_checkpointing = enable_gradient_checkpointing(model)
        with bitsandbytes_gaudi_probe_guard():
            model = get_peft_model(
                model,
                LoraConfig(
                    r=rank,
                    lora_alpha=alpha,
                    target_modules=target_modules,
                    lora_dropout=0.0,
                    bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION,
                ),
            )
        unsloth_model_type = model_type_for_proof(model)
        trainable_params, total_params = model.get_nb_trainable_parameters()
        optimizer, optimizer_name = make_optimizer(model, learning_rate)
        model.train()

        losses: list[float] = []
        optimizer_updates: list[dict[str, Any]] = []
        trainable_tensors = trainable_named_parameters(model)
        initial_trainable_sha = trainable_state_sha256(trainable_tensors)
        initial_trainable_l2 = trainable_l2_norm(trainable_tensors)
        batch_size = min(4, len(samples))
        environment = training_environment(torch)
        contract = training_contract(
            gradient_checkpointing=gradient_checkpointing,
            input_require_grads=False,
            optimizer_name=optimizer_name,
            unsloth_model_type=unsloth_model_type,
        )
        with trace_path.open("w", encoding="utf-8", newline="\n") as trace:
            write_trace(
                trace,
                {
                    "event": "run_start",
                    **contract,
                    "started_at": started_at.isoformat(),
                    "adapter_kind": TrainingAdapterKind.hme_retrieval.value,
                    "base_model": model_id,
                    "dataset_path": str(dataset_path.resolve()),
                    "dataset_sha256": sha256_file(dataset_path),
                    "sample_count": len(samples),
                    "max_steps": max_steps,
                    "max_length": max_length,
                    "rank": rank,
                    "alpha": alpha,
                    "target_modules": target_modules,
                    "trainable_params": int(trainable_params),
                    "total_params": int(total_params),
                    "optimizer": optimizer_name,
                    "learning_rate": learning_rate,
                    "environment": environment,
                },
            )
            for step in range(max_steps):
                window = [samples[(step + offset) % len(samples)] for offset in range(batch_size)]
                queries = [str(sample.inputs[0]["text"]) for sample in window]
                documents = [str(sample.output["positive"]) for sample in window]
                query_batch = encode_retrieval_batch(queries, processor, model.device, max_length)
                document_batch = encode_retrieval_batch(documents, processor, model.device, max_length)
                query_embeddings = pool_embeddings(model(**query_batch), query_batch["attention_mask"])
                document_embeddings = pool_embeddings(model(**document_batch), document_batch["attention_mask"])
                logits = query_embeddings @ document_embeddings.T / 0.05
                labels = torch.arange(logits.shape[0], device=logits.device)
                loss = F.cross_entropy(logits.float(), labels)
                loss.backward()
                grad_l2 = gradient_l2_norm(trainable_tensors)
                before_update = trainable_state_cpu(trainable_tensors)
                optimizer.step()
                update_l2 = trainable_update_l2(before_update, trainable_tensors)
                optimizer_updates.append(
                    {
                        "step": step + 1,
                        "gradient_l2": grad_l2,
                        "update_l2": update_l2,
                    }
                )
                optimizer.zero_grad(set_to_none=True)
                actual_loss = float(loss.detach().cpu())
                losses.append(actual_loss)
                write_trace(
                    trace,
                    {
                        "event": "step",
                        "step": step + 1,
                        "sample_ids": [sample.id for sample in window],
                        "task": "hme_retrieval_contrastive_pair",
                        "loss": actual_loss,
                        "optimizer_step": True,
                        "gradient_l2": grad_l2,
                        "update_l2": update_l2,
                    },
                )

        final_trainable_sha = trainable_state_sha256(trainable_tensors)
        final_trainable_l2 = trainable_l2_norm(trainable_tensors)
        max_update_l2 = max((item["update_l2"] for item in optimizer_updates), default=None)
        loss_delta = losses[0] - losses[-1] if len(losses) >= 2 else None
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        self.write_lora_config(output_dir, TrainingAdapterKind.hme_retrieval, rank, alpha, model_id)
        weight_stats = safetensor_weight_stats(output_dir / "adapter_model.safetensors")
        finished_at = datetime.now(timezone.utc)
        wall_seconds = time.perf_counter() - wall_start
        summary_path = output_dir / "training_summary.json"
        proof_path = output_dir / "training_proof.json"
        training_proof = {
            **contract,
            "adapter_kind": TrainingAdapterKind.hme_retrieval.value,
            "base_model": model_id,
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": sha256_file(dataset_path),
            "sample_count": len(samples),
            "max_steps": max_steps,
            "max_length": max_length,
            "rank": rank,
            "alpha": alpha,
            "target_modules": target_modules,
            "trainable_params": int(trainable_params),
            "total_params": int(total_params),
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "gradient_accumulation_steps": 1,
            "bf16": bool(torch.cuda.is_available()),
            "environment": environment,
            "initial_trainable_sha256": initial_trainable_sha,
            "final_trainable_sha256": final_trainable_sha,
            "initial_trainable_l2": initial_trainable_l2,
            "final_trainable_l2": final_trainable_l2,
            "optimizer_update_count": len(optimizer_updates),
            "max_optimizer_update_l2": max_update_l2,
            "loss_delta": loss_delta,
            "loss_history": losses,
            "optimizer_updates": optimizer_updates,
            "adapter_weight_sha256": weight_stats["sha256"],
            "adapter_weight_stats": weight_stats,
        }
        write_training_summary(proof_path, training_proof)
        write_training_summary(
            summary_path,
            {
                **contract,
                "adapter_kind": TrainingAdapterKind.hme_retrieval.value,
                "base_model": model_id,
                "dataset_path": str(dataset_path.resolve()),
                "dataset_sha256": sha256_file(dataset_path),
                "sample_count": len(samples),
                "max_steps": max_steps,
                "max_length": max_length,
                "rank": rank,
                "alpha": alpha,
                "target_modules": target_modules,
                "trainable_params": int(trainable_params),
                "total_params": int(total_params),
                "optimizer": optimizer_name,
                "learning_rate": learning_rate,
                "gradient_accumulation_steps": 1,
                "bf16": bool(torch.cuda.is_available()),
                "loss_history": losses,
                "final_loss": losses[-1] if losses else None,
                "initial_trainable_sha256": initial_trainable_sha,
                "final_trainable_sha256": final_trainable_sha,
                "initial_trainable_l2": initial_trainable_l2,
                "final_trainable_l2": final_trainable_l2,
                "optimizer_update_count": len(optimizer_updates),
                "max_optimizer_update_l2": max_update_l2,
                "loss_delta": loss_delta,
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "wall_seconds": wall_seconds,
                "environment": environment,
                "peak_cuda_memory_mb": cuda_peak_memory_mb(torch),
                "adapter_weight_sha256": weight_stats["sha256"],
                "adapter_weight_stats": weight_stats,
            },
        )
        with trace_path.open("a", encoding="utf-8", newline="\n") as trace:
            write_trace(
                trace,
                {
                    "event": "run_finish",
                    **contract,
                    "finished_at": finished_at.isoformat(),
                    "wall_seconds": wall_seconds,
                    "final_loss": losses[-1] if losses else None,
                    "initial_trainable_sha256": initial_trainable_sha,
                    "final_trainable_sha256": final_trainable_sha,
                    "optimizer_update_count": len(optimizer_updates),
                    "max_optimizer_update_l2": max_update_l2,
                    "loss_delta": loss_delta,
                    "peak_cuda_memory_mb": cuda_peak_memory_mb(torch),
                    "adapter_weight_sha256": weight_stats["sha256"],
                },
            )
        manifest = AdapterManifest(
            adapter_id=adapter_id(TrainingAdapterKind.hme_retrieval, dataset_path, model_id),
            kind=TrainingAdapterKind.hme_retrieval,
            created_at=datetime.now(timezone.utc),
            proof_version=TRAINING_PROOF_VERSION,
            training_method=TRAINING_METHOD,
            precision=TRAINING_PRECISION,
            framework=TRAINING_FRAMEWORK,
            base_model=model_id,
            dataset_path=str(dataset_path.resolve()),
            dataset_sha256=sha256_file(dataset_path),
            sample_count=len(samples),
            max_steps=max_steps,
            max_length=max_length,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            trainable_params=int(trainable_params),
            total_params=int(total_params),
            optimizer=optimizer_name,
            optimizer_family=optimizer_family(optimizer_name),
            learning_rate=learning_rate,
            gradient_accumulation_steps=1,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_mode=GRADIENT_CHECKPOINTING_MODE if gradient_checkpointing else None,
            autocast_dtype=AUTOCAST_DTYPE,
            bf16=torch.cuda.is_available(),
            unsloth_version=package_version("unsloth"),
            triton_version=package_version("triton-windows") or package_version("triton"),
            xformers_version=package_version("xformers"),
            unsloth_model_type=unsloth_model_type,
            final_loss=losses[-1] if losses else None,
            loss_history=losses,
            initial_trainable_sha256=initial_trainable_sha,
            final_trainable_sha256=final_trainable_sha,
            initial_trainable_l2=initial_trainable_l2,
            final_trainable_l2=final_trainable_l2,
            optimizer_update_count=len(optimizer_updates),
            max_optimizer_update_l2=max_update_l2,
            loss_delta=loss_delta,
            training_proof_path=str(proof_path.resolve()),
            training_proof_sha256=sha256_file(proof_path),
            training_trace_path=str(trace_path.resolve()),
            training_trace_sha256=sha256_file(trace_path),
            training_summary_path=str(summary_path.resolve()),
            training_summary_sha256=sha256_file(summary_path),
            training_started_at=started_at,
            training_finished_at=finished_at,
            training_wall_seconds=wall_seconds,
            training_device=str(environment.get("device_name") or environment.get("device") or ""),
            torch_version=str(environment.get("torch_version") or ""),
            cuda_version=(
                str(environment["torch_cuda_version"])
                if environment.get("torch_cuda_version") is not None
                else None
            ),
            adapter_weight_sha256=str(weight_stats["sha256"] or ""),
            adapter_tensor_count=int(weight_stats["tensor_count"]),
            adapter_lora_tensor_count=int(weight_stats["lora_tensor_count"]),
            adapter_total_elements=int(weight_stats["total_elements"]),
            adapter_nonzero_elements=int(weight_stats["nonzero_elements"]),
            adapter_max_abs=weight_stats["max_abs"],
            adapter_weights_finite=bool(weight_stats["all_finite"]),
            artifacts=relative_artifacts(output_dir),
            sha256=sha256_tree(output_dir),
            cartridge_id=cartridge_id,
        )
        (output_dir / "adapter_manifest.json").write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return manifest

    def _atoms_for_path(self, path: Path) -> list[EvidenceAtom]:
        resolved = path.resolve()
        atoms: list[EvidenceAtom] = []
        for atom in self.db.list_atoms():
            try:
                if Path(atom.source).resolve() == resolved:
                    atoms.append(atom)
            except OSError:
                if atom.source == str(path):
                    atoms.append(atom)
        return atoms

    def _latest_cartridge_id(self) -> str | None:
        rows = self.db.list_cartridges()
        return str(rows[0]["cartridge_id"]) if rows else None

    def _model_for_kind(self, kind: TrainingAdapterKind) -> str:
        if kind == TrainingAdapterKind.hme_router:
            return self.settings.planner_model_id
        return self.settings.model_id

    def _atom_id_for_path(self, path: Path, text: str) -> str:
        atoms = self._atoms_for_path(path)
        return atoms[0].atom_id if atoms else f"{path.stem}:{sha256_bytes(text.encode('utf-8'))[:12]}"


def encode_training_sample(
    sample: TrainingSample,
    processor: Any,
    device: Any,
    max_length: int,
) -> dict[str, Any]:
    messages = messages_for_sample(sample, include_answer=True)
    prompt_messages = messages_for_sample(sample, include_answer=False)
    image_inputs, video_inputs = vision_inputs(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    labels = inputs["input_ids"].clone()
    prompt_length = min(int(prompt_inputs["input_ids"].shape[-1]), int(labels.shape[-1]))
    labels[:, :prompt_length] = -100
    pad_token_id = getattr(processor, "tokenizer", processor).pad_token_id
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    if int((labels != -100).sum().item()) == 0:
        raise ValueError(f"Sample {sample.id} produced no trainable tokens")
    inputs["labels"] = labels
    if int(inputs["input_ids"].shape[-1]) > max_length:
        for key in ("input_ids", "attention_mask", "labels"):
            if key in inputs:
                inputs[key] = inputs[key][:, -max_length:]
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def encode_retrieval_batch(
    texts: list[str],
    processor: Any,
    device: Any,
    max_length: int,
) -> dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", processor)
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def pool_embeddings(outputs: Any, attention_mask: Any) -> Any:
    import torch.nn.functional as F

    hidden = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return F.normalize(pooled, dim=-1)


def messages_for_sample(sample: TrainingSample, include_answer: bool) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = []
    for item in sample.inputs:
        item_type = item.get("type")
        if item_type in {"image", "pdf_page", "ui_screenshot"} and item.get("path"):
            path = str(item["path"])
            if path.startswith("object://"):
                user_content.append({"type": "text", "text": f"Image object reference: {path}"})
            else:
                image_part: dict[str, Any] = {"type": "image", "image": path}
                for key in ("min_pixels", "max_pixels"):
                    if key in item:
                        image_part[key] = item[key]
                user_content.append(image_part)
        elif item_type == "text" and item.get("text"):
            user_content.append({"type": "text", "text": str(item["text"])})
        else:
            user_content.append({"type": "text", "text": json.dumps(item, ensure_ascii=False)})
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if include_answer:
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(sample.output, ensure_ascii=False, separators=(",", ":")),
            }
        )
    return messages


def vision_inputs(messages: list[dict[str, Any]]) -> tuple[list[Any] | None, list[Any] | None]:
    has_media = any(
        isinstance(content, list)
        and any(part.get("type") in {"image", "video"} for part in content if isinstance(part, dict))
        for content in (message.get("content") for message in messages)
    )
    if not has_media:
        return None, None
    from qwen_vl_utils import process_vision_info

    return process_vision_info(messages)


def infer_lora_targets(model: Any, kind: TrainingAdapterKind) -> list[str]:
    import torch

    core_leafs = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    mlp_leafs = {"gate_proj", "up_proj", "down_proj"}
    found_leafs: set[str] = set()
    exact: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        lower = name.lower()
        if kind == TrainingAdapterKind.hme_core_lora and leaf in core_leafs:
            found_leafs.add(leaf)
        elif kind == TrainingAdapterKind.hme_vision_lora and ("visual" in lower or "vision" in lower):
            exact.add(name)
        elif kind == TrainingAdapterKind.hme_ttt_vl and (leaf in mlp_leafs or "visual" in lower):
            exact.add(name if "visual" in lower else leaf)
        elif kind == TrainingAdapterKind.hme_visual_mtp and (
            "mtp" in lower or "draft" in lower or name == "lm_head"
        ):
            exact.add(name)
        elif kind in {TrainingAdapterKind.hme_router, TrainingAdapterKind.hme_retrieval} and leaf in core_leafs:
            found_leafs.add(leaf)
    targets = sorted(exact or found_leafs)
    if not targets:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and not name.endswith("lm_head"):
                targets.append(name.rsplit(".", 1)[-1])
        targets = sorted(set(targets))
    if not targets:
        raise ValueError(f"No LoRA target modules found for {kind.value}")
    return targets


def trainable_named_parameters(model: Any) -> list[tuple[str, Any]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def trainable_state_cpu(named_parameters: list[tuple[str, Any]]) -> dict[str, Any]:
    return {name: param.detach().float().cpu().clone() for name, param in named_parameters}


def trainable_state_sha256(named_parameters: list[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, param in named_parameters:
        tensor = param.detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def trainable_l2_norm(named_parameters: list[tuple[str, Any]]) -> float:
    total = 0.0
    for _, param in named_parameters:
        tensor = param.detach().float()
        total += float(tensor.square().sum().detach().cpu())
    return math.sqrt(total)


def gradient_l2_norm(named_parameters: list[tuple[str, Any]]) -> float:
    total = 0.0
    for _, param in named_parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += float(grad.square().sum().detach().cpu())
    return math.sqrt(total)


def trainable_update_l2(before_update: dict[str, Any], named_parameters: list[tuple[str, Any]]) -> float:
    total = 0.0
    for name, param in named_parameters:
        delta = param.detach().float().cpu() - before_update[name]
        total += float(delta.square().sum())
    return math.sqrt(total)


def require_cuda_bf16(torch_module: Any) -> None:
    if not torch_module.cuda.is_available():
        raise RuntimeError("PRD 12.2 training requires CUDA BF16 on the local RTX 4090.")
    bf16_supported = getattr(torch_module.cuda, "is_bf16_supported", lambda: False)
    if not bool(bf16_supported()):
        raise RuntimeError("PRD 12.2 training requires BF16-capable CUDA hardware.")


def activate_unsloth_training() -> None:
    required = {
        "unsloth": package_version("unsloth"),
        "unsloth_zoo": package_version("unsloth_zoo"),
        "triton": package_version("triton-windows") or package_version("triton"),
        "xformers": package_version("xformers"),
    }
    missing = [name for name, installed in required.items() if not installed]
    if missing:
        raise RuntimeError(
            "PRD 12.2 training requires installed Unsloth dependencies: " + ", ".join(missing)
        )
    with bitsandbytes_gaudi_probe_guard():
        __import__("unsloth")


def enable_gradient_checkpointing(model: Any) -> bool:
    if not hasattr(model, "gradient_checkpointing_enable"):
        return False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    return True


def model_type_for_proof(model: Any) -> str | None:
    current = model
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        config = getattr(current, "config", None)
        model_type = getattr(config, "model_type", None)
        if isinstance(model_type, str) and model_type:
            return model_type
        current = getattr(current, "model", None)
    return None


def gradient_checkpointing_enabled(model: Any) -> bool:
    if bool(getattr(model, "is_gradient_checkpointing", False)):
        return True
    config = getattr(model, "config", None)
    if bool(getattr(config, "gradient_checkpointing", False)):
        return True
    return bool(getattr(model, "_gradient_checkpointing_func", None))


def make_optimizer(model: Any, learning_rate: float) -> tuple[Any, str]:
    params = [param for param in model.parameters() if param.requires_grad]
    try:
        with bitsandbytes_gaudi_probe_guard():
            import bitsandbytes as bnb
    except Exception as exc:
        raise RuntimeError("PRD 12.2 training requires bitsandbytes PagedAdamW8bit.") from exc
    try:
        return bnb.optim.PagedAdamW8bit(params, lr=learning_rate), "paged_adamw_8bit"
    except Exception as exc:
        raise RuntimeError("bitsandbytes PagedAdamW8bit failed to initialize.") from exc


def optimizer_family(optimizer_name: str) -> str:
    if optimizer_name == "paged_adamw_8bit":
        return "8-bit AdamW / paged optimizer"
    return optimizer_name


@contextmanager
def bitsandbytes_gaudi_probe_guard() -> Any:
    original_run = subprocess.run

    def guarded_run(*args: Any, **kwargs: Any) -> Any:
        command = args[0] if args else kwargs.get("args")
        if isinstance(command, str) and "habana-torch-plugin" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        return original_run(*args, **kwargs)

    subprocess.run = guarded_run
    try:
        yield
    finally:
        subprocess.run = original_run


def training_contract(
    *,
    gradient_checkpointing: bool,
    input_require_grads: bool,
    optimizer_name: str,
    unsloth_model_type: str | None,
) -> dict[str, Any]:
    return {
        "proof_version": TRAINING_PROOF_VERSION,
        "training_method": TRAINING_METHOD,
        "precision": TRAINING_PRECISION,
        "framework": TRAINING_FRAMEWORK,
        "optimizer_family": optimizer_family(optimizer_name),
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_mode": GRADIENT_CHECKPOINTING_MODE if gradient_checkpointing else None,
        "input_require_grads": input_require_grads,
        "autocast_dtype": AUTOCAST_DTYPE,
        "unsloth_version": package_version("unsloth"),
        "triton_version": package_version("triton-windows") or package_version("triton"),
        "xformers_version": package_version("xformers"),
        "unsloth_model_type": unsloth_model_type,
    }


def training_environment(torch_module: Any) -> dict[str, Any]:
    cuda_available = bool(torch_module.cuda.is_available())
    device_name = torch_module.cuda.get_device_name(0) if cuda_available else platform.processor()
    capability = torch_module.cuda.get_device_capability(0) if cuda_available else None
    bf16_supported = (
        bool(getattr(torch_module.cuda, "is_bf16_supported", lambda: False)())
        if cuda_available
        else False
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": "cuda:0" if cuda_available else "cpu",
        "device_name": device_name,
        "cuda_available": cuda_available,
        "bf16_supported": bf16_supported,
        "cuda_capability": list(capability) if capability else None,
        "torch_version": getattr(torch_module, "__version__", None),
        "torch_cuda_version": getattr(torch_module.version, "cuda", None),
        "transformers_version": package_version("transformers"),
        "peft_version": package_version("peft"),
        "bitsandbytes_version": package_version("bitsandbytes"),
        "unsloth_version": package_version("unsloth"),
        "unsloth_zoo_version": package_version("unsloth_zoo"),
        "triton_version": package_version("triton-windows") or package_version("triton"),
        "xformers_version": package_version("xformers"),
    }


def package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def cuda_peak_memory_mb(torch_module: Any) -> float | None:
    if not torch_module.cuda.is_available():
        return None
    return float(torch_module.cuda.max_memory_allocated() / 1024 / 1024)


def safetensor_weight_stats(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
        "tensor_count": 0,
        "lora_tensor_count": 0,
        "total_elements": 0,
        "nonzero_elements": 0,
        "nonzero_tensor_count": 0,
        "finite_tensor_count": 0,
        "all_finite": False,
        "max_abs": None,
        "mean_abs": None,
        "dtype_counts": {},
        "key_sample": [],
        "valid": False,
    }
    if not path.exists():
        return report

    import torch
    from safetensors import safe_open

    total_abs = 0.0
    max_abs = 0.0
    dtype_counts: dict[str, int] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        report["tensor_count"] = len(keys)
        report["key_sample"] = keys[:12]
        for key in keys:
            tensor = handle.get_tensor(key)
            numel = int(tensor.numel())
            dtype = str(tensor.dtype).replace("torch.", "")
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            report["total_elements"] += numel
            if "lora_" in key:
                report["lora_tensor_count"] += 1
            if numel == 0:
                continue
            tensor_float = tensor.detach().float()
            finite = bool(torch.isfinite(tensor_float).all().item())
            if finite:
                report["finite_tensor_count"] += 1
            nonzero = int(torch.count_nonzero(tensor).item())
            report["nonzero_elements"] += nonzero
            if nonzero > 0:
                report["nonzero_tensor_count"] += 1
            abs_tensor = tensor_float.abs()
            tensor_max_abs = float(abs_tensor.max().item())
            max_abs = max(max_abs, tensor_max_abs)
            total_abs += float(abs_tensor.sum().item())

    report["dtype_counts"] = dtype_counts
    report["all_finite"] = (
        report["tensor_count"] > 0 and report["finite_tensor_count"] == report["tensor_count"]
    )
    if int(report["total_elements"]) > 0:
        report["max_abs"] = max_abs
        report["mean_abs"] = total_abs / int(report["total_elements"])
    report["valid"] = (
        report["exists"]
        and int(report["tensor_count"]) > 0
        and int(report["lora_tensor_count"]) > 0
        and int(report["total_elements"]) > 0
        and int(report["nonzero_elements"]) > 0
        and bool(report["all_finite"])
        and report["max_abs"] is not None
        and float(report["max_abs"]) > 0.0
    )
    return report


def write_trace(handle: TextIO, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


def write_training_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def split_markdown_sections(text: str) -> list[dict[str, str]]:
    matches = list(re.finditer(r"^(#{2,3})\s+(.+?)\s*$", text, flags=re.MULTILINE))
    sections: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append({"title": match.group(2).strip(), "body": text[start:end].strip()})
    return sections


def section_by_title(text: str, title_fragment: str) -> str:
    for section in split_markdown_sections(text):
        if title_fragment in section["title"]:
            return section["body"]
    return ""


def write_jsonl(path: Path, samples: list[TrainingSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(sample.model_dump_json() + "\n")


def prepare_training_image(source: Path, media_dir: Path, max_side: int = 896) -> Path:
    from PIL import Image

    image = Image.open(source).convert("RGB")
    image.thumbnail((max_side, max_side))
    target = media_dir / f"{source.stem}_{sha256_file(source)[:10]}.png"
    image.save(target)
    return target


def render_prd_training_card(text: str, atom_id: str, target: Path) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1024, 768), "#f7f5ef")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    title_font = ImageFont.truetype(str(font_path), 34) if font_path.exists() else ImageFont.load_default()
    body_font = ImageFont.truetype(str(font_path), 21) if font_path.exists() else ImageFont.load_default()
    small_font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    draw.rectangle((28, 28, 996, 740), outline="#1f5f4a", width=3)
    draw.text((52, 48), "Ely-Eye PRD Training Card", fill="#09201b", font=title_font)
    draw.text((52, 100), f"Evidence Atom: {atom_id}", fill="#1f5f4a", font=small_font)
    source = (
        section_by_title(text, "12.1 训练目标")
        + "\n"
        + section_by_title(text, "12.2 训练策略")
        + "\n"
        + section_by_title(text, "12.3 训练数据结构")
    )
    y = 148
    for line in wrap_text(compact_text(source, 1500), width=42):
        draw.text((52, y), line, fill="#09201b", font=body_font)
        y += 31
        if y > 700:
            break
    image.save(target)
    return target


def wrap_text(text: str, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        current += char
        if len(current) >= width or char == "\n":
            lines.append(current.strip())
            current = ""
    if current.strip():
        lines.append(current.strip())
    return lines


def compact_text(text: str, limit: int) -> str:
    compacted = re.sub(r"\s+", " ", text).strip()
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 3].rstrip() + "..."


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", value).strip("-").lower()
    return slug[:80] or "section"


def adapter_id(kind: TrainingAdapterKind, dataset_path: Path, model_id: str) -> str:
    raw = f"{kind.value}:{model_id}:{dataset_path.resolve()}:{datetime.now(timezone.utc).isoformat()}"
    return f"{kind.value}_{sha256_bytes(raw.encode('utf-8'))[:12]}"


def relative_artifacts(root: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "adapter_manifest.json":
            artifacts[path.stem] = path.relative_to(root).as_posix()
    return artifacts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        shutil.copyfileobj(handle, _DigestWriter(digest))
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "adapter_manifest.json":
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, _DigestWriter(digest))
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _DigestWriter:
    def __init__(self, digest: "hashlib._Hash") -> None:
        self.digest = digest

    def write(self, data: bytes) -> int:
        self.digest.update(data)
        return len(data)
