from __future__ import annotations

import base64
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import Settings, get_settings
from .schemas import AdapterManifest, CompiledContext, RuntimeStatus, TrainingAdapterKind
from .storage import ObjectStore


class RuntimeUnavailableError(RuntimeError):
    pass


@dataclass
class RuntimeGeneration:
    text: str
    backend: str
    adapter_path: str | None = None
    adapter_kind: str | None = None
    adapter_id: str | None = None
    adapter_sha256: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class QwenRuntime:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.objects = ObjectStore(self.settings)
        self._processor: Any | None = None
        self._model: Any | None = None

    def status(self) -> RuntimeStatus:
        gpu = gpu_status()
        adapter = resolve_runtime_adapter(self.settings)
        if self.settings.runtime_backend == "sglang":
            available, detail = self._sglang_status()
        else:
            available, detail = self._transformers_status()
        return RuntimeStatus(
            backend=self.settings.runtime_backend,
            model_id=self.settings.model_id,
            available=available,
            detail=detail,
            adapter_path=str(adapter.path) if adapter else None,
            adapter_kind=adapter.manifest.kind if adapter else None,
            adapter_id=adapter.manifest.adapter_id if adapter else None,
            adapter_sha256=adapter.manifest.sha256 if adapter else None,
            gpu_name=gpu.get("name"),
            cuda=gpu.get("cuda"),
            vram_total_mb=gpu.get("total_mb"),
            vram_used_mb=gpu.get("used_mb"),
        )

    def generate(self, question: str, context: CompiledContext) -> RuntimeGeneration:
        if self.settings.runtime_backend == "sglang":
            return self._generate_sglang(question, context)
        return self._generate_transformers(question, context)

    def generate_prompt(self, prompt: str, image_paths: list[Path] | None = None) -> RuntimeGeneration:
        if self.settings.runtime_backend == "sglang":
            return self._generate_sglang_prompt(prompt, image_paths or [])
        return self._generate_transformers_prompt(prompt, image_paths or [])

    def _sglang_status(self) -> tuple[bool, str]:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{self.settings.sglang_base_url}/models")
            if response.status_code == 200:
                return True, "SGLang OpenAI-compatible endpoint is reachable."
            return False, f"SGLang endpoint returned HTTP {response.status_code}."
        except httpx.HTTPError as exc:
            return False, f"SGLang endpoint is not reachable: {exc}"

    def _transformers_status(self) -> tuple[bool, str]:
        try:
            import torch

            adapter_detail = ""
            adapter = resolve_runtime_adapter(self.settings)
            if adapter is not None:
                adapter_detail = f" Adapter selected: {adapter.path}."
            elif self.settings.adapter_dir is not None:
                return False, f"Configured adapter path is missing: {self.settings.adapter_dir}"
            if torch.cuda.is_available():
                return True, f"Transformers backend can use CUDA.{adapter_detail}"
            return True, f"Transformers backend is available on CPU.{adapter_detail}"
        except Exception as exc:
            return False, f"Transformers backend import failed: {exc}"

    def _generate_sglang(self, question: str, context: CompiledContext) -> RuntimeGeneration:
        prompt = build_prompt(question, context)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in self._context_images(context)[:6]:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(image_path)},
                }
            )
        payload = {
            "model": self.settings.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_new_tokens,
        }
        try:
            with httpx.Client(timeout=None) as client:
                response = client.post(
                    f"{self.settings.sglang_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.settings.sglang_api_key}"},
                    json=payload,
                )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeUnavailableError(f"SGLang generation failed: {exc}") from exc
        data = response.json()
        adapter = resolve_runtime_adapter(self.settings)
        usage = data.get("usage") if isinstance(data, dict) else None
        return RuntimeGeneration(
            text=data["choices"][0]["message"]["content"],
            backend="sglang",
            adapter_path=str(adapter.path) if adapter else None,
            adapter_kind=adapter.manifest.kind.value if adapter else None,
            adapter_id=adapter.manifest.adapter_id if adapter else None,
            adapter_sha256=adapter.manifest.sha256 if adapter else None,
            input_tokens=usage.get("prompt_tokens") if isinstance(usage, dict) else None,
            output_tokens=usage.get("completion_tokens") if isinstance(usage, dict) else None,
        )

    def _generate_sglang_prompt(self, prompt: str, image_paths: list[Path]) -> RuntimeGeneration:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths[:6]:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(image_path)},
                }
            )
        payload = {
            "model": self.settings.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_new_tokens,
        }
        try:
            with httpx.Client(timeout=None) as client:
                response = client.post(
                    f"{self.settings.sglang_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.settings.sglang_api_key}"},
                    json=payload,
                )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeUnavailableError(f"SGLang generation failed: {exc}") from exc
        data = response.json()
        adapter = resolve_runtime_adapter(self.settings)
        usage = data.get("usage") if isinstance(data, dict) else None
        return RuntimeGeneration(
            text=data["choices"][0]["message"]["content"],
            backend="sglang",
            adapter_path=str(adapter.path) if adapter else None,
            adapter_kind=adapter.manifest.kind.value if adapter else None,
            adapter_id=adapter.manifest.adapter_id if adapter else None,
            adapter_sha256=adapter.manifest.sha256 if adapter else None,
            input_tokens=usage.get("prompt_tokens") if isinstance(usage, dict) else None,
            output_tokens=usage.get("completion_tokens") if isinstance(usage, dict) else None,
        )

    def _generate_transformers(self, question: str, context: CompiledContext) -> RuntimeGeneration:
        model, processor = self._load_transformers()
        prompt = build_prompt(question, context)
        images = [str(path) for path in self._context_images(context)[:6]]
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                + [{"type": "image", "image": image_path} for image_path in images],
            }
        ]
        import torch

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image_inputs: list[Any] | None = None
        video_inputs: list[Any] | None = None
        if images:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.settings.max_new_tokens,
            "do_sample": self.settings.temperature > 0,
        }
        if self.settings.temperature > 0:
            generate_kwargs["temperature"] = self.settings.temperature
        with torch.inference_mode():
            generated = model.generate(**generate_kwargs)
        input_length = int(inputs["input_ids"].shape[-1])
        generated = generated[:, input_length:]
        decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
        adapter = resolve_runtime_adapter(self.settings)
        return RuntimeGeneration(
            text=decoded,
            backend="transformers",
            adapter_path=str(adapter.path) if adapter else None,
            adapter_kind=adapter.manifest.kind.value if adapter else None,
            adapter_id=adapter.manifest.adapter_id if adapter else None,
            adapter_sha256=adapter.manifest.sha256 if adapter else None,
            input_tokens=input_length,
            output_tokens=int(generated.shape[-1]),
        )

    def _generate_transformers_prompt(self, prompt: str, image_paths: list[Path]) -> RuntimeGeneration:
        model, processor = self._load_transformers()
        images = [str(path) for path in image_paths[:6]]
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                + [{"type": "image", "image": image_path} for image_path in images],
            }
        ]
        import torch

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image_inputs: list[Any] | None = None
        video_inputs: list[Any] | None = None
        if images:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.settings.max_new_tokens,
            "do_sample": self.settings.temperature > 0,
        }
        if self.settings.temperature > 0:
            generate_kwargs["temperature"] = self.settings.temperature
        with torch.inference_mode():
            generated = model.generate(**generate_kwargs)
        input_length = int(inputs["input_ids"].shape[-1])
        generated = generated[:, input_length:]
        decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
        adapter = resolve_runtime_adapter(self.settings)
        return RuntimeGeneration(
            text=decoded,
            backend="transformers",
            adapter_path=str(adapter.path) if adapter else None,
            adapter_kind=adapter.manifest.kind.value if adapter else None,
            adapter_id=adapter.manifest.adapter_id if adapter else None,
            adapter_sha256=adapter.manifest.sha256 if adapter else None,
            input_tokens=input_length,
            output_tokens=int(generated.shape[-1]),
        )

    def _load_transformers(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForCausalLM

            model_cls = AutoModelForCausalLM

        self._processor = AutoProcessor.from_pretrained(self.settings.model_id, trust_remote_code=True)
        self._model = model_cls.from_pretrained(
            self.settings.model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        adapter = resolve_runtime_adapter(self.settings)
        if adapter is not None:
            from .training import bitsandbytes_gaudi_probe_guard

            with bitsandbytes_gaudi_probe_guard():
                from peft import PeftModel

                self._model = PeftModel.from_pretrained(self._model, adapter.path)
        self._model.eval()
        return self._model, self._processor

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    def _context_images(self, context: CompiledContext) -> list[Path]:
        paths: list[Path] = []
        for hit in context.hits:
            image_ref = hit.atom.image_ref
            if image_ref:
                paths.append(self.objects.resolve(image_ref))
        return paths


@dataclass(frozen=True)
class RuntimeAdapter:
    path: Path
    manifest: AdapterManifest


def resolve_runtime_adapter(
    settings: Settings,
    kind: TrainingAdapterKind = TrainingAdapterKind.hme_core_lora,
) -> RuntimeAdapter | None:
    if settings.adapter_dir is not None:
        path = settings.adapter_dir.expanduser().resolve()
        manifest_path = path / "adapter_manifest.json"
        if path.exists() and manifest_path.exists():
            return RuntimeAdapter(
                path=path,
                manifest=AdapterManifest.model_validate_json(manifest_path.read_text(encoding="utf-8")),
            )
        return None

    candidates: list[RuntimeAdapter] = []
    for manifest_path in settings.adapters_dir.glob("*/adapter_manifest.json"):
        manifest = AdapterManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if manifest.kind == kind and manifest.base_model == settings.model_id:
            candidates.append(RuntimeAdapter(path=manifest_path.parent.resolve(), manifest=manifest))
    candidates.sort(
        key=lambda item: (
            item.manifest.cartridge_id is not None,
            (item.path / "adapter_model.safetensors").stat().st_mtime
            if (item.path / "adapter_model.safetensors").exists()
            else 0.0,
        ),
        reverse=True,
    )
    return candidates[0] if candidates else None


def build_prompt(question: str, context: CompiledContext) -> str:
    contract = "\n".join(f"- {item}" for item in context.plan.verifier_contract)
    return f"""You are Ely-Eye, a local multimodal context operating system.

Use /no_think mode. Return final JSON only. Use only the provided evidence. Include atom ids in citations.

Verifier contract:
{contract}

Context plan:
{context.plan.model_dump_json(indent=2)}

Packed evidence:
{context.packed_text}

Question:
{question}

Return compact JSON only:
{{
  "answer": "120 words or fewer",
  "citations": ["atom_id strings only"],
  "contradiction_notes": ["strings"],
  "confidence": 0.0
}}
"""


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def gpu_status() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return {}
    if not output:
        return {}
    first = output.splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 4:
        return {}
    return {
        "name": parts[0],
        "total_mb": int(float(parts[1])),
        "used_mb": int(float(parts[2])),
        "cuda": f"driver {parts[3]}",
    }


def parse_runtime_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise
