from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ELY_EYE_", env_file=".env", extra="ignore")

    home: Path = Field(default_factory=lambda: Path.cwd() / ".ely_eye")
    model_id: str = "Qwen/Qwen3.5-9B"
    planner_model_id: str = "Qwen/Qwen3.5-0.8B"
    embedding_model_id: str = "Qwen/Qwen3-VL-Embedding-8B"
    adapter_dir: Path | None = None
    runtime_backend: Literal["sglang", "transformers"] = "transformers"
    sglang_base_url: str = "http://127.0.0.1:8000/v1"
    sglang_api_key: str = "ely-eye-local"
    dense_embeddings_enabled: bool = False
    live_context_tokens: int = 262_144
    extreme_context_tokens: int = 1_010_000
    library_target_tokens: int = 100_000_000
    upload_max_bytes: int = 20 * 1024 * 1024 * 1024
    video_sample_seconds: float = 5.0
    max_text_file_bytes: int = 10 * 1024 * 1024
    retrieval_top_k: int = 24
    max_new_tokens: int = 512
    temperature: float = 0.1
    train_max_length: int = 8192
    train_gradient_accumulation_steps: int = 8
    train_learning_rate: float = 2e-4

    @property
    def data_dir(self) -> Path:
        return self.home / "data"

    @property
    def object_dir(self) -> Path:
        return self.home / "objects"

    @property
    def cartridge_dir(self) -> Path:
        return self.home / "cartridges"

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    @property
    def training_dir(self) -> Path:
        return self.home / "training"

    @property
    def adapters_dir(self) -> Path:
        return self.home / "adapters"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "ely_eye.sqlite"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.object_dir,
            self.cartridge_dir,
            self.cache_dir,
            self.training_dir,
            self.adapters_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
