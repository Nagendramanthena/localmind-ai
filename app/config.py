"""
Centralized configuration — loads from .env with sensible defaults.

All tunable parameters live here so every module imports from one place.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Ollama ────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:7b"

    # ── Embedding / ONNX ──────────────────────────────────────
    embedding_model_path: str = "./models/onnx"
    onnx_providers: str = "auto"  # "auto", "DmlExecutionProvider", "CPUExecutionProvider"

    # ── Retrieval ─────────────────────────────────────────────
    faiss_index_path: str = "./data/index"
    documents_path: str = "./data/documents"
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Agent ─────────────────────────────────────────────────
    confidence_threshold: float = 0.6
    max_retries: int = 2

    # ── Server ────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── Derived helpers ───────────────────────────────────────

    @property
    def onnx_provider_list(self) -> list[str]:
        """
        Return the ordered list of ONNX Runtime execution providers.

        On Windows, attempt DirectML first for GPU acceleration;
        everywhere else, fall back to CPU.
        """
        if self.onnx_providers != "auto":
            return [self.onnx_providers]

        providers: list[str] = []
        if platform.system() == "Windows":
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    @property
    def embedding_model_dir(self) -> Path:
        return Path(self.embedding_model_path).resolve()

    @property
    def faiss_index_dir(self) -> Path:
        return Path(self.faiss_index_path).resolve()

    @property
    def documents_dir(self) -> Path:
        return Path(self.documents_path).resolve()


# Module-level singleton — import this everywhere.
settings = Settings()
