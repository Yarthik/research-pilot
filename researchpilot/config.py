"""Central configuration.

Every field can be set with an environment variable of the same name in upper case
(for example LLM_PROVIDER or FINAL_K), or in a `.env` file in the working directory.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

RETRIEVAL_MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


class Settings(BaseModel):
    # --- LLM: change provider/model without touching code ---
    llm_provider: str = "groq"  # groq | openai | anthropic | google_genai | ollama
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.0
    llm_max_retries: int = Field(3, ge=1)  # retries on rate limits / transient errors

    # --- Local models (free, CPU, ONNX via FastEmbed) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # --- Storage ---
    data_dir: str = "storage"
    collection_name: str = "documents"

    # --- Chunking (characters) ---
    chunk_size: int = Field(1000, gt=100)
    chunk_overlap: int = Field(150, ge=0)

    # --- Retrieval ---
    retrieval_mode: str = "hybrid"
    candidate_k: int = Field(20, ge=1)  # candidates from each retriever before fusion / reranking
    final_k: int = Field(5, ge=1)  # chunks kept per query

    # --- Agent ---
    max_retrieval_rounds: int = Field(2, ge=1)
    max_evidence: int = Field(15, ge=1)  # chunks passed to the answer step

    # --- Optional web search ---
    tavily_api_key: str | None = None

    # --- Service / demo safety ---
    api_key: str | None = None  # if set, the API requires an X-API-Key header
    allow_ingest: bool = True  # turn off on a public demo to make the corpus read-only
    max_upload_mb: int = Field(20, ge=1)
    cors_origins: str = "*"  # comma-separated list for the API
    demo_max_questions: int | None = Field(None, ge=1)  # UI questions per browser session; unset = unlimited

    @field_validator("retrieval_mode")
    @classmethod
    def _check_mode(cls, value: str) -> str:
        if value not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval_mode must be one of {RETRIEVAL_MODES}")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def _check_overlap(cls, value: int, info) -> int:
        size = info.data.get("chunk_size")
        if size is not None and value >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value

    @field_validator("tavily_api_key", "api_key", mode="before")
    @classmethod
    def _empty_is_none(cls, value):
        return value or None

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables (and .env, loaded by get_settings)."""
        values = {}
        for name in cls.model_fields:
            raw = os.environ.get(name.upper())
            if raw is not None and raw.strip() != "":
                values[name] = raw.strip()
        return cls(**values)


@lru_cache
def get_settings() -> Settings:
    # Export .env values to os.environ so provider SDKs can also find their API keys.
    load_dotenv()
    return Settings.from_env()


def reset_settings() -> None:
    """Clear the cached settings (used by tests and after changing environment variables)."""
    get_settings.cache_clear()
