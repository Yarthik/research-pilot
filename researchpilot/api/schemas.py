"""Request and response models for the API (they also drive the OpenAPI docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

Mode = Literal["vector", "bm25", "hybrid", "hybrid_rerank"]


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=2000, examples=["How does LoRA reduce the number of trainable parameters?"]
    )
    mode: Mode | None = Field(None, description="Retrieval mode; defaults to the server setting")
    max_rounds: int | None = Field(None, ge=1, le=5, description="Max retrieval rounds; defaults to the server setting")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class Citation(BaseModel):
    n: int
    title: str
    source: str
    page: int | None = None
    origin: str
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    invalid_citations: list[int] = []
    insufficient: bool
    sub_queries: list[str] | None = None
    round: int | None = None
    evidence_count: int
    seconds: float
    trace: list[dict]


class IngestUrlsRequest(BaseModel):
    urls: list[HttpUrl] = Field(..., min_length=1, max_length=20)


class IngestedItem(BaseModel):
    source: str
    title: str
    chunks: int
    pages: int


class FailedItem(BaseModel):
    source: str
    error: str


class IngestResponse(BaseModel):
    ingested: list[IngestedItem]
    failed: list[FailedItem]


class SourceInfo(BaseModel):
    source: str
    title: str
    chunks: int


class DeleteResponse(BaseModel):
    source: str
    deleted_chunks: int


class HealthResponse(BaseModel):
    status: str
    version: str
    chunks: int
    llm_provider: str
    llm_model: str
    retrieval_mode: str
    web_search: bool
    ingest_enabled: bool
