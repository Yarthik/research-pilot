"""Request and response models for the API (they also drive the OpenAPI docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

Mode = Literal["vector", "bm25", "hybrid", "hybrid_rerank"]


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=2000, examples=["How does LoRA reduce the number of trainable parameters?"]
    )
    mode: Mode | None = Field(None, description="Retrieval mode; defaults to the server setting")
    max_rounds: int | None = Field(None, ge=1, le=5, description="Max retrieval rounds; defaults to the server setting")
    history: list[Turn] | None = Field(
        None,
        max_length=20,
        description=(
            "Earlier messages of the conversation, oldest first, so follow-ups like 'What is it about?' can be "
            "understood. Only the last 6 are used, each shortened to 600 characters. Omit for a single question."
        ),
    )

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
    standalone_question: str | None = Field(None, description="The follow-up rewritten from the history, if it was")
    clarification: bool = Field(False, description="True if the answer asks which document is meant")


class IngestUrlsRequest(BaseModel):
    urls: list[HttpUrl] = Field(..., min_length=1, max_length=20)


class IngestedItem(BaseModel):
    source: str
    title: str
    chunks: int
    pages: int
    english: bool = Field(True, description="False if the text isn't English: answers about it may be less reliable")


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
