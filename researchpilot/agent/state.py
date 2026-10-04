"""Shared state passed between LangGraph nodes."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Source(TypedDict):
    id: str
    text: str
    title: str
    source: str
    page: int | None
    origin: str  # "document" or "web"


class ResearchState(TypedDict, total=False):
    # input
    question: str
    mode: str | None  # retrieval mode override
    max_rounds: int | None  # override for max_retrieval_rounds (used by the eval ablation)
    # planning / retrieval loop
    sub_queries: list[str]
    new_queries: list[str]
    needs_web: bool
    evidence: list[Source]
    round: int
    sufficient: bool
    # output
    answer: str
    citations: list[dict]
    invalid_citations: list[int]
    insufficient: bool
    trace: Annotated[list[dict], operator.add]  # each node appends its step log
