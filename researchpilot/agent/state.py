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
    owner: str | None  # UI session whose private uploads are searched too (retrieval/session.py)
    history: list[dict]  # earlier messages, {"role": "user" | "assistant", "content": str}, oldest first
    attachments: list[dict]  # files attached since the last question, {"source", "title"} (UI sessions only)
    # conversation context (set by the condense step; absent without history or attachments)
    standalone: str  # the question rewritten to stand on its own; used instead of `question` when set
    scope: list[str]  # sources retrieval is limited to (the attached files the question is about)
    # planning / retrieval loop
    sub_queries: list[str]
    new_queries: list[str]
    needs_web: bool
    evidence: list[Source]
    round: int
    sufficient: bool
    verdict: str | None  # the grader's "sufficient" | "retry" | "out_of_scope" (None: none or malformed)
    # output
    clarification: bool  # the answer lists the documents instead of answering (a question that names none)
    answer: str
    citations: list[dict]
    invalid_citations: list[int]
    insufficient: bool
    trace: Annotated[list[dict], operator.add]  # each node appends its step log
