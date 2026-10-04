"""Structured outputs the LLM returns at each step."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    sub_queries: list[str] = Field(description="1 to 3 short, focused search queries that together answer the question")
    needs_web: bool = Field(
        default=False,
        description="True only if the question needs recent or external information unlikely to be in the documents",
    )


class ContextGrade(BaseModel):
    sufficient: bool = Field(description="True if the context contains enough information to answer the question fully")
    missing: str = Field(default="", description="Briefly, what information is missing. Empty string if sufficient.")
    rewritten_queries: list[str] = Field(
        default_factory=list,
        description="Up to 2 new search queries to find the missing information. Empty list if sufficient.",
    )


class Answer(BaseModel):
    answer: str = Field(description="The answer in markdown, citing sources inline like [1] or [2][3]")
    cited_sources: list[int] = Field(
        default_factory=list, description="Numbers of the sources actually used in the answer"
    )
    insufficient_information: bool = Field(
        default=False, description="True if the sources did not contain enough information to answer"
    )
