"""Structured outputs the LLM returns at each step."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StandaloneQuestion(BaseModel):
    question: str = Field(description="The latest message rewritten as one standalone question")
    about_attached_files: bool = Field(
        default=False,
        description="True only if the latest message is about the newly attached file(s)",
    )


class ResearchPlan(BaseModel):
    sub_queries: list[str] = Field(description="1 to 3 short, focused search queries that together answer the question")
    needs_web: bool = Field(
        default=False,
        description="True only if the question needs recent or external information unlikely to be in the documents",
    )


GRADE_VERDICTS = ("sufficient", "retry", "out_of_scope")


class ContextGrade(BaseModel):
    # A plain string rather than an enum, so an unexpected value is handled by the grade step (it falls
    # back to the old two-way logic) instead of failing validation and losing the rewritten queries.
    verdict: str = Field(
        default="",
        description=(
            '"sufficient" (the context answers the question), "retry" (the context is about the question\'s '
            'subject but a specific detail is missing) or "out_of_scope" (the context is about something else)'
        ),
    )
    missing: str = Field(default="", description="Briefly, what information is missing. Empty string if sufficient.")
    rewritten_queries: list[str] = Field(
        default_factory=list,
        description='Up to 2 new search queries to find the missing information. Empty unless the verdict is "retry".',
    )


class Answer(BaseModel):
    answer: str = Field(description="The answer in markdown, citing sources inline like [1] or [2][3]")
    cited_sources: list[int] = Field(
        default_factory=list, description="Numbers of the sources actually used in the answer"
    )
    insufficient_information: bool = Field(
        default=False, description="True if the sources did not contain enough information to answer"
    )
