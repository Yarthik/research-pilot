"""LLM-as-judge for answer quality: correctness against a reference and faithfulness to sources."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from researchpilot import llm as llm_module
from researchpilot.config import get_settings


class CorrectnessVerdict(BaseModel):
    verdict: Literal["correct", "partially_correct", "incorrect"] = Field(
        description="correct: covers the key facts of the reference; partially_correct: some key facts missing or "
        "imprecise; incorrect: wrong, contradicts the reference, or doesn't answer"
    )
    reason: str = Field(description="One sentence explaining the verdict")


class ClaimCheck(BaseModel):
    claim: str = Field(description="One factual claim made in the answer")
    supported: bool = Field(description="True if the cited/provided sources state or directly imply this claim")


class FaithfulnessVerdict(BaseModel):
    claims: list[ClaimCheck] = Field(description="Every factual claim in the answer, each checked against the sources")


CORRECTNESS_PROMPT = """You grade answers from a research assistant against a reference answer.

Question: {question}

Reference answer: {reference}

Assistant's answer: {answer}

Judge only whether the assistant's answer conveys the key facts of the reference. Extra correct detail is fine. Different wording is fine. Return a verdict and a one-sentence reason."""

FAITHFULNESS_PROMPT = """You check whether an answer is grounded in its sources.

Question: {question}

Sources:
{sources}

Answer: {answer}

List every factual claim the answer makes (ignore statements like "the sources don't say"). For each claim, decide whether the sources above state it or directly imply it. Use only the sources, not your own knowledge."""

SCORES = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


def get_judge():
    """Judge model: JUDGE_PROVIDER / JUDGE_MODEL, defaulting to the main LLM."""
    s = get_settings()
    provider = os.environ.get("JUDGE_PROVIDER") or s.llm_provider
    model = os.environ.get("JUDGE_MODEL") or s.llm_model
    return llm_module.make_llm(provider, model, temperature=0.0), f"{provider}/{model}"


def judge_correctness(judge, question: str, reference: str, answer: str) -> dict:
    result = llm_module.structured(CorrectnessVerdict, judge).invoke(
        CORRECTNESS_PROMPT.format(question=question, reference=reference, answer=answer)
    )
    return {"verdict": result.verdict, "score": SCORES[result.verdict], "reason": result.reason}


def judge_faithfulness(judge, question: str, answer: str, sources: list[dict]) -> dict:
    if not sources:
        return {"score": None, "claims": 0, "unsupported": []}
    formatted = "\n\n".join(f"[{s['n']}] {s['text'][:1500]}" for s in sources)
    result = llm_module.structured(FaithfulnessVerdict, judge).invoke(
        FAITHFULNESS_PROMPT.format(question=question, sources=formatted, answer=answer)
    )
    claims = result.claims or []
    if not claims:
        return {"score": None, "claims": 0, "unsupported": []}
    supported = sum(c.supported for c in claims)
    return {
        "score": round(supported / len(claims), 4),
        "claims": len(claims),
        "unsupported": [c.claim for c in claims if not c.supported],
    }
