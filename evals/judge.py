"""LLM-as-judge for answer quality: correctness against a reference and faithfulness to sources."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from researchpilot import llm as llm_module
from researchpilot.config import get_settings


class CorrectnessVerdict(BaseModel):
    verdict: Literal["correct", "partially_correct", "incorrect"] = Field(
        description="correct: answers everything the question asks, consistent with the reference; "
        "partially_correct: answers only part of what the question asks, or is vague on a key fact; "
        "incorrect: wrong, contradicts the reference, or doesn't answer"
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

Judge whether the assistant's answer gives the key facts needed to answer the question, using the reference as the source of truth. Key facts are the parts of the reference that answer what the question asks. Details in the reference beyond what the question asks (background, extra comparisons, explanations of why) are optional: leaving them out is not a reason to lower the verdict. Extra correct detail is fine. Different wording is fine.

- correct: answers everything the question asks, consistent with the reference.
- partially_correct: answers only part of what the question asks, or is vague or imprecise on a key fact.
- incorrect: wrong, contradicts the reference, or does not answer the question (including saying the sources don't cover it).

Return a verdict and a one-sentence reason."""

# Bump when CORRECTNESS_PROMPT changes: verdicts record it, and `python -m evals rejudge` re-grades old ones.
# v2: key facts are what the question asks; extra context in the reference is optional (v1 was too strict).
CORRECTNESS_PROMPT_VERSION = 2

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
    return {
        "verdict": result.verdict,
        "score": SCORES[result.verdict],
        "reason": result.reason,
        "prompt_version": CORRECTNESS_PROMPT_VERSION,
    }


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
