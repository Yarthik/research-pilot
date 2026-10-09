"""Evaluation metrics. Pure functions, no models or API calls, so they are unit-tested."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, turn punctuation/hyphens/line breaks into single spaces, pad with spaces.

    PDF extraction is messy ("RAG-Token", "RAG Token", "rank-\\ndecomposition"); normalising both
    the chunk and the keyword makes matching robust, and the padding gives whole-word matches.
    """
    return " " + _NON_ALNUM.sub(" ", text.lower()).strip() + " "


def contains_any(text: str, keywords: list[str]) -> bool:
    haystack = normalize(text)
    return any(normalize(k) in haystack for k in keywords if k.strip())


def same_source(a: str, b: str) -> bool:
    """URLs may come back with a version suffix (…/2106.09685v2) or trailing slash."""
    a, b = a.rstrip("/").lower(), b.rstrip("/").lower()
    return a == b or a.startswith(b + "v") or b.startswith(a + "v")


@dataclass
class Hit:
    source: str
    text: str


def is_relevant(hit: Hit, expected_sources: list[str], keywords: list[str]) -> bool:
    """A retrieved chunk is relevant if it comes from an expected source and contains a key phrase."""
    return any(same_source(hit.source, s) for s in expected_sources) and contains_any(hit.text, keywords)


def first_relevant_rank(hits: list[Hit], expected_sources: list[str], keywords: list[str]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(hit, expected_sources, keywords):
            return rank
    return None


def source_recall(hits: list[Hit], expected_sources: list[str], keywords: list[str]) -> float:
    """Fraction of expected sources with at least one relevant chunk retrieved (matters for multi-doc questions)."""
    if not expected_sources:
        return 0.0
    found = sum(any(is_relevant(h, [s], keywords) for h in hits) for s in expected_sources)
    return found / len(expected_sources)


def retrieval_scores(hits: list[Hit], expected_sources: list[str], keywords: list[str]) -> dict:
    rank = first_relevant_rank(hits, expected_sources, keywords)
    return {
        "hit": 1.0 if rank else 0.0,
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
        "first_relevant_rank": rank,
        "source_recall": source_recall(hits, expected_sources, keywords),
    }


def mean(values: list[float]) -> float:
    values = [v for v in values if v is not None]
    return round(statistics.fmean(values), 4) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    values = sorted(v for v in values if v is not None)
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round(pct / 100 * (len(values) - 1))))
    return round(values[index], 4)


ABSTAIN_PATTERNS = re.compile(
    r"(do(es)? not (contain|include|mention|provide|cover|say)|don't (contain|mention|cover|have)|"
    r"not (mentioned|covered|found|provided|available) in|no (information|mention)|"
    r"couldn't find|could not find|cannot (answer|find|be answered)|can't answer|insufficient information|"
    r"not enough information|unable to (answer|find))",
    re.IGNORECASE,
)


def abstained(answer: str, insufficient_flag: bool) -> bool:
    """Did the agent decline to answer? Uses the structured flag, or wording when the flag is missing."""
    return bool(insufficient_flag) or bool(ABSTAIN_PATTERNS.search(answer or ""))


def declined(answer: str, insufficient_flag: bool, cited: list) -> bool:
    """For answerable questions: did the agent decline instead of answering?

    A cited answer counts as an answer even when the agent flags that some information was missing
    (e.g. "state of the art [3], but the exact figure isn't in the excerpts").
    """
    return not cited and abstained(answer, insufficient_flag)
