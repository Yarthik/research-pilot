"""LangGraph nodes: plan -> retrieve -> grade -> (retrieve again | generate).

Every LLM step degrades gracefully: if a model fails structured output or an API call errors,
the node falls back to a sensible default instead of failing the whole request.
"""

from __future__ import annotations

import logging
import time
from itertools import zip_longest

from .. import llm as llm_module
from ..config import get_settings
from ..ingestion.chunking import Chunk
from ..llm import FATAL_LLM_ERRORS
from ..retrieval import hybrid
from ..tools.web_search import web_search, web_search_available
from .citations import normalize_citations, parse_citations, split_valid
from .evidence import select_evidence
from .prompts import ANSWER_PROMPT, GRADE_PROMPT, PLAN_PROMPT
from .schemas import Answer, ContextGrade, ResearchPlan
from .state import ResearchState, Source

log = logging.getLogger(__name__)

NO_EVIDENCE_ANSWER = (
    "I couldn't find anything relevant in the ingested documents. "
    "Try ingesting more sources or rephrasing the question."
)


def _elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 2)


def _chunk_to_source(chunk: Chunk) -> Source:
    return {
        "id": chunk.id,
        "text": chunk.text,
        "title": chunk.title,
        "source": chunk.source,
        "page": chunk.page,
        "origin": "document",
    }


def format_context(evidence: list[Source], max_chars: int = 1500) -> str:
    parts = []
    for number, item in enumerate(evidence, start=1):
        page = f", p. {item['page']}" if item.get("page") else ""
        parts.append(f"[{number}] {item['title']}{page} ({item['origin']})\n{item['text'][:max_chars]}")
    return "\n\n".join(parts)


def _clean_queries(queries: list[str] | None, limit: int) -> list[str]:
    cleaned = [q.strip() for q in (queries or []) if q and q.strip()]
    return list(dict.fromkeys(cleaned))[:limit]


# --------------------------------------------------------------------------- plan
def plan(state: ResearchState) -> dict:
    start = time.perf_counter()
    question = state["question"]
    queries: list[str] = []
    needs_web = False
    error = None
    try:
        result = llm_module.structured(ResearchPlan).invoke(
            PLAN_PROMPT.format(question=question, web_available=web_search_available())
        )
        if result:
            queries = _clean_queries(result.sub_queries, 3)
            needs_web = bool(result.needs_web) and web_search_available()
    except FATAL_LLM_ERRORS:
        raise  # a missing key or package isn't transient: surface it instead of degrading
    except Exception as exc:  # smaller models sometimes fail structured output
        error = str(exc)[:200]
        log.warning("plan step failed, using the raw question: %s", exc)
    queries = queries or [question]
    step = {"step": "plan", "queries": queries, "needs_web": needs_web, "seconds": _elapsed(start)}
    if error:
        step["error"] = error
    return {"sub_queries": queries, "needs_web": needs_web, "round": 0, "evidence": [], "trace": [step]}


# --------------------------------------------------------------------------- retrieve
def retrieve(state: ResearchState) -> dict:
    start = time.perf_counter()
    settings = get_settings()
    retriever = hybrid.get_retriever()
    if state.get("owner"):  # a UI session: its private uploads too, never another session's
        from ..retrieval.session import get_registry

        retriever = get_registry().retriever(state["owner"], retriever)
    mode = state.get("mode") or settings.retrieval_mode
    existing = list(state.get("evidence") or [])
    seen = {item["id"] for item in existing}
    new: list[Source] = []
    added_web = 0

    per_query: list[list[Source]] = []
    for query in state["sub_queries"]:
        results = [_chunk_to_source(chunk) for chunk, _score in retriever.retrieve(query, mode=mode)]
        if state.get("needs_web"):
            results += web_search(query)
        per_query.append(results)
    # Interleave by rank (each sub-query's best hit first): when there is no scorer and the pool is
    # trimmed, the lowest-ranked hits of every sub-query go, rather than whole sub-queries.
    for item in (item for group in zip_longest(*per_query) for item in group if item is not None):
        if item["id"] not in seen:
            seen.add(item["id"])
            new.append(item)
            added_web += item.get("origin") == "web"

    # Re-score the whole pool against the question so retry and web results compete fairly.
    scorer = retriever.score if mode == "hybrid_rerank" else None
    evidence = select_evidence(state["question"], existing, new, settings.max_evidence, scorer)
    kept_new = len({item["id"] for item in evidence} & {item["id"] for item in new})

    round_number = state.get("round", 0) + 1
    return {
        "evidence": evidence,
        "round": round_number,
        "trace": [
            {
                "step": "retrieve",
                "round": round_number,
                "mode": mode,
                "queries": state["sub_queries"],
                "new_chunks": len(new) - added_web,
                "new_web_results": added_web,
                "kept_new": kept_new,
                "evidence": len(evidence),
                "seconds": _elapsed(start),
            }
        ],
    }


# --------------------------------------------------------------------------- grade
def _max_rounds(state: ResearchState) -> int:
    return state.get("max_rounds") or get_settings().max_retrieval_rounds


def grade(state: ResearchState) -> dict:
    start = time.perf_counter()
    evidence = state.get("evidence") or []
    sufficient, missing, new_queries, error, skipped = False, "", [], None, False

    if not evidence:
        missing = "nothing relevant was retrieved"
    elif state.get("round", 0) >= _max_rounds(state):
        # No retries left: skip the LLM call and go straight to answering.
        sufficient, skipped = True, True
    else:
        try:
            result = llm_module.structured(ContextGrade).invoke(
                GRADE_PROMPT.format(question=state["question"], context=format_context(evidence, 800))
            )
            sufficient = bool(result.sufficient) if result else True
            if result and not result.sufficient:
                missing = result.missing
                previous = {q.lower() for q in state.get("sub_queries") or []}
                new_queries = [q for q in _clean_queries(result.rewritten_queries, 2) if q.lower() not in previous]
        except FATAL_LLM_ERRORS:
            raise
        except Exception as exc:
            error = str(exc)[:200]
            log.warning("grade step failed, proceeding to answer: %s", exc)
            sufficient = True

    step = {
        "step": "grade",
        # A skipped grade made no verdict: None in the trace, while the state still routes to answering.
        "sufficient": None if skipped else sufficient,
        "missing": missing,
        "new_queries": new_queries,
        "seconds": _elapsed(start),
    }
    if skipped:
        step["skipped"] = True
    if error:
        step["error"] = error
    update: dict = {"sufficient": sufficient, "new_queries": new_queries, "trace": [step]}
    if new_queries:
        update["sub_queries"] = new_queries
    return update


def route_after_grade(state: ResearchState) -> str:
    if state.get("sufficient") or not state.get("new_queries") or state.get("round", 0) >= _max_rounds(state):
        return "generate"
    return "retrieve"


# --------------------------------------------------------------------------- generate
def generate(state: ResearchState) -> dict:
    start = time.perf_counter()
    evidence = state.get("evidence") or []
    if not evidence:
        return {
            "answer": NO_EVIDENCE_ANSWER,
            "citations": [],
            "invalid_citations": [],
            "insufficient": True,
            "trace": [{"step": "generate", "sources": 0, "cited": [], "seconds": _elapsed(start)}],
        }

    prompt = ANSWER_PROMPT.format(question=state["question"], context=format_context(evidence))
    error = None
    try:
        result = llm_module.structured(Answer).invoke(prompt)
        if result is None or not result.answer.strip():
            raise ValueError("empty structured output")
        answer, listed, insufficient = result.answer.strip(), result.cited_sources, result.insufficient_information
    except FATAL_LLM_ERRORS:
        raise
    except Exception as exc:
        error = str(exc)[:200]
        log.warning("structured answer failed, falling back to plain text: %s", exc)
        try:
            answer, listed, insufficient = llm_module.complete(prompt).strip(), [], False
        except FATAL_LLM_ERRORS:
            raise
        except Exception as inner:
            log.error("answer generation failed: %s", inner)
            answer = f"Sorry, the language model request failed: {str(inner)[:200]}"
            listed, insufficient = [], True

    # Trust the markers actually written in the text over the model's self-reported list.
    answer = normalize_citations(answer)
    numbers = parse_citations(answer) or list(dict.fromkeys(listed))
    valid, invalid = split_valid(numbers, len(evidence))
    citations = [{"n": n, **evidence[n - 1]} for n in valid]
    step = {"step": "generate", "sources": len(evidence), "cited": valid, "seconds": _elapsed(start)}
    if invalid:
        step["invalid_citations"] = invalid
    if error:
        step["error"] = error
    return {
        "answer": answer,
        "citations": citations,
        "invalid_citations": invalid,
        "insufficient": bool(insufficient),
        "trace": [step],
    }
