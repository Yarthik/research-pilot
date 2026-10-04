"""Assemble the LangGraph workflow and expose ask() / ask_stream()."""

from __future__ import annotations

import time
from collections.abc import Iterator
from functools import lru_cache

from .nodes import generate, grade, plan, retrieve, route_after_grade
from .state import ResearchState

RESULT_KEYS = ("question", "answer", "citations", "invalid_citations", "insufficient", "sub_queries", "round", "trace")


def build_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("generate", generate)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {"retrieve": "retrieve", "generate": "generate"})
    graph.add_edge("generate", END)
    return graph.compile()


@lru_cache
def get_graph():
    return build_graph()


def _initial_state(question: str, mode: str | None, max_rounds: int | None) -> dict:
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    state: dict = {"question": question, "trace": []}
    if mode:
        state["mode"] = mode
    if max_rounds:
        state["max_rounds"] = max_rounds
    return state


def _result(state: dict, seconds: float) -> dict:
    result = {key: state.get(key) for key in RESULT_KEYS}
    result["citations"] = result["citations"] or []
    result["invalid_citations"] = result["invalid_citations"] or []
    result["trace"] = result["trace"] or []
    result["insufficient"] = bool(result["insufficient"])
    result["evidence_count"] = len(state.get("evidence") or [])
    result["seconds"] = seconds
    return result


def ask(question: str, mode: str | None = None, max_rounds: int | None = None) -> dict:
    """Run the agent on one question and return the answer, citations and step trace."""
    start = time.perf_counter()
    state = get_graph().invoke(_initial_state(question, mode, max_rounds))
    return _result(dict(state), round(time.perf_counter() - start, 2))


def ask_stream(question: str, mode: str | None = None, max_rounds: int | None = None) -> Iterator[dict]:
    """Yield {"type": "step", ...} events as each node finishes, then {"type": "result", ...}."""
    start = time.perf_counter()
    final: dict = {}
    for stream_mode, chunk in get_graph().stream(
        _initial_state(question, mode, max_rounds), stream_mode=["updates", "values"]
    ):
        if stream_mode == "values":
            final = dict(chunk)
            continue
        for node, update in chunk.items():
            for step in (update or {}).get("trace", []):
                yield {"type": "step", "node": node, **step}
    yield {"type": "result", **_result(final, round(time.perf_counter() - start, 2))}
