"""Assemble the LangGraph workflow and expose ask() / ask_stream()."""

from __future__ import annotations

import time
from collections.abc import Iterator
from functools import lru_cache

from .conversation import clean_history
from .nodes import (
    clarify,
    condense,
    generate,
    grade,
    plan,
    retrieve,
    route_after_condense,
    route_after_grade,
    route_start,
)
from .state import ResearchState

RESULT_KEYS = ("question", "answer", "citations", "invalid_citations", "insufficient", "sub_queries", "round", "trace")


def build_graph():
    """START -> [condense] -> plan -> retrieve -> grade -> (retrieve | generate) -> END.

    A question with chat history (or with newly attached files it refers to) is condensed into a
    standalone question first; one that names no document is answered by `clarify` with the list of
    documents. Any other question goes straight to `plan`, exactly as without conversation context.
    """
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(ResearchState)
    graph.add_node("condense", condense)
    graph.add_node("clarify", clarify)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("generate", generate)

    graph.add_conditional_edges(START, route_start, {"condense": "condense", "clarify": "clarify", "plan": "plan"})
    graph.add_conditional_edges("condense", route_after_condense, {"clarify": "clarify", "plan": "plan"})
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {"retrieve": "retrieve", "generate": "generate"})
    graph.add_edge("generate", END)
    graph.add_edge("clarify", END)
    return graph.compile()


@lru_cache
def get_graph():
    return build_graph()


def _initial_state(
    question: str,
    mode: str | None,
    max_rounds: int | None,
    owner: str | None = None,
    history: list | None = None,
    attachments: list[dict] | None = None,
) -> dict:
    question = (question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    state: dict = {"question": question, "trace": []}
    if mode:
        state["mode"] = mode
    if max_rounds:
        state["max_rounds"] = max_rounds
    if owner:
        state["owner"] = owner
    history = clean_history(history)
    if history:
        state["history"] = history
    attachments = [a for a in attachments or [] if isinstance(a, dict) and a.get("source")]
    if attachments and owner:  # attached files exist only in a UI session
        state["attachments"] = attachments
    return state


def _result(state: dict, seconds: float) -> dict:
    result = {key: state.get(key) for key in RESULT_KEYS}
    result["citations"] = result["citations"] or []
    result["invalid_citations"] = result["invalid_citations"] or []
    result["trace"] = result["trace"] or []
    result["insufficient"] = bool(result["insufficient"])
    result["evidence_count"] = len(state.get("evidence") or [])
    result["seconds"] = seconds
    # Only with conversation context: the question the agent actually worked on, and whether it asked back.
    if state.get("standalone"):
        result["standalone_question"] = state["standalone"]
    if state.get("clarification"):
        result["clarification"] = True
    return result


def ask(
    question: str,
    mode: str | None = None,
    max_rounds: int | None = None,
    owner: str | None = None,
    history: list | None = None,
    attachments: list[dict] | None = None,
) -> dict:
    """Run the agent on one question and return the answer, citations and step trace.

    `owner` is a UI session id: that session's private uploads are searched along with the shared corpus.
    `history` is the earlier chat ({"role": "user" | "assistant", "content": str}, oldest first), used to
    understand follow-up questions; `attachments` are that session's files attached since its last question.
    """
    start = time.perf_counter()
    state = get_graph().invoke(_initial_state(question, mode, max_rounds, owner, history, attachments))
    return _result(dict(state), round(time.perf_counter() - start, 2))


def ask_stream(
    question: str,
    mode: str | None = None,
    max_rounds: int | None = None,
    owner: str | None = None,
    history: list | None = None,
    attachments: list[dict] | None = None,
) -> Iterator[dict]:
    """Yield {"type": "step", ...} events as each node finishes, then {"type": "result", ...}."""
    start = time.perf_counter()
    final: dict = {}
    for stream_mode, chunk in get_graph().stream(
        _initial_state(question, mode, max_rounds, owner, history, attachments), stream_mode=["updates", "values"]
    ):
        if stream_mode == "values":
            final = dict(chunk)
            continue
        for node, update in chunk.items():
            for step in (update or {}).get("trace", []):
                yield {"type": "step", "node": node, **step}
    yield {"type": "result", **_result(final, round(time.perf_counter() - start, 2))}
