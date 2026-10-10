"""The Streamlit page itself, run headlessly with AppTest over the fake retriever and LLM."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("langgraph")

from streamlit.testing.v1 import AppTest  # noqa: E402

from researchpilot.config import reset_settings  # noqa: E402
from tests.conftest import fake_embed, make_chunk  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PLAN = {"sub_queries": ["LoRA rank decomposition"], "needs_web": False}
ANSWER = {"answer": "LoRA injects rank decomposition matrices [1].", "cited_sources": [1]}


def _app(monkeypatch, **env) -> AppTest:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    reset_settings()
    return AppTest.from_file(str(APP), default_timeout=60)


def _example_buttons(at: AppTest) -> list:
    return [b for b in at.button if b.key and b.key.startswith("ex-")]


def test_no_question_limit_by_default(loaded_retriever, monkeypatch):
    at = _app(monkeypatch).run()
    assert not at.exception
    assert not at.chat_input[0].disabled
    assert _example_buttons(at)
    assert not any("Demo limit" in c.value for c in at.sidebar.caption)


def test_question_limit_reached_disables_the_chat(loaded_retriever, monkeypatch):
    at = _app(monkeypatch, DEMO_MAX_QUESTIONS="2")
    at.session_state["questions_asked"] = 2
    at.run()
    assert not at.exception
    assert at.chat_input[0].disabled
    assert any("allows 2 questions per session" in i.value for i in at.info)
    assert not _example_buttons(at)  # examples would otherwise bypass the disabled input
    assert any("0 of 2 left" in c.value for c in at.sidebar.caption)


def test_asking_counts_towards_the_limit(loaded_retriever, fake_llm, monkeypatch):
    fake_llm.responses = {"ResearchPlan": [PLAN], "ContextGrade": [{"sufficient": True}], "Answer": [ANSWER]}
    at = _app(monkeypatch, DEMO_MAX_QUESTIONS="1").run()
    assert not at.chat_input[0].disabled
    assert any("1 of 1 left" in c.value for c in at.sidebar.caption)

    at.chat_input[0].set_value("How does LoRA work?").run()
    assert not at.exception
    assert at.session_state["questions_asked"] == 1
    assert any("rank decomposition" in m.value for m in at.markdown)  # the answer is still shown
    assert at.chat_input[0].disabled
    assert any("allows 1 question per session" in i.value for i in at.info)


def test_auto_ingest_retries_only_the_missing_sample_papers(retriever, monkeypatch):
    import streamlit as st

    from researchpilot.cli import read_sources_file
    from researchpilot.ingestion import pipeline

    samples = read_sources_file(str(APP.parents[1] / "data" / "sample_sources.txt"))
    calls = []

    def fake_ingest(urls, on_progress=None):
        calls.append(list(urls))
        failed = [{"source": urls[-1], "error": "timeout"}] if len(calls) == 1 else []  # first try: last one fails
        ok = [u for u in urls if not any(f["source"] == u for f in failed)]
        retriever.store.add([make_chunk(u, f"text of {u}", u, u) for u in ok], [fake_embed("x") for _ in ok])
        return {"ingested": [{"source": u, "title": u, "chunks": 1} for u in ok], "failed": failed}

    monkeypatch.setattr(pipeline, "ingest", fake_ingest)
    st.cache_resource.clear()
    _app(monkeypatch, AUTO_INGEST_SAMPLES="true", ALLOW_INGEST="false").run()
    assert calls == [samples]
    AppTest.from_file(str(APP), default_timeout=60).run()  # next page load
    assert calls == [samples, samples[-1:]]  # only the paper that failed is fetched again
    AppTest.from_file(str(APP), default_timeout=60).run()
    assert len(calls) == 2  # all present now: nothing more to do
