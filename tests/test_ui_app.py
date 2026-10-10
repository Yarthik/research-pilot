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


@pytest.mark.parametrize(
    "daily, expected",
    [(True, "today's free-tier limit"), (False, "try again in a minute")],
)
def test_rate_limit_shows_a_friendly_message_not_the_raw_error(
    loaded_retriever, fake_llm, monkeypatch, daily, expected
):
    from tests.conftest import quota_error

    fake_llm.responses = {"ResearchPlan": [quota_error(daily=daily)]}
    at = _app(monkeypatch, DEMO_MAX_QUESTIONS="2").run()
    at.chat_input[0].set_value("How does LoRA work?").run()

    assert not at.exception  # no traceback
    assert not at.error  # no red error box
    assert any(expected in w.value for w in at.warning)
    page = " ".join(el.value for group in (at.markdown, at.warning, at.error, at.caption) for el in group)
    assert "org_secret" not in page and "429" not in page and "Sorry, the language model request failed" not in page

    # an unanswered question doesn't use up the visitor's allowance, and the message stays on later reruns
    assert at.session_state["questions_asked"] == 0
    assert any("2 of 2 left" in c.value for c in at.sidebar.caption)
    at.run()
    assert any(expected in w.value for w in at.warning)
    assert not at.chat_input[0].disabled


def test_a_boolean_secret_makes_the_page_read_only(loaded_retriever, monkeypatch):
    # Streamlit Community Cloud does not copy TOML booleans (ALLOW_INGEST = false) into the environment.
    monkeypatch.setenv("ALLOW_INGEST", "x")
    monkeypatch.delenv("ALLOW_INGEST")  # registers the cleanup for the variable the app is about to set
    at = _app(monkeypatch)
    at.secrets["ALLOW_INGEST"] = False
    at.secrets["DEMO_MAX_QUESTIONS"] = "3"
    at.run()
    assert not at.exception
    assert any("fixed set of documents" in i.value for i in at.sidebar.info)
    assert not at.get("file_uploader")
    assert any("3 of 3 left" in c.value for c in at.sidebar.caption)
