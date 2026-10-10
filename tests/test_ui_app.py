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


def _page_text(at: AppTest) -> str:
    groups = (at.header, at.markdown, at.caption, at.info, at.warning, at.expander, at.button, at.toggle)
    return " ".join(str(getattr(el, "value", None) or getattr(el, "label", "")) for g in groups for el in g)


def test_own_documents_mode_copy_settings_and_footer(loaded_retriever, monkeypatch):
    from researchpilot import __version__

    at = _app(monkeypatch).run()
    assert not at.exception
    assert at.header[0].value == "Ask questions about your documents"
    assert at.selectbox[0].label == "Search method"
    labels = ["Hybrid (recommended)", "Hybrid + reranker", "Keyword (BM25)", "Semantic (vector)"]
    assert at.selectbox[0].options == labels
    assert at.selectbox[0].help and "Keyword (BM25)" in at.selectbox[0].help
    steps = at.toggle[0]
    assert steps.label == "Show agent steps" and steps.value is True

    how = next(e for e in at.expander if e.label == "How it works")
    text = " ".join(m.value for m in how.markdown)
    assert "**Plan**" in text and "**Grade**" in text and "Currently selected: **Hybrid (recommended)**" in text
    assert "github.com/Yarthik/research-pilot#evaluation" in text

    sidebar = " ".join(c.value for c in at.sidebar.caption)
    assert "Agentic RAG with verifiable citations" in sidebar
    assert "Built by Kartik Kaushik" in sidebar and "https://github.com/Yarthik/research-pilot" in sidebar
    assert " via " in sidebar and f"v{__version__}" in sidebar
    page = _page_text(at) + sidebar
    assert "local mode" not in page and "chunks" not in page
    assert len(_example_buttons(at)) == 4


def test_demo_mode_names_the_sample_papers_with_clean_linked_titles(retriever, monkeypatch):
    from researchpilot.cli import read_sources_file

    samples = read_sources_file(str(APP.parents[1] / "data" / "sample_sources.txt"))
    titles = {url: f"Paper {i}" for i, url in enumerate(samples)}
    lora = next(url for url in samples if "2106.09685" in url)
    titles[lora] = "LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS"  # an index from before the title fix
    retriever.store.add(
        [make_chunk(u, f"text of {u}", u, titles[u]) for u in samples], [fake_embed("x") for _ in samples]
    )
    at = _app(monkeypatch, ALLOW_INGEST="false").run()
    assert not at.exception
    assert at.header[0].value == "Ask questions about five landmark AI papers"
    links = " ".join(m.value for m in at.sidebar.markdown)
    assert f"[LoRA: Low-Rank Adaptation of Large Language Models]({lora})" in links
    assert "LORA: LOW-RANK" not in links
    assert any("fixed set of documents" in i.value for i in at.sidebar.info)


def test_read_only_mode_with_other_documents_has_a_neutral_heading(loaded_retriever, monkeypatch):
    at = _app(monkeypatch, ALLOW_INGEST="false").run()
    assert at.header[0].value == "Ask questions about these documents"


def test_answer_shows_agent_steps_and_a_natural_footer(loaded_retriever, fake_llm, monkeypatch):
    fake_llm.responses = {"ResearchPlan": [PLAN], "ContextGrade": [{"sufficient": True}], "Answer": [ANSWER]}
    at = _app(monkeypatch).run()
    at.chat_input[0].set_value("How does LoRA work?").run()
    assert not at.exception
    steps = next(e for e in at.expander if e.label.startswith("Agent steps"))
    assert any("Planned 1 search: LoRA rank decomposition" in m.value for m in steps.markdown)
    assert any(c.value.startswith("Answered in") and "1 search round ·" in c.value for c in at.caption)
    assert "(s)" not in " ".join(c.value for c in at.caption)

    at.toggle[0].set_value(False).run()  # steps hidden, the answer stays
    assert not any(e.label.startswith("Agent steps") for e in at.expander)
    assert any("rank decomposition" in m.value for m in at.markdown)


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
