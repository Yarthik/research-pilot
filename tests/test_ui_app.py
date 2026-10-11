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
    assert not any("questions left" in c.value for c in at.sidebar.caption)


def test_question_limit_reached_disables_the_chat(loaded_retriever, monkeypatch):
    at = _app(monkeypatch, DEMO_MAX_QUESTIONS="2")
    at.session_state["questions_asked"] = 2
    at.run()
    assert not at.exception
    assert at.chat_input[0].disabled
    assert any("allows 2 questions per session" in i.value for i in at.info)
    assert not _example_buttons(at)  # examples would otherwise bypass the disabled input
    assert any("0 of 2 questions left" in c.value for c in at.sidebar.caption)


def test_asking_counts_towards_the_limit(loaded_retriever, fake_llm, monkeypatch):
    fake_llm.responses = {"ResearchPlan": [PLAN], "ContextGrade": [{"sufficient": True}], "Answer": [ANSWER]}
    at = _app(monkeypatch, DEMO_MAX_QUESTIONS="1").run()
    assert not at.chat_input[0].disabled
    assert any("1 of 1 questions left" in c.value for c in at.sidebar.caption)

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
    first = _app(monkeypatch, AUTO_INGEST_SAMPLES="true", ALLOW_INGEST="false").run()
    assert calls == [samples]
    assert not first.exception
    assert any("Setting up the demo" in m.value for m in first.markdown)  # the first-start panel explains the wait
    assert any("1 paper couldn't be downloaded" in w.value for w in first.warning)
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
    assert any("2 of 2 questions left" in c.value for c in at.sidebar.caption)
    at.run()
    assert any(expected in w.value for w in at.warning)
    assert not at.chat_input[0].disabled


HEADING = "ResearchPilot"
SUBTITLE = (
    "Ask a question. The agent searches, checks its evidence, searches again if needed, and answers with citations."
)
DEMO_PLACEHOLDER = "Ask about Transformers, RAG, LoRA, ReAct or chain-of-thought…"
GENERIC_PLACEHOLDER = "Ask a question about your documents…"


def _placeholder(at: AppTest) -> str:
    return at.chat_input[0].proto.placeholder


def _assert_landing_copy(at: AppTest) -> None:
    """The same heading and subtitle in every mode."""
    assert [t.value for t in at.title] == [HEADING]
    assert any(m.value == SUBTITLE for m in at.markdown)


def _page_text(at: AppTest) -> str:
    groups = (at.header, at.markdown, at.caption, at.info, at.warning, at.expander, at.button, at.toggle)
    return " ".join(str(getattr(el, "value", None) or getattr(el, "label", "")) for g in groups for el in g)


def test_own_documents_mode_copy_settings_and_footer(loaded_retriever, monkeypatch):
    from researchpilot import __version__

    at = _app(monkeypatch).run()
    assert not at.exception
    _assert_landing_copy(at)
    assert _placeholder(at) == GENERIC_PLACEHOLDER  # ALLOW_INGEST=true: the visitor's own documents
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


def test_demo_mode_suggests_the_papers_topics_and_shows_clean_linked_titles(retriever, monkeypatch):
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
    _assert_landing_copy(at)
    assert _placeholder(at) == DEMO_PLACEHOLDER
    links = " ".join(m.value for m in at.sidebar.markdown)
    assert f"[LoRA: Low-Rank Adaptation of Large Language Models]({lora})" in links
    assert "LORA: LOW-RANK" not in links
    assert any("fixed set of documents" in i.value for i in at.sidebar.info)


def test_read_only_mode_with_other_documents_has_a_generic_placeholder(loaded_retriever, monkeypatch):
    at = _app(monkeypatch, ALLOW_INGEST="false").run()
    _assert_landing_copy(at)
    assert _placeholder(at) == GENERIC_PLACEHOLDER


def test_answer_shows_agent_steps_and_a_natural_footer(loaded_retriever, fake_llm, monkeypatch):
    fake_llm.responses = {"ResearchPlan": [PLAN], "ContextGrade": [{"sufficient": True}], "Answer": [ANSWER]}
    at = _app(monkeypatch).run()
    at.chat_input[0].set_value("How does LoRA work?").run()
    assert not at.exception
    steps = next(e for e in at.expander if e.label.startswith("Agent steps"))
    assert any("Planned 1 search: LoRA rank decomposition" in m.value for m in steps.markdown)
    assert any(c.value.startswith("Answered in") and "1 search round ·" in c.value for c in at.caption)
    answer = next(m.value for m in at.markdown if "LoRA injects" in m.value)
    assert '<a class="rp-cite" href="#m1-1"' in answer  # the [1] marker is a chip linking to its source card
    assert any('id="m1-1"' in m.value and "rp-src-title" in m.value for m in at.markdown)
    assert any(e.label == "Show full passage" for e in at.expander)
    assert "(s)" not in " ".join(c.value for c in at.caption)

    at.toggle[0].set_value(False).run()  # steps hidden, the answer stays
    assert not any(e.label.startswith("Agent steps") for e in at.expander)
    assert any("rank decomposition" in m.value for m in at.markdown)


ZEBRAS = b"Zebras migrate across the Serengeti every year, following the rains."


def _session_app(monkeypatch, owner: str, **env) -> AppTest:
    at = _app(monkeypatch, ALLOW_INGEST="false", ALLOW_SESSION_UPLOADS="true", **env)
    at.session_state["owner"] = owner  # what the page would generate for a new browser session
    at.session_state["upload_round"] = 0
    return at


def _sidebar_text(at: AppTest) -> str:
    return " ".join(m.value for m in at.sidebar.markdown) + " ".join(c.value for c in at.sidebar.caption)


def test_private_uploads_are_listed_only_in_their_session_and_can_be_removed(loaded_retriever, monkeypatch):
    from researchpilot.retrieval.session import get_registry

    get_registry().add_file("alice", "zebras.txt", ZEBRAS)
    alice = _session_app(monkeypatch, "alice").run()
    assert not alice.exception
    text = _sidebar_text(alice)
    assert "Your documents" in text and ":material/draft: zebras" in text and "Private to this browser tab" in text
    assert "Shared with every visitor and read-only" in text
    assert alice.get("file_uploader")  # one more file allowed
    assert _placeholder(alice) == GENERIC_PLACEHOLDER

    bob = _session_app(monkeypatch, "bob").run()
    assert "Your documents" in _sidebar_text(bob) and "zebras" not in _sidebar_text(bob)

    next(b for b in alice.sidebar.button if b.key == "rm-zebras.txt").click().run()
    assert get_registry().documents("alice") == []
    assert "zebras" not in _sidebar_text(alice)


def test_questions_search_the_askers_uploads_and_still_count_towards_the_limit(loaded_retriever, fake_llm, monkeypatch):
    from researchpilot.retrieval.session import get_registry

    get_registry().add_file("alice", "zebras.txt", ZEBRAS)
    fake_llm.responses = {
        "ResearchPlan": [{"sub_queries": ["zebras migrate Serengeti"], "needs_web": False}],
        "ContextGrade": [{"sufficient": True}],
        "Answer": [{"answer": "They follow the rains [1].", "cited_sources": [1]}],
    }
    alice = _session_app(monkeypatch, "alice", DEMO_MAX_QUESTIONS="1").run()
    alice.chat_input[0].set_value("Why do zebras migrate?").run()
    assert not alice.exception
    assert any('class="rp-src-title">zebras' in m.value for m in alice.markdown)  # cited from her private file
    assert alice.chat_input[0].disabled and alice.session_state["questions_asked"] == 1

    bob = _session_app(monkeypatch, "bob").run()
    bob.chat_input[0].set_value("Why do zebras migrate?").run()
    assert not any("rp-src-title" in m.value and "zebras" in m.value for m in bob.markdown)


@pytest.mark.parametrize("env", [{"ALLOW_INGEST": "false"}, {"ALLOW_INGEST": "true", "ALLOW_SESSION_UPLOADS": "true"}])
def test_private_uploads_only_appear_when_enabled_on_a_read_only_demo(loaded_retriever, monkeypatch, env):
    at = _app(monkeypatch, **env).run()
    assert not at.exception
    assert "Your documents" not in _sidebar_text(at)


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
    assert any("3 of 3 questions left" in c.value for c in at.sidebar.caption)
