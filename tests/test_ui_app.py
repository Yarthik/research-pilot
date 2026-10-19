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
    assert not any("left" in c.value for c in at.caption)  # no limit: no questions-left line


def test_question_limit_reached_disables_the_chat(loaded_retriever, monkeypatch):
    at = _app(monkeypatch, DEMO_MAX_QUESTIONS="2")
    at.session_state["questions_asked"] = 2
    at.run()
    assert not at.exception
    assert at.chat_input[0].disabled
    assert any("allows 2 questions per session" in i.value for i in at.info)
    assert not _example_buttons(at)  # examples would otherwise bypass the disabled input
    assert any(c.value == "No questions left" for c in at.caption)


def test_asking_counts_towards_the_limit(loaded_retriever, fake_llm, monkeypatch):
    fake_llm.responses = {"ResearchPlan": [PLAN], "ContextGrade": [{"sufficient": True}], "Answer": [ANSWER]}
    at = _app(monkeypatch, DEMO_MAX_QUESTIONS="1").run()
    assert not at.chat_input[0].disabled
    assert any(c.value == "1 question left" for c in at.caption)

    at.chat_input[0].set_value("How does LoRA work?").run()
    assert not at.exception
    assert at.session_state["questions_asked"] == 1
    assert any(c.value == "No questions left" for c in at.caption)
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
    assert any(c.value == "2 questions left" for c in at.caption)
    at.run()
    assert any(expected in w.value for w in at.warning)
    assert not at.chat_input[0].disabled


HEADING = "ResearchPilot"
SUBTITLE = (
    "Ask a question. The agent searches, checks its evidence, searches again if needed, and answers with citations."
)
PLACEHOLDER = "Ask a question…"  # the same in every mode


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
    at = _app(monkeypatch).run()
    assert not at.exception
    _assert_landing_copy(at)
    assert _placeholder(at) == PLACEHOLDER  # ALLOW_INGEST=true: the visitor's own documents
    assert at.selectbox[0].label == "Search method"
    labels = ["Hybrid (recommended)", "Hybrid + reranker", "Keyword (BM25)", "Semantic (vector)"]
    assert at.selectbox[0].options == labels
    assert at.selectbox[0].help and "Keyword (BM25)" in at.selectbox[0].help
    assert not at.toggle  # agent steps are always available as one collapsed line
    new_chat = at.button(key="rp-new-chat")
    assert new_chat.disabled  # always in the sidebar, enabled once a chat exists, so nothing shifts
    assert at.chat_input[0].proto.accept_file == 0  # shared uploads are in the sidebar, not the composer

    assert not any(m.value.lstrip().startswith("1. **Plan**") for m in at.markdown)  # a dialog, not inline
    at.button(key="rp-how").click().run()
    text = " ".join(m.value for m in at.markdown)
    assert "**Plan**" in text and "**Grade**" in text and "Currently selected: **Hybrid (recommended)**" in text
    assert "github.com/Yarthik/research-pilot#evaluation" in text

    sidebar = " ".join(c.value for c in at.sidebar.caption)
    assert "Built by Kartik Kaushik" in sidebar and "https://github.com/Yarthik/research-pilot" in sidebar
    assert "  \ngpt-oss-120b via Groq" in sidebar  # the model on its own line, so it never breaks at a hyphen
    assert any(e.label == "Manage library" for e in at.sidebar.expander)  # ALLOW_INGEST=true: editable
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
    assert _placeholder(at) == PLACEHOLDER
    links = " ".join(m.value for m in at.sidebar.markdown)
    title = "LoRA: Low-Rank Adaptation of Large Language Models"
    assert f'href="{lora}" title="{title}"' in links  # the full, title-cased title on hover
    assert '<span class="rp-doc-t">LoRA</span>' in links  # a short name in the list
    assert "LORA: LOW-RANK" not in links
    assert any("read-only" in c.value for c in at.sidebar.caption)
    assert not at.sidebar.expander  # nothing to manage in a read-only demo


def test_read_only_mode_with_other_documents_has_a_generic_placeholder(loaded_retriever, monkeypatch):
    at = _app(monkeypatch, ALLOW_INGEST="false").run()
    _assert_landing_copy(at)
    assert _placeholder(at) == PLACEHOLDER


def test_answer_shows_collapsed_steps_chips_and_source_cards(loaded_retriever, fake_llm, monkeypatch):
    fake_llm.responses = {"ResearchPlan": [PLAN], "ContextGrade": [{"sufficient": True}], "Answer": [ANSWER]}
    at = _app(monkeypatch).run()
    at.chat_input[0].set_value("How does LoRA work?").run()
    assert not at.exception
    (status,) = [s for s in at.status if s.label.startswith("Searched")]
    assert status.label.startswith("Searched 1 round · ") and " passages · " in status.label  # singular/plural
    assert any("Planned 1 search: LoRA rank decomposition" in m.value for m in status.markdown)
    answer = next(m.value for m in at.markdown if "LoRA injects" in m.value)
    assert '<a class="rp-cite" href="#m1-1"' in answer  # the [1] marker is a chip linking to its source card
    assert any('class="rp-anchor" id="m1-1"' in m.value for m in at.markdown)  # the chip's target
    card = at.button(key="src-m1-1")
    assert "**LoRA**" in card.label and ":gray[p. 1]" in card.label
    card.click().run()  # opens the full passage in a dialog
    assert any(m.value.startswith("LoRA freezes the pretrained weights") for m in at.markdown)
    assert not at.get("link_button")  # lora.pdf is not an arXiv URL, so no "Open PDF" button
    assert any("From your file lora.pdf" in c.value for c in at.caption)
    assert "(s)" not in " ".join(s.label for s in at.status)


ZEBRAS = b"Zebras migrate across the Serengeti every year, following the rains."


def _session_app(monkeypatch, owner: str, **env) -> AppTest:
    at = _app(monkeypatch, ALLOW_INGEST="false", ALLOW_SESSION_UPLOADS="true", **env)
    at.session_state["owner"] = owner  # what the page would generate for a new browser session
    return at


def _sidebar_text(at: AppTest) -> str:
    return " ".join(m.value for m in at.sidebar.markdown) + " ".join(c.value for c in at.sidebar.caption)


def test_private_uploads_are_listed_only_in_their_session_and_can_be_removed(loaded_retriever, monkeypatch):
    from researchpilot.retrieval.session import get_registry

    get_registry().add_file("alice", "zebras.txt", ZEBRAS)
    alice = _session_app(monkeypatch, "alice").run()
    assert not alice.exception
    text = _sidebar_text(alice)
    assert "Your documents" in text and '<span class="rp-doc-t">zebras</span>' in text and "Private to this tab" in text
    assert "Shared with every visitor · read-only" in text
    assert alice.chat_input[0].proto.accept_file == 2  # attach files in the composer (multiple)
    assert list(alice.chat_input[0].proto.file_type) == [".pdf", ".txt", ".md"]
    assert alice.button(key="rm-zebras.txt").label == "zebras"  # a removable chip by the composer
    assert _placeholder(alice) == PLACEHOLDER

    bob = _session_app(monkeypatch, "bob").run()
    assert "zebras" not in _sidebar_text(bob) and "Your documents" not in _sidebar_text(bob)
    assert not [b for b in bob.button if b.key and b.key.startswith("rm-")]
    assert bob.chat_input[0].proto.accept_file == 2  # bob can attach his own

    alice.button(key="rp-how").click().run()
    how = " ".join(m.value for m in alice.markdown)
    assert "with the + button in the message box" in how and "paperclip" not in how

    alice.button(key="rm-zebras.txt").click().run()
    assert get_registry().documents("alice") == []
    assert "zebras" not in _sidebar_text(alice) and "Your documents" not in _sidebar_text(alice)


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
    assert any("**zebras**" in b.label for b in alice.button if b.key and b.key.startswith("src-"))  # her file
    assert alice.chat_input[0].disabled and alice.session_state["questions_asked"] == 1

    bob = _session_app(monkeypatch, "bob").run()
    bob.chat_input[0].set_value("Why do zebras migrate?").run()
    assert not any("zebras" in b.label for b in bob.button if b.key and b.key.startswith("src-"))


@pytest.mark.parametrize("env", [{"ALLOW_INGEST": "false"}, {"ALLOW_INGEST": "true", "ALLOW_SESSION_UPLOADS": "true"}])
def test_private_uploads_only_appear_when_enabled_on_a_read_only_demo(loaded_retriever, monkeypatch, env):
    at = _app(monkeypatch, **env).run()
    assert not at.exception
    assert at.chat_input[0].proto.accept_file == 0  # no attach button
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
    assert any("read-only" in c.value for c in at.sidebar.caption)
    assert not at.get("file_uploader") and not at.sidebar.expander
    assert at.chat_input[0].proto.accept_file == 0
    assert any(c.value == "3 questions left" for c in at.caption)


def test_source_cards_are_in_citation_order(loaded_retriever, fake_llm, monkeypatch):
    fake_llm.responses = {
        "ResearchPlan": [PLAN],
        "ContextGrade": [{"sufficient": True}],
        "Answer": [{"answer": "Frozen weights [2]. Low-rank matrices [1].", "cited_sources": [2, 1]}],
    }
    at = _app(monkeypatch).run()
    at.chat_input[0].set_value("How does LoRA work?").run()
    assert not at.exception
    cards = [b.key for b in at.button if b.key and b.key.startswith("src-")]
    assert cards == ["src-m1-1", "src-m1-2"]  # cited [2] first, shown in number order


def test_an_unexpected_failure_shows_a_friendly_message_and_is_not_counted(loaded_retriever, monkeypatch):
    import sys

    sys.path.insert(0, str(APP.parent))
    import backend

    def failing_stream(self, question, mode=None, owner=None):
        yield {"type": "step", "step": "plan", "queries": ["q"], "needs_web": False, "seconds": 0.1}
        raise RuntimeError("disk I/O error at /secret/path")

    monkeypatch.setattr(backend.LocalBackend, "ask_stream", failing_stream)
    at = _app(monkeypatch, DEMO_MAX_QUESTIONS="2").run()
    at.chat_input[0].set_value("How does LoRA work?").run()
    assert not at.exception
    assert [e.value for e in at.error] == [
        "Something went wrong while answering, so this question wasn't counted. Please try again in a moment."
    ]
    page = " ".join(el.value for group in (at.markdown, at.error, at.caption) for el in group)
    assert "disk I/O" not in page and "/secret/path" not in page  # details go to the log only
    assert at.session_state["questions_asked"] == 0 and any(c.value == "2 questions left" for c in at.caption)
    assert not at.status  # no live status left hanging on "Searching the documents…"
    at.run()
    assert len(at.error) == 1  # kept in the history, not lost on the next rerun


def test_the_composer_says_when_the_demo_limit_is_reached(loaded_retriever, monkeypatch):
    at = _app(monkeypatch, DEMO_MAX_QUESTIONS="1")
    at.session_state["questions_asked"] = 1
    at.run()
    assert at.chat_input[0].disabled and _placeholder(at) == "Demo limit reached"
