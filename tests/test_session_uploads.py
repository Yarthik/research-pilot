"""Private, temporary uploads per UI session (researchpilot/retrieval/session.py)."""

import pytest

from researchpilot.config import RETRIEVAL_MODES
from researchpilot.retrieval import session as session_module
from researchpilot.retrieval.session import (
    MAX_CHARS,
    MAX_FILE_MB,
    MAX_FILES,
    MAX_PDF_PAGES,
    SessionRegistry,
    SessionRetriever,
    SessionsFull,
    SessionUploadError,
)
from tests.conftest import CORPUS
from tests.test_config_and_loaders import make_pdf

ALICE_TEXT = b"Zebras migrate across the Serengeti every year, following the rains and fresh grass."
BOB_TEXT = b"Quokkas live on Rottnest Island and are known as the happiest marsupials."
SHARED_IDS = {c.id for c in CORPUS}


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def registry(loaded_retriever, clock):
    """A registry over the fake shared corpus (embeddings are the conftest fakes)."""
    return SessionRegistry(clock=clock)


def _ids(results) -> set[str]:
    return {chunk.id for chunk, _ in results}


def _sources(results) -> set[str]:
    return {chunk.source for chunk, _ in results}


# --------------------------------------------------------------------------- isolation
@pytest.mark.parametrize("mode", RETRIEVAL_MODES)
def test_a_session_never_retrieves_another_sessions_chunks(registry, loaded_retriever, mode):
    registry.add_file("alice", "zebras.txt", ALICE_TEXT)
    registry.add_file("bob", "quokkas.txt", BOB_TEXT)
    alice = registry.retriever("alice", loaded_retriever)
    bob = registry.retriever("bob", loaded_retriever)
    stranger = registry.retriever("stranger", loaded_retriever)

    for query in ("zebras migrate Serengeti", "quokkas Rottnest marsupials", "LoRA trainable parameters"):
        for k in (1, 5, 50):
            assert "quokkas.txt" not in _sources(alice.retrieve(query, mode=mode, final_k=k))
            assert "zebras.txt" not in _sources(bob.retrieve(query, mode=mode, final_k=k))
            assert not _sources(stranger.retrieve(query, mode=mode, final_k=k)) & {"zebras.txt", "quokkas.txt"}
    # each session does find its own document
    assert "zebras.txt" in _sources(alice.retrieve("zebras migrate Serengeti", mode=mode))
    assert "quokkas.txt" in _sources(bob.retrieve("quokkas Rottnest marsupials", mode=mode))


@pytest.mark.parametrize("mode", RETRIEVAL_MODES)
def test_a_session_without_uploads_gets_the_shared_retriever_unchanged(registry, loaded_retriever, mode):
    registry.add_file("alice", "zebras.txt", ALICE_TEXT)
    assert registry.retriever("bob", loaded_retriever) is loaded_retriever
    assert registry.retriever(None, loaded_retriever) is loaded_retriever
    before = loaded_retriever.retrieve("LoRA trainable parameters", mode=mode, final_k=6)
    assert _ids(before) <= SHARED_IDS
    assert loaded_retriever.store.count() == len(CORPUS)  # the shared store is never written


def test_one_sessions_files_do_not_change_another_sessions_scores(registry, loaded_retriever):
    registry.add_file("bob", "quokkas.txt", BOB_TEXT)
    before = registry.retriever("bob", loaded_retriever).retrieve("LoRA weights quokkas", mode="bm25", final_k=10)
    registry.add_file("alice", "zebras.txt", ALICE_TEXT + b" LoRA weights")  # shares query terms with bob's search
    after = registry.retriever("bob", loaded_retriever).retrieve("LoRA weights quokkas", mode="bm25", final_k=10)
    assert [(c.id, s) for c, s in before] == [(c.id, s) for c, s in after]


def test_session_chunks_get_their_own_ids_even_if_named_like_a_shared_source(registry, loaded_retriever):
    registry.add_file("alice", "lora.pdf.txt", ALICE_TEXT)
    alice = registry.retriever("alice", loaded_retriever)
    assert isinstance(alice, SessionRetriever)
    own = [chunk for chunk in alice.docs.chunks]
    assert own and all(chunk.id.startswith("session-") for chunk in own)


def test_the_agent_searches_only_the_askers_private_documents(registry, loaded_retriever, fake_llm, monkeypatch):
    from researchpilot.agent.graph import ask

    monkeypatch.setattr(session_module, "get_registry", lambda: registry)
    registry.add_file("alice", "zebras.txt", ALICE_TEXT)
    plan = {"sub_queries": ["zebras migrate Serengeti"], "needs_web": False}
    fake_llm.responses = {
        "ResearchPlan": [plan],
        "ContextGrade": [{"verdict": "sufficient"}],
        "Answer": [{"answer": "They follow the rains [1].", "cited_sources": [1]}],
    }
    alice = ask("Why do zebras migrate?", owner="alice")
    assert alice["citations"][0]["source"] == "zebras.txt"
    for owner in ("bob", None):
        result = ask("Why do zebras migrate?", owner=owner)
        assert all(c["source"] != "zebras.txt" for c in result["citations"])
        answer_prompt = next(prompt for name, prompt in fake_llm.calls[::-1] if name == "Answer")
        assert "Serengeti" not in answer_prompt  # not even in the evidence the model saw


# --------------------------------------------------------------------------- limits
def test_only_pdf_txt_and_md_files_are_accepted(registry):
    with pytest.raises(SessionUploadError, match="only PDF, TXT and MD"):
        registry.add_file("alice", "slides.pptx", b"x")
    registry.add_file("alice", "notes.md", b"# Notes\nZebras are striped.")
    assert [d["source"] for d in registry.documents("alice")] == ["notes.md"]


def test_files_over_the_size_limit_are_rejected(registry):
    too_big = b"a " * (MAX_FILE_MB * 1024 * 1024 // 2 + 1)
    with pytest.raises(SessionUploadError, match=f"the limit is {MAX_FILE_MB} MB"):
        registry.add_file("alice", "big.txt", too_big)
    assert registry.documents("alice") == []


def test_at_most_two_files_per_session_and_a_same_name_upload_replaces(registry, loaded_retriever):
    assert MAX_FILES == 2
    registry.add_file("alice", "one.txt", ALICE_TEXT)
    registry.add_file("alice", "two.txt", BOB_TEXT)
    with pytest.raises(SessionUploadError, match="up to 2 files"):
        registry.add_file("alice", "three.txt", b"Three")
    registry.add_file("alice", "one.txt", b"Penguins huddle together to keep warm in the Antarctic winter.")
    assert sorted(d["source"] for d in registry.documents("alice")) == ["one.txt", "two.txt"]
    alice = registry.retriever("alice", loaded_retriever)
    texts = " ".join(c.text for c in alice.docs.chunks)
    assert "Penguins" in texts and "Zebras" not in texts  # the old one.txt is gone, not duplicated


def test_long_pdfs_are_indexed_up_to_the_page_limit(registry, loaded_retriever):
    # Was: rejected outright, while the question was still sent. Now the first pages are kept and the visitor told.
    total = MAX_PDF_PAGES + 30
    info = registry.add_file("alice", "long.pdf", make_pdf([f"Zebra page marker{i + 1}" for i in range(total)]))
    assert info["pages"] == MAX_PDF_PAGES
    assert info["kept"] == {"unit": "pages", "kept": MAX_PDF_PAGES, "total": total}
    texts = " ".join(c.text for c in registry.retriever("alice", loaded_retriever).docs.chunks)
    assert f"marker{MAX_PDF_PAGES} " in texts + " " and f"marker{MAX_PDF_PAGES + 1}" not in texts
    info = registry.add_file("alice", "ok.pdf", make_pdf([f"Zebra page {i}" for i in range(MAX_PDF_PAGES)]))
    assert info["pages"] == MAX_PDF_PAGES and info["kept"] is None  # within the limits: nothing to say


@pytest.mark.parametrize(
    "name, data",
    [("empty.txt", b"   \n "), ("scan.pdf", make_pdf([], streams=["q Q"]))],
)
def test_files_without_text_get_a_clear_error(registry, name, data):
    with pytest.raises(SessionUploadError, match="no extractable text"):
        registry.add_file("alice", name, data)
    assert registry.active_sessions() == 0


def test_unreadable_pdfs_are_rejected(registry):
    with pytest.raises(SessionUploadError, match="couldn't be read as a PDF"):
        registry.add_file("alice", "broken.pdf", b"%PDF-1.4 this is not really a PDF")


def test_text_heavy_files_are_cut_at_the_character_limit_on_a_word_boundary(registry, loaded_retriever):
    words = MAX_CHARS // 5 + 5000
    info = registry.add_file("alice", "huge.txt", b"word " * words)
    assert info["kept"] == {"unit": "characters", "kept": MAX_CHARS, "total": words * 5}
    chunks = registry.retriever("alice", loaded_retriever).docs.chunks
    assert sum(len(c.text) for c in chunks) <= MAX_CHARS + len(chunks) * 200  # chunk overlap repeats a little
    assert all(c.text.split()[-1] == "word" for c in chunks)  # no word cut in half


def test_a_dense_pdf_is_cut_at_the_character_limit_too(registry, monkeypatch):
    monkeypatch.setattr(session_module, "MAX_CHARS", 500)
    pages = [" ".join(f"zebra{i}x{j}" for j in range(20)) for i in range(10)]  # ~200 characters a page
    info = registry.add_file("alice", "dense.pdf", make_pdf(pages))
    assert info["kept"]["unit"] == "pages" and info["kept"]["total"] == 10 and 2 <= info["kept"]["kept"] <= 3


# --------------------------------------------------------------------------- questions about one attached file
RESUME = b"Alex Morgan, frontend engineer. Six years building web apps with React, TypeScript and Node.js."


@pytest.mark.parametrize("mode", RETRIEVAL_MODES)
def test_a_scoped_retriever_searches_only_the_given_files(registry, loaded_retriever, mode):
    registry.add_file("alice", "resume.txt", RESUME)
    registry.add_file("alice", "zebras.txt", ALICE_TEXT)
    scoped = registry.retriever("alice", loaded_retriever, scope=["resume.txt"])
    assert scoped.scope == ["resume.txt"]
    # "ReAct" in the shared corpus matches the resume's "React" for BM25 (lower-cased): scoping keeps it out
    assert _sources(scoped.retrieve("React experience", mode=mode, final_k=5)) == {"resume.txt"}
    # the file's opening passage is always offered, even when the words of the question don't match it
    assert "resume.txt" in _sources(scoped.retrieve("What is it about?", mode=mode, final_k=5))
    assert registry.retriever("bob", loaded_retriever, scope=["resume.txt"]) is loaded_retriever  # not bob's


def test_a_scope_whose_files_are_gone_falls_back_to_the_sessions_retriever(registry, loaded_retriever):
    registry.add_file("alice", "zebras.txt", ALICE_TEXT)
    retriever = registry.retriever("alice", loaded_retriever, scope=["removed.pdf"])
    assert isinstance(retriever, SessionRetriever) and not hasattr(retriever, "scope")


def test_the_number_of_sessions_holding_uploads_is_capped(loaded_retriever, clock):
    registry = SessionRegistry(clock=clock, max_sessions=2)
    registry.add_file("a", "a.txt", ALICE_TEXT)
    registry.add_file("b", "b.txt", BOB_TEXT)
    with pytest.raises(SessionsFull, match="at capacity"):
        registry.add_file("c", "c.txt", ALICE_TEXT)
    registry.add_file("a", "a2.txt", BOB_TEXT)  # a session that already holds files can still add
    registry.remove("b", "b.txt")
    registry.add_file("c", "c.txt", ALICE_TEXT)  # room again
    assert registry.active_sessions() == 2


# --------------------------------------------------------------------------- removal and clean-up
def test_remove_deletes_the_file_and_its_chunks(registry, loaded_retriever):
    registry.add_file("alice", "zebras.txt", ALICE_TEXT)
    registry.add_file("alice", "quokkas.txt", BOB_TEXT)
    assert registry.remove("alice", "zebras.txt") is True
    alice = registry.retriever("alice", loaded_retriever)
    for mode in RETRIEVAL_MODES:
        assert "zebras.txt" not in _sources(alice.retrieve("zebras migrate Serengeti", mode=mode, final_k=50))
    assert registry.remove("alice", "zebras.txt") is False
    registry.remove("alice", "quokkas.txt")
    assert registry.active_sessions() == 0  # no files left: the session no longer counts towards the cap
    assert registry.retriever("alice", loaded_retriever) is loaded_retriever


def test_idle_sessions_are_cleaned_up_after_an_hour(registry, loaded_retriever, clock):
    registry.add_file("alice", "zebras.txt", ALICE_TEXT)
    registry.add_file("bob", "quokkas.txt", BOB_TEXT)
    clock.now += 3000
    registry.touch("bob")  # bob is active: page loads keep his files
    clock.now += 700  # alice idle for 3700 s, bob for 700 s
    assert registry.cleanup() == ["alice"]
    assert registry.documents("alice") == [] and registry.documents("bob")
    assert registry.retriever("alice", loaded_retriever) is loaded_retriever
    clock.now += 3000
    registry.retriever("bob", loaded_retriever)  # asking a question also counts as activity
    clock.now += 3000
    assert registry.cleanup() == []
    clock.now += 700
    registry.touch("someone-else")  # any visitor's page load cleans up idle sessions
    assert registry.active_sessions() == 0


def test_the_process_registry_has_a_background_cleanup_thread(monkeypatch):
    import threading

    started = []
    monkeypatch.setattr(threading.Thread, "start", lambda self: started.append(self.name))
    session_module.get_registry.cache_clear()
    first = session_module.get_registry()
    assert session_module.get_registry() is first
    assert started == ["session-uploads-cleanup"]
