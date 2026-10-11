"""Streamlit chat UI for ResearchPilot.

    streamlit run app/streamlit_app.py

Runs the agent in-process by default. Set API_URL (e.g. http://api:8000) to use the FastAPI
service instead, which is how docker compose wires the two containers together.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "app"):  # importable without `pip install -e .`
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit as st  # noqa: E402
from backend import get_backend  # noqa: E402
from render import (  # noqa: E402
    GITHUB_URL,
    answer_footer,
    describe_step,
    model_label,
    plural,
    rate_limit_notice,
    short_title,
    streamlit_markdown,
)
from runtime_env import ensure_modern_sqlite, load_streamlit_secrets  # noqa: E402

load_streamlit_secrets(st.secrets)  # hosted apps keep settings in secrets; this also covers TOML booleans

from researchpilot import __version__  # noqa: E402
from researchpilot.cli import read_sources_file  # noqa: E402
from researchpilot.config import get_settings  # noqa: E402
from researchpilot.ingestion.loaders import title_case_if_all_caps  # noqa: E402
from researchpilot.llm import LLMConfigError, LLMRateLimitError  # noqa: E402
from researchpilot.retrieval.session import MAX_FILE_MB, MAX_FILES, MAX_PDF_PAGES, SessionUploadError  # noqa: E402

SAMPLE_FILE = ROOT / "data" / "sample_sources.txt"
ICON = ":material/travel_explore:"
USER_AVATAR = ":material/person:"
EVAL_URL = f"{GITHUB_URL}#evaluation"
# One per kind of behaviour, from the eval set (evals/dataset.jsonl: l1 shortened, m2, re3, u2).
EXAMPLES = [
    ("One paper", "How does LoRA reduce the number of trainable parameters?"),
    ("Compares two papers", "How do RAG and LoRA differ in the way they adapt a pre-trained language model?"),
    (
        "May search twice",
        "What tool does ReAct use for knowledge-intensive tasks like HotpotQA, and which actions does it support?",
    ),
    ("Not in the papers", "What learning rate was used to train GPT-4?"),
]
MODE_LABELS = {  # selectbox order
    "hybrid": "Hybrid (recommended)",
    "hybrid_rerank": "Hybrid + reranker",
    "bm25": "Keyword (BM25)",
    "vector": "Semantic (vector)",
}
MODE_HELP = {
    "hybrid": "keyword and semantic search, merged by rank fusion; the best accuracy in the evaluation",
    "hybrid_rerank": "hybrid, then a cross-encoder re-reads the top passages; about 1 s slower, not better here",
    "bm25": "exact word matches; good for names and numbers",
    "vector": "matches meaning, even when the wording differs",
}
STEP_ICONS = {
    "plan": ":material/checklist:",
    "retrieve": ":material/search:",
    "grade": ":material/fact_check:",
    "generate": ":material/edit_note:",
}
SUBTITLE = (
    "Ask a question. The agent searches, checks its evidence, searches again if needed, and answers with citations."
)
DEMO_PLACEHOLDER = "Ask about Transformers, RAG, LoRA, ReAct or chain-of-thought…"
GENERIC_PLACEHOLDER = "Ask a question about your documents…"
PRIVATE_NOTE = (
    "Private to this browser tab and temporary: removed when you click Remove, reload the page, "
    "or after an hour without activity."
)

st.set_page_config(page_title="ResearchPilot", page_icon=ICON, layout="wide")

try:
    ensure_modern_sqlite()  # before anything imports chromadb
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()


@st.cache_resource
def backend():
    return get_backend()


# --------------------------------------------------------------------------- helpers
def display_title(title: str, limit: int = 90) -> str:
    return short_title(title_case_if_all_caps(title or ""), limit)


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    st.markdown("**Sources**")
    for c in citations:
        page = f", p. {c['page']}" if c.get("page") else ""
        icon = ":material/public:" if c.get("origin") == "web" else ":material/description:"
        with st.expander(f"[{c['n']}] {display_title(c['title'])}{page}", icon=icon):
            st.caption(c["source"])
            st.markdown(streamlit_markdown(c["text"]))


def render_steps(trace: list[dict]) -> None:
    with st.expander(f"Agent steps ({len(trace)})", icon=":material/route:"):
        for step in trace:
            st.markdown(f"{STEP_ICONS.get(step.get('step'), '•')} {describe_step(step)}")


def render_result(result: dict, show_steps: bool) -> None:
    if show_steps and result.get("trace"):
        render_steps(result["trace"])
    if result.get("insufficient"):
        covered = "don't fully answer" if result.get("citations") else "don't cover"
        st.warning(f"The documents {covered} this question.", icon=":material/info:")
    st.markdown(streamlit_markdown(result["answer"]))
    render_citations(result.get("citations", []))
    st.caption(answer_footer(result))


def show_report(status, report: dict) -> None:
    for item in report["ingested"]:
        status.write(f":material/check: {display_title(item['title'])} ({plural(item['chunks'], 'passage')})")
    for item in report["failed"]:
        status.write(f":material/close: {item['source']}: {item['error']}")
    status.update(label=f"Added {len(report['ingested'])}, failed {len(report['failed'])}", state="complete")


def ingest_urls(urls: list[str]) -> None:
    with st.status(f"Adding {plural(len(urls), 'source')}...", expanded=True) as status:
        report = backend().ingest_urls(urls, on_progress=lambda s, i, n: status.write(f"[{i}/{n}] {s}"))
        show_report(status, report)


@st.cache_resource(show_spinner="Loading the sample papers (first start only)...")
def auto_ingest_samples() -> bool:
    """Ingest any sample papers missing from the index, so a fresh deployment works immediately.

    Returns False if a download failed; the caller then clears the cache so the next page load retries.
    """
    if not SAMPLE_FILE.exists():
        return True
    present = {s["source"] for s in backend().sources()}
    missing = [url for url in read_sources_file(str(SAMPLE_FILE)) if url not in present]
    return not missing or not backend().ingest_urls(missing)["failed"]


def only_sample_papers(sources: list[dict]) -> bool:
    samples = set(read_sources_file(str(SAMPLE_FILE))) if SAMPLE_FILE.exists() else set()
    return bool(sources) and all(s["source"] in samples for s in sources)


def session_uploads_panel(owner: str, my_docs: list[dict]) -> None:
    """The visitor's private files: list with Remove buttons, and an uploader while there is room."""
    st.markdown("**Your documents**")
    st.caption(f":material/lock: {PRIVATE_NOTE}")
    for doc in my_docs:
        left, right = st.columns([3, 2], vertical_alignment="center")
        left.markdown(f":material/draft: {display_title(doc['title'], 60)}")
        if right.button("Remove", key=f"rm-{doc['source']}", help=f"Remove {doc['source']}"):
            backend().remove_session_file(owner, doc["source"])
            st.rerun()
    for kind, text in st.session_state.pop("upload_notes", []):
        if kind == "ok":
            st.success(text, icon=":material/check:")
        else:
            st.error(text, icon=":material/error:")

    room = MAX_FILES - len(my_docs)
    if room <= 0:
        st.caption(f"You've added {MAX_FILES} files, the most for this demo. Remove one to add another.")
        return
    uploads = st.file_uploader(
        f"Add your own file: PDF, TXT or MD, up to {MAX_FILE_MB} MB and {MAX_PDF_PAGES} pages "
        f"({plural(room, 'more file')} allowed)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        max_upload_size=MAX_FILE_MB,
        key=f"session-upload-{st.session_state.upload_round}",
    )
    if not uploads:
        return
    notes = []
    with st.spinner("Reading and indexing your file..."):
        for upload in uploads[:room]:
            try:
                added = backend().add_session_file(owner, upload.name, upload.getvalue())
                notes.append(("ok", f"Added {display_title(added['title'], 60)} ({plural(added['pages'], 'page')})."))
            except SessionUploadError as exc:
                notes.append(("error", str(exc)))
            except Exception as exc:  # unexpected: keep the page working
                notes.append(("error", f"{upload.name} couldn't be added: {exc}"))
    if len(uploads) > room:
        skipped = ", ".join(u.name for u in uploads[room:])
        notes.append(("error", f"Only {plural(room, 'more file')} can be added, so {skipped} was skipped."))
    st.session_state.upload_notes = notes
    st.session_state.upload_round += 1  # a fresh uploader: clears the widget and frees the uploaded bytes
    st.rerun()


def how_it_works(mode: str, max_rounds: int) -> str:
    retry = (
        f"If something is missing, it rewrites the searches and looks again (up to {plural(max_rounds, 'round')})."
        if max_rounds > 1
        else "Retrying is switched off, so it answers after one search round."
    )
    modes = "\n".join(f"- **{label}**: {MODE_HELP[m]}" for m, label in MODE_LABELS.items())
    return f"""
1. **Plan**: splits your question into one to three focused searches.
2. **Retrieve**: finds the most relevant passages in the documents.
3. **Grade**: checks whether those passages are enough to answer.
4. **Retry or answer**: {retry} Then it answers, citing the passages it used, or says the documents don't cover it.

**Search methods** (choose one under Settings in the sidebar):

{modes}

Currently selected: **{MODE_LABELS.get(mode, mode)}**. How well each method works is measured in the
[evaluation]({EVAL_URL}).
"""


# --------------------------------------------------------------------------- startup
try:
    info = backend().info()
    # In API mode the API service handles AUTO_INGEST_SAMPLES itself.
    auto_ingest = os.environ.get("AUTO_INGEST_SAMPLES", "").lower() in {"1", "true", "yes"}
    if backend().name == "local" and auto_ingest and not auto_ingest_samples():
        auto_ingest_samples.clear()  # a download failed: try the missing papers again on the next load
    sources = backend().sources()
except Exception as exc:
    st.error(f"Could not reach the document index: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0  # not reset by "Clear chat", so the demo limit holds
if "owner" not in st.session_state:
    st.session_state.owner = uuid.uuid4().hex  # this browser session's id for its private uploads
    st.session_state.upload_round = 0

# Optional per-session cap for a public demo (DEMO_MAX_QUESTIONS); None means unlimited.
settings = get_settings()
question_limit = settings.demo_max_questions
questions_left = None if question_limit is None else max(question_limit - st.session_state.questions_asked, 0)
limit_reached = questions_left == 0
can_edit = bool(info.get("ingest_enabled"))
# Private per-visitor uploads: only when the shared corpus is read-only, and only in-process (not via the API).
owner = st.session_state.owner
session_uploads = (
    settings.allow_session_uploads and not can_edit and getattr(backend(), "supports_session_uploads", False)
)
my_docs: list[dict] = []
if session_uploads:
    backend().touch_session(owner)  # keeps this visitor's uploads alive; removes other idle sessions' uploads
    my_docs = backend().session_documents(owner)

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"{ICON} **ResearchPilot**")  # small: the page heading carries the name
    st.caption("Agentic RAG with verifiable citations")

    st.markdown("**Settings**")
    modes = list(MODE_LABELS)
    default_mode = info.get("retrieval_mode", "hybrid")
    mode = st.selectbox(
        "Search method",
        modes,
        format_func=lambda m: MODE_LABELS[m],
        index=modes.index(default_mode) if default_mode in modes else 0,
        help="\n\n".join(f"**{MODE_LABELS[m]}**: {MODE_HELP[m]}." for m in modes),
    )
    show_steps = st.toggle("Show agent steps", value=True, help="Show how the agent searched for each answer.")
    if st.session_state.messages and st.button("Clear chat", icon=":material/delete_sweep:"):
        st.session_state.messages = []
        st.rerun()

    if question_limit is not None:
        st.caption(f":material/forum: Demo limit: {questions_left} of {question_limit} left in this session")

    st.divider()
    st.markdown(f"**Documents ({len(sources)})**")
    if session_uploads:
        st.caption("Shared with every visitor and read-only.")
    for s in sources:
        title = display_title(s["title"])
        link = f"[{title}]({s['source']})" if s["source"].startswith(("http://", "https://")) else title
        if can_edit:
            left, right = st.columns([6, 1], vertical_alignment="center")
            left.markdown(f":material/description: {link}")
            if right.button("", icon=":material/close:", key=f"del-{s['source']}", help=f"Remove {s['source']}"):
                backend().delete(s["source"])
                st.rerun()
        else:
            st.markdown(f":material/description: {link}")

    if can_edit:
        st.markdown("**Add documents**")
        uploads = st.file_uploader(
            f"PDF, TXT or Markdown (max {info.get('max_upload_mb', 20)} MB each)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if uploads and st.button("Add uploaded files"):
            with st.status(f"Adding {plural(len(uploads), 'file')}...", expanded=True) as status:
                show_report(status, backend().ingest_files([(u.name, u.getvalue()) for u in uploads]))
            st.rerun()
        url = st.text_input("Or a URL (web page or PDF)", placeholder="https://arxiv.org/pdf/2005.11401")
        if url and st.button("Add URL"):
            ingest_urls([url.strip()])
            st.rerun()
        if not sources and SAMPLE_FILE.exists() and st.button("Load 5 sample AI papers", type="primary"):
            ingest_urls(read_sources_file(str(SAMPLE_FILE)))
            st.rerun()
    elif session_uploads:
        session_uploads_panel(owner, my_docs)
    else:
        st.info("This demo uses a fixed set of documents.", icon=":material/lock:")

    st.divider()
    st.caption(f"Built by Kartik Kaushik · [GitHub]({GITHUB_URL})")
    st.caption(f"{model_label(info['llm_provider'], info['llm_model'])} · v{__version__}")

# --------------------------------------------------------------------------- chat
st.title("ResearchPilot", anchor=False)
st.markdown(SUBTITLE)
with st.expander("How it works", icon=":material/help_outline:"):
    st.markdown(how_it_works(mode, settings.max_retrieval_rounds))

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=USER_AVATAR if message["role"] == "user" else ICON):
        if message["role"] == "user":
            st.markdown(message["content"])
        elif "notice" in message:  # the question couldn't be answered (e.g. the LLM quota is used up)
            st.warning(message["notice"], icon=":material/hourglass_top:")
        else:
            render_result(message["result"], show_steps)

# Before the first question the input sits under the intro, so the page opens at the top; a pinned
# input makes Streamlit scroll to the bottom, which hides the heading on a phone. Once the chat has
# started, the input is pinned to the bottom as usual.
landing = not st.session_state.messages
input_area = st.container() if landing else st
# The sample papers' topics in the read-only demo; a generic prompt once the visitor's own files are in play.
demo_corpus = not can_edit and not my_docs and only_sample_papers(sources)
has_documents = bool(sources or my_docs)
question = input_area.chat_input(
    DEMO_PLACEHOLDER if demo_corpus else GENERIC_PLACEHOLDER, disabled=not has_documents or limit_reached
)
if not has_documents:
    st.info("No documents yet. Add some from the sidebar to start.")
elif limit_reached:
    st.info(
        f"Thanks for trying ResearchPilot! This public demo allows {question_limit} "
        f"question{'' if question_limit == 1 else 's'} per session "
        f"to keep it free for everyone. To keep exploring, run it yourself from the [GitHub repository]({GITHUB_URL}).",
        icon=":material/favorite:",
    )
elif not st.session_state.messages and not question:
    examples = st.empty()  # cleared once a question is asked, so the buttons don't linger above the chat
    with examples.container():
        st.markdown("**Or try one of these:**")
        for row in range(0, len(EXAMPLES), 2):  # one row per pair, so phones stack them in order
            pairs = enumerate(EXAMPLES[row : row + 2], start=row)
            for col, (i, (kind, example)) in zip(st.columns(2), pairs, strict=False):  # odd count: one empty column
                if col.button(f"**{kind}:** {example}", key=f"ex-{i}", width="stretch", wrap=True):
                    question = example
    if question:
        examples.empty()

if question and not limit_reached:
    st.session_state.questions_asked += 1
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)
    with st.chat_message("assistant", avatar=ICON):
        result, notice = None, None
        progress = st.empty()  # live steps while the agent works; replaced by the answer when it's done
        try:
            with progress.container(), st.status("Researching...", expanded=True) as status:
                try:
                    for event in backend().ask_stream(question, mode=mode, owner=owner if session_uploads else None):
                        if event["type"] == "step":
                            status.write(f"{STEP_ICONS.get(event['step'], '•')} {describe_step(event)}")
                        elif event["type"] == "result":
                            result = event
                except LLMRateLimitError as exc:  # the provider's quota is used up: friendly message, not the error
                    notice = rate_limit_notice(exc.daily)
                    status.update(label="The LLM's free-tier limit was reached", state="error", expanded=False)
                else:
                    status.update(label="Done", state="complete", expanded=False)
        except LLMConfigError as exc:
            st.error(f"The LLM isn't configured: {exc}. Set LLM_PROVIDER, LLM_MODEL and the API key in `.env`.")
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
        if result:
            progress.empty()
            render_result(result, show_steps)
            st.session_state.messages.append({"role": "assistant", "result": result})
        else:
            st.session_state.questions_asked -= 1  # only answered questions count towards the demo limit
            if notice:
                st.warning(notice, icon=":material/hourglass_top:")
                st.session_state.messages.append({"role": "assistant", "notice": notice})
        if result and (question_limit is not None or landing):
            # Update the questions-left count and disable the input once the limit is hit; after the
            # first question, also move the input from under the intro to the bottom of the page.
            st.rerun()
