"""Streamlit chat UI for ResearchPilot.

    streamlit run app/streamlit_app.py

Runs the agent in-process by default. Set API_URL (e.g. http://api:8000) to use the FastAPI
service instead, which is how docker compose wires the two containers together.
Components and styling are in app/ui.py and app/styles.css; text helpers in app/render.py.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "app"):  # importable without `pip install -e .`
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit as st  # noqa: E402
import ui  # noqa: E402
from backend import get_backend  # noqa: E402
from render import (  # noqa: E402
    GITHUB_URL,
    answer_html,
    describe_step,
    how_it_works,
    kept_note,
    model_label,
    next_activity,
    plural,
    rate_limit_notice,
    short_name,
    short_title,
    steps_summary,
)
from render import questions_left as questions_left_text  # noqa: E402
from runtime_env import ensure_modern_sqlite, load_streamlit_secrets  # noqa: E402

load_streamlit_secrets(st.secrets)  # hosted apps keep settings in secrets; this also covers TOML booleans

from researchpilot.cli import read_sources_file  # noqa: E402
from researchpilot.config import get_settings  # noqa: E402
from researchpilot.ingestion.loaders import title_case_if_all_caps  # noqa: E402
from researchpilot.llm import LLMConfigError, LLMRateLimitError  # noqa: E402
from researchpilot.retrieval.session import MAX_FILE_MB, MAX_FILES, MAX_PDF_PAGES, SessionUploadError  # noqa: E402

SAMPLE_FILE = ROOT / "data" / "sample_sources.txt"
ICON = ":material/travel_explore:"
USER_AVATAR = ":material/person:"
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
SUBTITLE = (
    "Ask a question. The agent searches, checks its evidence, searches again if needed, and answers with citations."
)
PLACEHOLDER = "Ask a question…"
LIMIT_PLACEHOLDER = "Demo limit reached"
FAILURE_MESSAGE = "Something went wrong while answering, so this question wasn't counted. Please try again in a moment."
NOT_SENT = "Your question wasn't sent, so it wasn't counted; it's still in the message box."
HISTORY_MESSAGES = 6  # earlier messages sent with a question, for follow-ups (the agent caps them too)
log = logging.getLogger("researchpilot.app")

st.set_page_config(page_title="ResearchPilot", page_icon=ICON, layout="centered")
ui.inject_styles()

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


def doc_entry(title: str, url: str | None = None) -> tuple[str, str, str | None]:
    """(short name, full title, URL) for a sidebar list."""
    full = display_title(title, 200)
    return short_name(full), full, url


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def render_result(result: dict, prefix: str) -> None:
    """An answer: collapsed agent steps, the answer with citation chips, and the sources grid.

    `prefix` makes the source cards' anchors unique per message (citation chips link to them).
    """
    if result.get("trace"):
        ui.steps_status(result["trace"], steps_summary(result), prefix)
    if result.get("insufficient"):
        covered = "don't fully answer" if result.get("citations") else "don't cover"
        st.info(f"The documents {covered} this question.", icon=":material/info:")
    citations = [dict(c, title=display_title(c.get("title", ""))) for c in result.get("citations", [])]
    st.markdown(answer_html(result["answer"], citations, prefix), unsafe_allow_html=True)
    ui.source_grid(citations, result["answer"], prefix)


def how_it_works_text(mode: str) -> str:
    text = how_it_works(MODE_LABELS, MODE_HELP, mode, get_settings().max_retrieval_rounds)
    if session_uploads:
        text += (
            f"\n**Your own files**: attach up to {MAX_FILES} PDF, TXT or MD files ({MAX_FILE_MB} MB each) with "
            f"the + button in the message box; for a longer file, the first {MAX_PDF_PAGES} pages are searched. "
            'Ask "What is it about?" right after attaching one and only that file is searched. Only this '
            "browser tab can search them, and they're removed when you remove them, reload the page, or after "
            "an hour idle.\n"
        )
    return text


def attach_files(files) -> tuple[list[dict], list[str]]:
    """Add files from the composer to this session's private index. Returns (added files, error messages).

    A file over the page or text limit is added with only its first part, and a notice says so.
    """
    added, errors = [], []
    with st.spinner("Reading and indexing your file…"):
        for upload in files:
            try:
                added.append(backend().add_session_file(owner, upload.name, upload.getvalue()))
            except SessionUploadError as exc:  # too big, no text, unreadable, too many files, at capacity
                errors.append(str(exc))
            except Exception:  # unexpected: keep the page working; the details go to the log
                log.exception("adding %r failed", upload.name)
                errors.append(f"{upload.name} couldn't be added. Please try again, or try another file.")
    st.session_state.attach_notes = [note for note in map(kept_note, added) if note]
    return added, errors


def chat_history() -> list[dict]:
    """The last answered exchanges, for follow-up questions: {"role", "content"}, oldest first.

    An answer carries the titles it cited, so "that paper" can be resolved; a question carries the
    files attached with it. Questions that failed (rate limit, error) are left out.
    """
    history = []
    for asked, reply in zip(messages[0::2], messages[1::2], strict=False):
        if asked.get("role") != "user" or "result" not in reply:
            continue
        question = asked["content"]
        if asked.get("files"):
            question += f" (attached: {', '.join(asked['files'])})"
        result = reply["result"]
        titles = list(dict.fromkeys(display_title(c.get("title", "")) for c in result.get("citations", [])))
        answer = (f"(Sources: {'; '.join(titles)}) " if titles else "") + result.get("answer", "")
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return history[-HISTORY_MESSAGES:]


def clear_notices() -> None:
    """The attachment notices stay until the visitor's next action, which starts afresh."""
    for key in ("attach_notes", "upload_errors"):
        st.session_state.pop(key, None)


# --------------------------------------------------------------------------- shared library (ALLOW_INGEST=true)
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


def manage_library(sources: list[dict]) -> None:
    """Upload, add by URL and delete, for a deployment whose shared library is editable."""
    with st.expander("Manage library", icon=":material/tune:"):
        for s in sources:
            left, right = st.columns([6, 1], vertical_alignment="center")
            left.caption(display_title(s["title"], 60))
            if right.button("", icon=":material/close:", key=f"del-{s['source']}", help=f"Remove {s['source']}"):
                backend().delete(s["source"])
                st.rerun()
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


# --------------------------------------------------------------------------- first start
@st.cache_resource
def setup_lock() -> threading.Lock:
    return threading.Lock()  # one session indexes the sample papers; others wait for it


def missing_samples() -> list[str]:
    present = {s["source"] for s in backend().sources()}
    return [url for url in read_sources_file(str(SAMPLE_FILE)) if url not in present]


def sample_labels() -> dict[str, str]:
    """URL -> short name, from the comments in data/sample_sources.txt ("<url>   LoRA")."""
    labels = {}
    for line in SAMPLE_FILE.read_text(encoding="utf-8").splitlines():
        url, _, label = line.strip().partition(" ")
        if url.startswith("http") and label.strip():
            labels[url] = label.strip()
    return labels


def prepare_samples() -> None:
    """Index any sample papers missing from the index (AUTO_INGEST_SAMPLES), with visible progress.

    Runs on a fresh deployment or after a wake-up. A failed download is retried on the next page load.
    """
    if not SAMPLE_FILE.exists() or not missing_samples():
        return
    with st.container(key="rp-setup"):
        st.markdown("**Setting up the demo**")
        st.caption(
            "The app has just started, so it is downloading the five papers and indexing them for search. "
            "This happens once after a restart and takes about two minutes."
        )
        bar = st.progress(0.0, text="Waiting to start…")
        labels = sample_labels()
        with setup_lock():
            todo = missing_samples()  # another visitor may have finished it meanwhile
            report = {"failed": []}
            if todo:
                report = backend().ingest_urls(
                    todo,
                    on_progress=lambda url, i, n: bar.progress(
                        (i - 1) / n, text=f"Indexing {labels.get(url, url)} ({i} of {n})…"
                    ),
                )
        if report["failed"]:
            bar.progress(1.0, text="Some papers couldn't be downloaded.")
            st.warning(
                f"{plural(len(report['failed']), 'paper')} couldn't be downloaded; "
                "reload the page in a minute to try again.",
                icon=":material/sync_problem:",
            )
            return
    st.rerun()  # ready: draw the page without the setup panel


# --------------------------------------------------------------------------- startup
try:
    info = backend().info()
    # In API mode the API service handles AUTO_INGEST_SAMPLES itself.
    auto_ingest = os.environ.get("AUTO_INGEST_SAMPLES", "").lower() in {"1", "true", "yes"}
    if backend().name == "local" and auto_ingest:
        prepare_samples()
    sources = backend().sources()
except Exception as exc:
    st.error(f"Could not reach the document index: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0  # not reset by "New chat", so the demo limit holds
if "owner" not in st.session_state:
    st.session_state.owner = uuid.uuid4().hex  # this browser session's id for its private uploads
if "restore_text" in st.session_state:  # a question held back by a failed attachment: back in the box
    st.session_state.composer = st.session_state.pop("restore_text")

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
messages = st.session_state.messages
landing = not messages

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    with st.container(key="rp-brand"):
        st.markdown(f"{ICON} **ResearchPilot**")
    # Always present (disabled when empty), so nothing shifts when a chat starts.
    if st.button("New chat", icon=":material/edit_square:", key="rp-new-chat", width="stretch", disabled=landing):
        st.session_state.messages = []
        st.rerun()

    mode_keys = list(MODE_LABELS)
    default_mode = info.get("retrieval_mode", "hybrid")
    mode = st.selectbox(
        "Search method",
        mode_keys,
        format_func=lambda m: MODE_LABELS[m],
        index=mode_keys.index(default_mode) if default_mode in mode_keys else 0,
        help="\n\n".join(f"**{MODE_LABELS[m]}**: {MODE_HELP[m]}." for m in mode_keys),
    )

    ui.section_label("Library")
    library = [doc_entry(s["title"], s["source"] if is_url(s["source"]) else None) for s in sources]
    ui.doc_list(library, key="rp-library")
    if can_edit:
        manage_library(sources)
    elif sources:
        st.caption("Shared with every visitor · read-only")

    if my_docs:
        ui.section_label("Your documents")
        ui.doc_list([doc_entry(d["title"]) for d in my_docs], key="rp-mydocs")
        st.caption(":material/lock: Private to this tab · removed after an hour idle")

    st.caption(
        f"Built by Kartik Kaushik · [GitHub]({GITHUB_URL})  \n{model_label(info['llm_provider'], info['llm_model'])}"
    )

# --------------------------------------------------------------------------- chat
has_documents = bool(sources or my_docs)
left_text = None if questions_left is None else questions_left_text(questions_left)
# The composer accepts files up to the server's upload cap, not MAX_FILE_MB: a file it refuses in the browser
# never reaches the page, so the question would go without it. The registry then rejects anything over
# MAX_FILE_MB, before reading it, and the question is held back with a clear message.
composer_options = (
    {
        "accept_file": "multiple",
        "file_type": ["pdf", "txt", "md"],
        "max_upload_size": max(MAX_FILE_MB, int(st.get_option("server.maxUploadSize"))),
    }
    if session_uploads
    else {}
)


def composer(area):
    """The chat input; with private uploads on, its attach button adds files to this session's index."""
    return area.chat_input(
        LIMIT_PLACEHOLDER if limit_reached else PLACEHOLDER,
        key="composer",
        disabled=not has_documents or limit_reached,
        **composer_options,
    )


def status_notices() -> None:
    if not has_documents:
        st.info("No documents yet. Add some from the sidebar to start.")
    elif limit_reached:
        st.info(
            f"Thanks for trying ResearchPilot! This public demo allows {question_limit} "
            f"question{'' if question_limit == 1 else 's'} per session "
            "to keep it free for everyone. To keep exploring, run it yourself from the "
            f"[GitHub repository]({GITHUB_URL}).",
            icon=":material/favorite:",
        )


clicked_example = None
if landing:
    # One vertically centred group; the stylesheet centres it while .st-key-rp-empty is on the page.
    with st.container(key="rp-empty"):
        with st.container(key="rp-hero"):
            st.markdown(ICON)
            st.title("ResearchPilot", anchor=False)
            st.markdown(SUBTITLE)
        # The same order as the pinned composer: attached-file chips and their notices, composer, meta line.
        removed = ui.file_chips(my_docs) if my_docs else None
        ui.composer_notices()
        submitted = composer(st.container(key="rp-composer"))
        how_clicked = ui.composer_meta(left_text, with_how_link=True)
        status_notices()
        if has_documents and not limit_reached:
            clicked_example = ui.suggestions(EXAMPLES)
else:
    for index, message in enumerate(messages):
        with st.chat_message(message["role"], avatar=USER_AVATAR if message["role"] == "user" else ICON):
            if message["role"] == "user":
                st.markdown(message["content"])
                if message.get("files"):
                    st.caption(f":material/attach_file: {', '.join(message['files'])}")
            elif "notice" in message:  # the question couldn't be answered (e.g. the LLM quota is used up)
                st.warning(message["notice"], icon=":material/hourglass_top:")
            elif "error" in message:  # it failed for another reason (details in the log)
                st.error(message["error"], icon=":material/error:")
            else:
                render_result(message["result"], prefix=f"m{index}")
    with st.bottom:
        removed = ui.file_chips(my_docs) if my_docs else None
        ui.composer_notices()
    submitted = composer(st)  # pinned to the bottom
    with st.bottom:
        how_clicked = ui.composer_meta(left_text, with_how_link=False)
    if "pending" not in st.session_state:  # not above an answer that's still being written
        status_notices()

if how_clicked:
    ui.how_it_works_dialog(how_it_works_text(mode))
if removed:
    clear_notices()
    backend().remove_session_file(owner, removed)
    st.rerun()

# What was submitted: text from the composer or a suggestion, and any attached files.
if isinstance(submitted, str):
    question, files = submitted.strip() or None, []
elif submitted:
    question, files = (submitted.text or "").strip() or None, list(submitted.files or [])
else:
    question, files = None, []
question = question or clicked_example
if question or files:
    clear_notices()
if files:
    added, errors = attach_files(files)
    # Files attached since the last question are "new": a question that refers to them searches only them.
    fresh = [s for s in st.session_state.get("fresh", []) if s not in {d["source"] for d in added}]
    st.session_state.fresh = fresh + [d["source"] for d in added]
    if errors:  # don't answer without the file the visitor meant to ask about
        st.session_state.upload_errors = ["  \n".join(errors) + (f"  \n{NOT_SENT}" if question else "")]
        if question:
            st.session_state.restore_text = question
        st.rerun()
    if not question:
        st.rerun()  # show the new chips
if question and not limit_reached:
    # Show the question and a live status at once in the chat layout, then answer on that run.
    fresh = set(st.session_state.pop("fresh", []))
    current = backend().session_documents(owner) if session_uploads else []
    attachments = [{"source": d["source"], "title": d["title"]} for d in current if d["source"] in fresh]
    st.session_state.pending = {"question": question, "history": chat_history(), "attachments": attachments}
    st.session_state.questions_asked += 1
    user_message = {"role": "user", "content": question}
    if attachments:
        user_message["files"] = [short_name(a["title"]) for a in attachments]
    messages.append(user_message)
    st.rerun()

pending = st.session_state.pop("pending", None)
if isinstance(pending, str):  # queued by an older version of this page (a session kept across a reload)
    pending = {"question": pending}
if pending:
    with st.chat_message("assistant", avatar=ICON):
        result, notice, failure = None, None, None
        progress = st.empty()  # the live steps; replaced by the finished answer (with its collapsed steps)
        with (
            progress.container(),
            st.container(key="rp-steps-live"),
            st.status("Searching the documents…", expanded=True) as status,
        ):
            done = st.container()  # finished steps, in order
            doing = st.empty()  # what the agent is doing now; replaced when each step finishes
            first = "Reading it with the earlier messages…" if pending.get("history") else "Planning the searches…"
            doing.markdown(f":material/hourglass_top: {first}")  # shown at once
            try:
                for event in backend().ask_stream(
                    pending["question"],
                    mode=mode,
                    owner=owner if session_uploads else None,
                    history=pending.get("history"),
                    attachments=pending.get("attachments"),
                ):
                    if event["type"] == "step":
                        done.markdown(f"{ui.STEP_ICONS.get(event['step'], '•')} {describe_step(event)}")
                        activity = next_activity(event)
                        if activity:
                            doing.markdown(f":material/hourglass_top: {activity}")
                        else:
                            doing.empty()
                    elif event["type"] == "result":
                        result = event
            except LLMRateLimitError as exc:  # the provider's quota is used up: friendly message, not the error
                notice = rate_limit_notice(exc.daily)
            except LLMConfigError as exc:  # for whoever runs the app, so it names the settings
                failure = f"The LLM isn't configured: {exc}. Set LLM_PROVIDER, LLM_MODEL and the API key in `.env`."
            except Exception:  # anything else: a friendly message; the details go to the log, not the page
                log.exception("answering %r failed", pending["question"])
                failure = FAILURE_MESSAGE
            doing.empty()
            if result:
                status.update(label="Done", state="complete", expanded=False)
            else:
                stopped = "The LLM's free-tier limit was reached" if notice else "Stopped"
                status.update(label=stopped, state="error", expanded=False)
        if result:
            progress.empty()
            render_result(result, prefix=f"m{len(messages)}")
            messages.append({"role": "assistant", "result": result})
            if result.get("clarification"):  # it only listed the documents (no model call): not counted
                st.session_state.questions_asked -= 1
            st.rerun()  # update the questions-left line, and disable the composer once the limit is hit
        st.session_state.questions_asked -= 1  # only answered questions count towards the demo limit
        # Its files are still new: asked again, "What is it about?" still means them.
        st.session_state.fresh = [a["source"] for a in pending.get("attachments") or []]
        if notice:
            messages.append({"role": "assistant", "notice": notice})
        else:
            messages.append({"role": "assistant", "error": failure or FAILURE_MESSAGE})
        st.rerun()  # show it from the history, with the refunded question in the count
