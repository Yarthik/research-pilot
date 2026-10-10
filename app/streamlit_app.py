"""Streamlit chat UI for ResearchPilot.

    streamlit run app/streamlit_app.py

Runs the agent in-process by default. Set API_URL (e.g. http://api:8000) to use the FastAPI
service instead, which is how docker compose wires the two containers together.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "app"):  # importable without `pip install -e .`
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import streamlit as st  # noqa: E402
from backend import get_backend  # noqa: E402
from render import GITHUB_URL, rate_limit_notice, short_title, streamlit_markdown  # noqa: E402
from runtime_env import ensure_modern_sqlite, load_streamlit_secrets  # noqa: E402

load_streamlit_secrets(st.secrets)  # hosted apps keep settings in secrets; this also covers TOML booleans

from researchpilot import __version__  # noqa: E402
from researchpilot.cli import read_sources_file  # noqa: E402
from researchpilot.config import RETRIEVAL_MODES, get_settings  # noqa: E402
from researchpilot.llm import LLMConfigError, LLMRateLimitError  # noqa: E402

SAMPLE_FILE = ROOT / "data" / "sample_sources.txt"
EXAMPLES = [
    "How does LoRA reduce the number of trainable parameters?",
    "Compare ReAct and chain-of-thought prompting.",
    "What are RAG-Sequence and RAG-Token, and how do they differ?",
    "Why does the Transformer scale dot products by 1/sqrt(d_k)?",
]
MODE_LABELS = {
    "hybrid_rerank": "Hybrid + cross-encoder rerank (slower)",
    "hybrid": "Hybrid: BM25 + vector, RRF (default)",
    "vector": "Vector only",
    "bm25": "BM25 only",
}
STEP_ICONS = {"plan": "🧭", "retrieve": "🔎", "grade": "⚖️", "generate": "✍️"}

st.set_page_config(page_title="ResearchPilot", page_icon="🧭", layout="wide")

try:
    ensure_modern_sqlite()  # before anything imports chromadb
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()


@st.cache_resource
def backend():
    return get_backend()


# --------------------------------------------------------------------------- helpers
def describe_step(event: dict) -> str:
    step = event.get("step")
    seconds = f" ({event.get('seconds', 0)}s)"
    if step == "plan":
        queries = event.get("queries", [])
        web = " + web search" if event.get("needs_web") else ""
        return f"Planned {len(queries)} search(es){web}: {'; '.join(queries)}{seconds}"
    if step == "retrieve":
        web = f", {event['new_web_results']} web result(s)" if event.get("new_web_results") else ""
        found = f"{event.get('new_chunks', 0)} new chunk(s){web}"
        return f"Round {event.get('round')}: {found}, {event.get('evidence', 0)} in context{seconds}"
    if step == "grade":
        if event.get("skipped"):
            return f"Grader skipped (no rounds left){seconds}"
        if event.get("sufficient"):
            return f"Context is sufficient{seconds}"
        if event.get("new_queries"):
            return f"Missing: {event.get('missing')} → searching again: {'; '.join(event['new_queries'])}{seconds}"
        return f"Context incomplete: {event.get('missing')}{seconds}"
    if step == "generate":
        return f"Answered from {event.get('sources')} source(s), cited {len(event.get('cited', []))}{seconds}"
    return str(event)


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    st.markdown("**Sources**")
    for c in citations:
        page = f", p. {c['page']}" if c.get("page") else ""
        icon = "🌐" if c.get("origin") == "web" else "📄"
        with st.expander(f"[{c['n']}] {icon} {short_title(c['title'], 90)}{page}"):
            st.caption(c["source"])
            st.markdown(streamlit_markdown(c["text"]))


def render_result(result: dict, show_trace: bool) -> None:
    if result.get("insufficient"):
        st.warning("The documents don't fully answer this question.", icon="⚠️")
    st.markdown(streamlit_markdown(result["answer"]))
    render_citations(result.get("citations", []))
    st.caption(
        f"{result.get('seconds', 0)}s · {result.get('round') or 0} retrieval round(s) · "
        f"{result.get('evidence_count', 0)} chunks in context"
    )
    if show_trace and result.get("trace"):
        with st.expander("Agent trace"):
            st.json(result["trace"])


def show_report(status, report: dict) -> None:
    for item in report["ingested"]:
        status.write(f"✅ {short_title(item['title'])} ({item['chunks']} chunks)")
    for item in report["failed"]:
        status.write(f"❌ {item['source']}: {item['error']}")
    status.update(label=f"Ingested {len(report['ingested'])}, failed {len(report['failed'])}", state="complete")


def ingest_urls(urls: list[str]) -> None:
    with st.status(f"Ingesting {len(urls)} source(s)...", expanded=True) as status:
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

# Optional per-session cap for a public demo (DEMO_MAX_QUESTIONS); None means unlimited.
question_limit = get_settings().demo_max_questions
questions_left = None if question_limit is None else max(question_limit - st.session_state.questions_asked, 0)
limit_reached = questions_left == 0

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧭 ResearchPilot")
    st.caption(f"v{__version__} · LLM `{info['llm_provider']}/{info['llm_model']}` · {backend().name} mode")

    modes = list(RETRIEVAL_MODES[::-1])
    default_mode = info.get("retrieval_mode", "hybrid")
    mode = st.selectbox(
        "Retrieval mode",
        modes,
        format_func=lambda m: MODE_LABELS[m],
        index=modes.index(default_mode) if default_mode in modes else 0,
    )
    show_trace = st.toggle("Show agent trace", value=False)
    if question_limit is not None:
        st.caption(f"Demo limit: {questions_left} of {question_limit} left in this session")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader(f"Documents ({len(sources)})")
    st.caption(f"{sum(s['chunks'] for s in sources)} chunks indexed")
    can_edit = bool(info.get("ingest_enabled"))
    for s in sources:
        left, right = st.columns([6, 1])
        left.markdown(f"📄 **{short_title(s['title'])}**  \n{s['chunks']} chunks")
        if can_edit and right.button("✕", key=f"del-{s['source']}", help=f"Remove {s['source']}"):
            backend().delete(s["source"])
            st.rerun()

    if can_edit:
        st.divider()
        st.subheader("Add documents")
        uploads = st.file_uploader(
            f"PDF, TXT or Markdown (max {info.get('max_upload_mb', 20)} MB each)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if uploads and st.button("Ingest uploaded files"):
            with st.status(f"Ingesting {len(uploads)} file(s)...", expanded=True) as status:
                show_report(status, backend().ingest_files([(u.name, u.getvalue()) for u in uploads]))
            st.rerun()
        url = st.text_input("Or a URL (web page or PDF)", placeholder="https://arxiv.org/pdf/2005.11401")
        if url and st.button("Ingest URL"):
            ingest_urls([url.strip()])
            st.rerun()
        if not sources and SAMPLE_FILE.exists() and st.button("Load 5 sample AI papers", type="primary"):
            ingest_urls(read_sources_file(str(SAMPLE_FILE)))
            st.rerun()
    else:
        st.info("This demo uses a fixed set of documents.")

# --------------------------------------------------------------------------- chat
st.header("Ask your documents")
st.caption(
    "A LangGraph agent plans searches, retrieves with hybrid BM25 + vector search (optional cross-encoder reranking), "
    "checks whether the context is enough (searching again if not), and answers with citations."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        elif "notice" in message:  # the question couldn't be answered (e.g. the LLM quota is used up)
            st.warning(message["notice"], icon="⏳")
        else:
            render_result(message["result"], show_trace)

question = st.chat_input("Ask a question about your documents", disabled=not sources or limit_reached)
if not sources:
    st.info("No documents yet. Add some from the sidebar to start.")
elif limit_reached:
    st.info(
        f"Thanks for trying ResearchPilot! This public demo allows {question_limit} "
        f"question{'' if question_limit == 1 else 's'} per session "
        f"to keep it free for everyone. To keep exploring, run it yourself from the [GitHub repository]({GITHUB_URL}).",
        icon="🙏",
    )
elif not st.session_state.messages and not question:
    examples = st.empty()  # cleared once a question is asked, so the buttons don't linger above the chat
    with examples.container():
        st.markdown("**Try one of these:**")
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLES):
            if cols[i % 2].button(example, key=f"ex-{i}"):
                question = example
    if question:
        examples.empty()

if question and not limit_reached:
    st.session_state.questions_asked += 1
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        result, notice = None, None
        try:
            with st.status("Researching...", expanded=True) as status:
                try:
                    for event in backend().ask_stream(question, mode=mode):
                        if event["type"] == "step":
                            status.write(f"{STEP_ICONS.get(event['step'], '•')} {describe_step(event)}")
                        elif event["type"] == "result":
                            result = event
                except LLMRateLimitError as exc:  # the provider's quota is used up: friendly message, not the error
                    notice = rate_limit_notice(exc.daily)
                    status.update(label="The LLM's free-tier limit was reached", state="error", expanded=False)
                else:
                    label = f"Done in {result['seconds']}s" if result else "Finished"
                    status.update(label=label, state="complete", expanded=False)
        except LLMConfigError as exc:
            st.error(f"The LLM isn't configured: {exc}. Set LLM_PROVIDER, LLM_MODEL and the API key in `.env`.")
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
        if result:
            render_result(result, show_trace)
            st.session_state.messages.append({"role": "assistant", "result": result})
        else:
            st.session_state.questions_asked -= 1  # only answered questions count towards the demo limit
            if notice:
                st.warning(notice, icon="⏳")
                st.session_state.messages.append({"role": "assistant", "notice": notice})
        if result and question_limit is not None:
            st.rerun()  # update the questions-left count, and disable the input once the limit is hit
