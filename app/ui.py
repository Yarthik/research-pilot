"""The page's components and look: one stylesheet (app/styles.css) and the Streamlit pieces that use it.

The stylesheet starts with the design tokens (spacing, type sizes, radii, colours); everything else
uses them. Colours that must work in both themes are derived from the current text colour with
color-mix(), so one rule gives a muted grey on white and on the dark background (muted text at 72%
of the text colour is at least 4.9:1 on every background, WCAG AA). The accent is the theme's
primary colour (.streamlit/config.toml), mixed with the text colour where it's small text.

Streamlit has no public class names for most elements, so the selectors use its data-testid
attributes and the st-key-* classes of keyed containers and widgets. Checked against Streamlit 1.65.
"""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st
from render import (
    best_snippet,
    citing_sentences,
    describe_step,
    doc_list_html,
    fix_small_caps,
    pdf_page_url,
    short_name,
    shorten_urls,
    source_anchor,
    source_button_label,
    streamlit_markdown,
)

STYLES = Path(__file__).with_name("styles.css")
CARD_SNIPPET_CHARS = 170  # about two lines in a half-width card; the stylesheet clamps it to two
STEP_ICONS = {
    "plan": ":material/checklist:",
    "retrieve": ":material/search:",
    "grade": ":material/fact_check:",
    "generate": ":material/edit_note:",
}


def inject_styles() -> None:
    # Read on every run (about 12 KB): Streamlit reloads edited Python modules but not this file.
    st.html(f"<style>{STYLES.read_text(encoding='utf-8')}</style>")


def section_label(text: str) -> None:
    """The one section-header style, used in the sidebar and above the sources."""
    st.markdown(f'<div class="rp-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def doc_list(items: list[tuple[str, str, str | None]], key: str) -> None:
    """One line per document: (short name, full title, URL); the full title shows on hover."""
    with st.container(key=key):
        st.markdown(doc_list_html(items), unsafe_allow_html=True)


# --------------------------------------------------------------------------- empty state
def suggestions(examples: list[tuple[str, str]]) -> str | None:
    """Four compact, left-aligned suggestion rows. Returns the clicked question."""
    clicked = None
    with st.container(key="rp-suggest"):
        for i, (kind, question) in enumerate(examples):
            if st.button(f":gray[{kind}]  \n{question}", key=f"ex-{i}", width="stretch"):
                clicked = question
    return clicked


@st.dialog("How it works", width="medium")
def how_it_works_dialog(text: str) -> None:
    st.markdown(text)


def composer_meta(questions_left: str | None, with_how_link: bool) -> bool:
    """The line under the composer: a "How it works" link and the questions left. True if the link was clicked."""
    clicked = False
    with st.container(key="rp-meta", horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        if with_how_link:
            clicked = st.button("How it works", type="tertiary", key="rp-how")
        if questions_left:
            st.caption(questions_left)
    return clicked


def file_chips(docs: list[dict]) -> str | None:
    """The visitor's attached files as small removable chips. Returns the source to remove, if clicked."""
    removed = None
    with st.container(key="rp-files", horizontal=True, gap="small"):
        for doc in docs:
            label = short_name(doc["title"])
            if st.button(
                label,
                key=f"rm-{doc['source']}",
                icon=":material/close:",
                icon_position="right",
                help=f"Remove {doc['title']} (private to this tab)",
            ):
                removed = doc["source"]
    return removed


# --------------------------------------------------------------------------- answers
def steps_status(trace: list[dict], summary: str, prefix: str) -> None:
    """A finished answer's agent steps: one collapsed line that expands to the steps."""
    with st.container(key=f"rp-steps-{prefix}"), st.status(summary, state="complete", expanded=False):
        for step in trace:
            st.markdown(f"{STEP_ICONS.get(step.get('step'), '•')} {describe_step(step)}")


def source_dialog(citation: dict) -> None:
    """The full passage of one source, titled with its number, with a link to the PDF page if there is one."""

    @st.dialog(f"Source {citation['n']}", width="medium")
    def show() -> None:
        page = f" · p. {citation['page']}" if citation.get("page") else ""
        st.markdown(
            f'<div class="rp-dialog-title">{html.escape(citation.get("title", ""))}'
            f'<span class="rp-meta">{page}</span></div>',
            unsafe_allow_html=True,
        )
        with st.container(key="rp-passage"):
            st.markdown(streamlit_markdown(shorten_urls(fix_small_caps(citation.get("text", "")), markdown=True)))
        source = citation.get("source") or ""
        pdf = pdf_page_url(source, citation.get("page"))
        if pdf:
            st.link_button(f"Open PDF at page {citation['page']}", pdf, icon=":material/open_in_new:")
        elif source.startswith(("http://", "https://")):
            st.link_button("Open source", source, icon=":material/open_in_new:")
        else:
            st.caption(f"From your file {source}")

    show()


def source_grid(citations: list[dict], answer: str, prefix: str) -> None:
    """Cited sources as a two-column grid of compact cards (one column on a phone), in citation order.

    Each card is a single button that opens the full passage. An empty anchor before it is the
    target of the answer's citation chips, which scroll to the card and highlight it. Cards show a
    short name (the full title is on hover and in the dialog).
    """
    if not citations:
        return
    citations = sorted(citations, key=lambda c: c["n"])  # [1], [2], [3], whatever order retrieval found them
    section_label("Sources")
    with st.container(key=f"rp-sources-{prefix}"):
        for row in range(0, len(citations), 2):
            pair = citations[row : row + 2]
            for col, citation in zip(st.columns(2, gap="small"), pair, strict=False):  # odd count: one empty
                number = citation["n"]
                snippet = best_snippet(
                    citation.get("text", ""), citing_sentences(answer, number), limit=CARD_SNIPPET_CHARS
                )
                with col.container(key=f"rp-src-{prefix}-{number}"):
                    st.markdown(
                        f'<span class="rp-anchor" id="{source_anchor(prefix, number)}"></span>', unsafe_allow_html=True
                    )
                    card = dict(citation, title=short_name(citation.get("title", "")))
                    label = source_button_label(card, shorten_urls(fix_small_caps(snippet)))
                    full_title = citation.get("title", "")
                    if st.button(label, key=f"src-{prefix}-{number}", width="stretch", wrap=True, help=full_title):
                        source_dialog(citation)
