"""The page's look: one stylesheet (app/styles.css) and the few Streamlit helpers that depend on it.

Colours that must work in both themes are derived from the current text colour with color-mix(),
so the same rule gives a muted grey on white and on the dark background. Muted text at 72% of the
text colour is at least 4.9:1 on every background (WCAG AA). The accent is the theme's primary
colour (.streamlit/config.toml); it's used for borders and fills, and mixed with the text colour
where it's small text, because pure #2F6FE4 is only 4.1:1 on the dark background.

Streamlit has no public class names for most elements, so the selectors use its data-testid
attributes and the st-key-* classes of keyed containers. Checked against Streamlit 1.65.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from render import best_snippet, citing_sentences, meter_html, shorten_urls, source_card_html, streamlit_markdown

STYLES = Path(__file__).with_name("styles.css")


def inject_styles() -> None:
    # Read on every run (about 10 KB): Streamlit reloads edited Python modules but not this file.
    st.html(f"<style>{STYLES.read_text(encoding='utf-8')}</style>")


def section_label(text: str, sidebar_style: bool = True) -> None:
    """A small heading: an uppercase label in the sidebar, a muted one in the chat column."""
    css_class = "rp-side-label" if sidebar_style else "rp-section-label"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def example_cards(examples: list[tuple[str, str]]) -> str | None:
    """Suggestion cards in a 2x2 grid (stacked in order on a phone). Returns the clicked question."""
    clicked = None
    with st.container(key="rp-examples"):
        for row in range(0, len(examples), 2):
            pairs = enumerate(examples[row : row + 2], start=row)
            for col, (i, (kind, question)) in zip(st.columns(2), pairs, strict=False):  # odd count: one empty
                if col.button(f":gray[{kind}]  \n{question}", key=f"ex-{i}", width="stretch", wrap=True):
                    clicked = question
    return clicked


def source_cards(citations: list[dict], answer: str, prefix: str) -> None:
    """One card per cited source: title · page, a snippet, a link, and the full passage on demand."""
    if not citations:
        return
    st.markdown('<div class="rp-sources-label">Sources</div>', unsafe_allow_html=True)
    for citation in citations:
        snippet = best_snippet(citation.get("text", ""), citing_sentences(answer, citation["n"]))
        with st.container(key=f"rp-src-{prefix}-{citation['n']}"):
            st.markdown(source_card_html(citation, snippet, prefix), unsafe_allow_html=True)
            with st.expander("Show full passage"):
                st.markdown(streamlit_markdown(shorten_urls(citation.get("text", ""), markdown=True)))


def question_meter(left: int, total: int) -> None:
    st.caption(f"{left} of {total} questions left")
    st.markdown(meter_html(left, total), unsafe_allow_html=True)
