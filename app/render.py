"""Text clean-up before Streamlit renders it as Markdown (pure functions, unit-tested)."""

from __future__ import annotations

import re

# "$" + an amount + a space, punctuation or the end: currency, not the start of $...$ maths ($10^4$, $2$).
_CURRENCY = re.compile(r"(?<!\\)\$(?=\d[\d,.]*(?:[^\d,.$^_{}\\]|$))")
_DISPLAY_MATH = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


def streamlit_markdown(text: str) -> str:
    """Make LLM output and PDF passages render correctly in Streamlit Markdown.

    Streamlit renders LaTeX only between $...$ and $$...$$, so models' \\( ... \\) and \\[ ... \\]
    delimiters are converted. It also treats any pair of dollar signs as maths, which would garble
    amounts like "$36,500 ... $50", so dollar amounts are escaped first.
    """
    text = _CURRENCY.sub(r"\\$", text or "")
    text = _DISPLAY_MATH.sub(lambda m: f"$${m.group(1).strip()}$$", text)
    return _INLINE_MATH.sub(lambda m: f"${m.group(1).strip()}$", text)


def short_title(title: str, limit: int = 80) -> str:
    """Shorten a title on a word boundary, with an ellipsis, instead of cutting a word in half."""
    title = " ".join((title or "").split())
    if len(title) <= limit:
        return title
    cut = title[: limit + 1].rsplit(" ", 1)[0] if " " in title[:limit] else title[:limit]
    return cut.rstrip(" ,;:-–—") + "…"


GITHUB_URL = "https://github.com/Yarthik/research-pilot"


def rate_limit_notice(daily: bool) -> str:
    """What visitors see when the LLM provider's rate limit or quota stops a question (never the raw error)."""
    if daily:
        return (
            "The demo has reached today's free-tier limit. Please try again later, "
            f"or see the [GitHub repository]({GITHUB_URL}) to run it yourself."
        )
    return "The demo is very busy right now (the free tier's per-minute limit was hit). Please try again in a minute."
