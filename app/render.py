"""Text clean-up before Streamlit renders it as Markdown (pure functions, unit-tested)."""

from __future__ import annotations

import re

_CURRENCY = re.compile(r"(?<!\\)\$(?=\d)")
_DISPLAY_MATH = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


def streamlit_markdown(text: str) -> str:
    """Make LLM output and PDF passages render correctly in Streamlit Markdown.

    Streamlit renders LaTeX only between $...$ and $$...$$, so models' \\( ... \\) and \\[ ... \\]
    delimiters are converted. It also treats any pair of dollar signs as maths, which would garble
    amounts like "$36,500 ... $50", so a dollar sign followed by a digit is escaped first.
    """
    text = _CURRENCY.sub(r"\\$", text or "")
    text = _DISPLAY_MATH.sub(lambda m: f"$${m.group(1).strip()}$$", text)
    return _INLINE_MATH.sub(lambda m: f"${m.group(1).strip()}$", text)
