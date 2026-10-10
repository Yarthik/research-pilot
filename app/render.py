"""Text clean-up before Streamlit renders it as Markdown (pure functions, unit-tested)."""

from __future__ import annotations

import re

_DISPLAY_MATH = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_CODE = re.compile(r"(```.*?```|`[^`\n]+`)", re.DOTALL)  # Markdown code: shown verbatim, left alone


def _inline_math_end(text: str, start: int) -> int | None:
    """Index of the "$" closing inline maths opened at `start`, or None if that "$" is a literal dollar.

    Pandoc's rule: the opening "$" must be followed by a non-space; the closing "$" is the next
    unescaped one, which must follow a non-space and not be followed by a digit. So "$2dr$," and
    "$d^2$" are maths, while "costs $5 and $10" is currency. Maths doesn't span a blank line.
    """
    if start + 1 >= len(text) or text[start + 1].isspace():
        return None
    i = start + 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text.startswith("\n\n", i):
            return None
        if text[i] == "$":
            after = text[i + 1 : i + 2]
            return None if text[i - 1].isspace() or after.isdigit() else i
        i += 1
    return None


def _escape_literal_dollars(text: str) -> str:
    """Escape every "$" that doesn't delimit $...$ or $$...$$ maths, so Streamlit can't pair it up."""
    out = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\\":
            out.append(text[i : i + 2])  # an escape, e.g. an already escaped \$
            i += 2
        elif char != "$":
            out.append(char)
            i += 1
        elif text.startswith("$$", i):
            end = text.find("$$", i + 2)
            if end == -1:
                out.append(r"\$\$")
                i += 2
            else:
                out.append(text[i : end + 2])
                i = end + 2
        elif (end := _inline_math_end(text, i)) is not None:
            out.append(text[i : end + 1])
            i = end + 1
        else:
            out.append(r"\$")
            i += 1
    return "".join(out)


def _prose_markdown(text: str) -> str:
    text = _escape_literal_dollars(text)
    text = _DISPLAY_MATH.sub(lambda m: f"$${m.group(1).strip()}$$", text)
    return _INLINE_MATH.sub(lambda m: f"${m.group(1).strip()}$", text)


def streamlit_markdown(text: str) -> str:
    """Make LLM output and PDF passages render correctly in Streamlit Markdown.

    Streamlit renders LaTeX only between $...$ and $$...$$, so models' \\( ... \\) and \\[ ... \\]
    delimiters are converted. It also treats any pair of dollar signs as maths, which would garble
    amounts like "$36,500 ... $50", so dollars that aren't maths delimiters (by Pandoc's rule) are
    escaped first. Code spans and fenced code blocks are left as they are.
    """
    parts = _CODE.split(text or "")
    return "".join(part if n % 2 else _prose_markdown(part) for n, part in enumerate(parts))


def short_title(title: str, limit: int = 80) -> str:
    """Shorten a title on a word boundary, with an ellipsis, instead of cutting a word in half."""
    title = " ".join((title or "").split())
    if len(title) <= limit:
        return title
    cut = title[: limit + 1].rsplit(" ", 1)[0] if " " in title[:limit] else title[:limit]
    return cut.rstrip(" ,;:-–—") + "…"


GITHUB_URL = "https://github.com/Yarthik/research-pilot"
PROVIDER_NAMES = {
    "groq": "Groq",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google_genai": "Google",
    "ollama": "Ollama",
}


def plural(count: int, word: str, many: str | None = None) -> str:
    """E.g. "1 search round", "2 search rounds"."""
    return f"{count} {word if count == 1 else many or word + 's'}"


def model_label(provider: str, model: str) -> str:
    """E.g. "openai/gpt-oss-120b" on groq -> "gpt-oss-120b via Groq"."""
    return f"{(model or '').rsplit('/', 1)[-1]} via {PROVIDER_NAMES.get(provider, provider)}"


def answer_footer(result: dict) -> str:
    rounds = result.get("round") or 0
    passages = result.get("evidence_count", 0)
    return (
        f"Answered in {result.get('seconds', 0)} s · {plural(rounds, 'search round')} · "
        f"{plural(passages, 'passage')} considered"
    )


def describe_step(event: dict) -> str:
    """One plain-English line for an agent step (a streamed step event or a trace entry)."""
    step = event.get("step")
    seconds = f" ({event.get('seconds', 0)} s)"
    if step == "plan":
        queries = event.get("queries", [])
        if event.get("error"):
            return f"Couldn't plan (the model call failed), so it searches for the question as asked{seconds}"
        web = " and a web search" if event.get("needs_web") else ""
        return f"Planned {plural(len(queries), 'search', 'searches')}{web}: {'; '.join(queries)}{seconds}"
    if step == "retrieve":
        web = f" and {plural(event['new_web_results'], 'web result')}" if event.get("new_web_results") else ""
        found = f"{plural(event.get('new_chunks', 0), 'new passage')}{web}"
        return f"Search round {event.get('round')}: found {found}; {event.get('evidence', 0)} in use{seconds}"
    if step == "grade":
        if event.get("skipped"):
            return "Last search round done, so it answers with what it found"
        if event.get("error"):
            return f"Couldn't check the passages (the model call failed), so it answers with what it found{seconds}"
        if event.get("sufficient"):
            return f"Checked the passages: enough to answer{seconds}"
        if event.get("new_queries"):
            return f"Missing: {event.get('missing')} → searching again: {'; '.join(event['new_queries'])}{seconds}"
        return f"Something is missing: {event.get('missing')}{seconds}"
    if step == "generate":
        cited = len(event.get("cited", [])) or "none"
        return f"Wrote the answer from {plural(event.get('sources', 0), 'passage')}, citing {cited}{seconds}"
    return str(event)


def rate_limit_notice(daily: bool) -> str:
    """What visitors see when the LLM provider's rate limit or quota stops a question (never the raw error)."""
    if daily:
        return (
            "The demo has reached today's free-tier limit. Please try again later, "
            f"or see the [GitHub repository]({GITHUB_URL}) to run it yourself."
        )
    return "The demo is very busy right now (the free tier's per-minute limit was hit). Please try again in a minute."
