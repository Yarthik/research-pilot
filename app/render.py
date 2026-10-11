"""Text clean-up and small HTML pieces for the Streamlit page (pure functions, unit-tested)."""

from __future__ import annotations

import html
import re

from researchpilot.agent.citations import parse_citations
from researchpilot.retrieval.bm25 import tokenize

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


# --------------------------------------------------------------------------- citation chips
# The marker forms the agent's parser accepts: [1], [2][3], [1, 4], [2-4]. Not a Markdown link "[1](url)".
_CITE_GROUP = re.compile(r"\[(\s*\d+(?:\s*[-–—]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–—]\s*\d+)?)*\s*)\](?!\()")
# Maths left by streamlit_markdown: unescaped $$...$$ or $...$ (literal dollars are already escaped).
_MATH_SPAN = re.compile(r"(\$\$.+?\$\$|(?<!\\)\$(?:\\.|[^$\\\n])+\$)", re.DOTALL)
_TAG_START = re.compile(r"<(?=[A-Za-z/!?])")


def source_anchor(prefix: str, number: int) -> str:
    """The HTML id of a source card, the target of its citation chips."""
    return f"{prefix}-{number}"


def cite_chips(text: str, valid: set[int], prefix: str, titles: dict[int, str] | None = None) -> str:
    """Turn citation markers into chips linking to their source cards; copying still gives "[1]".

    The brackets and commas stay in the HTML, visually hidden, so selecting and copying the answer
    yields the original plain text. Numbers that aren't valid sources stay as they were.
    """
    titles = titles or {}

    def hidden(text: str) -> str:
        return f'<span class="rp-cite-copy">{html.escape(text)}</span>'

    def chip(token: str, number: int) -> str:
        label = f"Source {number}: {titles[number]}" if titles.get(number) else f"Source {number}"
        return (
            f'<a class="rp-cite" href="#{source_anchor(prefix, number)}" title="{html.escape(label)}">'
            f"{html.escape(token.strip())}</a>"
        )

    def replace(match: re.Match) -> str:
        pieces = re.split(r"(\s*,\s*)", match.group(1))  # numbers or ranges, with the separators kept
        firsts = [int(re.search(r"\d+", piece).group()) for piece in pieces[::2]]
        if not any(number in valid for number in firsts):
            return match.group(0)
        out = ['<span class="rp-cites">', hidden("[")]
        for i, piece in enumerate(pieces):
            if i % 2:
                out.append(hidden(piece))
            else:
                number = firsts[i // 2]
                out.append(chip(piece, number) if number in valid else html.escape(piece))
        out += [hidden("]"), "</span>"]
        return "".join(out)

    return _CITE_GROUP.sub(replace, text)


def answer_html(answer: str, citations: list[dict], prefix: str) -> str:
    """The answer as Markdown with citation chips, safe to render with unsafe_allow_html=True.

    Maths and code are left alone. Anything in the model's text that could open an HTML tag is
    escaped, so only the chips added here are HTML (an answer can quote an uploaded file).
    """
    valid = {c["n"] for c in citations}
    titles = {c["n"]: c.get("title", "") for c in citations}
    out = []
    for n, part in enumerate(_CODE.split(streamlit_markdown(answer))):
        if n % 2:  # a code span or block: shown verbatim
            out.append(part)
            continue
        for m, segment in enumerate(_MATH_SPAN.split(part)):
            out.append(segment if m % 2 else cite_chips(_TAG_START.sub("&lt;", segment), valid, prefix, titles))
    return "".join(out)


# --------------------------------------------------------------------------- source cards
_URL = re.compile(r"https?://[^\s<>()\[\]{}\"'`]+")
_ARXIV = re.compile(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/([^?#\s]+?)(?:\.pdf)?/?$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]$])")
SNIPPET_CHARS = 250


def short_url(url: str, limit: int = 40) -> str:
    """'https://www.example.com/a/very/long/path' -> 'example.com/a/very/lo…'."""
    text = re.sub(r"^https?://(www\.)?", "", url).rstrip("/")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def shorten_urls(text: str, limit: int = 40, markdown: bool = False) -> str:
    """Shorten URLs in passage text so they don't dominate; with `markdown`, keep them as links."""

    def replace(match: re.Match) -> str:
        url, tail = match.group(0), ""
        while url and url[-1] in ".,;:!?":  # sentence punctuation isn't part of the URL
            url, tail = url[:-1], url[-1] + tail
        short = short_url(url, limit)
        if markdown:
            return f"[{_MARKDOWN_SPECIAL.sub(r'\\\1', short)}]({url}){tail}"
        return short + tail

    return _URL.sub(replace, text or "")


def citing_sentences(answer: str, number: int) -> str:
    """The answer's sentences that cite `number` (the whole answer if none do)."""
    sentences = [s for s in _SENTENCE_END.split(answer or "") if number in parse_citations(s)]
    return " ".join(sentences) or answer or ""


def best_snippet(passage: str, context: str, limit: int = SNIPPET_CHARS) -> str:
    """About `limit` characters of `passage`, starting at the sentence that best matches `context`.

    Falls back to the start of the passage when no sentence shares a word with the context.
    """
    text = " ".join((passage or "").split())
    if len(text) <= limit:
        return text
    wanted = set(tokenize(context or ""))
    best_start, best_score, offset = 0, 0, 0
    for sentence in _SENTENCE_END.split(text):
        start = text.find(sentence, offset)
        score = len(set(tokenize(sentence)) & wanted)
        if score > best_score:
            best_start, best_score = start, score
        offset = start + len(sentence)
    start = best_start
    if len(text) - start < limit:  # near the end: show a full window that still contains the sentence
        start = text.find(" ", len(text) - limit) + 1
    window = text[start : start + limit + 1]
    if start + limit < len(text):  # cut on a word boundary
        window = window.rsplit(" ", 1)[0].rstrip(" ,;:.-–—") + "…"
    return ("…" if start > 0 else "") + window


def pdf_page_url(source: str, page: int | None) -> str | None:
    """A link that opens an arXiv paper's PDF at `page` (#page=N), or None for other sources."""
    match = _ARXIV.match(source or "")
    if not match or not page:
        return None
    return f"https://arxiv.org/pdf/{match.group(1)}#page={page}"


def source_card_html(citation: dict, snippet: str, prefix: str) -> str:
    """Header, snippet and link of one source card (the full passage is an expander below it)."""
    number, page, source = citation["n"], citation.get("page"), citation.get("source") or ""
    meta = f'<span class="rp-src-meta"> · p. {page}</span>' if page else ""
    pdf = pdf_page_url(source, page)
    if pdf:
        footer = _external_link(pdf, f"Open PDF at page {page} ↗")
    elif source.startswith(("http://", "https://")):
        footer = _external_link(source, f"{short_url(source)} ↗")
    else:
        footer = f'<span class="rp-src-file">{html.escape(source)}</span>'
    return (
        f'<div class="rp-src" id="{source_anchor(prefix, number)}">'
        f'<div class="rp-src-head"><span class="rp-src-n">{number}</span>'
        f'<span class="rp-src-title">{html.escape(citation.get("title") or "")}{meta}</span></div>'
        f'<p class="rp-src-snippet">{html.escape(shorten_urls(snippet))}</p>{footer}</div>'
    )


def _external_link(url: str, text: str) -> str:
    return (
        f'<a class="rp-src-open" href="{html.escape(url)}" target="_blank" rel="noopener noreferrer">'
        f"{html.escape(text)}</a>"
    )


def meter_html(left: int, total: int) -> str:
    """A thin bar showing how many demo questions are left."""
    share = 0 if total <= 0 else round(100 * max(0, min(left, total)) / total)
    return (
        f'<div class="rp-meter" role="progressbar" aria-label="Questions left" aria-valuemin="0" '
        f'aria-valuemax="{total}" aria-valuenow="{left}"><span style="width: {share}%"></span></div>'
    )
