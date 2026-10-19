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


SHORT_NAME_CHARS = 30  # about one line in the sidebar and in a half-width source card


def short_name(title: str, limit: int = SHORT_NAME_CHARS) -> str:
    """A short display name: the part before the first colon ("LoRA: Low-Rank ..." -> "LoRA"),
    otherwise the title cut at a word boundary. Display only; the full title stays in the metadata."""
    title = " ".join((title or "").split())
    head, colon, _ = title.partition(":")
    if colon and head.strip() and len(head.strip()) <= limit:
        return head.strip()
    return short_title(title, limit)


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


def steps_summary(result: dict) -> str:
    """The collapsed agent-steps line, e.g. "Searched 1 round · 5 passages · 4.4 s"."""
    rounds = result.get("round") or 0
    passages = result.get("evidence_count", 0)
    seconds = f"{round(float(result.get('seconds') or 0), 1):g}"  # 11.01 -> "11", 4.42 -> "4.4"
    if result.get("clarification"):  # answered with the list of documents, without searching
        return f"No search needed · {seconds} s"
    return f"Searched {plural(rounds, 'round')} · {plural(passages, 'passage')} · {seconds} s"


def questions_left(left: int) -> str:
    """'4 questions left', '1 question left', 'No questions left'."""
    return "No questions left" if left <= 0 else f"{plural(left, 'question')} left"


def how_it_works(mode_labels: dict[str, str], mode_help: dict[str, str], mode: str, max_rounds: int) -> str:
    """The "How it works" dialog: the agent's loop, the search methods and the selected one."""
    retry = (
        f"If something is missing, it rewrites the searches and looks again (up to {plural(max_rounds, 'round')})."
        if max_rounds > 1
        else "Retrying is switched off, so it answers after one search round."
    )
    modes = "\n".join(f"- **{label}**: {mode_help[m]}" for m, label in mode_labels.items())
    return f"""
1. **Plan**: splits your question into one to three focused searches.
2. **Retrieve**: finds the most relevant passages in the documents.
3. **Grade**: checks whether those passages are enough to answer.
4. **Retry or answer**: {retry} When the passages are about something else entirely, it stops early: \
searching again can't help. Then it answers, citing the passages it used, or says the documents don't cover it.

**Follow-up questions**: it remembers the last few turns, so you can ask "What about its memory use?" or \
"Which paper was that?". Before planning, it rewrites a follow-up into a question that stands on its own (one \
extra model call, skipped for a first question). If a question doesn't say which document it means, it lists \
them and asks.

**Search methods** (choose one in the sidebar):

{modes}

Currently selected: **{mode_labels.get(mode, mode)}**. How well each method works is measured in the
[evaluation]({GITHUB_URL}#evaluation).
"""


def _names(titles: list[str]) -> str:
    """ "your file X" / "your files X and Y" (short names)."""
    names = [short_name(t) for t in titles]
    return f"your file {names[0]}" if len(names) == 1 else f"your files {', '.join(names[:-1])} and {names[-1]}"


def _that(titles: list[str]) -> str:
    return "that file" if len(titles) == 1 else "those files"


def kept_note(doc: dict) -> str | None:
    """For a file cut to the demo's limits: "Indexed the first 50 of 160 pages of X." (None otherwise)."""
    kept = doc.get("kept")
    if not kept:
        return None
    return f"Indexed the first {kept['kept']:,} of {kept['total']:,} {kept['unit']} of {short_name(doc['title'])}."


def file_chip_label(doc: dict) -> str:
    """An attached file's chip: its short name, and how much was indexed if it was cut."""
    kept = doc.get("kept")
    name = short_name(doc["title"])
    if not kept:
        return name
    if kept["unit"] == "pages":
        return f"{name} · {kept['kept']} of {kept['total']} pages"
    return f"{name} · first {kept['kept'] // 1000}k characters"


def describe_step(event: dict) -> str:
    """One plain-English line for an agent step (a streamed step event or a trace entry)."""
    step = event.get("step")
    seconds = f" ({event.get('seconds', 0)} s)"
    if step == "condense":
        files = event.get("files")
        if files:
            return f"Your question is about {_names(files)}, so it searches only {_that(files)}{seconds}"
        if event.get("error"):
            return f"Couldn't use the earlier messages (the model call failed), so it searches as asked{seconds}"
        if event.get("rewritten"):
            return f"Read with the earlier messages as: {event.get('question')}{seconds}"
        return f"Checked the earlier messages: the question stands on its own{seconds}"
    if step == "clarify":
        if event.get("kind") == "library":
            return "You asked about the documents as a whole, so it lists them instead of searching"
        return "Your question doesn't say which document it means, so it lists them instead of searching"
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
        if event.get("verdict") == "out_of_scope":
            return f"Checked the passages: the documents don't cover this, so it doesn't search again{seconds}"
        if event.get("new_queries"):
            return f"Missing: {event.get('missing')} → searching again: {'; '.join(event['new_queries'])}{seconds}"
        return f"Something is missing: {event.get('missing')}{seconds}"
    if step == "generate":
        cited = len(event.get("cited", [])) or "none"
        return f"Wrote the answer from {plural(event.get('sources', 0), 'passage')}, citing {cited}{seconds}"
    return str(event)


def next_activity(event: dict) -> str | None:
    """The live status's "now doing" line after a step finishes (None once the answer is written)."""
    step = event.get("step")
    if step == "condense":
        return "Planning the searches…"
    if step == "plan":
        return "Searching the documents…"
    if step == "retrieve":
        return "Checking the passages…"
    if step == "grade":
        return "Searching again…" if event.get("new_queries") else "Writing the answer…"
    return None


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
        # PDF text sometimes glues a URL to the word before it ("athttps://..."); keep them apart.
        lead = " " if match.start() and match.string[match.start() - 1].isalnum() else ""
        short = short_url(url, limit)
        if markdown:
            return f"{lead}[{_MARKDOWN_SPECIAL.sub(r'\\\1', short)}]({url}){tail}"
        return lead + short + tail

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


# --------------------------------------------------------------------------- display text
_SMALL_CAPS_SPLIT = re.compile(r"\b([A-Z]) ([A-Z]{2,})\b")
_VOWELS = set("AEIOUY")
# Consonant pairs that can begin an English word; "PP", "NT", "XP" can't.
# fmt: off
_ONSETS = {
    "BL", "BR", "CH", "CL", "CR", "DR", "DW", "FL", "FR", "GH", "GL", "GN", "GR", "KN", "KR", "PH", "PL", "PR",
    "PS", "PT", "QU", "RH", "SC", "SH", "SK", "SL", "SM", "SN", "SP", "SQ", "ST", "SW", "SZ", "TH", "TR", "TS",
    "TW", "WH", "WR",
}
# fmt: on


# A word with its last letter split off ("REAC T", "AC T"): kerning in small-caps titles and headings.
_STRAY_LAST_LETTER = re.compile(r"\b([A-Z]{2,}) ([A-Z])\b(?![.\d])")
_WORDS = re.compile(r"[A-Za-z]+")
_GLUED_SYMBOL = re.compile(r"[a-z]+[A-Z]")
# Short words that often follow a one-letter symbol ("BETWEEN W AND ∆W"): "W AND" isn't "WAND".
_FUNCTION_WORDS = {"AND", "ARE", "AS", "AT", "BE", "BY", "FOR", "FROM", "IN", "IS", "IT", "OF", "ON", "OR", "THE", "TO"}


def fix_small_caps(text: str, context: str = "") -> str:
    """Rejoin words that PDF extraction split at a small-caps boundary ("A PPLYING" -> "APPLYING").

    Display only (passages and snippets), so the index and the retrieval eval are unaffected.
    A capital other than A or I is never a word on its own, so "R ELATED" is rejoined; after A or I
    the rest is rejoined only if it can't start a word, which keeps "A NEW", "A STUDY" and "I AM".

    A stray last letter ("REAC T: S YNERGIZING") is rejoined only when the joined word also appears
    elsewhere in the text or in `context` (e.g. the title "ReAct: ..."), since "PLAN B", "ROUGE L",
    "QK T" and "BETWEEN W" are real separate words: nothing else marks those splits. The same check
    applies when a single capital is followed by a short function word ("W AND" stays, "F OR" -> "FOR").
    """
    text = text or ""
    # Not words that pypdf glued to a maths symbol ("betweenW", "forB"): they would vouch for "BETWEEN W".
    known = {word.lower() for word in _WORDS.findall(f"{text} {context}") if not _GLUED_SYMBOL.fullmatch(word)}

    def rejoin_last(match: re.Match) -> str:
        head, letter = match.groups()
        return head + letter if (head + letter).lower() in known else match.group(0)

    def replace(match: re.Match) -> str:
        head, tail = match.groups()
        if tail in _FUNCTION_WORDS:
            return head + tail if (head + tail).lower() in known else match.group(0)
        if head in "AI":
            pair = tail[:2]
            if pair[0] in _VOWELS or pair[1] in _VOWELS or pair in _ONSETS:
                return match.group(0)
        return head + tail

    # Stray last letters first, so "AC T TRAJECTORY" becomes "ACT TRAJECTORY", not "AC TTRAJECTORY".
    return _SMALL_CAPS_SPLIT.sub(replace, _STRAY_LAST_LETTER.sub(rejoin_last, text))


_MARKDOWN_LABEL = re.compile(r"([\\`*_~#|<>$:])")


def escape_markdown(text: str) -> str:
    """Plain text for a Streamlit Markdown label: no formatting, maths, links or colour directives."""
    text = " ".join((text or "").split()).replace("[", "(").replace("]", ")")
    return _MARKDOWN_LABEL.sub(r"\\\1", text)


def source_button_label(citation: dict, snippet: str) -> str:
    """A source card's label: number, title, page, then a snippet (laid out by app/styles.css).

    The parts are a grey span (number), bold (title), an optional grey span (page) and a final
    grey span (snippet), so the stylesheet can place each one without per-card rules.
    """
    page = f" :gray[p. {citation['page']}]" if citation.get("page") else ""
    return (
        f":gray[{citation['n']}] **{escape_markdown(citation.get('title', ''))}**{page}  \n"
        f":gray[{escape_markdown(snippet)}]"
    )


def doc_list_html(items: list[tuple[str, str, str | None]]) -> str:
    """One-line entries of (short name, full title, URL): the full title on hover, a ↗ if there's a URL."""
    rows = []
    for name, title, url in items:
        label = f'<span class="rp-doc-t">{html.escape(name)}</span>'
        if url:
            rows.append(
                f'<li><a href="{html.escape(url)}" title="{html.escape(title)}" target="_blank" '
                f'rel="noopener noreferrer">{label}<span class="rp-doc-x" aria-hidden="true">↗</span></a></li>'
            )
        else:
            rows.append(f'<li title="{html.escape(title)}">{label}</li>')
    return f'<ul class="rp-docs">{"".join(rows)}</ul>'
