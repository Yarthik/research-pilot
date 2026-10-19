"""Load raw text from local PDFs, text files, PDF URLs and web pages."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

USER_AGENT = "ResearchPilot/1.0 (open-source research assistant)"
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


class LoaderError(ValueError):
    """Raised when a source can't be loaded or has no extractable text."""


@dataclass
class Document:
    source: str  # file name or URL
    title: str
    pages: list[tuple[int | None, str]] = field(default_factory=list)  # (page number, text)
    total_pages: int | None = None  # a PDF's page count, also when only the first pages were read

    @property
    def char_count(self) -> int:
        return sum(len(text) for _, text in self.pages)


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_pdf_text(text: str) -> str:
    """Normalise extracted PDF text for search and display.

    NFKC turns ligatures into plain letters ("ﬁne-tuning" -> "fine-tuning"), so BM25 matches them.
    Text in a font without a Unicode map comes out as control-character soup that uses a control
    character where the space should be: lines where those outnumber real spaces are dropped, and
    stray control characters elsewhere (unmapped bullets) become spaces.
    """
    lines = []
    for line in unicodedata.normalize("NFKC", text).splitlines():
        garbage = len(CONTROL_CHARS.findall(line))
        if garbage > line.count(" "):
            continue
        lines.append(CONTROL_CHARS.sub(" ", line) if garbage else line)
    return "\n".join(lines)


TITLE_MIN_RATIO = 1.3  # a title's font is clearly larger than the body font
SMALL_CAPS_RATIO = 0.75  # small capitals in a title are ~80% of its size


def _title_from_spans(spans: list[tuple[float, float, str]]) -> str | None:
    """Pick a PDF title from page-1 text spans (font size, baseline y, text) in content order.

    The title is the first run of text clearly larger than the body font. This skips copyright
    notices and venue headers above it, joins titles that wrap onto two lines, and rejoins
    small-caps titles, which PDFs store as alternating big/small fragments ("L" + " OW" -> "LOW").
    """
    if not spans:
        return None
    chars: Counter[float] = Counter()
    for size, _, text in spans:
        chars[size] += len(text)
    body = chars.most_common(1)[0][0]
    start = next((i for i, (size, _, _) in enumerate(spans) if size >= TITLE_MIN_RATIO * body), None)
    if start is None:
        return None

    title_size = spans[start][0]
    parts: list[str] = []
    sizes = set()
    prev = None
    for size, y, text in spans[start:]:
        if size < SMALL_CAPS_RATIO * title_size:
            break
        if prev:
            prev_size, prev_y, prev_text = prev
            if y != prev_y:
                parts.append("\n")
            elif size < prev_size and re.search(r"(^|[^A-Za-z])[A-Z]$", prev_text) and re.match(r" [A-Z]", text):
                text = text[1:]  # big initial + small-caps rest of the same word
        parts.append(text)
        sizes.add(size)
        prev = (size, y, text)

    title = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", "".join(parts))  # word hyphenated across lines
    title = " ".join(title.split())
    if len(sizes) > 1:  # small caps: a stray final letter split off by kerning ("REAC T" -> "REACT")
        title = re.sub(r"\b([A-Z]{2,}) ([B-HJ-NP-Z])\b", r"\1\2", title)
    return title[:200] or None


# Spellings that case alone can't recover from an ALL-CAPS title.
KNOWN_TERMS = {
    term.upper(): term
    for term in (
        "LoRA", "QLoRA", "ReAct", "BERT", "RoBERTa", "LLaMA", "GPT", "RAG", "LLMs", "AI", "QA", "BLEU",
        "ImageNet", "GitHub", "PyTorch", "TensorFlow", "arXiv",
    )
}  # fmt: skip
MINOR_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "nor", "of", "on", "or"}
MINOR_WORDS |= {"over", "per", "the", "to", "via", "vs", "with"}


def title_case_if_all_caps(title: str) -> str:
    """Turn an ALL-CAPS title ("LORA: LOW-RANK ADAPTATION OF ...") into title case.

    Known terms get their usual spelling ("LoRA"); words with digits or without vowels ("GPT-3",
    "NLP", "CNN") and Roman numerals stay upper-case, as they are likely acronyms. Titles that have
    any lower-case letter, or fewer than three words, are returned unchanged.
    """
    if title != title.upper() or len(re.findall(r"[^\W\d_]{2,}", title)) < 3:
        return title

    def fix(match: re.Match) -> str:
        word = match.group(0)
        starts_phrase = not title[: match.start()].strip() or title[: match.start()].rstrip()[-1] in ":?!.—–"
        if word in KNOWN_TERMS:
            return KNOWN_TERMS[word]
        if re.search(r"\d", word) or not re.search(r"[AEIOUY]", word) or re.fullmatch(r"[IVX]{2,}", word):
            return word
        if word.lower() in MINOR_WORDS and not starts_phrase:
            return word.lower()
        return word.capitalize()

    return re.sub(r"[^\W_]+(?:['’][^\W_]+)*", fix, title)


def _read_pdf(data: bytes, max_pages: int | None = None) -> tuple[list[tuple[int | None, str]], str | None, int]:
    """(pages with text, title, total page count). With `max_pages`, only the first pages are read."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(BytesIO(data))
    except PdfReadError as exc:
        raise LoaderError(f"not a readable PDF: {exc}") from exc
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise LoaderError("PDF is password-protected") from exc

    spans: list[tuple[float, float, str]] = []

    def collect_span(text, cm, tm, _font, size):
        # Horizontal text only: rotated arXiv side stamps and figure labels are not titles.
        if text.strip() and tm[1] == tm[2] == 0 and cm[1] == cm[2] == 0:
            spans.append((round(abs(size * tm[3] * cm[3]), 1), round(tm[5] * cm[3] + cm[5], 1), text))

    pages = []
    total = len(reader.pages)
    for number, page in enumerate(reader.pages, start=1):
        if max_pages is not None and number > max_pages:
            break
        text = clean_pdf_text(page.extract_text(visitor_text=collect_span if number == 1 else None) or "")
        if text.strip():
            pages.append((number, text))

    title = None
    meta_title = (reader.metadata.title or "").strip() if reader.metadata else ""
    if meta_title and not meta_title.lower().startswith(("microsoft word", "untitled")):
        title = meta_title
    else:
        title = _title_from_spans([(size, y, clean_pdf_text(text)) for size, y, text in spans])
    if not title and pages:
        first_line = next((line.strip() for line in pages[0][1].splitlines() if line.strip()), "")
        title = first_line[:120] or None
    return pages, title_case_if_all_caps(title) if title else None, total


def load_pdf_bytes(data: bytes, source: str, max_pages: int | None = None) -> Document:
    pages, title, total = _read_pdf(data, max_pages)
    return Document(source=source, title=title or Path(source).stem, pages=pages, total_pages=total)


def load_pdf_file(path: str | Path) -> Document:
    path = Path(path)
    return load_pdf_bytes(path.read_bytes(), path.name)


def load_text_file(path: str | Path) -> Document:
    path = Path(path)
    return Document(
        source=path.name, title=path.stem, pages=[(None, path.read_text(encoding="utf-8", errors="replace"))]
    )


def load_url(url: str, timeout: float = 60) -> Document:
    import httpx

    try:
        response = httpx.get(url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise LoaderError(f"could not download {url}: {exc}") from exc

    content_type = response.headers.get("content-type", "").lower()
    if "pdf" in content_type or response.content[:5] == b"%PDF-" or url.lower().endswith(".pdf"):
        pages, title, total = _read_pdf(response.content)
        return Document(source=url, title=title or url.rstrip("/").split("/")[-1], pages=pages, total_pages=total)

    import trafilatura

    html = response.text
    text = trafilatura.extract(html, include_comments=False, include_tables=True) or ""
    metadata = trafilatura.extract_metadata(html)
    title = metadata.title if metadata and metadata.title else url
    return Document(source=url, title=title, pages=[(None, text)])


def load_source(source: str) -> Document:
    """Dispatch on the kind of source: URL, PDF file or text/markdown file."""
    source = source.strip()
    if source.startswith(("http://", "https://")):
        return load_url(source)
    path = Path(source).expanduser()
    if not path.exists():
        raise LoaderError(f"file not found: {source}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_file(path)
    if suffix in TEXT_SUFFIXES:
        return load_text_file(path)
    raise LoaderError(f"unsupported file type '{suffix}' (use .pdf, .txt or .md)")
