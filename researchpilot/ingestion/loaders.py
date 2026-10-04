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


def _read_pdf(data: bytes) -> tuple[list[tuple[int | None, str]], str | None]:
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
    for number, page in enumerate(reader.pages, start=1):
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
    return pages, title


def load_pdf_bytes(data: bytes, source: str) -> Document:
    pages, title = _read_pdf(data)
    return Document(source=source, title=title or Path(source).stem, pages=pages)


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
        pages, title = _read_pdf(response.content)
        return Document(source=url, title=title or url.rstrip("/").split("/")[-1], pages=pages)

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
