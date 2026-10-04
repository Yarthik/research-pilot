"""Load raw text from local PDFs, text files, PDF URLs and web pages."""

from __future__ import annotations

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

    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((number, text))

    title = None
    meta_title = (reader.metadata.title or "").strip() if reader.metadata else ""
    if meta_title and not meta_title.lower().startswith(("microsoft word", "untitled")):
        title = meta_title
    elif pages:
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
