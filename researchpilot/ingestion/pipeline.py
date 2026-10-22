"""Ingest sources: load -> chunk -> embed -> store (ChromaDB + BM25)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from ..config import get_settings
from ..retrieval import hybrid
from . import quality
from .chunking import chunk_document
from .loaders import Document, LoaderError, load_source

log = logging.getLogger(__name__)
ProgressCallback = Callable[[str, int, int], None]  # (source, index, total)


UNREADABLE = "its text can't be read: it is stored in a non-standard font encoding or as images"


def ingest_document(doc: Document, refresh: bool = True) -> dict:
    """Chunk, embed and store one already-loaded document. Re-ingesting a source replaces it.

    A document whose text is mojibake (a PDF in a legacy font) is refused; one in another language is
    stored, and `english` in the result is False (see ingestion/quality.py).
    """
    from ..retrieval import embeddings

    settings = get_settings()
    retriever = hybrid.get_retriever()
    chunks = chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise LoaderError("no extractable text (scanned PDFs need OCR first)")
    check = quality.assess(doc.text)
    if not check.readable:
        log.info("refused %s: unreadable text layer %s", doc.source, check.stats)
        raise LoaderError(UNREADABLE)
    if not check.english:
        log.warning("%s doesn't look like English; ResearchPilot is tuned for English", doc.source)
    retriever.store.delete_source(doc.source)
    retriever.store.add(chunks, embeddings.embed_texts([c.text for c in chunks]))
    if refresh:
        retriever.refresh()
    return {
        "source": doc.source,
        "title": doc.title,
        "chunks": len(chunks),
        "pages": len(doc.pages),
        "english": check.english,
    }


def ingest(sources: Iterable[str], on_progress: ProgressCallback | None = None) -> dict:
    """Ingest URLs / file paths. One failing source doesn't stop the others."""
    sources = [s.strip() for s in sources if s and s.strip()]
    report: dict = {"ingested": [], "failed": []}
    for index, source in enumerate(sources, start=1):
        if on_progress:
            on_progress(source, index, len(sources))
        try:
            report["ingested"].append(ingest_document(load_source(source), refresh=False))
        except Exception as exc:
            log.warning("failed to ingest %s: %s", source, exc)
            report["failed"].append({"source": source, "error": str(exc)})
    hybrid.get_retriever().refresh()  # rebuild the BM25 index once, after all sources
    return report


def delete_source(source: str) -> int:
    """Remove a source from the index. Returns the number of chunks deleted."""
    retriever = hybrid.get_retriever()
    deleted = retriever.store.delete_source(source)
    retriever.refresh()
    return deleted
