"""Private, temporary documents for one browser session, searched together with the shared corpus.

The public demo (ALLOW_SESSION_UPLOADS) lets a visitor add a couple of files without changing what
anyone else sees.

Design: a session's uploads live in their own small in-memory index: the chunks, a matrix of
normalised embeddings, and a BM25 index over the shared chunks plus the session's own. The shared
ChromaDB collection is never written, so it stays read-only and nothing is left on disk after a
restart. A session's retriever holds only the shared corpus and that session's index, so isolation
comes from what it can reach, not from a filter that every query has to remember. BM25's statistics
(IDF, average length) cover only what that session can see, so one visitor's files never shift
another visitor's scores. A session without uploads gets the shared retriever itself, so retrieval
there is exactly as before.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..caching import build_once
from ..config import get_settings
from ..ingestion import quality
from ..ingestion.chunking import Chunk, chunk_document
from ..ingestion.loaders import Document, decode_text, load_pdf_bytes
from .bm25 import BM25Index
from .hybrid import HybridRetriever

log = logging.getLogger(__name__)

MAX_FILES = 2  # per session
MAX_FILE_MB = 5
MAX_PDF_PAGES = 50
MAX_CHARS = 400_000  # per file, about 50 dense pages: bounds the chunks (and memory) of a text-heavy file
MAX_SESSIONS = 10  # sessions holding uploads at the same time
IDLE_SECONDS = 3600  # uploads of a session idle for longer are removed
CLEANUP_INTERVAL = 300  # seconds between background clean-ups
SUFFIXES = (".pdf", ".txt", ".md")


class SessionUploadError(ValueError):
    """A file was rejected. The message is written for the visitor."""


class SessionsFull(SessionUploadError):
    """Too many sessions hold uploads right now."""


def unreadable_message(name: str) -> str:
    """Why a file whose text layer is mojibake was refused (see ingestion/quality.py)."""
    if name.lower().endswith(".pdf"):
        return f"{name}'s text can't be read: the PDF stores its text in a non-standard font encoding or as images."
    return f"{name}'s text can't be read: it isn't text in a standard encoding."


@dataclass
class PreparedFile:
    """A file that passed every check and is chunked, but not yet embedded (that is the slow part)."""

    name: str
    chunks: list[Chunk]
    info: dict  # {"source", "title", "chunks", "pages", "kept", "english"}


@dataclass
class SessionDocs:
    """One session's private documents."""

    last_used: float
    files: dict[str, dict] = field(default_factory=dict)  # source -> {"source", "title", "chunks", "pages"}
    chunks: list[Chunk] = field(default_factory=list)
    vectors: np.ndarray | None = None  # one normalised row per chunk
    _bm25: tuple[BM25Index, BM25Index] | None = None  # (shared index it was built from, shared + own)

    def search(self, embedding: list[float], k: int) -> list[tuple[Chunk, float]]:
        """Top-k own chunks by cosine similarity (the same measure as the shared ChromaDB search)."""
        if not self.chunks or self.vectors is None:
            return []
        query = np.asarray(embedding, dtype=np.float32)
        query /= np.linalg.norm(query) or 1.0
        similarities = self.vectors @ query
        top = np.argsort(-similarities, kind="stable")[:k]
        return [(self.chunks[i], float(similarities[i])) for i in top]

    def bm25(self, shared: BM25Index) -> BM25Index:
        """BM25 over the shared chunks plus this session's, rebuilt when either side changes."""
        cached = self._bm25
        if cached is None or cached[0] is not shared:
            cached = self._bm25 = (shared, BM25Index(shared.chunks + self.chunks))
        return cached[1]

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        rows = np.asarray(vectors, dtype=np.float32)
        rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
        self.chunks = self.chunks + chunks
        self.vectors = rows if self.vectors is None else np.vstack([self.vectors, rows])
        self._bm25 = None

    def drop(self, source: str) -> bool:
        if self.files.pop(source, None) is None:
            return False
        keep = [i for i, chunk in enumerate(self.chunks) if chunk.source != source]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if self.vectors is not None and keep else None
        self._bm25 = None
        return True


class SessionRetriever(HybridRetriever):
    """The shared corpus plus one session's private documents, with the same modes and fusion."""

    def __init__(self, shared: HybridRetriever, docs: SessionDocs) -> None:
        super().__init__(store=shared.store, embed_query=shared.embed_query, score=shared.score)
        self.shared = shared
        self.docs = docs

    @property
    def bm25(self) -> BM25Index:
        return self.docs.bm25(self.shared.bm25)

    def vector_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        embedding = self.embed_query(query)
        hits = self.store.search(embedding, k) + self.docs.search(embedding, k)
        return sorted(hits, key=lambda hit: hit[1], reverse=True)[:k]


class ScopedRetriever(HybridRetriever):
    """Only some of a session's own files: the ones a question is about ("What is this file about?").

    Same modes and fusion, over those files' chunks alone, so a word that also names something in the
    shared papers (a resume's "React" and the ReAct paper) can't pull those in.
    """

    def __init__(self, shared: HybridRetriever, docs: SessionDocs, sources: list[str]) -> None:
        super().__init__(store=shared.store, embed_query=shared.embed_query, score=shared.score)
        keep = [i for i, chunk in enumerate(docs.chunks) if chunk.source in sources]
        self.scope = sorted({docs.chunks[i].source for i in keep})
        self.view = SessionDocs(last_used=docs.last_used, chunks=[docs.chunks[i] for i in keep])
        self.view.vectors = docs.vectors[keep] if docs.vectors is not None and keep else None
        self._index = BM25Index(self.view.chunks)

    @property
    def bm25(self) -> BM25Index:
        return self._index

    def vector_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        return self.view.search(self.embed_query(query), k)

    def retrieve(self, query: str, mode: str | None = None, final_k: int | None = None) -> list[tuple[Chunk, float]]:
        """The best matches, plus each file's opening passage (title, abstract, summary) if it isn't among them:
        that is what a question like "What is this about?" needs, and a search for its words can miss it."""
        results = super().retrieve(query, mode=mode, final_k=final_k)
        found = {chunk.id for chunk, _ in results}
        for source in self.scope:
            first = min((c for c in self.view.chunks if c.source == source), key=lambda c: c.chunk_index)
            if first.id not in found:
                results.append((first, 0.0))
        return results


def _megabytes(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"


def _cut(doc: Document, max_chars: int) -> Document:
    """The document's first `max_chars` characters, ending on a word boundary."""
    pages, used = [], 0
    for number, text in doc.pages:
        if used + len(text) > max_chars:
            room = max_chars - used
            head = text[:room].rsplit(" ", 1)[0] if " " in text[:room] else text[:room]
            if head.strip():
                pages.append((number, head))
            break
        pages.append((number, text))
        used += len(text)
    return replace(doc, pages=pages)


def _load(name: str, data: bytes) -> tuple[Document, dict | None]:
    """Read an uploaded file, keeping only what fits the page and text limits.

    Returns the document and, if it was cut, what was kept: {"unit": "pages" | "characters", "kept", "total"}.
    A PDF is read only up to MAX_PDF_PAGES, so a long one costs no more than a short one.
    """
    kept = None
    if name.lower().endswith(".pdf"):
        try:
            doc = load_pdf_bytes(data, name, max_pages=MAX_PDF_PAGES)
        except Exception as exc:
            raise SessionUploadError(
                f"{name} couldn't be read as a PDF (it may be damaged or password-protected)."
            ) from exc
        total = doc.total_pages or len(doc.pages)
        too_long = doc.char_count > MAX_CHARS
        if too_long:
            doc = _cut(doc, MAX_CHARS)
        if too_long or total > MAX_PDF_PAGES:
            last = max((number or 0 for number, _ in doc.pages), default=0)
            kept = {"unit": "pages", "kept": last, "total": total}
    else:
        doc = Document(source=name, title=Path(name).stem, pages=[(None, decode_text(data))])
        if doc.char_count > MAX_CHARS:
            kept = {"unit": "characters", "kept": MAX_CHARS, "total": doc.char_count}
            doc = _cut(doc, MAX_CHARS)
    return doc, kept


class SessionRegistry:
    """All sessions' private documents in this process, with limits and idle clean-up."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        embed: Callable[[list[str]], list[list[float]]] | None = None,
        max_sessions: int = MAX_SESSIONS,
        idle_seconds: float = IDLE_SECONDS,
    ) -> None:
        self._clock = clock
        self._embed = embed
        self.max_sessions = max_sessions
        self.idle_seconds = idle_seconds
        self._sessions: dict[str, SessionDocs] = {}
        self._lock = threading.Lock()
        # One upload is embedded at a time, so concurrent uploads can't stack up ONNX Runtime's memory.
        self._embed_lock = threading.Lock()

    # sessions ------------------------------------------------------------------
    def active_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

    def cleanup(self) -> list[str]:
        """Remove the uploads of sessions idle for longer than `idle_seconds`. Returns their ids."""
        now = self._clock()
        with self._lock:
            expired = [owner for owner, docs in self._sessions.items() if now - docs.last_used > self.idle_seconds]
            for owner in expired:
                del self._sessions[owner]
        if expired:
            log.info("removed the uploads of %d idle session(s)", len(expired))
        return expired

    def touch(self, owner: str) -> None:
        """Record activity for `owner` (keeps its uploads alive) and clean up idle sessions."""
        self.cleanup()
        with self._lock:
            if owner in self._sessions:
                self._sessions[owner].last_used = self._clock()

    def documents(self, owner: str) -> list[dict]:
        with self._lock:
            docs = self._sessions.get(owner)
            return sorted(docs.files.values(), key=lambda f: f["title"].lower()) if docs else []

    def retriever(self, owner: str | None, shared: HybridRetriever, scope: list[str] | None = None) -> HybridRetriever:
        """The retriever for `owner`: the shared one if the session has no uploads.

        With `scope`, only those of the session's files; if none of them is still there (removed, or
        cleaned up after an idle hour), the session's normal retriever instead.
        """
        with self._lock:
            docs = self._sessions.get(owner) if owner else None
            if docs is None or not docs.chunks:
                return shared
            docs.last_used = self._clock()
            if scope and any(chunk.source in scope for chunk in docs.chunks):
                return ScopedRetriever(shared, docs, scope)
            return SessionRetriever(shared, docs)

    # files -----------------------------------------------------------------------
    def _check_room(self, owner: str, name: str, also_adding: list[str] = ()) -> None:
        docs = self._sessions.get(owner)
        if docs is None and len(self._sessions) >= self.max_sessions:
            raise SessionsFull(
                "Private uploads are at capacity right now, because other visitors are using them. "
                "Please try again later; you can still ask about the shared papers."
            )
        attached = set(docs.files) if docs is not None else set()
        if name not in attached and len(attached | set(also_adding)) >= MAX_FILES:
            raise SessionUploadError(f"You can add up to {MAX_FILES} files. Remove one to add another.")

    def prepare_files(self, owner: str, files: list[tuple[str, bytes]]) -> list[PreparedFile | SessionUploadError]:
        """Check, read and chunk files for `owner`, without embedding them: one result per file, in order.

        Fast, so a file that can't be added (too big, no text, unreadable text, one too many) is refused
        before a question that comes with it is sent. Files in the same batch count towards the limit.
        """
        results: list[PreparedFile | SessionUploadError] = []
        names: list[str] = []
        for name, data in files:
            try:
                prepared = self._prepare(owner, name, data, also_adding=names)
            except SessionUploadError as exc:
                results.append(exc)
                continue
            results.append(prepared)
            names.append(prepared.name)
        return results

    def _prepare(self, owner: str, name: str, data: bytes, also_adding: list[str]) -> PreparedFile:
        name = Path(name or "").name or "upload.txt"
        if not name.lower().endswith(SUFFIXES):
            raise SessionUploadError(f"{name}: only PDF, TXT and MD files can be added.")
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            raise SessionUploadError(f"{name} is {_megabytes(len(data))}; the limit is {MAX_FILE_MB} MB.")
        self.cleanup()
        with self._lock:
            self._check_room(owner, name, also_adding)

        doc, kept = _load(name, data)
        settings = get_settings()
        chunks = chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise SessionUploadError(f"{name} has no extractable text (scanned PDFs need OCR first).")
        check = quality.assess(doc.text)
        if not check.readable:
            log.info("refused %s: unreadable text layer %s", name, check.stats)
            raise SessionUploadError(unreadable_message(name))
        # Distinct ids, so a file named like a shared source can't be merged with it during fusion.
        chunks = [replace(chunk, id=f"session-{chunk.id}") for chunk in chunks]
        info = {
            "source": name,
            "title": doc.title,
            "chunks": len(chunks),
            "pages": len(doc.pages),
            "kept": kept,
            "english": check.english,
        }
        return PreparedFile(name=name, chunks=chunks, info=info)

    def index(self, owner: str, prepared: PreparedFile) -> dict:
        """Embed a prepared file into `owner`'s private documents (replacing a same-named one). The slow part."""
        embed = self._embed
        if embed is None:
            from . import embeddings

            embed = embeddings.embed_texts
        with self._embed_lock:
            vectors = embed([chunk.text for chunk in prepared.chunks])
        with self._lock:
            self._check_room(owner, prepared.name)  # again: another upload may have finished meanwhile
            docs = self._sessions.setdefault(owner, SessionDocs(last_used=self._clock()))
            docs.drop(prepared.name)
            docs.add(prepared.chunks, vectors)
            docs.files[prepared.name] = prepared.info
            docs.last_used = self._clock()
        return prepared.info

    def add_file(self, owner: str, name: str, data: bytes) -> dict:
        """Check, chunk and embed one file into `owner`'s private documents (replacing a same-named one)."""
        (result,) = self.prepare_files(owner, [(name, data)])
        if isinstance(result, SessionUploadError):
            raise result
        return self.index(owner, result)

    def remove(self, owner: str, source: str) -> bool:
        with self._lock:
            docs = self._sessions.get(owner)
            removed = bool(docs and docs.drop(source))
            if docs is not None and not docs.files:
                del self._sessions[owner]
            return removed

    # background clean-up -----------------------------------------------------------
    def run_cleanup_forever(self, interval: float = CLEANUP_INTERVAL) -> None:
        while True:
            time.sleep(interval)
            try:
                self.cleanup()
            except Exception:  # never let the clean-up thread die
                log.exception("session clean-up failed")


@build_once
def get_registry() -> SessionRegistry:
    """The process-wide registry; a daemon thread removes idle sessions even when nobody visits."""
    registry = SessionRegistry()
    threading.Thread(target=registry.run_cleanup_forever, name="session-uploads-cleanup", daemon=True).start()
    return registry
