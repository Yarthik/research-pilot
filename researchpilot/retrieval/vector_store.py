"""ChromaDB wrapper storing chunks, embeddings and metadata on disk."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ..config import get_settings
from ..ingestion.chunking import Chunk

log = logging.getLogger(__name__)


def writable_dir(path: Path) -> Path:
    """`path` if the index can be stored there, else a folder under the system temp directory.

    Some hosts mount the app's folder read-only. A temp index is lost on restart, but the app keeps
    working (and re-ingests the sample papers when AUTO_INGEST_SAMPLES is set).
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.touch()
        probe.unlink()
        return path
    except OSError as exc:
        fallback = Path(tempfile.gettempdir()) / "researchpilot" / path.name
        log.warning("cannot write the index to %s (%s); using %s, which is lost on restart", path, exc, fallback)
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def summarize_sources(metadatas: list[dict]) -> list[dict]:
    """Group chunk metadata into one entry per source."""
    summary: dict[str, dict] = {}
    for meta in metadatas:
        source = meta.get("source", "")
        entry = summary.setdefault(source, {"source": source, "title": meta.get("title") or source, "chunks": 0})
        entry["chunks"] += 1
    return sorted(summary.values(), key=lambda e: e["title"].lower())


class VectorStore:
    def __init__(self, path: str | Path | None = None, collection_name: str | None = None) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        s = get_settings()
        path = writable_dir(Path(path) if path else Path(s.data_dir) / "chroma")
        self.client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            collection_name or s.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,  # we always pass our own embeddings
        )

    def count(self) -> int:
        return self.collection.count()

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        batch = 1000  # stay well under Chroma's max batch size
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            self.collection.upsert(
                ids=[c.id for c in part],
                embeddings=embeddings[start : start + batch],
                documents=[c.text for c in part],
                metadatas=[c.metadata() for c in part],
            )

    def search(self, query_embedding: list[float], k: int) -> list[tuple[Chunk, float]]:
        count = self.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            (Chunk.from_metadata(cid, doc, meta), 1.0 - dist)  # cosine distance -> similarity
            for cid, doc, meta, dist in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            )
        ]

    def all_chunks(self) -> list[Chunk]:
        result = self.collection.get(include=["documents", "metadatas"])
        return [
            Chunk.from_metadata(cid, doc, meta)
            for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"], strict=True)
        ]

    def delete_source(self, source: str) -> int:
        existing = self.collection.get(where={"source": source}, include=["metadatas"])
        ids = existing["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def sources(self) -> list[dict]:
        return summarize_sources(self.collection.get(include=["metadatas"])["metadatas"])
