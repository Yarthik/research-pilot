"""ChromaDB wrapper storing chunks, embeddings and metadata on disk."""

from __future__ import annotations

from pathlib import Path

from ..config import get_settings
from ..ingestion.chunking import Chunk


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
        path = Path(path) if path else Path(s.data_dir) / "chroma"
        path.mkdir(parents=True, exist_ok=True)
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
