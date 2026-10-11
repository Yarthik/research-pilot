"""Hybrid retrieval: vector + BM25, fused with Reciprocal Rank Fusion, optionally reranked."""

from __future__ import annotations

import threading
from collections.abc import Callable

from ..caching import build_once
from ..config import RETRIEVAL_MODES, get_settings
from ..ingestion.chunking import Chunk
from .bm25 import BM25Index

MODES = RETRIEVAL_MODES
EmbedFn = Callable[[str], list[float]]
ScoreFn = Callable[[str, list[str]], list[float]]


def reciprocal_rank_fusion(result_lists: list[list[Chunk]], k: int = 60) -> list[tuple[Chunk, float]]:
    """Merge ranked lists: each chunk scores sum(1 / (k + rank)) over the lists it appears in.

    RRF only uses ranks, so it can combine BM25 scores and cosine similarities, which live on
    different scales. k=60 is the constant from the original paper (Cormack et al., 2009).
    """
    scores: dict[str, float] = {}
    lookup: dict[str, Chunk] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            lookup.setdefault(chunk.id, chunk)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(lookup[cid], score) for cid, score in ranked]


class HybridRetriever:
    """Retrieves chunks in one of four modes. Dependencies are injectable for tests."""

    def __init__(self, store=None, embed_query: EmbedFn | None = None, score: ScoreFn | None = None) -> None:
        if store is None:
            from .vector_store import VectorStore

            store = VectorStore()
        self.store = store
        self._embed_query = embed_query
        self._score = score
        self._bm25: BM25Index | None = None
        self._lock = threading.Lock()

    # dependencies -----------------------------------------------------------
    def embed_query(self, query: str) -> list[float]:
        if self._embed_query is None:
            from .embeddings import embed_query

            self._embed_query = embed_query
        return self._embed_query(query)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Cross-encoder relevance scores (also used by the agent to trim evidence)."""
        if self._score is None:
            from .reranker import score

            self._score = score
        return self._score(query, texts)

    @property
    def bm25(self) -> BM25Index:
        with self._lock:
            if self._bm25 is None:
                self._bm25 = BM25Index(self.store.all_chunks())
            return self._bm25

    def refresh(self) -> None:
        """Call after the store changes so the BM25 index is rebuilt on next use."""
        with self._lock:
            self._bm25 = None

    # retrieval ----------------------------------------------------------------
    def vector_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Top-k chunks by cosine similarity (overridden for a session's private documents)."""
        return self.store.search(self.embed_query(query), k)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        scores = self.score(query, [c.text for c in chunks])
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [(chunk, float(s)) for chunk, s in ranked[:top_k]]

    def retrieve(self, query: str, mode: str | None = None, final_k: int | None = None) -> list[tuple[Chunk, float]]:
        s = get_settings()
        mode = mode or s.retrieval_mode
        final_k = final_k or s.final_k
        if mode not in MODES:
            raise ValueError(f"Unknown retrieval mode '{mode}'. Use one of {MODES}.")
        if not query.strip():
            return []

        if mode == "vector":
            return self.vector_search(query, final_k)
        if mode == "bm25":
            return self.bm25.search(query, final_k)

        candidate_k = max(s.candidate_k, final_k)
        vector_hits = [c for c, _ in self.vector_search(query, candidate_k)]
        keyword_hits = [c for c, _ in self.bm25.search(query, candidate_k)]
        fused = reciprocal_rank_fusion([vector_hits, keyword_hits])
        if mode == "hybrid":
            return fused[:final_k]
        return self.rerank(query, [c for c, _ in fused[:candidate_k]], final_k)


@build_once
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
