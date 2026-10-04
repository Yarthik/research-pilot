"""Keyword retrieval with Okapi BM25.

BM25 catches exact terms, acronyms, names and numbers that embeddings can blur. The scorer is
implemented here (about 30 lines with NumPy) rather than pulled from a library, so the ranking
formula is visible and tested.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from ..ingestion.chunking import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
# fmt: off
_STOPWORDS = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has",
    "have", "how", "if", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "were", "what", "when", "where", "which", "who", "why", "will", "with",
])
# fmt: on


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    """Okapi BM25: score(q, d) = sum over q terms of IDF(t) * tf*(k1+1) / (tf + k1*(1 - b + b*|d|/avgdl))."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n_docs = len(corpus)
        self.doc_len = np.array([len(doc) for doc in corpus], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        self.term_freqs = [Counter(doc) for doc in corpus]
        doc_freq: Counter = Counter()
        for doc in corpus:
            doc_freq.update(set(doc))
        # Lucene-style IDF: always positive, even for terms in most documents
        self.idf = {t: math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs)
        if not self.n_docs or self.avgdl == 0:
            return scores
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([freqs.get(term, 0) for freqs in self.term_freqs], dtype=float)
            scores += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.bm25 = BM25([tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.scores(tokenize(query))
        top = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]
