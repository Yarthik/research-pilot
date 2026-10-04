"""Cross-encoder reranking: scores each (query, passage) pair jointly for higher precision."""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings


@lru_cache
def _model(name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=name)


def score(query: str, texts: list[str]) -> list[float]:
    """Relevance score for each text (higher is more relevant)."""
    if not texts:
        return []
    return [float(s) for s in _model(get_settings().reranker_model).rerank(query, texts)]


def warmup() -> None:
    score("warmup", ["warmup"])
