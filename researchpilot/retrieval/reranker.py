"""Cross-encoder reranking: scores each (query, passage) pair jointly for higher precision."""

from __future__ import annotations

from ..caching import build_once
from ..config import get_settings

BATCH_SIZE = 8  # fastembed defaults to 64, which needs far more memory for 512-token passage pairs


@build_once
def _model(name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=name)


def score(query: str, texts: list[str]) -> list[float]:
    """Relevance score for each text (higher is more relevant)."""
    if not texts:
        return []
    return [float(s) for s in _model(get_settings().reranker_model).rerank(query, texts, batch_size=BATCH_SIZE)]


def warmup() -> None:
    score("warmup", ["warmup"])
