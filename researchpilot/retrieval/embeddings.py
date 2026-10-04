"""Local, free embeddings with FastEmbed (ONNX, runs on CPU, no API key)."""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings

BATCH_SIZE = 64


@lru_cache
def _model(name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _model(get_settings().embedding_model)
    return [vector.tolist() for vector in model.embed(texts, batch_size=BATCH_SIZE)]


def embed_query(query: str) -> list[float]:
    model = _model(get_settings().embedding_model)
    # query_embed adds the query instruction that BGE models expect
    return next(iter(model.query_embed(query))).tolist()


def warmup() -> None:
    """Download and load the embedding model (used at Docker build time)."""
    embed_query("warmup")
