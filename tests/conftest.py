"""Shared test fixtures.

Tests never call an LLM API or download models: the LLM, embeddings, vector store and
reranker are replaced by small deterministic fakes that implement the same interfaces.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from researchpilot import config, llm
from researchpilot.ingestion.chunking import Chunk
from researchpilot.retrieval import embeddings, hybrid, session
from researchpilot.retrieval.bm25 import tokenize
from researchpilot.retrieval.vector_store import summarize_sources

DIM = 256


# --------------------------------------------------------------------------- fakes
def fake_embed(text: str) -> list[float]:
    """Hashed bag-of-words vector: texts sharing words get similar vectors."""
    vector = [0.0] * DIM
    for token in tokenize(text):
        vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def fake_score(query: str, texts: list[str]) -> list[float]:
    """Word-overlap 'cross-encoder'."""
    q = set(tokenize(query))
    return [len(q & set(tokenize(t))) / (len(q) or 1) for t in texts]


class InMemoryStore:
    """Same interface as VectorStore, backed by a dict."""

    def __init__(self) -> None:
        self.items: dict[str, tuple[Chunk, list[float]]] = {}

    def count(self) -> int:
        return len(self.items)

    def add(self, chunks, vectors) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.items[chunk.id] = (chunk, vector)

    def search(self, query_embedding, k):
        def cosine(vector):
            return sum(a * b for a, b in zip(query_embedding, vector, strict=True))

        scored = [(chunk, cosine(vector)) for chunk, vector in self.items.values()]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def all_chunks(self):
        return [chunk for chunk, _ in self.items.values()]

    def delete_source(self, source: str) -> int:
        ids = [cid for cid, (chunk, _) in self.items.items() if chunk.source == source]
        for cid in ids:
            del self.items[cid]
        return len(ids)

    def sources(self):
        return summarize_sources([c.metadata() for c in self.all_chunks()])


@dataclass
class FakeLLM:
    """Scripted chat model. `responses` maps a schema name to a list of outputs (or exceptions)."""

    responses: dict = field(default_factory=dict)
    text: str = "Plain text answer [1]."
    calls: list = field(default_factory=list)

    def with_structured_output(self, schema):
        fake = self

        class _Runnable:
            def invoke(self, prompt):
                fake.calls.append((schema.__name__, prompt))
                queue = fake.responses.get(schema.__name__)
                if not queue:
                    raise RuntimeError(f"no scripted response for {schema.__name__}")
                item = queue.pop(0) if len(queue) > 1 else queue[0]
                if isinstance(item, Exception):
                    raise item
                return item if isinstance(item, schema) else schema(**item)

        return _Runnable()

    def invoke(self, prompt):
        self.calls.append(("text", prompt))
        return SimpleNamespace(content=self.text)


def make_chunk(cid: str, text: str, source: str = "doc.pdf", title: str = "Doc", page: int | None = 1) -> Chunk:
    return Chunk(cid, text, source, title, page, 0)


CORPUS = [
    make_chunk(
        "lora1",
        "LoRA freezes the pretrained weights and injects trainable rank decomposition matrices.",
        "lora.pdf",
        "LoRA",
        1,
    ),
    make_chunk(
        "lora2", "LoRA reduces trainable parameters by 10,000 times and GPU memory by 3 times.", "lora.pdf", "LoRA", 2
    ),
    make_chunk(
        "tr1",
        "The Transformer relies entirely on multi-head attention instead of recurrence.",
        "attention.pdf",
        "Attention",
        1,
    ),
    make_chunk(
        "tr2",
        "Positional encodings use sine and cosine functions of different frequencies.",
        "attention.pdf",
        "Attention",
        6,
    ),
    make_chunk(
        "re1",
        "ReAct generates reasoning traces and task-specific actions in an interleaved manner.",
        "react.pdf",
        "ReAct",
        1,
    ),
    make_chunk(
        "cot1", "Chain-of-thought prompting uses a series of intermediate reasoning steps.", "cot.pdf", "CoT", 1
    ),
]


# --------------------------------------------------------------------------- fixtures
def _clear_caches() -> None:
    funcs = [config.get_settings, hybrid.get_retriever, llm.get_llm, session.get_registry]
    try:
        from researchpilot.agent import graph

        funcs.append(graph.get_graph)
    except ImportError:  # langgraph not installed
        pass
    for func in funcs:  # some may be monkeypatched with plain functions at this point
        getattr(func, "cache_clear", lambda: None)()


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Fresh settings per test, no real API keys, storage in a temp dir."""
    for name in list(config.Settings.model_fields):
        monkeypatch.delenv(name.upper(), raising=False)
    for key in (
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "TAVILY_API_KEY",
        "API_URL",
        "AUTO_INGEST_SAMPLES",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)  # ignore a developer's .env
    _clear_caches()
    yield
    _clear_caches()


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def retriever(monkeypatch, store):
    """A real HybridRetriever over the in-memory store with fake embeddings and reranker."""
    r = hybrid.HybridRetriever(store=store, embed_query=fake_embed, score=fake_score)
    monkeypatch.setattr(hybrid, "get_retriever", lambda: r)
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts: [fake_embed(t) for t in texts])
    monkeypatch.setattr(embeddings, "embed_query", fake_embed)
    return r


@pytest.fixture
def loaded_retriever(retriever, store):
    store.add(CORPUS, [fake_embed(c.text) for c in CORPUS])
    retriever.refresh()
    return retriever


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "get_llm", lambda: fake)
    return fake


def quota_error(daily: bool = True) -> Exception:
    """What a provider SDK raises when the free-tier quota is used up (worded like Groq's, with an org id)."""
    window = "tokens per day (TPD)" if daily else "tokens per minute (TPM)"
    return RuntimeError(f"Error code: 429 - Rate limit reached for model `m` in organization `org_secret` on {window}")
