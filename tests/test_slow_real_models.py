"""End-to-end retrieval with the real local models and ChromaDB (no LLM).

Downloads ~150 MB of models on first run, so it is excluded from CI:  pytest -m slow
"""

import pytest

pytestmark = pytest.mark.slow


def test_real_hybrid_rerank_retrieval(tmp_path, monkeypatch):
    pytest.importorskip("fastembed")
    pytest.importorskip("chromadb")
    from researchpilot.ingestion.pipeline import ingest
    from researchpilot.retrieval import hybrid

    doc = tmp_path / "facts.md"
    doc.write_text(
        "LoRA freezes the pretrained model weights and injects trainable rank decomposition matrices.\n\n"
        "The Transformer is based solely on attention mechanisms, dispensing with recurrence.\n\n"
        "ReAct interleaves reasoning traces with actions such as Wikipedia searches.\n\n"
        "Chain-of-thought prompting adds intermediate reasoning steps to few-shot exemplars.",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHUNK_SIZE", "120")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    from researchpilot.config import reset_settings

    reset_settings()
    report = ingest([str(doc)])
    assert report["ingested"][0]["chunks"] >= 4
    for mode in ("vector", "bm25", "hybrid", "hybrid_rerank"):
        top = hybrid.get_retriever().retrieve("How does low-rank adaptation work?", mode=mode, final_k=2)
        assert top, mode
    best = hybrid.get_retriever().retrieve("How does low-rank adaptation work?", mode="hybrid_rerank", final_k=1)
    assert "rank decomposition" in best[0][0].text
