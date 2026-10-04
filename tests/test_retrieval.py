import pytest

from researchpilot.retrieval.bm25 import BM25, BM25Index, tokenize
from researchpilot.retrieval.hybrid import reciprocal_rank_fusion

from .conftest import make_chunk


def make(cid: str, text: str = ""):
    return make_chunk(cid, text or cid)


# --------------------------------------------------------------------------- RRF
def test_rrf_prefers_chunks_ranked_by_both_retrievers():
    a, b, c = make("a"), make("b"), make("c")
    fused = reciprocal_rank_fusion([[a, b], [c, b]])
    assert fused[0][0].id == "b"
    assert {chunk.id for chunk, _ in fused} == {"a", "b", "c"}


def test_rrf_scores_follow_the_formula():
    a, b = make("a"), make("b")
    fused = dict((chunk.id, score) for chunk, score in reciprocal_rank_fusion([[a, b], [a]], k=60))
    assert fused["a"] == pytest.approx(2 / 61)
    assert fused["b"] == pytest.approx(1 / 62)


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []


# --------------------------------------------------------------------------- BM25
def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the LoRA rank?") == ["lora", "rank"]
    assert tokenize("GPT-3 175B") == ["gpt", "3", "175b"]


def test_bm25_finds_exact_terms():
    chunks = [
        make("1", "LoRA injects trainable rank decomposition matrices"),
        make("2", "The Transformer relies entirely on attention"),
        make("3", "ReAct interleaves reasoning and acting"),
    ]
    results = BM25Index(chunks).search("rank decomposition", 2)
    assert results[0][0].id == "1"
    assert len(results) == 1  # chunks without any query term are not returned


def test_bm25_rare_terms_weigh_more():
    bm25 = BM25([["common", "rare"], ["common"], ["common"], ["common"]])
    assert bm25.idf["rare"] > bm25.idf["common"] > 0


def test_bm25_length_normalisation():
    short = ["lora"] + ["x"] * 2
    long = ["lora"] + ["x"] * 50
    scores = BM25([short, long]).scores(["lora"])
    assert scores[0] > scores[1]


def test_bm25_empty_index_and_unknown_terms():
    assert BM25Index([]).search("anything", 5) == []
    assert BM25Index([make("1", "hello world")]).search("unrelated", 5) == []


# --------------------------------------------------------------------------- HybridRetriever
@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid", "hybrid_rerank"])
def test_every_mode_finds_the_relevant_chunk(loaded_retriever, mode):
    results = loaded_retriever.retrieve("rank decomposition matrices in LoRA", mode=mode, final_k=3)
    assert results, mode
    assert results[0][0].id == "lora1"
    assert len(results) <= 3


def test_unknown_mode_is_rejected(loaded_retriever):
    with pytest.raises(ValueError):
        loaded_retriever.retrieve("query", mode="magic")


def test_blank_query_returns_nothing(loaded_retriever):
    assert loaded_retriever.retrieve("   ") == []


def test_refresh_picks_up_new_chunks(loaded_retriever, store):
    from .conftest import fake_embed

    assert loaded_retriever.retrieve("zebra", mode="bm25") == []
    new = make_chunk("z", "A zebra appears.")
    store.add([new], [fake_embed(new.text)])
    loaded_retriever.refresh()
    assert loaded_retriever.retrieve("zebra", mode="bm25")[0][0].id == "z"


def test_empty_store(retriever):
    assert retriever.retrieve("anything", mode="hybrid_rerank") == []
