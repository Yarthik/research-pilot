from evals import metrics
from evals.report import answers_table, retrieval_table, update_readme
from evals.run import DATASET, load_dataset, run_answers, run_retrieval, summarize_answers


# --------------------------------------------------------------------------- metrics
def test_normalize_handles_pdf_messiness():
    assert metrics.contains_any("RAG-Token models generate", ["rag token"])
    assert metrics.contains_any("reduces parameters by 10,000 times", ["10000 times", "10,000 times"])
    assert metrics.contains_any("achieving 28.4 BLEU on WMT", ["28.4 BLEU"])
    assert not metrics.contains_any("bartender", ["bart"])  # whole words only


def test_same_source_ignores_arxiv_versions():
    assert metrics.same_source("https://arxiv.org/pdf/2106.09685v2", "https://arxiv.org/pdf/2106.09685")
    assert not metrics.same_source("https://arxiv.org/pdf/2106.0968", "https://arxiv.org/pdf/2106.09685")


def test_retrieval_scores():
    hits = [
        metrics.Hit("a.pdf", "nothing"),
        metrics.Hit("b.pdf", "rank decomposition"),
        metrics.Hit("a.pdf", "rank decomposition"),
    ]
    scores = metrics.retrieval_scores(hits, ["a.pdf"], ["rank decomposition"])
    assert scores["hit"] == 1.0 and scores["first_relevant_rank"] == 3
    assert scores["reciprocal_rank"] == 1 / 3
    assert metrics.retrieval_scores(hits, ["c.pdf"], ["rank"])["hit"] == 0.0


def test_source_recall_for_multi_doc_questions():
    hits = [metrics.Hit("a.pdf", "key phrase")]
    assert metrics.source_recall(hits, ["a.pdf", "b.pdf"], ["key phrase"]) == 0.5


def test_abstained():
    assert metrics.abstained("Anything", True)
    assert metrics.abstained("The documents do not contain information about GPT-4.", False)
    assert not metrics.abstained("LoRA uses rank decomposition [1].", False)


def test_mean_and_percentile():
    assert metrics.mean([1.0, 0.0, None]) == 0.5 and metrics.mean([]) == 0.0
    assert metrics.percentile([1, 2, 3, 4, 5], 50) == 3 and metrics.percentile([], 95) == 0.0


# --------------------------------------------------------------------------- dataset
def test_dataset_is_valid():
    rows = load_dataset(DATASET)
    assert len(rows) >= 25
    assert {r["type"] for r in rows} == {"single", "multi", "unanswerable"}
    for r in rows:
        if r["type"] != "unanswerable":
            assert all(s.startswith("https://arxiv.org/pdf/") for s in r["sources"])


def test_dataset_sources_match_the_sample_file():
    from pathlib import Path

    from researchpilot.cli import read_sources_file

    samples = set(read_sources_file(str(Path(DATASET).parents[1] / "data" / "sample_sources.txt")))
    needed = {s for r in load_dataset(DATASET) for s in r["sources"]}
    assert needed <= samples


# --------------------------------------------------------------------------- runners
DATA = [
    {
        "id": "q1",
        "question": "rank decomposition matrices",
        "type": "single",
        "sources": ["lora.pdf"],
        "keywords": ["rank decomposition"],
        "reference": "r",
    },
    {
        "id": "q2",
        "question": "reasoning traces and intermediate reasoning steps",
        "type": "multi",
        "sources": ["react.pdf", "cot.pdf"],
        "keywords": ["reasoning traces", "intermediate reasoning steps"],
        "reference": "r",
    },
    {
        "id": "u1",
        "question": "capital of Australia",
        "type": "unanswerable",
        "sources": [],
        "keywords": [],
        "reference": "n/a",
    },
]


def test_run_retrieval(loaded_retriever):
    result = run_retrieval(DATA, ["bm25", "hybrid_rerank"], k=3, retriever=loaded_retriever)
    assert set(result["summary"]) == {"bm25", "hybrid_rerank"}
    s = result["summary"]["hybrid_rerank"]
    assert s["questions"] == 2 and s["hit_rate@3"] == 1.0
    assert s["multi_doc_source_recall@3"] == 1.0
    assert result["never_found"] == []
    table = retrieval_table(result)
    assert "| Hybrid + rerank | 100.0% |" in table


def test_run_answers_with_fake_agent_and_judge(fake_llm):
    def fake_ask(question, mode=None, max_rounds=None):
        if "Australia" in question:
            return {
                "answer": "The documents do not mention this.",
                "insufficient": True,
                "citations": [],
                "round": 1,
                "seconds": 0.1,
            }
        citation = {
            "n": 1,
            "text": "LoRA uses rank decomposition matrices.",
            "title": "LoRA",
            "source": "lora.pdf",
            "page": 1,
            "origin": "document",
        }
        return {
            "answer": "Rank decomposition [1].",
            "insufficient": False,
            "citations": [citation],
            "invalid_citations": [],
            "round": 2,
            "seconds": 0.2,
        }

    fake_llm.responses = {
        "CorrectnessVerdict": [{"verdict": "correct", "reason": "matches"}],
        "FaithfulnessVerdict": [{"claims": [{"claim": "uses rank decomposition", "supported": True}]}],
    }
    result = run_answers(DATA, ask_fn=fake_ask, judge=fake_llm)
    s = result["summary"]
    assert s["questions"] == 3 and s["errors"] == 0
    assert s["correctness"] == 1.0 and s["faithfulness"] == 1.0
    assert s["abstention_accuracy"] == 1.0 and s["false_abstention_rate"] == 0.0
    assert s["answers_with_citations"] == 1.0
    assert "| demo |" in answers_table({"demo": result})


def test_summarize_answers_counts_errors_and_uncited_answers():
    rows = [
        {"id": "a", "type": "single", "error": "boom"},
        {
            "id": "b",
            "type": "single",
            "abstained": False,
            "cited": [],
            "invalid_citations": [3],
            "rounds": 1,
            "seconds": 1.0,
            "correctness": {"verdict": "incorrect", "score": 0.0},
            "faithfulness": {"score": 0.0},
        },
    ]
    s = summarize_answers(rows)
    assert s["errors"] == 1 and s["correctness"] == 0.0
    assert s["invalid_citation_rate"] == 1.0 and s["answers_with_citations"] == 0.0


def test_update_readme_replaces_only_the_marked_section(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "intro\n<!-- EVAL:RETRIEVAL:START -->\nold\n<!-- EVAL:RETRIEVAL:END -->\noutro\n", encoding="utf-8"
    )
    assert update_readme("RETRIEVAL", "| new |\n", readme)
    assert (
        readme.read_text(encoding="utf-8")
        == "intro\n<!-- EVAL:RETRIEVAL:START -->\n| new |\n<!-- EVAL:RETRIEVAL:END -->\noutro\n"
    )
    assert not update_readme("ANSWERS", "x", readme)
