import json

import pytest

from evals import metrics
from evals.report import answers_table, retrieval_table, update_readme
from evals.run import (
    DATASET,
    QuotaExhausted,
    is_rate_limit,
    load_dataset,
    rejudge_correctness,
    run_answers,
    run_retrieval,
    summarize_answers,
)


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


class _RateLimitError(Exception):
    status_code = 429


def _scripted_ask(calls, fail_on=None, fail_with=None):
    """A fake agent that answers every question, except `fail_on`, which hits a rate limit."""

    def ask(question, mode=None, max_rounds=None):
        calls.append(question)
        if question == fail_on:
            if fail_with == "trace":  # swallowed by an agent step: shows up only in the trace
                return {
                    "answer": "Sorry, the language model request failed.",
                    "insufficient": True,
                    "citations": [],
                    "round": 1,
                    "seconds": 0.1,
                    "trace": [{"step": "generate", "error": "Error code: 429 - rate_limit_exceeded (tokens per day)"}],
                }
            raise _RateLimitError("Rate limit reached for model on tokens per day (TPD)")
        return {"answer": "An answer.", "insufficient": False, "citations": [], "round": 1, "seconds": 0.1}

    return ask


def test_run_answers_stops_on_quota_and_resumes_from_checkpoint(fake_llm, tmp_path):
    fake_llm.responses = {"CorrectnessVerdict": [{"verdict": "correct", "reason": "ok"}]}
    checkpoint = tmp_path / "answers.json"
    calls = []
    with pytest.raises(QuotaExhausted, match="q2"):
        run_answers(
            DATA, ask_fn=_scripted_ask(calls, fail_on=DATA[1]["question"]), judge=fake_llm, checkpoint=checkpoint
        )
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert [r["id"] for r in saved["details"]] == ["q1"]  # the rate-limited question is not recorded
    assert saved["complete"] is False

    calls.clear()
    result = run_answers(DATA, ask_fn=_scripted_ask(calls), judge=fake_llm, checkpoint=checkpoint)
    assert calls == [DATA[1]["question"], DATA[2]["question"]]  # q1 was not asked again
    assert [r["id"] for r in result["details"]] == ["q1", "q2", "u1"] and result["complete"] is True
    assert result["summary"]["errors"] == 0


def test_run_answers_treats_a_rate_limit_swallowed_by_the_agent_as_a_stop(fake_llm, tmp_path):
    checkpoint = tmp_path / "answers.json"
    ask = _scripted_ask([], fail_on=DATA[0]["question"], fail_with="trace")
    with pytest.raises(QuotaExhausted):
        run_answers(DATA, ask_fn=ask, judge=fake_llm, checkpoint=checkpoint)
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["details"] == []


def test_run_answers_refuses_a_checkpoint_from_another_configuration(fake_llm, tmp_path):
    fake_llm.responses = {"CorrectnessVerdict": [{"verdict": "correct", "reason": "ok"}]}
    checkpoint = tmp_path / "answers.json"
    run_answers(DATA[:1], ask_fn=_scripted_ask([]), judge=fake_llm, checkpoint=checkpoint, max_rounds=1)
    with pytest.raises(SystemExit, match="different configuration"):
        run_answers(DATA[:1], ask_fn=_scripted_ask([]), judge=fake_llm, checkpoint=checkpoint, max_rounds=2)


def test_resumed_run_keeps_its_date_and_table_shows_each_runs_date(fake_llm, tmp_path):
    fake_llm.responses = {"CorrectnessVerdict": [{"verdict": "correct", "reason": "ok"}]}
    checkpoint = tmp_path / "answers.json"
    run_answers(DATA[:1], ask_fn=_scripted_ask([]), judge=fake_llm, checkpoint=checkpoint, max_rounds=1)
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    saved["meta"]["date"] = "2026-10-09 13:14 UTC"  # pretend it ran the day before
    checkpoint.write_text(json.dumps(saved), encoding="utf-8")

    first = run_answers(DATA[:1], ask_fn=_scripted_ask([]), judge=fake_llm, checkpoint=checkpoint, max_rounds=1)
    assert first["meta"]["date"] == "2026-10-09 13:14 UTC"
    second = {**first, "meta": {**first["meta"], "date": "2026-10-10 14:00 UTC"}}
    table = answers_table({"1 round": first, "2 rounds": second})
    assert "runs 1 round: 2026-10-09 13:14 UTC; 2 rounds: 2026-10-10 14:00 UTC" in table
    assert "(prompt v2)" in table


def test_declined_requires_no_citations():
    assert metrics.declined("The sources do not mention this.", True, [])
    # a cited answer that flags missing detail is still an answer, not an abstention
    assert not metrics.declined("State of the art [3]; the exact figure is not given.", True, [3])
    assert not metrics.declined("LoRA uses rank decomposition.", False, [])


def test_rejudge_regrades_old_verdicts_and_recomputes_abstention(fake_llm, tmp_path):
    from evals.judge import CORRECTNESS_PROMPT_VERSION

    checkpoint = tmp_path / "answers.json"
    old = {"verdict": "partially_correct", "score": 0.5, "reason": "too strict"}  # v1: no prompt_version
    current = {"verdict": "correct", "score": 1.0, "reason": "ok", "prompt_version": CORRECTNESS_PROMPT_VERSION}
    rows = [
        {"id": "q1", "type": "single", "answer": "Rank decomposition [1].", "insufficient": True, "cited": [1],
         "invalid_citations": [], "rounds": 1, "seconds": 1.0, "abstained": True, "correctness": old},
        {"id": "q2", "type": "multi", "answer": "Both [1].", "insufficient": False, "cited": [1],
         "invalid_citations": [], "rounds": 1, "seconds": 1.0, "abstained": False, "correctness": current},
    ]  # fmt: skip
    meta = {"judge": "custom", "judge_prompt": 1}
    checkpoint.write_text(json.dumps({"meta": meta, "complete": True, "details": rows}), encoding="utf-8")
    fake_llm.responses = {"CorrectnessVerdict": [{"verdict": "correct", "reason": "answers the question"}]}

    saved = rejudge_correctness(checkpoint, DATA, judge=fake_llm)
    names = [name for name, _ in fake_llm.calls]
    assert names.count("CorrectnessVerdict") == 1  # only the old verdict was re-graded
    q1, q2 = saved["details"]
    assert (
        q1["correctness"]["verdict"] == "correct" and q1["correctness"]["prompt_version"] == CORRECTNESS_PROMPT_VERSION
    )
    assert q1["abstained"] is False  # cited answer: no longer counted as an abstention
    assert q2["correctness"] == current
    assert saved["meta"]["judge_prompt"] == CORRECTNESS_PROMPT_VERSION
    assert saved["summary"]["false_abstention_rate"] == 0.0
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["summary"]["correctness"] == 1.0


def test_is_rate_limit():
    assert is_rate_limit(_RateLimitError("x"))
    assert is_rate_limit("Error code: 429 - {'code': 'rate_limit_exceeded'}")
    assert not is_rate_limit(ValueError("could not parse structured output")) and not is_rate_limit("")


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


def test_run_answers_stops_on_the_agents_own_rate_limit_error(fake_llm, tmp_path):
    from researchpilot.llm import LLMRateLimitError

    def ask(question, mode=None, max_rounds=None):
        raise LLMRateLimitError(daily=True)

    assert is_rate_limit(LLMRateLimitError())
    checkpoint = tmp_path / "answers.json"
    with pytest.raises(QuotaExhausted):
        run_answers(DATA, ask_fn=ask, judge=fake_llm, checkpoint=checkpoint)
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["details"] == []  # nothing recorded as a failure
