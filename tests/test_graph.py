"""The full LangGraph workflow, end to end, with a scripted LLM and fake retrieval."""

import pytest

pytest.importorskip("langgraph")

from researchpilot.agent.graph import ask, ask_stream  # noqa: E402

PLAN = {"sub_queries": ["LoRA rank decomposition"], "needs_web": False}
ANSWER = {
    "answer": "LoRA injects rank decomposition matrices [1] and cuts parameters 10,000x [2].",
    "cited_sources": [1, 2],
}


def test_happy_path_one_round(loaded_retriever, fake_llm):
    fake_llm.responses = {"ResearchPlan": [PLAN], "ContextGrade": [{"sufficient": True}], "Answer": [ANSWER]}
    result = ask("How does LoRA work?")
    assert [s["step"] for s in result["trace"]] == ["plan", "retrieve", "grade", "generate"]
    assert result["round"] == 1
    assert result["citations"] and result["citations"][0]["source"] == "lora.pdf"
    assert result["insufficient"] is False
    assert result["seconds"] >= 0


def test_self_correcting_retry(loaded_retriever, fake_llm):
    fake_llm.responses = {
        "ResearchPlan": [PLAN],
        "ContextGrade": [
            {
                "sufficient": False,
                "missing": "memory savings",
                "rewritten_queries": ["LoRA GPU memory trainable parameters"],
            }
        ],
        "Answer": [ANSWER],
    }
    result = ask("How does LoRA work and how much memory does it save?")
    steps = [s["step"] for s in result["trace"]]
    assert steps == ["plan", "retrieve", "grade", "retrieve", "grade", "generate"]
    assert result["round"] == 2
    second_retrieve = result["trace"][3]
    assert second_retrieve["queries"] == ["LoRA GPU memory trainable parameters"]
    # the final grade is skipped (no rounds left), so the LLM was asked to grade only once
    assert sum(1 for name, _ in fake_llm.calls if name == "ContextGrade") == 1
    # ...and its trace step says so instead of claiming the evidence was judged sufficient
    first_grade, final_grade = result["trace"][2], result["trace"][4]
    assert first_grade["sufficient"] is False and "skipped" not in first_grade
    assert final_grade["skipped"] is True and final_grade["sufficient"] is None


def test_max_rounds_one_disables_retry(loaded_retriever, fake_llm):
    fake_llm.responses = {"ResearchPlan": [PLAN], "Answer": [ANSWER]}
    result = ask("How does LoRA work?", max_rounds=1)
    assert [s["step"] for s in result["trace"]] == ["plan", "retrieve", "grade", "generate"]
    assert not any(name == "ContextGrade" for name, _ in fake_llm.calls)
    assert result["trace"][2]["skipped"] is True
    assert result["answer"]  # routing unchanged: a skipped grade still goes on to answer


def test_mode_override_reaches_retrieval(loaded_retriever, fake_llm):
    fake_llm.responses = {"ResearchPlan": [PLAN], "ContextGrade": [{"sufficient": True}], "Answer": [ANSWER]}
    result = ask("How does LoRA work?", mode="bm25")
    assert result["trace"][1]["mode"] == "bm25"


def test_empty_index_gives_a_clear_answer(retriever, fake_llm):
    fake_llm.responses = {"ResearchPlan": [PLAN]}
    result = ask("Anything?")
    assert result["insufficient"] is True and result["citations"] == []


def test_blank_question_is_rejected():
    with pytest.raises(ValueError):
        ask("   ")


def test_stream_yields_steps_then_result(loaded_retriever, fake_llm):
    fake_llm.responses = {"ResearchPlan": [PLAN], "ContextGrade": [{"sufficient": True}], "Answer": [ANSWER]}
    events = list(ask_stream("How does LoRA work?"))
    assert [e["type"] for e in events] == ["step", "step", "step", "step", "result"]
    assert [e["step"] for e in events[:-1]] == ["plan", "retrieve", "grade", "generate"]
    assert events[-1]["answer"].startswith("LoRA injects")


def test_rate_limit_stops_the_whole_question_instead_of_degrading(loaded_retriever, fake_llm):
    from researchpilot.llm import LLMRateLimitError
    from tests.conftest import quota_error

    fake_llm.responses = {"ResearchPlan": [quota_error(daily=True)]}
    with pytest.raises(LLMRateLimitError) as caught:
        ask("How does LoRA work?")
    assert caught.value.daily is True
    assert len(fake_llm.calls) == 1  # failed fast: no doomed grade/answer calls after the first one
    assert "org_secret" not in str(caught.value)
