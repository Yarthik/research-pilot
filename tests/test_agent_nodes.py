"""Agent nodes tested one at a time with a scripted LLM (no LangGraph needed)."""

import pytest

from researchpilot.agent import nodes
from researchpilot.agent.schemas import Answer
from researchpilot.llm import LLMRateLimitError
from tests.conftest import make_chunk, quota_error


def test_plan_uses_llm_queries(fake_llm):
    fake_llm.responses["ResearchPlan"] = [
        {"sub_queries": ["LoRA rank", " LoRA rank ", "", "GPU memory"], "needs_web": True}
    ]
    out = nodes.plan({"question": "Tell me about LoRA"})
    assert out["sub_queries"] == ["LoRA rank", "GPU memory"]  # cleaned and de-duplicated
    assert out["needs_web"] is False  # no Tavily key configured
    assert out["round"] == 0 and out["evidence"] == []
    assert out["trace"][0]["step"] == "plan"


def test_plan_falls_back_to_the_question(fake_llm):
    fake_llm.responses["ResearchPlan"] = [RuntimeError("rate limited")]
    out = nodes.plan({"question": "What is LoRA?"})
    assert out["sub_queries"] == ["What is LoRA?"]
    assert "rate limited" in out["trace"][0]["error"]


def test_retrieve_collects_dedupes_and_counts(loaded_retriever):
    state = {"question": "LoRA", "sub_queries": ["LoRA rank decomposition", "LoRA trainable parameters"], "round": 0}
    out = nodes.retrieve(state)
    ids = [e["id"] for e in out["evidence"]]
    assert len(ids) == len(set(ids))
    assert "lora1" in ids and "lora2" in ids
    assert out["round"] == 1
    step = out["trace"][0]
    assert step["step"] == "retrieve" and step["evidence"] == len(ids)


def test_retrieve_respects_the_evidence_cap(loaded_retriever, monkeypatch):
    monkeypatch.setenv("MAX_EVIDENCE", "2")
    from researchpilot.config import reset_settings

    reset_settings()
    out = nodes.retrieve(
        {"question": "attention lora react", "sub_queries": ["attention", "lora", "react"], "round": 0}
    )
    assert len(out["evidence"]) == 2


def test_retrieve_second_round_keeps_new_evidence(loaded_retriever, monkeypatch):
    monkeypatch.setenv("MAX_EVIDENCE", "2")
    from researchpilot.config import reset_settings

    reset_settings()
    first = {"id": "tr1", "text": "irrelevant filler", "title": "x", "source": "x", "page": 1, "origin": "document"}
    second = {**first, "id": "tr2"}
    state = {
        "question": "rank decomposition matrices",
        "sub_queries": ["rank decomposition matrices"],
        "round": 1,
        "evidence": [first, second],
    }
    out = nodes.retrieve(state)
    assert "lora1" in [e["id"] for e in out["evidence"]]  # the retry's best hit made it in
    assert out["trace"][0]["kept_new"] >= 1


def test_retrieve_without_scorer_keeps_every_sub_querys_best_hits(loaded_retriever, monkeypatch):
    # In modes without a cross-encoder, trimming must not drop whole sub-queries.
    monkeypatch.setenv("MAX_EVIDENCE", "3")
    from researchpilot.config import reset_settings

    reset_settings()
    ranked = {q: [(make_chunk(f"{q}{r}", f"{q} text {r}"), 1.0 / r) for r in (1, 2, 3)] for q in ("a", "b", "c")}
    monkeypatch.setattr(loaded_retriever, "retrieve", lambda query, mode=None: ranked[query])
    out = nodes.retrieve({"question": "q", "sub_queries": ["a", "b", "c"], "round": 0, "mode": "hybrid"})
    assert [e["id"] for e in out["evidence"]] == ["a1", "b1", "c1"]


def test_retrieve_adds_web_results(loaded_retriever, monkeypatch):
    web = [{"id": "web:1", "text": "news", "title": "News", "source": "https://x.com", "page": None, "origin": "web"}]
    monkeypatch.setattr(nodes, "web_search", lambda q: web)
    out = nodes.retrieve({"question": "LoRA", "sub_queries": ["LoRA"], "round": 0, "needs_web": True})
    assert "web:1" in [e["id"] for e in out["evidence"]]
    assert out["trace"][0]["new_web_results"] == 1


def test_grade_requests_new_queries(fake_llm):
    fake_llm.responses["ContextGrade"] = [
        {"verdict": "retry", "missing": "GPU memory", "rewritten_queries": ["LoRA GPU memory", "old query"]}
    ]
    state = {
        "question": "q",
        "evidence": [{"id": "a", "text": "t", "title": "T", "page": 1, "origin": "document"}],
        "round": 1,
        "sub_queries": ["old query"],
    }
    out = nodes.grade(state)
    assert out["sufficient"] is False
    assert out["new_queries"] == ["LoRA GPU memory"]  # repeated query dropped
    assert out["sub_queries"] == ["LoRA GPU memory"]
    assert nodes.route_after_grade({**state, **out}) == "retrieve"


def test_grade_skips_llm_on_last_round(fake_llm):
    state = {
        "question": "q",
        "evidence": [{"id": "a", "text": "t", "title": "T", "page": 1, "origin": "document"}],
        "round": 2,
    }
    out = nodes.grade(state)
    assert out["sufficient"] is True and not fake_llm.calls
    assert nodes.route_after_grade({**state, **out}) == "generate"


def test_grade_without_evidence_goes_to_answer(fake_llm):
    out = nodes.grade({"question": "q", "evidence": [], "round": 1})
    assert nodes.route_after_grade({"round": 1, **out}) == "generate"


def test_grade_failure_proceeds_to_answer(fake_llm):
    fake_llm.responses["ContextGrade"] = [RuntimeError("boom")]
    state = {
        "question": "q",
        "evidence": [{"id": "a", "text": "t", "title": "T", "page": 1, "origin": "document"}],
        "round": 1,
    }
    out = nodes.grade(state)
    assert out["sufficient"] is True and "boom" in out["trace"][0]["error"]


def test_route_respects_max_rounds_override():
    state = {"sufficient": False, "new_queries": ["x"], "round": 1, "max_rounds": 1}
    assert nodes.route_after_grade(state) == "generate"
    assert nodes.route_after_grade({**state, "max_rounds": 3}) == "retrieve"


EVIDENCE = [
    {
        "id": "a",
        "text": "LoRA uses rank decomposition.",
        "title": "LoRA",
        "source": "lora.pdf",
        "page": 1,
        "origin": "document",
    },
    {"id": "b", "text": "It cuts memory 3x.", "title": "LoRA", "source": "lora.pdf", "page": 2, "origin": "document"},
]


def test_generate_builds_citations_from_the_text(fake_llm):
    fake_llm.responses["Answer"] = [Answer(answer="Rank decomposition [1], 3x memory [2][7].", cited_sources=[1])]
    out = nodes.generate({"question": "q", "evidence": EVIDENCE})
    assert [c["n"] for c in out["citations"]] == [1, 2]
    assert out["citations"][1]["page"] == 2
    assert out["invalid_citations"] == [7]
    assert out["insufficient"] is False


def test_generate_normalizes_lenticular_citations(fake_llm):
    # gpt-oss writes 【n†Lx-Ly】 instead of [n]; the answer shown to users must use [n].
    fake_llm.responses["Answer"] = [
        Answer(answer="Rank decomposition cuts memory【1†L1-L4】【2†L3】.", cited_sources=[1])
    ]
    out = nodes.generate({"question": "q", "evidence": EVIDENCE})
    assert out["answer"] == "Rank decomposition cuts memory[1][2]."
    assert [c["n"] for c in out["citations"]] == [1, 2]


def test_generate_falls_back_to_plain_text(fake_llm):
    fake_llm.responses["Answer"] = [ValueError("bad JSON")]
    fake_llm.text = "LoRA uses rank decomposition [1]."
    out = nodes.generate({"question": "q", "evidence": EVIDENCE})
    assert out["answer"] == "LoRA uses rank decomposition [1]."
    assert [c["n"] for c in out["citations"]] == [1]


def test_generate_survives_total_llm_failure(fake_llm, monkeypatch):
    fake_llm.responses["Answer"] = [ValueError("bad JSON")]

    def broken(_prompt):
        raise RuntimeError("API down")

    monkeypatch.setattr(fake_llm, "invoke", broken)
    out = nodes.generate({"question": "q", "evidence": EVIDENCE})
    assert "API down" in out["answer"] and out["insufficient"] is True


def test_generate_without_evidence(fake_llm):
    out = nodes.generate({"question": "q", "evidence": []})
    assert out["insufficient"] is True and out["citations"] == [] and not fake_llm.calls


def test_format_context_numbers_sources():
    text = nodes.format_context(EVIDENCE, max_chars=10)
    assert text.startswith("[1] LoRA, p. 1 (document)\nLoRA uses ")
    assert "[2] LoRA, p. 2" in text


def test_llm_configuration_errors_are_not_swallowed(monkeypatch):
    """A missing API key should reach the user as a clear error, not as a degraded answer."""
    import pytest

    from researchpilot import llm

    def missing_key():
        raise llm.LLMConfigError("Could not initialise groq/llama: GROQ_API_KEY not set")

    monkeypatch.setattr(llm, "get_llm", missing_key)
    with pytest.raises(llm.LLMConfigError):
        nodes.plan({"question": "q"})
    with pytest.raises(llm.LLMConfigError):
        nodes.generate({"question": "q", "evidence": EVIDENCE})


def test_a_rate_limit_is_raised_not_hidden_by_a_degraded_step(loaded_retriever, fake_llm):
    # Plan, grade and generate degrade on ordinary failures, but a rate limit would only repeat on every
    # step and end in a raw provider error shown as the answer. It must reach the caller instead.
    fake_llm.responses = {"ResearchPlan": [quota_error()]}
    with pytest.raises(LLMRateLimitError):
        nodes.plan({"question": "q"})
    fake_llm.responses = {"ContextGrade": [quota_error()]}
    with pytest.raises(LLMRateLimitError):
        nodes.grade({"question": "q", "evidence": EVIDENCE, "round": 1})
    fake_llm.responses = {"Answer": [quota_error()]}
    with pytest.raises(LLMRateLimitError):
        nodes.generate({"question": "q", "evidence": EVIDENCE})


def test_a_rate_limit_in_the_plain_text_fallback_is_raised_too(fake_llm, monkeypatch):
    fake_llm.responses = {"Answer": [RuntimeError("could not parse structured output")]}
    monkeypatch.setattr(fake_llm, "invoke", lambda prompt: (_ for _ in ()).throw(quota_error()))
    with pytest.raises(LLMRateLimitError):
        nodes.generate({"question": "q", "evidence": EVIDENCE})
