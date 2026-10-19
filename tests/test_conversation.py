"""Conversation context: follow-ups (condense), attached-file scope, vague questions and the grader's verdicts."""

import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from researchpilot.agent import conversation, nodes  # noqa: E402
from researchpilot.agent.graph import ask  # noqa: E402
from researchpilot.retrieval import session as session_module  # noqa: E402
from researchpilot.retrieval.session import SessionRegistry  # noqa: E402
from tests.conftest import quota_error  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLAN = {"sub_queries": ["LoRA rank decomposition"], "needs_web": False}
ANSWER = {"answer": "LoRA injects rank decomposition matrices [1].", "cited_sources": [1]}
SUFFICIENT = {"verdict": "sufficient"}
HISTORY = [
    {"role": "user", "content": "How does LoRA reduce the number of trainable parameters?"},
    {"role": "assistant", "content": "LoRA freezes the weights and trains low-rank matrices [1]."},
]
EVIDENCE = [{"id": "a", "text": "t", "title": "T", "page": 1, "origin": "document"}]
RESUME = b"Alex Morgan, frontend engineer. Six years building web apps with React, TypeScript and Node.js."


def _steps(result) -> list[str]:
    return [step["step"] for step in result["trace"]]


def _calls(fake_llm, name: str) -> list[str]:
    return [prompt for schema, prompt in fake_llm.calls if schema == name]


# --------------------------------------------------------------------------- no history: exactly as before
def test_a_question_without_history_takes_the_same_path_and_calls_as_before(loaded_retriever, fake_llm):
    fake_llm.responses = {"ResearchPlan": [PLAN], "ContextGrade": [SUFFICIENT], "Answer": [ANSWER]}
    result = ask("How does LoRA work?")
    assert _steps(result) == ["plan", "retrieve", "grade", "generate"]
    assert [name for name, _ in fake_llm.calls] == ["ResearchPlan", "ContextGrade", "Answer"]  # no condense call
    assert "standalone_question" not in result and "clarification" not in result
    assert "Question: How does LoRA work?" in _calls(fake_llm, "Answer")[0]


@pytest.mark.parametrize("history", [None, [], [{"role": "system", "content": "x"}, {"role": "user", "content": " "}]])
def test_empty_or_malformed_history_counts_as_none(loaded_retriever, fake_llm, history):
    fake_llm.responses = {"ResearchPlan": [PLAN], "ContextGrade": [SUFFICIENT], "Answer": [ANSWER]}
    assert _steps(ask("How does LoRA work?", history=history))[0] == "plan"


def test_every_eval_and_example_question_is_clear():
    """The vague-question check must never catch a real question (it would answer with a list instead)."""
    rows = [json.loads(line) for line in (ROOT / "evals" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    examples = [
        "How does LoRA reduce the number of trainable parameters?",
        "How do RAG and LoRA differ in the way they adapt a pre-trained language model?",
        "What is attention?",
        "Who are the authors?",
        "What is it used for in ReAct?",
        "Summarize the LoRA paper",
        "What does this say about GPU memory?",
    ]
    for question in [r["question"] for r in rows] + examples:
        assert not conversation.is_vague(question), question


@pytest.mark.parametrize(
    "question, vague, library",
    [
        ("What is it about?", True, False),
        ("what's this paper about", True, False),
        ("Summarize this document.", True, False),
        ("Tell me the main points", True, False),
        ("What are these papers about?", True, True),
        ("What documents do you have?", True, True),
        ("What is the library about?", True, True),
        ("What is LoRA about?", False, False),
        ("Is it fast?", False, False),
    ],
)
def test_vague_questions_and_library_questions(question, vague, library):
    assert conversation.is_vague(question) is vague
    if vague:
        assert conversation.asks_about_library(question) is library


# --------------------------------------------------------------------------- vague questions without context
def test_a_vague_question_lists_the_documents_instead_of_searching(loaded_retriever, fake_llm):
    result = ask("What is it about?")
    assert _steps(result) == ["clarify"] and not fake_llm.calls  # no LLM call, no search
    assert result["clarification"] is True and result["insufficient"] is False and result["citations"] == []
    assert result["answer"].startswith("Which document do you mean?")
    for title in ("LoRA", "Attention", "ReAct", "CoT"):
        assert f"- {title}" in result["answer"]
    assert result["trace"][0]["kind"] == "ambiguous"


def test_a_question_about_the_library_describes_it(loaded_retriever, fake_llm):
    result = ask("What are these papers about?")
    assert _steps(result) == ["clarify"] and result["trace"][0]["kind"] == "library"
    assert result["answer"].startswith("I can search 4 documents:")


def test_a_vague_question_with_a_single_document_is_just_searched(retriever, store, fake_llm):
    from tests.conftest import CORPUS, fake_embed

    lora = [c for c in CORPUS if c.source == "lora.pdf"]
    store.add(lora, [fake_embed(c.text) for c in lora])
    retriever.refresh()
    fake_llm.responses = {"ResearchPlan": [PLAN], "ContextGrade": [SUFFICIENT], "Answer": [ANSWER]}
    assert _steps(ask("What is it about?"))[0] == "plan"  # only one thing it can mean


def test_library_answer_marks_the_visitors_own_files_and_title_cases():
    kind, text = conversation.library_answer(
        "What is it about?",
        [{"source": "x", "title": "LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS"}],
        [{"source": "cv.pdf", "title": "Alex Morgan CV"}],
    )
    assert kind == "ambiguous"
    assert "- LoRA: Low-Rank Adaptation of Large Language Models\n- Alex Morgan CV (your file)" in text
    assert 'for example "What is Alex Morgan CV about?"' in text


# --------------------------------------------------------------------------- follow-ups
def test_a_follow_up_is_rewritten_into_a_standalone_question(loaded_retriever, fake_llm):
    fake_llm.responses = {
        "StandaloneQuestion": [{"question": "What is the LoRA paper about?"}],
        "ResearchPlan": [PLAN],
        "ContextGrade": [SUFFICIENT],
        "Answer": [ANSWER],
    }
    result = ask("What is it about?", history=HISTORY)
    assert _steps(result) == ["condense", "plan", "retrieve", "grade", "generate"]
    assert result["question"] == "What is it about?"  # as asked
    assert result["standalone_question"] == "What is the LoRA paper about?"
    condense_prompt = _calls(fake_llm, "StandaloneQuestion")[0]
    assert "User: How does LoRA reduce" in condense_prompt and "Latest message: What is it about?" in condense_prompt
    for name in ("ResearchPlan", "ContextGrade", "Answer"):  # every later step works on the rewrite
        assert "What is the LoRA paper about?" in _calls(fake_llm, name)[0]


def test_a_failed_condense_call_falls_back_to_the_question_as_asked(loaded_retriever, fake_llm):
    fake_llm.responses = {
        "StandaloneQuestion": [RuntimeError("boom")],
        "ResearchPlan": [PLAN],
        "ContextGrade": [SUFFICIENT],
        "Answer": [ANSWER],
    }
    result = ask("How much memory does it save?", history=HISTORY)
    assert "boom" in result["trace"][0]["error"] and "standalone_question" not in result
    assert "Question: How much memory does it save?" in _calls(fake_llm, "Answer")[0]


def test_a_failed_condense_on_a_vague_follow_up_asks_which_document(loaded_retriever, fake_llm):
    fake_llm.responses = {"StandaloneQuestion": [RuntimeError("boom")]}
    assert _steps(ask("What is it about?", history=HISTORY)) == ["condense", "clarify"]


def test_a_rate_limit_in_the_condense_step_is_raised(loaded_retriever, fake_llm):
    from researchpilot.llm import LLMRateLimitError

    fake_llm.responses = {"StandaloneQuestion": [quota_error()]}
    with pytest.raises(LLMRateLimitError):
        ask("What is it about?", history=HISTORY)


def test_history_is_capped_to_the_last_messages_and_shortened():
    long = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "x " * 1000} for i in range(10)]
    cleaned = conversation.clean_history(long)
    assert len(cleaned) == conversation.MAX_HISTORY_MESSAGES and cleaned[0]["content"].startswith("m4 ")
    assert all(len(turn["content"]) <= conversation.MAX_MESSAGE_CHARS + 2 for turn in cleaned)


# --------------------------------------------------------------------------- attached files
@pytest.fixture
def registry(loaded_retriever, monkeypatch):
    registry = SessionRegistry()
    monkeypatch.setattr(session_module, "get_registry", lambda: registry)
    registry.add_file("alice", "resume.txt", RESUME)
    return registry


ATTACHED = [{"source": "resume.txt", "title": "resume"}]


def _retrieved_sources(result) -> set[str]:
    return {c["source"] for c in result["citations"]}


def test_what_is_it_about_with_a_new_file_searches_only_that_file(registry, fake_llm):
    fake_llm.responses = {
        "ResearchPlan": [{"sub_queries": ["React experience"], "needs_web": False}],
        "ContextGrade": [SUFFICIENT],
        "Answer": [{"answer": "A frontend engineer's resume [1].", "cited_sources": [1]}],
    }
    result = ask("What is it about?", owner="alice", attachments=ATTACHED)
    assert _steps(result) == ["condense", "plan", "retrieve", "grade", "generate"]
    assert not _calls(fake_llm, "StandaloneQuestion")  # nothing to resolve: no extra LLM call
    assert result["trace"][2]["scope"] == ["resume.txt"]
    answer_prompt = _calls(fake_llm, "Answer")[0]
    assert "Alex Morgan" in answer_prompt and "ReAct" not in answer_prompt  # the shared ReAct paper stays out


def test_a_vague_question_with_a_new_file_mid_conversation_means_that_file(registry, fake_llm):
    fake_llm.responses = {"ResearchPlan": [PLAN], "ContextGrade": [SUFFICIENT], "Answer": [ANSWER]}
    result = ask("What is it about?", owner="alice", history=HISTORY, attachments=ATTACHED)
    assert not _calls(fake_llm, "StandaloneQuestion") and result["trace"][2]["scope"] == ["resume.txt"]


def test_a_specific_question_about_the_new_file_is_scoped_when_the_model_says_so(registry, fake_llm):
    fake_llm.responses = {
        "StandaloneQuestion": [{"question": "Does Alex Morgan know React?", "about_attached_files": True}],
        "ResearchPlan": [{"sub_queries": ["React"], "needs_web": False}],
        "ContextGrade": [SUFFICIENT],
        "Answer": [ANSWER],
    }
    result = ask("Does this person know React?", owner="alice", attachments=ATTACHED)
    assert result["trace"][0]["files"] == ["resume"] and result["trace"][2]["scope"] == ["resume.txt"]
    assert '"resume.txt"' in _calls(fake_llm, "StandaloneQuestion")[0]
    assert "ReAct" not in _calls(fake_llm, "Answer")[0]


def test_a_question_that_also_names_other_documents_is_not_scoped(registry, fake_llm):
    fake_llm.responses = {
        "StandaloneQuestion": [{"question": "Compare the resume with ReAct", "about_attached_files": False}],
        "ResearchPlan": [{"sub_queries": ["ReAct reasoning"], "needs_web": False}],
        "ContextGrade": [SUFFICIENT],
        "Answer": [ANSWER],
    }
    result = ask("Compare this with ReAct", owner="alice", attachments=ATTACHED)
    assert "scope" not in result["trace"][2]
    assert "ReAct" in _calls(fake_llm, "Answer")[0]


def test_a_question_that_ignores_the_new_file_skips_the_condense_step(registry, fake_llm):
    fake_llm.responses = {"ResearchPlan": [PLAN], "ContextGrade": [SUFFICIENT], "Answer": [ANSWER]}
    assert _steps(ask("How does LoRA reduce parameters?", owner="alice", attachments=ATTACHED))[0] == "plan"


def test_the_model_cannot_scope_a_question_without_new_files(registry, fake_llm):
    fake_llm.responses = {
        "StandaloneQuestion": [{"question": "What is LoRA about?", "about_attached_files": True}],
        "ResearchPlan": [PLAN],
        "ContextGrade": [SUFFICIENT],
        "Answer": [ANSWER],
    }
    result = ask("What is it about?", owner="alice", history=HISTORY)
    assert "scope" not in result["trace"][2]


def test_a_removed_file_falls_back_to_searching_everything(registry, fake_llm):
    fake_llm.responses = {"ResearchPlan": [PLAN], "ContextGrade": [SUFFICIENT], "Answer": [ANSWER]}
    registry.remove("alice", "resume.txt")
    result = ask("What is it about?", owner="alice", attachments=ATTACHED)
    assert result["trace"][2].get("scope_missing") is True


def test_attachments_need_a_session(loaded_retriever, fake_llm):
    fake_llm.responses = {"ResearchPlan": [PLAN], "ContextGrade": [SUFFICIENT], "Answer": [ANSWER]}
    assert _steps(ask("How does LoRA work?", attachments=ATTACHED))[0] == "plan"  # ignored without an owner


# --------------------------------------------------------------------------- the grader's three-way verdict
def _grade(fake_llm, response, round_=1, sub_queries=("old query",)):
    fake_llm.responses["ContextGrade"] = [response]
    state = {"question": "q", "evidence": EVIDENCE, "round": round_, "sub_queries": list(sub_queries)}
    out = nodes.grade(state)
    return out, nodes.route_after_grade({**state, **out})


def test_verdict_sufficient_answers(fake_llm):
    out, route = _grade(fake_llm, {"verdict": "sufficient", "rewritten_queries": ["ignored"]})
    assert route == "generate" and out["verdict"] == "sufficient" and out["sufficient"] is True


def test_verdict_retry_searches_again(fake_llm):
    response = {"verdict": "retry", "missing": "the actions", "rewritten_queries": ["ReAct actions"]}
    out, route = _grade(fake_llm, response)
    assert route == "retrieve" and out["sub_queries"] == ["ReAct actions"] and out["trace"][0]["verdict"] == "retry"


def test_verdict_out_of_scope_goes_straight_to_the_answer(fake_llm):
    out, route = _grade(fake_llm, {"verdict": "out_of_scope", "missing": "GPT-4's learning rate"})
    assert route == "generate" and out["verdict"] == "out_of_scope" and out["new_queries"] == []
    assert out["trace"][0]["missing"] == "GPT-4's learning rate"


def test_out_of_scope_with_queries_is_treated_as_a_retry(fake_llm):
    out, route = _grade(fake_llm, {"verdict": "out_of_scope", "rewritten_queries": ["GPT-4 learning rate"]})
    assert route == "retrieve" and out["verdict"] == "retry"


@pytest.mark.parametrize(
    "raw, expected", [("Out of scope", "out_of_scope"), ("RETRY", "retry"), ('"sufficient"', "sufficient")]
)
def test_verdict_spelling_variants_are_understood(raw, expected):
    assert nodes.parse_verdict(raw) == expected


@pytest.mark.parametrize("raw", ["", "maybe", "insufficient", None])
def test_a_malformed_verdict_falls_back_to_the_old_rule(fake_llm, raw):
    with_queries, route = _grade(fake_llm, {"verdict": raw or "", "rewritten_queries": ["new query"]})
    assert route == "retrieve" and with_queries["verdict"] is None  # new queries: retry, as before
    without, route = _grade(fake_llm, {"verdict": raw or "", "rewritten_queries": []})
    assert route == "generate" and without["sufficient"] is False  # none: answer, as before


def test_an_out_of_scope_question_searches_once(loaded_retriever, fake_llm):
    fake_llm.responses = {
        "ResearchPlan": [{"sub_queries": ["GPT-4 learning rate"], "needs_web": False}],
        "ContextGrade": [{"verdict": "out_of_scope", "missing": "GPT-4's training details"}],
        "Answer": [{"answer": "The documents don't say.", "cited_sources": [], "insufficient_information": True}],
    }
    result = ask("What learning rate was used to train GPT-4?")
    assert _steps(result) == ["plan", "retrieve", "grade", "generate"] and result["round"] == 1
    assert result["insufficient"] is True
