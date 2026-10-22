"""The answer step keeps to its passages (agent/grounding.py and the unreadable-passage handling in nodes.py)."""

from researchpilot.agent import nodes
from researchpilot.agent.grounding import check_citations, sentence_spans, supports
from researchpilot.agent.schemas import Answer
from tests.conftest import fake_embed, make_chunk
from tests.test_quality import GERMAN, HINDI, legacy_font


def _passage(text: str, n: int = 1, title: str = "LoRA") -> dict:
    return {"id": f"p{n}", "text": text, "title": title, "source": f"{title}.pdf", "page": n, "origin": "document"}


LORA = _passage("LoRA freezes the pretrained weights and injects trainable rank decomposition matrices.", 1)
MEMORY = _passage("LoRA reduces trainable parameters by 10,000 times and GPU memory by 3 times.", 2)
ATTENTION = _passage("The Transformer relies entirely on multi-head attention instead of recurrence.", 3, "Attention")


def test_support_needs_shared_words_and_accepts_word_forms():
    assert supports("LoRA injects rank decomposition matrices [1].", LORA["text"])
    assert supports("The pretrained weights are frozen and stay fixed [1].", LORA["text"])  # "weights", "pretrained"
    assert supports("Weights initialized to zero [1].", "We use a zero initialization for the weights.")
    assert not supports("Durga Saptashati is a collection of 700 verses [1].", LORA["text"])
    assert supports("See [1].", LORA["text"])  # nothing to check


def test_sentences_keep_their_markers_and_lines():
    text = "First claim [1]. Second claim.[2] Third?\n- a bullet [3]\nLast one"
    pieces = [text[a:b].strip() for a, b in sentence_spans(text)]
    assert pieces == ["First claim [1].", "Second claim.[2]", "Third?", "- a bullet [3]", "Last one"]
    decimal = "It cuts memory by 2.5 times [1]. Then e.g. this."
    assert [decimal[a:b].strip() for a, b in sentence_spans(decimal)] == [
        "It cuts memory by 2.5 times [1].",
        "Then e.g.",
        "this.",
    ]


def test_unsupported_markers_and_sentences_are_removed():
    evidence = [LORA, MEMORY, ATTENTION]
    answer = (
        "LoRA freezes the pretrained weights [1][3]. It reduces GPU memory by 3 times [2].\n\n"
        "Durga Saptashati has 700 verses in praise of the goddess [3]."
    )
    checked = check_citations(answer, evidence)
    assert checked.answer == "LoRA freezes the pretrained weights [1]. It reduces GPU memory by 3 times [2]."
    assert [r["n"] for r in checked.removed] == [3, 3]
    assert checked.dropped == ["Durga Saptashati has 700 verses in praise of the goddess [3]."]


def test_paragraphs_and_lists_survive_the_check():
    answer = "Two things:\n\n- LoRA injects rank decomposition matrices [1].\n- It cuts GPU memory by 3 times [2].\n"
    assert check_citations(answer, [LORA, MEMORY]).answer == answer


def test_passages_in_another_language_are_not_word_checked():
    german = _passage(GERMAN, 1, "Menschenrechte")
    assert check_citations("All people are born free and equal [1].", [german]).removed == []


def test_generate_abstains_when_nothing_it_cites_supports_it(fake_llm):
    # The reported failure: the model answered from the title and its own knowledge, citing unrelated passages.
    fake_llm.responses["Answer"] = [
        Answer(answer="It is a collection of 700 verses in praise of the goddess Durga [1][2].", cited_sources=[1, 2])
    ]
    out = nodes.generate({"question": "What is it about?", "evidence": [LORA, MEMORY]})
    assert out["answer"] == nodes.UNSUPPORTED_ANSWER
    assert out["insufficient"] is True and out["citations"] == []
    (step,) = out["trace"]
    assert step["abstained_unsupported"] and len(step["unsupported"]) == 2


def test_generate_keeps_a_supported_answer_and_reports_cut_citations(fake_llm):
    answer = "LoRA freezes the pretrained weights [1]. The Transformer uses multi-head attention [1][3]."
    fake_llm.responses["Answer"] = [Answer(answer=answer, cited_sources=[1, 3])]
    out = nodes.generate({"question": "q", "evidence": [LORA, MEMORY, ATTENTION]})
    assert out["answer"] == "LoRA freezes the pretrained weights [1]. The Transformer uses multi-head attention [3]."
    assert [c["n"] for c in out["citations"]] == [1, 3]
    assert out["trace"][0]["unsupported"] == [{"n": 1, "sentence": "The Transformer uses multi-head attention [1][3]."}]


def test_the_prompt_rules_out_titles_and_outside_knowledge():
    prompt = nodes.ANSWER_PROMPT
    assert "not a source's title or file name" in prompt and "garbled or unreadable" in prompt


def test_unreadable_passages_never_reach_the_model(loaded_retriever, fake_llm):
    garbled = [make_chunk(f"g{i}", legacy_font(HINDI, seed=i), "durga.pdf", "Durga Saptashati", i) for i in (0, 1, 3)]
    loaded_retriever.store.add(garbled, [fake_embed("Durga Saptashati") for _ in garbled])  # the closest matches
    loaded_retriever.refresh()
    state = {"question": "Durga Saptashati", "sub_queries": ["Durga Saptashati"], "mode": "vector", "round": 0}
    update = nodes.retrieve(state)
    assert not any(item["source"] == "durga.pdf" for item in update["evidence"])
    assert update["trace"][0]["unreadable"] == 3 and update["unreadable"] == ["Durga Saptashati"]

    # Nothing readable left: it says why, without calling the model.
    out = nodes.generate({"question": "q", "evidence": [], "unreadable": update["unreadable"]})
    assert out["answer"].startswith("I can't answer this from the documents: the passages that matched in")
    assert '"Durga Saptashati"' in out["answer"] and "can't be read" in out["answer"]
    assert out["insufficient"] is True and not fake_llm.calls


def test_sources_listed_without_inline_markers_are_checked_against_the_whole_answer(fake_llm):
    answer = "LoRA freezes the pretrained weights and trains rank decomposition matrices."
    fake_llm.responses["Answer"] = [Answer(answer=answer, cited_sources=[1, 3, 9])]
    out = nodes.generate({"question": "q", "evidence": [LORA, MEMORY, ATTENTION]})
    assert [c["n"] for c in out["citations"]] == [1]  # [3] shares nothing with the answer
    assert out["invalid_citations"] == [9]


def test_a_garbled_passage_supports_nothing():
    garbled = _passage(legacy_font(HINDI, seed=1), 2, "Durga Saptashati")
    checked = check_citations("LoRA freezes the pretrained weights [1][2].", [LORA, garbled])
    assert checked.answer == "LoRA freezes the pretrained weights [1]." and [r["n"] for r in checked.removed] == [2]
    assert check_citations("It has 700 verses [1].", [garbled]).dropped == ["It has 700 verses [1]."]


def test_reporting_words_dont_count_and_each_part_of_a_sentence_may_have_its_own_passage():
    tern = _passage("The tern feeds along the tide line at dawn; note 17 records its call.", 1, "Birds")
    plover = _passage("The plover feeds along the tide line at dawn.", 2, "Birds")
    for answer in ("The guide describes the tern [1] and the plover [2].", "The guide describes the tern [1]."):
        assert check_citations(answer, [tern, plover]).removed == []


def test_an_answer_the_check_doesnt_touch_keeps_its_exact_layout():
    answer = "LoRA freezes the pretrained weights [1].  \nIt cuts GPU memory by 3 times [2].\n\n    code stays indented"
    assert check_citations(answer, [LORA, MEMORY]).answer == answer
    cut = check_citations("The weights stay frozen [1][3] and pretrained [1].", [LORA, MEMORY, ATTENTION])
    assert cut.answer == "The weights stay frozen [1] and pretrained [1]."
