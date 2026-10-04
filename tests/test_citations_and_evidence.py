from researchpilot.agent.citations import normalize_citations, parse_citations, split_valid
from researchpilot.agent.evidence import select_evidence


def test_normalize_lenticular_citations():
    text = "Low-rank【1†L1-L4】【2】, ranges【3-4†source】, list【1, 5】, untouched 【note】 and [6]."
    assert normalize_citations(text) == "Low-rank[1][2], ranges[3-4], list[1, 5], untouched 【note】 and [6]."
    assert parse_citations(normalize_citations(text)) == [1, 2, 3, 4, 5, 6]
    assert normalize_citations(None) == ""


def test_parse_simple_and_adjacent_citations():
    assert parse_citations("LoRA freezes weights [1]. It adds matrices [2][3].") == [1, 2, 3]


def test_parse_lists_and_ranges():
    assert parse_citations("See [1, 4] and [2-3] and [5–6].") == [1, 4, 2, 3, 5, 6]


def test_parse_ignores_duplicates_and_non_citations():
    assert parse_citations("[2] again [2], an array a[i], a link [text](url), [x]") == [2]


def test_parse_ignores_absurd_ranges():
    assert parse_citations("[1-9999]") == []


def test_parse_empty():
    assert parse_citations("") == [] and parse_citations(None) == []


def test_split_valid():
    assert split_valid([1, 3, 0, 9], 3) == ([1, 3], [0, 9])


def item(i: str, text: str = "") -> dict:
    return {"id": i, "text": text or i}


def test_select_keeps_everything_under_the_cap():
    pool = select_evidence("q", [item("a")], [item("b")], cap=5)
    assert [p["id"] for p in pool] == ["a", "b"]


def test_select_uses_scorer_over_the_whole_pool():
    existing = [item("old1", "nothing"), item("old2", "lora rank")]
    new = [item("new1", "lora rank decomposition"), item("new2", "unrelated")]

    def scorer(question, texts):
        return [len(set(question.split()) & set(t.split())) for t in texts]

    kept = select_evidence("lora rank decomposition", existing, new, cap=2, scorer=scorer)
    assert [k["id"] for k in kept] == ["new1", "old2"]  # best first, regardless of round


def test_retry_results_survive_when_first_round_filled_the_cap():
    """Regression test: the old code kept evidence[:cap], dropping everything a retry found."""
    existing = [item(f"old{i}") for i in range(15)]
    new = [item(f"new{i}") for i in range(10)]
    kept = select_evidence("q", existing, new, cap=15)
    assert len(kept) == 15
    assert sum(k["id"].startswith("new") for k in kept) >= 7


def test_select_falls_back_when_scorer_fails():
    def broken(_q, _t):
        raise RuntimeError("model unavailable")

    kept = select_evidence("q", [item("a"), item("b")], [item("c"), item("d")], cap=2, scorer=broken)
    assert [k["id"] for k in kept] == ["a", "c"]
