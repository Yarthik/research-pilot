import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from render import short_title, streamlit_markdown  # noqa: E402


def test_latex_delimiters_become_streamlit_math():
    text = r"B \( \in \mathbb{R}^{d\times r} \) and \[ W = W_0 + BA \] end"
    assert streamlit_markdown(text) == r"B $\in \mathbb{R}^{d\times r}$ and $$W = W_0 + BA$$ end"


def test_multiline_display_math():
    assert streamlit_markdown("\\[\nh = W_0 x\n\\]") == "$$h = W_0 x$$"


def test_currency_is_escaped_so_it_is_not_read_as_maths():
    text = "Her salary is $36,500 and lunch costs $5."
    assert streamlit_markdown(text) == r"Her salary is \$36,500 and lunch costs \$5."


def test_dollar_maths_starting_with_a_digit_is_not_mistaken_for_currency():
    text = r"about $10^4$ times fewer, rank $2$, and $10{,}000\times$"
    assert streamlit_markdown(text) == text


def test_maths_starting_with_a_digit_is_maths_by_pandocs_rule():
    # Seen live: "$2dr$" rendered as raw text because "$2" was escaped as money.
    for text in ("LoRA trains only $2dr$, not $d^2$", "$2dr$,", "$d^2$", "($2dr$)", "$2dr$."):
        assert streamlit_markdown(text) == text


def test_dollars_that_cannot_open_or_close_maths_are_currency():
    assert streamlit_markdown("costs $5 and $10") == r"costs \$5 and \$10"
    assert streamlit_markdown("from $5 to $10.") == r"from \$5 to \$10."
    # The closing "$" is the next one, and it follows a space, so "$5 ... $" is not maths.
    assert streamlit_markdown("costs $5 and $d^2$") == r"costs \$5 and $d^2$"
    # A closing "$" followed by a digit doesn't close, so neither "$" is maths.
    assert streamlit_markdown("$x$5") == r"\$x\$5"
    # An opening "$" followed by a space doesn't open.
    assert streamlit_markdown("$ x$ and $ y") == r"\$ x\$ and \$ y"
    assert streamlit_markdown("a lone $") == r"a lone \$"


def test_maths_does_not_span_a_blank_line_but_may_wrap_a_line():
    assert streamlit_markdown("$5 now\n\nand 6$") == "\\$5 now\n\nand 6\\$"
    assert streamlit_markdown("$a +\nb$") == "$a +\nb$"


def test_display_maths_and_escapes_are_kept():
    assert streamlit_markdown("$$W = W_0 + BA$$ costs $5") == r"$$W = W_0 + BA$$ costs \$5"
    assert streamlit_markdown("$$5") == r"\$\$5"  # unmatched "$$" is literal
    assert streamlit_markdown(r"$\$5 + x$ and \$5") == r"$\$5 + x$ and \$5"


def test_code_is_left_verbatim():
    text = "Run `echo $HOME $5` then:\n```bash\nexport PRICE=$5\n```\nwhich costs $5"
    expected = "Run `echo $HOME $5` then:\n```bash\nexport PRICE=$5\n```\nwhich costs \\$5"
    assert streamlit_markdown(text) == expected


def test_short_title_cuts_on_a_word_boundary():
    title = "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
    assert short_title(title, 60) == "Chain-of-Thought Prompting Elicits Reasoning in Large…"
    assert short_title("Attention Is All You Need") == "Attention Is All You Need"
    assert short_title("a" * 100, 10) == "a" * 10 + "…"  # no spaces: hard cut
    assert short_title("") == "" and short_title(None) == ""


def test_existing_dollar_maths_and_plain_text_are_untouched():
    text = r"rank $r$ is small, already \$5"
    assert streamlit_markdown(text) == text
    assert streamlit_markdown("") == "" and streamlit_markdown(None) == ""


def test_plural_and_model_label():
    from render import model_label, plural

    assert plural(1, "search round") == "1 search round" and plural(2, "search round") == "2 search rounds"
    assert plural(0, "passage") == "0 passages" and plural(3, "search", "searches") == "3 searches"
    assert model_label("groq", "openai/gpt-oss-120b") == "gpt-oss-120b via Groq"
    assert model_label("anthropic", "claude-haiku-4-5") == "claude-haiku-4-5 via Anthropic"
    assert model_label("custom", "m") == "m via custom"


def test_answer_footer_reads_naturally():
    from render import answer_footer

    one = answer_footer({"seconds": 2.5, "round": 1, "evidence_count": 1})
    assert one == "Answered in 2.5 s · 1 search round · 1 passage considered"
    two = answer_footer({"seconds": 3.4, "round": 2, "evidence_count": 9})
    assert two == "Answered in 3.4 s · 2 search rounds · 9 passages considered"
    assert "(s)" not in one + two and "chunk" not in one + two


def test_agent_steps_are_described_in_plain_english():
    from render import describe_step

    plan = describe_step({"step": "plan", "queries": ["LoRA rank"], "seconds": 0.7})
    assert plan == "Planned 1 search: LoRA rank (0.7 s)"
    assert describe_step({"step": "plan", "queries": ["a", "b"], "needs_web": True, "seconds": 1}).startswith(
        "Planned 2 searches and a web search: a; b"
    )
    assert "Couldn't plan" in describe_step({"step": "plan", "queries": ["q"], "error": "boom", "seconds": 1})
    found = describe_step({"step": "retrieve", "round": 2, "new_chunks": 1, "evidence": 6, "seconds": 0.1})
    assert found == "Search round 2: found 1 new passage; 6 in use (0.1 s)"
    retry = describe_step({"step": "grade", "sufficient": False, "missing": "the actions", "new_queries": ["x"]})
    assert retry.startswith("Missing: the actions → searching again: x")
    assert describe_step({"step": "grade", "sufficient": True, "seconds": 1}).startswith("Checked the passages")
    assert describe_step({"step": "grade", "sufficient": None, "skipped": True}).startswith("Last search round")
    assert "Couldn't check" in describe_step({"step": "grade", "sufficient": True, "error": "boom", "seconds": 1})
    answered = describe_step({"step": "generate", "sources": 8, "cited": [], "seconds": 1})
    assert answered == "Wrote the answer from 8 passages, citing none (1 s)"
    steps = [plan, found, retry, answered]
    assert not any("(s)" in s or "(es)" in s or "chunk" in s for s in steps)


def test_rate_limit_notice_is_friendly_and_points_to_the_repo():
    from render import GITHUB_URL, rate_limit_notice

    daily = rate_limit_notice(daily=True)
    assert "today's free-tier limit" in daily and "try again later" in daily and GITHUB_URL in daily
    busy = rate_limit_notice(daily=False)
    assert "try again in a minute" in busy and "today" not in busy
    for text in (daily, busy):
        assert "429" not in text and "Error" not in text
