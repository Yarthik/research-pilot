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


def test_rate_limit_notice_is_friendly_and_points_to_the_repo():
    from render import GITHUB_URL, rate_limit_notice

    daily = rate_limit_notice(daily=True)
    assert "today's free-tier limit" in daily and "try again later" in daily and GITHUB_URL in daily
    busy = rate_limit_notice(daily=False)
    assert "try again in a minute" in busy and "today" not in busy
    for text in (daily, busy):
        assert "429" not in text and "Error" not in text
