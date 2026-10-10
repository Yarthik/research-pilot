import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from render import streamlit_markdown  # noqa: E402


def test_latex_delimiters_become_streamlit_math():
    text = r"B \( \in \mathbb{R}^{d\times r} \) and \[ W = W_0 + BA \] end"
    assert streamlit_markdown(text) == r"B $\in \mathbb{R}^{d\times r}$ and $$W = W_0 + BA$$ end"


def test_multiline_display_math():
    assert streamlit_markdown("\\[\nh = W_0 x\n\\]") == "$$h = W_0 x$$"


def test_currency_is_escaped_so_it_is_not_read_as_maths():
    text = "Her salary is $36,500 and lunch costs $5."
    assert streamlit_markdown(text) == r"Her salary is \$36,500 and lunch costs \$5."


def test_existing_dollar_maths_and_plain_text_are_untouched():
    text = r"rank $r$ is small, already \$5"
    assert streamlit_markdown(text) == text
    assert streamlit_markdown("") == "" and streamlit_markdown(None) == ""
