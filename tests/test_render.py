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


# --------------------------------------------------------------------------- citation chips
CITES = [
    {"n": 1, "title": "LoRA", "source": "https://arxiv.org/pdf/2106.09685", "page": 4},
    {"n": 2, "title": 'Notes "draft"', "source": "notes.txt", "page": None},
]


def _visible_and_copy_text(html_text: str) -> tuple[str, str]:
    """What a reader sees (hidden copy-only spans removed) and what copying gives (all text)."""
    import re

    visible = re.sub(r'<span class="rp-cite-copy">.*?</span>', "", html_text)
    strip = lambda s: re.sub(r"<[^>]+>", "", s)  # noqa: E731
    return strip(visible), strip(html_text).replace("&lt;", "<")


def test_citations_become_chips_that_link_to_their_source_cards():
    from render import cite_chips

    out = cite_chips("Frozen weights [1][2], see [1, 2].", {1, 2}, "m3", {1: "LoRA"})
    assert out.count('class="rp-cite"') == 4
    assert 'href="#m3-1"' in out and 'href="#m3-2"' in out and 'title="Source 1: LoRA"' in out
    visible, copied = _visible_and_copy_text(out)
    assert visible == "Frozen weights 12, see 12."  # chips only
    assert copied == "Frozen weights [1][2], see [1, 2]."  # copying keeps the plain-text form


def test_ranges_invalid_numbers_and_links_are_handled():
    from render import cite_chips

    out = cite_chips("Range [1-3], unknown [7], mixed [2, 9], a link [1](https://x.org).", {1, 2}, "p")
    assert '<a class="rp-cite" href="#p-1" title="Source 1">1-3</a>' in out  # a range links to its first source
    assert "unknown [7]" in out  # not a source: left as written
    assert 'href="#p-2"' in out and ">9</a>" not in out and "</span>9<span" in out  # 9 stays plain text
    assert "[1](https://x.org)" in out  # a Markdown link is not a citation
    assert _visible_and_copy_text(out)[1] == "Range [1-3], unknown [7], mixed [2, 9], a link [1](https://x.org)."


def test_answer_html_keeps_maths_and_code_and_neutralises_html():
    from render import answer_html

    out = answer_html(
        "Use $W[1]$ and `x[1]` here [1]. <script>alert(1)</script> <img src=x onerror=y> a < b", CITES, "m0"
    )
    assert "$W[1]$" in out and "`x[1]`" in out  # untouched: not chips inside maths or code
    assert out.count('class="rp-cite"') == 1
    assert "<script" not in out and "<img" not in out and "&lt;script>" in out and "&lt;img" in out
    assert "a < b" in out  # a comparison isn't a tag
    assert 'title="Source 2: Notes &quot;draft&quot;"' not in out  # source 2 isn't cited here
    assert "Notes &quot;draft&quot;" in answer_html("See [2].", CITES, "m0")  # titles are escaped in attributes


# --------------------------------------------------------------------------- snippets, URLs, cards
def test_snippet_starts_at_the_sentence_that_best_matches_the_answer():
    from render import best_snippet, citing_sentences

    passage = " ".join(["Background sentence number one about many unrelated things."] * 6) + (
        " LoRA freezes the pretrained weights and trains rank decomposition matrices. "
        + " ".join(["A closing remark about evaluation and other topics entirely."] * 6)
    )
    answer = "It freezes the weights [1]. RAG retrieves passages [2]."
    context = citing_sentences(answer, 1)
    assert context == "It freezes the weights [1]."
    snippet = best_snippet(passage, context)
    assert snippet.startswith("…LoRA freezes the pretrained weights") and snippet.endswith("…")
    assert len(snippet) <= 252
    assert " " in snippet and not snippet[1:-1].endswith(" ")  # cut on a word boundary


def test_snippet_falls_back_to_the_start_and_keeps_short_passages_whole():
    from render import best_snippet, citing_sentences

    passage = "First sentence here. " + "Filler words go on and on. " * 20
    snippet = best_snippet(passage, "Something about quantum chromodynamics.")
    assert snippet.startswith("First sentence here.") and snippet.endswith("…") and len(snippet) <= 251
    assert best_snippet("  Short\npassage.  ", "anything") == "Short passage."
    # A sentence near the end still gets a full window that contains it.
    tail = best_snippet("Filler words go on and on. " * 20 + "Zebras migrate.", "zebras")
    assert tail.startswith("…") and tail.endswith("Zebras migrate.") and len(tail) >= 200
    assert citing_sentences("No citations here.", 1) == "No citations here."


def test_long_urls_are_shortened_in_passages():
    from render import short_url, shorten_urls

    url = "https://www.example.com/a/very/long/path/that/goes/on/and/on/forever.html"
    assert short_url(url) == "example.com/a/very/long/path/that/goes/…" and len(short_url(url)) == 40
    assert short_url("https://arxiv.org/abs/2106.09685/") == "arxiv.org/abs/2106.09685"
    assert shorten_urls(f"See {url}, then stop.") == "See example.com/a/very/long/path/that/goes/…, then stop."
    linked = shorten_urls(f"Code: {url}.", markdown=True)
    assert linked == f"Code: [example.com/a/very/long/path/that/goes/…]({url})."
    assert shorten_urls("see https://x.org/my_file_name.") == "see x.org/my_file_name."
    assert (
        shorten_urls("see https://x.org/my_file_name", markdown=True)
        == r"see [x.org/my\_file\_name](https://x.org/my_file_name)"
    )
    assert shorten_urls("") == "" and shorten_urls(None) == ""


def test_pdf_links_open_arxiv_papers_at_the_cited_page():
    from render import pdf_page_url

    assert pdf_page_url("https://arxiv.org/pdf/2106.09685", 4) == "https://arxiv.org/pdf/2106.09685#page=4"
    assert pdf_page_url("https://arxiv.org/abs/2005.11401v4", 2) == "https://arxiv.org/pdf/2005.11401v4#page=2"
    assert pdf_page_url("https://arxiv.org/pdf/2106.09685.pdf", 1) == "https://arxiv.org/pdf/2106.09685#page=1"
    assert pdf_page_url("https://arxiv.org/pdf/2106.09685", None) is None
    assert pdf_page_url("https://example.com/paper.pdf", 3) is None
    assert pdf_page_url("notes.pdf", 3) is None


# --------------------------------------------------------------------------- ui-v3 text pieces
def test_steps_summary_and_questions_left_read_naturally():
    from render import questions_left, steps_summary

    assert steps_summary({"round": 1, "evidence_count": 5, "seconds": 4.4}) == "Searched 1 round · 5 passages · 4.4 s"
    assert steps_summary({"round": 2, "evidence_count": 1, "seconds": 9}) == "Searched 2 rounds · 1 passage · 9 s"
    assert [questions_left(n) for n in (4, 1, 0, -1)] == [
        "4 questions left",
        "1 question left",
        "No questions left",
        "No questions left",
    ]


def test_how_it_works_explains_the_loop_and_the_selected_method():
    from render import how_it_works

    labels = {"hybrid": "Hybrid (recommended)", "bm25": "Keyword (BM25)"}
    help_text = {"hybrid": "both", "bm25": "words"}
    text = how_it_works(labels, help_text, "bm25", 2)
    assert "**Plan**" in text and "up to 2 rounds" in text and "- **Keyword (BM25)**: words" in text
    assert "Currently selected: **Keyword (BM25)**" in text and "#evaluation" in text
    assert "Retrying is switched off" in how_it_works(labels, help_text, "hybrid", 1)


def test_small_caps_splits_are_rejoined_only_where_safe():
    from render import fix_small_caps

    text = "4.2 A PPLYING LORA TO TRANSFORMER, 1 I NTRODUCTION, 2 R ELATED WORK, E XPERIMENTS"
    assert fix_small_caps(text) == "4.2 APPLYING LORA TO TRANSFORMER, 1 INTRODUCTION, 2 RELATED WORK, EXPERIMENTS"
    for kept in ("A NEW METHOD", "A STUDY OF", "I AM HERE", "A PROMPT", "See Table A here", "W_0 + B A x"):
        assert fix_small_caps(kept) == kept
    assert fix_small_caps("") == "" and fix_small_caps(None) == ""


def test_source_card_labels_are_escaped_markdown():
    from render import escape_markdown, source_button_label

    assert escape_markdown("Use *bold* [12] $5 :red[x] a_b <b>") == r"Use \*bold\* (12) \$5 \:red(x) a\_b \<b\>"
    label = source_button_label({"n": 2, "title": "LoRA: Low_Rank", "page": 4}, "The *matrix* [3] costs $5.")
    assert label == r":gray[2] **LoRA\: Low\_Rank** :gray[p. 4]  " + "\n" + r":gray[The \*matrix\* (3) costs \$5.]"
    no_page = source_button_label({"n": 1, "title": "notes", "page": None}, "Text.")
    assert no_page == ":gray[1] **notes**  \n:gray[Text.]"


def test_document_lists_show_short_names_with_full_escaped_titles():
    from render import doc_list_html

    out = doc_list_html([("A <b>", 'A <b>"bold"</b> title', "https://arxiv.org/pdf/1"), ("notes", "notes", None)])
    assert out.startswith('<ul class="rp-docs">') and out.count("<li") == 2
    assert 'href="https://arxiv.org/pdf/1" title="A &lt;b&gt;&quot;bold&quot;&lt;/b&gt; title"' in out
    assert '<span class="rp-doc-t">A &lt;b&gt;</span>' in out
    assert 'rel="noopener noreferrer"' in out and '<span class="rp-doc-x" aria-hidden="true">↗</span>' in out
    assert '<li title="notes"><span class="rp-doc-t">notes</span></li>' in out
    assert "<b>" not in out and "\n" not in out  # one line: an HTML block in Markdown must not be split


def test_short_names_use_the_part_before_a_colon_or_a_word_boundary():
    from render import SHORT_NAME_CHARS, short_name

    assert short_name("LoRA: Low-Rank Adaptation of Large Language Models") == "LoRA"
    assert short_name("ReAct: Synergizing Reasoning and Acting in Language Models") == "ReAct"
    assert short_name("Attention Is All You Need") == "Attention Is All You Need"
    assert short_name("Chain-of-Thought Prompting Elicits Reasoning in Large Language Models") == (
        "Chain-of-Thought Prompting…"
    )
    assert short_name("Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks") == (
        "Retrieval-Augmented Generation…"
    )
    long_head = "A heading far too long to be a short name: the rest"
    assert short_name(long_head).endswith("…") and len(short_name(long_head)) <= SHORT_NAME_CHARS + 1
    assert short_name(": leading colon") == ": leading colon" and short_name("") == ""


def test_the_live_status_says_what_the_agent_does_next():
    from render import next_activity

    assert next_activity({"step": "plan"}) == "Searching the documents…"
    assert next_activity({"step": "retrieve"}) == "Checking the passages…"
    assert next_activity({"step": "grade", "new_queries": ["x"]}) == "Searching again…"
    assert next_activity({"step": "grade", "sufficient": True}) == "Writing the answer…"
    assert next_activity({"step": "grade", "skipped": True}) == "Writing the answer…"
    assert next_activity({"step": "generate"}) is None


def test_a_url_glued_to_the_previous_word_gets_a_space():
    from render import shorten_urls

    assert shorten_urls("GPT-2 athttps://github.com/microsoft/LoRA.") == "GPT-2 at github.com/microsoft/LoRA."
    linked = shorten_urls("athttps://x.org/a", markdown=True)
    assert linked == "at [x.org/a](https://x.org/a)"
    assert shorten_urls("(https://x.org)") == "(x.org)"  # punctuation before a URL is left alone
