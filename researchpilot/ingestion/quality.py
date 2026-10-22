"""Is extracted text readable, and is it English? Checked when a document is added and before passages reach the model.

ResearchPilot is English-only for now: the embedding model (bge-small-en) and the keyword tokenizer
(ASCII words) are English. Two things can go wrong with a document:

- Its text layer is corrupt. A PDF typeset with a pre-Unicode ("legacy") font stores glyph codes
  that only that font understands, so the extracted text is mojibake ("ÃÔÊ Å‖¡—SfIM§¬…", or
  letter soup such as "Hkkjr ,d fo'kky ns'k gS"). Searching or quoting it is useless and invites the
  model to guess, so such text is refused at ingest and dropped from the evidence.
- It's real text in another language. That is indexed, but the visitor is told that answers may be
  less reliable, since neither search signal is tuned for it.

Both checks are word statistics, with no model: the share of words that look like words of some
writing system (letters of one script with natural casing, a vowel in a Latin word, no symbols
inside), and the share that are common English function words. Calibrated on every chunk of the
sample papers, real text in 20 languages and synthetic corrupted text layers (see CLAUDE.md).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Words are separated by spaces, brackets and quotes, or by the punctuation of scripts written without
# spaces (CJK) and the sentence marks of Devanagari and Arabic, so a Chinese sentence is a run of words
# rather than one long "word" with commas inside.
_TOKEN = re.compile(r"[^\s()\[\]{}<>\"“”«»\u3000-\u303f\uff00-\uffef\u0964\u0965\u060c\u061b\u061f]+")
# Punctuation that may surround a word: stripped from both ends before the word is judged.
_EDGE = "\"'`“”‘’«»„‚‹›()[]{}<>.,;:!?¡¿…—–-•·*#%/\\|=+~^"
# ASCII punctuation allowed inside a word: "don't", "GPT-3", "e.g.", "and/or", "R&D", "x_1", URLs, and
# what PDF tables glue together ("32,li", "rq=rv"). Other symbols inside a word ("Å‖¡—", "§¬") mark mojibake.
_JOINERS = frozenset("'’-‐‑.,:;/&_+=%?#~@")
_VOWELS = frozenset("aeiouy")
# Common English function words. Words that are just as common in other languages written in the Latin
# alphabet ("a", "in", "die", "an", "was", "on", "for", "is", "to") are left out on purpose.
# fmt: off
_ENGLISH = frozenset([
    "the", "of", "and", "that", "with", "this", "are", "from", "which", "have", "has", "were", "been", "they",
    "their", "these", "those", "would", "should", "could", "there", "between", "each", "both", "into", "than",
    "such", "however", "because", "through", "about", "after", "before", "while", "what", "when", "how", "who",
    "them", "our", "your", "we", "you", "he", "she", "his", "her", "it", "its", "be", "by", "not", "but", "or",
    "as", "at", "can", "more", "other", "only", "also",
])
# fmt: on

MIN_WORDS = 20  # with fewer words and letters than this, there is too little to judge: the text is taken as it is
MIN_LETTERS = 60  # (Chinese and Japanese have few spaces, so few "words": 60 characters are enough)
# Share of words that look like words. Whole documents: the sample papers 0.96 or more, real text in 20 languages
# 0.97 or more (as a PDF delivers it, NFKC-normalised), Markdown and source code 0.88 or more; corrupted text
# layers 0.67 or less.
DOCUMENT_MIN_CLEAN = 0.75
PASSAGE_MIN_CLEAN = 0.5  # one chunk is noisier (a table of hyperparameters scores 0.60), so a lower bar
MIN_ENGLISH = 0.05  # English: 0.13 (a resume) to 0.29; other Latin-script languages: 0.015 or less
MIN_LATIN = 0.7  # share of letters in the Latin alphabet, below which a text isn't English


@dataclass(frozen=True)
class TextCheck:
    readable: bool
    english: bool
    words: int  # tokens with at least one letter
    stats: dict = field(default_factory=dict, compare=False)


def _letter(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "LM"


_EAST_ASIAN = {"CJK", "HIRAGANA", "KATAKANA", "HALFWIDTH", "FULLWIDTH"}


def _script(ch: str) -> str:
    """The writing system of a letter, from its Unicode name ("LATIN", "GREEK", "DEVANAGARI"...)."""
    try:
        script = unicodedata.name(ch).split(" ", 1)[0]
    except ValueError:
        return "?"
    return "EAST_ASIAN" if script in _EAST_ASIAN else script


def _latin(ch: str) -> bool:
    return _script(ch) == "LATIN"


def _has_vowel(letters: list[str]) -> bool:
    return any(unicodedata.normalize("NFD", ch)[0].lower() in _VOWELS for ch in letters)


def assess(text: str, min_clean: float = DOCUMENT_MIN_CLEAN) -> TextCheck:
    """Word statistics of `text` and the two verdicts. Too short a text to judge passes both."""
    words = clean = garbled = latin_words = vowelless = odd_case = english = 0
    letters = latin_letters = 0
    for token in _TOKEN.findall(text or ""):
        core = token.strip(_EDGE)
        alpha = [ch for ch in core if unicodedata.category(ch)[0] == "L"]
        if not alpha:  # numbers, punctuation
            continue
        words += 1
        letters += len(alpha)
        if any(not (_letter(ch) or ch.isdigit() or ch in _JOINERS) for ch in core):
            garbled += 1
            continue
        # PDF text is NFKC-normalised, which turns much of a legacy font's Latin-1 soup into letters: "µ" into
        # Greek "μ", "¸" into a combining cedilla. A real word neither starts with a combining mark nor mixes
        # alphabets ("W8μû"). Chinese and Japanese mix their scripts, and run Latin words in without a space.
        scripts = {_script(ch) for ch in alpha}
        if unicodedata.category(core[0])[0] == "M" or (len(scripts) > 1 and "EAST_ASIAN" not in scripts):
            garbled += 1
            continue
        if not all(_latin(ch) for ch in alpha):  # another script: no casing or vowel rules to apply
            clean += 1
            continue
        latin_words += 1
        latin_letters += len(alpha)
        english += core.lower() in _ENGLISH
        # Mojibake in the Latin alphabet: "vkSj" (a capital after a lower-case start; "iPhone" is rare in
        # prose) and "Hkkjr" (no vowel; acronyms such as "NLP" are all capitals and don't count).
        odd = alpha[0].islower() and any(ch.isupper() for ch in alpha[1:])
        no_vowel = len(alpha) >= 2 and not core.isupper() and not _has_vowel(alpha)
        odd_case += odd
        vowelless += no_vowel
        clean += not (odd or no_vowel)
    n = words or 1
    stats = {
        "words": words,
        "clean": clean / n,
        "garbled": garbled / n,
        "vowelless": vowelless / (latin_words or 1),
        "odd_case": odd_case / (latin_words or 1),
        "latin": latin_letters / (letters or 1),
        "english": english / n,
    }
    if words < MIN_WORDS and letters < MIN_LETTERS:
        return TextCheck(readable=True, english=True, words=words, stats=stats)
    return TextCheck(
        readable=stats["clean"] >= min_clean,
        english=stats["latin"] >= MIN_LATIN and stats["english"] >= MIN_ENGLISH,
        words=words,
        stats=stats,
    )


def readable_passage(text: str) -> bool:
    """False for a retrieved passage that is clearly mojibake (a lower bar than for a whole document)."""
    return assess(text, PASSAGE_MIN_CLEAN).readable
