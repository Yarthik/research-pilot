"""Readable text and English text (ingestion/quality.py): corrupted text layers are refused, other languages flagged."""

import random

import pytest

from researchpilot.ingestion.quality import assess, readable_passage

ENGLISH = (
    "We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable "
    "rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of "
    "trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the "
    "number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times."
)
RESUME = (
    "Alex Morgan - Frontend Engineer - alex.morgan@example.com - Lisbon. Summary: Frontend engineer with six years "
    "of experience building web applications with React, TypeScript and Node.js. Experience: Senior Frontend "
    "Engineer, Brightline Analytics (2022 - present). Rebuilt the customer dashboard in React 18 with hooks and "
    "React Query; page loads 40% faster. Skills: React, Next.js, TypeScript, GraphQL, Jest, Playwright, AWS."
)
# A hyperparameter table as pypdf extracts it: the noisiest passage in the sample papers.
TABLE = (
    "Method Hyperparameters # Trainable Parameters WikiSQL MNLI-m Fine-Tune - 175B 73.8 89.5 PrefixEmbed lp = 32,li "
    "= 8 0.4 M 55.9 84.9 lp = 64,li = 8 0.9 M 58.7 88.1 lp = 128,li = 8 1.7 M 60.6 88.0 PrefixLayer lp = 2,li = 2 "
    "5.1 M 68.5 89.2 lp = 8,li = 0 10.1 M 69.8 88.2 AdapterH r = 1 7.1 M 71.9 89.8 r = 4 21.2 M 73.2 91.0 LoRA+PE "
    "rq =rv = 8,lp = 8,li = 4 37.8 M 75.0 91.4"
)
# The Universal Declaration of Human Rights: the start of the preamble and Article 1.
GERMAN = (
    "Da die Anerkennung der angeborenen Würde und der gleichen und unveräußerlichen Rechte aller Mitglieder der "
    "Gemeinschaft der Menschen die Grundlage von Freiheit, Gerechtigkeit und Frieden in der Welt bildet. Alle "
    "Menschen sind frei und gleich an Würde und Rechten geboren. Sie sind mit Vernunft und Gewissen begabt und "
    "sollen einander im Geist der Brüderlichkeit begegnen."
)
SPANISH = (
    "Considerando que la libertad, la justicia y la paz en el mundo tienen por base el reconocimiento de la "
    "dignidad intrínseca y de los derechos iguales e inalienables de todos los miembros de la familia humana. "
    "Todos los seres humanos nacen libres e iguales en dignidad y derechos y, dotados como están de razón y "
    "conciencia, deben comportarse fraternalmente los unos con los otros."
)
HINDI = (
    "चूंकि मानव परिवार के सभी सदस्यों के जन्मजात गौरव और समान तथा अविच्छिन्न अधिकार की स्वीकृति ही विश्व-शान्ति, "
    "न्याय और स्वतन्त्रता की बुनियाद है। सभी मनुष्यों को गौरव और अधिकारों के मामले में जन्मजात स्वतन्त्रता और "
    "समानता प्राप्त है। उन्हें बुद्धि और अन्तरात्मा की देन प्राप्त है और परस्पर उन्हें भाईचारे के भाव से बर्ताव "
    "करना चाहिए।"
)
CHINESE = (
    "人人生而自由，在尊严和权利上一律平等。他们赋有理性和良心，并应以兄弟关系的精神相对待。人人有资格享有本宣言所载的一切"
    "权利和自由，不分种族、肤色、性别、语言、宗教、政治或其他见解、国籍或社会出身、财产、出生或其他身分等任何区别。"
)
# What a PDF set in a legacy Devanagari font extracts as: the owner's sample, and letter soup (the Kruti Dev style).
SYMBOL_SOUP = " ".join(["ÃÔÊ Å‖¡—SfIM§¬… ¶µ±Ç ÐÞ§ßð þ¿Æ Øæ ø¶ µ±°ºª «»Ç"] * 6)
LETTER_SOUP = (
    "Hkkjr ,d fo'kky ns'k gSA ;gk¡ vusd /keZ] Hkk\"kk,¡ vkSj tkfr;k¡ gSaA Hkkjr dh jkt/kkuh ubZ fnYyh gSA "
    "Hkkjr dk lafo/kku fo'o dk lcls cM+k fyf[kr lafo/kku gSA ;g ns'k xaxk vkSj ;equk tSlh ufn;ksa dk ns'k gSA"
)


def legacy_font(text: str, seed: int = 0) -> str:
    """Each character mapped to an arbitrary WinAnsi character, as a font without a Unicode map comes out."""
    rng = random.Random(seed)
    alphabet = [chr(c) for c in range(0x21, 0x7F) if chr(c) not in "()\\"] + [chr(c) for c in range(0xA1, 0x100)]
    table: dict[str, str] = {}
    return "".join(ch if ch.isspace() else table.setdefault(ch, rng.choice(alphabet)) for ch in text)


@pytest.mark.parametrize("text", [ENGLISH, RESUME, ENGLISH + " " + TABLE])
def test_english_text_is_readable_and_english(text):
    check = assess(text)
    assert check.readable and check.english, check.stats


def test_a_table_passage_is_kept():
    assert readable_passage(TABLE)


@pytest.mark.parametrize("text", [GERMAN, SPANISH, HINDI, CHINESE], ids=["de", "es", "hi", "zh"])
def test_other_languages_are_readable_but_not_english(text):
    check = assess(text)
    assert check.readable and not check.english, check.stats
    assert readable_passage(text)


@pytest.mark.parametrize(
    "text",
    [
        SYMBOL_SOUP,
        LETTER_SOUP,
        legacy_font(HINDI),
        legacy_font(HINDI, seed=1),
        legacy_font(ENGLISH),
        HINDI.encode("utf-8").decode("cp1252", errors="replace"),  # UTF-8 read as Windows-1252
    ],
    ids=["symbols", "letters", "legacy-hi", "legacy-hi-2", "legacy-en", "double-utf8"],
)
def test_corrupted_text_layers_are_unreadable(text):
    assert not assess(text).readable, assess(text).stats
    assert not readable_passage(text)


def test_short_text_is_taken_as_it_is():
    assert assess("ÃÔÊ Å‖¡—").readable  # too little to judge (a heading, a figure label)
    assert assess("").readable
