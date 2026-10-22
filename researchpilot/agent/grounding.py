"""Keeping an answer to what its passages say.

The answer prompt asks the model to use only the numbered passages, but a prompt is a request, not a
guarantee: given passages it can't read, a model may answer from a document's title and its own
knowledge and cite the passages anyway. So the answer step also checks, without another model call:

- Unreadable passages never reach it: the retrieve step drops them (see ingestion/quality.py), and
  with nothing readable left the agent says so instead of answering.
- Every citation must point to a passage that shares words with the sentence it is attached to.
  A marker whose passage shares nothing with its sentence is removed; a sentence left with no
  supported citation is removed; if no cited sentence is left, the answer becomes an abstention.

The word check is lenient on purpose: two shared content words, or one when the sentence has only one
or cites several passages (each may back one part: "the tern [1] and the plover [2]"). Words sharing
their first five letters match ("initialized", "initialization"), and reporting words ("the guide
describes", "the paper states") don't count. On the answer eval's answers every real citation passes,
while 101 of 111 citations re-pointed at a garbled copy of their passage are caught (the rest are garbled
text that the passage check takes for another language). It catches citations with no textual link to
their sentence, not a mix-up between two passages on the same topic: those share words too, and a
stricter rule would also cut correct citations. A garbled passage supports nothing, and the check is
skipped for passages in another language, which an English sentence can't share words with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..ingestion.quality import PASSAGE_MIN_CLEAN, assess
from ..retrieval.bm25 import tokenize
from .citations import parse_citations

# One citation group as the parser reads it: [1], [1, 2], [2-4].
_CITE = re.compile(r"\[\s*\d+(?:\s*[,\-–—]\s*\d+)*\s*\]")
# The end of a sentence: . ! or ? (with any citation markers right after it) and white space, or a line break.
_BOUNDARY = re.compile(r"(?<=[.!?])(?:\s*\[\s*\d+(?:\s*[,\-–—]\s*\d+)*\s*\])*(?=\s)|\n")
# Words that say nothing about the content, so sharing them proves nothing: generic research words, and
# the words an answer uses to report what a document says.
# fmt: off
_GENERIC = frozenset([
    "also", "both", "such", "use", "used", "uses", "using", "based", "paper", "papers", "work", "works", "study",
    "approach", "method", "methods", "model", "models", "result", "results", "show", "shows", "shown", "table",
    "figure", "section", "authors", "author", "propose", "proposed", "provide", "provides", "one", "two", "first",
    "second", "new", "different", "many", "much", "well", "like", "see", "according", "source", "sources", "passage",
    "passages", "document", "documents", "guide", "text", "article", "report", "file", "describe", "describes",
    "described", "mention", "mentions", "mentioned", "say", "says", "said", "state", "states", "stated", "explain",
    "explains", "discuss", "discusses", "discussed", "note", "notes", "noted", "list", "lists", "includes", "include",
])
# fmt: on
STEM = 5  # words sharing their first five letters count as the same word


@dataclass
class Checked:
    answer: str
    removed: list[dict] = field(default_factory=list)  # {"n": passage number, "sentence": text}, one per marker cut
    dropped: list[str] = field(default_factory=list)  # sentences removed because nothing they cite supports them


def _stems(words) -> set[str]:
    return {w[:STEM] if len(w) > STEM and not w.isdigit() else w for w in words}


def supports(sentence: str, passage: str, part: bool = False) -> bool:
    """Whether `passage` shares enough words with `sentence` to possibly support it.

    `part`: the sentence cites several passages, so this one may support only part of it (one shared word).
    """
    wanted = _stems(t for t in tokenize(_CITE.sub(" ", sentence)) if t not in _GENERIC)
    if not wanted:
        return True  # nothing to check ("See [1].")
    return len(wanted & _stems(tokenize(passage))) >= (1 if part else min(2, len(wanted)))


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence or line of `text`, each with the citation markers that close it."""
    spans, start = [], 0
    for match in _BOUNDARY.finditer(text):
        spans.append((start, match.end()))
        start = match.end()
    spans.append((start, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]


def _without(sentence: str, numbers: set[int]) -> str:
    """The sentence with `numbers` taken out of its markers ("[1][3]" without 3 -> "[1]", "[2, 3]" -> "[2]")."""

    def fix(match: re.Match) -> str:
        kept = [n for n in parse_citations(match.group(0)) if n not in numbers]
        return f"[{', '.join(map(str, kept))}]" if kept else ""

    edited = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", _CITE.sub(fix, sentence))  # "frozen [3] and" -> "frozen and"
    return re.sub(r"[ \t]+(?=[.,;:!?)]|$)", "", edited)  # "frozen [3]." -> "frozen."


def _kind(text: str) -> str:
    """ "garbled" (supports nothing), "other" (another language: not word-checked) or "english"."""
    check = assess(text, PASSAGE_MIN_CLEAN)
    return "garbled" if not check.readable else "english" if check.english else "other"


def _supported(sentence: str, text: str, kind: str, part: bool = False) -> bool:
    return kind == "other" or (kind == "english" and supports(sentence, text, part))


def check_listed(answer: str, evidence: list[dict], listed: list[int]) -> list[int]:
    """The passages a model listed without citing them inline, minus those that share no words with the answer."""
    keep = []
    for n in dict.fromkeys(listed):
        if 1 <= n <= len(evidence):
            text = evidence[n - 1]["text"]
            if _supported(answer, text, _kind(text)):
                keep.append(n)
    return keep


def check_citations(answer: str, evidence: list[dict]) -> Checked:
    """Remove the citations, then the sentences, that the cited passages don't support (see the module docstring)."""
    kinds = {}
    removed, dropped, edits = [], [], []
    for start, end in sentence_spans(answer):
        sentence = answer[start:end]
        cited = [n for n in parse_citations(sentence) if 1 <= n <= len(evidence)]
        bad = set()
        for n in cited:
            text = evidence[n - 1]["text"]
            kinds.setdefault(n, _kind(text))
            if not _supported(sentence, text, kinds[n], part=len(cited) > 1):
                bad.add(n)
        if not bad:
            continue
        removed += [{"n": n, "sentence": sentence.strip()} for n in sorted(bad)]
        if bad == set(cited):
            dropped.append(sentence.strip())
            edits.append((start, end, ""))
        else:
            edits.append((start, end, _without(sentence, bad)))
    for start, end, text in reversed(edits):
        answer = answer[:start] + text + answer[end:]
    if edits:  # a dropped paragraph can leave extra blank lines; an untouched answer keeps its exact layout
        answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    return Checked(answer=answer, removed=removed, dropped=dropped)
