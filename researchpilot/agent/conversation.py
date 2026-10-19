"""Conversation context for the agent: chat history, attached files and questions that name no document.

The agent answers one question at a time. A follow-up such as "What is it about?" only makes sense
with the conversation before it, so when there is history the condense step (nodes.condense) rewrites
it into a standalone question first. Without history or attachments none of this runs, except a
cheap word check: a question with no subject at all ("What is it about?") gets a list of the
documents to choose from instead of a search that would return an arbitrary paper.
"""

from __future__ import annotations

import re

from ..ingestion.loaders import title_case_if_all_caps
from ..retrieval.bm25 import tokenize

MAX_HISTORY_MESSAGES = 6  # the last three exchanges
MAX_MESSAGE_CHARS = 600  # per message: the gist of an answer, not all of it
ROLES = {"user": "User", "assistant": "Assistant"}


def clean_history(history: list | None) -> list[dict]:
    """The last few well-formed messages, each shortened, so the condense prompt stays small."""
    turns = []
    for turn in history or []:
        role = turn.get("role") if isinstance(turn, dict) else getattr(turn, "role", None)
        content = turn.get("content") if isinstance(turn, dict) else getattr(turn, "content", None)
        if role in ROLES and isinstance(content, str) and content.strip():
            text = " ".join(content.split())
            if len(text) > MAX_MESSAGE_CHARS:
                text = text[:MAX_MESSAGE_CHARS].rsplit(" ", 1)[0] + " …"
            turns.append({"role": role, "content": text})
    return turns[-MAX_HISTORY_MESSAGES:]


def format_history(history: list[dict]) -> str:
    return "\n".join(f"{ROLES[turn['role']]}: {turn['content']}" for turn in history) or "(no earlier messages)"


def format_attachments(attachments: list[dict]) -> str:
    if not attachments:
        return "No files were attached with the latest message."
    names = "; ".join(
        f'"{item["source"]}"' + (f' (title: "{item["title"]}")' if item.get("title") not in (None, "") else "")
        for item in attachments
    )
    return f"Newly attached with the latest message: {names}"


# --------------------------------------------------------------------------- questions that name no document
_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


REFERENCES = _words("it its itself this that these those they them their he she him her his here there one ones")
SINGULAR_DOCS = _words("document doc paper file pdf article text attachment upload resume cv report")
PLURAL_DOCS = _words("documents docs papers files pdfs articles texts sources library collection corpus")
# Words that ask for a summary or description without naming a subject.
GENERIC = _words(
    """about say says said tell me us summarize summarise summary summarized summarised overview explain
    describe description mean means meaning main point points idea ideas topic topics subject key gist
    contain contains cover covers covered discuss discusses discussed talk talks give brief briefly please
    you your i we my our all whole so really exactly anything something s t d ll m re ve go goes going
    happen happens work works get got show list available can could would should may might have has had
    any some each every other ask know need want like"""
)


def content_words(question: str) -> list[str]:
    """The words that name a subject: everything but stop words, references, document nouns and asks."""
    ignored = REFERENCES | SINGULAR_DOCS | PLURAL_DOCS | GENERIC
    return [w for w in tokenize(question or "") if w not in ignored]


def is_vague(question: str) -> bool:
    """True when the question names no subject ("What is it about?", "Summarize this paper").

    Deliberately narrow: any other word ("What is attention?", "Who are the authors?") makes the
    question clear, so a question that names anything is searched exactly as before.
    """
    return not content_words(question)


def refers_to_context(question: str) -> bool:
    """True when the question points at something ("it", "this file") or names no subject."""
    words = set(_WORD.findall((question or "").lower()))
    return bool(words & (REFERENCES | SINGULAR_DOCS | PLURAL_DOCS)) or is_vague(question)


def asks_about_library(question: str) -> bool:
    """A vague question about the documents as a whole ("What are these papers about?")."""
    words = set(_WORD.findall((question or "").lower()))
    return bool(words & PLURAL_DOCS) or bool(words & {"they", "them", "these", "those"})


def short_name(title: str, limit: int = 30) -> str:
    head, colon, _ = title.partition(":")
    return head.strip() if colon and head.strip() and len(head.strip()) <= limit else title


def library_answer(question: str, documents: list[dict], private: list[dict]) -> tuple[str, str]:
    """(kind, answer) for a vague question: what the library holds, or which document is meant."""
    shared = [title_case_if_all_caps(" ".join((d.get("title") or d["source"]).split())) for d in documents]
    own = [" ".join((d.get("title") or d["source"]).split()) for d in private]
    lines = [f"- {title}" for title in shared]
    if own:
        lines += [f"- {title} (your file)" for title in own]
    listing = "\n".join(lines)
    example = short_name(own[0]) if own else min((short_name(title) for title in shared), key=len)
    tip = f'Ask about one of them by name, for example "What is {example} about?"'
    if asks_about_library(question):
        count = len(shared) + len(own)
        noun = "document" if count == 1 else "documents"
        return "library", f"I can search {count} {noun}:\n\n{listing}\n\n{tip}"
    return "ambiguous", f"Which document do you mean? I can search these:\n\n{listing}\n\n{tip}"
