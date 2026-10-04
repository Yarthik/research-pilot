"""Sentence-aware chunking with overlap, keeping page numbers for citations."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .loaders import Document


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    title: str
    page: int | None
    chunk_index: int

    def metadata(self) -> dict:
        # ChromaDB metadata values can't be None, so -1 means "no page".
        return {
            "source": self.source,
            "title": self.title,
            "page": self.page if self.page is not None else -1,
            "chunk_index": self.chunk_index,
        }

    @classmethod
    def from_metadata(cls, chunk_id: str, text: str, meta: dict) -> Chunk:
        page = meta.get("page", -1)
        return cls(
            id=chunk_id,
            text=text,
            source=meta.get("source", ""),
            title=meta.get("title", ""),
            page=None if page in (-1, None) else int(page),
            chunk_index=int(meta.get("chunk_index", 0)),
        )


def clean_text(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)  # re-join words hyphenated across lines
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single line breaks -> spaces (PDF wrapping)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    if " " in tail:  # start the overlap on a word boundary
        tail = tail[tail.index(" ") + 1 :]
    return tail.strip()


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Pack sentences into chunks of about `size` characters, carrying `overlap` characters forward."""
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]

    units = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    chunks: list[str] = []
    current = ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        while len(unit) > size:  # hard-split very long sentences
            if current:
                chunks.append(current)
                current = ""
            chunks.append(unit[:size])
            unit = unit[size - overlap :]
        if not current:
            current = unit
        elif len(current) + 1 + len(unit) <= size:
            current = f"{current} {unit}"
        else:
            chunks.append(current)
            # carry overlap forward, but never let a chunk grow past `size`
            tail = _overlap_tail(current, min(overlap, size - len(unit) - 1))
            current = f"{tail} {unit}" if tail else unit
    if current:
        chunks.append(current)
    return chunks


def chunk_document(doc: Document, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page, text in doc.pages:
        for piece in split_text(text, size, overlap):
            index = len(chunks)
            chunk_id = hashlib.sha1(f"{doc.source}::{page}::{index}".encode()).hexdigest()[:16]
            chunks.append(Chunk(chunk_id, piece, doc.source, doc.title, page, index))
    return chunks
