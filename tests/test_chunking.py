import pytest

from researchpilot.ingestion.chunking import Chunk, chunk_document, clean_text, split_text
from researchpilot.ingestion.loaders import Document

TEXT = " ".join(f"Sentence number {i} is here." for i in range(200))


def test_short_text_is_one_chunk():
    assert split_text("Hello world.", 100, 10) == ["Hello world."]


def test_empty_text_gives_no_chunks():
    assert split_text("   \n\n ", 100, 10) == []


def test_chunks_never_exceed_size():
    for size, overlap in [(200, 40), (120, 100), (500, 0)]:
        chunks = split_text(TEXT, size, overlap)
        assert len(chunks) > 1
        assert all(len(c) <= size for c in chunks), (size, overlap)


def test_no_content_lost():
    joined = " ".join(split_text(TEXT, 200, 40))
    for i in range(200):
        assert f"number {i} is" in joined


def test_consecutive_chunks_overlap():
    chunks = split_text(TEXT, 200, 40)
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert set(a.split()[-3:]) & set(b.split()[:10])


def test_long_unbroken_text_is_hard_split():
    chunks = split_text("x" * 500, 200, 50)
    assert all(chunks) and all(len(c) <= 200 for c in chunks)
    assert sum(len(c) for c in chunks) >= 500


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text(TEXT, 100, 100)


def test_clean_text_joins_pdf_line_wraps():
    assert clean_text("trans-\nformer models\nare great") == "transformer models are great"


def test_clean_text_keeps_paragraph_breaks():
    assert clean_text("First paragraph.\n\n\n\nSecond.") == "First paragraph.\n\nSecond."


def test_metadata_round_trip():
    chunk = Chunk("abc", "text", "file.pdf", "Title", None, 3)
    assert Chunk.from_metadata("abc", "text", chunk.metadata()) == chunk
    with_page = Chunk("abc", "text", "file.pdf", "Title", 7, 3)
    assert Chunk.from_metadata("abc", "text", with_page.metadata()) == with_page


def test_chunk_document_keeps_pages_and_unique_ids():
    doc = Document(source="a.pdf", title="A", pages=[(1, TEXT), (2, "Short page.")])
    chunks = chunk_document(doc, 300, 50)
    assert chunks[0].page == 1 and chunks[-1].page == 2
    assert len({c.id for c in chunks}) == len(chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_ids_are_stable_across_runs():
    doc = Document(source="a.pdf", title="A", pages=[(1, TEXT)])
    assert [c.id for c in chunk_document(doc, 300, 50)] == [c.id for c in chunk_document(doc, 300, 50)]
