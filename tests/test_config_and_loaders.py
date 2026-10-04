import pytest
from pydantic import ValidationError

from researchpilot.config import Settings, get_settings, reset_settings
from researchpilot.ingestion.loaders import LoaderError, load_pdf_bytes, load_source


# --------------------------------------------------------------------------- config
def test_defaults():
    s = get_settings()
    assert s.llm_provider == "groq" and s.retrieval_mode == "hybrid_rerank"
    assert s.tavily_api_key is None and s.allow_ingest is True


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("FINAL_K", "8")
    monkeypatch.setenv("ALLOW_INGEST", "false")
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    monkeypatch.setenv("TAVILY_API_KEY", "")  # empty means "not set"
    reset_settings()
    s = get_settings()
    assert s.final_k == 8 and s.allow_ingest is False and s.retrieval_mode == "bm25"
    assert s.tavily_api_key is None


def test_invalid_values_are_rejected():
    with pytest.raises(ValidationError):
        Settings(retrieval_mode="magic")
    with pytest.raises(ValidationError):
        Settings(chunk_size=500, chunk_overlap=500)
    with pytest.raises(ValidationError):
        Settings(final_k=0)


# --------------------------------------------------------------------------- loaders
def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with one text line per page (no PDF library needed)."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode()}\nendstream")
        content_id = len(objects)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            "/Resources << /Font << /F1 3 0 R >> >> >>"
        )
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


def test_pdf_pages_and_title():
    doc = load_pdf_bytes(make_pdf(["LoRA Low-Rank Adaptation", "Second page text"]), "lora.pdf")
    assert doc.source == "lora.pdf"
    assert [n for n, _ in doc.pages] == [1, 2]
    assert "Second page" in doc.pages[1][1]
    assert doc.title.startswith("LoRA")


def test_pdf_file_and_text_file(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(make_pdf(["Hello PDF"]))
    assert load_source(str(pdf)).pages[0][1].strip().startswith("Hello PDF")

    md = tmp_path / "notes.md"
    md.write_text("# Notes\nSome text", encoding="utf-8")
    doc = load_source(str(md))
    assert doc.title == "notes" and doc.pages == [(None, "# Notes\nSome text")]


def test_loader_errors(tmp_path):
    with pytest.raises(LoaderError):
        load_source(str(tmp_path / "missing.pdf"))
    bad = tmp_path / "data.xlsx"
    bad.write_bytes(b"x")
    with pytest.raises(LoaderError):
        load_source(str(bad))
    with pytest.raises(LoaderError):
        load_pdf_bytes(b"this is not a pdf", "fake.pdf")
