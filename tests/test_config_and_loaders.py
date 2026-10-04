import pytest
from pydantic import ValidationError

from researchpilot.config import Settings, get_settings, reset_settings
from researchpilot.ingestion.loaders import LoaderError, clean_pdf_text, load_pdf_bytes, load_source


# --------------------------------------------------------------------------- config
def test_defaults():
    s = get_settings()
    assert s.llm_provider == "groq" and s.retrieval_mode == "hybrid"
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
def make_pdf(pages: list[str], streams: list[str] | None = None) -> bytes:
    """Build a minimal valid PDF with one text line per page (no PDF library needed).

    `streams` replaces the generated content with raw PDF content streams, one per page.
    """
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for stream in streams or [f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET" for text in pages]:
        stream = stream.encode()
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


def test_pdf_title_is_large_font_run_with_small_caps_rejoined():
    # Like arXiv papers: a small header line above a two-line small-caps title, then body text.
    lines = {  # baseline y -> fragments (font size, text) that flow along the line
        760: [(10, "Published as a conference paper at ICLR 2023")],
        720: [(17, "R"), (14, "E"), (17, "A"), (14, "C T"), (17, ": L"), (14, " OW"), (17, "-R"), (14, " ANK")],
        700: [(17, "L"), (14, "AN-")],
        680: [(14, "GUAGE")],
        640: [(10, "Body text of the paper, which is set in the smaller body font size.")],
        625: [(10, "More body text so that the body font is the most common size.")],
    }
    stream = (
        "BT "
        + " ".join(
            f"1 0 0 1 72 {y} Tm " + " ".join(f"/F1 {size} Tf ({text}) Tj" for size, text in fragments)
            for y, fragments in lines.items()
        )
        + " ET"
    )
    doc = load_pdf_bytes(make_pdf([], streams=[stream]), "react.pdf")
    assert doc.title == "REACT: LOW-RANK LANGUAGE"


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


def test_clean_pdf_text_fixes_ligatures_and_drops_unmapped_font_lines():
    text = (
        "The major downside of \ufb01ne-tuning is ef\ufb01ciency.\n"
        "$FW\x03\x14\x1d\x037KLQN>)LUVW\x03,\x03QHHG\n"  # figure text in a font with no Unicode map
        "So the answer is 300. \x17 Explanation for error category"
    )
    assert clean_pdf_text(text) == (
        "The major downside of fine-tuning is efficiency.\nSo the answer is 300.   Explanation for error category"
    )
