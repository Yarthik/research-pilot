from researchpilot import cli
from researchpilot.ingestion.pipeline import delete_source, ingest

LONG_TEXT = " ".join(f"Zebras have stripes number {i}." for i in range(100))


def test_ingest_files_and_replace_on_reingest(retriever, store, tmp_path):
    path = tmp_path / "zebras.txt"
    path.write_text(LONG_TEXT, encoding="utf-8")
    progress = []
    report = ingest([str(path), str(tmp_path / "missing.pdf")], on_progress=lambda s, i, n: progress.append((i, n)))

    assert report["ingested"][0]["source"] == "zebras.txt" and report["ingested"][0]["chunks"] > 1
    assert report["failed"][0]["source"].endswith("missing.pdf")
    assert progress == [(1, 2), (2, 2)]
    count = store.count()
    assert retriever.retrieve("zebras stripes", mode="bm25")  # BM25 index was refreshed

    ingest([str(path)])  # re-ingesting replaces instead of duplicating
    assert store.count() == count


def test_empty_file_is_reported(retriever, tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   ", encoding="utf-8")
    report = ingest([str(path)])
    assert report["ingested"] == [] and "no extractable text" in report["failed"][0]["error"]


def test_delete_source(loaded_retriever, store):
    assert delete_source("lora.pdf") == 2
    assert delete_source("lora.pdf") == 0
    assert "lora.pdf" not in {s["source"] for s in store.sources()}


def test_read_sources_file(tmp_path):
    path = tmp_path / "sources.txt"
    path.write_text("# comment\n\nhttps://a.org/x.pdf   A title\n  paper.pdf\n", encoding="utf-8")
    assert cli.read_sources_file(str(path)) == ["https://a.org/x.pdf", "paper.pdf"]


def test_cli_ingest_sources_delete(retriever, tmp_path, capsys):
    path = tmp_path / "zebras.txt"
    path.write_text(LONG_TEXT, encoding="utf-8")
    assert cli.main(["ingest", str(path)]) == 0
    assert "OK" in capsys.readouterr().out
    assert cli.main(["ingest", "--if-empty", str(path)]) == 0
    assert "skipping" in capsys.readouterr().out
    assert cli.main(["sources"]) == 0
    assert "zebras" in capsys.readouterr().out
    assert cli.main(["delete", "zebras.txt"]) == 0
    assert cli.main(["delete", "zebras.txt"]) == 1


def test_cli_trace_marks_skipped_grade(loaded_retriever, fake_llm, capsys):
    fake_llm.responses = {
        "ResearchPlan": [{"sub_queries": ["LoRA rank decomposition"], "needs_web": False}],
        "Answer": [{"answer": "LoRA injects rank decomposition matrices [1].", "cited_sources": [1]}],
    }
    assert cli.main(["ask", "How does LoRA work?", "--rounds", "1", "--trace"]) == 0
    out = capsys.readouterr().out
    assert "grade: grader skipped (no rounds left)" in out
    assert "sufficient=" not in out


def test_cli_reports_errors_cleanly(retriever, capsys):
    assert cli.main(["ingest", "/no/such/file.pdf"]) == 1
    assert "FAIL" in capsys.readouterr().out
