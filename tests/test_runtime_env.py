"""Hosting helpers: secrets -> environment, old-SQLite fallback, read-only app folder."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from runtime_env import ensure_modern_sqlite, export_secrets_to_env, load_streamlit_secrets  # noqa: E402

from researchpilot.config import get_settings, reset_settings  # noqa: E402
from researchpilot.retrieval.vector_store import writable_dir  # noqa: E402


def test_export_covers_the_types_streamlit_leaves_out():
    env = {}
    secrets = {
        "GROQ_API_KEY": "gsk_x",
        "DEMO_MAX_QUESTIONS": 5,
        "LLM_TEMPERATURE": 0.2,
        "ALLOW_INGEST": False,  # Streamlit does not promote booleans to the environment
        "AUTO_INGEST_SAMPLES": True,
        "general": {"nested": "tables are ignored"},
        "a_list": [1, 2],
    }
    exported = export_secrets_to_env(secrets, env)
    assert env == {
        "GROQ_API_KEY": "gsk_x",
        "DEMO_MAX_QUESTIONS": "5",
        "LLM_TEMPERATURE": "0.2",
        "ALLOW_INGEST": "false",
        "AUTO_INGEST_SAMPLES": "true",
    }
    assert sorted(exported) == sorted(env)


def test_existing_environment_variables_win():
    env = {"ALLOW_INGEST": "true"}
    assert export_secrets_to_env({"ALLOW_INGEST": False, "LLM_MODEL": "m"}, env) == ["LLM_MODEL"]
    assert env["ALLOW_INGEST"] == "true"


def test_secrets_become_settings(monkeypatch):
    env = {}
    export_secrets_to_env({"ALLOW_INGEST": False, "DEMO_MAX_QUESTIONS": "5", "LLM_MODEL": "openai/gpt-oss-120b"}, env)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    reset_settings()
    s = get_settings()
    assert s.allow_ingest is False and s.demo_max_questions == 5 and s.llm_model == "openai/gpt-oss-120b"


def test_load_streamlit_secrets_tolerates_a_missing_secrets_file():
    class NoSecrets:
        def __iter__(self):
            raise FileNotFoundError("No secrets found")

    assert load_streamlit_secrets(NoSecrets()) == []


def test_sqlite_is_left_alone_when_new_enough():
    import sqlite3

    ensure_modern_sqlite()
    assert sys.modules["sqlite3"] is sqlite3


def test_old_sqlite_is_swapped_for_pysqlite3(monkeypatch):
    import sqlite3

    fake = type(sys)("pysqlite3")
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 34, 0))
    monkeypatch.setitem(sys.modules, "pysqlite3", fake)
    monkeypatch.setitem(sys.modules, "sqlite3", sqlite3)  # restored afterwards
    ensure_modern_sqlite()
    assert sys.modules["sqlite3"] is fake


def test_old_sqlite_without_pysqlite3_explains_the_fix(monkeypatch):
    import sqlite3

    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 34, 0))
    monkeypatch.setitem(sys.modules, "pysqlite3", None)  # makes `import pysqlite3` fail
    with pytest.raises(RuntimeError, match="pysqlite3-binary"):
        ensure_modern_sqlite()


def test_index_folder_falls_back_to_temp_when_it_cannot_be_written(tmp_path, monkeypatch, caplog):
    blocker = tmp_path / "read-only-mount"
    blocker.write_text("a file where the folder should be", encoding="utf-8")  # mkdir under it fails
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "tmp"))
    chosen = writable_dir(blocker / "chroma")
    assert chosen == tmp_path / "tmp" / "researchpilot" / "chroma" and chosen.is_dir()
    assert "cannot write the index" in caplog.text


def test_writable_index_folder_is_used_as_is(tmp_path):
    target = tmp_path / "storage" / "chroma"
    assert writable_dir(target) == target and target.is_dir()
    assert not (target / ".write-test").exists()  # the probe file is cleaned up


def test_local_models_run_in_small_batches(monkeypatch):
    """ONNX Runtime's memory grows with batch size; 64 needed ~2-3 GB, too much for a 2.7 GB host."""
    import numpy as np

    from researchpilot.retrieval import embeddings, reranker

    seen = {}

    class FakeEmbedder:
        def embed(self, texts, batch_size=None):
            seen["embed"] = batch_size
            return (np.zeros(3) for _ in texts)

    class FakeReranker:
        def rerank(self, query, documents, batch_size=None):
            seen["rerank"] = batch_size
            return [0.0 for _ in documents]

    monkeypatch.setattr(embeddings, "_model", lambda name: FakeEmbedder())
    monkeypatch.setattr(reranker, "_model", lambda name: FakeReranker())
    embeddings.embed_texts(["a", "b"])
    reranker.score("q", ["a", "b"])
    assert seen == {"embed": embeddings.BATCH_SIZE, "rerank": reranker.BATCH_SIZE}
    assert embeddings.BATCH_SIZE <= 16 and reranker.BATCH_SIZE <= 16
