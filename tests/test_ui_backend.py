"""The two UI backends behave the same: in-process, and over HTTP against the real API app."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langgraph")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from backend import ApiBackend, LocalBackend, get_backend  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

PLAN = {"sub_queries": ["LoRA"], "needs_web": False}
ANSWER = {"answer": "LoRA uses rank decomposition matrices [1].", "cited_sources": [1]}


@pytest.fixture(params=["local", "api"])
def backend(request, loaded_retriever, fake_llm):
    fake_llm.responses = {"ResearchPlan": [PLAN], "ContextGrade": [{"verdict": "sufficient"}], "Answer": [ANSWER]}
    if request.param == "local":
        return LocalBackend()
    from researchpilot.api.main import create_app

    return ApiBackend("http://testserver", client=TestClient(create_app()))


def test_info_and_sources(backend):
    assert backend.info()["ingest_enabled"] is True
    assert len(backend.sources()) == 4


def test_ask_stream(backend):
    events = list(backend.ask_stream("How does LoRA work?", mode="hybrid"))
    assert events[-1]["type"] == "result"
    assert events[-1]["citations"][0]["source"] == "lora.pdf"


def test_upload_and_delete(backend):
    report = backend.ingest_files([("notes.txt", b"Zebras are striped. " * 20)])
    assert report["ingested"][0]["source"] == "notes.txt"
    assert backend.delete("notes.txt") >= 1


def test_get_backend_picks_mode(monkeypatch):
    assert isinstance(get_backend(), LocalBackend)
    monkeypatch.setenv("API_URL", "http://localhost:8000")
    assert isinstance(get_backend(), ApiBackend)


@pytest.mark.parametrize("daily", [True, False])
def test_both_backends_raise_the_same_rate_limit_error(backend, fake_llm, daily):
    from researchpilot.llm import LLMRateLimitError
    from tests.conftest import quota_error

    fake_llm.responses = {"ResearchPlan": [quota_error(daily=daily)]}
    with pytest.raises(LLMRateLimitError) as caught:
        list(backend.ask_stream("How does LoRA work?"))
    assert caught.value.daily is daily
    assert "org_secret" not in str(caught.value)
