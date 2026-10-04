import pytest

from researchpilot import llm

from .conftest import FakeLLM


class AuthenticationError(Exception):
    pass


def test_structured_and_complete_use_the_configured_model(fake_llm):
    fake_llm.text = "hello"
    assert llm.complete("hi") == "hello"


def test_complete_joins_content_blocks(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(fake, "invoke", lambda _p: type("M", (), {"content": [{"text": "a"}, {"text": "b"}]})())
    assert llm.complete("hi", llm=fake) == "ab"


@pytest.mark.parametrize(
    "exc",
    [AuthenticationError("bad key"), RuntimeError("Invalid API key provided"), RuntimeError("model_not_found")],
)
def test_auth_errors_become_config_errors(exc):
    from pydantic import BaseModel

    class Dummy(BaseModel):
        x: int = 0

    fake = FakeLLM(responses={"Dummy": [exc]})
    with pytest.raises(llm.LLMConfigError):
        llm.structured(Dummy, fake).invoke("prompt")


def test_transient_errors_pass_through():
    from pydantic import BaseModel

    class Dummy(BaseModel):
        x: int = 0

    fake = FakeLLM(responses={"Dummy": [TimeoutError("read timed out")]})
    with pytest.raises(TimeoutError):
        llm.structured(Dummy, fake).invoke("prompt")


def test_unknown_provider_gives_a_helpful_error():
    pytest.importorskip("langchain")
    with pytest.raises(llm.LLMConfigError):
        llm.make_llm("not-a-provider", "model")
