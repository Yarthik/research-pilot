import pytest
from pydantic import BaseModel

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


class _FakeGroq:
    """Records which structured-output method was requested; each method has a scripted outcome."""

    _llm_type = "groq-chat"

    def __init__(self, outcomes):
        self.outcomes, self.requested, self.called = outcomes, [], []

    def with_structured_output(self, schema, method="function_calling", strict=None):
        from langchain_core.runnables import RunnableLambda

        self.requested.append((method, strict))

        def run(_prompt):
            self.called.append(method)
            outcome = self.outcomes[method]
            if isinstance(outcome, Exception):
                raise outcome
            return schema(**outcome)

        return RunnableLambda(run)


def _groq_error(cls, status):
    import httpx

    response = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat"))
    return cls(f"Error code: {status}", response=response, body=None)


class _Plan(BaseModel):
    queries: list[str]


def test_groq_uses_native_json_schema_mode():
    fake = _FakeGroq({"json_schema": {"queries": ["a"]}, "function_calling": {"queries": ["tool"]}})
    assert llm.structured(_Plan, fake).invoke("p").queries == ["a"]
    assert ("json_schema", True) in fake.requested and fake.called == ["json_schema"]


def test_groq_json_schema_rejection_falls_back_to_tool_calling(caplog):
    groq = pytest.importorskip("groq")
    fake = _FakeGroq({"json_schema": _groq_error(groq.BadRequestError, 400), "function_calling": {"queries": ["t"]}})
    assert llm.structured(_Plan, fake).invoke("p").queries == ["t"]
    assert fake.called == ["json_schema", "function_calling"]
    assert "using tool calling" in caplog.text


def test_groq_rate_limit_is_not_swallowed_by_the_fallback():
    groq = pytest.importorskip("groq")
    fake = _FakeGroq({"json_schema": _groq_error(groq.RateLimitError, 429), "function_calling": {"queries": ["t"]}})
    with pytest.raises(llm.LLMRateLimitError) as caught:
        llm.structured(_Plan, fake).invoke("p")
    assert isinstance(caught.value.__cause__, groq.RateLimitError)  # the original error stays chained
    assert fake.called == ["json_schema"]  # no second attempt through tool calling


def test_other_providers_keep_tool_calling(fake_llm):
    fake_llm.responses = {"_Plan": [{"queries": ["x"]}]}
    assert llm.structured(_Plan, fake_llm).invoke("p").queries == ["x"]


def _rate_limit_error(message: str, status: int = 429):
    groq = pytest.importorskip("groq")
    return groq.RateLimitError(f"Error code: {status} - {message}", response=_groq_response(status), body=None)


def _groq_response(status):
    import httpx

    return httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat"))


DAILY = "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_secret` on tokens per day (TPD)"
PER_MINUTE = (
    "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_secret` on tokens per minute (TPM)"
)


@pytest.mark.parametrize("message, daily", [(DAILY, True), (PER_MINUTE, False)])
def test_rate_limits_become_a_friendly_error_without_the_providers_text(message, daily):
    fake = FakeLLM(responses={"_Plan": [_rate_limit_error(message)]})
    with pytest.raises(llm.LLMRateLimitError) as caught:
        llm.structured(_Plan, fake).invoke("p")
    error = caught.value
    assert error.daily is daily
    assert "org_secret" not in str(error) and "429" not in str(error)  # nothing from the provider leaks
    assert "org_secret" in error.detail  # ...but it is kept for the logs


def test_complete_also_raises_the_rate_limit_error():
    class Limited(FakeLLM):
        def invoke(self, prompt):
            raise _rate_limit_error(DAILY)

    with pytest.raises(llm.LLMRateLimitError):
        llm.complete("hi", llm=Limited())


@pytest.mark.parametrize(
    "exc, expected",
    [
        (RuntimeError("Error code: 429 - {'code': 'rate_limit_exceeded'}"), True),
        (RuntimeError("429 RESOURCE_EXHAUSTED. Quota exceeded"), True),
        (type("RateLimitError", (Exception,), {})("slow down"), True),  # OpenAI / Anthropic class name
        (TimeoutError("read timed out"), False),
        (ValueError("could not parse structured output"), False),
    ],
)
def test_is_rate_limit_error(exc, expected):
    assert llm.is_rate_limit_error(exc) is expected
