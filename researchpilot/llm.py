"""Provider-agnostic chat models. Switch providers with LLM_PROVIDER / LLM_MODEL."""

from __future__ import annotations

import logging
from functools import lru_cache

from .config import get_settings

log = logging.getLogger(__name__)

PROVIDER_PACKAGES = {
    "groq": "langchain-groq",
    "openai": "langchain-openai",
    "anthropic": "langchain-anthropic",
    "google_genai": "langchain-google-genai",
    "ollama": "langchain-ollama",
}


class LLMConfigError(RuntimeError):
    """Raised when the configured LLM provider can't be initialised."""


class LLMRateLimitError(RuntimeError):
    """The provider's rate limit or quota stopped a request, even after retries.

    The message is generic on purpose: the provider's own text (which can include organisation ids and
    usage figures) is kept in `detail` for logs and never shown to users. `daily` is True when a
    per-day quota is used up, so waiting a minute won't help.
    """

    def __init__(
        self,
        message: str = "The language model's rate limit or quota was reached.",
        *,
        daily: bool = False,
        detail: str = "",
    ):
        super().__init__(message)
        self.daily, self.detail = daily, detail


# Errors the agent must not paper over with a degraded answer: retrying or guessing won't help.
FATAL_LLM_ERRORS = (LLMConfigError, LLMRateLimitError)


def make_llm(provider: str, model: str, temperature: float = 0.0):
    """Create a LangChain chat model for any supported provider."""
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise LLMConfigError("langchain is not installed: pip install -r requirements.txt") from exc
    try:
        return init_chat_model(model, model_provider=provider, temperature=temperature)
    except ImportError as exc:
        package = PROVIDER_PACKAGES.get(provider, f"the LangChain integration for '{provider}'")
        raise LLMConfigError(f"LLM provider '{provider}' needs {package}: pip install {package}") from exc
    except Exception as exc:
        raise LLMConfigError(f"Could not initialise {provider}/{model}: {exc}") from exc


@lru_cache
def get_llm():
    s = get_settings()
    return make_llm(s.llm_provider, s.llm_model, s.llm_temperature)


def with_retry(runnable):
    """Wrap a runnable with exponential-backoff retries (free API tiers rate-limit often)."""
    attempts = get_settings().llm_max_retries
    if attempts <= 1 or not hasattr(runnable, "with_retry"):
        return runnable
    return runnable.with_retry(stop_after_attempt=attempts, wait_exponential_jitter=True)


AUTH_ERROR_NAMES = {"AuthenticationError", "PermissionDeniedError"}


def is_config_error(exc: Exception) -> bool:
    """True for errors that retrying won't fix: bad or missing API keys, unknown models."""
    if isinstance(exc, LLMConfigError):
        return True
    if type(exc).__name__ in AUTH_ERROR_NAMES or getattr(exc, "status_code", None) in (401, 403):
        return True
    text = str(exc).lower()
    unknown_model = "model_not_found" in text or ("model" in text and "does not exist" in text)
    return "api key" in text or "api_key" in text or unknown_model


RATE_LIMIT_NAMES = {"RateLimitError", "ResourceExhausted", "TooManyRequests"}
_DAILY_MARKERS = ("per day", "(tpd)", "(rpd)", "daily", "insufficient_quota", "billing")


def is_rate_limit_error(exc: Exception) -> bool:
    """True for HTTP 429 / quota errors from any provider (Groq, OpenAI, Anthropic, Google)."""
    if isinstance(exc, LLMRateLimitError):
        return True
    if type(exc).__name__ in RATE_LIMIT_NAMES or getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "rate_limit_exceeded" in text or "resource_exhausted" in text or "error code: 429" in text


def _rate_limit_error(exc: Exception) -> LLMRateLimitError:
    text = str(exc)
    daily = any(marker in text.lower() for marker in _DAILY_MARKERS)
    return LLMRateLimitError(daily=daily, detail=f"{type(exc).__name__}: {text[:500]}")


class _Guarded:
    """Invokes a runnable and turns authentication, configuration and rate-limit failures into
    LLMConfigError and LLMRateLimitError (after the retries wrapped around the runnable)."""

    def __init__(self, runnable) -> None:
        self.runnable = runnable

    def invoke(self, prompt):
        try:
            return self.runnable.invoke(prompt)
        except Exception as exc:
            if is_config_error(exc) and not isinstance(exc, LLMConfigError):
                raise LLMConfigError(f"LLM request failed ({type(exc).__name__}): {str(exc)[:300]}") from exc
            if is_rate_limit_error(exc) and not isinstance(exc, LLMRateLimitError):
                limited = _rate_limit_error(exc)
                log.warning("LLM rate limit reached (daily=%s): %s", limited.daily, limited.detail)
                raise limited from exc
            raise


def _with_schema(llm, schema):
    """Structured output: Groq's native JSON-schema mode where possible, tool calling otherwise.

    With tool calling, gpt-oss on Groq sometimes writes the JSON as plain text instead of calling the
    tool, and Groq rejects the reply (400 tool_use_failed). JSON-schema mode avoids that; strict=True
    constrains decoding for models that support it (langchain-groq drops it for the others). A 400
    from JSON-schema mode (unsupported model, output failing validation) falls back to tool calling.
    Rate limits are not caught here, so they still reach the retry and quota handling.
    """
    tool_calling = llm.with_structured_output(schema)
    if getattr(llm, "_llm_type", "") != "groq-chat":
        return tool_calling
    from groq import BadRequestError
    from langchain_core.runnables import RunnableLambda

    def fall_back(prompt):
        log.warning("JSON-schema structured output was rejected for %s; using tool calling", schema.__name__)
        return tool_calling.invoke(prompt)

    native = llm.with_structured_output(schema, method="json_schema", strict=True)
    return native.with_fallbacks([RunnableLambda(fall_back)], exceptions_to_handle=(BadRequestError,))


def structured(schema, llm=None) -> _Guarded:
    """A runnable that returns an instance of the Pydantic `schema`."""
    llm = llm or get_llm()
    return _Guarded(with_retry(_with_schema(llm, schema)))


def complete(prompt: str, llm=None) -> str:
    """Plain-text completion with retries."""
    llm = llm or get_llm()
    message = _Guarded(with_retry(llm)).invoke(prompt)
    content = getattr(message, "content", message)
    if isinstance(content, list):  # some providers return a list of content blocks
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)
