"""Provider-agnostic chat models. Switch providers with LLM_PROVIDER / LLM_MODEL."""

from __future__ import annotations

from functools import lru_cache

from .config import get_settings

PROVIDER_PACKAGES = {
    "groq": "langchain-groq",
    "openai": "langchain-openai",
    "anthropic": "langchain-anthropic",
    "google_genai": "langchain-google-genai",
    "ollama": "langchain-ollama",
}


class LLMConfigError(RuntimeError):
    """Raised when the configured LLM provider can't be initialised."""


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


class _Guarded:
    """Invokes a runnable and turns authentication/configuration failures into LLMConfigError."""

    def __init__(self, runnable) -> None:
        self.runnable = runnable

    def invoke(self, prompt):
        try:
            return self.runnable.invoke(prompt)
        except Exception as exc:
            if is_config_error(exc) and not isinstance(exc, LLMConfigError):
                raise LLMConfigError(f"LLM request failed ({type(exc).__name__}): {str(exc)[:300]}") from exc
            raise


def structured(schema, llm=None) -> _Guarded:
    """A runnable that returns an instance of the Pydantic `schema`."""
    llm = llm or get_llm()
    return _Guarded(with_retry(llm.with_structured_output(schema)))


def complete(prompt: str, llm=None) -> str:
    """Plain-text completion with retries."""
    llm = llm or get_llm()
    message = _Guarded(with_retry(llm)).invoke(prompt)
    content = getattr(message, "content", message)
    if isinstance(content, list):  # some providers return a list of content blocks
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)
