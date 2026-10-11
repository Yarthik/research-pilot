"""A cache for expensive singletons that is safe when several Streamlit sessions start at once."""

from __future__ import annotations

import threading
from collections.abc import Callable
from functools import lru_cache, wraps


def build_once[T](func: Callable[..., T]) -> Callable[..., T]:
    """Like functools.lru_cache, but concurrent first calls build the value once; the rest wait for it.

    lru_cache alone lets two threads both miss and both build. For the ChromaDB client that broke
    page loads when visitors arrived together on a cold start ("Could not connect to tenant"), and
    for the ONNX models it would load a second copy into memory.
    """
    cached = lru_cache(func)
    lock = threading.Lock()

    @wraps(func)
    def wrapper(*args):
        with lock:
            return cached(*args)

    wrapper.cache_clear = cached.cache_clear  # type: ignore[attr-defined]
    wrapper.cache_info = cached.cache_info  # type: ignore[attr-defined]
    return wrapper
