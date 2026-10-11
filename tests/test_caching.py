"""build_once: expensive singletons are built once even when sessions start at the same time."""

import threading
import time

from researchpilot.caching import build_once
from researchpilot.retrieval import embeddings, hybrid, reranker, session


def test_concurrent_first_calls_build_the_value_once():
    builds = []

    @build_once
    def slow_singleton():
        builds.append(1)
        time.sleep(0.2)  # long enough for every thread to arrive while the first is still building
        return object()

    results = []
    threads = [threading.Thread(target=lambda: results.append(slow_singleton())) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(builds) == 1
    assert len({id(r) for r in results}) == 1

    slow_singleton.cache_clear()
    slow_singleton()
    assert len(builds) == 2


def test_the_expensive_singletons_use_it():
    # The ChromaDB client (a race broke page loads on a cold start) and the ONNX models.
    for func in (hybrid.get_retriever, session.get_registry, embeddings._model, reranker._model):
        assert func.__code__.co_name == "wrapper" and hasattr(func, "cache_clear")
