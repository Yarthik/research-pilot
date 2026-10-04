"""Optional web search via Tavily. Disabled unless TAVILY_API_KEY is set."""

from __future__ import annotations

import hashlib
import logging

from ..config import get_settings

log = logging.getLogger(__name__)


def web_search_available() -> bool:
    return bool(get_settings().tavily_api_key)


def web_search(query: str, max_results: int = 3) -> list[dict]:
    """Return results shaped like agent evidence. Never raises: failures return []."""
    if not web_search_available():
        return []
    try:
        from tavily import TavilyClient

        response = TavilyClient(api_key=get_settings().tavily_api_key).search(query, max_results=max_results)
    except Exception as exc:
        log.warning("web search failed: %s", exc)
        return []
    results = []
    for item in response.get("results", []):
        url = item.get("url", "")
        text = (item.get("content") or "").strip()
        if not url or not text:
            continue
        results.append(
            {
                "id": "web:" + hashlib.sha1(url.encode()).hexdigest()[:16],
                "text": text,
                "title": item.get("title") or url,
                "source": url,
                "page": None,
                "origin": "web",
            }
        )
    return results
