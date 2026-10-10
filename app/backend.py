"""How the UI talks to ResearchPilot.

- LocalBackend runs everything in the Streamlit process (one container, e.g. Hugging Face Spaces).
- ApiBackend calls the FastAPI service over HTTP (docker compose: UI and API as two services).

Set API_URL to use the API; leave it unset to run locally.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path


class LocalBackend:
    name = "local"

    def info(self) -> dict:
        from researchpilot.config import get_settings

        s = get_settings()
        return {
            "llm_provider": s.llm_provider,
            "llm_model": s.llm_model,
            "retrieval_mode": s.retrieval_mode,
            "ingest_enabled": s.allow_ingest,
            "max_upload_mb": s.max_upload_mb,
        }

    def _retriever(self):
        from researchpilot.retrieval.hybrid import get_retriever

        return get_retriever()

    def sources(self) -> list[dict]:
        return self._retriever().store.sources()

    def ingest_urls(self, urls: list[str], on_progress=None) -> dict:
        from researchpilot.ingestion.pipeline import ingest

        return ingest(urls, on_progress=on_progress)

    def ingest_files(self, files: list[tuple[str, bytes]]) -> dict:
        from researchpilot.config import get_settings
        from researchpilot.ingestion.loaders import Document, load_pdf_bytes
        from researchpilot.ingestion.pipeline import ingest_document

        limit = get_settings().max_upload_mb * 1024 * 1024
        report: dict = {"ingested": [], "failed": []}
        for name, data in files:
            try:
                if len(data) > limit:
                    raise ValueError(f"larger than {get_settings().max_upload_mb} MB")
                if name.lower().endswith(".pdf"):
                    doc = load_pdf_bytes(data, name)
                else:
                    doc = Document(source=name, title=Path(name).stem, pages=[(None, data.decode("utf-8", "replace"))])
                report["ingested"].append(ingest_document(doc, refresh=False))
            except Exception as exc:
                report["failed"].append({"source": name, "error": str(exc)})
        self._retriever().refresh()
        return report

    def delete(self, source: str) -> int:
        from researchpilot.ingestion.pipeline import delete_source

        return delete_source(source)

    def ask_stream(self, question: str, mode: str | None = None) -> Iterator[dict]:
        from researchpilot.agent.graph import ask_stream

        yield from ask_stream(question, mode=mode)


class ApiBackend:
    name = "api"

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 300, client=None) -> None:
        headers = {"X-API-Key": api_key} if api_key else {}
        if client is None:
            import httpx

            client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)
        else:
            client.headers.update(headers)
        self.client = client

    def _check(self, response):
        if response.status_code >= 400:
            try:
                body = response.json()
                detail = body.get("detail", response.text)
            except ValueError:
                body, detail = {}, response.text
            if response.status_code == 429:  # the API's LLM provider is rate-limited
                from researchpilot.llm import LLMRateLimitError

                raise LLMRateLimitError(daily=bool(body.get("daily")))
            raise RuntimeError(f"API error {response.status_code}: {detail}")
        return response

    def info(self) -> dict:
        data = self._check(self.client.get("/health")).json()
        data.setdefault("max_upload_mb", 20)
        data["ingest_enabled"] = data.get("ingest_enabled", True)
        return data

    def sources(self) -> list[dict]:
        return self._check(self.client.get("/sources")).json()

    def ingest_urls(self, urls: list[str], on_progress=None) -> dict:
        if on_progress:
            on_progress(", ".join(urls), 1, 1)
        return self._check(self.client.post("/ingest/urls", json={"urls": urls})).json()

    def ingest_files(self, files: list[tuple[str, bytes]]) -> dict:
        payload = [("files", (name, data)) for name, data in files]
        return self._check(self.client.post("/ingest/files", files=payload)).json()

    def delete(self, source: str) -> int:
        return self._check(self.client.delete("/sources", params={"source": source})).json()["deleted_chunks"]

    def ask_stream(self, question: str, mode: str | None = None) -> Iterator[dict]:
        body = {"question": question, "mode": mode}
        with self.client.stream("POST", "/ask/stream", json=body) as response:
            if response.status_code >= 400:
                response.read()
                self._check(response)
            for line in response.iter_lines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("type") == "error":
                    if event.get("code") == "rate_limit":
                        from researchpilot.llm import LLMRateLimitError

                        raise LLMRateLimitError(daily=bool(event.get("daily")))
                    raise RuntimeError(event.get("detail", "unknown error"))
                yield event


def get_backend():
    url = os.environ.get("API_URL", "").strip()
    if url:
        return ApiBackend(url, api_key=os.environ.get("API_KEY") or None)
    return LocalBackend()
