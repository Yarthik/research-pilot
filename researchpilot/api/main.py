"""FastAPI service.

uvicorn researchpilot.api.main:app --port 8000
open http://localhost:8000/docs for the interactive API docs
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader

from .. import __version__
from ..config import get_settings
from ..llm import LLMConfigError, LLMRateLimitError
from .schemas import (
    AskRequest,
    AskResponse,
    DeleteResponse,
    HealthResponse,
    IngestResponse,
    IngestUrlsRequest,
    SourceInfo,
)

log = logging.getLogger(__name__)
ALLOWED_UPLOADS = {".pdf", ".txt", ".md", ".markdown"}
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    expected = get_settings().api_key
    if expected and not (key and secrets.compare_digest(key, expected)):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


def require_ingest_allowed() -> None:
    if not get_settings().allow_ingest:
        raise HTTPException(status_code=403, detail="Ingestion is disabled on this deployment (ALLOW_INGEST=false)")


def _retriever():
    from ..retrieval.hybrid import get_retriever

    return get_retriever()


SAMPLE_FILE = Path(__file__).resolve().parents[2] / "data" / "sample_sources.txt"


def _auto_ingest_samples() -> None:
    """Ingest the sample papers in the background when the index is empty (AUTO_INGEST_SAMPLES=true)."""
    try:
        if _retriever().store.count() == 0 and SAMPLE_FILE.exists():
            from ..cli import read_sources_file
            from ..ingestion.pipeline import ingest

            report = ingest(read_sources_file(str(SAMPLE_FILE)))
            log.info("auto-ingested %d sample source(s), %d failed", len(report["ingested"]), len(report["failed"]))
    except Exception:
        log.exception("auto-ingest of sample sources failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("AUTO_INGEST_SAMPLES", "").lower() in {"1", "true", "yes"}:
        threading.Thread(target=_auto_ingest_samples, daemon=True).start()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        lifespan=lifespan,
        title="ResearchPilot API",
        version=__version__,
        description="Agentic research assistant: hybrid retrieval + a self-correcting LangGraph agent with citations.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(LLMConfigError)
    async def _llm_config_error(_request, exc: LLMConfigError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(LLMRateLimitError)
    async def _llm_rate_limited(_request, exc: LLMRateLimitError):
        return JSONResponse(status_code=429, content={"detail": str(exc), "daily": exc.daily})

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        s = get_settings()
        try:
            chunks = _retriever().store.count()
            status = "ok"
        except Exception as exc:  # the API still answers so the problem is visible
            log.error("vector store unavailable: %s", exc)
            chunks, status = 0, "degraded"
        from ..tools.web_search import web_search_available

        return HealthResponse(
            status=status,
            version=__version__,
            chunks=chunks,
            llm_provider=s.llm_provider,
            llm_model=s.llm_model,
            retrieval_mode=s.retrieval_mode,
            web_search=web_search_available(),
            ingest_enabled=s.allow_ingest,
        )

    @app.get("/sources", response_model=list[SourceInfo], tags=["documents"], dependencies=[Depends(require_api_key)])
    def list_sources() -> list[dict]:
        return _retriever().store.sources()

    @app.post(
        "/ingest/urls",
        response_model=IngestResponse,
        tags=["documents"],
        dependencies=[Depends(require_api_key), Depends(require_ingest_allowed)],
    )
    def ingest_urls(body: IngestUrlsRequest) -> dict:
        from ..ingestion.pipeline import ingest

        return ingest([str(u) for u in body.urls])

    @app.post(
        "/ingest/files",
        response_model=IngestResponse,
        tags=["documents"],
        dependencies=[Depends(require_api_key), Depends(require_ingest_allowed)],
    )
    def ingest_files(files: list[UploadFile] = File(...)) -> dict:
        from ..ingestion.loaders import Document, load_pdf_bytes
        from ..ingestion.pipeline import ingest_document

        limit = get_settings().max_upload_mb * 1024 * 1024
        report: dict = {"ingested": [], "failed": []}
        for upload in files:
            name = Path(upload.filename or "upload").name
            try:
                suffix = Path(name).suffix.lower()
                if suffix not in ALLOWED_UPLOADS:
                    raise ValueError(f"unsupported file type '{suffix}' (use .pdf, .txt or .md)")
                data = upload.file.read(limit + 1)
                if len(data) > limit:
                    raise ValueError(f"file is larger than {get_settings().max_upload_mb} MB")
                if suffix == ".pdf":
                    doc = load_pdf_bytes(data, name)
                else:
                    doc = Document(source=name, title=Path(name).stem, pages=[(None, data.decode("utf-8", "replace"))])
                report["ingested"].append(ingest_document(doc, refresh=False))
            except Exception as exc:
                report["failed"].append({"source": name, "error": str(exc)})
        _retriever().refresh()
        return report

    @app.delete(
        "/sources",
        response_model=DeleteResponse,
        tags=["documents"],
        dependencies=[Depends(require_api_key), Depends(require_ingest_allowed)],
    )
    def delete(source: str = Query(..., description="The source exactly as listed by GET /sources")) -> dict:
        from ..ingestion.pipeline import delete_source

        deleted = delete_source(source)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"No chunks found for source '{source}'")
        return {"source": source, "deleted_chunks": deleted}

    @app.post("/ask", response_model=AskResponse, tags=["agent"], dependencies=[Depends(require_api_key)])
    def ask_question(body: AskRequest) -> dict:
        from ..agent.graph import ask

        return ask(body.question, mode=body.mode, max_rounds=body.max_rounds)

    @app.post("/ask/stream", tags=["agent"], dependencies=[Depends(require_api_key)])
    def ask_question_stream(body: AskRequest) -> StreamingResponse:
        """Newline-delimited JSON: one {"type": "step"} line per agent step, then {"type": "result"}."""
        from ..agent.graph import ask_stream

        def events():
            try:
                for event in ask_stream(body.question, mode=body.mode, max_rounds=body.max_rounds):
                    yield json.dumps(event, default=str) + "\n"
            except LLMRateLimitError as exc:  # already logged; send a typed event clients can show nicely
                event = {"type": "error", "code": "rate_limit", "detail": str(exc), "daily": exc.daily}
                yield json.dumps(event) + "\n"
            except Exception as exc:
                log.exception("streaming request failed")
                yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"

        return StreamingResponse(events(), media_type="application/x-ndjson")

    return app


app = create_app()
