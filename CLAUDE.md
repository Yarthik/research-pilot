# ResearchPilot: notes for Claude Code

Portfolio project: an agentic RAG assistant (LangGraph agent, hybrid BM25 + vector retrieval with
RRF and a cross-encoder reranker, FastAPI, Streamlit, evaluation harness). The owner will show it
to employers, so it must run cleanly end to end. See README.md for the full description.

## Status

All code is written. It was developed in a sandbox with no access to PyPI, Hugging Face or arXiv,
so these parts have NOT yet run against the real libraries and need verifying first:

- ChromaDB calls in `researchpilot/retrieval/vector_store.py` (client settings, `embedding_function=None`, `get`/`delete`/`query` arguments)
- FastEmbed model names and APIs in `retrieval/embeddings.py` and `retrieval/reranker.py`
- LangGraph `stream(..., stream_mode=["updates", "values"])` in `agent/graph.py`
- `init_chat_model` and `with_structured_output` with the configured provider (`llm.py`)
- FastAPI app and tests (`tests/test_api.py`, `tests/test_ui_backend.py`)
- The Streamlit UI (`app/streamlit_app.py`), never rendered yet
- The Dockerfile and docker-compose, never built yet
- Key phrases in `evals/dataset.jsonl`, written from memory of the papers and not checked against the PDFs

Already verified: pure-Python logic (chunking, BM25, RRF, citations, evidence selection, agent
nodes, CLI, metrics); 92 tests pass and `ruff` is clean.

## Commands

```bash
pip install -r requirements-dev.txt
pytest -m "not slow"                  # fast suite, uses fakes (no keys, no downloads)
pytest -m slow                        # real local models + ChromaDB
ruff check . && ruff format --check .
python -m researchpilot ingest --file data/sample_sources.txt
python -m researchpilot ask "How does LoRA reduce the number of trainable parameters?" --trace
streamlit run app/streamlit_app.py
uvicorn researchpilot.api.main:app --port 8000
python -m evals retrieval --update-readme
python -m evals answers --compare-retry --limit 10 --delay 2 --update-readme
docker compose up --build
```

## Conventions

- Fix the code to match the real library APIs; keep the architecture and public behaviour.
- Tests must not need API keys or downloads (except `-m slow`); use the fakes in `tests/conftest.py`.
- Pin versions in `requirements.txt` to the ones that actually work once verified.
- Never commit `.env` or API keys. Never invent evaluation numbers: README tables are filled only by `--update-readme` from real runs.
- If an eval question is never retrieved by any mode, check its key phrases against the actual chunk text before changing anything else.
- Commit after each working step with a clear message.

## Definition of done

1. `pytest -m "not slow"` and `pytest -m slow` pass; ruff is clean.
2. Sample papers ingest; `ask` returns cited answers for the README example questions, including one that triggers a retry and one that is not in the documents.
3. Streamlit UI works: sources list, upload, URL ingest, live agent steps, citations, mode switch.
4. API works: `/docs`, `/ask`, `/ask/stream`, upload, delete, API-key and read-only modes.
5. Retrieval eval runs and its table is in the README; the answer eval runs on a subset.
6. `docker compose up --build` works and the image builds.
7. Pushed to GitHub with CI green; README placeholders (`YOUR-USERNAME`, demo link) filled in.
