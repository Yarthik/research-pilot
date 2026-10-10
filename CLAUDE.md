# ResearchPilot: notes for Claude Code

Portfolio project: an agentic RAG assistant (LangGraph agent, hybrid BM25 + vector retrieval with
RRF and a cross-encoder reranker, FastAPI, Streamlit, evaluation harness). The owner will show it
to employers, so it must run cleanly end to end. See README.md for the full description.

## Status

All code is written. It was developed in a sandbox with no access to PyPI, Hugging Face or arXiv,
so these parts have NOT yet run against the real libraries and need verifying first:

- LLM providers other than Groq (only initialised with dummy keys)
- The FastAPI app against a live server (`/docs`, `/ask/stream`, uploads); its tests pass with fakes
- The Streamlit UI (`app/streamlit_app.py`), never rendered yet
- The Dockerfile and docker-compose, never built yet

Already verified: pure-Python logic (chunking, BM25, RRF, citations, evidence selection, agent
nodes, CLI, metrics). Requires Python 3.12+; tested on 3.12.4 (3.14 is untested): 123 fast tests
(`pytest -m "not slow"`) and 1 slow test (`pytest -m slow`) pass, `ruff` is clean, and
`requirements.txt` is pinned to the versions that ran. Verified against the real libraries in
Stage 1: ChromaDB calls in `retrieval/vector_store.py`, FastEmbed embeddings and reranker (slow
test, all four retrieval modes), the LangGraph `stream(..., stream_mode=["updates", "values"])`
in `agent/graph.py` (run by the graph tests with a fake LLM), and `init_chat_model` +
`with_structured_output` for all five providers (initialised with dummy keys, no network call).

Verified in Stage 2 with Groq `openai/gpt-oss-120b` (the old default `llama-3.3-70b-versatile` is
no longer served): the five sample papers ingest with correct titles; `ask` returns cited answers
from the right papers, retries when round 1 misses part of a question, and says when the documents
don't cover a question. PDF text is NFKC-normalised (ligatures) and lines from fonts without a
Unicode map are dropped; gpt-oss's `【1†L1-L4】` citation markers are rewritten to `[1]`.

Stage 3: the retrieval eval runs and its table is in the README. Every answerable question's key
phrases match real chunk text (`l1`/`m2` gained "decomposition matrices", which a chunk boundary
split from "rank"). `l5` is never retrieved: a genuine failure (BM25 has no stemming, so
"initialized" misses "initialization", and pypdf glues math letters to words, e.g. "forB"). The
cross-encoder lowers MRR versus plain hybrid; see the Stage 3 commit message for the analysis.

Stage 3b: default `RETRIEVAL_MODE` is now `hybrid` (best hit rate and MRR, ~11 ms); three
rerankers (MiniLM-L-6, jina-v1-turbo, bge-reranker-base) all scored below plain RRF, so
`hybrid_rerank` is opt-in and `RERANKER_MODEL` stays MiniLM-L-6. Snowball stemming for BM25 was
tried and reverted (lower hybrid hit rate and MRR). Without a scorer, retrieve interleaves results
by rank across sub-queries so evidence trimming can't drop a whole sub-query. Experiment results are
in `evals/results/experiments/`; README "Findings" summarises them. Don't re-tune on these 25
questions without a larger eval set.

Stage 4: the answer eval ran on all 29 questions, 1 round (2026-10-09) and 2 rounds (2026-10-10),
agent `openai/gpt-oss-120b`, judge `qwen/qwen3.8-27b` (set via `JUDGE_PROVIDER`/`JUDGE_MODEL` env
vars), judge prompt v2. Both configs score 88% correctness; only 6 questions retried and only `re3`
improved. Runs checkpoint to `evals/results/answers_<mode>_<n>round.json`, resume on re-run, and stop
cleanly on 429/quota errors (including ones the agent swallows in its fallbacks). Judge prompt v1
penalised reference details the question didn't ask for; `python -m evals rejudge` re-grades saved
answers. Known issue: gpt-oss on Groq sometimes returns JSON text instead of a tool call for
structured output (`tool_use_failed`, 3 grade steps in the 2-round run); `method="json_schema"` in
`llm.structured` is the likely fix, but changing it means re-running both configs.

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
python -m evals answers --rounds 1 --delay 30          # then --rounds 2 (another day if quota is short)
python -m evals answers --compare-retry --update-readme # combine the two saved runs, no LLM calls
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
7. Pushed to GitHub with CI green; README demo-link placeholder filled in.
