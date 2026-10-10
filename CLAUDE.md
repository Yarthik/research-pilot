# ResearchPilot: notes for Claude Code

Portfolio project: an agentic RAG assistant (LangGraph agent, hybrid BM25 + vector retrieval with
RRF and a cross-encoder reranker, FastAPI, Streamlit, evaluation harness). The owner will show it
to employers, so it must run cleanly end to end. See README.md for the full description.

## Where things stand (read first)

**Done** (details in the stage notes below; everything is committed and pushed to `main`): real-library
fixes and pinned versions; clean PDF titles/text; retrieval eval (default `hybrid`, reranker opt-in);
answer eval on all 29 questions (1 vs 2 rounds, qwen judge); Groq JSON-schema structured output; API
and Streamlit UI verified end to end; friendly rate-limit handling (`LLMRateLimitError`, see "Rate
limits"); `DEMO_MAX_QUESTIONS` per-session limit; Streamlit Community Cloud prep including the memory fix
(cold-start peak 3.25 GB -> 0.68 GB) and the secrets/SQLite/temp-folder guards.

**Tests:** 170 collected = 169 fast (`-m "not slow"`) + 1 slow; ruff clean. Update when it changes.

**Hosting plan:** the main host is Streamlit Community Cloud (free, ~2.7 GB RAM): entry point
`app/streamlit_app.py`, Python 3.12, settings as quoted top-level TOML secrets
(`.streamlit/secrets.toml.example`; steps in `docs/DEPLOY.md`). It is NOT deployed yet; the owner does that
from share.streamlit.io. The Hugging Face Docker Space (`scripts/deploy_hf_space.sh`) now needs a paid
plan and stays optional, as do Docker Compose and any container host.

**Open follow-ups:**
1. Deploy, then fill in the README demo link (placeholder: "add your Streamlit Community Cloud link
   here") and run the checklist at the end of `docs/DEPLOY.md` on the live app (the host's SQLite version,
   CPU count and folder permissions are untested).
2. The GitHub repo looked private (the repo page and CI badge returned 404 when signed out) and CI status
   was not checked. Employers need it public; confirm the Actions run is green.
3. Pointless retries on out-of-scope questions: the grader prompt says not to search again when the
   documents clearly don't cover a topic, but gpt-oss still rewrites the query (3 of the 6 retries in the
   2-round eval were unanswerable questions, e.g. the FIFA one). It costs a round of latency and LLM calls.
   Ideas: skip the retry when the new queries nearly repeat the old ones, or tighten the grader prompt;
   measure with the answer eval.
4. `l5` (how LoRA initialises A and B) is never retrieved by any mode, so the agent honestly abstains. The
   passage exists (page 4), but pypdf glues maths letters to words ("forB") and BM25 has no stemming.
5. BM25 has no stemming. Snowball stemming was tried and reverted (hybrid hit rate 96% -> 92%, lower MRR);
   retry only with a larger eval set.
6. Optional: ship the sample corpus's chunks and vectors as a snapshot, so a cold start or wake-up
   doesn't download and embed five PDFs (about 90 s here, 2-4 min expected on the free tier).
7. Minor: a rate-limited visitor waits ~13 s before the friendly message (3 agent retries x 3 SDK
   retries). The daily-vs-per-minute wording depends on Groq's message text.

## Status

All code is written. It was developed in a sandbox with no access to PyPI, Hugging Face or arXiv,
so these parts have NOT yet run against the real libraries and need verifying first:

- LLM providers other than Groq (only initialised with dummy keys)
- `docker compose up --build` locally: skipped by decision, since Docker Desktop can't run on the
  owner's machine (Windows virtualization is disabled). CI builds the image and smoke-tests the API
  container on every push, which stands in for this check.
- The live demo on Streamlit Community Cloud (prepared in Stage 7; the owner deploys it from share.streamlit.io).
  Hugging Face Docker Spaces now need a paid plan, so that path (Stage 6) is kept but is no longer the main one.

Already verified: pure-Python logic (chunking, BM25, RRF, citations, evidence selection, agent
nodes, CLI, metrics). Requires Python 3.12+; tested on 3.12.4 (3.14 is untested): 169 fast tests
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
answers.

Stage 5: `llm.structured` uses Groq's native JSON-schema mode (`method="json_schema"`, strict on
gpt-oss), falling back to tool calling on a 400; this fixed gpt-oss's intermittent `tool_use_failed`.
The answer eval predates it. The API was checked end to end against a live server (all endpoints,
upload/URL ingest/delete, `API_KEY` 401s, `ALLOW_INGEST=false` 403s). The owner checked the
Streamlit UI by hand (sources, live steps, citations, mode switch, PDF upload/delete, URL ingest,
retry, out-of-scope question, clear chat). UI text goes through `app/render.py` (LaTeX delimiters,
currency escaping, word-boundary titles); `.streamlit/config.toml` sets the minimal toolbar and the
20 MB upload limit.

Stage 6 (Hugging Face Space prep): `DEMO_MAX_QUESTIONS` caps UI questions per browser session
(unset = unlimited; "Clear chat" doesn't reset it; tested with Streamlit AppTest in
`tests/test_ui_app.py`). `AUTO_INGEST_SAMPLES` in the UI now ingests only missing sample papers and
retries on the next page load if a download failed. A simulated fresh Space (empty storage,
`AUTO_INGEST_SAMPLES=true`, `ALLOW_INGEST=false`) ingested the 5 papers in about 2 minutes and hid
upload/delete. `scripts/deploy_hf_space.sh` exports with `core.autocrlf=false` (Git for Windows'
`git archive` otherwise writes CRLF) and has `DRY_RUN=1`; it was dry-run from Git Bash, never
pushed by Claude. Steps and Space variables are in docs/DEPLOY.md.

Rate limits: a provider 429/quota error becomes `LLMRateLimitError` (generic message, `daily` flag, raw
text only in `detail` and the logs) in `llm._Guarded`. The agent nodes re-raise it (`FATAL_LLM_ERRORS`)
instead of degrading, since before that a quota-exhausted question ended with the raw Groq JSON, org id
included, as its answer. The API returns 429 (`/ask`) or a typed `rate_limit` stream event; the UI
shows `render.rate_limit_notice` and doesn't count the question. The daily-vs-minute wording relies on
Groq's message containing "per day" / (TPD) / (RPD); otherwise the per-minute notice is shown. The
eval's quota stop recognises the new exception.

Stage 7 (Streamlit Community Cloud prep; ~2.7 GB RAM limit): profiled the real `streamlit run` server
over a websocket session with an empty index and model cache and settings delivered only through
`secrets.toml`. The first measurement was 3.25 GB peak, caused by ONNX Runtime: fastembed's default
batch of 64 peaked near 2 GB and the arena allocator never gave it back. `embeddings.BATCH_SIZE` and
`reranker.BATCH_SIZE` are now 8 (vectors are bit-identical, so the retrieval eval is unchanged;
cold-start ingest went from 145 s to 91 s); peak is now 0.68 GB, 0.95 GB with the reranker selected.
Don't raise those batch sizes without re-measuring (`tests/test_runtime_env.py` pins them). Streamlit
promotes root-level string/int/float secrets to os.environ at server start but NOT booleans, so
`app/runtime_env.py` copies the rest (`ALLOW_INGEST = false` would otherwise leave uploads on for a
public demo); the docs still tell the owner to quote every value. Also added: an old-system-SQLite
guard for ChromaDB (swap in `pysqlite3` if installed, else a clear error), and a fallback to a temp
folder when the index folder isn't writable. The friendly Groq-quota message was verified end to end
by pointing the real Groq SDK at a local server that returns HTTP 429. Not verifiable from here: the
Community Cloud host itself (its SQLite version, CPU count, folder permissions); `docs/DEPLOY.md` has
the troubleshooting for each. Optional follow-up for a faster cold start: ship the sample corpus's
chunks and vectors as a snapshot instead of downloading and embedding five PDFs on every wake-up.

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
docker compose up --build              # can't run on the owner's machine (no virtualization); CI covers it
```

## Conventions

- Environment: Windows + Git Bash, venv in `.venv` (Python 3.12.4), so call `.venv/Scripts/python.exe -m ...`.
  `.gitattributes` forces LF for code files; `core.fileMode` is false locally.
- Never open or print `.env` (it holds the Groq key; the app loads it itself). Pass other settings as
  environment variables, and never write a real key into a file.
- Groq free tier: about 8K tokens/min and 200K tokens/day per model. A question costs 2-3 LLM calls, the
  answer eval about 4-5 per question per config, so keep LLM use modest and run evals one config per day.

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
6. `docker compose up --build` works and the image builds (checked by CI only; see Commands).
7. Pushed to GitHub with CI green; README demo-link placeholder filled in once the app is deployed.
