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
(cold-start peak 3.25 GB -> 0.68 GB) and the secrets/SQLite/temp-folder guards. Deployed (2026-10-11)
at https://research-pilot-yarthik.streamlit.app/; the GitHub repo (Yarthik/research-pilot) is public and
its full history was scanned for secrets (none found). After going live: `$...$` maths vs currency now
follows Pandoc's rule (`app/render.py`), and ALL-CAPS PDF titles are title-cased at ingest
(`loaders.title_case_if_all_caps`; existing indexes keep old titles until re-ingested). UI polish
(2026-10-11): visitor-facing copy that adapts to `ALLOW_INGEST`, a "How it works" expander, a blue accent
in `.streamlit/config.toml` (set under `[theme.light]` and `[theme.dark]`; a plain `[theme]` would drop
dark mode), a grouped sidebar (brand, Settings, demo limit, linked Documents, footer), plain-English
agent steps (on by default), and four example questions from the eval set (l1 shortened, m2, re3, u2), each
checked live. Before the first question the chat input sits inline under the intro: a pinned input
makes Streamlit scroll to the bottom on load, which hid the heading on phones. Step and footer wording
lives in `app/render.py`. Then (2026-10-11): landing heading "ResearchPilot" with a fixed subtitle in every
mode; the chat placeholder names the sample papers' topics only when the read-only sample corpus is all
there is. `ALLOW_SESSION_UPLOADS` (default false; UI only, needs `ALLOW_INGEST=false`) gives each browser
session up to 2 private, temporary files (PDF/TXT/MD, 5 MB, 50 pages, 400k chars; no URLs), see "Private
uploads" below. Also fixed a cold-start race: concurrent first page loads each built a ChromaDB client
(`lru_cache` isn't thread-safe) and some visitors got "Could not reach the document index";
`researchpilot/caching.build_once` now guards the retriever, the ONNX models and the session registry.
UI redesign (2026-10-12): centred 820 px reading column (layout centered), Inter (loaded by an @import in `app/styles.css`; the theme's "Inter:<url>" form is ignored in [theme.light]/[theme.dark]), smaller heading scale, compact hero, 2x2 suggestion cards, citation chips that jump to source cards (snippet from the best-matching sentence, "Open PDF at page N" for arXiv, full passage in an expander), one neutral style for every alert, a first-start setup panel with progress, a questions-left meter, and hidden Streamlit chrome. All styling is in `app/styles.css` (read on every run), the helpers that depend on it in `app/ui.py`, the pure pieces in `app/render.py` (unit-tested). Muted colours are color-mix() of the text colour so one rule works in both themes; a Playwright contrast audit found every styled text at 4.76:1 or more. Don't hide `stToolbar`: it holds the phone's open-sidebar button.
UI v3 (2026-10-17, branch `ui-v3`, awaiting the owner's review; not merged to main): subtraction pass. Tokens at the top of `app/styles.css` (spacing 4-48, type 12/13/15/28, radius 8/12, colours as color-mix of the text colour on :root so dialogs, which render outside .stApp, get them). Empty state is one vertically centred group (the column's inner block has flex: 1, so it gets justify-content: center too); a submitted question is queued in `session_state.pending` and the page reruns, so the chat layout and live `st.status` appear at once. Private uploads go through `st.chat_input(accept_file="multiple")` into the same session registry (chips by the composer remove them; `st.bottom` holds the chips and the questions-left line when pinned). Steps are one `st.status` collapsing to `render.steps_summary`; sources are a two-column grid of `st.button` cards (label laid out by CSS grid: number, title, page, 2-line snippet) opening `ui.source_dialog`; chips target an anchor before each card. "How it works" is an `st.dialog`; the sidebar is brand, New chat (always present), Search method, Library, Your documents (only when attached), footer; admin controls (ALLOW_INGEST=true) sit in a *Manage library* expander. `render.fix_small_caps` rejoins "A PPLYING"-style splits in displayed passages only. Gotchas: Streamlit gives Markdown containers margin-bottom -16px (cancel it where paragraphs have no margin); Markdown icons are `span[role="img"]`; innerText applies text-transform ("YOUR DOCUMENTS"). Playwright contrast audit: every styled text 4.91:1 or more in both themes.
UI v3 polish (2026-10-19, same branch): one placeholder "Ask a question…" ("Demo limit reached" when disabled, with a dashed composer); source cards sorted by citation number; the live status keeps finished steps plus one "now doing" line (`render.next_activity`); short display names (`render.short_name`: text before the first colon, else word-boundary cut at 30 chars; full title on hover and in the dialog); library arrows share one right edge; footer model on its own line; tooltips capped at `--rp-tooltip-width`; the How-it-works baseline fix is a shared line-height (Streamlit buttons set 1.6). Unexpected answer failures now show a friendly message kept in the history (details only in the log) and refund the question. A screenshot harness that runs the real app with the tests' FakeLLM (scenarios short/long/slow/uncovered/error/ratelimit/setup) lives in Claude's scratchpad, not the repo.

**Tests:** 238 collected = 237 fast (`-m "not slow"`) + 1 slow; ruff clean (on ui-v3). Update when it changes.

**Hosting plan:** the main host is Streamlit Community Cloud (free, ~2.7 GB RAM): entry point
`app/streamlit_app.py`, Python 3.12, settings as quoted top-level TOML secrets
(`.streamlit/secrets.toml.example`; steps in `docs/DEPLOY.md`). Live at
https://research-pilot-yarthik.streamlit.app/ (linked at the top of the README); it redeploys on every
push to `main`. The Hugging Face Docker Space (`scripts/deploy_hf_space.sh`) now needs a paid plan and
stays optional, as do Docker Compose and any container host.

**Open follow-ups:**
1. Run the checklist at the end of `docs/DEPLOY.md` on the live app (the host's SQLite version, CPU
   count and folder permissions were untested before deploying).
2. Confirm the GitHub Actions run is green now that the repo is public.
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
- The live demo on Streamlit Community Cloud (prepared in Stage 7, deployed by the owner on 2026-10-11;
  the DEPLOY.md checklist hasn't been run on it yet). Hugging Face Docker Spaces now need a paid plan, so that path (Stage 6) is kept but is no longer the main one.

Already verified: pure-Python logic (chunking, BM25, RRF, citations, evidence selection, agent
nodes, CLI, metrics). Requires Python 3.12+; tested on 3.12.4 (3.14 is untested): 237 fast tests
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

Private uploads (`researchpilot/retrieval/session.py`): each session's files live in their own in-memory
index (chunks, normalised vectors, BM25 over shared chunks + own); `SessionRetriever` subclasses
`HybridRetriever` and only overrides `vector_search` (shared top-k merged with own top-k by cosine) and
`bm25`, so modes and RRF are unchanged. A session without uploads gets the shared retriever object itself.
The shared ChromaDB collection is never written. Chosen over an owner field + filtered search because that
would write to the shared index, could reorder shared HNSW results, and would let other sessions' files
shift BM25 IDF. The agent gets the session via `ask_stream(..., owner=...)` -> state `owner` -> retrieve
node. Limits: 2 files, 5 MB, 50 pages, 400k chars per file, 10 sessions at once, 1 h idle (lazy clean-up on
any page load plus a daemon thread every 5 min); embedding is serialised by a lock. Verified: retrieval
eval and all 232 exact rankings unchanged (u4, an unanswerable question, varies run to run in vector modes
even on the old code: HNSW on an off-topic query); two concurrent sessions each adding a 5 MB PDF and asking
peaked at 457 MB; live, session A answered from its private PDF while session B, at the same time, could
not see it. arXiv is unreachable from Claude's sandbox, so test PDFs there are generated locally.

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
7. Pushed to the public GitHub repo with CI green; the README links the live demo.
