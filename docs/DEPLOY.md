# Deploying ResearchPilot

The public demo runs on Streamlit Community Cloud (Option 1): <https://research-pilot-yarthik.streamlit.app/>

| Option | Cost | Best for |
| --- | --- | --- |
| [1. Streamlit Community Cloud](#option-1-streamlit-community-cloud-free-the-public-demo) | Free | The public demo link (recommended) |
| [2. Docker Compose](#option-2-docker-compose-ui--api-on-one-machine) | Your machine | UI and API as separate services |
| [3. Hugging Face Docker Space](#option-3-hugging-face-docker-space-needs-a-paid-plan) | Paid plan (PRO or billing credits) | One container on Hugging Face |
| [4. Any container host](#option-4-any-container-host) | Varies | Running the image anywhere |

## Option 1: Streamlit Community Cloud (free, the public demo)

Community Cloud runs `app/streamlit_app.py` straight from the GitHub repository, with no Docker. One process holds the UI, the agent and the document index. The free tier has roughly 2.7 GB of memory; a cold start of this app peaks at about 0.7 GB (see [Memory](#memory)).

**What you need:** the repository pushed to GitHub, and a Groq API key (free at console.groq.com).

### Steps

1. **Push the code.** `git push` to `main` on GitHub (Community Cloud deploys from the repo, and redeploys on every push).
2. **Sign in.** Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and allow access to the repository.
3. **Create the app.** *Create app* → *Deploy a public app from GitHub* (button names may differ slightly). Then fill in:
   - **Repository:** `Yarthik/research-pilot`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **App URL:** pick a name, e.g. `research-pilot`
4. **Advanced settings.**
   - **Python version: 3.12** (the project requires 3.12 or newer).
   - **Secrets:** paste the TOML below, with your own key. It is the same as [.streamlit/secrets.toml.example](../.streamlit/secrets.toml.example).

   ```toml
   GROQ_API_KEY = "gsk_replace_me"

   LLM_PROVIDER = "groq"
   LLM_MODEL = "openai/gpt-oss-120b"

   AUTO_INGEST_SAMPLES = "true"
   ALLOW_INGEST = "false"
   DEMO_MAX_QUESTIONS = "5"
   ALLOW_SESSION_UPLOADS = "false"
   ```

   | Secret | Why |
   | --- | --- |
   | `GROQ_API_KEY` | The LLM key. It lives only in the secrets store, never in the repo or the page. |
   | `LLM_PROVIDER`, `LLM_MODEL` | Which model answers. `openai/gpt-oss-120b` is the one the evaluation used. |
   | `AUTO_INGEST_SAMPLES` | The first page load downloads and ingests the five sample papers. |
   | `ALLOW_INGEST` | `false` hides upload, URL ingest and delete, so visitors can ask but not change the documents. |
   | `DEMO_MAX_QUESTIONS` | Questions per visitor session, to protect the free Groq quota (see [below](#protecting-the-groq-quota)). |
   | `ALLOW_SESSION_UPLOADS` | `"true"` lets each visitor add up to 2 private files (PDF, TXT or MD; 5 MB each, of which the first 50 pages are searched) next to the shared papers, while `ALLOW_INGEST = "false"` keeps the shared papers read-only. See [Private uploads](#private-uploads-allow_session_uploads). |

   Put every secret at the top level of the file (not under a `[section]`) and quote the values, including `"true"`, `"false"` and numbers. Top-level secrets become environment variables, which is how the app reads its settings. Streamlit itself skips unquoted TOML booleans such as `ALLOW_INGEST = false`; the app has a fallback for that, but quoting is the safer habit.
5. **Deploy.** Installing the requirements takes about 5-10 minutes; watch *Manage app* for the log. The first visitor then triggers the model download (about 70 MB) and the ingest of the five papers, which took about 90 seconds on a 4-core machine and may take 2-4 minutes on the free tier's shared CPU. After that the app answers immediately.
6. **Check it** with the [checklist](#checklist-before-sharing-the-link), then put the app URL at the top of the GitHub README.

**Changing things later.** Edit secrets in *Manage app → Settings → Secrets* (the app restarts). Code changes deploy by pushing to `main`.

### What to expect on Community Cloud

- **Sleeping.** Free apps go to sleep after about 12 hours without visitors and wake when someone clicks *Yes, get this app back up*. Waking restarts the app, so the first load ingests the papers again (2-4 minutes). Open the link yourself a few minutes before showing it to someone.
- **Storage is temporary.** The index is written to `storage/` in the app folder, or to the system temp folder if the app folder is read-only; the FastEmbed model cache is in the temp folder. Both disappear on a restart and are rebuilt by `AUTO_INGEST_SAMPLES`.
- **Configuration files** are read from the repository: `requirements.txt` (pinned to the versions that were tested) and `.streamlit/config.toml` (minimal toolbar, 20 MB upload limit). Visual styling is in `app/styles.css`.
- **Streamlit's own chrome.** `app/styles.css` hides everything inside the app that can go without breaking it (the decoration bar, main menu, running indicator and deploy button; the toolbar itself stays because it holds the button that opens the sidebar on a phone), and `.streamlit/config.toml` sets the minimal toolbar. What Community Cloud draws around the app is outside the app's page, so CSS can't reach it: the hosting badge and the creator's avatar in the bottom-right corner, and the *Manage app* button, which only the owner sees. Sharing the link with `?embed=true` drops Community Cloud's surroundings (untested here).
- **Local runs** can use the same secrets: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (git-ignored) and run `streamlit run app/streamlit_app.py`.

### Memory

Measured on the real `streamlit run` server, starting from an empty index and model cache, with settings delivered only through `secrets.toml` and one question asked over a real session (Windows, 4 cores; Linux numbers should be similar):

| Scenario | Peak memory |
| --- | ---: |
| Idle server, no visitor | 66 MB |
| Cold start (download models, ingest 5 papers) and one question | 680 MB |
| The same, with a visitor switching on the cross-encoder reranker | 950 MB |
| *Before the fix: embedding in batches of 64* | *3,250 MB* |
| Two visitors at once, each adding a 5 MB PDF (45 and 30 dense pages) and asking a question; index already built, models loaded during the run | 457 MB |

The cause of the 3.2 GB was ONNX Runtime: embedding in batches of 64 peaked near 2 GB and the allocator never returned it. Batches of 8 give bit-identical vectors, use a quarter of the memory and ingest faster. `tests/test_runtime_env.py` pins the batch sizes.

### Private uploads (`ALLOW_SESSION_UPLOADS`)

With `ALLOW_SESSION_UPLOADS = "true"` (and `ALLOW_INGEST = "false"`), the message box gets an attach button (+) for up to 2 files per visitor: PDF, TXT or MD, at most 5 MB each. Of a longer file only the first 50 pages (or 400,000 characters of text) are read and indexed, and the file's chip and a notice by the message box say so ("Indexed the first 50 of 160 pages"). A file's text is checked when it's read: one whose text can't be read (a PDF in a pre-Unicode font extracts as symbols) is refused with "X's text can't be read: the PDF stores its text in a non-standard font encoding or as images.", and one in another language is indexed with a notice by its chip that ResearchPilot is tuned for English. If a file can't be added (no text, unreadable text, over 5 MB, a damaged PDF, a third file), the question sent with it is held back: it isn't answered or counted, the error shows by the message box, and the typed text stays there. The message box accepts files up to the server's 20 MB upload cap so that these reach the page and get that message; a file over 20 MB is refused by Streamlit in the browser with its own error badge. An attached file shows as a small chip by the message box (click ✕ to remove it) and under *Your documents* in the sidebar, and stays searchable for every later question in that session. No URL ingestion in this mode. The files are searched together with the shared papers, in every retrieval mode, but only in that browser session:

- Each session's files live in a small in-memory index of their own (chunks, vectors and a BM25 index over the shared papers plus that session's files). The shared ChromaDB index is never written, so it stays read-only and nothing is left on disk.
- A session can only reach the shared index and its own, so one visitor can never retrieve another's files, and one visitor's files don't change another's keyword scores. A session without uploads searches exactly as before.
- Files are removed when the visitor clicks *Remove*, reloads the page (a new session), or after an hour without activity (a background clean-up runs every 5 minutes). At most 10 sessions can hold files at once; beyond that, visitors see a friendly "at capacity" message and can still ask about the shared papers.
- Uploads are embedded one at a time, so concurrent uploads don't stack up memory. Adding a dense 30-45-page PDF took 30-50 seconds on a 4-core machine, including loading the embedding model the first time; expect longer on the free tier.
- A question sent with a newly attached file that refers to it ("What is it about?") searches only that file. Later questions search the file together with the shared papers.
- The question limit still applies. The API is unaffected: it keeps its single shared corpus and `ALLOW_INGEST`.

The limits are constants at the top of `researchpilot/retrieval/session.py`.

### Protecting the Groq quota

A question costs about 2-3 LLM calls and roughly 3-6K tokens. Groq's free tier allows 200K tokens a day for `gpt-oss-120b`, so the whole app can serve roughly 30-60 questions a day. `DEMO_MAX_QUESTIONS` is per browser session: it stops one visitor from using up the day's quota, but a new session starts again at zero, and Groq's daily limit remains the hard cap.

When the quota (or a per-minute limit) is used up, visitors see a friendly "the demo has reached today's free-tier limit" message with a link to the GitHub repo instead of an error, and a question that couldn't be answered doesn't count towards their allowance. The provider's own error text, which can include organisation ids, goes only to the app log (*Manage app → logs*).

### Troubleshooting

| What you see | Cause and fix |
| --- | --- |
| The page says *This host's SQLite is ... but ChromaDB needs 3.35 or newer* | The host's system SQLite is old. Add the line `pysqlite3-binary` to `requirements.txt` (Linux wheels exist for Python 3.12 and 3.13), push, and the app switches to it automatically. |
| Upload, URL and delete controls are visible | `ALLOW_INGEST` is missing, misspelled, or sits under a `[section]` in the secrets. |
| No attach button (+) in the message box | `ALLOW_SESSION_UPLOADS` isn't `"true"`, or `ALLOW_INGEST` isn't `"false"` (with shared uploads on, the private ones are hidden). |
| "No documents yet" after a restart | `AUTO_INGEST_SAMPLES` is missing or not `"true"`. A failed paper download is retried on the next page load. |
| The install fails on a dependency | Check that the Python version in *Advanced settings* is 3.12. |
| "Free-tier limit" message right after deploying | The Groq key is rate-limited or exhausted; check the app log for the provider's message. |

## Option 2: Docker Compose (UI + API on one machine)

```bash
cp .env.example .env        # add your API key
docker compose up --build
```

- UI: http://localhost:8501
- API: http://localhost:8000/docs

The API owns the index (a named volume, `storage`) and ingests the sample papers on first start; the UI calls the API over HTTP. Set `API_KEY` in `.env` to protect the API; the UI passes it along automatically.

## Option 3: Hugging Face Docker Space (needs a paid plan)

Hugging Face now requires a paid plan (PRO or billing credits) for Docker Spaces, so this is no longer the free option; use [Option 1](#option-1-streamlit-community-cloud-free-the-public-demo) for a free demo. A Docker Space runs the same image as `docker build`: one container, with the Streamlit UI on port 7860 and the agent running in-process.

1. **Create the Space.** On huggingface.co: *New Space* → name it (e.g. `research-pilot`) → SDK **Docker** → *Blank* template → visibility *Public*, with the hardware your plan allows.
2. **Add the settings.** In the Space: *Settings → Variables and secrets*.

   | Name | Type | Value |
   | --- | --- | --- |
   | `GROQ_API_KEY` | **Secret** | your Groq key |
   | `LLM_MODEL` | Variable | `openai/gpt-oss-120b` |
   | `AUTO_INGEST_SAMPLES` | Variable | `true` |
   | `ALLOW_INGEST` | Variable | `false` |
   | `DEMO_MAX_QUESTIONS` | Variable | `5` |

   They mean the same as in Option 1; see [Protecting the Groq quota](#protecting-the-groq-quota).
3. **Create a Hugging Face token.** *Settings → Access Tokens → Create new token* → type **Write** (or a fine-grained token with write access to this Space). Copy it; don't paste it into any file.
4. **Push the code from Git Bash.** The script deploys your last commit, so commit first. It swaps in `deploy/huggingface/README_SPACE.md` as the Space's README (Spaces needs a YAML header there; your GitHub README stays as it is). Replace `YOUR-HF-USERNAME`:

   ```bash
   cd "/d/AI Projects/research-pilot"
   git status                        # should be clean: only committed work is deployed
   read -rsp "HF token: " HF_TOKEN && export HF_TOKEN && echo   # typed, not echoed, and not saved in history
   DRY_RUN=1 ./scripts/deploy_hf_space.sh YOUR-HF-USERNAME/research-pilot   # optional: build the upload, no push
   ./scripts/deploy_hf_space.sh YOUR-HF-USERNAME/research-pilot
   unset HF_TOKEN
   ```

5. **Wait for the build** (*Logs → Build*, about 5-10 minutes the first time). The first page load then shows "Loading the sample papers (first start only)..." for about 2 minutes.

Free Spaces used to sleep after inactivity and had no persistent disk; the same applies here, and the papers are ingested again on the first page load after a restart. To update the Space later, commit and run step 4 again.

## Option 4: any container host

The image runs the UI by default and listens on `$PORT` (default 7860). To run the API instead, override the command:

```bash
docker run -p 8000:8000 -e SERVICE=api -e PORT=8000 --env-file .env researchpilot \
  uvicorn researchpilot.api.main:app --host 0.0.0.0 --port 8000
```

Mount a volume at `/app/storage` to keep the index across restarts.

## Checklist before sharing the link

- [ ] The sidebar lists the five papers with clean titles (or `/health` on the API shows a non-zero chunk count), and the heading reads "Ask questions about five landmark AI papers"
- [ ] One example question returns an answer with citations
- [ ] `ALLOW_INGEST=false` on public deployments: under *Library* the sidebar says "Shared with every visitor · read-only" and there is no *Manage library* section
- [ ] "N questions left" shows under the message box if `DEMO_MAX_QUESTIONS` is set
- [ ] With `ALLOW_SESSION_UPLOADS = "true"`: a small PDF attached with the + button can be asked about, a second browser (or a private window) doesn't see it, and its chip's ✕ removes it
- [ ] The API key is set as a *secret* (Community Cloud's Secrets box or a Hugging Face *Secret*), never in a file, and `.env` is not committed
