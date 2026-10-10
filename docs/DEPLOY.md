# Deploying ResearchPilot

## Option 1: Hugging Face Spaces (free, public demo link)

A Docker Space runs the same image as `docker build`: one container, with the Streamlit UI on port 7860 and the agent running in-process.

1. **Create the Space.** On huggingface.co: *New Space* → name it (e.g. `research-pilot`) → SDK **Docker** → *Blank* template → hardware *CPU basic* (free) → visibility *Public*.
2. **Add the settings.** In the Space: *Settings → Variables and secrets*.

   | Name | Type | Value | Why |
   | --- | --- | --- | --- |
   | `GROQ_API_KEY` | **Secret** | your Groq key | Secrets are hidden from visitors and from the Space's files |
   | `LLM_MODEL` | Variable | `openai/gpt-oss-120b` | The model the evaluation used (also the default) |
   | `AUTO_INGEST_SAMPLES` | Variable | `true` | The first page load ingests the five sample papers (about 2 minutes) |
   | `ALLOW_INGEST` | Variable | `false` | Visitors can ask questions but not upload or delete documents |
   | `DEMO_MAX_QUESTIONS` | Variable | `5` | Questions per browser session, to protect the free Groq quota |

   Sizing `DEMO_MAX_QUESTIONS`: a question costs about 2–3 LLM calls and roughly 3–6K tokens. Groq's free tier allows 200K tokens a day for `gpt-oss-120b`, so the whole Space can serve roughly 30–60 questions a day. The limit is per browser session: it stops one visitor from using up the day's quota, but a new session starts again at zero, and Groq's daily limit remains the hard cap.
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

5. **Wait for the build.** Watch *Logs → Build* on the Space page: about 5–10 minutes the first time (it installs the requirements and downloads the embedding and reranker models into the image). Then open the Space: the first load shows "Loading the sample papers (first start only)..." for about 2 minutes.
6. **Check it** with the checklist at the end of this page, then put the Space link at the top of the GitHub README.

Free Spaces sleep after about 48 hours without visitors, and their disk isn't persistent: after a restart the papers are ingested again automatically on the first page load. If a paper fails to download, the next page load retries the missing ones. Open the link yourself a few minutes before showing it to someone.

To update the Space later, commit and run step 4 again (the push replaces the Space's contents).

## Option 2: Docker Compose (UI + API on one machine)

```bash
cp .env.example .env        # add your API key
docker compose up --build
```

- UI: http://localhost:8501
- API: http://localhost:8000/docs

The API owns the index (a named volume, `storage`) and ingests the sample papers on first start; the UI calls the API over HTTP. Set `API_KEY` in `.env` to protect the API; the UI passes it along automatically.

## Option 3: any container host

The image runs the UI by default and listens on `$PORT` (default 7860). To run the API instead, override the command:

```bash
docker run -p 8000:8000 -e SERVICE=api -e PORT=8000 --env-file .env researchpilot \
  uvicorn researchpilot.api.main:app --host 0.0.0.0 --port 8000
```

Mount a volume at `/app/storage` to keep the index across restarts.

## Checklist before sharing the link

- [ ] `/health` (API) or the UI sidebar shows the expected LLM and a non-zero chunk count (5 papers, 583 chunks for the samples)
- [ ] One example question from the UI returns an answer with citations
- [ ] `ALLOW_INGEST=false` on public deployments: the sidebar says "This demo uses a fixed set of documents" and has no upload or ✕ buttons
- [ ] The sidebar shows "Demo limit: N of N left in this session" if `DEMO_MAX_QUESTIONS` is set
- [ ] The API key is set as a *secret*, not a plain variable, and `.env` is not committed
