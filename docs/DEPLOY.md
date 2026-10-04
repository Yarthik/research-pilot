# Deploying ResearchPilot

## Option 1: Hugging Face Spaces (free, public demo link)

A Docker Space runs the same image as `docker build`, with the Streamlit UI on port 7860.

1. **Create the Space.** On huggingface.co: *New Space* → name it (e.g. `research-pilot`) → SDK **Docker** → *Blank* template → hardware *CPU basic* (free).
2. **Add settings.** In the Space: *Settings → Variables and secrets*.
   - Secret `GROQ_API_KEY` = your key (or the key for whichever provider you set below)
   - Variable `AUTO_INGEST_SAMPLES` = `true` (ingests the five sample papers on first start)
   - Variable `ALLOW_INGEST` = `false` (recommended for a public link: visitors can ask but not change the documents)
   - Optional: `LLM_PROVIDER`, `LLM_MODEL` if you don't want the Groq default
3. **Push the code.** Commit your work, create a Hugging Face access token with *write* permission, then:

   ```bash
   export HF_TOKEN=hf_xxx
   ./scripts/deploy_hf_space.sh YOUR-HF-USERNAME/research-pilot
   ```

   The script pushes your last commit and swaps in `deploy/huggingface/README_SPACE.md` as the Space's README (Spaces needs a YAML header there; your GitHub README stays as it is).
4. **Wait for the build** (about 5–10 minutes the first time; it downloads the models into the image). The first page load then ingests the sample papers, which takes a minute or two.
5. Put the Space link at the top of the GitHub README.

Free Spaces sleep after inactivity and their disk isn't persistent: after a restart the papers are ingested again automatically. Open the link yourself a few minutes before showing it to someone.

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

- [ ] `/health` (API) or the UI sidebar shows the expected LLM and a non-zero chunk count
- [ ] One example question from the UI returns an answer with citations
- [ ] `ALLOW_INGEST=false` on public deployments
- [ ] The API key is set as a *secret*, not a plain variable, and `.env` is not committed
