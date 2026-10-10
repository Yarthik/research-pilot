# One image for both the Streamlit UI (default) and the FastAPI service.
#   docker build -t researchpilot .
#   docker run -p 7860:7860 --env-file .env researchpilot                       # UI
#   docker run -p 8000:8000 --env-file .env researchpilot \
#       uvicorn researchpilot.api.main:app --host 0.0.0.0 --port 8000           # API
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/app \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    PORT=7860

# Hugging Face Spaces runs containers as UID 1000
RUN useradd --create-home --uid 1000 --home-dir /app user
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the embedding + reranker models at build time so the first request is fast.
COPY --chown=user researchpilot ./researchpilot
RUN mkdir -p /app/storage /app/.cache && chown -R user:user /app
USER user
RUN python -m researchpilot warmup

COPY --chown=user . .

EXPOSE 7860 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "scripts/healthcheck.py"]

CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false --server.maxUploadSize=${MAX_UPLOAD_MB:-20}"]
