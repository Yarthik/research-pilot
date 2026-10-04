.PHONY: help install dev test test-slow lint format ingest ui api eval-retrieval eval-answers eval docker compose

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-16s %s\n", $$1, $$2}'

install:  ## install runtime dependencies
	pip install -r requirements.txt

dev:  ## install runtime + dev dependencies and the package in editable mode
	pip install -r requirements-dev.txt && pip install -e . --no-deps

test:  ## fast tests (no API keys, no downloads)
	pytest -m "not slow"

test-slow:  ## tests that download the real local models
	pytest -m slow

lint:  ## ruff lint + format check
	ruff check . && ruff format --check .

format:  ## auto-format
	ruff check . --fix && ruff format .

ingest:  ## ingest the 5 sample papers
	python -m researchpilot ingest --file data/sample_sources.txt

ui:  ## run the Streamlit UI on http://localhost:8501
	streamlit run app/streamlit_app.py

api:  ## run the API on http://localhost:8000 (docs at /docs)
	uvicorn researchpilot.api.main:app --reload --port 8000

eval-retrieval:  ## retrieval benchmark (free, no LLM)
	python -m evals retrieval --update-readme

eval-answers:  ## answer-quality benchmark with the retry ablation (uses the LLM)
	python -m evals answers --compare-retry --delay 2 --update-readme

eval: eval-retrieval eval-answers  ## both benchmarks

docker:  ## build the image
	docker build -t researchpilot .

compose:  ## run UI + API with docker compose
	docker compose up --build
