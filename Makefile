.PHONY: install run run-http tunnel test test-pg test-qdrant test-real lint fmt db psql feedback qdrant ingest ask search rag-status docker-build docker-run dockerhub-push \
        tf-fmt tf-validate tf-test gcp-setup push-secrets github-setup slack-manifest restart logs url

ENV ?= test
TAG ?= $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo local)
DOCKERHUB_REPO ?= anjaiahspr/slack-fastapi-langgraph-agent
PG_URL ?= postgresql://agent:agent@localhost:5432/agent

# ------------------------------------------------------------------ local (SQLite)
install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt

# Socket Mode + SQLite at ./data/agent.db — no public URL, no database server
run:
	uvicorn app.main:app --reload --port 8016

# HTTP mode locally (expose with `make tunnel`)
run-http:
	SLACK_MODE=http uvicorn app.main:app --reload --port 8016

tunnel:
	ngrok http 8016

# ------------------------------------------------------------------ RAG (Qdrant)
COURSE ?= llm
Q ?= Can I still join the course?

# Local Qdrant in Docker (dashboard: http://localhost:6333/dashboard)
qdrant:
	docker compose up -d qdrant

# Ingest everything configured for a course (faq, github, slack, youtube): make ingest COURSE=llm
ingest:
	python -m app.rag.cli ingest $(COURSE) $(if $(SOURCES),--sources $(SOURCES),)

ask:
	python -m app.rag.cli ask $(COURSE) "$(Q)"

search:
	python -m app.rag.cli search $(COURSE) "$(Q)"

rag-status:
	python -m app.rag.cli status

# Look inside the local database
feedback:
	sqlite3 -header -column data/agent.db "SELECT f.updated_at, f.feedback_type, f.user_id, substr(r.question,1,60) AS question FROM agent_feedback f JOIN agent_runs r USING (run_id) ORDER BY f.updated_at DESC LIMIT 20;"

# ------------------------------------------------------------------ tests & lint
test:
	pytest -q

# RAG tests against a real Qdrant too (start one with `make qdrant`)
test-qdrant:
	TEST_QDRANT_URL=http://localhost:6333 pytest -q

# Retrieval quality on the real DataTalks.Club LLM Zoomcamp FAQ + repo (clones them into /tmp/dtc)
test-real:
	@[ -d /tmp/dtc/faq ] || git clone --depth 1 https://github.com/DataTalksClub/faq /tmp/dtc/faq
	@[ -d /tmp/dtc/llm-zoomcamp ] || git clone --depth 1 https://github.com/DataTalksClub/llm-zoomcamp /tmp/dtc/llm-zoomcamp
	REAL_FAQ_DIR=/tmp/dtc/faq REAL_COURSE_REPO=/tmp/dtc/llm-zoomcamp TEST_QDRANT_URL=$${TEST_QDRANT_URL:-http://localhost:6333} pytest -q -s tests/test_rag_real_data.py

# Same tests plus the PostgreSQL ones (start one with `make db`)
test-pg:
	TEST_DATABASE_URL=$(PG_URL) pytest -q

lint:
	ruff check app tests examples && ruff format --check app tests examples && shellcheck -x -S warning scripts/*.sh

fmt:
	ruff format app tests examples && ruff check --fix app tests examples && terraform fmt -recursive infra

# Local PostgreSQL in Docker, to rehearse the cloud setup or run test-pg
db:
	docker compose --profile pg up -d postgres

psql:
	docker compose exec postgres psql -U agent -d agent

# ------------------------------------------------------------------ docker
docker-build:
	docker build -t slack-agent:local .

docker-run:
	docker run --rm -p 8000:8080 --env-file .env -e DATABASE_URL=sqlite:////tmp/data/agent.db slack-agent:local

# linux/amd64 is what Cloud Run runs (also right from an Apple Silicon Mac). `docker login -u anjaiahspr` first.
dockerhub-push:
	docker buildx build --platform linux/amd64 -t $(DOCKERHUB_REPO):$(TAG) -t $(DOCKERHUB_REPO):latest --push .

# ------------------------------------------------------------------ terraform
tf-fmt:
	terraform fmt -recursive infra

tf-validate:
	terraform -chdir=infra/platform init -backend=false -input=false >/dev/null && terraform -chdir=infra/platform validate
	terraform -chdir=infra/cloudrun init -backend=false -input=false >/dev/null && terraform -chdir=infra/cloudrun validate

# Plans against mocked providers — no cloud account needed
tf-test: tf-validate
	terraform -chdir=infra/platform test
	terraform -chdir=infra/cloudrun test

# ------------------------------------------------------------------ Google Cloud (one-time + ops)
gcp-setup:
	bash scripts/gcp-setup.sh

push-secrets:
	bash scripts/push-secrets.sh $(filter test prod,$(ENV))

github-setup:
	bash scripts/github-setup.sh

slack-manifest:
	bash scripts/slack-manifest.sh $(ENV)

url:
	@. scripts/config.sh && gcloud run services describe $$SERVICE-$(ENV) --project $$PROJECT_ID --region $$REGION --format='value(status.url)'

# New revision so instances re-read Secret Manager (after push-secrets). The next deploy drops the marker.
restart:
	@. scripts/config.sh && gcloud run services update $$SERVICE-$(ENV) --project $$PROJECT_ID --region $$REGION --update-env-vars=SECRETS_REFRESHED_AT=$$(date +%s)

logs:
	@. scripts/config.sh && gcloud run services logs read $$SERVICE-$(ENV) --project $$PROJECT_ID --region $$REGION --limit 100
