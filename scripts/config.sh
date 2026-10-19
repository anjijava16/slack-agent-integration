# shellcheck shell=bash
# Edit these once. Every script sources this file.
# Defaults are taken from anjijava16/glcoud-run-agents/scripts/gcp-setup.sh, so the
# Slack agent lands in the same Google Cloud project and reuses its shared pieces
# (Terraform state bucket, Workload Identity pool, openrouter-api-key secret).

PROJECT_ID="${PROJECT_ID:-project-52e8c95d-822f-4563-a7e}"
ACCOUNT="${ACCOUNT:-mamathaanjireddy@gmail.com}"
REGION="${REGION:-us-central1}"

GITHUB_REPO="${GITHUB_REPO:-anjijava16/slack-fastapi-langgraph-agent}"  # owner/name, exactly as on GitHub
GITHUB_BRANCH="${GITHUB_BRANCH:-develop}"                                # the branch that deploys

SERVICE="${SERVICE:-slack-agent}"            # Cloud Run services become slack-agent-test / slack-agent-prod
ENVIRONMENTS="${ENVIRONMENTS:-test prod}"    # cloud environments (local runs on SQLite, needs nothing)
DB_TIER="${DB_TIER:-db-f1-micro}"            # Cloud SQL tier (~USD 10/month for f1-micro)

STATE_BUCKET="${STATE_BUCKET:-${PROJECT_ID}-tfstate}"   # shared with glcoud-run-agents
WIF_POOL_ID="${WIF_POOL_ID:-github-pool}"              # shared with glcoud-run-agents
OPENROUTER_SECRET_ID="${OPENROUTER_SECRET_ID:-openrouter-api-key}"  # shared with glcoud-run-agents

LLM_MODEL="${LLM_MODEL:-openai/gpt-4o-mini}"   # any OpenRouter model with tool calling
