#!/usr/bin/env bash
# Everything GitHub Actions needs, via the GitHub CLI. Run after scripts/gcp-setup.sh.
#
#   gh auth login                 # once
#   bash scripts/github-setup.sh
#
# Repository VARIABLES (not secret; GCP trusts GitHub through Workload Identity
# Federation, so no Google key is stored anywhere):
#   GCP_PROJECT_ID GCP_REGION GCP_WIF_PROVIDER GCP_DEPLOYER_SA GCP_RUNTIME_SA
#   GAR_REPOSITORY CLOUDSQL_CONNECTION_NAME TF_STATE_BUCKET SERVICE_NAME
#   LLM_MODEL SLACK_MODE SLACK_CHANNEL_ID REQUIRE_API_KEY DEPLOY_PROD DOCKERHUB_REPO
#   QDRANT_URL RERANK_PROVIDER  (RAG turns on in the cloud once QDRANT_URL is set)
# Repository SECRETS (from ~/.zprofile.sh):
#   DOCKERHUB_USERNAME  DOCKERHUB_TOKEN   (Docker Hub access token, Read & Write)
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/lib.sh
source scripts/lib.sh

[ -f .gcp-setup.env ] || { echo "No .gcp-setup.env. Run scripts/gcp-setup.sh first."; exit 1; }
# shellcheck disable=SC1091
source .gcp-setup.env
command -v gh >/dev/null || { echo "!! Install the GitHub CLI: https://cli.github.com"; exit 1; }

step "Repository $GITHUB_REPO"
if ! gh repo view "$GITHUB_REPO" --json nameWithOwner -q .nameWithOwner >/dev/null 2>&1; then
  echo "   not found, creating it (private)"
  gh repo create "$GITHUB_REPO" --private --description "Slack agent: FastAPI + LangGraph on Cloud Run"
fi

setvar() { gh variable set "$1" --repo "$GITHUB_REPO" --body "$2"; echo "   variable $1 = $2"; }
setsecret() { printf '%s' "$2" | gh secret set "$1" --repo "$GITHUB_REPO"; echo "   secret   $1 = (set, ${#2} chars)"; }

step "GCP / deploy variables"
for v in GCP_PROJECT_ID GCP_REGION GCP_WIF_PROVIDER GCP_DEPLOYER_SA GCP_RUNTIME_SA GAR_REPOSITORY \
         CLOUDSQL_CONNECTION_NAME TF_STATE_BUCKET SERVICE_NAME; do
  setvar "$v" "${!v}"
done

step "App settings"
setvar LLM_MODEL "$LLM_MODEL"
setvar SLACK_MODE "http"           # Cloud Run: Slack POSTs to /slack/events
setvar REQUIRE_API_KEY "true"      # /api/* needs X-API-Key (secret <service>-<env>-api-key)
[ -n "${SLACK_CHANNEL_ID:-}" ] && setvar SLACK_CHANNEL_ID "$SLACK_CHANNEL_ID"
# Course FAQ RAG: the cluster URL isn't secret; the API key went to Secret Manager (push-secrets.sh).
QDRANT_URL_VALUE=$(read_secret QDRANT_URL)
if [ -n "$QDRANT_URL_VALUE" ] && [ "$QDRANT_URL_VALUE" != "http://localhost:6333" ]; then
  setvar QDRANT_URL "$QDRANT_URL_VALUE"
else
  echo "   QDRANT_URL not set (or still localhost): RAG stays off in the cloud. Later: gh variable set QDRANT_URL --body https://..."
fi
gh variable get RERANK_PROVIDER --repo "$GITHUB_REPO" >/dev/null 2>&1 || setvar RERANK_PROVIDER "fastembed"
# prod deploys after test only when this is "true". Flip it once prod has its own Slack app.
gh variable get DEPLOY_PROD --repo "$GITHUB_REPO" >/dev/null 2>&1 || setvar DEPLOY_PROD "false"

step "Docker Hub (optional)"
DH_USER=$(read_secret DOCKERHUB_USERNAME)
DH_TOKEN=$(read_secret DOCKERHUB_TOKEN)
if [ -n "$DH_USER" ] && [ -n "$DH_TOKEN" ]; then
  setsecret DOCKERHUB_USERNAME "$DH_USER"
  setsecret DOCKERHUB_TOKEN "$DH_TOKEN"
  setvar DOCKERHUB_REPO "${DH_USER}/${GITHUB_REPO#*/}"
else
  echo "   skipped: DOCKERHUB_USERNAME / DOCKERHUB_TOKEN not in your shell or $PROFILE_FILE"
fi
unset DH_TOKEN

step "GitHub environments (add required reviewers to 'production' for a manual gate)"
for e in test production; do
  gh api --method PUT "repos/$GITHUB_REPO/environments/$e" >/dev/null && echo "   $e"
done

step "Done"
gh variable list --repo "$GITHUB_REPO"
gh secret list --repo "$GITHUB_REPO"
cat <<NEXT

Push the code to $GITHUB_BRANCH to deploy:
  git init -b $GITHUB_BRANCH 2>/dev/null || true
  git remote add origin https://github.com/$GITHUB_REPO.git 2>/dev/null || true
  git add -A && git commit -m "Slack agent: FastAPI + LangGraph on Cloud Run"
  git push -u origin $GITHUB_BRANCH
Then point Slack at the new service:  bash scripts/slack-manifest.sh test
NEXT
