#!/usr/bin/env bash
# One-time Google Cloud setup for the Slack agent. Safe to re-run.
#
#   bash scripts/gcp-setup.sh            # use bash, not sh
#
# What it does:
#   1. gcloud: account/project, billing check, APIs, shared pieces that may already
#      exist from glcoud-run-agents (state bucket, WIF pool, openrouter-api-key)
#   2. terraform apply infra/platform: Cloud SQL Postgres (+ a database and user per
#      environment), Artifact Registry, service accounts, WIF provider, secrets
#   3. scripts/push-secrets.sh: copies your Slack + OpenRouter keys from ~/.zprofile.sh
#      into Secret Manager
#   4. writes .gcp-setup.env for scripts/github-setup.sh
#
# Needs: gcloud (logged in as the project owner) and terraform >= 1.6.
# Edit scripts/config.sh first.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
# shellcheck source=scripts/config.sh
source scripts/config.sh

for bin in gcloud terraform; do
  command -v "$bin" >/dev/null || { echo "!! $bin is not installed"; exit 1; }
done
if ! [[ "$SERVICE" =~ ^[a-z][a-z0-9-]{2,20}$ ]]; then
  echo "!! SERVICE '$SERVICE': 3-21 chars, lowercase letters, digits, hyphens."; exit 1
fi
echo "project: $PROJECT_ID | repo: $GITHUB_REPO@$GITHUB_BRANCH | service: $SERVICE | envs: $ENVIRONMENTS"

step "Account and project"
gcloud config set account "$ACCOUNT"
gcloud config set project "$PROJECT_ID"
gcloud config set billing/quota_project "$PROJECT_ID" >/dev/null
gcloud auth application-default set-quota-project "$PROJECT_ID" >/dev/null 2>&1 || true
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
echo "   project number: $PROJECT_NUMBER"
# Terraform authenticates with Application Default Credentials.
if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
  echo "   Terraform needs application-default credentials; opening the browser login..."
  gcloud auth application-default login
fi

step "Billing check"
gcloud services enable cloudbilling.googleapis.com serviceusage.googleapis.com >/dev/null 2>&1 || true
BILLING=$(gcloud billing projects describe "$PROJECT_ID" --format='value(billingEnabled)' 2>/dev/null || echo unknown)
if [ "$BILLING" = "False" ]; then
  echo "!! Billing is not linked to $PROJECT_ID (Cloud SQL needs it)."
  echo "   gcloud billing projects link $PROJECT_ID --billing-account=<ACCOUNT_ID>"
  exit 1
fi
echo "   billing: $BILLING"

step "Enable APIs (1-2 minutes the first time)"
gcloud services enable \
  run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com \
  sqladmin.googleapis.com iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com \
  cloudresourcemanager.googleapis.com serviceusage.googleapis.com storage.googleapis.com \
  logging.googleapis.com monitoring.googleapis.com cloudtrace.googleapis.com

step "Terraform state bucket gs://$STATE_BUCKET (shared)"
if ! exists gcloud storage buckets describe "gs://$STATE_BUCKET"; then
  gcloud storage buckets create "gs://$STATE_BUCKET" --location="$REGION" \
    --uniform-bucket-level-access --public-access-prevention
fi
gcloud storage buckets update "gs://$STATE_BUCKET" --versioning >/dev/null
echo "   ok"

step "Workload Identity pool $WIF_POOL_ID (shared)"
exists gcloud iam workload-identity-pools describe "$WIF_POOL_ID" --location=global || \
  gcloud iam workload-identity-pools create "$WIF_POOL_ID" --location=global --display-name="GitHub Actions"
echo "   ok"

step "OpenRouter key secret $OPENROUTER_SECRET_ID (shared)"
exists gcloud secrets describe "$OPENROUTER_SECRET_ID" || \
  gcloud secrets create "$OPENROUTER_SECRET_ID" --replication-policy=automatic
OR_KEY=$(read_secret OPENROUTER_API_KEY)
if [ -n "$OR_KEY" ]; then
  put_secret "$OPENROUTER_SECRET_ID" "$OR_KEY"
elif ! has_secret_value "$OPENROUTER_SECRET_ID"; then
  OR_KEY=$(ask_hidden "Paste your OpenRouter API key (sk-or-v1-...)")
  put_secret "$OPENROUTER_SECRET_ID" "$OR_KEY"
else
  echo "   already set (OPENROUTER_API_KEY not found in $PROFILE_FILE, keeping the stored one)"
fi
unset OR_KEY

step "terraform apply infra/platform (Cloud SQL creation takes ~10 minutes the first time)"
ENV_LIST=$(printf '"%s",' $ENVIRONMENTS); ENV_LIST="[${ENV_LIST%,}]"
TF="terraform -chdir=infra/platform"
$TF init -input=false -reconfigure -backend-config="bucket=$STATE_BUCKET" >/dev/null
TF_VARS=(
  -var="project_id=$PROJECT_ID" -var="region=$REGION" -var="service_name=$SERVICE"
  -var="environments=$ENV_LIST" -var="github_repository=$GITHUB_REPO" -var="github_branch=$GITHUB_BRANCH"
  -var="state_bucket=$STATE_BUCKET" -var="wif_pool_id=$WIF_POOL_ID"
  -var="openrouter_secret_id=$OPENROUTER_SECRET_ID" -var="db_tier=$DB_TIER"
)
APPROVE=(); [ "${AUTO_APPROVE:-0}" = "1" ] && APPROVE=(-auto-approve)
# Without AUTO_APPROVE=1 it shows the plan and asks "yes". If it fails on a brand-new
# service account ("does not exist"), IAM just hasn't caught up: run the script again.
$TF apply -input=false "${APPROVE[@]}" "${TF_VARS[@]}"
out() { $TF output -raw "$1"; }

step "Secrets from $PROFILE_FILE"
bash scripts/push-secrets.sh

step "Done"
SLACK_CHANNEL_ID_VALUE=$(read_secret SLACK_CHANNEL_ID)
cat > .gcp-setup.env <<ENV
GCP_PROJECT_ID=$(out GCP_PROJECT_ID)
GCP_REGION=$(out GCP_REGION)
GCP_WIF_PROVIDER=$(out GCP_WIF_PROVIDER)
GCP_DEPLOYER_SA=$(out GCP_DEPLOYER_SA)
GCP_RUNTIME_SA=$(out GCP_RUNTIME_SA)
GAR_REPOSITORY=$(out GAR_REPOSITORY)
CLOUDSQL_CONNECTION_NAME=$(out CLOUDSQL_CONNECTION_NAME)
TF_STATE_BUCKET=$STATE_BUCKET
SERVICE_NAME=$SERVICE
LLM_MODEL=$LLM_MODEL
SLACK_CHANNEL_ID=$SLACK_CHANNEL_ID_VALUE
GITHUB_REPO=$GITHUB_REPO
GITHUB_BRANCH=$GITHUB_BRANCH
ENV
sed 's/^/   /' .gcp-setup.env
echo
echo "Next: bash scripts/github-setup.sh   (puts these into GitHub with the gh CLI)"
