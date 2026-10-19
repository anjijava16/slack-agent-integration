#!/usr/bin/env bash
# Copy Slack credentials from ~/.zprofile.sh (or your shell) into Secret Manager,
# one set per environment. Re-run any time you rotate a token; unchanged values
# are skipped. Values travel through stdin only, never on a command line.
#
#   bash scripts/push-secrets.sh            # all environments in config.sh
#   bash scripts/push-secrets.sh test       # just one
#
# Names it looks for (first one found wins):
#   test:  TEST_SLACK_BOT_TOKEN       or SLACK_BOT_TOKEN
#          TEST_SLACK_SIGNING_SECRET  or SLACK_SIGNING_SECRET      (asked for if missing)
#          TEST_SLACK_APP_TOKEN       or SLACK_APP_TOKEN or SLACK_SOCKET_TOKEN   (optional)
#   prod:  PROD_SLACK_BOT_TOKEN ...   then the same fallbacks
#   both:  <ENV>_QDRANT_API_KEY or QDRANT_API_KEY      (Qdrant Cloud, for the course FAQ RAG)
#          <ENV>_COHERE_API_KEY or COHERE_API_KEY      (optional, Cohere re-ranking)
# A prod deploy that falls back to the test bot works, but one Slack app can only
# send events to ONE request URL, so give prod its own Slack app (PROD_* names).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/lib.sh
source scripts/lib.sh
# shellcheck source=scripts/config.sh
source scripts/config.sh
gcloud config set project "$PROJECT_ID" >/dev/null 2>&1

envs="${1:-$ENVIRONMENTS}"
for env in $envs; do
  ENV_UPPER=$(printf '%s' "$env" | tr '[:lower:]' '[:upper:]')
  prefix="${SERVICE}-${env}"
  echo "-- $env"

  # Bot token (required)
  pick "${ENV_UPPER}_SLACK_BOT_TOKEN" SLACK_BOT_TOKEN; v="$PICK_VALUE"
  if [ -n "$v" ]; then
    echo "   bot token from \$$PICK_NAME"
    case "$v" in xoxb-*) ;; *) warn "bot token doesn't start with xoxb-; storing it anyway" ;; esac
    put_secret "$prefix-slack-bot-token" "$v"
  elif ! has_secret_value "$prefix-slack-bot-token"; then
    put_secret "$prefix-slack-bot-token" "$(ask_hidden "Slack Bot User OAuth Token for $env (xoxb-...)")"
  fi

  # Signing secret (required for HTTP mode, which is what Cloud Run uses)
  pick "${ENV_UPPER}_SLACK_SIGNING_SECRET" SLACK_SIGNING_SECRET; v="$PICK_VALUE"
  if [ -n "$v" ]; then
    echo "   signing secret from \$$PICK_NAME"
    put_secret "$prefix-slack-signing-secret" "$v"
  elif ! has_secret_value "$prefix-slack-signing-secret"; then
    echo "   SLACK_SIGNING_SECRET is not in $PROFILE_FILE."
    echo "   Find it at api.slack.com/apps -> your app -> Basic Information -> App Credentials -> Signing Secret"
    put_secret "$prefix-slack-signing-secret" "$(ask_hidden "Slack Signing Secret for $env")"
  fi

  # App-level token (only used with SLACK_MODE=socket)
  pick "${ENV_UPPER}_SLACK_APP_TOKEN" SLACK_APP_TOKEN SLACK_SOCKET_TOKEN; v="$PICK_VALUE"
  if [ -n "$v" ]; then
    echo "   app token from \$$PICK_NAME (used only if SLACK_MODE=socket)"
    put_secret "$prefix-slack-app-token" "$v"
  fi

  # Qdrant Cloud API key (RAG). One cluster can serve both envs: collections are prefixed per env.
  pick "${ENV_UPPER}_QDRANT_API_KEY" QDRANT_API_KEY; v="$PICK_VALUE"
  if [ -n "$v" ]; then
    echo "   qdrant api key from \$$PICK_NAME"
    put_secret "$prefix-qdrant-api-key" "$v"
  elif ! has_secret_value "$prefix-qdrant-api-key"; then
    warn "QDRANT_API_KEY not found: RAG stays off until you add it (and the QDRANT_URL GitHub variable)"
  fi

  # Cohere (optional re-ranker)
  pick "${ENV_UPPER}_COHERE_API_KEY" COHERE_API_KEY; v="$PICK_VALUE"
  if [ -n "$v" ]; then
    echo "   cohere api key from \$$PICK_NAME"
    put_secret "$prefix-cohere-api-key" "$v"
  fi

  # API key for /api/* — generated once, never needs to be in your profile
  if ! has_secret_value "$prefix-api-key"; then
    put_secret "$prefix-api-key" "$(openssl rand -hex 24)"
    echo "   read it with: gcloud secrets versions access latest --secret=$prefix-api-key"
  else
    echo "   $prefix-api-key: already set"
  fi
done
echo
echo "Running services read secrets when an instance starts. To pick up new values now:"
for env in $envs; do echo "  make restart ENV=$env"; done
