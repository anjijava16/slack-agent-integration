#!/usr/bin/env bash
# Print a Slack app manifest pointing at a deployed environment, ready to paste.
#
#   bash scripts/slack-manifest.sh test     # or prod
#
# New app:      api.slack.com/apps -> Create New App -> From a manifest -> paste
# Existing app: api.slack.com/apps -> your app -> App Manifest -> paste -> Save
#   (Slack then sends a url_verification request; the service answers it.)
# Slack apps are tied to one request URL, so use one app per environment.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/config.sh
source scripts/config.sh
env="${1:?usage: slack-manifest.sh test|prod}"
svc="${SERVICE}-${env}"

url=$(gcloud run services describe "$svc" --project "$PROJECT_ID" --region "$REGION" --format='value(status.url)' 2>/dev/null || true)
[ -n "$url" ] || { echo "!! Cloud Run service $svc not found. Has CD deployed it yet?" >&2; exit 1; }

name="Ops Buddy"; cmd_suffix=""
if [ "$env" != "prod" ]; then name="Ops Buddy ($env)"; cmd_suffix="-$env"; fi

mkdir -p slack_manifests/out
outfile="slack_manifests/out/manifest.$env.yaml"
sed -e "s#https://YOUR-HOST#${url}#g" \
    -e "s#^  name: Ops Buddy#  name: ${name}#" \
    -e "s#^    display_name: Ops Buddy#    display_name: ${name}#" \
    -e "s#command: /agent-add#command: /agent-add${cmd_suffix}#" \
    -e "s#command: /agent\$#command: /agent${cmd_suffix}#" \
    -e "s#command: /faq\$#command: /faq${cmd_suffix}#" \
    slack_manifests/manifest.http.yaml > "$outfile"

cat "$outfile"
echo
echo "# Written to $outfile"
echo "# Request URL: ${url}/slack/events"
if [ -n "$cmd_suffix" ]; then
  echo "# Slash commands are /agent${cmd_suffix}, /agent-add${cmd_suffix} and /faq${cmd_suffix} here, so they don't clash with prod."
  echo "# The CD workflow already sets SLACK_CMD_AGENT / SLACK_CMD_ADD to match."
fi
