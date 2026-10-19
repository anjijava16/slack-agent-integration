#!/usr/bin/env bash
# Exercise the REST API. Start the server first:  make run
set -euo pipefail
BASE=${BASE:-http://localhost:8000}
KEY_HEADER=()
[[ -n "${API_KEY:-}" ]] && KEY_HEADER=(-H "X-API-Key: $API_KEY")

echo "== health";            curl -s "$BASE/health"; echo
echo "== policy + math";     curl -s "${KEY_HEADER[@]}" -X POST "$BASE/api/agent/invoke" -H 'Content-Type: application/json' \
  -d '{"message":"How many PTO days do I get per year and how many is that per quarter?","thread_id":"api:demo"}'; echo
echo "== follow-up (memory)"; curl -s "${KEY_HEADER[@]}" -X POST "$BASE/api/agent/invoke" -H 'Content-Type: application/json' \
  -d '{"message":"And how many roll over?","thread_id":"api:demo"}'; echo
echo "== ticket";            curl -s "${KEY_HEADER[@]}" -X POST "$BASE/api/agent/invoke" -H 'Content-Type: application/json' \
  -d '{"message":"My VPN drops every 5 minutes since this morning, open a ticket","user_id":"U_DEMO"}'; echo
echo "== history";           curl -s "${KEY_HEADER[@]}" "$BASE/api/agent/threads/api:demo"; echo
if [[ -n "${SLACK_CHANNEL_ID:-}" ]]; then
  echo "== agent -> slack";  curl -s "${KEY_HEADER[@]}" -X POST "$BASE/api/agent/invoke" -H 'Content-Type: application/json' \
    -d "{\"message\":\"Summarize the deploy freeze policy\",\"post_to_slack_channel\":\"$SLACK_CHANNEL_ID\"}"; echo
  echo "== notify";          curl -s "${KEY_HEADER[@]}" -X POST "$BASE/api/slack/notify" -H 'Content-Type: application/json' \
    -d '{"text":"Nightly ETL finished :white_check_mark:"}'; echo
fi

echo "== vote on an answer"
RUN=$(curl -s "${KEY_HEADER[@]}" -X POST "$BASE/api/agent/invoke" -H 'Content-Type: application/json' \
  -d '{"message":"How many PTO days roll over?"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["run_id"])')
curl -s "${KEY_HEADER[@]}" -X POST "$BASE/api/feedback" -H 'Content-Type: application/json' \
  -d "{\"run_id\":\"$RUN\",\"feedback_type\":\"upvote\",\"user_id\":\"demo-user\"}"; echo
echo "== run + votes";       curl -s "${KEY_HEADER[@]}" "$BASE/api/runs/$RUN"; echo
echo "== feedback summary";  curl -s "${KEY_HEADER[@]}" "$BASE/api/feedback/summary?days=7"; echo

# ------------------------------------------------------------------ course FAQ RAG (needs Qdrant: make qdrant)
echo "== rag status";        curl -s "${KEY_HEADER[@]}" "$BASE/api/rag/status"; echo
echo "== ingest llm faq (background job)"
JOB=$(curl -s "${KEY_HEADER[@]}" -X POST "$BASE/api/ingest/faq" -H 'Content-Type: application/json' \
  -d '{"course":"llm"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')
for _ in $(seq 1 60); do
  STATE=$(curl -s "${KEY_HEADER[@]}" "$BASE/api/ingest/jobs/$JOB" | python3 -c 'import sys,json;print(json.load(sys.stdin)["status"])')
  [ "$STATE" = succeeded ] || [ "$STATE" = failed ] && break; sleep 2
done
curl -s "${KEY_HEADER[@]}" "$BASE/api/ingest/jobs/$JOB"; echo
echo "== rag search (hybrid, no LLM)"
curl -s "${KEY_HEADER[@]}" -X POST "$BASE/api/rag/search" -H 'Content-Type: application/json' \
  -d '{"course":"llm","query":"Connection refused elasticsearch 9200","final_top_n":3}'; echo
echo "== rag query (answer + sources)"
curl -s "${KEY_HEADER[@]}" -X POST "$BASE/api/rag/query" -H 'Content-Type: application/json' \
  -d '{"course":"llm","question":"I just found the course, can I still join?"}'; echo
