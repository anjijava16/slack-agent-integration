"""
👍/👎 feedback with plain (sync) Bolt + Socket Mode, written to PostgreSQL.

This is your upvote/downvote snippet with the JSONL file swapped for a Postgres
table, kept in the same style so the two are easy to compare. The full app does
the same thing async (asyncpg) in app/db/feedback_store.py.

What changed vs. the JSONL version, and why:
  - feedback.jsonl + a threading.Lock  ->  a psycopg connection pool. The DB does
    the locking, and every replica writes to the same place.
  - append-only lines                   ->  INSERT ... ON CONFLICT (run_id, user_id)
    DO UPDATE. One vote per person per answer; clicking again or switching
    👍 -> 👎 updates the row instead of piling up duplicates.
  - `logger` inside add_feedback()      ->  module-level logger. In the original,
    `logger` only existed as a listener argument, so add_feedback() would raise
    NameError at its last line unless a global logger happened to exist.
  - every answer is stored in agent_runs first, so a vote can be joined to the
    question/answer it judged (the button's value is that run_id).

Run:
    docker compose up -d postgres          # or any Postgres you have
    pip install -r requirements-examples.txt
    python examples/03_feedback_postgres_sync.py

Needs in .env: SLACK_BOT_TOKEN, SLACK_SOCKET_TOKEN (xapp-...), OPENAI_API_KEY,
DATABASE_URL=postgresql://agent:agent@localhost:5432/agent
"""

import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("feedback-example")

app = App(token=os.environ["SLACK_BOT_TOKEN"])
llm = OpenAI()

# --------------------------------------------------
# Database: a small pool, schema applied once at startup
# --------------------------------------------------
pool = ConnectionPool(os.environ["DATABASE_URL"], min_size=1, max_size=5, open=True)
SCHEMA = (Path(__file__).resolve().parent.parent / "app" / "db" / "schema.sql").read_text()
with pool.connection() as conn:
    conn.execute(SCHEMA)  # idempotent: CREATE ... IF NOT EXISTS


# --------------------------------------------------
# 1. @mention -> answer with 👍/👎 buttons
# --------------------------------------------------
@app.event("app_mention")
def handle_mention(event, say, context):
    question = event.get("text", "").replace(f"<@{context.bot_user_id}>", "").strip()
    thread_ts = event.get("thread_ts") or event["ts"]
    started = datetime.now(UTC)

    response = llm.responses.create(model="gpt-4o", input=question or "Say hi briefly.")
    answer = response.output_text

    run_id = str(uuid.uuid4())
    save_run(
        run_id=run_id,
        thread_id=f"slack:{event['channel']}:{thread_ts}",
        user_id=event.get("user"),
        channel_id=event["channel"],
        question=question,
        answer=answer,
        duration_ms=int((datetime.now(UTC) - started).total_seconds() * 1000),
    )

    say(
        text=answer,
        thread_ts=thread_ts,
        blocks=[
            {"type": "section", "text": {"type": "mrkdwn", "text": answer[:2900]}},
            {
                "type": "actions",
                "block_id": "agent_feedback",
                "elements": [
                    {
                        "type": "button",
                        "action_id": "upvote",
                        "value": run_id,
                        "text": {"type": "plain_text", "text": ":thumbsup: Helpful", "emoji": True},
                    },
                    {
                        "type": "button",
                        "action_id": "downvote",
                        "value": run_id,
                        "text": {"type": "plain_text", "text": ":thumbsdown: Not helpful", "emoji": True},
                    },
                ],
            },
        ],
    )


# --------------------------------------------------
# 4a. @app.action() — Upvote
# --------------------------------------------------
@app.action("upvote")
def handle_upvote(ack, body, client):
    ack()
    tally = add_feedback(body, "upvote")
    confirm(client, body, tally)
    logger.info("Positive feedback received")


# --------------------------------------------------
# 4b. @app.action() — Downvote
# --------------------------------------------------
@app.action("downvote")
def handle_downvote(ack, body, client):
    ack()
    tally = add_feedback(body, "downvote")
    confirm(client, body, tally)
    logger.info("Negative feedback received")


# --------------------------------------------------
# Shared feedback function — now writes to PostgreSQL
# --------------------------------------------------
UPSERT_FEEDBACK = """
INSERT INTO agent_feedback (feedback_id, run_id, feedback_type, user_id, team_id,
                            channel_id, message_ts, created_at, updated_at)
VALUES (%(feedback_id)s, %(run_id)s, %(feedback_type)s, %(user_id)s, %(team_id)s,
        %(channel_id)s, %(message_ts)s, %(created_at)s, %(created_at)s)
ON CONFLICT (run_id, user_id) DO UPDATE
   SET feedback_type = EXCLUDED.feedback_type,
       message_ts    = EXCLUDED.message_ts,
       updated_at    = EXCLUDED.updated_at
"""

TALLY = """
SELECT COUNT(*) FILTER (WHERE feedback_type = 'upvote'),
       COUNT(*) FILTER (WHERE feedback_type = 'downvote')
FROM agent_feedback WHERE run_id = %s
"""


def add_feedback(body, feedback_type: str):
    actions = body.get("actions", [])
    action = actions[0] if actions else {}

    user = body.get("user", {})
    message = body.get("message", {})
    channel = body.get("channel", {})
    team = body.get("team") or {}

    record = {
        "feedback_id": str(uuid.uuid4()),
        "run_id": action.get("value"),
        "feedback_type": feedback_type,
        "user_id": user.get("id"),
        "team_id": team.get("id"),
        "channel_id": channel.get("id"),
        "message_ts": message.get("ts"),
        "created_at": datetime.now(UTC),
    }

    try:
        # `with pool.connection()` checks out a connection, commits on success,
        # rolls back on error, and returns it to the pool.
        with pool.connection() as conn:
            conn.execute(UPSERT_FEEDBACK, record)
            ups, downs = conn.execute(TALLY, (record["run_id"],)).fetchone()
    except Exception:
        logger.exception("Could not save feedback run_id=%s", record["run_id"])
        return None

    logger.info("Feedback saved: type=%s run_id=%s (+%s/-%s)", feedback_type, record["run_id"], ups, downs)
    return ups, downs


def save_run(**run):
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_runs (run_id, thread_id, source, user_id, channel_id,
                                    question, answer, tool_calls, model, duration_ms)
            VALUES (%(run_id)s, %(thread_id)s, 'slack_mention', %(user_id)s, %(channel_id)s,
                    %(question)s, %(answer)s, %(tool_calls)s, 'gpt-4o', %(duration_ms)s)
            ON CONFLICT (run_id) DO NOTHING
            """,
            {**run, "tool_calls": Jsonb([])},
        )


def confirm(client, body, tally):
    """Ephemeral = only the person who clicked sees it."""
    text = (
        ":warning: Couldn't save your vote, please try again."
        if tally is None
        else f"Thanks! This answer: :thumbsup: {tally[0]}  :thumbsdown: {tally[1]}"
    )
    client.chat_postEphemeral(channel=body["channel"]["id"], user=body["user"]["id"], text=text)


@app.event("message")
def ignore_other_messages():
    pass


# --------------------------------------------------
# Start the Slack Socket Mode connection
# --------------------------------------------------
if __name__ == "__main__":
    try:
        SocketModeHandler(app, os.environ.get("SLACK_SOCKET_TOKEN") or os.environ["SLACK_APP_TOKEN"]).start()
    finally:
        pool.close()
