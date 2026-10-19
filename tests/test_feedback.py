"""
👍/👎 feedback tests.

The store tests run against every implementation: in-memory and SQLite always,
and real PostgreSQL when TEST_DATABASE_URL is set, e.g.

    TEST_DATABASE_URL=postgresql://agent:agent@localhost:5432/agent pytest -q

(`docker compose up -d postgres` gives you one on that URL.) The Postgres
fixture TRUNCATEs the two tables, so point it at a throwaway database.
"""

import asyncio
import os
import uuid
from datetime import UTC

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from slack_bolt.request.async_request import AsyncBoltRequest

from app.config import Settings
from app.db.feedback_store import (
    InMemoryFeedbackStore,
    PostgresFeedbackStore,
    RunRecord,
    SqliteFeedbackStore,
    UnknownRunError,
    feedback_from_body,
)
from app.main import create_app
from app.services.agent_service import AgentService
from app.slack.bolt_app import create_bolt_app
from tests.fakes import ScriptedChatModel, tool_call
from tests.test_slack import RecordingClient

PG_URL = os.environ.get("TEST_DATABASE_URL")


def click(run_id: str, action_id: str = "upvote", user: str = "U1", channel: str = "C9", ts: str = "999.0001") -> dict:
    """A trimmed but realistic block_actions payload, as Slack sends it."""
    return {
        "type": "block_actions",
        "team": {"id": "T1", "domain": "acme"},
        "user": {"id": user, "username": "anji", "team_id": "T1"},
        "api_app_id": "A1",
        "container": {"type": "message", "message_ts": ts, "channel_id": channel, "is_ephemeral": False},
        "channel": {"id": channel, "name": "it-help"},
        "message": {"type": "message", "ts": ts, "text": "answer…"},
        "response_url": "https://hooks.slack.com/actions/T1/1/abc",
        "trigger_id": "1.2.3",
        "actions": [
            {
                "type": "button",
                "action_id": action_id,
                "block_id": "agent_feedback",
                "value": run_id,
                "action_ts": "1700000001.0",
            }
        ],
    }


def a_run(run_id: str | None = None, **kw) -> RunRecord:
    return RunRecord(
        run_id=run_id or str(uuid.uuid4()),
        thread_id=kw.get("thread_id", "slack:C9:1.1"),
        source=kw.get("source", "slack_mention"),
        user_id="U42",
        channel_id="C9",
        question=kw.get("question", "when is the deploy freeze?"),
        answer=kw.get("answer", "2nd Friday of December to Jan 2 (KB-004)."),
        tool_calls=[{"name": "search_knowledge_base", "args": {"query": "deploy freeze"}}],
        model="gpt-4o",
        duration_ms=1234,
    )


# ------------------------------------------------------------------ payload parsing
def test_feedback_from_body_extracts_fields():
    rid = str(uuid.uuid4())
    fb = feedback_from_body(click(rid, "downvote", user="U7", channel="C3", ts="111.222"), "downvote")
    assert (fb.run_id, fb.feedback_type, fb.user_id, fb.channel_id, fb.message_ts, fb.team_id) == (
        rid,
        "downvote",
        "U7",
        "C3",
        "111.222",
        "T1",
    )
    uuid.UUID(fb.feedback_id)


@pytest.mark.parametrize("value", [None, "", "slack:C1:1.1", "not-a-uuid"])
def test_feedback_from_body_rejects_bad_run_ids(value):
    body = click("x")
    body["actions"][0]["value"] = value
    with pytest.raises(ValueError):
        feedback_from_body(body, "upvote")


# ------------------------------------------------------------------ store contract (both backends)
@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def store(request, tmp_path):
    if request.param in ("memory", "sqlite"):
        s = InMemoryFeedbackStore() if request.param == "memory" else SqliteFeedbackStore(str(tmp_path / "agent.db"))
        await s.init()
        yield s
        await s.close()
        return
    if not PG_URL:
        pytest.skip("set TEST_DATABASE_URL to run against PostgreSQL")
    s = PostgresFeedbackStore(PG_URL)
    await s.init()
    await s._pool.execute("TRUNCATE agent_feedback, agent_runs, slack_event_claims, ingest_jobs")
    yield s
    await s.close()


async def test_vote_lifecycle(store):
    run = a_run()
    await store.record_run(run)

    t1 = await store.add_feedback(feedback_from_body(click(run.run_id, user="U1"), "upvote"))
    assert (t1.upvotes, t1.downvotes, t1.changed) == (1, 0, "new")

    # same person clicks 👍 again -> no duplicate row
    t2 = await store.add_feedback(feedback_from_body(click(run.run_id, user="U1"), "upvote"))
    assert (t2.upvotes, t2.downvotes, t2.changed) == (1, 0, "unchanged")

    # changes their mind
    t3 = await store.add_feedback(feedback_from_body(click(run.run_id, "downvote", user="U1"), "downvote"))
    assert (t3.upvotes, t3.downvotes, t3.changed) == (0, 1, "changed")

    # someone else agrees with the original answer
    t4 = await store.add_feedback(feedback_from_body(click(run.run_id, user="U2"), "upvote"))
    assert (t4.upvotes, t4.downvotes, t4.changed) == (1, 1, "new")

    got = await store.get_run(run.run_id)
    assert got["question"] == run.question
    assert got["tool_calls"] == [{"name": "search_knowledge_base", "args": {"query": "deploy freeze"}}]
    assert sorted((v["user_id"], v["feedback_type"]) for v in got["feedback"]) == [("U1", "downvote"), ("U2", "upvote")]


async def test_unknown_run_is_rejected(store):
    with pytest.raises(UnknownRunError):
        await store.add_feedback(feedback_from_body(click(str(uuid.uuid4())), "upvote"))


async def test_record_run_is_idempotent(store):
    run = a_run()
    await store.record_run(run)
    await store.record_run(run)  # Slack retry, double call — must not raise
    assert (await store.get_run(run.run_id))["answer"] == run.answer


async def test_summary(store):
    good, bad = a_run(question="good one"), a_run(question="bad one", source="slash_command")
    await store.record_run(good)
    await store.record_run(bad)
    for user in ("U1", "U2", "U3"):
        await store.add_feedback(feedback_from_body(click(good.run_id, user=user), "upvote"))
    await store.add_feedback(feedback_from_body(click(bad.run_id, "downvote", user="U1"), "downvote"))
    await store.add_feedback(feedback_from_body(click(bad.run_id, "downvote", user="U2"), "downvote"))

    s = await store.summary(days=7)
    assert (s["runs"], s["upvotes"], s["downvotes"], s["approval_rate"]) == (2, 3, 2, 0.6)
    assert s["most_downvoted"][0]["question"] == "bad one"
    assert {r["source"]: (r["upvotes"], r["downvotes"]) for r in s["by_source"]} == {
        "slack_mention": (3, 0),
        "slash_command": (0, 2),
    }


async def test_get_run_unknown(store):
    assert await store.get_run(str(uuid.uuid4())) is None
    assert await store.get_run("garbage") is None


# ------------------------------------------------------------------ Slack: answer -> click -> row
async def _wait_for(pred, tries=100):
    for _ in range(tries):
        if pred():
            return
        await asyncio.sleep(0.02)


async def test_slack_upvote_and_downvote_end_to_end(monkeypatch):
    llm = ScriptedChatModel(
        script=[tool_call("search_knowledge_base", {"query": "pto"}), AIMessage(content="21 days a year (KB-003).")]
    )
    store = InMemoryFeedbackStore()
    agent = AgentService(llm, store=store, model_name="gpt-4o")
    settings = Settings(_env_file=None, database_url="", slack_mode="socket", slack_bot_token="xoxb-test", openai_api_key="")
    bolt = create_bolt_app(settings, agent, store=store)
    rec = RecordingClient(monkeypatch)

    # 1. someone mentions the bot -> agent answers -> run recorded, buttons carry run_id
    mention = {
        "type": "event_callback",
        "team_id": "T1",
        "event_id": "EvF1",
        "event": {"type": "app_mention", "user": "U42", "text": "<@UBOT> pto?", "ts": "1.1", "channel": "C9"},
    }
    await bolt.async_dispatch(AsyncBoltRequest(body=mention, mode="socket_mode"))
    await _wait_for(lambda: any(n == "chat_update" for n, _ in rec.calls))

    update = next(kw for n, kw in rec.calls if n == "chat_update")
    actions = next(b for b in update["blocks"] if b["type"] == "actions")
    assert [e["action_id"] for e in actions["elements"]] == ["upvote", "downvote"]
    run_id = actions["elements"][0]["value"]
    assert store.runs[run_id]["question"] == "pto?"
    assert store.runs[run_id]["tool_calls"] == [{"name": "search_knowledge_base", "args": {"query": "pto"}}]

    # 2. U1 clicks 👍, U2 clicks 👎
    await bolt.async_dispatch(AsyncBoltRequest(body=click(run_id, "upvote", user="U1"), mode="socket_mode"))
    await bolt.async_dispatch(AsyncBoltRequest(body=click(run_id, "downvote", user="U2"), mode="socket_mode"))
    await _wait_for(lambda: sum(n == "chat_postEphemeral" for n, _ in rec.calls) == 2)

    assert {u: v["feedback_type"] for (_, u), v in store.feedback.items()} == {"U1": "upvote", "U2": "downvote"}
    ephemerals = [kw for n, kw in rec.calls if n == "chat_postEphemeral"]
    assert ephemerals[0]["user"] == "U1" and ":thumbsup: 1" in ephemerals[0]["text"]
    assert ephemerals[1]["user"] == "U2" and ":thumbsdown: 1" in ephemerals[1]["text"]


# ------------------------------------------------------------------ REST
def test_rest_feedback_endpoints():
    store = InMemoryFeedbackStore()
    llm = ScriptedChatModel(script=[AIMessage(content="It's 42.")])
    settings = Settings(_env_file=None, database_url="", slack_bot_token="", openai_api_key="")
    with TestClient(create_app(settings=settings, llm=llm, store=store)) as c:
        run_id = c.post("/api/agent/invoke", json={"message": "meaning of life?"}).json()["run_id"]

        r = c.post("/api/feedback", json={"run_id": run_id, "feedback_type": "upvote", "user_id": "web-1"})
        assert r.json() == {"run_id": run_id, "upvotes": 1, "downvotes": 0, "changed": "new"}

        run = c.get(f"/api/runs/{run_id}").json()
        assert run["source"] == "api" and run["feedback"][0]["user_id"] == "web-1"

        summary = c.get("/api/feedback/summary?days=1").json()
        assert summary["upvotes"] == 1 and summary["approval_rate"] == 1.0

        assert (
            c.post("/api/feedback", json={"run_id": str(uuid.uuid4()), "feedback_type": "upvote", "user_id": "x"}).status_code
            == 404
        )
        assert c.post("/api/feedback", json={"run_id": "nope", "feedback_type": "upvote", "user_id": "x"}).status_code == 422
        assert c.post("/api/feedback", json={"run_id": run_id, "feedback_type": "meh", "user_id": "x"}).status_code == 422
        assert c.get(f"/api/runs/{uuid.uuid4()}").status_code == 404


async def test_event_claims_dedupe(store):
    assert await store.claim_event("C1:1700000000.000100") is True
    assert await store.claim_event("C1:1700000000.000100") is False  # Slack retry / other instance
    assert await store.claim_event("C1:1700000000.000200") is True


async def test_ingest_jobs_lifecycle(store):
    from datetime import datetime

    from app.db.jobs import JobRepo

    jobs = JobRepo(store)
    job = await jobs.create("llm", "faq", {"x": 1})
    assert job["status"] == "queued" and job["params"] == {"x": 1}
    await jobs.update(job["job_id"], status="running", started_at=datetime.now(UTC))
    await jobs.update(job["job_id"], status="succeeded", result={"embedded": 3, "errors": []}, finished_at=datetime.now(UTC))
    got = await jobs.get(job["job_id"])
    assert got["status"] == "succeeded" and got["result"] == {"embedded": 3, "errors": []} and got["finished_at"]
    other = await jobs.create("de", "github", {})
    assert [j["job_id"] for j in await jobs.list(course="llm")] == [job["job_id"]]
    assert {j["job_id"] for j in await jobs.list()} == {job["job_id"], other["job_id"]}
    assert await jobs.get("00000000-0000-0000-0000-000000000000") is None
