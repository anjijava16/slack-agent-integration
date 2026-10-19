"""
REST API.

These endpoints give you the *other* direction of the integration:
  - Your systems -> FastAPI -> agent -> (optionally) Slack       POST /api/agent/invoke
  - Your systems -> FastAPI -> Slack (plain notification)       POST /api/slack/notify
  - Debugging: what does the agent remember for a thread?        GET  /api/agent/threads/{id}

Slack itself only ever talks to /slack/events (HTTP mode) or to the WebSocket
(Socket Mode). See app/main.py.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from slack_sdk.errors import SlackApiError

from app.api.schemas import (
    FeedbackIn,
    FeedbackOut,
    HistoryMessage,
    HistoryResponse,
    InvokeRequest,
    InvokeResponse,
    NotifyRequest,
    NotifyResponse,
    ToolCallOut,
)
from app.config import Settings
from app.db.feedback_store import FeedbackRecord, FeedbackStore, UnknownRunError
from app.services.agent_service import AgentService
from app.slack.blocks import answer_blocks, fallback_text

router = APIRouter(prefix="/api", tags=["agent"])


# ---------------------------------------------------------------- dependencies
def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_agent(request: Request) -> AgentService:
    return request.app.state.agent


def get_store(request: Request) -> FeedbackStore:
    return request.app.state.store


def get_slack_client(request: Request):
    bolt = getattr(request.app.state, "bolt", None)
    if bolt is None:
        raise HTTPException(503, "Slack is not configured (SLACK_BOT_TOKEN missing).")
    return bolt.client


async def require_api_key(settings: Settings = Depends(get_settings_dep), x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(401, "Invalid or missing X-API-Key header.")


# ---------------------------------------------------------------- routes
@router.post("/agent/invoke", response_model=InvokeResponse, dependencies=[Depends(require_api_key)])
async def invoke_agent(body: InvokeRequest, request: Request, agent: AgentService = Depends(get_agent)):
    thread_id = body.thread_id or f"api:{uuid.uuid4()}"
    result = await agent.run(body.message, thread_id=thread_id, user_id=body.user_id, source="api")

    slack_ts = None
    if body.post_to_slack_channel:
        client = get_slack_client(request)
        try:
            resp = await client.chat_postMessage(
                channel=body.post_to_slack_channel,
                text=fallback_text(result),
                blocks=[
                    {"type": "context", "elements": [{"type": "mrkdwn", "text": f"API request: _{body.message}_"}]},
                    *answer_blocks(result, with_feedback=False),
                ],
            )
            slack_ts = resp["ts"]
        except SlackApiError as exc:
            raise HTTPException(502, f"Agent answered, but Slack rejected the post: {exc.response['error']}")

    return InvokeResponse(
        answer=result.answer,
        thread_id=result.thread_id,
        run_id=result.run_id,
        tool_calls=[ToolCallOut(name=t.name, args=t.args, output=t.output) for t in result.tool_calls],
        duration_ms=result.duration_ms,
        blocked=result.blocked,
        slack_message_ts=slack_ts,
    )


@router.post("/slack/notify", response_model=NotifyResponse, dependencies=[Depends(require_api_key)])
async def notify_slack(body: NotifyRequest, request: Request, settings: Settings = Depends(get_settings_dep)):
    """The 'push a message into a channel' script, as an endpoint."""
    channel = body.channel or settings.slack_channel_id
    if not channel:
        raise HTTPException(400, "No channel given and SLACK_CHANNEL_ID is not set.")
    client = get_slack_client(request)
    try:
        resp = await client.chat_postMessage(channel=channel, text=body.text, thread_ts=body.thread_ts)
    except SlackApiError as exc:
        # Common ones: not_in_channel (invite the bot), channel_not_found, missing_scope
        raise HTTPException(502, f"Slack error: {exc.response['error']}")
    return NotifyResponse(ok=True, channel=resp["channel"], ts=resp["ts"])


@router.get("/agent/threads/{thread_id:path}", response_model=HistoryResponse, dependencies=[Depends(require_api_key)])
async def thread_history(thread_id: str, agent: AgentService = Depends(get_agent)):
    snapshot = await agent.graph.aget_state({"configurable": {"thread_id": thread_id}})
    messages = (snapshot.values or {}).get("messages", [])
    if not messages:
        raise HTTPException(404, f"No conversation stored for thread {thread_id!r}.")
    out = []
    for m in messages:
        role = {HumanMessage: "user", AIMessage: "assistant", ToolMessage: "tool"}.get(type(m), m.type)
        out.append(
            HistoryMessage(
                role=role,
                content=m.content if isinstance(m.content, str) else str(m.content),
                tool_calls=[c["name"] for c in getattr(m, "tool_calls", []) or []],
            )
        )
    return HistoryResponse(thread_id=thread_id, messages=out)


# ---------------------------------------------------------------- runs & feedback
@router.post("/feedback", response_model=FeedbackOut, tags=["feedback"], dependencies=[Depends(require_api_key)])
async def post_feedback(body: FeedbackIn, store: FeedbackStore = Depends(get_store)):
    """Same upsert the Slack buttons use, for votes from other UIs."""
    try:
        run_id = str(uuid.UUID(body.run_id))
    except ValueError:
        raise HTTPException(422, "run_id must be a UUID")
    record = FeedbackRecord(
        feedback_id=str(uuid.uuid4()),
        run_id=run_id,
        feedback_type=body.feedback_type,
        user_id=body.user_id,
        team_id=None,
        channel_id=body.channel_id,
        message_ts=None,
        created_at=datetime.now(UTC).isoformat(),
    )
    try:
        tally = await store.add_feedback(record)
    except UnknownRunError:
        raise HTTPException(404, f"No run with id {run_id}")
    return FeedbackOut(run_id=run_id, upvotes=tally.upvotes, downvotes=tally.downvotes, changed=tally.changed)


@router.get("/feedback/summary", tags=["feedback"], dependencies=[Depends(require_api_key)])
async def feedback_summary(
    days: int = Query(7, ge=1, le=365), worst: int = Query(5, ge=1, le=50), store: FeedbackStore = Depends(get_store)
):
    """Totals, approval rate, split by source, and the most downvoted answers to go fix."""
    return await store.summary(days=days, worst=worst)


@router.get("/runs/{run_id}", tags=["feedback"], dependencies=[Depends(require_api_key)])
async def get_run(run_id: str, store: FeedbackStore = Depends(get_store)):
    """One answer with its question, tool calls, latency and every vote on it."""
    run = await store.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"No run with id {run_id}")
    return run
