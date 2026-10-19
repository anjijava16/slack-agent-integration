from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class InvokeRequest(BaseModel):
    message: str = Field(..., min_length=1, examples=["How many PTO days do I get, and how many is that per quarter?"])
    thread_id: str | None = Field(None, description="Reuse to continue a conversation. New uuid if empty.")
    user_id: str = Field("api-user", description="Who is asking. Shows up as the ticket requester.")
    post_to_slack_channel: str | None = Field(
        None, description="If set (e.g. C0123ABCD), the answer is also posted to this Slack channel."
    )


class ToolCallOut(BaseModel):
    name: str
    args: dict
    output: str


class InvokeResponse(BaseModel):
    answer: str
    thread_id: str
    run_id: str = Field(description="Pass to POST /api/feedback to vote on this answer")
    tool_calls: list[ToolCallOut]
    duration_ms: int
    blocked: bool
    slack_message_ts: str | None = None


class NotifyRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["Nightly ETL finished :white_check_mark:"])
    channel: str | None = Field(None, description="Defaults to SLACK_CHANNEL_ID")
    thread_ts: str | None = None


class NotifyResponse(BaseModel):
    ok: bool
    channel: str
    ts: str


class HistoryMessage(BaseModel):
    role: str
    content: str
    tool_calls: list[str] = []


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[HistoryMessage]


class FeedbackIn(BaseModel):
    """Vote from a non-Slack client (web UI, eval harness). Slack votes come in via the buttons."""

    run_id: str
    feedback_type: Literal["upvote", "downvote"]
    user_id: str = Field(..., min_length=1)
    channel_id: str | None = None


class FeedbackOut(BaseModel):
    run_id: str
    upvotes: int
    downvotes: int
    changed: Literal["new", "changed", "unchanged"]
