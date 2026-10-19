"""
Slack tests without Slack.

1. HTTP mode: signed requests hit /slack/events through FastAPI exactly like
   Slack would send them (signature, url_verification, slash command).
2. Listener logic: we dispatch an app_mention event straight into the Bolt app
   with the Web API client stubbed, then check what would have been posted.
"""

import asyncio
import hashlib
import hmac
import json
import time
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from slack_bolt.request.async_request import AsyncBoltRequest
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.web.async_slack_response import AsyncSlackResponse

from app.config import Settings
from app.main import create_app
from app.services.agent_service import AgentService
from app.slack.bolt_app import create_bolt_app
from tests.fakes import ScriptedChatModel, tool_call

SECRET = "test-signing-secret"


def signed_headers(body: str, content_type: str) -> dict:
    ts = str(int(time.time()))
    sig = "v0=" + hmac.new(SECRET.encode(), f"v0:{ts}:{body}".encode(), hashlib.sha256).hexdigest()
    return {"X-Slack-Request-Timestamp": ts, "X-Slack-Signature": sig, "Content-Type": content_type}


def fake_response(data: dict) -> AsyncSlackResponse:
    return AsyncSlackResponse(client=None, http_verb="POST", api_url="", req_args={}, data=data, headers={}, status_code=200)


class RecordingClient:
    """Stubs the Web API methods our listeners call. Bolt hands each listener a
    per-request copy of the client, so we patch the class, not an instance."""

    def __init__(self, monkeypatch):
        self.calls: list[tuple[str, dict]] = []
        for name in ("chat_postMessage", "chat_update", "reactions_add", "reactions_remove", "chat_postEphemeral"):
            monkeypatch.setattr(AsyncWebClient, name, self._recorder(name))
        monkeypatch.setattr(AsyncWebClient, "auth_test", self._auth_test)

    def _recorder(self, name):
        calls = self.calls

        async def method(_client, **kwargs):
            calls.append((name, kwargs))
            return fake_response({"ok": True, "ts": "999.0001", "channel": kwargs.get("channel")})

        return method

    @staticmethod
    async def _auth_test(_client, **kwargs):
        return fake_response({"ok": True, "team_id": "T1", "user_id": "UBOT", "bot_id": "B1", "url": "https://x.slack.com/"})


def http_settings(**kw) -> Settings:
    return Settings(
        _env_file=None,
        database_url="",
        slack_mode="http",
        slack_bot_token="xoxb-test",
        slack_signing_secret=SECRET,
        openai_api_key="",
        **kw,
    )


# ------------------------------------------------------------------ HTTP mode
def test_url_verification_challenge():
    with TestClient(create_app(settings=http_settings(), llm=ScriptedChatModel())) as c:
        body = json.dumps({"type": "url_verification", "challenge": "abc123", "token": "x"})
        r = c.post("/slack/events", content=body, headers=signed_headers(body, "application/json"))
        assert r.status_code == 200
        assert r.json()["challenge"] == "abc123"


def test_unsigned_request_rejected():
    with TestClient(create_app(settings=http_settings(), llm=ScriptedChatModel())) as c:
        body = json.dumps({"type": "url_verification", "challenge": "abc"})
        r = c.post(
            "/slack/events",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Slack-Signature": "v0=bad",
                "X-Slack-Request-Timestamp": str(int(time.time())),
            },
        )
        assert r.status_code == 401


def test_slash_command_add_over_http(monkeypatch):
    RecordingClient(monkeypatch)
    app = create_app(settings=http_settings(), llm=ScriptedChatModel())
    with TestClient(app) as c:
        form = urlencode(
            {
                "command": "/agent-add",
                "text": "2 3.5",
                "user_id": "U1",
                "channel_id": "C1",
                "team_id": "T1",
                "response_url": "https://hooks.slack.com/x",
                "trigger_id": "1",
            }
        )
        r = c.post("/slack/events", content=form, headers=signed_headers(form, "application/x-www-form-urlencoded"))
        assert r.status_code == 200
        assert "5.5" in r.text


# ------------------------------------------------------------------ listener logic
@pytest.mark.asyncio
async def test_app_mention_runs_agent_and_updates_placeholder(monkeypatch):
    llm = ScriptedChatModel(
        script=[
            tool_call("search_knowledge_base", {"query": "deploy freeze"}),
            AIMessage(content="**No deploys** from the 2nd Friday of Dec to Jan 2 (KB-004)."),
        ]
    )
    settings = Settings(_env_file=None, database_url="", slack_mode="socket", slack_bot_token="xoxb-test", openai_api_key="")
    bolt = create_bolt_app(settings, AgentService(llm))
    rec = RecordingClient(monkeypatch)

    body = {
        "type": "event_callback",
        "team_id": "T1",
        "api_app_id": "A1",
        "event_id": "Ev1",
        "event": {
            "type": "app_mention",
            "user": "U42",
            "text": "<@UBOT> when is the deploy freeze?",
            "ts": "1700000000.000100",
            "channel": "C9",
            "event_ts": "1700000000.000100",
        },
    }
    await bolt.async_dispatch(AsyncBoltRequest(body=body, mode="socket_mode"))

    for _ in range(100):  # listener runs after the ack; give it a moment
        if any(name == "chat_update" for name, _ in rec.calls):
            break
        await asyncio.sleep(0.02)

    names = [n for n, _ in rec.calls]
    assert "chat_postMessage" in names and "chat_update" in names
    placeholder = next(kw for n, kw in rec.calls if n == "chat_postMessage")
    assert placeholder["thread_ts"] == "1700000000.000100"  # replied in thread
    update = next(kw for n, kw in rec.calls if n == "chat_update")
    assert update["text"].startswith("*No deploys*")  # markdown converted to mrkdwn
    assert any("search_knowledge_base" in json.dumps(b) for b in update["blocks"])
    # the bot mention was stripped before reaching the LLM
    assert llm.calls[0][-1].content == "when is the deploy freeze?"


@pytest.mark.asyncio
async def test_duplicate_event_is_ignored(monkeypatch):
    llm = ScriptedChatModel(script=[AIMessage(content="once")])
    settings = Settings(_env_file=None, database_url="", slack_mode="socket", slack_bot_token="xoxb-test", openai_api_key="")
    bolt = create_bolt_app(settings, AgentService(llm))
    rec = RecordingClient(monkeypatch)
    event = {"type": "app_mention", "user": "U1", "text": "<@UBOT> hi there", "ts": "1.1", "channel": "C1"}
    body = {"type": "event_callback", "team_id": "T1", "event_id": "Ev2", "event": event}
    await bolt.async_dispatch(AsyncBoltRequest(body=body, mode="socket_mode"))
    await bolt.async_dispatch(AsyncBoltRequest(body=dict(body, event_id="Ev2-retry"), mode="socket_mode"))
    await asyncio.sleep(0.3)
    assert sum(1 for n, _ in rec.calls if n == "chat_update") == 1
