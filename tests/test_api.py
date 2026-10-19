from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.config import Settings
from app.main import create_app
from tests.fakes import ScriptedChatModel, tool_call


def make_client(llm, **overrides):
    settings = Settings(_env_file=None, database_url="", slack_bot_token="", openai_api_key="", **overrides)
    return TestClient(create_app(settings=settings, llm=llm))


def test_health():
    with make_client(ScriptedChatModel()) as c:
        body = c.get("/health").json()
    assert body == {
        "status": "ok",
        "env": "local",
        "slack": "disabled",
        "model": "gpt-4o",
        "database": "in-memory",
        "rag": "qdrant:memory",
    }
    with make_client(ScriptedChatModel()) as c:
        assert c.get("/ready").json() == {"status": "ready", "database": "memory", "qdrant": "ok"}


def test_invoke_and_history():
    llm = ScriptedChatModel(
        script=[
            tool_call("calculator", {"expression": "21/4"}),
            AIMessage(content="21 PTO days a year, about *5.25* per quarter (KB-003)."),
        ]
    )
    with make_client(llm) as c:
        r = c.post("/api/agent/invoke", json={"message": "PTO per quarter?", "thread_id": "api:demo"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["tool_calls"][0] == {"name": "calculator", "args": {"expression": "21/4"}, "output": "5.25"}
        assert body["thread_id"] == "api:demo"

        hist = c.get("/api/agent/threads/api:demo").json()
        assert [m["role"] for m in hist["messages"]] == ["user", "assistant", "tool", "assistant"]
        assert c.get("/api/agent/threads/nope").status_code == 404


def test_api_key_enforced():
    with make_client(ScriptedChatModel(script=[AIMessage(content="ok")]), api_key="s3cret") as c:
        assert c.post("/api/agent/invoke", json={"message": "hi"}).status_code == 401
        ok = c.post("/api/agent/invoke", json={"message": "hi"}, headers={"X-API-Key": "s3cret"})
        assert ok.status_code == 200


def test_notify_without_slack_is_503():
    with make_client(ScriptedChatModel()) as c:
        assert c.post("/api/slack/notify", json={"text": "hi", "channel": "C1"}).status_code == 503


def test_slack_events_404_when_http_mode_off():
    with make_client(ScriptedChatModel()) as c:
        assert c.post("/slack/events", json={}).status_code == 404
