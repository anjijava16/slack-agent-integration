"""
Storage per environment:
  local       -> SQLite file (feedback, runs, event claims AND conversation memory)
  test / prod -> PostgreSQL (refuses to start on anything else)

These tests restart the whole app against the same database and check that
conversation memory and votes survive — which is the point of moving off
in-memory state before Cloud Run starts scaling instances up and down.
"""

import os
import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from app.config import OPENROUTER_BASE_URL, Settings
from app.db.feedback_store import db_kind, sqlite_path
from app.main import create_app
from tests.fakes import ScriptedChatModel

PG_URL = os.environ.get("TEST_DATABASE_URL")


def boot(database_url: str, replies: list[str]) -> TestClient:
    settings = Settings(_env_file=None, database_url=database_url, slack_bot_token="", openai_api_key="")
    return TestClient(create_app(settings=settings, llm=ScriptedChatModel(script=[AIMessage(content=r) for r in replies])))


def _restart_roundtrip(database_url: str):
    thread = f"api:persist-{uuid.uuid4()}"
    # --- first "instance"
    with boot(database_url, ["Nice to meet you, Anji."]) as c:
        assert c.get("/ready").status_code == 200
        run_id = c.post("/api/agent/invoke", json={"message": "My name is Anji", "thread_id": thread}).json()["run_id"]
        c.post("/api/feedback", json={"run_id": run_id, "feedback_type": "upvote", "user_id": "U1"})

    # --- brand-new app object, same database (= Cloud Run starting a fresh instance)
    llm = ScriptedChatModel(script=[AIMessage(content="Your name is Anji.")])
    settings = Settings(_env_file=None, database_url=database_url, slack_bot_token="", openai_api_key="")
    with TestClient(create_app(settings=settings, llm=llm)) as c:
        c.post("/api/agent/invoke", json={"message": "What's my name?", "thread_id": thread})
        # the new instance sent the OLD conversation to the model
        assert [m.content for m in llm.calls[0][1:]] == ["My name is Anji", "Nice to meet you, Anji.", "What's my name?"]
        run = c.get(f"/api/runs/{run_id}").json()
        assert run["feedback"][0]["user_id"] == "U1"
        history = c.get(f"/api/agent/threads/{thread}").json()["messages"]
        assert len(history) == 4


def test_sqlite_survives_restart(tmp_path):
    _restart_roundtrip(f"sqlite:///{tmp_path}/agent.db")
    assert (tmp_path / "agent.db").exists()


@pytest.mark.skipif(not PG_URL, reason="set TEST_DATABASE_URL to run against PostgreSQL")
def test_postgres_survives_restart():
    _restart_roundtrip(PG_URL)


@pytest.mark.skipif(not PG_URL, reason="set TEST_DATABASE_URL to run against PostgreSQL")
def test_postgres_unix_socket_dsn():
    """Cloud Run reaches Cloud SQL through a unix socket under /cloudsql/<instance>.
    The DSN shape is postgresql://user:pw@/db?host=<socket dir>. If a local socket dir
    is given (POSTGRES_SOCKET_DIR), prove both drivers (asyncpg + psycopg) accept it."""
    sock = os.environ.get("POSTGRES_SOCKET_DIR")
    if not sock:
        pytest.skip("set POSTGRES_SOCKET_DIR (e.g. /var/run/postgresql) to test socket DSNs")
    from urllib.parse import urlparse

    u = urlparse(PG_URL)
    _restart_roundtrip(f"postgresql://{u.username}:{u.password}@{u.path}?host={sock}")


# ------------------------------------------------------------------ config rules
def test_cloud_envs_refuse_sqlite():
    for env in ("test", "prod"):
        with pytest.raises(ValidationError, match="requires a PostgreSQL"):
            Settings(_env_file=None, app_env=env, database_url="sqlite:///./data/agent.db")
        with pytest.raises(ValidationError, match="requires a PostgreSQL"):
            Settings(_env_file=None, app_env=env, database_url="")
    ok = Settings(_env_file=None, app_env="prod", database_url="postgresql://u:p@/db?host=/cloudsql/p:r:i", rag_enabled=False)
    assert ok.db_kind == "postgres"


def test_local_default_is_sqlite(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    s = Settings(_env_file=None)
    assert s.app_env == "local" and s.db_kind == "sqlite" and sqlite_path(s.database_url) == "./data/agent.db"


def test_openrouter_key_from_env_sets_base_url(monkeypatch):
    for k in ("OPENAI_API_KEY", "LLM_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-abc")
    s = Settings(_env_file=None, database_url="")
    assert s.openai_api_key == "sk-or-v1-abc" and s.openai_base_url == OPENROUTER_BASE_URL


def test_db_kind_parsing():
    assert db_kind("postgres://u:p@h/db") == "postgres"
    assert db_kind("sqlite+aiosqlite:////tmp/x.db") == "sqlite"
    assert db_kind("") == "memory"
    with pytest.raises(ValueError):
        db_kind("mysql://u:p@h/db")
