"""
LangGraph conversation memory, stored in the same database as everything else.

    sqlite:///...     -> AsyncSqliteSaver   (local; same file as agent_runs)
    postgresql://...  -> AsyncPostgresSaver (Cloud SQL; psycopg pool)
    "" (empty)        -> InMemorySaver      (unit tests)

Why this matters on Cloud Run: instances are created and destroyed all the
time (scale to zero, deploys, autoscaling). With InMemorySaver a Slack thread
"forgets" everything whenever its next message lands on a fresh instance. With
a database checkpointer any instance can pick up any thread.

LangGraph creates its own tables (checkpoints, checkpoint_writes, ...) on
setup(); they don't collide with agent_runs / agent_feedback.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.db.feedback_store import db_kind, sqlite_path

log = logging.getLogger(__name__)

_SETUP_LOCK_ID = 724002  # Postgres advisory lock: one replica runs LangGraph's migrations at a time


@asynccontextmanager
async def open_checkpointer(database_url: str, pool_max: int = 5) -> AsyncIterator[BaseCheckpointSaver]:
    kind = db_kind(database_url)

    if kind == "memory":
        yield InMemorySaver()
        return

    if kind == "sqlite":
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        path = sqlite_path(database_url)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(path)
        try:
            await conn.execute("PRAGMA journal_mode = WAL")
            saver = AsyncSqliteSaver(conn)
            await saver.setup()
            log.info("checkpointer: SQLite at %s", path)
            yield saver
        finally:
            await conn.close()
        return

    # postgres
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        conninfo=database_url,  # libpq URL; ?host=/cloudsql/<instance> works for Cloud Run's socket
        min_size=1,
        max_size=max(2, pool_max),
        open=False,
        # What AsyncPostgresSaver expects: autocommit, dict rows, no server-side
        # prepared statements (safe behind PgBouncer / Cloud SQL proxies).
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open(wait=True, timeout=30)
    try:
        saver = AsyncPostgresSaver(pool)
        async with pool.connection() as lock_conn:
            await lock_conn.execute("SELECT pg_advisory_lock(%s)", (_SETUP_LOCK_ID,))
            try:
                await saver.setup()
            finally:
                await lock_conn.execute("SELECT pg_advisory_unlock(%s)", (_SETUP_LOCK_ID,))
        log.info("checkpointer: PostgreSQL")
        yield saver
    finally:
        await pool.close()
