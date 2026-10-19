"""
Persistence for agent runs and 👍/👎 feedback.

Three implementations behind one interface, picked from DATABASE_URL:

    sqlite:///./data/agent.db        SqliteFeedbackStore    local development (default)
    postgresql://user:pw@host/db     PostgresFeedbackStore  test / prod on Cloud Run (Cloud SQL)
    "" (empty)                       InMemoryFeedbackStore  unit tests

Every store also owns `slack_event_claims`, which de-duplicates Slack events
across all instances of the service (see claim_event()).

Why asyncpg and not psycopg/SQLAlchemy here: the whole app runs on one asyncio
event loop (FastAPI + Bolt AsyncApp + LangGraph). A blocking DB driver inside a
Bolt listener would freeze every other Slack event and REST request while it
waits on the network. asyncpg is natively async and fast. The sync, psycopg
version of the same idea is in examples/03_feedback_postgres_sync.py.
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

log = logging.getLogger(__name__)

FeedbackType = Literal["upvote", "downvote"]
SCHEMA_SQL = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
SQLITE_SCHEMA_SQL = (Path(__file__).parent / "schema_sqlite.sql").read_text(encoding="utf-8")
CLAIM_TTL = timedelta(days=1)  # how long a Slack event key is remembered


def _now() -> datetime:
    return datetime.now(UTC)


# ----------------------------------------------------------------------------- records
@dataclass
class RunRecord:
    run_id: str
    thread_id: str
    source: str
    user_id: str | None
    channel_id: str | None
    question: str
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    model: str | None = None
    duration_ms: int | None = None
    blocked: bool = False


@dataclass
class FeedbackRecord:
    feedback_id: str
    run_id: str
    feedback_type: FeedbackType
    user_id: str
    team_id: str | None
    channel_id: str | None
    message_ts: str | None
    created_at: str


@dataclass
class FeedbackTally:
    run_id: str
    upvotes: int
    downvotes: int
    changed: Literal["new", "changed", "unchanged"]  # what this click did


class UnknownRunError(Exception):
    """The run_id on the button isn't in agent_runs (e.g. DB was down when it was answered)."""


def feedback_from_body(body: dict, feedback_type: FeedbackType) -> FeedbackRecord:
    """Turn a Slack block_actions payload into a FeedbackRecord.

    Same fields as the original add_feedback() helper:
        body["actions"][0]["value"]  -> run_id (we put it on the button)
        body["user"]["id"]           -> who clicked
        body["channel"]["id"]        -> where
        body["message"]["ts"]        -> which message held the buttons
    """
    actions = body.get("actions") or []
    action = actions[0] if actions else {}
    user = body.get("user") or {}
    channel = body.get("channel") or {}
    message = body.get("message") or {}
    team = body.get("team") or {}

    run_id = action.get("value")
    if not run_id:
        raise ValueError("button has no value (run_id)")
    try:
        run_id = str(uuid.UUID(run_id))
    except ValueError as exc:
        raise ValueError(f"button value is not a run id: {run_id!r}") from exc
    if not user.get("id"):
        raise ValueError("payload has no user id")

    return FeedbackRecord(
        feedback_id=str(uuid.uuid4()),
        run_id=run_id,
        feedback_type=feedback_type,
        user_id=user["id"],
        team_id=team.get("id") or user.get("team_id"),
        channel_id=channel.get("id"),
        message_ts=message.get("ts") or (body.get("container") or {}).get("message_ts"),
        created_at=datetime.now(UTC).isoformat(),
    )


# ----------------------------------------------------------------------------- interface
class FeedbackStore(Protocol):
    kind: str  # "postgres" | "sqlite" | "memory"

    async def init(self) -> None: ...
    async def claim_event(self, key: str) -> bool: ...
    async def close(self) -> None: ...
    async def record_run(self, run: RunRecord) -> None: ...
    async def add_feedback(self, fb: FeedbackRecord) -> FeedbackTally: ...
    async def get_run(self, run_id: str) -> dict | None: ...
    async def summary(self, days: int = 7, worst: int = 5) -> dict: ...
    async def ping(self) -> bool: ...


# ----------------------------------------------------------------------------- Postgres
_INSERT_RUN = """
INSERT INTO agent_runs (run_id, thread_id, source, user_id, channel_id, question, answer,
                        tool_calls, model, duration_ms, blocked)
VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11)
ON CONFLICT (run_id) DO NOTHING
"""

# Upsert keyed on (run_id, user_id): one vote per person per answer, and they can
# change their mind. RETURNING tells us what happened:
#   xmax = 0                          -> brand-new row
#   previous type differs (via CTE)   -> vote switched
_UPSERT_FEEDBACK = """
WITH prev AS (
    SELECT feedback_type FROM agent_feedback WHERE run_id = $2::uuid AND user_id = $4
)
INSERT INTO agent_feedback (feedback_id, run_id, feedback_type, user_id, team_id,
                            channel_id, message_ts, created_at, updated_at)
VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $8)
ON CONFLICT (run_id, user_id) DO UPDATE
   SET feedback_type = EXCLUDED.feedback_type,
       channel_id    = COALESCE(EXCLUDED.channel_id, agent_feedback.channel_id),
       message_ts    = COALESCE(EXCLUDED.message_ts, agent_feedback.message_ts),
       updated_at    = EXCLUDED.updated_at
RETURNING (xmax = 0) AS inserted, (SELECT feedback_type FROM prev) AS previous_type
"""

_TALLY = """
SELECT COUNT(*) FILTER (WHERE feedback_type = 'upvote')   AS upvotes,
       COUNT(*) FILTER (WHERE feedback_type = 'downvote') AS downvotes
FROM agent_feedback WHERE run_id = $1::uuid
"""


class PostgresFeedbackStore:
    kind = "postgres"

    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 5, apply_schema: bool = True) -> None:
        self._dsn = dsn
        self._min, self._max = min_size, max_size
        self._apply_schema = apply_schema
        self._pool = None

    async def init(self) -> None:
        import asyncpg  # imported lazily so the app runs without it when DATABASE_URL is empty

        self._pool = await asyncpg.create_pool(self._dsn, min_size=self._min, max_size=self._max, command_timeout=10)
        if self._apply_schema:
            async with self._pool.acquire() as conn:
                # Serialise schema creation across replicas starting at the same time.
                async with conn.transaction():
                    await conn.execute("SELECT pg_advisory_xact_lock(724001)")
                    await conn.execute(SCHEMA_SQL)
        log.info("feedback store: PostgreSQL ready")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

    async def ping(self) -> bool:
        try:
            return await self._pool.fetchval("SELECT 1") == 1
        except Exception:  # noqa: BLE001
            return False

    async def claim_event(self, key: str) -> bool:
        """True for the first caller across ALL instances; False for duplicates."""
        claimed = await self._pool.fetchval(
            "INSERT INTO slack_event_claims (event_key) VALUES ($1) ON CONFLICT DO NOTHING RETURNING true", key
        )
        if random.random() < 0.01:  # cheap housekeeping, ~1 in 100 events
            await self._pool.execute("DELETE FROM slack_event_claims WHERE claimed_at < $1", _now() - CLAIM_TTL)
        return bool(claimed)

    async def record_run(self, run: RunRecord) -> None:
        await self._pool.execute(
            _INSERT_RUN,
            run.run_id,
            run.thread_id,
            run.source,
            run.user_id,
            run.channel_id,
            run.question,
            run.answer,
            json.dumps(run.tool_calls),
            run.model,
            run.duration_ms,
            run.blocked,
        )

    async def add_feedback(self, fb: FeedbackRecord) -> FeedbackTally:
        import asyncpg

        created = datetime.fromisoformat(fb.created_at)
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                try:
                    row = await conn.fetchrow(
                        _UPSERT_FEEDBACK,
                        fb.feedback_id,
                        fb.run_id,
                        fb.feedback_type,
                        fb.user_id,
                        fb.team_id,
                        fb.channel_id,
                        fb.message_ts,
                        created,
                    )
                except asyncpg.ForeignKeyViolationError as exc:
                    raise UnknownRunError(fb.run_id) from exc
                tally = await conn.fetchrow(_TALLY, fb.run_id)
        if row["inserted"]:
            changed = "new"
        elif row["previous_type"] != fb.feedback_type:
            changed = "changed"
        else:
            changed = "unchanged"
        return FeedbackTally(fb.run_id, tally["upvotes"], tally["downvotes"], changed)

    async def get_run(self, run_id: str) -> dict | None:
        try:
            uuid.UUID(run_id)
        except ValueError:
            return None
        run = await self._pool.fetchrow("SELECT * FROM agent_runs WHERE run_id = $1::uuid", run_id)
        if run is None:
            return None
        votes = await self._pool.fetch(
            "SELECT feedback_type, user_id, channel_id, message_ts, created_at, updated_at "
            "FROM agent_feedback WHERE run_id = $1::uuid ORDER BY created_at",
            run_id,
        )
        out = _jsonable(dict(run))
        out["tool_calls"] = json.loads(out["tool_calls"]) if isinstance(out["tool_calls"], str) else out["tool_calls"]
        out["feedback"] = [_jsonable(dict(v)) for v in votes]
        return out

    async def summary(self, days: int = 7, worst: int = 5) -> dict:
        totals = await self._pool.fetchrow(
            """
            SELECT (SELECT COUNT(*) FROM agent_runs WHERE created_at >= now() - make_interval(days => $1)) AS runs,
                   COUNT(*) FILTER (WHERE feedback_type = 'upvote')   AS upvotes,
                   COUNT(*) FILTER (WHERE feedback_type = 'downvote') AS downvotes
            FROM agent_feedback WHERE updated_at >= now() - make_interval(days => $1)
            """,
            days,
        )
        by_source = await self._pool.fetch(
            """
            SELECT source, SUM(upvotes)::int AS upvotes, SUM(downvotes)::int AS downvotes
            FROM agent_run_scores WHERE created_at >= now() - make_interval(days => $1)
            GROUP BY source ORDER BY source
            """,
            days,
        )
        worst_rows = await self._pool.fetch(
            """
            SELECT run_id, created_at, question, answer, upvotes, downvotes
            FROM agent_run_scores
            WHERE created_at >= now() - make_interval(days => $1) AND downvotes > 0
            ORDER BY downvotes DESC, upvotes ASC, created_at DESC
            LIMIT $2
            """,
            days,
            worst,
        )
        return _summary_dict(days, dict(totals), [dict(r) for r in by_source], [_jsonable(dict(r)) for r in worst_rows])


# ----------------------------------------------------------------------------- SQLite
class SqliteFeedbackStore:
    """Local development store: one file, no server. Same tables and behaviour
    as Postgres (schema_sqlite.sql). One aiosqlite connection (its own thread)
    serialises writes, which is exactly what SQLite wants."""

    kind = "sqlite"

    def __init__(self, path: str) -> None:
        self._path = path
        self._db = None

    async def init(self) -> None:
        import aiosqlite

        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None -> autocommit; we open transactions explicitly.
        self._db = await aiosqlite.connect(self._path, isolation_level=None)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode = WAL")  # readers don't block the writer
        await self._db.execute("PRAGMA foreign_keys = ON")  # off by default in SQLite!
        await self._db.execute("PRAGMA busy_timeout = 5000")
        await self._db.executescript(SQLITE_SCHEMA_SQL)
        log.info("feedback store: SQLite at %s", self._path)

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()

    async def ping(self) -> bool:
        try:
            async with self._db.execute("SELECT 1") as cur:
                return (await cur.fetchone())[0] == 1
        except Exception:  # noqa: BLE001
            return False

    async def claim_event(self, key: str) -> bool:
        cur = await self._db.execute(
            "INSERT INTO slack_event_claims (event_key, claimed_at) VALUES (?, ?) ON CONFLICT DO NOTHING",
            (key, _now().isoformat()),
        )
        if random.random() < 0.01:
            await self._db.execute("DELETE FROM slack_event_claims WHERE claimed_at < ?", ((_now() - CLAIM_TTL).isoformat(),))
        return cur.rowcount == 1

    async def record_run(self, run: RunRecord) -> None:
        await self._db.execute(
            """
            INSERT INTO agent_runs (run_id, thread_id, source, user_id, channel_id, question, answer,
                                    tool_calls, model, duration_ms, blocked, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id) DO NOTHING
            """,
            (
                run.run_id,
                run.thread_id,
                run.source,
                run.user_id,
                run.channel_id,
                run.question,
                run.answer,
                json.dumps(run.tool_calls),
                run.model,
                run.duration_ms,
                int(run.blocked),
                _now().isoformat(),
            ),
        )

    async def add_feedback(self, fb: FeedbackRecord) -> FeedbackTally:
        import sqlite3

        db = self._db
        # BEGIN IMMEDIATE takes the write lock up front, so "read previous vote,
        # then upsert" can't interleave with another writer.
        await db.execute("BEGIN IMMEDIATE")
        try:
            async with db.execute(
                "SELECT feedback_type FROM agent_feedback WHERE run_id = ? AND user_id = ?", (fb.run_id, fb.user_id)
            ) as cur:
                prev = await cur.fetchone()
            try:
                await db.execute(
                    """
                    INSERT INTO agent_feedback (feedback_id, run_id, feedback_type, user_id, team_id,
                                                channel_id, message_ts, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (run_id, user_id) DO UPDATE
                       SET feedback_type = excluded.feedback_type,
                           channel_id    = COALESCE(excluded.channel_id, agent_feedback.channel_id),
                           message_ts    = COALESCE(excluded.message_ts, agent_feedback.message_ts),
                           updated_at    = excluded.updated_at
                    """,
                    (
                        fb.feedback_id,
                        fb.run_id,
                        fb.feedback_type,
                        fb.user_id,
                        fb.team_id,
                        fb.channel_id,
                        fb.message_ts,
                        fb.created_at,
                        fb.created_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:  # FOREIGN KEY constraint failed
                raise UnknownRunError(fb.run_id) from exc
            async with db.execute(
                "SELECT COUNT(*) FILTER (WHERE feedback_type = 'upvote'), "
                "COUNT(*) FILTER (WHERE feedback_type = 'downvote') FROM agent_feedback WHERE run_id = ?",
                (fb.run_id,),
            ) as cur:
                ups, downs = await cur.fetchone()
            await db.execute("COMMIT")
        except BaseException:
            await db.execute("ROLLBACK")
            raise
        if prev is None:
            changed = "new"
        elif prev[0] != fb.feedback_type:
            changed = "changed"
        else:
            changed = "unchanged"
        return FeedbackTally(fb.run_id, ups, downs, changed)

    async def get_run(self, run_id: str) -> dict | None:
        async with self._db.execute("SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        run = dict(row)
        run["tool_calls"] = json.loads(run["tool_calls"])
        run["blocked"] = bool(run["blocked"])
        async with self._db.execute(
            "SELECT feedback_type, user_id, channel_id, message_ts, created_at, updated_at "
            "FROM agent_feedback WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        ) as cur:
            run["feedback"] = [dict(r) for r in await cur.fetchall()]
        return run

    async def summary(self, days: int = 7, worst: int = 5) -> dict:
        since = (_now() - timedelta(days=days)).isoformat()

        async def rows(sql: str, args: tuple) -> list[dict]:
            async with self._db.execute(sql, args) as cur:
                return [dict(r) for r in await cur.fetchall()]

        totals = (
            await rows(
                """
                SELECT (SELECT COUNT(*) FROM agent_runs WHERE created_at >= ?) AS runs,
                       COUNT(*) FILTER (WHERE feedback_type = 'upvote')   AS upvotes,
                       COUNT(*) FILTER (WHERE feedback_type = 'downvote') AS downvotes
                FROM agent_feedback WHERE updated_at >= ?
                """,
                (since, since),
            )
        )[0]
        by_source = await rows(
            "SELECT source, SUM(upvotes) AS upvotes, SUM(downvotes) AS downvotes FROM agent_run_scores "
            "WHERE created_at >= ? GROUP BY source ORDER BY source",
            (since,),
        )
        worst_rows = await rows(
            "SELECT run_id, created_at, question, answer, upvotes, downvotes FROM agent_run_scores "
            "WHERE created_at >= ? AND downvotes > 0 ORDER BY downvotes DESC, upvotes ASC, created_at DESC LIMIT ?",
            (since, worst),
        )
        return _summary_dict(days, totals, by_source, worst_rows)


# ----------------------------------------------------------------------------- in-memory
class InMemoryFeedbackStore:
    """Same behaviour as the Postgres store, kept in dicts. Used when
    DATABASE_URL is empty (unit tests)."""

    kind = "memory"

    def __init__(self) -> None:
        self.runs: dict[str, dict] = {}
        self.feedback: dict[tuple[str, str], dict] = {}
        self.claims: set[str] = set()

    async def claim_event(self, key: str) -> bool:
        if key in self.claims:
            return False
        self.claims.add(key)
        return True

    async def init(self) -> None:
        log.warning("feedback store: in-memory (set DATABASE_URL to persist to PostgreSQL)")

    async def close(self) -> None:
        return None

    async def ping(self) -> bool:
        return True

    async def record_run(self, run: RunRecord) -> None:
        self.runs.setdefault(run.run_id, {**asdict(run), "created_at": datetime.now(UTC)})

    async def add_feedback(self, fb: FeedbackRecord) -> FeedbackTally:
        if fb.run_id not in self.runs:
            raise UnknownRunError(fb.run_id)
        key = (fb.run_id, fb.user_id)
        prev = self.feedback.get(key)
        now = datetime.fromisoformat(fb.created_at)
        if prev is None:
            self.feedback[key] = {**asdict(fb), "created_at": now, "updated_at": now}
            changed = "new"
        else:
            changed = "unchanged" if prev["feedback_type"] == fb.feedback_type else "changed"
            prev.update(feedback_type=fb.feedback_type, updated_at=now)
        ups, downs = self._tally(fb.run_id)
        return FeedbackTally(fb.run_id, ups, downs, changed)

    def _tally(self, run_id: str) -> tuple[int, int]:
        votes = [v["feedback_type"] for (rid, _), v in self.feedback.items() if rid == run_id]
        return votes.count("upvote"), votes.count("downvote")

    async def get_run(self, run_id: str) -> dict | None:
        run = self.runs.get(run_id)
        if run is None:
            return None
        votes = [
            {k: v[k] for k in ("feedback_type", "user_id", "channel_id", "message_ts", "created_at", "updated_at")}
            for (rid, _), v in self.feedback.items()
            if rid == run_id
        ]
        return _jsonable({**run, "feedback": [_jsonable(v) for v in votes]})

    async def summary(self, days: int = 7, worst: int = 5) -> dict:
        by_source: dict[str, list[int]] = {}
        scored = []
        for rid, run in self.runs.items():
            ups, downs = self._tally(rid)
            s = by_source.setdefault(run["source"], [0, 0])
            s[0] += ups
            s[1] += downs
            if downs:
                scored.append((downs, -ups, run, ups))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        totals = {
            "runs": len(self.runs),
            "upvotes": sum(s[0] for s in by_source.values()),
            "downvotes": sum(s[1] for s in by_source.values()),
        }
        return _summary_dict(
            days,
            totals,
            [{"source": k, "upvotes": v[0], "downvotes": v[1]} for k, v in sorted(by_source.items())],
            [
                _jsonable(
                    {
                        "run_id": r["run_id"],
                        "created_at": r["created_at"],
                        "question": r["question"],
                        "answer": r["answer"],
                        "upvotes": ups,
                        "downvotes": downs,
                    }
                )
                for downs, _, r, ups in scored[:worst]
            ],
        )


# ----------------------------------------------------------------------------- helpers
def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, uuid.UUID):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def _summary_dict(days: int, totals: dict, by_source: list[dict], worst: list[dict]) -> dict:
    votes = (totals.get("upvotes") or 0) + (totals.get("downvotes") or 0)
    return {
        "window_days": days,
        "runs": totals.get("runs") or 0,
        "upvotes": totals.get("upvotes") or 0,
        "downvotes": totals.get("downvotes") or 0,
        "approval_rate": round((totals.get("upvotes") or 0) / votes, 3) if votes else None,
        "by_source": by_source,
        "most_downvoted": worst,
    }


def db_kind(database_url: str) -> str:
    """ "postgres" | "sqlite" | "memory" for a DATABASE_URL."""
    url = (database_url or "").strip().lower()
    if not url or url == "memory":
        return "memory"
    if url.startswith(("postgresql://", "postgres://")):
        return "postgres"
    if url.startswith(("sqlite://", "sqlite+aiosqlite://")):
        return "sqlite"
    raise ValueError(
        f"Unsupported DATABASE_URL {database_url.split('@')[-1]!r}. "
        "Use sqlite:///./data/agent.db or postgresql://user:pass@host:5432/db"
    )


def sqlite_path(database_url: str) -> str:
    """sqlite:///./data/agent.db -> ./data/agent.db ; sqlite:////tmp/x.db -> /tmp/x.db"""
    rest = database_url.split("://", 1)[1]
    if not rest.startswith("/"):
        raise ValueError("SQLite URLs need three slashes: sqlite:///relative.db or sqlite:////abs/path.db")
    path = rest[1:]
    return path or ":memory:"


def build_feedback_store(database_url: str, min_size: int = 1, max_size: int = 5) -> FeedbackStore:
    kind = db_kind(database_url)
    if kind == "postgres":
        return PostgresFeedbackStore(database_url, min_size=min_size, max_size=max_size)
    if kind == "sqlite":
        return SqliteFeedbackStore(sqlite_path(database_url))
    return InMemoryFeedbackStore()
