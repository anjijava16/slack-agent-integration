"""
Ingestion job records, in whichever database the app uses (same connection pools
as the feedback store). Status must live in the database, not in memory: on Cloud
Run the request that polls a job can land on a different instance from the one
running it.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from app.db.feedback_store import FeedbackStore

FIELDS = ("job_id", "course", "kind", "params", "status", "result", "error", "created_at", "started_at", "finished_at")


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(v: Any) -> Any:
    return v.isoformat() if isinstance(v, datetime) else v


def _decode(row: dict) -> dict:
    out = {k: _iso(row.get(k)) for k in FIELDS}
    for k in ("params", "result"):
        if isinstance(out[k], str):
            out[k] = json.loads(out[k])
    return out


class JobRepo:
    def __init__(self, store: FeedbackStore) -> None:
        self.store = store
        self.kind = store.kind
        self._mem: dict[str, dict] = {}

    async def create(self, course: str, kind: str, params: dict) -> dict:
        job = {
            "job_id": str(uuid.uuid4()),
            "course": course,
            "kind": kind,
            "params": params,
            "status": "queued",
            "result": None,
            "error": None,
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
        }
        if self.kind == "postgres":
            await self.store._pool.execute(
                "INSERT INTO ingest_jobs (job_id, course, kind, params, status, created_at) VALUES ($1,$2,$3,$4::jsonb,$5,$6)",
                job["job_id"],
                course,
                kind,
                json.dumps(params),
                "queued",
                job["created_at"],
            )
        elif self.kind == "sqlite":
            await self.store._db.execute(
                "INSERT INTO ingest_jobs (job_id, course, kind, params, status, created_at) VALUES (?,?,?,?,?,?)",
                (job["job_id"], course, kind, json.dumps(params), "queued", job["created_at"].isoformat()),
            )
        else:
            self._mem[job["job_id"]] = dict(job)
        return _decode(job)

    async def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        if self.kind == "memory":
            self._mem[job_id].update(fields)
            return
        cols = list(fields)
        if self.kind == "postgres":
            sets = ", ".join(f"{c} = ${i + 2}{'::jsonb' if c == 'result' else ''}" for i, c in enumerate(cols))
            vals = [json.dumps(fields[c]) if c == "result" else fields[c] for c in cols]
            await self.store._pool.execute(f"UPDATE ingest_jobs SET {sets} WHERE job_id = $1", job_id, *vals)
        else:
            sets = ", ".join(f"{c} = ?" for c in cols)
            vals = [json.dumps(fields[c]) if c == "result" else _iso(fields[c]) for c in cols]
            await self.store._db.execute(f"UPDATE ingest_jobs SET {sets} WHERE job_id = ?", (*vals, job_id))

    async def get(self, job_id: str) -> dict | None:
        if self.kind == "memory":
            row = self._mem.get(job_id)
            return _decode(row) if row else None
        if self.kind == "postgres":
            row = await self.store._pool.fetchrow("SELECT * FROM ingest_jobs WHERE job_id = $1", job_id)
            return _decode(dict(row)) if row else None
        async with self.store._db.execute("SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,)) as cur:
            row = await cur.fetchone()
        return _decode(dict(row)) if row else None

    async def list(self, limit: int = 20, course: str | None = None) -> list[dict]:
        if self.kind == "memory":
            rows = [r for r in self._mem.values() if course in (None, r["course"])]
            rows.sort(key=lambda r: r["created_at"], reverse=True)
            return [_decode(r) for r in rows[:limit]]
        if self.kind == "postgres":
            rows = await self.store._pool.fetch(
                "SELECT * FROM ingest_jobs WHERE ($1::text IS NULL OR course = $1) ORDER BY created_at DESC LIMIT $2",
                course,
                limit,
            )
            return [_decode(dict(r)) for r in rows]
        async with self.store._db.execute(
            "SELECT * FROM ingest_jobs WHERE (? IS NULL OR course = ?) ORDER BY created_at DESC LIMIT ?",
            (course, course, limit),
        ) as cur:
            return [_decode(dict(r)) for r in await cur.fetchall()]
