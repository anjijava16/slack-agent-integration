-- ============================================================================
-- Same tables as schema.sql, in SQLite's dialect. Used for local development
-- (DATABASE_URL=sqlite:///./data/agent.db). Cloud environments use PostgreSQL.
--
--   UUID        -> TEXT
--   JSONB       -> TEXT holding JSON
--   TIMESTAMPTZ -> TEXT, ISO-8601 in UTC ("2026-10-18T14:05:00+00:00"), which
--                  sorts and compares correctly as a string
-- ============================================================================
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id       TEXT    PRIMARY KEY,
    thread_id    TEXT    NOT NULL,
    source       TEXT    NOT NULL,
    user_id      TEXT,
    channel_id   TEXT,
    question     TEXT    NOT NULL,
    answer       TEXT    NOT NULL,
    tool_calls   TEXT    NOT NULL DEFAULT '[]',
    model        TEXT,
    duration_ms  INTEGER,
    blocked      INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_agent_runs_thread     ON agent_runs (thread_id);
CREATE INDEX IF NOT EXISTS ix_agent_runs_created_at ON agent_runs (created_at);

CREATE TABLE IF NOT EXISTS agent_feedback (
    feedback_id    TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES agent_runs (run_id) ON DELETE CASCADE,
    feedback_type  TEXT NOT NULL CHECK (feedback_type IN ('upvote', 'downvote')),
    user_id        TEXT NOT NULL,
    team_id        TEXT,
    channel_id     TEXT,
    message_ts     TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    UNIQUE (run_id, user_id)
);
CREATE INDEX IF NOT EXISTS ix_agent_feedback_created_at ON agent_feedback (created_at);

CREATE VIEW IF NOT EXISTS agent_run_scores AS
SELECT r.run_id, r.created_at, r.source, r.user_id, r.channel_id, r.question, r.answer,
       r.tool_calls, r.duration_ms,
       COUNT(f.feedback_id) FILTER (WHERE f.feedback_type = 'upvote')   AS upvotes,
       COUNT(f.feedback_id) FILTER (WHERE f.feedback_type = 'downvote') AS downvotes
FROM agent_runs r
LEFT JOIN agent_feedback f ON f.run_id = r.run_id
GROUP BY r.run_id;

CREATE TABLE IF NOT EXISTS slack_event_claims (
    event_key   TEXT PRIMARY KEY,
    claimed_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_jobs (
    job_id       TEXT PRIMARY KEY,
    course       TEXT NOT NULL,
    kind         TEXT NOT NULL,
    params       TEXT NOT NULL DEFAULT '{}',
    status       TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    result       TEXT,
    error        TEXT,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    finished_at  TEXT
);
CREATE INDEX IF NOT EXISTS ix_ingest_jobs_created_at ON ingest_jobs (created_at);
