-- ============================================================================
-- Agent runs + user feedback (upvote / downvote) — PostgreSQL 13+
--
-- Applied automatically at app startup (idempotent: IF NOT EXISTS everywhere),
-- and also mounted into the Postgres container's /docker-entrypoint-initdb.d.
-- ============================================================================

-- One row per agent answer. The Slack buttons carry run_id, so every vote can
-- be joined back to the exact question, answer, tools and latency it judged.
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id       UUID        PRIMARY KEY,
    thread_id    TEXT        NOT NULL,             -- LangGraph thread (slack:C123:1700000000.0001)
    source       TEXT        NOT NULL,             -- slack_mention | slack_dm | slash_command | api
    user_id      TEXT,                             -- Slack user who asked
    channel_id   TEXT,
    question     TEXT        NOT NULL,
    answer       TEXT        NOT NULL,
    tool_calls   JSONB       NOT NULL DEFAULT '[]'::jsonb,   -- [{"name":..., "args":{...}}]
    model        TEXT,
    duration_ms  INTEGER,
    blocked      BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_agent_runs_thread     ON agent_runs (thread_id);
CREATE INDEX IF NOT EXISTS ix_agent_runs_created_at ON agent_runs (created_at DESC);

-- One row per (answer, voter). Clicking again, or switching from 👍 to 👎,
-- updates the same row instead of adding a duplicate — see the UNIQUE constraint
-- and the INSERT ... ON CONFLICT in app/db/feedback_store.py.
CREATE TABLE IF NOT EXISTS agent_feedback (
    feedback_id    UUID        PRIMARY KEY,
    run_id         UUID        NOT NULL REFERENCES agent_runs (run_id) ON DELETE CASCADE,
    feedback_type  TEXT        NOT NULL CHECK (feedback_type IN ('upvote', 'downvote')),
    user_id        TEXT        NOT NULL,           -- Slack user who clicked
    team_id        TEXT,
    channel_id     TEXT,
    message_ts     TEXT,                            -- the Slack message that held the buttons
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_agent_feedback_run_user UNIQUE (run_id, user_id)
);

CREATE INDEX IF NOT EXISTS ix_agent_feedback_created_at ON agent_feedback (created_at DESC);
CREATE INDEX IF NOT EXISTS ix_agent_feedback_type       ON agent_feedback (feedback_type);

-- Handy for dashboards and eval sets: one row per answer with its vote counts.
CREATE OR REPLACE VIEW agent_run_scores AS
SELECT
    r.run_id,
    r.created_at,
    r.source,
    r.user_id,
    r.channel_id,
    r.question,
    r.answer,
    r.tool_calls,
    r.duration_ms,
    COUNT(f.*) FILTER (WHERE f.feedback_type = 'upvote')   AS upvotes,
    COUNT(f.*) FILTER (WHERE f.feedback_type = 'downvote') AS downvotes
FROM agent_runs r
LEFT JOIN agent_feedback f ON f.run_id = r.run_id
GROUP BY r.run_id;

-- Slack event de-duplication shared by every instance. Slack can deliver the
-- same event twice (retries, message + app_mention); the first instance to
-- INSERT the key wins, the rest see a conflict and skip. Old rows are pruned
-- by the app.
CREATE TABLE IF NOT EXISTS slack_event_claims (
    event_key   TEXT        PRIMARY KEY,           -- "<channel>:<ts>"
    claimed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_slack_event_claims_at ON slack_event_claims (claimed_at);

-- Background ingestion jobs (POST /api/ingest/... returns a job id; poll GET /api/ingest/jobs/{id}).
CREATE TABLE IF NOT EXISTS ingest_jobs (
    job_id       TEXT        PRIMARY KEY,
    course       TEXT        NOT NULL,
    kind         TEXT        NOT NULL,              -- faq | github | slack | youtube | url | files | text | course
    params       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    status       TEXT        NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    result       JSONB,
    error        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_ingest_jobs_created_at ON ingest_jobs (created_at DESC);
