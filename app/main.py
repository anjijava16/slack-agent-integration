"""
FastAPI entrypoint — wires Slack, the LangGraph agent and the REST API together.

    uvicorn app.main:app --reload --port 8000

One process, one event loop, three ways in:

    Slack (Socket Mode)  ──WebSocket──▶ AsyncSocketModeHandler ─┐
    Slack (HTTP mode)    ──POST /slack/events──▶ AsyncSlackRequestHandler ─┼─▶ Bolt AsyncApp ─▶ AgentService ─▶ LangGraph
    Your systems         ──POST /api/agent/invoke ───────────────────────────────────────────────▶ AgentService
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.language_models.chat_models import BaseChatModel
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from app.agent.llm import build_llm
from app.api.ingest_routes import router as ingest_router
from app.api.rag_routes import router as rag_router
from app.api.routes import router as api_router
from app.config import Settings, get_settings
from app.db.checkpointer import open_checkpointer
from app.db.feedback_store import FeedbackStore, build_feedback_store
from app.rag.service import RagService
from app.rag.tools import make_course_search_tool
from app.services.agent_service import AgentService
from app.slack.bolt_app import create_bolt_app

log = logging.getLogger("app")


def create_app(
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    store: FeedbackStore | None = None,
    rag: RagService | None = None,
) -> FastAPI:
    """App factory. Tests call create_app(settings=..., llm=FakeLLM(), store=...) to run
    the whole stack without OpenAI, Slack or Postgres."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # ---- startup ------------------------------------------------------
        app.state.settings = settings
        log.info("starting env=%s database=%s slack_mode=%s", settings.app_env, settings.db_kind, settings.slack_mode)
        exit_stack = AsyncExitStack()
        # Database first: the agent records every run, the Slack buttons record votes.
        feedback_store = store or build_feedback_store(
            settings.database_url, min_size=settings.db_pool_min, max_size=settings.db_pool_max
        )
        await feedback_store.init()
        exit_stack.push_async_callback(feedback_store.close)
        app.state.store = feedback_store
        # Conversation memory lives in the same database (SQLite file / Cloud SQL).
        checkpointer = await exit_stack.enter_async_context(
            open_checkpointer(settings.database_url, pool_max=settings.db_pool_max)
        )
        llm_ = llm or build_llm(settings)
        # RAG: course FAQ bot over Qdrant. Built before the agent so the agent gets its search tool.
        app.state.rag = None
        if settings.rag_enabled:
            app.state.rag = rag or RagService(settings, llm_, feedback_store, checkpointer)
            # Load the embedding model / connect to Qdrant in the background: a slow model
            # download must not delay Slack or the health check.
            app.state.rag_warmup = asyncio.create_task(app.state.rag.warmup())
            exit_stack.callback(app.state.rag_warmup.cancel)
            exit_stack.push_async_callback(app.state.rag.close)
        app.state.agent = AgentService(
            llm_,
            max_steps=settings.agent_max_steps,
            max_input_chars=settings.agent_max_input_chars,
            store=feedback_store,
            model_name=settings.llm_model,
            checkpointer=checkpointer,
            extra_tools=[make_course_search_tool(app.state.rag)] if app.state.rag else None,
        )
        app.state.bolt = None
        app.state.bolt_http = None
        socket_handler = None

        if settings.slack_bot_token:
            bolt = create_bolt_app(settings, app.state.agent, store=feedback_store, rag=app.state.rag)
            app.state.bolt = bolt
            if settings.slack_mode == "socket":
                if not settings.slack_app_token:
                    raise RuntimeError("SLACK_MODE=socket needs SLACK_APP_TOKEN (xapp-...) / SLACK_SOCKET_TOKEN")
                socket_handler = AsyncSocketModeHandler(bolt, settings.slack_app_token)
                # connect_async() opens the WebSocket and returns; the receive loop
                # runs as a background task on this same event loop.
                await socket_handler.connect_async()
                log.info("Slack connected via Socket Mode")
            else:
                if not settings.slack_signing_secret:
                    raise RuntimeError("SLACK_MODE=http needs SLACK_SIGNING_SECRET")
                app.state.bolt_http = AsyncSlackRequestHandler(bolt)
                log.info("Slack ready in HTTP mode at POST /slack/events")
        else:
            log.warning("SLACK_BOT_TOKEN not set — running REST API only, Slack disabled")

        yield

        # ---- shutdown -----------------------------------------------------
        if socket_handler is not None:
            await socket_handler.close_async()
            log.info("Socket Mode connection closed")
        await exit_stack.aclose()

    app = FastAPI(
        title="Slack × FastAPI × LangGraph agent",
        version="1.0.0",
        description="An agent that lives in Slack, runs on FastAPI and thinks with LangGraph.",
        lifespan=lifespan,
    )
    app.include_router(api_router)
    app.include_router(rag_router)
    app.include_router(ingest_router)

    @app.get("/health", tags=["ops"])
    async def health(request: Request):
        """Liveness: the process is up. Deliberately does NOT touch the database, so a
        slow Cloud SQL doesn't make Cloud Run kill healthy instances.
        (Not /healthz: Cloud Run reserves paths ending in "z" on run.app URLs.)"""
        s: Settings = request.app.state.settings
        return {
            "status": "ok",
            "env": s.app_env,
            "slack": "disabled" if request.app.state.bolt is None else s.slack_mode,
            "model": s.llm_model,
            "database": {"memory": "in-memory"}.get(s.db_kind, s.db_kind),
            "rag": f"qdrant:{s.qdrant_mode}" if request.app.state.rag else "disabled",
        }

    @app.get("/ready", tags=["ops"])
    async def ready(request: Request):
        """Readiness: can we reach the database? Used by the CD smoke test."""
        db_ok = await request.app.state.store.ping()
        body = {"status": "ready", "database": request.app.state.settings.db_kind}
        rag = request.app.state.rag
        qdrant_ok = True
        if rag is not None:
            qdrant_ok = await rag.ping()
            body["qdrant"] = "ok" if qdrant_ok else "unreachable"
        if not db_ok:
            body["status"] = "database unreachable"
        elif not qdrant_ok:
            body["status"] = "qdrant unreachable"
        return JSONResponse(body, status_code=200 if db_ok and qdrant_ok else 503)

    @app.post("/slack/events", tags=["slack"], include_in_schema=False)
    async def slack_events(request: Request):
        """Single URL for Event Subscriptions, Slash Commands and Interactivity.
        Bolt verifies the X-Slack-Signature header, answers url_verification
        challenges, acks within 3s and dispatches to the right listener."""
        handler: AsyncSlackRequestHandler | None = request.app.state.bolt_http
        if handler is None:
            raise HTTPException(404, "Slack HTTP mode is not enabled (SLACK_MODE=socket or no bot token).")
        return await handler.handle(request)

    return app


# `uvicorn app.main:app` — built lazily from env vars.
app = create_app()
