"""
Central configuration.

Everything comes from environment variables (or a .env file). pydantic-settings
validates the values once at startup, so a missing token fails loudly at boot
instead of deep inside a Slack handler at 2am.

Names match the ones used in the original Socket Mode script
(SLACK_BOT_TOKEN, SLACK_SOCKET_TOKEN, SLACK_CHANNEL_ID) so an existing .env
keeps working. SLACK_APP_TOKEN is accepted as an alias for SLACK_SOCKET_TOKEN.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.db.feedback_store import db_kind

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True)

    # ---------------- App ----------------
    app_name: str = "slack-langgraph-agent"
    # local -> your laptop (SQLite). test / prod -> Cloud Run (must use PostgreSQL).
    app_env: Literal["local", "test", "prod"] = "local"
    log_level: str = "INFO"

    # ---------------- Slack ----------------
    # "socket" -> outbound WebSocket to Slack (no public URL needed, great for laptops)
    # "http"   -> Slack POSTs events to https://<your-host>/slack/events
    slack_mode: Literal["socket", "http"] = "socket"

    slack_bot_token: str = Field(default="", description="xoxb-... Bot User OAuth Token")
    slack_signing_secret: str = Field(default="", description="Used to verify HTTP requests from Slack")
    slack_app_token: str = Field(
        default="",
        validation_alias=AliasChoices("SLACK_APP_TOKEN", "SLACK_SOCKET_TOKEN"),
        description="xapp-... App-Level token with connections:write (Socket Mode only)",
    )
    slack_channel_id: str = Field(default="", description="Default channel for /api/slack/notify")

    # Slash command names. Slack command names are global per workspace, so keep
    # them configurable instead of hard-coding them in the handler decorators.
    slack_cmd_agent: str = "/agent"
    slack_cmd_add: str = "/agent-add"

    # Shared secret that protects the REST endpoints (/api/*). Empty = no auth (local dev).
    api_key: str = ""

    # ---------------- LLM ----------------
    # Any OpenAI-compatible endpoint works: OpenAI, Azure OpenAI (via base url),
    # OpenRouter, LiteLLM proxy, vLLM, Ollama's /v1 endpoint, ...
    # The key can come from OPENAI_API_KEY, OPENROUTER_API_KEY or LLM_API_KEY.
    # Cloud Run gets it from Secret Manager (openrouter-api-key) as LLM_API_KEY.
    openai_api_key: str = Field(default="", validation_alias=AliasChoices("LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"))
    openai_base_url: str | None = None
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.2
    llm_timeout_s: float = 60.0

    # ---------------- Database (runs + 👍/👎 feedback) ----------------
    # postgresql://user:password@host:5432/dbname   (plain libpq URL, no "+asyncpg")
    #   local:       sqlite:///./data/agent.db               (default)
    #   test / prod: postgresql://user:pw@/db?host=/cloudsql/<project>:<region>:<instance>
    #                (Cloud Run injects it from Secret Manager)
    #   ""           in-memory, nothing survives a restart (unit tests)
    database_url: str = "sqlite:///./data/agent.db"
    db_pool_min: int = 1
    db_pool_max: int = 5

    # ---------------- Agent ----------------
    agent_max_steps: int = 12  # LangGraph recursion_limit (protects against tool loops)
    agent_max_input_chars: int = 4000  # guardrail for giant pastes

    # ---------------- RAG: course FAQ bot (LlamaIndex + Qdrant) ----------------
    rag_enabled: bool = True
    rag_courses_file: str = "config/courses.yaml"
    rag_default_course: str = ""  # answer with RAG in channels not listed in courses.yaml
    rag_channel_map: str = ""  # "C0123=llm,C0456=de" — extra channel routes without editing the YAML
    slack_cmd_faq: str = "/faq"
    slack_workspace_url: str = ""  # https://<team>.slack.com/ for source links; auto-detected if empty

    # Qdrant: local (docker) | cloud | embedded (no server, on disk) | memory (tests)
    qdrant_mode: Literal["local", "cloud", "embedded", "memory"] = "local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_path: str = "./data/qdrant"
    qdrant_collection_prefix: str = "faq_"
    qdrant_timeout_s: int = 30

    # Embeddings (dense) and keyword (sparse) vectors
    embed_provider: Literal["fastembed", "openai", "huggingface", "hash"] = "fastembed"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_base_url: str = ""  # EMBED_PROVIDER=openai: any OpenAI-compatible /embeddings endpoint
    embed_api_key: str = ""
    embed_cache_dir: str | None = None  # fastembed model cache (Docker image sets it)
    sparse_provider: Literal["bm25", "fastembed"] = "bm25"  # bm25 = built-in, no download
    sparse_model: str = "Qdrant/bm25"

    # Retrieval pipeline: top_k retrieved -> recency_top_k after time weighting -> final_top_n after re-ranking
    rag_search_mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    rag_top_k: int = 20
    rag_recency_top_k: int = 10
    rag_final_top_n: int = 4
    rag_hybrid_alpha: float = 0.5  # weight of dense vs keyword in RRF (0 = keyword only, 1 = dense only)
    rag_recency_half_life_days: float = 30.0
    rag_recency_weight: float = 0.3
    rerank_provider: Literal["auto", "none", "cohere", "fastembed"] = "auto"  # auto = cohere if key else none
    cohere_api_key: str = ""
    rerank_model: str = ""  # default: rerank-v3.5 (cohere) / Xenova/ms-marco-MiniLM-L-6-v2 (fastembed)
    rag_llm_temperature: float = 0.2

    # Ingestion
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 50
    github_token: str = Field(default="", validation_alias=AliasChoices("GITHUB_TOKEN", "GH_TOKEN"))
    faq_repo: str = "DataTalksClub/faq@main"  # owner/repo@branch holding _questions/<course>/...
    faq_local_path: str = ""  # use a local checkout of the FAQ repo instead of downloading it

    @model_validator(mode="after")
    def _check(self) -> Settings:
        # An OpenRouter key (sk-or-...) with no base URL would be sent to OpenAI and
        # fail with 401. Point it at OpenRouter automatically.
        if not self.openai_base_url and self.openai_api_key.startswith("sk-or-"):
            self.openai_base_url = OPENROUTER_BASE_URL
        # Cloud Run's disk is in-memory and per instance: a SQLite file there is
        # silently lost on every scale-down. Refuse to start rather than lose data.
        if self.app_env in ("test", "prod") and db_kind(self.database_url) != "postgres":
            raise ValueError(f"APP_ENV={self.app_env} requires a PostgreSQL DATABASE_URL (got {db_kind(self.database_url)})")
        # Same reasoning for vectors: embedded/memory Qdrant lives inside one container.
        if self.rag_enabled and self.app_env in ("test", "prod") and self.qdrant_mode in ("embedded", "memory"):
            raise ValueError(f"APP_ENV={self.app_env} with RAG needs QDRANT_MODE=cloud (or local with a server URL)")
        if self.rag_enabled and self.qdrant_mode == "cloud" and not (self.qdrant_api_key and self.qdrant_url):
            raise ValueError("QDRANT_MODE=cloud needs QDRANT_URL and QDRANT_API_KEY")
        if not self.rag_top_k >= self.rag_recency_top_k >= self.rag_final_top_n >= 1:
            raise ValueError("Need RAG_TOP_K >= RAG_RECENCY_TOP_K >= RAG_FINAL_TOP_N >= 1")
        return self

    @property
    def db_kind(self) -> str:
        return db_kind(self.database_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()
