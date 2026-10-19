"""
LLM factory.

ChatOpenAI talks to anything that speaks the OpenAI Chat Completions API, so
the same code works with OpenAI, OpenRouter, a LiteLLM proxy, vLLM or Ollama by
changing OPENAI_BASE_URL + LLM_MODEL. Tests pass their own fake model to
build_graph() and never touch this file.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import Settings


def build_llm(settings: Settings) -> BaseChatModel:
    if not settings.openai_api_key:
        raise RuntimeError("No LLM key: set OPENROUTER_API_KEY (or OPENAI_API_KEY / LLM_API_KEY).")
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        timeout=settings.llm_timeout_s,
        max_retries=2,
        # OpenRouter uses these for attribution; other providers ignore them.
        default_headers={"HTTP-Referer": "https://github.com/anjijava16", "X-Title": settings.app_name},
    )
