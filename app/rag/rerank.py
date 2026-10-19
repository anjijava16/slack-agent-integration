"""
Re-rankers: the last step of retrieval. A cross-encoder reads the question and each
candidate together, which is far more accurate than comparing two independent
embeddings, but too slow to run over the whole collection — hence retrieve 20,
keep 10 after recency weighting, re-rank those down to 4.

    RERANK_PROVIDER=cohere     Cohere Rerank API (rerank-v3.5), like the original bot (COHERE_API_KEY)
    RERANK_PROVIDER=fastembed  local cross-encoder, no API key (Xenova/ms-marco-MiniLM-L-6-v2)
    RERANK_PROVIDER=none       keep the hybrid + recency order
    RERANK_PROVIDER=auto       cohere if COHERE_API_KEY is set, otherwise none
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import httpx

log = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str

    def scores(self, query: str, texts: list[str]) -> list[float]: ...


class NoRerank:
    name = "none"

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(len(texts) - i) for i in range(len(texts))]  # keep incoming order


class CohereRerank:
    name = "cohere"

    def __init__(self, api_key: str, model: str = "rerank-v3.5", timeout: float = 20.0) -> None:
        self.api_key, self.model, self.timeout = api_key, model or "rerank-v3.5", timeout

    def scores(self, query: str, texts: list[str]) -> list[float]:
        resp = httpx.post(
            "https://api.cohere.com/v2/rerank",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "query": query, "documents": texts, "top_n": len(texts)},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        out = [0.0] * len(texts)
        for r in resp.json()["results"]:
            out[r["index"]] = float(r["relevance_score"])
        return out


class FastEmbedRerank:
    name = "fastembed"

    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2", cache_dir: str | None = None) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._enc: Any = TextCrossEncoder(model_name=model or "Xenova/ms-marco-MiniLM-L-6-v2", cache_dir=cache_dir)

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._enc.rerank(query, texts)]


def build_reranker(provider: str, *, cohere_api_key: str = "", model: str = "", cache_dir: str | None = None) -> Reranker:
    if provider == "auto":
        provider = "cohere" if cohere_api_key else "none"
    if provider == "cohere":
        if not cohere_api_key:
            raise ValueError("RERANK_PROVIDER=cohere needs COHERE_API_KEY")
        return CohereRerank(cohere_api_key, model)
    if provider == "fastembed":
        return FastEmbedRerank(model, cache_dir)
    return NoRerank()
