"""
Dense embedding models, as LlamaIndex BaseEmbedding objects.

    EMBED_PROVIDER=fastembed     local ONNX model, no API key (default; BAAI/bge-small-en-v1.5, 384 dims).
                                 Downloads the model from Hugging Face the first time; the Docker
                                 image bakes it in so Cloud Run cold starts don't download.
    EMBED_PROVIDER=openai        any OpenAI-compatible /embeddings endpoint (OpenAI, Azure via
                                 base URL, LiteLLM proxy, Ollama, vLLM): EMBED_BASE_URL + EMBED_API_KEY
    EMBED_PROVIDER=huggingface   sentence-transformers, e.g. BAAI/bge-base-en-v1.5 like the original bot
                                 (needs `pip install llama-index-embeddings-huggingface`, pulls torch)
    EMBED_PROVIDER=hash          deterministic hashing vectors with NO semantic meaning. Only for
                                 tests and offline demos; keyword (sparse) search still works.

Whatever you pick, the collection remembers its vector size: switching models later
means re-ingesting (POST /api/ingest/collections/{course}/reset, then ingest again).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import httpx
from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import PrivateAttr

from app.rag.sparse import tokenize


class HashEmbedding(BaseEmbedding):
    """Feature-hashing bag of words, L2-normalised. Similar texts share tokens, so
    retrieval isn't random, but this is NOT a semantic model."""

    dim: int = 256

    @classmethod
    def class_name(cls) -> str:
        return "HashEmbedding"

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in tokenize(text) or ["_empty_"]:
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vec(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)


class FastEmbedEmbedding(BaseEmbedding):
    """fastembed's ONNX models (CPU, no torch). Query and passage prefixes are handled by fastembed."""

    _model: Any = PrivateAttr()

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str | None = None, **kwargs: Any):
        super().__init__(model_name=model_name, **kwargs)
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    @classmethod
    def class_name(cls) -> str:
        return "FastEmbedEmbedding"

    def _get_query_embedding(self, query: str) -> list[float]:
        return next(iter(self._model.query_embed([query]))).tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        return next(iter(self._model.passage_embed([text]))).tolist()

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self._model.passage_embed(texts)]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


class OpenAICompatibleEmbedding(BaseEmbedding):
    """POST {base_url}/embeddings — works with OpenAI and anything that copies its API."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    timeout: float = 60.0

    @classmethod
    def class_name(cls) -> str:
        return "OpenAICompatibleEmbedding"

    def _post(self, inputs: list[str]) -> list[list[float]]:
        resp = httpx.post(
            f"{self.base_url.rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            json={"model": self.model_name, "input": inputs},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._post([query])[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._post([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._post(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


def build_embed_model(provider: str, model: str, *, base_url: str = "", api_key: str = "", cache_dir: str | None = None):
    if provider == "hash":
        return HashEmbedding(model_name="hash-256", embed_batch_size=256)
    if provider == "fastembed":
        return FastEmbedEmbedding(model_name=model, cache_dir=cache_dir, embed_batch_size=64)
    if provider == "openai":
        return OpenAICompatibleEmbedding(
            model_name=model, base_url=base_url or "https://api.openai.com/v1", api_key=api_key, embed_batch_size=96
        )
    if provider == "huggingface":
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        except ImportError as exc:  # pragma: no cover - optional heavy dependency
            raise RuntimeError("EMBED_PROVIDER=huggingface needs: pip install llama-index-embeddings-huggingface") from exc
        return HuggingFaceEmbedding(model_name=model)
    raise ValueError(f"Unknown EMBED_PROVIDER {provider!r}")
