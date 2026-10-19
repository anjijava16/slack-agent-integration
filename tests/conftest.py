"""
Test defaults: RAG runs fully offline — in-memory Qdrant, the hashing embedding,
no re-ranker — so the suite needs no Docker, no model downloads and no API keys.
The real-Qdrant tests opt in with TEST_QDRANT_URL (see tests/test_rag_*.py).
"""

import os

for key, value in {
    "QDRANT_MODE": "memory",
    "EMBED_PROVIDER": "hash",
    "RERANK_PROVIDER": "none",
    "QDRANT_COLLECTION_PREFIX": "test_",
}.items():
    os.environ.setdefault(key, value)
for key in ("GITHUB_TOKEN", "GH_TOKEN", "COHERE_API_KEY", "QDRANT_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
    os.environ.pop(key, None)
