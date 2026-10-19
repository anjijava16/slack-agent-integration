"""Shared builders for the RAG tests: a tiny FAQ repo, a course repo, a course catalogue
and a RagService wired to in-memory (or a real) Qdrant with the offline hash embedding."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from app.config import Settings
from app.db.feedback_store import InMemoryFeedbackStore
from app.rag.courses import CourseCatalog, RepoSpec
from app.rag.qdrant_store import QdrantCollections, QdrantConfig
from app.rag.service import RagService

QDRANT_URL = os.environ.get("TEST_QDRANT_URL")
BACKENDS = ["memory", "server"]

FAQ = {
    "general/001_aaa111_join.md": (
        "aaa111",
        "Can I still join the course after the start date?",
        "Yes. You can join late; homework deadlines still apply but the final project has two attempts.",
    ),
    "module-1/002_bbb222_es.md": (
        "bbb222",
        "Elasticsearch: ConnectionError: Connection refused on localhost:9200",
        "Start the container first: `docker run -p 9200:9200 elasticsearch:8.4.3`. "
        "If it exits immediately, give Docker at least 4 GB of memory.",
    ),
    "module-1/003_ccc333_gemini.md": (
        "ccc333",
        "How do I use Gemini through the OpenAI compatible endpoint?",
        "Point the OpenAI client at base_url=https://generativelanguage.googleapis.com/v1beta/openai/ "
        "and pass your GEMINI_API_KEY.",
    ),
    "module-2/004_ddd444_qdrant.md": (
        "ddd444",
        "Qdrant: which port does the REST API use?",
        "6333 for REST and 6334 for gRPC. In Docker, publish both: -p 6333:6333 -p 6334:6334.",
    ),
}


def make_faq_repo(root: Path, faq_dir: str = "llm-zoomcamp", entries: dict | None = None) -> Path:
    for rel, (fid, q, a) in (entries or FAQ).items():
        p = root / "_questions" / faq_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"---\nid: {fid}\nquestion: {json.dumps(q)}\nsort_order: 1\n---\n\n{a}\n", encoding="utf-8")
    (root / "_questions" / faq_dir / "_metadata.yaml").write_text("course: llm\n")
    return root


def make_course_repo(root: Path) -> Path:
    files = {
        "README.md": "# LLM Zoomcamp\n\n## Modules\n\nModule 1: RAG. Module 2: vector search with minsearch and Qdrant.",
        "01-intro/README.md": "# Intro\n\nInstall requirements with `pip install -r requirements.txt`. Use Python 3.12.",
        "02-vector-search/search.py": "from minsearch import Index\nindex = Index(text_fields=['question'])\n",
        "02-vector-search/notebook.ipynb": json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": ["# Qdrant hybrid search\n", "![x](data:image/png;base64,AAAA)"]},
                    {
                        "cell_type": "code",
                        "source": "client.query_points(collection_name='zoomcamp-faq')",
                        "outputs": [{"text": "BIG OUTPUT"}],
                    },
                ]
            }
        ),
        "cohorts/2024/old.md": "# Old cohort\n\nThis should be excluded.",
        "images/diagram.md": "should be excluded",
        ".github/workflows/ci.md": "hidden, excluded",
        "data/big.csv": "a,b\n1,2\n",
    }
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return root


def make_catalog(tmp: Path, repo_path: Path | None = None) -> CourseCatalog:
    yaml_text = f"""
courses:
  llm:
    name: LLM Zoomcamp
    short_name: LLM
    cohort_year: 2025
    course_start_date: 2 June 2025
    channels: [CLLM]
    slack_ingest: [CLLM]
    slack_days: 30
    faq_dir: llm-zoomcamp
    repos:
      - owner: DataTalksClub
        repo: llm-zoomcamp
        branch: main
        exclude_dirs: [cohorts/2024, images]
        {"local_path: " + str(repo_path) if repo_path else ""}
    youtube_videos: [vid123]
  de:
    name: Data Engineering Zoomcamp
    channels: [CDE]
    faq_dir: data-engineering-zoomcamp
"""
    path = tmp / "courses.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return CourseCatalog.load(path)


def make_settings(backend: str = "memory", **kw) -> Settings:
    base = dict(
        _env_file=None,
        database_url="",
        slack_bot_token="",
        openai_api_key="",
        qdrant_mode="memory" if backend == "memory" else "local",
        qdrant_url=QDRANT_URL or "http://localhost:6333",
        qdrant_collection_prefix=f"t{uuid.uuid4().hex[:8]}_",
        embed_provider="hash",
        rerank_provider="none",
        github_token="",
    )
    base.update(kw)
    return Settings(**base)


def make_rag(tmp: Path, llm, backend: str = "memory", **kw) -> RagService:
    settings = make_settings(backend, faq_local_path=str(make_faq_repo(tmp / "faq")), **kw)
    catalog = make_catalog(tmp, make_course_repo(tmp / "repo"))
    return RagService(settings, llm, InMemoryFeedbackStore(), catalog=catalog)


def collections_for(backend: str) -> QdrantCollections:
    s = make_settings(backend)
    return QdrantCollections(QdrantConfig(mode=s.qdrant_mode, url=s.qdrant_url, prefix=s.qdrant_collection_prefix))


__all__ = ["RepoSpec", "make_rag", "make_catalog", "make_settings", "collections_for", "BACKENDS", "QDRANT_URL"]
