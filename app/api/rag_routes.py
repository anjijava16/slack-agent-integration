"""
Query router — RAG over the course collections.

    GET  /api/rag/status     Qdrant/embedding/reranker config, per-course point counts
    GET  /api/rag/courses    the course catalogue (channels, sources)
    POST /api/rag/query      question -> answer + sources (+ run_id for /api/feedback)
    POST /api/rag/search     retrieval only: see exactly what the LLM would get
"""

from __future__ import annotations

import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.routes import require_api_key
from app.rag.service import RagService, RagUnavailable

router = APIRouter(prefix="/api/rag", tags=["rag"], dependencies=[Depends(require_api_key)])


def get_rag(request: Request) -> RagService:
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise HTTPException(503, "RAG is disabled (RAG_ENABLED=false)")
    return rag


def resolve_course(rag: RagService, course: str | None, channel_id: str | None) -> str:
    if course:
        if course not in rag.catalog:
            raise HTTPException(404, f"unknown course {course!r}; known: {', '.join(rag.catalog.keys())}")
        return course
    found = rag.catalog.for_channel(channel_id)
    if found is None:
        raise HTTPException(422, f"give `course` (one of {', '.join(rag.catalog.keys())}) or a mapped `channel_id`")
    return found.key


class QueryIn(BaseModel):
    question: str = Field(..., min_length=1, examples=["How do I use Gemini with the OpenAI-compatible endpoint?"])
    course: str | None = Field(None, examples=["llm"])
    channel_id: str | None = Field(None, description="Alternative to course: route like Slack does")
    thread_id: str | None = Field(None, description="Reuse to ask follow-ups with memory")
    user_id: str = "api-user"


class SourceOut(BaseModel):
    title: str | None
    url: str | None
    source_type: str | None
    score: float | None = None
    rerank_score: float | None = None


class QueryOut(BaseModel):
    answer: str
    course: str
    run_id: str
    thread_id: str
    search_query: str
    sources: list[SourceOut]
    search: dict
    duration_ms: int
    blocked: bool


class SearchIn(BaseModel):
    query: str = Field(..., min_length=1)
    course: str
    mode: Literal["hybrid", "dense", "sparse"] | None = None
    top_k: int | None = Field(None, ge=1, le=100)
    final_top_n: int | None = Field(None, ge=1, le=50)
    source_types: list[Literal["faq", "github", "slack", "youtube", "web", "file", "text"]] | None = None
    rerank: bool = True


@router.get("/status")
async def status(rag: RagService = Depends(get_rag)):
    return await rag.status()


@router.get("/courses")
async def courses(rag: RagService = Depends(get_rag)):
    return [
        {
            "key": c.key,
            "name": c.name,
            "cohort_year": c.cohort_year,
            "course_start_date": c.course_start_date,
            "channels": c.channels,
            "faq_dir": c.faq_dir,
            "repos": [f"{r.full_name}@{r.branch}" for r in c.repos],
            "slack_ingest": c.slack_ingest,
            "youtube_videos": c.youtube_videos,
        }
        for c in rag.catalog
    ]


@router.post("/query", response_model=QueryOut)
async def query(body: QueryIn, rag: RagService = Depends(get_rag)):
    course = resolve_course(rag, body.course, body.channel_id)
    try:
        res = await rag.answer(
            body.question,
            course,
            thread_id=body.thread_id or f"api:{uuid.uuid4()}",
            user_id=body.user_id,
            channel_id=body.channel_id or "api",
            source="api_rag",
        )
    except RagUnavailable as exc:
        raise HTTPException(503, f"RAG pipeline unavailable: {exc}")
    return QueryOut(
        answer=res.answer,
        course=res.course,
        run_id=res.run_id,
        thread_id=res.thread_id.split(":", 2)[2],
        search_query=res.search_query,
        sources=[SourceOut(**s) for s in res.sources],
        search=res.search_meta,
        duration_ms=res.duration_ms,
        blocked=res.blocked,
    )


@router.post("/search")
async def search(body: SearchIn, rag: RagService = Depends(get_rag)):
    resolve_course(rag, body.course, None)
    try:
        res = await rag.search(
            body.course,
            body.query,
            mode=body.mode,
            top_k=body.top_k,
            final_top_n=body.final_top_n,
            source_types=body.source_types,
            rerank=body.rerank,
        )
    except RagUnavailable as exc:
        raise HTTPException(503, f"RAG pipeline unavailable: {exc}")
    return {
        "query": res.query,
        "mode": res.mode,
        "reranker": res.reranker,
        "funnel": res.funnel,
        "timings_ms": res.timings_ms,
        "hits": [h.as_dict() for h in res.hits],
    }
