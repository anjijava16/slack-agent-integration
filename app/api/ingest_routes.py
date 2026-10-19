"""
Ingestion router — load sources into a course's Qdrant collection.

Long ingestions run as background jobs: the call returns 202 + a job record, poll
GET /api/ingest/jobs/{job_id}. Pass "wait": true to block and get the stats back
(handy in scripts and tests).

    POST /api/ingest/course/{course}   everything configured in courses.yaml (faq, github, slack, youtube)
    POST /api/ingest/faq               DataTalksClub/faq entries for the course
    POST /api/ingest/github            course repos (or the repos you pass)
    POST /api/ingest/slack             answered threads from the last N days
    POST /api/ingest/youtube           video transcripts
    POST /api/ingest/urls              web pages
    POST /api/ingest/text              a snippet of text (synchronous)
    POST /api/ingest/files             multipart upload: pdf, md, txt, ipynb, html, code
    GET  /api/ingest/jobs[/{id}]       job status
    POST /api/ingest/collections/{course}/reset   drop the collection (e.g. after changing EMBED_MODEL)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.rag_routes import get_rag
from app.api.routes import require_api_key
from app.rag.courses import DEFAULT_REPO_EXCLUDE_DIRS, DEFAULT_REPO_INCLUDE, RepoSpec
from app.rag.service import SOURCES, RagService

router = APIRouter(prefix="/api/ingest", tags=["ingest"], dependencies=[Depends(require_api_key)])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class _Base(BaseModel):
    course: str
    wait: bool = Field(False, description="Run inline and return the stats instead of a job")


class CourseIn(BaseModel):
    sources: list[str] | None = Field(None, description=f"subset of {list(SOURCES)}; default all")
    wait: bool = False


class RepoIn(BaseModel):
    owner: str
    repo: str
    branch: str = "main"
    include_ext: list[str] = Field(default_factory=lambda: list(DEFAULT_REPO_INCLUDE))
    exclude_dirs: list[str] = Field(default_factory=lambda: list(DEFAULT_REPO_EXCLUDE_DIRS))
    local_path: str | None = Field(None, description="Local checkout; only allowed when APP_ENV=local")


class GithubIn(_Base):
    repos: list[RepoIn] | None = None


class SlackIn(_Base):
    channels: list[str] | None = None
    days: int | None = Field(None, ge=1, le=3650)


class YoutubeIn(_Base):
    video_ids: list[str] | None = None


class UrlsIn(_Base):
    urls: list[str] = Field(..., min_length=1)


class TextIn(BaseModel):
    course: str
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    url: str = ""
    source_key: str | None = Field(None, description="Same key again replaces the earlier text")
    timestamp: float | None = Field(None, description="unix seconds; enables recency weighting")


def _check_course(rag: RagService, course: str) -> None:
    if course not in rag.catalog:
        raise HTTPException(404, f"unknown course {course!r}; known: {', '.join(rag.catalog.keys())}")


async def _run(rag: RagService, course: str, kind: str, params: dict, work, wait: bool):
    _check_course(rag, course)
    if wait:
        try:
            result = await work()
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return result.as_dict() if hasattr(result, "as_dict") else result
    job = await rag.start_job(course, kind, params, work)
    return JSONResponse(job, status_code=202)


@router.post("/course/{course}")
async def ingest_course(course: str, body: CourseIn | None = None, rag: RagService = Depends(get_rag)):
    body = body or CourseIn()
    bad = set(body.sources or []) - set(SOURCES)
    if bad:
        raise HTTPException(422, f"unknown sources {sorted(bad)}; use {list(SOURCES)}")
    return await _run(
        rag, course, "course", {"sources": body.sources}, lambda: rag.ingest_course(course, body.sources), body.wait
    )


@router.post("/faq")
async def ingest_faq(body: _Base, rag: RagService = Depends(get_rag)):
    return await _run(rag, body.course, "faq", {}, lambda: rag.ingest_faq(body.course), body.wait)


@router.post("/github")
async def ingest_github(body: GithubIn, request: Request, rag: RagService = Depends(get_rag)):
    specs = None
    if body.repos:
        if any(r.local_path for r in body.repos) and request.app.state.settings.app_env != "local":
            raise HTTPException(403, "local_path is only accepted when APP_ENV=local")
        specs = [RepoSpec(**r.model_dump()) for r in body.repos]
    params = {"repos": [f"{s.full_name}@{s.branch}" for s in specs] if specs else None}
    return await _run(rag, body.course, "github", params, lambda: rag.ingest_github(body.course, specs), body.wait)


@router.post("/slack")
async def ingest_slack(body: SlackIn, rag: RagService = Depends(get_rag)):
    params = {"channels": body.channels, "days": body.days}
    return await _run(
        rag, body.course, "slack", params, lambda: rag.ingest_slack(body.course, body.channels, body.days), body.wait
    )


@router.post("/youtube")
async def ingest_youtube(body: YoutubeIn, rag: RagService = Depends(get_rag)):
    return await _run(
        rag,
        body.course,
        "youtube",
        {"video_ids": body.video_ids},
        lambda: rag.ingest_youtube(body.course, body.video_ids),
        body.wait,
    )


@router.post("/urls")
async def ingest_urls(body: UrlsIn, rag: RagService = Depends(get_rag)):
    return await _run(rag, body.course, "url", {"urls": body.urls}, lambda: rag.ingest_urls(body.course, body.urls), body.wait)


@router.post("/text")
async def ingest_text(body: TextIn, rag: RagService = Depends(get_rag)):
    _check_course(rag, body.course)
    stats = await rag.ingest_text(
        body.course, body.text, body.title, url=body.url, source_key=body.source_key, timestamp=body.timestamp
    )
    return stats.as_dict()


@router.post("/files")
async def ingest_files(
    course: str = Form(...),
    files: list[UploadFile] = File(...),
    wait: bool = Form(False),
    rag: RagService = Depends(get_rag),
):
    payload = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{f.filename} is larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
        payload.append((f.filename or "upload.txt", data))
    params = {"files": [n for n, _ in payload]}
    return await _run(rag, course, "files", params, lambda: rag.ingest_files(course, payload), wait)


@router.get("/jobs")
async def list_jobs(course: str | None = None, limit: int = Query(20, ge=1, le=200), rag: RagService = Depends(get_rag)):
    return await rag.jobs.list(limit=limit, course=course)


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, rag: RagService = Depends(get_rag)):
    job = await rag.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    return job


@router.post("/collections/{course}/reset")
async def reset_collection(course: str, rag: RagService = Depends(get_rag)):
    _check_course(rag, course)
    return {"course": course, "collection": rag.collections.name(course), "deleted": await rag.reset(course)}
