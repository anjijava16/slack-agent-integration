"""
RagService — the one object the Slack handlers, REST routers, agent tool and CLI use.

    answer()   question -> RAG graph -> answer + sources (+ recorded in agent_runs for 👍/👎)
    search()   retrieval only (debug endpoint, agent tool)
    ingest_*() readers -> Indexer, as background jobs recorded in ingest_jobs
    status()   per-course collection stats

Heavy pieces (embedding model download, Qdrant connection) are created lazily, so
the Slack bot and the agent keep working even if Qdrant is down or the model
can't be downloaded; RAG requests then fail with a clear message instead.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.config import Settings
from app.db.feedback_store import FeedbackStore, RunRecord
from app.db.jobs import JobRepo
from app.rag.courses import CourseCatalog, RepoSpec
from app.rag.embeddings import build_embed_model
from app.rag.graph import build_rag_graph
from app.rag.indexer import Indexer, IndexStats
from app.rag.qdrant_store import QdrantCollections, QdrantConfig
from app.rag.readers import make_document
from app.rag.readers.github import checkout, load_faq, load_repo
from app.rag.readers.slack import SlackHistoryReader
from app.rag.readers.web import load_bytes, load_url
from app.rag.readers.youtube import load_youtube
from app.rag.rerank import build_reranker
from app.rag.retriever import RetrievalConfig, Retriever, SearchResult

log = logging.getLogger(__name__)

SOURCES = ("faq", "github", "slack", "youtube")


class RagUnavailable(RuntimeError):
    pass


INIT_RETRY_AFTER_S = 60.0


async def run_in_daemon_thread(fn):
    """Like asyncio.to_thread, but on a daemon thread: a slow model download can never
    hold up shutdown (the default executor's threads are joined at exit)."""
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def deliver(ok: bool, value) -> None:  # runs on the event loop
        if not fut.done():
            fut.set_result(value) if ok else fut.set_exception(value)

    def target():
        # Pass the outcome as arguments: the name bound by `except ... as exc` is deleted
        # when the except block ends, so a lambda closing over it would fail later.
        try:
            outcome = (True, fn())
        except BaseException as exc:  # noqa: BLE001
            outcome = (False, exc)
        try:
            loop.call_soon_threadsafe(deliver, *outcome)
        except RuntimeError:  # the loop closed while we were working (shutdown): nobody is waiting
            pass

    threading.Thread(target=target, name="rag-init", daemon=True).start()
    return await fut


@dataclass
class RagAnswer:
    answer: str
    course: str
    thread_id: str
    run_id: str = ""
    search_query: str = ""
    sources: list[dict] = field(default_factory=list)
    search_meta: dict = field(default_factory=dict)
    duration_ms: int = 0
    blocked: bool = False


def parse_repo(ref: str) -> RepoSpec:
    """'owner/repo@branch' -> RepoSpec"""
    name, _, branch = ref.partition("@")
    owner, _, repo = name.partition("/")
    if not owner or not repo:
        raise ValueError(f"expected owner/repo[@branch], got {ref!r}")
    return RepoSpec(owner=owner, repo=repo, branch=branch or "main")


class RagService:
    def __init__(
        self,
        settings: Settings,
        llm: BaseChatModel,
        store: FeedbackStore,
        checkpointer: BaseCheckpointSaver | None = None,
        *,
        catalog: CourseCatalog | None = None,
        collections: QdrantCollections | None = None,
        embed_model=None,
        reranker=None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.store = store
        self.jobs = JobRepo(store)
        self.catalog = catalog or CourseCatalog.load(
            settings.rag_courses_file, settings.rag_default_course, settings.rag_channel_map
        )
        self.qcfg = QdrantConfig(
            mode=settings.qdrant_mode,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            path=settings.qdrant_path,
            prefix=settings.qdrant_collection_prefix,
            timeout=settings.qdrant_timeout_s,
            hybrid=settings.rag_search_mode != "dense",
            sparse_provider=settings.sparse_provider,
            sparse_model=settings.sparse_model,
        )
        self._collections = collections
        self._embed = embed_model
        self._reranker = reranker
        self._checkpointer = checkpointer or InMemorySaver()
        self._graph = None
        self._retriever: Retriever | None = None
        self._indexer: Indexer | None = None
        self._init_error: str | None = None
        self._init_failed_at = 0.0
        self._init_lock = asyncio.Lock()
        self._ingest_lock = asyncio.Lock()  # one ingestion at a time per instance (embedding is CPU-bound)
        self._locks: dict[str, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ lifecycle
    @property
    def collections(self) -> QdrantCollections:
        if self._collections is None:
            self._collections = QdrantCollections(self.qcfg)
        return self._collections

    def _build(self) -> None:
        s = self.settings
        if self._embed is None:
            self._embed = build_embed_model(
                s.embed_provider, s.embed_model, base_url=s.embed_base_url, api_key=s.embed_api_key, cache_dir=s.embed_cache_dir
            )
        if self._reranker is None:
            self._reranker = build_reranker(
                s.rerank_provider, cohere_api_key=s.cohere_api_key, model=s.rerank_model, cache_dir=s.embed_cache_dir
            )
        rcfg = RetrievalConfig(
            mode=s.rag_search_mode,
            top_k=s.rag_top_k,
            recency_top_k=s.rag_recency_top_k,
            final_top_n=s.rag_final_top_n,
            alpha=s.rag_hybrid_alpha,
            half_life_days=s.rag_recency_half_life_days,
            recency_weight=s.rag_recency_weight,
        )
        self._retriever = Retriever(self.collections, self._embed, rcfg, self._reranker)
        self._indexer = Indexer(self.collections, self._embed, s.rag_chunk_size, s.rag_chunk_overlap)
        self._graph = build_rag_graph(
            self.llm,
            self.catalog,
            lambda course, q: self._retriever.search(course, q),
            checkpointer=self._checkpointer,
            max_input_chars=s.agent_max_input_chars,
        )

    async def ready(self) -> None:
        """Build the pipeline once (model load can take a while: runs in a thread)."""
        if self._graph is not None:
            return
        async with self._init_lock:
            if self._graph is not None:
                return
            # Don't hammer a broken dependency (model download, Qdrant) on every message.
            if self._init_error and time.monotonic() - self._init_failed_at < INIT_RETRY_AFTER_S:
                raise RagUnavailable(self._init_error)
            try:
                await run_in_daemon_thread(self._build)
                self._init_error = None
            except Exception as exc:  # noqa: BLE001
                self._init_error = f"{type(exc).__name__}: {exc}"
                self._init_failed_at = time.monotonic()
                log.error("RAG pipeline failed to initialise: %s", self._init_error)
                raise RagUnavailable(self._init_error) from exc

    async def warmup(self) -> None:
        """Called in the background at startup: downloads/loads the embedding model early."""
        try:
            await self.ready()
            await run_in_daemon_thread(lambda: self._indexer.dim)
            log.info("RAG ready (qdrant=%s, embed=%s, rerank=%s)", self.qcfg.mode, self.settings.embed_model, self._reranker.name)
        except Exception:  # noqa: BLE001 - logged; requests will report it
            pass

    async def close(self) -> None:
        for t in list(self._tasks):
            t.cancel()

    # ------------------------------------------------------------------ query
    async def search(self, course: str, query: str, **kwargs: Any) -> SearchResult:
        self.catalog.get(course)
        await self.ready()
        return await asyncio.to_thread(self._retriever.search, course, query, **kwargs)

    async def answer(
        self,
        question: str,
        course: str,
        *,
        thread_id: str,
        user_id: str = "api-user",
        channel_id: str = "api",
        source: str = "api_rag",
    ) -> RagAnswer:
        self.catalog.get(course)
        await self.ready()
        started = time.perf_counter()
        thread_key = f"rag:{course}:{thread_id}"
        lock = self._locks.setdefault(thread_key, asyncio.Lock())
        async with lock:
            state = await self._graph.ainvoke(
                {"question": question, "course": course, "user_id": user_id, "channel_id": channel_id, "source": source},
                config={"configurable": {"thread_id": thread_key}, "recursion_limit": 12},
            )
        hits = state.get("hits") or []
        sources, seen = [], set()
        for h in hits:
            key = h.get("url") or h.get("title")
            if key and key not in seen:
                seen.add(key)
                sources.append({k: h.get(k) for k in ("title", "url", "source_type", "score", "rerank_score")})
        result = RagAnswer(
            answer=state.get("final_answer") or "",
            course=course,
            thread_id=thread_key,
            run_id=str(uuid.uuid4()),
            search_query=state.get("search_query") or question,
            sources=sources,
            search_meta=state.get("search_meta") or {},
            duration_ms=int((time.perf_counter() - started) * 1000),
            blocked=bool(state.get("blocked_reason")),
        )
        try:
            await self.store.record_run(
                RunRecord(
                    run_id=result.run_id,
                    thread_id=thread_key,
                    source=source,
                    user_id=user_id,
                    channel_id=channel_id,
                    question=question,
                    answer=result.answer,
                    tool_calls=[
                        {
                            "name": "rag_retrieve",
                            "args": {
                                "course": course,
                                "query": result.search_query,
                                "sources": [s["url"] or s["title"] for s in sources],
                                **{k: result.search_meta.get(k) for k in ("mode", "reranker", "funnel")},
                            },
                        }
                    ],
                    model=self.settings.llm_model,
                    duration_ms=result.duration_ms,
                    blocked=result.blocked,
                )
            )
        except Exception:  # noqa: BLE001
            log.exception("could not record RAG run %s", result.run_id)
        return result

    # ------------------------------------------------------------------ status
    async def status(self) -> dict:
        info: dict[str, Any] = {
            "enabled": self.settings.rag_enabled,
            "qdrant": {"mode": self.qcfg.mode, "url": self.qcfg.url if self.qcfg.mode in ("local", "cloud") else None},
            "embed": {"provider": self.settings.embed_provider, "model": self.settings.embed_model},
            "sparse": self.settings.sparse_provider if self.qcfg.hybrid else None,
            "search_mode": self.settings.rag_search_mode,
            "reranker": self._reranker.name if self._reranker else self.settings.rerank_provider,
            "init_error": self._init_error,
        }
        try:
            reachable = await asyncio.to_thread(self.collections.ping)
        except Exception as exc:  # noqa: BLE001
            reachable, info["qdrant"]["error"] = False, str(exc)
        info["qdrant"]["reachable"] = reachable
        courses = []
        for c in self.catalog:
            row = {"key": c.key, "name": c.name, "collection": self.collections.name(c.key), "channels": c.channels}
            if reachable:
                row["points"] = await asyncio.to_thread(self.collections.count, c.key)
                row["by_source"] = await asyncio.to_thread(self.collections.source_counts, c.key)
            courses.append(row)
        info["courses"] = courses
        return info

    async def ping(self) -> bool:
        try:
            return await asyncio.to_thread(self.collections.ping)
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ ingestion: building blocks
    async def _index(self, course: str, docs, prune_group: str | None = None) -> IndexStats:
        await self.ready()
        async with self._ingest_lock:

            def run():
                with self.collections.guard():
                    return self._indexer.index(course, docs, prune_group=prune_group)

            return await asyncio.to_thread(run)

    async def ingest_faq(self, course: str) -> IndexStats:
        c = self.catalog.get(course)
        if not c.faq_dir:
            raise ValueError(f"course {course!r} has no faq_dir in courses.yaml")
        spec = parse_repo(self.settings.faq_repo)
        spec.local_path = self.settings.faq_local_path or None

        def read():
            with checkout(spec, self.settings.github_token) as root:
                return load_faq(Path(root), c.faq_dir, course)

        docs = await asyncio.to_thread(read)
        return await self._index(course, docs, prune_group=f"faq:{c.faq_dir}")

    async def ingest_github(self, course: str, repos: list[RepoSpec] | None = None) -> IndexStats:
        c = self.catalog.get(course)
        total = IndexStats()
        for spec in repos or c.repos:

            def read(spec=spec):
                with checkout(spec, self.settings.github_token) as root:
                    return load_repo(Path(root), spec, course)

            try:
                docs = await asyncio.to_thread(read)
                total.add(await self._index(course, docs, prune_group=f"github:{spec.full_name}@{spec.branch}"))
            except Exception as exc:  # noqa: BLE001 - one bad repo shouldn't stop the others
                log.exception("github ingest failed for %s", spec.full_name)
                total.errors.append(f"{spec.full_name}: {exc}")
        return total

    async def ingest_slack(self, course: str, channels: list[str] | None = None, days: int | None = None) -> IndexStats:
        c = self.catalog.get(course)
        if not self.settings.slack_bot_token:
            raise ValueError("Slack ingestion needs SLACK_BOT_TOKEN")
        channels = channels or c.slack_ingest
        if not channels:
            raise ValueError(f"no Slack channels to ingest for {course!r} (slack_ingest in courses.yaml)")
        reader = SlackHistoryReader.from_token(self.settings.slack_bot_token, self.settings.slack_workspace_url)
        docs = await reader.load(channels, days or c.slack_days, course)
        return await self._index(course, docs)  # no prune: older threads stay searchable

    async def ingest_youtube(self, course: str, video_ids: list[str] | None = None) -> IndexStats:
        c = self.catalog.get(course)
        total = IndexStats()
        for vid in video_ids or c.youtube_videos:
            try:
                docs = await asyncio.to_thread(load_youtube, [vid], course)
                total.add(await self._index(course, docs))
            except Exception as exc:  # noqa: BLE001
                log.exception("youtube ingest failed for %s", vid)
                total.errors.append(f"{vid}: {exc}")
        return total

    async def ingest_urls(self, course: str, urls: list[str]) -> IndexStats:
        self.catalog.get(course)
        total = IndexStats()
        for url in urls:
            try:
                total.add(await self._index(course, [await load_url(url, course)]))
            except Exception as exc:  # noqa: BLE001
                total.errors.append(f"{url}: {exc}")
        return total

    async def ingest_text(
        self, course: str, text: str, title: str, *, url: str = "", source_key: str | None = None, timestamp: float | None = None
    ) -> IndexStats:
        self.catalog.get(course)
        key = source_key or f"text:{uuid.uuid5(uuid.NAMESPACE_URL, title)}"
        doc = make_document(
            text,
            course=course,
            source_type="text",
            source_key=key,
            source_group="text",
            title=title,
            source_url=url,
            timestamp=timestamp,
            fmt="markdown",
        )
        return await self._index(course, [doc])

    async def ingest_files(self, course: str, files: list[tuple[str, bytes]]) -> IndexStats:
        self.catalog.get(course)
        total = IndexStats()
        for name, data in files:
            try:
                total.add(await self._index(course, load_bytes(name, data, course)))
            except Exception as exc:  # noqa: BLE001
                total.errors.append(f"{name}: {exc}")
        return total

    async def ingest_course(self, course: str, sources: list[str] | None = None) -> dict:
        """Everything configured for a course — the equivalent of the original fill_*_index()."""
        out: dict[str, Any] = {}
        for src in sources or SOURCES:
            try:
                fn = {
                    "faq": self.ingest_faq,
                    "github": self.ingest_github,
                    "slack": self.ingest_slack,
                    "youtube": self.ingest_youtube,
                }[src]
                out[src] = (await fn(course)).as_dict()
            except Exception as exc:  # noqa: BLE001
                log.exception("course ingest %s/%s failed", course, src)
                out[src] = {"error": f"{type(exc).__name__}: {exc}"}
        return out

    async def reset(self, course: str) -> bool:
        self.catalog.get(course)
        async with self._ingest_lock:
            return await asyncio.to_thread(self.collections.reset, course)

    # ------------------------------------------------------------------ jobs
    async def start_job(self, course: str, kind: str, params: dict, work) -> dict:
        """Record a job, run `work()` in the background, store its result."""
        self.catalog.get(course)
        job = await self.jobs.create(course, kind, params)

        async def runner():
            await self.jobs.update(job["job_id"], status="running", started_at=datetime.now(UTC))
            try:
                result = await work()
                payload = result.as_dict() if isinstance(result, IndexStats) else result
                await self.jobs.update(job["job_id"], status="succeeded", result=payload, finished_at=datetime.now(UTC))
            except Exception as exc:  # noqa: BLE001
                log.exception("ingest job %s failed", job["job_id"])
                await self.jobs.update(
                    job["job_id"], status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=datetime.now(UTC)
                )

        task = asyncio.create_task(runner(), name=f"ingest-{job['job_id']}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def wait_for_jobs(self) -> None:
        """Tests and the CLI: block until background jobs finish."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
