"""
Retrieval pipeline for one course collection:

    question ─▶ dense embedding ┐
             └▶ BM25 sparse     ┴─▶ Qdrant: two searches, top_k each ─▶ RRF fusion (top_k)
                                                                         │
                         recency weighting (Slack threads decay with age) ─▶ recency_top_k
                                                                         │
                                         re-ranker (cross-encoder) ─▶ final_top_n hits

Recency: score = relevance (normalised to 0..1) + weight * 0.5 ** (age_days / half_life_days)
for chunks that carry a timestamp (Slack threads). Answers from last week's thread beat
the same answer from two cohorts ago, while FAQ entries and repo files are left alone.
The original bot used LlamaIndex's TimeWeightedPostprocessor on thread_ts for the same reason.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from app.rag.qdrant_store import QdrantCollections
from app.rag.rerank import NoRerank, Reranker

log = logging.getLogger(__name__)

MODES = {
    "hybrid": VectorStoreQueryMode.HYBRID,
    "dense": VectorStoreQueryMode.DEFAULT,
    "sparse": VectorStoreQueryMode.SPARSE,
}


@dataclass
class Hit:
    id: str
    text: str
    score: float
    title: str
    url: str
    source_type: str
    source_key: str
    timestamp: float | None = None
    rerank_score: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchResult:
    hits: list[Hit]
    query: str
    mode: str
    reranker: str
    timings_ms: dict[str, int] = field(default_factory=dict)
    funnel: dict[str, int] = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    mode: str = "hybrid"
    top_k: int = 20
    recency_top_k: int = 10
    final_top_n: int = 4
    alpha: float = 0.5
    half_life_days: float = 30.0
    recency_weight: float = 0.3


class Retriever:
    def __init__(self, collections: QdrantCollections, embed_model, cfg: RetrievalConfig, reranker: Reranker | None = None):
        self.collections = collections
        self.embed_model = embed_model
        self.cfg = cfg
        self.reranker = reranker or NoRerank()

    def search(
        self,
        course: str,
        query: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        final_top_n: int | None = None,
        source_types: list[str] | None = None,
        rerank: bool = True,
        now: float | None = None,
    ) -> SearchResult:
        cfg = self.cfg
        mode = mode or cfg.mode
        top_k = top_k or cfg.top_k
        final_top_n = final_top_n or cfg.final_top_n
        recency_top_k = max(final_top_n, min(cfg.recency_top_k, top_k))
        result = SearchResult(hits=[], query=query, mode=mode, reranker=self.reranker.name if rerank else "none")
        if not self.collections.exists(course):
            return result

        t0 = time.perf_counter()
        embedding = None if mode == "sparse" else self.embed_model.get_query_embedding(query)
        t1 = time.perf_counter()
        filters = None
        if source_types:
            filters = MetadataFilters(filters=[MetadataFilter(key="source_type", value=source_types, operator=FilterOperator.IN)])
        vq = VectorStoreQuery(
            query_embedding=embedding,
            query_str=query,
            similarity_top_k=top_k,
            sparse_top_k=top_k,
            hybrid_top_k=top_k,
            alpha=cfg.alpha,
            mode=MODES[mode],
            filters=filters,
        )
        with self.collections.guard():
            res = self.collections.store(course).query(vq)
        t2 = time.perf_counter()

        hits = []
        for node, score in zip(res.nodes or [], res.similarities or [], strict=False):
            md = node.metadata or {}
            hits.append(
                Hit(
                    id=node.node_id,
                    text=node.get_content(),
                    score=float(score),
                    title=str(md.get("title", "")),
                    url=str(md.get("source_url", "")),
                    source_type=str(md.get("source_type", "")),
                    source_key=str(md.get("source_key", "")),
                    timestamp=md.get("timestamp"),
                )
            )
        result.funnel["retrieved"] = len(hits)

        # --- recency weighting
        if hits:
            top = max(h.score for h in hits) or 1.0
            now = now or time.time()
            for h in hits:
                h.score = h.score / top
                if h.timestamp:
                    age_days = max(0.0, (now - float(h.timestamp)) / 86400)
                    h.score += cfg.recency_weight * 0.5 ** (age_days / cfg.half_life_days)
            hits.sort(key=lambda h: h.score, reverse=True)
            hits = hits[:recency_top_k]
        result.funnel["after_recency"] = len(hits)
        t3 = time.perf_counter()

        # --- re-rank
        if hits and rerank and not isinstance(self.reranker, NoRerank):
            try:
                scores = self.reranker.scores(query, [h.text for h in hits])
                for h, s in zip(hits, scores, strict=True):
                    h.rerank_score = s
                hits.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
            except Exception as exc:  # noqa: BLE001 - a reranker outage shouldn't kill answers
                log.warning("rerank failed (%s), keeping hybrid+recency order", exc)
                result.reranker = f"{self.reranker.name} (failed)"
        result.hits = hits[:final_top_n]
        result.funnel["final"] = len(result.hits)
        t4 = time.perf_counter()
        result.timings_ms = {
            "embed": int((t1 - t0) * 1000),
            "search": int((t2 - t1) * 1000),
            "recency": int((t3 - t2) * 1000),
            "rerank": int((t4 - t3) * 1000),
        }
        return result
