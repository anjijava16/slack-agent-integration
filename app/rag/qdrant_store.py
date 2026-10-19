"""
Qdrant: connection modes, collections and the LlamaIndex vector store.

QDRANT_MODE
    local     a Qdrant server you run, default http://localhost:6333
              (`make qdrant` / `docker compose --profile qdrant up -d qdrant`)
    cloud     Qdrant Cloud: QDRANT_URL=https://<cluster>.cloud.qdrant.io:6333 + QDRANT_API_KEY
    embedded  no server at all, data in QDRANT_PATH (./data/qdrant); single process only
    memory    in-process and throwaway; unit tests

Collection layout (one per course, "<prefix><course>"):

    vectors:         "dense"  -> size = embedding dim, cosine
    sparse_vectors:  "sparse" -> BM25 term weights, modifier = IDF (Qdrant does the IDF)
    payload indexes: source_key, source_group, source_type (keyword), timestamp (float), doc_id

LlamaIndex's QdrantVectorStore writes the points (text + metadata in the payload,
dense + sparse vectors) and runs the dense and sparse searches; we fuse them with
Reciprocal Rank Fusion.
"""

from __future__ import annotations

import logging
import threading
from contextlib import nullcontext
from dataclasses import dataclass

from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from app.rag import sparse

log = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"
KEYWORD_INDEXES = ("source_key", "source_group", "source_type", "doc_id")


class CollectionMismatchError(RuntimeError):
    """The collection was built with a different embedding size than the current model."""


@dataclass
class QdrantConfig:
    mode: str = "local"  # local | cloud | embedded | memory
    url: str = "http://localhost:6333"
    api_key: str = ""
    path: str = "./data/qdrant"
    prefix: str = "faq_"
    timeout: int = 30
    hybrid: bool = True
    sparse_provider: str = "bm25"  # bm25 | fastembed
    sparse_model: str = "Qdrant/bm25"
    rrf_k: int = 60


def build_client(cfg: QdrantConfig) -> QdrantClient:
    if cfg.mode == "memory":
        return QdrantClient(location=":memory:")
    if cfg.mode == "embedded":
        return QdrantClient(path=cfg.path)
    if cfg.mode == "cloud":
        if not cfg.api_key:
            raise ValueError("QDRANT_MODE=cloud needs QDRANT_API_KEY (and QDRANT_URL of your cluster)")
        return QdrantClient(url=cfg.url, api_key=cfg.api_key, timeout=cfg.timeout)
    if cfg.mode == "local":
        return QdrantClient(url=cfg.url, api_key=cfg.api_key or None, timeout=cfg.timeout)
    raise ValueError(f"Unknown QDRANT_MODE {cfg.mode!r}")


def rrf_fusion(
    dense_result: VectorStoreQueryResult,
    sparse_result: VectorStoreQueryResult,
    alpha: float = 0.5,
    top_k: int = 10,
    k: int = 60,
) -> VectorStoreQueryResult:
    """Reciprocal Rank Fusion: score = alpha/(k + rank_dense) + (1-alpha)/(k + rank_sparse).

    Ranks, not raw scores, are combined, so cosine similarities (0..1) and BM25
    scores (unbounded) don't have to be put on one scale. A chunk found by both
    searches beats one found by a single search at the same rank."""
    scores: dict[str, float] = {}
    nodes: dict[str, object] = {}
    for weight, result in ((alpha, dense_result), (1.0 - alpha, sparse_result)):
        for rank, node in enumerate(result.nodes or [], start=1):
            scores[node.node_id] = scores.get(node.node_id, 0.0) + weight / (k + rank)
            nodes.setdefault(node.node_id, node)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return VectorStoreQueryResult(
        nodes=[nodes[i] for i, _ in ranked], similarities=[s for _, s in ranked], ids=[i for i, _ in ranked]
    )


class QdrantCollections:
    """Owns the client; hands out per-course vector stores; creates collections lazily."""

    def __init__(self, cfg: QdrantConfig, client: QdrantClient | None = None) -> None:
        self.cfg = cfg
        self.client = client or build_client(cfg)
        self._stores: dict[str, QdrantVectorStore] = {}
        # Embedded/in-memory Qdrant runs inside this process and isn't thread-safe;
        # searches and ingestion run in worker threads, so serialise them there.
        self._lock = threading.RLock() if cfg.mode in ("memory", "embedded") else None
        if cfg.hybrid and cfg.sparse_provider == "fastembed":
            self._sparse_doc_fn, self._sparse_query_fn = sparse.fastembed_sparse_fns(cfg.sparse_model)
        else:
            self._sparse_doc_fn, self._sparse_query_fn = sparse.sparse_doc_fn, sparse.sparse_query_fn

    def guard(self):
        return self._lock if self._lock is not None else nullcontext()

    def name(self, course_key: str) -> str:
        return f"{self.cfg.prefix}{course_key}"

    def ping(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False

    def exists(self, course_key: str) -> bool:
        return self.client.collection_exists(self.name(course_key))

    def ensure(self, course_key: str, dim: int) -> str:
        """Create the collection if needed; refuse if it exists with another vector size."""
        name = self.name(course_key)
        if self.client.collection_exists(name):
            info = self.client.get_collection(name)
            vectors = info.config.params.vectors
            existing = vectors[DENSE].size if isinstance(vectors, dict) and DENSE in vectors else None
            if existing is not None and existing != dim:
                raise CollectionMismatchError(
                    f"Collection {name!r} stores {existing}-dim vectors but the embedding model makes {dim}-dim ones. "
                    f"Reset it (POST /api/ingest/collections/{course_key}/reset) and ingest again."
                )
            return name
        sparse_cfg = None
        if self.cfg.hybrid:
            sparse_cfg = {
                SPARSE: qm.SparseVectorParams(
                    index=qm.SparseIndexParams(on_disk=False),
                    modifier=qm.Modifier.IDF if sparse.requires_idf(self.cfg.sparse_provider, self.cfg.sparse_model) else None,
                )
            }
        self.client.create_collection(
            collection_name=name,
            vectors_config={DENSE: qm.VectorParams(size=dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config=sparse_cfg,
        )
        if self.cfg.mode not in ("memory", "embedded"):  # local mode ignores payload indexes
            for field in KEYWORD_INDEXES:
                self.client.create_payload_index(name, field, qm.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(name, "timestamp", qm.PayloadSchemaType.FLOAT)
        log.info("created Qdrant collection %s (dim=%s, hybrid=%s)", name, dim, self.cfg.hybrid)
        return name

    def store(self, course_key: str) -> QdrantVectorStore:
        if course_key not in self._stores:
            self._stores[course_key] = QdrantVectorStore(
                collection_name=self.name(course_key),
                client=self.client,
                enable_hybrid=self.cfg.hybrid,
                dense_vector_name=DENSE,
                sparse_vector_name=SPARSE,
                sparse_doc_fn=self._sparse_doc_fn if self.cfg.hybrid else None,
                sparse_query_fn=self._sparse_query_fn if self.cfg.hybrid else None,
                hybrid_fusion_fn=(lambda d, s, alpha=0.5, top_k=10: rrf_fusion(d, s, alpha, top_k, self.cfg.rrf_k))
                if self.cfg.hybrid
                else None,
                index_doc_id=False,  # we create our own payload indexes in ensure()
                batch_size=64,
            )
        return self._stores[course_key]

    def count(self, course_key: str) -> int:
        if not self.exists(course_key):
            return 0
        return self.client.count(self.name(course_key), exact=True).count

    def reset(self, course_key: str) -> bool:
        self._stores.pop(course_key, None)
        name = self.name(course_key)
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
            return True
        return False

    def source_counts(self, course_key: str, field: str = "source_type") -> dict[str, int]:
        if not self.exists(course_key):
            return {}
        try:
            facets = self.client.facet(self.name(course_key), key=field, limit=50, exact=True)
            return {str(h.value): h.count for h in facets.hits}
        except Exception:  # noqa: BLE001 - facet needs a payload index (not in memory/embedded mode)
            counts: dict[str, int] = {}
            offset = None
            while True:
                points, offset = self.client.scroll(
                    self.name(course_key), limit=512, offset=offset, with_payload=[field], with_vectors=False
                )
                for p in points:
                    v = str((p.payload or {}).get(field))
                    counts[v] = counts.get(v, 0) + 1
                if offset is None:
                    return counts

    def ids_for(self, course_key: str, field: str, value: str) -> set[str]:
        """All point ids whose payload[field] == value."""
        ids: set[str] = set()
        offset = None
        flt = qm.Filter(must=[qm.FieldCondition(key=field, match=qm.MatchValue(value=value))])
        while True:
            points, offset = self.client.scroll(
                self.name(course_key), scroll_filter=flt, limit=1024, offset=offset, with_payload=False, with_vectors=False
            )
            ids.update(str(p.id) for p in points)
            if offset is None:
                return ids

    def keys_in_group(self, course_key: str, group: str) -> set[str]:
        keys: set[str] = set()
        offset = None
        flt = qm.Filter(must=[qm.FieldCondition(key="source_group", match=qm.MatchValue(value=group))])
        while True:
            points, offset = self.client.scroll(
                self.name(course_key),
                scroll_filter=flt,
                limit=1024,
                offset=offset,
                with_payload=["source_key"],
                with_vectors=False,
            )
            keys.update(str((p.payload or {}).get("source_key")) for p in points)
            if offset is None:
                return keys

    def delete_ids(self, course_key: str, ids: list[str]) -> None:
        if ids:
            self.client.delete(self.name(course_key), points_selector=qm.PointIdsList(points=ids))

    def delete_where(self, course_key: str, field: str, value: str) -> None:
        flt = qm.Filter(must=[qm.FieldCondition(key=field, match=qm.MatchValue(value=value))])
        self.client.delete(self.name(course_key), points_selector=qm.FilterSelector(filter=flt))
