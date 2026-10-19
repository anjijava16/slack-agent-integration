"""
Incremental indexing into Qdrant through LlamaIndex.

    Documents ──split──▶ nodes ──id = uuid5(source_key | chunk hash | n)──▶ diff with Qdrant
                                                                    ├─ new ids      → embed + upsert (dense + BM25)
                                                                    ├─ known ids    → skip (no embedding cost)
                                                                    └─ missing ids  → delete (chunk changed/removed)

A chunk's id is derived from its content, so re-running ingestion on unchanged
material embeds nothing. This replaces the original bot's Upstash Redis embedding
cache: the vector store itself is the cache.

`prune_group` additionally deletes whole sources that vanished upstream (an FAQ
entry removed from the repo, a file deleted from the course repo).
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, Document, MetadataMode, TextNode

from app.rag.qdrant_store import QdrantCollections
from app.rag.readers import NAMESPACE

log = logging.getLogger(__name__)


@dataclass
class IndexStats:
    documents: int = 0
    sources: int = 0
    chunks: int = 0
    embedded: int = 0
    unchanged: int = 0
    deleted_chunks: int = 0
    deleted_sources: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def add(self, other: IndexStats) -> None:
        for k in ("documents", "sources", "chunks", "embedded", "unchanged", "deleted_chunks", "deleted_sources"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        self.seconds = round(self.seconds + other.seconds, 2)
        self.errors.extend(other.errors)

    def as_dict(self) -> dict:
        return asdict(self)


class Indexer:
    def __init__(self, collections: QdrantCollections, embed_model, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        self.collections = collections
        self.embed_model = embed_model
        self.splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.markdown = MarkdownNodeParser()
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_model.get_text_embedding("dimension probe"))
        return self._dim

    # ---------------------------------------------------------------- chunking
    def split(self, docs: list[Document]) -> list[BaseNode]:
        """Markdown is split on headings first (keeps an FAQ answer or a README section
        together), then anything still too long goes through the sentence splitter."""
        md = [d for d in docs if d.metadata.get("format") == "markdown"]
        other = [d for d in docs if d.metadata.get("format") != "markdown"]
        nodes: list[BaseNode] = []
        if md:
            sections = self.markdown.get_nodes_from_documents(md)
            nodes.extend(self.splitter.get_nodes_from_documents(sections))
        if other:
            nodes.extend(self.splitter.get_nodes_from_documents(other))
        return [n for n in nodes if n.get_content().strip()]

    @staticmethod
    def _assign_ids(source_key: str, nodes: list[BaseNode]) -> None:
        seen: dict[str, int] = defaultdict(int)
        for i, node in enumerate(nodes):
            h = hashlib.sha256(node.get_content(metadata_mode=MetadataMode.EMBED).encode()).hexdigest()[:24]
            dup = seen[h]
            seen[h] += 1
            node.id_ = str(uuid.uuid5(NAMESPACE, f"{source_key}|{h}|{dup}"))
            node.metadata["content_hash"] = h
            node.metadata["chunk_index"] = i
            # LlamaIndex relationships carry the parent doc; drop big parent text from payloads
            node.relationships = {k: v for k, v in node.relationships.items() if k.name == "SOURCE"}

    # ---------------------------------------------------------------- main entry (sync; call in a thread)
    def index(self, course: str, docs: list[Document], prune_group: str | None = None) -> IndexStats:
        started = time.perf_counter()
        stats = IndexStats(documents=len(docs))
        self.collections.ensure(course, self.dim)
        store = self.collections.store(course)

        by_key: dict[str, list[Document]] = defaultdict(list)
        for d in docs:
            by_key[d.metadata["source_key"]].append(d)
        stats.sources = len(by_key)

        pending: list[TextNode] = []
        for key, kdocs in by_key.items():
            nodes = self.split(kdocs)
            self._assign_ids(key, nodes)
            stats.chunks += len(nodes)
            existing = self.collections.ids_for(course, "source_key", key)
            fresh = [n for n in nodes if n.id_ not in existing]
            stale = existing - {n.id_ for n in nodes}
            stats.unchanged += len(nodes) - len(fresh)
            if stale:
                self.collections.delete_ids(course, sorted(stale))
                stats.deleted_chunks += len(stale)
            pending.extend(fresh)
            if len(pending) >= 256:
                stats.embedded += self._embed_and_store(store, pending)
                pending = []
        if pending:
            stats.embedded += self._embed_and_store(store, pending)

        if prune_group:
            gone = self.collections.keys_in_group(course, prune_group) - set(by_key)
            for key in gone:
                self.collections.delete_where(course, "source_key", key)
            stats.deleted_sources = len(gone)

        stats.seconds = round(time.perf_counter() - started, 2)
        log.info("indexed course=%s %s", course, {k: v for k, v in stats.as_dict().items() if k != "errors"})
        return stats

    def _embed_and_store(self, store, nodes: list[TextNode]) -> int:
        texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]
        vectors = self.embed_model.get_text_embedding_batch(texts, show_progress=False)
        for node, vec in zip(nodes, vectors, strict=True):
            node.embedding = vec
        store.add(nodes)  # dense + BM25 sparse (sparse_doc_fn) in one upsert
        return len(nodes)
