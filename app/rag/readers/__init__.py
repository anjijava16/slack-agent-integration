"""
Readers turn a source into LlamaIndex Documents with a common metadata contract:

    course        course key ("llm")
    source_type   faq | github | slack | youtube | web | file | text
    source_key    one logical source (one FAQ entry, one file, one Slack thread, one video).
                  Re-ingesting the same key replaces its chunks; unchanged chunks are skipped.
    source_group  a set of keys synced together (all FAQ entries of a course, one repo, one channel).
                  A full sync of a group deletes keys that disappeared upstream.
    source_url    link shown under the answer in Slack
    title         human label for the link
    timestamp     unix seconds, optional (Slack threads) — used for recency weighting
"""

from __future__ import annotations

import uuid

from llama_index.core.schema import Document

# Technical fields that must not leak into the embedded text or the LLM prompt.
HIDDEN_KEYS = [
    "course",
    "source_type",
    "source_key",
    "source_group",
    "source_url",
    "timestamp",
    "content_hash",
    "chunk_index",
    "channel",
    "thread_ts",
    "file_path",
    "faq_id",
    "sort_order",
    "yt_start",
    "format",
    "page",
]

NAMESPACE = uuid.UUID("6f1c5a8e-3c47-4b3e-9b3a-5e9d2c7a1f42")


def make_document(
    text: str,
    *,
    course: str,
    source_type: str,
    source_key: str,
    source_group: str,
    title: str,
    source_url: str = "",
    timestamp: float | None = None,
    fmt: str = "text",
    **extra,
) -> Document:
    metadata = {
        "course": course,
        "source_type": source_type,
        "source_key": source_key,
        "source_group": source_group,
        "source_url": source_url,
        "title": title,
        "format": fmt,
        **{k: v for k, v in extra.items() if v is not None},
    }
    if timestamp is not None:
        metadata["timestamp"] = float(timestamp)
    return Document(
        text=text,
        id_=str(uuid.uuid5(NAMESPACE, f"{course}|{source_key}")),
        metadata=metadata,
        excluded_embed_metadata_keys=list(HIDDEN_KEYS),
        excluded_llm_metadata_keys=list(HIDDEN_KEYS),
    )
