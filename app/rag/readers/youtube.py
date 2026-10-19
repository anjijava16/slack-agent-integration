"""YouTube transcripts, chunked with a start time so each source link jumps to the right moment."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import httpx
from llama_index.core.schema import Document

from app.rag.readers import make_document

log = logging.getLogger(__name__)

Snippet = tuple[str, float]  # (text, start seconds)


def _default_fetch(video_id: str) -> list[Snippet]:
    from youtube_transcript_api import YouTubeTranscriptApi

    return [(s.text, float(s.start)) for s in YouTubeTranscriptApi().fetch(video_id)]


def _title(video_id: str) -> str:
    try:
        r = httpx.get(
            "https://www.youtube.com/oembed",
            params={"format": "json", "url": f"https://www.youtube.com/watch?v={video_id}"},
            timeout=15,
        )
        if r.status_code == 200:
            return r.json().get("title", video_id)
    except httpx.HTTPError:
        pass
    return video_id


def chunk_snippets(snippets: list[Snippet], chunk_words: int) -> list[tuple[str, float]]:
    """Group transcript snippets into ~chunk_words-word chunks, keeping each chunk's start time."""
    chunks: list[tuple[str, float]] = []
    buf: list[str] = []
    start, words = 0.0, 0
    for text, s_start in snippets:
        n = len(text.split())
        if buf and words + n > chunk_words:
            chunks.append((" ".join(buf), start))
            buf, words = [], 0
        if not buf:
            start = s_start
        buf.append(text.replace("\n", " "))
        words += n
    if buf:
        chunks.append((" ".join(buf), start))
    return chunks


def load_youtube(
    video_ids: Iterable[str],
    course: str,
    chunk_words: int = 300,
    fetch: Callable[[str], list[Snippet]] = _default_fetch,
    title_of: Callable[[str], str] = _title,
) -> list[Document]:
    docs: list[Document] = []
    for vid in video_ids:
        title = title_of(vid)
        chunks = chunk_snippets(fetch(vid), chunk_words)
        for text, start in chunks:
            t = int(start)
            docs.append(
                make_document(
                    text,
                    course=course,
                    source_type="youtube",
                    source_key=f"youtube:{vid}",
                    source_group=f"youtube:{vid}",
                    title=f"YouTube: {title} ({t // 60}:{t % 60:02d})",
                    source_url=f"https://www.youtube.com/watch?v={vid}&t={t}s",
                    yt_start=t,
                )
            )
        log.info("youtube: %s -> %s chunks", vid, len(chunks))
    return docs
