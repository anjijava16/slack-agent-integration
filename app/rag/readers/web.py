"""Web pages and uploaded/raw files."""

from __future__ import annotations

import io
import re
from html.parser import HTMLParser
from pathlib import PurePosixPath

import httpx
from llama_index.core.schema import Document

from app.rag.readers import make_document
from app.rag.readers.github import notebook_to_text

TEXT_EXT = {".md", ".markdown", ".txt", ".rst", ".py", ".sql", ".sh", ".yaml", ".yml", ".toml", ".json", ".csv", ".html"}


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "svg", "form"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "pre", "section", "article"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    p = _TextExtractor()
    p.feed(html)
    text = re.sub(r"[ \t]+", " ", "".join(p.parts))
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return p.title.strip(), text


async def load_url(url: str, course: str, http: httpx.AsyncClient | None = None) -> Document:
    client = http or httpx.AsyncClient(timeout=30, follow_redirects=True, headers={"User-Agent": "faq-bot/1.0"})
    resp = await client.get(url)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if "html" in ctype:
        title, text = html_to_text(resp.text)
    else:
        title, text = url, resp.text
    if not text.strip():
        raise ValueError(f"no text found at {url}")
    return make_document(
        text,
        course=course,
        source_type="web",
        source_key=f"url:{url}",
        source_group="web",
        title=title or url,
        source_url=url,
    )


def load_bytes(
    filename: str,
    data: bytes,
    course: str,
    *,
    source_key: str | None = None,
    title: str | None = None,
    url: str = "",
) -> list[Document]:
    """Uploaded file -> Documents. PDF pages become separate documents (page links work)."""
    ext = PurePosixPath(filename).suffix.lower()
    key = source_key or f"file:{filename}"
    common = dict(course=course, source_type="file", source_key=key, source_group="files", source_url=url)
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        docs = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(make_document(text, title=f"{title or filename} (p.{i})", page=i, **common))
        return docs
    raw = data.decode("utf-8", errors="replace")
    if ext == ".ipynb":
        return [make_document(notebook_to_text(raw), title=title or filename, fmt="markdown", **common)]
    if ext in (".html", ".htm"):
        page_title, text = html_to_text(raw)
        return [make_document(text, title=title or page_title or filename, **common)]
    if ext not in TEXT_EXT and ext:
        raise ValueError(f"unsupported file type {ext!r} (pdf, ipynb, html and text formats are)")
    fmt = "markdown" if ext in (".md", ".markdown") else "text"
    return [make_document(raw, title=title or filename, fmt=fmt, **common)]
