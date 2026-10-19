"""
GitHub repositories: course repos and the DataTalksClub/faq repo.

One HTTP request per repo: the tarball (codeload.github.com for public repos, the
API tarball endpoint when GITHUB_TOKEN is set — needed for private repos and
higher rate limits). No per-file API calls, so no 60-requests-an-hour wall like
the original GithubRepositoryReader hit without a token. A local checkout can be
used instead (RepoSpec.local_path), which is also how the tests run.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import httpx
from llama_index.core.schema import Document

from app.rag.courses import RepoSpec
from app.rag.readers import make_document

log = logging.getLogger(__name__)

MAX_FILE_BYTES = 300_000
_BASE64_IMG = re.compile(r"!\[[^\]]*\]\(\s*data:image/[^)]+;base64,[^)]+\)", re.IGNORECASE)


@contextmanager
def checkout(spec: RepoSpec, token: str = "", http: httpx.Client | None = None) -> Iterator[Path]:
    """Yield a directory with the repo's files: the local path, or a downloaded tarball."""
    if spec.local_path:
        yield Path(spec.local_path)
        return
    tmp = Path(tempfile.mkdtemp(prefix="repo-"))
    try:
        if token:
            url = f"https://api.github.com/repos/{spec.owner}/{spec.repo}/tarball/{spec.branch}"
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        else:
            url = f"https://codeload.github.com/{spec.owner}/{spec.repo}/tar.gz/refs/heads/{spec.branch}"
            headers = {}
        client = http or httpx.Client(timeout=120, follow_redirects=True)
        archive = tmp / "repo.tar.gz"
        with client.stream("GET", url, headers=headers) as resp:
            if resp.status_code == 404:
                raise FileNotFoundError(f"{spec.full_name}@{spec.branch} not found (private repo? set GITHUB_TOKEN)")
            resp.raise_for_status()
            with archive.open("wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
        with tarfile.open(archive) as tar:
            _safe_extract(tar, tmp / "src")
        roots = [p for p in (tmp / "src").iterdir() if p.is_dir()]
        yield roots[0] if len(roots) == 1 else tmp / "src"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        tar.extractall(dest, filter="data")  # Python 3.12+ (and 3.11.4+): blocks path traversal, links, devices
    except TypeError:  # pragma: no cover - very old Python
        base = dest.resolve()
        for m in tar.getmembers():
            if not (dest / m.name).resolve().is_relative_to(base) or m.issym() or m.islnk():
                raise ValueError(f"unsafe path in archive: {m.name}") from None
        tar.extractall(dest)


def notebook_to_text(raw: str) -> str:
    """Markdown and code cells, no outputs, no embedded images."""
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    parts = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        if not src.strip():
            continue
        parts.append(f"```python\n{src}\n```" if cell.get("cell_type") == "code" else src)
    return _BASE64_IMG.sub("", "\n\n".join(parts))


def _excluded(rel: PurePosixPath, exclude_dirs: list[str]) -> bool:
    rel_s = rel.as_posix()
    if any(part.startswith(".") for part in rel.parts[:-1]):  # .github, .tmp, .devcontainer ...
        return True
    return any(rel_s == d or rel_s.startswith(d.rstrip("/") + "/") or d in rel.parts[:-1] for d in exclude_dirs)


def iter_files(root: Path, include_ext: list[str], exclude_dirs: list[str]) -> Iterator[tuple[str, str]]:
    include = {e.lower() for e in include_ext}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        rel = PurePosixPath(path.relative_to(root).as_posix())
        ext = path.suffix.lower() or ("." + path.name.lower())
        if ext not in include or _excluded(rel, exclude_dirs):
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            log.info("skip %s (%s bytes)", rel, path.stat().st_size)
            continue
        yield rel.as_posix(), path.read_text(encoding="utf-8", errors="replace")


def load_repo(root: Path, spec: RepoSpec, course: str) -> list[Document]:
    docs = []
    group = f"github:{spec.full_name}@{spec.branch}"
    for rel, raw in iter_files(root, spec.include_ext, spec.exclude_dirs):
        ext = PurePosixPath(rel).suffix.lower()
        if ext == ".ipynb":
            text, fmt = notebook_to_text(raw), "markdown"
        elif ext == ".md":
            text, fmt = _BASE64_IMG.sub("", raw), "markdown"
        else:
            lang = ext.lstrip(".") or "text"
            text, fmt = f"File: {rel}\n```{lang}\n{raw}\n```", "code"
        if not text.strip():
            continue
        docs.append(
            make_document(
                text,
                course=course,
                source_type="github",
                source_key=f"github:{spec.full_name}:{rel}",
                source_group=group,
                title=f"GitHub {spec.repo}/{rel}",
                source_url=f"https://github.com/{spec.full_name}/blob/{spec.branch}/{rel}",
                fmt=fmt,
                file_path=rel,
            )
        )
    return docs


def load_faq(root: Path, faq_dir: str, course: str) -> list[Document]:
    """DataTalksClub/faq layout: _questions/<faq_dir>/<module>/<nnn>_<id>_<slug>.md with YAML frontmatter
    (id, question, sort_order). Each entry links to https://datatalks.club/faq/<faq_dir>.html#<id>."""
    import frontmatter

    base = root / "_questions" / faq_dir
    if not base.is_dir():
        raise FileNotFoundError(f"no FAQ folder _questions/{faq_dir} in the FAQ repo")
    docs = []
    for path in sorted(base.rglob("*.md")):
        try:
            post = frontmatter.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("bad frontmatter in %s: %s", path, exc)
            continue
        meta = post.metadata
        if not all(k in meta for k in ("id", "question")):
            continue
        rel = path.relative_to(base)
        module = rel.parts[0] if len(rel.parts) > 1 else "general"
        question = str(meta["question"]).strip()
        docs.append(
            make_document(
                f"{question}\n\n{post.content}",
                course=course,
                source_type="faq",
                source_key=f"faq:{faq_dir}:{meta['id']}",
                source_group=f"faq:{faq_dir}",
                title=f"FAQ · {module} · {question[:70]}{'…' if len(question) > 70 else ''}",
                source_url=f"https://datatalks.club/faq/{faq_dir}.html#{meta['id']}",
                fmt="markdown",
                faq_id=str(meta["id"]),
                module=module,
                question=question,
                sort_order=meta.get("sort_order"),
            )
        )
    return docs
