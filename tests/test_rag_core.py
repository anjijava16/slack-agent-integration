"""
RAG building blocks: BM25 sparse vectors, RRF, course catalogue, every reader,
incremental indexing and the retrieval pipeline.

Qdrant-backed tests run on in-memory Qdrant always, and on a real server too when
TEST_QDRANT_URL is set (e.g. `make qdrant` then TEST_QDRANT_URL=http://localhost:6333).
"""

import asyncio
import io
import tarfile
import time
from pathlib import Path

import httpx
import pytest
from llama_index.core.vector_stores.types import VectorStoreQueryResult

from app.rag import sparse
from app.rag.courses import CourseCatalog, RepoSpec
from app.rag.embeddings import HashEmbedding
from app.rag.indexer import Indexer
from app.rag.qdrant_store import CollectionMismatchError, rrf_fusion
from app.rag.readers import make_document
from app.rag.readers.github import checkout, load_faq, load_repo
from app.rag.readers.slack import SlackHistoryReader, permalink
from app.rag.readers.web import html_to_text, load_bytes, load_url
from app.rag.readers.youtube import chunk_snippets, load_youtube
from app.rag.retriever import RetrievalConfig, Retriever
from tests.rag_fixtures import BACKENDS, QDRANT_URL, collections_for, make_catalog, make_course_repo, make_faq_repo


@pytest.fixture(params=BACKENDS)
def collections(request):
    if request.param == "server" and not QDRANT_URL:
        pytest.skip("set TEST_QDRANT_URL to run against a Qdrant server")
    cols = collections_for(request.param)
    yield cols
    for c in ("llm", "de", "x"):
        cols.reset(c)


EMB = HashEmbedding(model_name="hash")


# ------------------------------------------------------------------ sparse / fusion
def test_tokenizer_keeps_identifiers_and_parts():
    toks = sparse.tokenize("ERR_CONNECTION_REFUSED in pg_hba.conf while running containers")
    assert "err_connection_refused" in toks and "pg_hba.conf" in toks
    assert "hba" in toks and "run" in toks and "contain" in toks  # parts + stemming
    assert "in" not in toks and "while" not in toks  # stopwords


def test_token_ids_are_stable_and_bm25_saturates():
    assert sparse.token_id("docker") == sparse.token_id("docker")  # not Python's salted hash()
    one = dict(zip(*sparse.encode_document("docker"), strict=True))
    many = dict(zip(*sparse.encode_document("docker " * 50), strict=True))
    w1, w50 = one[sparse.token_id("docker")], many[sparse.token_id("docker")]
    assert w50 > w1 and w50 < 2.2 * w1 + 1  # tf helps, with diminishing returns (k1 saturation)
    idx, vals = sparse.encode_query("docker docker compose")
    assert len(idx) == 2 and vals == [1.0, 1.0]


def test_rrf_rewards_agreement():
    from llama_index.core.schema import TextNode

    a, b, c = (TextNode(text=t, id_=t) for t in "abc")
    dense = VectorStoreQueryResult(nodes=[a, b], similarities=[0.9, 0.8], ids=["a", "b"])
    keyword = VectorStoreQueryResult(nodes=[c, b], similarities=[12.0, 3.0], ids=["c", "b"])
    fused = rrf_fusion(dense, keyword, alpha=0.5, top_k=3)
    assert fused.ids[0] == "b"  # in both lists beats #1 in only one
    assert set(fused.ids) == {"a", "b", "c"}


# ------------------------------------------------------------------ catalogue
def test_catalog_routing(tmp_path):
    cat = make_catalog(tmp_path)
    assert cat.for_channel("CLLM").key == "llm" and cat.for_channel("CDE").key == "de"
    assert cat.for_channel("COTHER") is None
    extra = CourseCatalog.load(tmp_path / "courses.yaml", default_course="de", extra_channels="CNEW=llm")
    assert extra.for_channel("CNEW").key == "llm" and extra.for_channel("COTHER").key == "de"
    with pytest.raises(ValueError):
        CourseCatalog.load(tmp_path / "courses.yaml", extra_channels="CX=nope")


def test_shipped_courses_yaml_loads():
    cat = CourseCatalog.load("config/courses.yaml")
    assert set(cat.keys()) == {"de", "ml", "mlops", "llm"}
    assert cat.for_channel("C06TEGTGM3J").key == "llm"  # same routing table as the original bot
    assert cat.get("ml").repos[0].branch == "master"


# ------------------------------------------------------------------ readers
def test_faq_reader(tmp_path):
    docs = load_faq(make_faq_repo(tmp_path), "llm-zoomcamp", "llm")
    assert len(docs) == 4
    es = next(d for d in docs if d.metadata["faq_id"] == "bbb222")
    assert es.metadata["source_url"] == "https://datatalks.club/faq/llm-zoomcamp.html#bbb222"
    assert es.metadata["module"] == "module-1" and es.text.startswith("Elasticsearch: ConnectionError")
    assert "source_url" in es.excluded_embed_metadata_keys and "title" not in es.excluded_embed_metadata_keys


def test_repo_reader_filters_and_notebooks(tmp_path):
    root = make_course_repo(tmp_path)
    spec = RepoSpec(owner="DataTalksClub", repo="llm-zoomcamp", exclude_dirs=["cohorts/2024", "images"])
    docs = {d.metadata["file_path"]: d for d in load_repo(root, spec, "llm")}
    assert set(docs) == {"README.md", "01-intro/README.md", "02-vector-search/search.py", "02-vector-search/notebook.ipynb"}
    nb = docs["02-vector-search/notebook.ipynb"].text
    assert "query_points" in nb and "base64" not in nb and "BIG OUTPUT" not in nb
    assert docs["README.md"].metadata["source_url"] == "https://github.com/DataTalksClub/llm-zoomcamp/blob/main/README.md"


def _tarball(root: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(root, arcname="llm-zoomcamp-main")
    return buf.getvalue()


@pytest.mark.parametrize("token", ["", "ghp_secret"])
def test_checkout_downloads_tarball(tmp_path, token):
    data = _tarball(make_course_repo(tmp_path / "src"))
    seen = {}

    def handler(request: httpx.Request):
        seen["url"], seen["auth"] = str(request.url), request.headers.get("authorization")
        return httpx.Response(200, content=data)

    spec = RepoSpec(owner="DataTalksClub", repo="llm-zoomcamp", branch="main")
    with checkout(spec, token, http=httpx.Client(transport=httpx.MockTransport(handler))) as root:
        assert (root / "README.md").exists()
        extracted = root
    assert not extracted.exists()  # temp dir cleaned up
    if token:
        assert seen["url"] == "https://api.github.com/repos/DataTalksClub/llm-zoomcamp/tarball/main"
        assert seen["auth"] == "Bearer ghp_secret"
    else:
        assert seen["url"] == "https://codeload.github.com/DataTalksClub/llm-zoomcamp/tar.gz/refs/heads/main"


def test_checkout_rejects_path_traversal(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("../../evil.txt")
        info.size = 4
        tar.addfile(info, io.BytesIO(b"evil"))
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=buf.getvalue())))
    with pytest.raises(Exception):  # noqa: B017 - tarfile's filter error type varies by Python version
        with checkout(RepoSpec(owner="o", repo="r"), "", http=client):
            pass


class FakeSlack:
    def __init__(self):
        now = time.time()
        self.history = [
            {"ts": f"{now - 3600:.6f}", "reply_count": 2, "text": "q1"},
            {"ts": f"{now - 7200:.6f}", "reply_count": 1, "text": "q2"},  # only the bot answered
            {"ts": f"{now - 9000:.6f}", "text": "unanswered"},
            {"ts": f"{now - 9500:.6f}", "reply_count": 1, "subtype": "channel_join"},
        ]
        self.replies = {
            self.history[0]["ts"]: [
                {"user": "U1", "text": "Docker says port 5432 is already allocated", "ts": self.history[0]["ts"]},
                {"user": "U2", "text": "Stop the local postgres or map -p 5433:5432", "ts": f"{now - 3000:.6f}"},
                {"user": "UBOT", "text": "bot answer", "ts": f"{now - 2900:.6f}"},
            ],
            self.history[1]["ts"]: [
                {"user": "U3", "text": "hello?", "ts": self.history[1]["ts"]},
                {"user": "UBOT", "bot_id": "B1", "text": "bot answer", "ts": f"{now - 7000:.6f}"},
            ],
        }

    async def auth_test(self):
        return {"user_id": "UBOT", "url": "https://dtc.slack.com/"}

    async def conversations_info(self, channel):
        return {"channel": {"name": "course-llm-zoomcamp"}}

    async def conversations_history(self, channel, oldest, cursor, limit):
        return {"messages": self.history, "has_more": False}

    async def conversations_replies(self, channel, ts, cursor, limit):
        return {"messages": self.replies[ts], "has_more": False}


async def test_slack_reader_keeps_answered_human_threads():
    docs = await SlackHistoryReader(FakeSlack()).load(["CLLM"], days=30, course="llm")
    assert len(docs) == 1
    d = docs[0]
    assert "5433:5432" in d.text and "bot answer" not in d.text
    assert d.metadata["source_url"].startswith("https://dtc.slack.com/archives/CLLM/p")
    assert d.metadata["title"].startswith("Slack #course-llm-zoomcamp: Docker says port")
    assert d.metadata["timestamp"] > float(FakeSlack().history[0]["ts"])  # latest reply, for recency
    assert permalink("https://x.slack.com/", "C1", "1700000000.000100") == "https://x.slack.com/archives/C1/p1700000000000100"


def test_youtube_chunks_with_timestamps():
    snippets = [(f"word{i} " * 50, float(i * 30)) for i in range(10)]  # 50 words every 30 s
    assert [round(s) for _, s in chunk_snippets(snippets, 120)] == [0, 60, 120, 180, 240]
    docs = load_youtube(["vid123"], "llm", chunk_words=120, fetch=lambda v: snippets, title_of=lambda v: "Office hours")
    assert len(docs) == 5
    assert docs[1].metadata["source_url"] == "https://www.youtube.com/watch?v=vid123&t=60s"
    assert docs[1].metadata["title"] == "YouTube: Office hours (1:00)"


async def test_web_and_files():
    title, text = html_to_text(
        "<html><head><title>Docs</title><style>x{}</style></head>"
        "<body><nav>menu</nav><h1>Setup</h1><p>Run make.</p></body></html>"
    )
    assert title == "Docs" and "Run make." in text and "menu" not in text and "x{}" not in text
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, html="<title>T</title><p>Hello course</p>", headers={"content-type": "text/html"})
        )
    )
    doc = await load_url("https://example.org/page", "llm", http=client)
    assert doc.metadata["source_key"] == "url:https://example.org/page" and "Hello course" in doc.text

    md = load_bytes("notes.md", b"# Notes\n\nUse uv.", "llm")
    assert md[0].metadata["format"] == "markdown" and md[0].metadata["source_key"] == "file:notes.md"
    nb = load_bytes("x.ipynb", b'{"cells":[{"cell_type":"code","source":"print(1)"}]}', "llm")
    assert "print(1)" in nb[0].text
    with pytest.raises(ValueError):
        load_bytes("photo.png", b"\x89PNG", "llm")


def _tiny_pdf(text: str) -> bytes:
    stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = io.BytesIO(), []
    out.write(b"%PDF-1.4\n")
    for i, body in enumerate(objs, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % i + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1))
    for off in offsets:
        out.write(b"%010d 00000 n \n" % off)
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref))
    return out.getvalue()


def test_pdf_pages():
    docs = load_bytes("syllabus.pdf", _tiny_pdf("Homework deadline is Monday"), "llm")
    assert len(docs) == 1 and "Homework deadline is Monday" in docs[0].text and docs[0].metadata["page"] == 1


# ------------------------------------------------------------------ indexing
def _doc(key, text, group="g", **kw):
    return make_document(
        text, course="llm", source_type=kw.pop("source_type", "text"), source_key=key, source_group=group, title=key, **kw
    )


def test_incremental_indexing(collections, tmp_path):
    ix = Indexer(collections, EMB)
    docs = load_faq(make_faq_repo(tmp_path), "llm-zoomcamp", "llm")
    first = ix.index("llm", docs, prune_group="faq:llm-zoomcamp")
    assert first.embedded == first.chunks == collections.count("llm") > 0

    again = ix.index("llm", docs, prune_group="faq:llm-zoomcamp")
    assert again.embedded == 0 and again.unchanged == first.chunks  # nothing re-embedded

    docs[0].set_content(docs[0].text + "\n\nUpdate: the 2026 cohort starts in June.")
    edited = ix.index("llm", docs[1:] + docs[:1], prune_group="faq:llm-zoomcamp")
    assert edited.embedded == 1 and edited.deleted_chunks == 1

    pruned = ix.index("llm", docs[:2], prune_group="faq:llm-zoomcamp")
    assert pruned.deleted_sources == 2
    assert collections.count("llm") == pruned.chunks


def test_dimension_mismatch_is_explicit(collections):
    Indexer(collections, EMB).index("x", [_doc("k", "hello docker")])
    small = HashEmbedding(model_name="small", dim=64)
    with pytest.raises(CollectionMismatchError, match="Reset it"):
        Indexer(collections, small).index("x", [_doc("k2", "hello again")])


# ------------------------------------------------------------------ retrieval
def _index_mixed(collections, tmp_path):
    ix = Indexer(collections, EMB)
    ix.index("llm", load_faq(make_faq_repo(tmp_path), "llm-zoomcamp", "llm"), prune_group="faq:llm-zoomcamp")
    now = time.time()
    ix.index(
        "llm",
        [
            _doc(
                "slack:old",
                "Docker port 5432 already allocated: stop local postgres",
                source_type="slack",
                timestamp=now - 400 * 86400,
                source_url="https://x/old",
            ),
            _doc(
                "slack:new",
                "Docker port 5432 already allocated: stop local postgres!",
                source_type="slack",
                timestamp=now - 1 * 86400,
                source_url="https://x/new",
            ),
        ],
    )
    return Retriever(collections, EMB, RetrievalConfig(top_k=10, recency_top_k=6, final_top_n=3))


def test_hybrid_finds_exact_error_text(collections, tmp_path):
    r = _index_mixed(collections, tmp_path)
    res = r.search("llm", "ConnectionError localhost:9200")
    assert res.hits[0].url.endswith("#bbb222")
    assert res.funnel["retrieved"] >= res.funnel["after_recency"] >= res.funnel["final"] == 3
    sparse_only = r.search("llm", "9200", mode="sparse")
    assert sparse_only.hits[0].url.endswith("#bbb222")


def test_recency_prefers_newer_slack_thread(collections, tmp_path):
    r = _index_mixed(collections, tmp_path)
    hits = r.search("llm", "port 5432 already allocated", source_types=["slack"]).hits
    assert [h.url for h in hits] == ["https://x/new", "https://x/old"]
    assert {h.source_type for h in hits} == {"slack"}


class FlippingReranker:
    name = "flip"

    def scores(self, query, texts):
        return list(range(len(texts)))  # reverses the order


class BrokenReranker:
    name = "broken"

    def scores(self, query, texts):
        raise httpx.ConnectError("cohere down")


def test_reranker_reorders_and_failure_falls_back(collections, tmp_path):
    base = _index_mixed(collections, tmp_path)
    plain = base.search("llm", "docker", final_top_n=6)
    flipped = Retriever(collections, EMB, base.cfg, FlippingReranker()).search("llm", "docker", final_top_n=6)
    assert [h.id for h in flipped.hits] == [h.id for h in reversed(plain.hits)]
    broken = Retriever(collections, EMB, base.cfg, BrokenReranker()).search("llm", "docker", final_top_n=6)
    assert [h.id for h in broken.hits] == [h.id for h in plain.hits] and broken.reranker == "broken (failed)"


def test_search_on_missing_collection_is_empty(collections):
    assert Retriever(collections, EMB, RetrievalConfig()).search("de", "anything").hits == []


# ------------------------------------------------------------------ init failures don't hang or crash
async def test_init_failure_is_reported_and_backed_off(tmp_path, monkeypatch):
    from app.rag import service as svc
    from tests.fakes import ScriptedChatModel
    from tests.rag_fixtures import make_rag

    rag = make_rag(tmp_path, ScriptedChatModel(script=[]), embed_provider="openai", embed_base_url="http://127.0.0.1:9")
    calls = []

    def broken_build():
        calls.append(1)
        raise ValueError("Could not load model BAAI/bge-small-en-v1.5 from any source.")

    monkeypatch.setattr(rag, "_build", broken_build)
    with pytest.raises(svc.RagUnavailable, match="Could not load model"):
        await rag.ready()
    with pytest.raises(svc.RagUnavailable, match="Could not load model"):
        await rag.ready()  # within the back-off window: no second attempt
    assert len(calls) == 1
    assert (await rag.status())["init_error"].startswith("ValueError: Could not load model")


async def test_daemon_thread_returns_values_and_errors():
    from app.rag.service import run_in_daemon_thread

    assert await run_in_daemon_thread(lambda: 41 + 1) == 42
    with pytest.raises(ZeroDivisionError):
        await run_in_daemon_thread(lambda: 1 / 0)


def test_daemon_thread_survives_loop_shutdown():
    import threading

    from app.rag.service import run_in_daemon_thread

    release = threading.Event()
    errors = []
    threading.excepthook, old_hook = (lambda args: errors.append(args.exc_value)), threading.excepthook
    try:

        async def start_and_leave():
            task = asyncio.ensure_future(run_in_daemon_thread(lambda: release.wait(5) or "late"))
            await asyncio.sleep(0.05)
            task.cancel()

        asyncio.run(start_and_leave())  # loop is closed after this returns
        release.set()  # the worker now finishes and tries to report back to a closed loop
        time.sleep(0.2)
    finally:
        threading.excepthook = old_hook
    assert errors == []
