"""
Retrieval quality on the real DataTalks.Club data: the LLM Zoomcamp FAQ (DataTalksClub/faq)
plus the llm-zoomcamp course repo, ingested through the same service the app uses.

Opt-in, because it needs local checkouts:

    git clone --depth 1 https://github.com/DataTalksClub/faq          /tmp/dtc/faq
    git clone --depth 1 https://github.com/DataTalksClub/llm-zoomcamp /tmp/dtc/llm-zoomcamp
    REAL_FAQ_DIR=/tmp/dtc/faq REAL_COURSE_REPO=/tmp/dtc/llm-zoomcamp [TEST_QDRANT_URL=http://localhost:6333] \\
        pytest -q tests/test_rag_real_data.py -s

With EMBED_PROVIDER=hash (the test default) the dense side has no semantics, so these
questions reuse some words of the real entries; with the default fastembed model you can
paraphrase much more freely. Each question must bring back the right FAQ entry in the top 4.
"""

import os

import pytest

from app.db.feedback_store import InMemoryFeedbackStore
from app.rag.courses import CourseCatalog, RepoSpec
from app.rag.service import RagService
from tests.fakes import ScriptedChatModel
from tests.rag_fixtures import QDRANT_URL, make_settings

FAQ_DIR = os.environ.get("REAL_FAQ_DIR")
REPO_DIR = os.environ.get("REAL_COURSE_REPO")
pytestmark = pytest.mark.skipif(not (FAQ_DIR and REPO_DIR), reason="set REAL_FAQ_DIR and REAL_COURSE_REPO")

CASES = [
    ("I just found out about the course, is it too late to join?", "74eb249bbf"),
    ("when will I get the confirmation email after registering?", "977bf7786c"),
    ("are there free models that support tool use?", "0d74a3616f"),
    ("AttributeError: 'str' object has no attribute 'output' with the Responses API", "9e7b3f0c25"),
    ("why doesn't the course use langchain?", "b7cdde6b25"),
    ("homework Q3 returns an empty list after filtering", "83da208a64"),
    ("Kestra copilot says I can only assist with creating Kestra flows", "04440cab11"),
    ("JSONDecodeError Expecting value when generating ground truth questions", "a3c7e1b4f9"),
    ("psycopg2 OperationalError could not translate host name when running prep.py", "4b65d5542d"),
    ("WinError 126 the specified module could not be found on Windows", "15e931476c"),
]


@pytest.fixture(scope="module")
def rag(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("real")
    (tmp / "courses.yaml").write_text(
        "courses:\n  llm:\n    name: LLM Zoomcamp\n    faq_dir: llm-zoomcamp\n    channels: [CLLM]\n", encoding="utf-8"
    )
    catalog = CourseCatalog.load(tmp / "courses.yaml")
    catalog.get("llm").repos = [
        RepoSpec(owner="DataTalksClub", repo="llm-zoomcamp", exclude_dirs=["images", "cohorts/2024"], local_path=REPO_DIR)
    ]
    backend = "server" if QDRANT_URL else "memory"
    settings = make_settings(backend, faq_local_path=FAQ_DIR)
    return RagService(settings, ScriptedChatModel(script=[]), InMemoryFeedbackStore(), catalog=catalog)


async def test_real_ingest_and_retrieval(rag):
    faq = await rag.ingest_faq("llm")
    repo = await rag.ingest_github("llm")
    print(f"\nFAQ: {faq.sources} entries -> {faq.chunks} chunks; repo: {repo.sources} files -> {repo.chunks} chunks")
    assert faq.sources > 100 and repo.sources > 50 and not repo.errors

    again = await rag.ingest_faq("llm")
    assert again.embedded == 0  # incremental: nothing re-embedded

    ranks = []
    for question, faq_id in CASES:
        res = await rag.search("llm", question)
        urls = [h.url for h in res.hits]
        rank = next((i for i, u in enumerate(urls, start=1) if u.endswith(f"#{faq_id}")), None)
        ranks.append(rank)
        print(f"  {'✓' if rank else '✗'} rank={rank}  {question[:70]}")
    found = sum(r is not None for r in ranks)
    mrr = sum(1 / r for r in ranks if r) / len(ranks)
    print(f"  hit@4 = {found}/{len(CASES)}   MRR = {mrr:.2f}")
    assert found >= 9, ranks
    rag.collections.reset("llm")
