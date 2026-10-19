"""
Command line for the course RAG, using the same settings (.env / environment) as the app.

    python -m app.rag.cli status
    python -m app.rag.cli ingest llm                         # faq + github + slack + youtube
    python -m app.rag.cli ingest llm --sources faq,github
    python -m app.rag.cli ingest-text llm "Office hours" "Thursdays 17:00 CET"
    python -m app.rag.cli search llm "elasticsearch connection refused" --mode sparse
    python -m app.rag.cli ask llm "Can I still join the course?"
    python -m app.rag.cli reset llm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from app.config import Settings
from app.db.feedback_store import build_feedback_store
from app.rag.service import SOURCES, RagService


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


async def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.rag.cli", description="Course RAG: ingest, search, ask")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    ing = sub.add_parser("ingest")
    ing.add_argument("course")
    ing.add_argument("--sources", default=",".join(SOURCES), help=f"comma list of {','.join(SOURCES)}")
    txt = sub.add_parser("ingest-text")
    txt.add_argument("course")
    txt.add_argument("title")
    txt.add_argument("text")
    srch = sub.add_parser("search")
    srch.add_argument("course")
    srch.add_argument("query")
    srch.add_argument("--mode", choices=["hybrid", "dense", "sparse"])
    srch.add_argument("--top", type=int, default=None)
    ask = sub.add_parser("ask")
    ask.add_argument("course")
    ask.add_argument("question")
    rst = sub.add_parser("reset")
    rst.add_argument("course")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    settings = Settings()
    store = build_feedback_store(settings.database_url)
    await store.init()
    llm = None
    if args.cmd == "ask":
        from app.agent.llm import build_llm

        llm = build_llm(settings)
    rag = RagService(settings, llm, store)
    try:
        if args.cmd == "status":
            _print(await rag.status())
        elif args.cmd == "ingest":
            _print(await rag.ingest_course(args.course, [s.strip() for s in args.sources.split(",") if s.strip()]))
        elif args.cmd == "ingest-text":
            _print((await rag.ingest_text(args.course, args.text, args.title)).as_dict())
        elif args.cmd == "search":
            res = await rag.search(args.course, args.query, mode=args.mode, final_top_n=args.top)
            for i, h in enumerate(res.hits, start=1):
                print(f"[{i}] {h.score:.3f}  {h.title}\n    {h.url}\n    {h.text[:200].strip()!r}")
            print(f"funnel={res.funnel} timings_ms={res.timings_ms} reranker={res.reranker}")
        elif args.cmd == "ask":
            res = await rag.answer(args.question, args.course, thread_id="cli", source="cli")
            print(res.answer, "\n\nSources:")
            for s in res.sources:
                print(f"  - {s['title']}  {s['url']}")
        elif args.cmd == "reset":
            print("deleted" if await rag.reset(args.course) else "nothing to delete")
    finally:
        await store.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
