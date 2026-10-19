"""The RAG pipeline as a tool for the general LangGraph agent.

Course channels get the dedicated RAG flow; everywhere else (DMs, other channels,
the REST agent) the agent decides when to call this tool."""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.rag.service import RagService


def make_course_search_tool(rag: RagService) -> StructuredTool:
    keys = rag.catalog.keys()
    described = "; ".join(f"{c.key} = {c.name}" for c in rag.catalog)

    class Args(BaseModel):
        question: str = Field(description="The student's question, self-contained, with exact error text if any")
        course: str = Field(description=f"Course key, one of: {described}")

    async def search_course_materials(question: str, course: str) -> str:
        if course not in keys:
            return f"Unknown course {course!r}. Use one of: {', '.join(keys)}."
        try:
            res = await rag.search(course, question)
        except Exception as exc:  # noqa: BLE001 - tell the model, let it answer gracefully
            return f"Course search is unavailable right now ({type(exc).__name__}). Say so to the user."
        if not res.hits:
            return f"Nothing relevant in the {course} materials (FAQ, Slack history, repo, videos)."
        parts = []
        for i, h in enumerate(res.hits, start=1):
            text = h.text.strip()
            parts.append(f"[{i}] {h.title} <{h.url}>\n{text[:1200]}{'…' if len(text) > 1200 else ''}")
        return "\n\n".join(parts)

    return StructuredTool.from_function(
        coroutine=search_course_materials,
        name="search_course_materials",
        description=(
            "Search a course's FAQ, Slack history, GitHub repo and video transcripts (hybrid keyword + semantic). "
            "Returns numbered excerpts with links. Use for any question about a course."
        ),
        args_schema=Args,
    )
