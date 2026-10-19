"""
The RAG flow as a LangGraph — the same building blocks as the agent graph, but a fixed path:

    START → guardrail ─blocked─────────────────────────────▶ END
                └─ok─▶ condense → retrieve ─no hits─▶ finalize → END
                                     └─hits─▶ generate ─▶ finalize

condense   rewrites a follow-up ("and on Windows?") into a standalone search query using
           the thread's history (LangGraph checkpointer, same thread_id scheme as the agent)
retrieve   hybrid search + recency + rerank in the course's Qdrant collection
generate   the course prompt + numbered excerpts → answer with [n] citations
finalize   sets final_answer; sources travel separately in state["hits"]
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from app.agent.graph import _INJECTION_MARKERS, _text_of
from app.rag import prompts
from app.rag.courses import CourseCatalog
from app.rag.retriever import SearchResult

log = logging.getLogger(__name__)


class RagState(MessagesState):
    course: str
    question: str
    search_query: str
    hits: list[dict]
    search_meta: dict
    final_answer: str
    blocked_reason: str
    user_id: str
    channel_id: str
    source: str


def build_rag_graph(
    llm: BaseChatModel,
    catalog: CourseCatalog,
    search: Callable[[str, str], SearchResult],
    checkpointer: BaseCheckpointSaver | None = None,
    max_input_chars: int = 4000,
    history_turns: int = 3,
):
    """`search(course_key, query)` is the (sync) retriever call; it runs in a worker thread."""

    async def guardrail(state: RagState) -> dict:
        q = state["question"].strip()
        reason = ""
        if not q:
            reason = "Ooops, your question is empty. Tag me together with your question."
        elif len(q) > max_input_chars:
            reason = f"That's {len(q):,} characters; please keep questions under {max_input_chars:,}."
        elif any(m in q.lower() for m in _INJECTION_MARKERS):
            reason = "I can only help with questions about the course."
        reset = {"blocked_reason": reason, "final_answer": reason, "hits": [], "search_meta": {}, "search_query": ""}
        if reason:
            return {**reset, "messages": [HumanMessage(content=q), AIMessage(content=reason)]}
        return {**reset, "messages": [HumanMessage(content=q)]}

    async def condense(state: RagState) -> dict:
        history = [m for m in state["messages"][:-1] if isinstance(m, HumanMessage | AIMessage)]
        question = state["question"].strip()
        if not history:
            return {"search_query": question}
        recent = history[-2 * history_turns :]
        transcript = "\n".join(
            f"{'Student' if isinstance(m, HumanMessage) else 'Assistant'}: {_text_of(m)[:600]}" for m in recent
        )
        try:
            out = await llm.ainvoke([HumanMessage(content=prompts.CONDENSE.format(history=transcript, question=question))])
            rewritten = _text_of(out).strip().strip('"') or question
        except Exception:  # noqa: BLE001 - fall back to the raw question
            log.warning("condense failed; searching with the raw question", exc_info=True)
            rewritten = question
        return {"search_query": rewritten}

    async def retrieve(state: RagState) -> dict:
        result = await asyncio.to_thread(search, state["course"], state["search_query"] or state["question"])
        return {
            "hits": [h.as_dict() for h in result.hits],
            "search_meta": {
                "mode": result.mode,
                "reranker": result.reranker,
                "funnel": result.funnel,
                "timings_ms": result.timings_ms,
                "query": result.query,
            },
        }

    async def generate(state: RagState) -> dict:
        course = catalog.get(state["course"])
        system = prompts.SYSTEM.format(
            course_name=course.name,
            cohort_year=course.cohort_year or "current",
            course_start_date=course.course_start_date or "see the course page",
            today=datetime.now(UTC).strftime("%d %B %Y"),
        )
        user = prompts.USER.format(context=prompts.format_context(state["hits"]), question=state["question"])
        # Earlier turns give the model conversational context; excerpts are only for this turn.
        prior = [m for m in state["messages"][:-1] if isinstance(m, HumanMessage | AIMessage)][-2 * history_turns :]
        out = await llm.ainvoke([SystemMessage(content=system), *prior, HumanMessage(content=user)])
        return {"messages": [AIMessage(content=_text_of(out).strip())]}

    async def no_hits(state: RagState) -> dict:
        text = prompts.NO_CONTEXT.format(course=catalog.get(state["course"]).name)
        return {"messages": [AIMessage(content=text)]}

    async def finalize(state: RagState) -> dict:
        return {"final_answer": _text_of(state["messages"][-1]).strip()}

    def after_guardrail(state: RagState) -> Literal["condense", "__end__"]:
        return END if state.get("blocked_reason") else "condense"

    def after_retrieve(state: RagState) -> Literal["generate", "no_hits"]:
        return "generate" if state.get("hits") else "no_hits"

    g = StateGraph(RagState)
    for name, fn in [
        ("guardrail", guardrail),
        ("condense", condense),
        ("retrieve", retrieve),
        ("generate", generate),
        ("no_hits", no_hits),
        ("finalize", finalize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "guardrail")
    g.add_conditional_edges("guardrail", after_guardrail, {"condense": "condense", END: END})
    g.add_edge("condense", "retrieve")
    g.add_conditional_edges("retrieve", after_retrieve, {"generate": "generate", "no_hits": "no_hits"})
    g.add_edge("generate", "finalize")
    g.add_edge("no_hits", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
