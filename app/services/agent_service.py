"""
AgentService — the one place that knows how to run the graph.

Slack handlers and REST routes both call AgentService.run(). They don't know
anything about LangGraph; the graph doesn't know anything about Slack. That
seam is what lets you test the agent without Slack, and swap Slack for Teams
without touching the agent.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from app.agent.graph import build_graph
from app.db.feedback_store import FeedbackStore, RunRecord

log = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    output: str


@dataclass
class AgentResult:
    answer: str
    thread_id: str
    run_id: str = ""  # one id per answer; the 👍/👎 buttons carry it back to us
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    duration_ms: int = 0
    blocked: bool = False


class AgentService:
    def __init__(
        self,
        llm: BaseChatModel,
        max_steps: int = 12,
        max_input_chars: int = 4000,
        store: FeedbackStore | None = None,
        model_name: str | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        extra_tools: list | None = None,
    ) -> None:
        # store: where every run is written (agent_runs table) so feedback can join to it.
        self._store = store
        self._model_name = model_name
        # Conversation memory per thread_id. main.py passes a SQLite (local) or
        # Postgres (cloud) saver; InMemorySaver is the fallback for tests.
        self._checkpointer = checkpointer or InMemorySaver()
        self._graph = build_graph(llm, checkpointer=self._checkpointer, max_input_chars=max_input_chars, extra_tools=extra_tools)
        self._max_steps = max_steps
        # Two Slack messages in the same thread arriving at once would race on the
        # same checkpoint. A lock per thread serialises them; different threads run in parallel.
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @property
    def graph(self):
        return self._graph

    async def run(
        self,
        text: str,
        *,
        thread_id: str,
        user_id: str = "api-user",
        channel_id: str = "api",
        source: str = "api",
    ) -> AgentResult:
        started = time.perf_counter()
        run_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": self._max_steps}
        inputs = {
            "messages": [HumanMessage(content=text)],
            "user_id": user_id,
            "channel_id": channel_id,
            "source": source,
        }

        async with self._locks[thread_id]:
            try:
                state = await self._graph.ainvoke(inputs, config=config)
            except GraphRecursionError:
                log.warning("agent hit recursion limit thread=%s", thread_id)
                result = AgentResult(
                    answer="I went around in circles on that one and stopped. Could you narrow the question down?",
                    thread_id=thread_id,
                    run_id=run_id,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
                await self._record(result, text, user_id, channel_id, source)
                return result

        result = AgentResult(
            answer=state.get("final_answer") or "",
            thread_id=thread_id,
            run_id=run_id,
            tool_calls=_tool_calls_this_turn(state["messages"]),
            duration_ms=int((time.perf_counter() - started) * 1000),
            blocked=bool(state.get("blocked_reason")),
        )
        await self._record(result, text, user_id, channel_id, source)
        log.info(
            "agent done run=%s thread=%s source=%s tools=%s ms=%s",
            run_id,
            thread_id,
            source,
            [t.name for t in result.tool_calls],
            result.duration_ms,
        )
        return result

    async def _record(self, result: AgentResult, question: str, user_id: str, channel_id: str, source: str) -> None:
        """Persist the run. Never fails the answer: a DB outage costs us analytics, not the user's reply."""
        if self._store is None:
            return
        try:
            await self._store.record_run(
                RunRecord(
                    run_id=result.run_id,
                    thread_id=result.thread_id,
                    source=source,
                    user_id=user_id,
                    channel_id=channel_id,
                    question=question,
                    answer=result.answer,
                    tool_calls=[{"name": t.name, "args": t.args} for t in result.tool_calls],
                    model=self._model_name,
                    duration_ms=result.duration_ms,
                    blocked=result.blocked,
                )
            )
        except Exception:  # noqa: BLE001
            log.exception("could not record run %s", result.run_id)


def _tool_calls_this_turn(messages) -> list[ToolCallRecord]:
    """Walk back to the latest HumanMessage and collect tool calls after it."""
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            start = i
            break
    outputs = {m.tool_call_id: str(m.content) for m in messages[start:] if isinstance(m, ToolMessage)}
    records = []
    for m in messages[start:]:
        if isinstance(m, AIMessage):
            for call in m.tool_calls:
                records.append(ToolCallRecord(call["name"], call.get("args", {}), outputs.get(call["id"], "")))
    return records
