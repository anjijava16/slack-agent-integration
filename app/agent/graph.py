"""
The LangGraph agent.

    START
      │
      ▼
  ┌──────────┐  blocked   ┌─────┐
  │guardrail │──────────▶│ END │
  └──────────┘            └─────┘
      │ ok                   ▲
      ▼                      │
  ┌──────────┐  no tool    ┌──────────┐
  │  agent   │───calls───▶│ finalize │
  └──────────┘             └──────────┘
    ▲     │ tool calls
    │     ▼
  ┌──────────┐
  │  tools   │
  └──────────┘

agent <-> tools is the ReAct loop: the model decides which tool to call, the
ToolNode runs it, the result goes back into the message list, and the model
looks again. The loop ends when the model replies without tool calls (or when
recursion_limit is hit).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.state import AgentState
from app.agent.tools import ALL_TOOLS

# Phrases that are almost always prompt-injection attempts. This is a cheap
# first line of defence, not a security boundary — real protection comes from
# tools that can't do damage (InjectedState, allow-lists, read-only creds).
_INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard your instructions",
    "reveal your system prompt",
)

_HISTORY_WINDOW = 30  # how many past messages from the thread we send to the LLM


def _text_of(message: AnyMessage) -> str:
    """Message content can be a string or a list of content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _window(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Keep the last N messages, but never start on an orphaned ToolMessage /
    tool-calling AIMessage — OpenAI rejects a tool result without its call."""
    recent = messages[-_HISTORY_WINDOW:]
    while recent and not isinstance(recent[0], HumanMessage):
        recent = recent[1:]
    return recent or messages[-1:]


def build_graph(
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    max_input_chars: int = 4000,
    extra_tools: list | None = None,
):
    # extra_tools: tools that need runtime objects, e.g. search_course_materials (needs the RAG service)
    tools = [*ALL_TOOLS, *(extra_tools or [])]
    llm_with_tools = llm.bind_tools(tools)

    # ------------------------------------------------------------- nodes
    def guardrail(state: AgentState) -> dict:
        last = state["messages"][-1]
        text = _text_of(last).strip()
        reason = ""
        if not text:
            reason = "Send me a question or a task and I'll get on it."
        elif len(text) > max_input_chars:
            reason = f"That message is {len(text):,} characters. Please keep it under {max_input_chars:,}."
        elif any(marker in text.lower() for marker in _INJECTION_MARKERS):
            reason = "I can't do that one. Ask me something about IT, HR policy, tickets, math or time."

        if reason:
            return {"blocked_reason": reason, "final_answer": reason, "messages": [AIMessage(content=reason)]}
        # Reset per-turn fields — the checkpointer remembers them from the last turn.
        return {"blocked_reason": "", "final_answer": ""}

    async def agent(state: AgentState) -> dict:
        # async so a slow LLM call never blocks FastAPI's event loop
        # (which is also serving Slack events and REST requests).
        system = SystemMessage(
            content=SYSTEM_PROMPT.format(
                user_id=state.get("user_id", "unknown"),
                channel_id=state.get("channel_id", "unknown"),
                source=state.get("source", "unknown"),
                now=datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
            )
        )
        response = await llm_with_tools.ainvoke([system, *_window(state["messages"])])
        return {"messages": [response]}

    def finalize(state: AgentState) -> dict:
        answer = _text_of(state["messages"][-1]).strip() or "I finished, but had nothing to say. Try rephrasing?"
        return {"final_answer": answer}

    # ------------------------------------------------------------- routing
    def after_guardrail(state: AgentState) -> Literal["agent", "__end__"]:
        return END if state.get("blocked_reason") else "agent"

    def after_agent(state: AgentState) -> Literal["tools", "finalize"]:
        # tools_condition returns "tools" when the last AIMessage has tool_calls, else END
        return "tools" if tools_condition(state) == "tools" else "finalize"

    # ------------------------------------------------------------- wiring
    builder = StateGraph(AgentState)
    builder.add_node("guardrail", guardrail)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges("guardrail", after_guardrail, {"agent": "agent", END: END})
    builder.add_conditional_edges("agent", after_agent, {"tools": "tools", "finalize": "finalize"})
    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)
