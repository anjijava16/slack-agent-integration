"""
The agent's state — the single object that flows through every node.

MessagesState gives us `messages: Annotated[list[AnyMessage], add_messages]`.
`add_messages` is a *reducer*: when a node returns {"messages": [msg]}, LangGraph
appends (or replaces by id) instead of overwriting the list. Every other key
uses the default reducer, which simply overwrites.
"""

from __future__ import annotations

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    # Who/where — filled by the caller (Slack handler or REST endpoint)
    user_id: str
    channel_id: str
    source: str  # "slack_mention" | "slack_dm" | "slash_command" | "api"

    # Set by the guardrail node; when non-empty the graph skips the LLM entirely
    blocked_reason: str

    # Set by the finalize node; this is what gets posted back to Slack
    final_answer: str
