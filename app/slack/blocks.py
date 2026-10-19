"""
Block Kit builders.

A Slack message has a plain `text` (used for notifications, screen readers
and as a fallback) and optional `blocks` (the rich layout). Always send both.
"""

from __future__ import annotations

from app.services.agent_service import AgentResult
from app.slack.formatting import chunk, to_mrkdwn

MAX_BLOCKS = 50  # Slack's per-message limit


def thinking_blocks(user_id: str | None = None) -> list[dict]:
    who = f"<@{user_id}> " if user_id else ""
    return [
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f":hourglass_flowing_sand: {who}working on it…"}],
        }
    ]


def answer_blocks(result: AgentResult, *, with_feedback: bool = True) -> list[dict]:
    blocks: list[dict] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": piece}} for piece in chunk(to_mrkdwn(result.answer))
    ]

    meta = []
    if result.tool_calls:
        names = ", ".join(dict.fromkeys(f"`{t.name}`" for t in result.tool_calls))  # de-dupe, keep order
        meta.append(f":hammer_and_wrench: {names}")
    meta.append(f":stopwatch: {result.duration_ms / 1000:.1f}s")
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": "  ·  ".join(meta)}]})

    if with_feedback and not result.blocked and result.run_id:
        blocks.append(feedback_actions(result.run_id))
    return blocks[:MAX_BLOCKS]


def feedback_actions(run_id: str) -> dict:
    """👍 / 👎 buttons. action_id picks the listener (@app.action("upvote") /
    @app.action("downvote")); value carries the run_id so the vote can be joined
    to the exact answer in Postgres (agent_runs.run_id)."""
    return {
        "type": "actions",
        "block_id": "agent_feedback",
        "elements": [
            {
                "type": "button",
                "action_id": "upvote",
                "text": {"type": "plain_text", "text": ":thumbsup: Helpful", "emoji": True},
                "value": run_id,
            },
            {
                "type": "button",
                "action_id": "downvote",
                "text": {"type": "plain_text", "text": ":thumbsdown: Not helpful", "emoji": True},
                "value": run_id,
            },
        ],
    }


def fallback_text(result: AgentResult) -> str:
    """Plain text for notifications. Keep it short — it shows in push previews."""
    text = to_mrkdwn(result.answer)
    return text if len(text) <= 3000 else text[:2997] + "..."


# ---------------------------------------------------------------- RAG (course FAQ) answers
_SOURCE_ICON = {
    "faq": ":question:",
    "slack": ":speech_balloon:",
    "github": ":octocat:",
    "youtube": ":tv:",
    "web": ":globe_with_meridians:",
    "file": ":page_facing_up:",
    "text": ":memo:",
}


def rag_thinking_blocks(course_name: str, user_id: str | None = None) -> list[dict]:
    who = f"<@{user_id}> " if user_id else ""
    return [
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":mag: {who}searching the *{course_name}* FAQ, Slack threads, course repo and videos…",
                }
            ],
        }
    ]


def _slack_link_text(title: str) -> str:
    return title.replace("|", "¦").replace(">", "›").replace("<", "‹")


def rag_answer_blocks(res, course_name: str, *, with_feedback: bool = True) -> list[dict]:
    """Answer, then a Sources section built from the retrieved chunks (not from the LLM,
    so links are always real), then a context line and the 👍/👎 buttons."""
    blocks: list[dict] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": piece}} for piece in chunk(to_mrkdwn(res.answer))
    ]
    if res.sources:
        lines = []
        for s in res.sources[:8]:
            icon = _SOURCE_ICON.get(s.get("source_type") or "", ":link:")
            title = _slack_link_text(s.get("title") or s.get("url") or "source")
            lines.append(f"{icon} <{s['url']}|{title}>" if s.get("url") else f"{icon} {title}")
        blocks += [{"type": "divider"}, {"type": "section", "text": {"type": "mrkdwn", "text": "*Sources*\n" + "\n".join(lines)}}]
    meta = res.search_meta or {}
    funnel = meta.get("funnel") or {}
    bits = [f":books: {course_name}"]
    if meta:
        bits.append(f"{meta.get('mode', 'hybrid')} search")
        if funnel:
            bits.append("→".join(str(funnel.get(k, "?")) for k in ("retrieved", "after_recency", "final")) + " chunks")
        if meta.get("reranker") and meta["reranker"] != "none":
            bits.append(f"rerank: {meta['reranker']}")
    bits.append(f":stopwatch: {res.duration_ms / 1000:.1f}s")
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": "  ·  ".join(bits)}]})
    if with_feedback and not res.blocked and res.run_id:
        blocks.append(feedback_actions(res.run_id))
    return blocks[:MAX_BLOCKS]
