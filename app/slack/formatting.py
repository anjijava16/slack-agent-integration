"""
Markdown -> Slack mrkdwn.

LLMs write GitHub-flavoured Markdown even when you ask them not to. Slack's
`mrkdwn` is a different dialect:

    Markdown            Slack mrkdwn
    **bold**            *bold*
    *italic* / _it_     _italic_
    ~~strike~~          ~strike~
    [text](url)         <url|text>
    # Heading           (not supported) -> *Heading*
    - item              (no list syntax) -> • item

Slack also requires &, < and > to be escaped as HTML entities, except inside
its own control sequences like <@U123>, <#C123>, <!here> and <https://...|label>.
"""

from __future__ import annotations

import re

_SLACK_TOKEN = re.compile(r"<(?:[@#!]|https?://|mailto:)[^<>]*>")
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_STRIKE = re.compile(r"~~(.+?)~~")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_BULLET = re.compile(r"^(\s*)[-*+]\s+")


def _escape(text: str) -> str:
    """Escape &, <, > but leave Slack's own <...> tokens alone."""
    out, last = [], 0
    for m in _SLACK_TOKEN.finditer(text):
        out.append(text[last : m.start()].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        out.append(m.group(0))
        last = m.end()
    out.append(text[last:].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return "".join(out)


def _convert_prose(text: str) -> str:
    text = _escape(text)
    lines = []
    for line in text.split("\n"):
        heading = _HEADING.match(line)
        if heading:
            line = f"*{heading.group(1).strip('*')}*"
        else:
            line = _BULLET.sub(lambda m: f"{m.group(1)}• ", line)
        lines.append(line)
    text = "\n".join(lines)
    text = _LINK.sub(lambda m: f"<{m.group(2)}|{m.group(1)}>", text)
    text = _BOLD.sub(lambda m: f"*{m.group(1) or m.group(2)}*", text)
    text = _STRIKE.sub(lambda m: f"~{m.group(1)}~", text)
    return text


def to_mrkdwn(markdown: str) -> str:
    """Convert LLM Markdown to Slack mrkdwn, leaving ``` code blocks untouched."""
    parts = re.split(r"(```.*?```)", markdown, flags=re.DOTALL)
    converted = []
    for part in parts:
        if part.startswith("```") and part.endswith("```") and len(part) >= 6:
            # Slack ignores the language hint (```python) and would print it, so drop it.
            body = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "```\n", part)
            converted.append(_escape(body))
        else:
            converted.append(_convert_prose(part))
    return "".join(converted)


def chunk(text: str, size: int = 2900) -> list[str]:
    """Split text into pieces that fit a section block (3000 char limit),
    preferring paragraph and line boundaries."""
    if len(text) <= size:
        return [text]
    pieces, current = [], ""
    for para in text.split("\n"):
        candidate = f"{current}\n{para}" if current else para
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            pieces.append(current)
        while len(para) > size:
            pieces.append(para[:size])
            para = para[size:]
        current = para
    if current:
        pieces.append(current)
    return pieces
