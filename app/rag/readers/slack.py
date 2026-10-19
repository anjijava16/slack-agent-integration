"""
Slack history: answered threads from the course channels, the most valuable source
in the original bot (students hit the same problems every cohort).

A thread is indexed when someone other than the bot replied. The bot's own answers
are left out so it doesn't learn from itself. Needs channels:history (+ groups:history
for private channels) and channels:read for channel names.
"""

from __future__ import annotations

import logging
import time

from llama_index.core.schema import Document
from slack_sdk.http_retry.builtin_async_handlers import AsyncRateLimitErrorRetryHandler
from slack_sdk.web.async_client import AsyncWebClient

from app.rag.readers import make_document

log = logging.getLogger(__name__)


def permalink(workspace_url: str, channel: str, ts: str) -> str:
    base = (workspace_url or "https://slack.com/").rstrip("/")
    return f"{base}/archives/{channel}/p{ts.replace('.', '')}"


class SlackHistoryReader:
    def __init__(self, client: AsyncWebClient, workspace_url: str = "", bot_user_id: str = "") -> None:
        self.client = client
        self.workspace_url = workspace_url
        self.bot_user_id = bot_user_id

    @classmethod
    def from_token(cls, token: str, workspace_url: str = "") -> SlackHistoryReader:
        client = AsyncWebClient(token=token)
        client.retry_handlers.append(AsyncRateLimitErrorRetryHandler(max_retry_count=5))
        return cls(client, workspace_url)

    async def _identify(self) -> None:
        if self.bot_user_id and self.workspace_url:
            return
        auth = await self.client.auth_test()
        self.bot_user_id = self.bot_user_id or auth.get("user_id", "")
        self.workspace_url = self.workspace_url or auth.get("url", "")

    async def load(self, channels: list[str], days: int, course: str) -> list[Document]:
        await self._identify()
        oldest = str(time.time() - days * 86400)
        docs: list[Document] = []
        for channel in channels:
            name = channel
            try:
                info = await self.client.conversations_info(channel=channel)
                name = info["channel"].get("name", channel)
            except Exception:  # noqa: BLE001 - channels:read missing; ids are fine
                pass
            cursor = None
            while True:
                page = await self.client.conversations_history(channel=channel, oldest=oldest, cursor=cursor, limit=200)
                for msg in page.get("messages", []):
                    if msg.get("reply_count", 0) == 0 or msg.get("subtype"):
                        continue
                    doc = await self._thread(channel, name, msg["ts"], course)
                    if doc is not None:
                        docs.append(doc)
                cursor = (page.get("response_metadata") or {}).get("next_cursor")
                if not page.get("has_more") or not cursor:
                    break
            log.info("slack: %s threads from #%s", sum(1 for d in docs if d.metadata["channel"] == channel), name)
        return docs

    async def _thread(self, channel: str, name: str, ts: str, course: str) -> Document | None:
        texts: list[str] = []
        authors: set[str] = set()
        latest = float(ts)
        cursor = None
        while True:
            page = await self.client.conversations_replies(channel=channel, ts=ts, cursor=cursor, limit=200)
            for m in page.get("messages", []):
                user = m.get("user", "")
                if m.get("bot_id") or user == self.bot_user_id or not m.get("text"):
                    continue
                texts.append(m["text"])
                authors.add(user)
                latest = max(latest, float(m.get("ts", ts)))
            cursor = (page.get("response_metadata") or {}).get("next_cursor")
            if not page.get("has_more") or not cursor:
                break
        if len(texts) < 2 or len(authors) < 1:  # a question nobody (human) answered
            return None
        first_line = texts[0].strip().splitlines()[0][:80]
        return make_document(
            "\n\n".join(texts),
            course=course,
            source_type="slack",
            source_key=f"slack:{channel}:{ts}",
            source_group=f"slack:{channel}",
            title=f"Slack #{name}: {first_line}",
            source_url=permalink(self.workspace_url, channel, ts),
            timestamp=latest,
            channel=channel,
            thread_ts=ts,
        )
