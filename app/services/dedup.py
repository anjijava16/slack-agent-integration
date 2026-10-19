"""
Event de-duplication.

Slack retries an event (up to 3 times) if it doesn't get a 2xx within 3
seconds, and both a `message` event and an `app_mention` event can fire for
the same text. Without de-dup the bot answers twice. Keys expire after a TTL so
memory stays bounded. For more than one replica, back this with Redis
(SET key 1 NX EX 600) — see README §12.
"""

from __future__ import annotations

import threading
import time


class TTLDeduper:
    def __init__(self, ttl_seconds: float = 600, max_items: int = 10_000) -> None:
        self._ttl = ttl_seconds
        self._max = max_items
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def first_time(self, key: str) -> bool:
        """True the first time a key is seen within the TTL window."""
        now = time.monotonic()
        with self._lock:
            if len(self._seen) > self._max:
                self._seen = {k: t for k, t in self._seen.items() if now - t < self._ttl}
            seen_at = self._seen.get(key)
            if seen_at is not None and now - seen_at < self._ttl:
                return False
            self._seen[key] = now
            return True
