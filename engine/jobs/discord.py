"""Post a message to the user's Discord channel through a webhook.

The URL lives in ``DISCORD_WEBHOOK_URL`` -- ``.env`` locally, a repository secret in GitHub
Actions -- and is never written anywhere else: anyone holding it can post to the channel.
Discord caps a message at 2000 characters, so longer text is split on line breaks.
"""

from __future__ import annotations

import os

import httpx

from engine.ingest.http_retry import request_with_retry

WEBHOOK_ENV = "DISCORD_WEBHOOK_URL"
MESSAGE_LIMIT = 2000


def webhook_url() -> str | None:
    return os.environ.get(WEBHOOK_ENV) or None


def chunk(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    """Split ``text`` into messages of at most ``limit`` characters, breaking between lines.
    A single line longer than ``limit`` is cut, never dropped."""
    chunks: list[str] = []
    current = ""
    for line in text.splitlines():
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def post(text: str, url: str, timeout: float = 15.0) -> None:
    """Post ``text``, split as needed. Raises on any non-2xx so a failed alert is loud."""
    with httpx.Client(timeout=timeout) as http:
        for message in chunk(text):
            request_with_retry(
                lambda message=message: http.post(url, json={"content": message, "allowed_mentions": {"parse": []}}))
