"""One retry policy for every outside call: Sleeper, ESPN, Discord.

The scheduled run on Mon 2026-10-26 died after 32 seconds -- early, in a fetch -- and with no
retry anywhere, one dropped connection or one 5xx from any of ~40 requests kills the whole
report. So a request is retried twice, after 2 s and 6 s, but only for what is plausibly
transient: timeouts and connection errors, 5xx, and 429 (rate limited). Any other 4xx is a real
answer and fails at once -- retrying a refusal is just hammering the endpoint (CLAUDE.md:
"do not silently retry into a rate limit"; two spaced retries are not that).
"""

from __future__ import annotations

import time
from typing import Callable

import httpx

RETRY_DELAYS_SECONDS: tuple[float, ...] = (2.0, 6.0)


def _transient(status: int) -> bool:
    return status >= 500 or status == 429


def request_with_retry(send: Callable[[], httpx.Response], delays: tuple[float, ...] = RETRY_DELAYS_SECONDS,
                       sleep: Callable[[float], None] = time.sleep) -> httpx.Response:
    """Call ``send`` until it gives a non-transient answer; raise the last error otherwise."""
    for attempt in range(len(delays) + 1):
        last = attempt == len(delays)
        try:
            response = send()
        except httpx.TransportError:
            if last:
                raise
            sleep(delays[attempt])
            continue
        if _transient(response.status_code) and not last:
            sleep(delays[attempt])
            continue
        response.raise_for_status()
        return response
    raise AssertionError("unreachable")
