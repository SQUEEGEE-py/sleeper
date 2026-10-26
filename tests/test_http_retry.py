"""The shared retry: transient failures retried twice, real refusals not at all. No network."""

from __future__ import annotations

import httpx
import pytest

from engine.ingest.http_retry import request_with_retry

REQUEST = httpx.Request("GET", "https://api.sleeper.app/v1/league/x")


def answers(*steps):
    """A fake ``send``: each step is a status code, or an exception to raise."""
    calls = []

    def send() -> httpx.Response:
        step = steps[len(calls)]
        calls.append(step)
        if isinstance(step, Exception):
            raise step
        return httpx.Response(step, request=REQUEST, json={})
    return send, calls


def test_a_dropped_connection_then_success_is_retried() -> None:
    send, calls = answers(httpx.ConnectError("reset", request=REQUEST), 200)
    waits = []
    assert request_with_retry(send, sleep=waits.append).status_code == 200
    assert len(calls) == 2 and waits == [2.0]


def test_server_errors_and_rate_limits_are_retried() -> None:
    send, calls = answers(503, 429, 200)
    waits = []
    assert request_with_retry(send, sleep=waits.append).status_code == 200
    assert waits == [2.0, 6.0]


def test_a_real_refusal_fails_at_once() -> None:
    send, calls = answers(404, 200)
    with pytest.raises(httpx.HTTPStatusError):
        request_with_retry(send, sleep=lambda _: None)
    assert len(calls) == 1


def test_gives_up_after_two_retries_with_the_last_error() -> None:
    send, calls = answers(500, 502, 503)
    with pytest.raises(httpx.HTTPStatusError, match="503"):
        request_with_retry(send, sleep=lambda _: None)
    assert len(calls) == 3
    send, calls = answers(*(httpx.ReadTimeout("slow", request=REQUEST) for _ in range(3)))
    with pytest.raises(httpx.ReadTimeout):
        request_with_retry(send, sleep=lambda _: None)


def test_failure_annotation_is_one_escaped_line() -> None:
    from engine.jobs.nightly import actions_error_annotation
    try:
        raise httpx.ConnectError("reset by peer\nsecond line 100%")
    except httpx.ConnectError as error:
        line = actions_error_annotation(error)
    assert line.startswith("::error title=Nightly advisor failed::ConnectError: reset by peer%0Asecond line 100%25")
    assert "\n" not in line and "test_http_retry.py" in line
