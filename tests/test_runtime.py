"""runtime.run_loop and utf8_output: a failed pass is logged and the loop goes on. No network."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from engine.jobs.runtime import run_loop, utf8_output


def test_a_failing_pass_is_logged_and_the_loop_continues(tmp_path: Path) -> None:
    passes: list[int] = []
    errors: list[tuple[str, Path]] = []

    def step() -> None:
        passes.append(len(passes))
        if len(passes) == 2:
            raise ValueError("bad pass")

    run_loop("job", step, interval=0, should_stop=lambda: len(passes) == 4,
             on_error=lambda error, path: errors.append((str(error), path)),
             sleep=lambda _: None, log_dir=tmp_path)

    assert passes == [0, 1, 2, 3]
    assert errors == [("bad pass", tmp_path / "job.log")]
    log = (tmp_path / "job.log").read_text(encoding="utf-8")
    assert "Traceback" in log and "ValueError: bad pass" in log and "in step" in log


def test_should_stop_is_checked_after_a_failure_so_once_still_exits(tmp_path: Path) -> None:
    calls: list[int] = []

    def step() -> None:
        calls.append(1)
        raise RuntimeError("always")

    run_loop("job", step, interval=0, should_stop=lambda: True, on_error=lambda *_: None,
             sleep=lambda _: None, log_dir=tmp_path)
    assert calls == [1]


def test_keyboard_interrupt_still_stops_the_loop(tmp_path: Path) -> None:
    def step() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_loop("job", step, interval=0, should_stop=lambda: False, on_error=lambda *_: None,
                 sleep=lambda _: None, log_dir=tmp_path)


def test_utf8_output_lets_a_cp1252_stream_print_any_name(monkeypatch: pytest.MonkeyPatch) -> None:
    # What redirected output looks like on Windows: a cp1252 text stream.
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    with pytest.raises(UnicodeEncodeError):
        stream.write("Vučević")
    monkeypatch.setattr(sys, "stdout", stream)
    utf8_output()
    print("Vučević Dončić")
    sys.stdout.flush()
    assert raw.getvalue().decode("utf-8").rstrip() == "Vučević Dončić"
