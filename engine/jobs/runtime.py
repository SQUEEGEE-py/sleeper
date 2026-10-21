"""What every long-running or scheduled job needs to stay alive and leave a trace.

    utf8_output()          -- call first in main(); output can then go to a file or a pipe
    run_loop(...)          -- the poll loop: one failed pass is logged and the loop goes on

Both exist because of the draft-night incident (2026-10-18, 18:31): the live advisor stopped
mid-draft with no traceback anywhere. Only the Sleeper fetch was inside a ``try``; anything
raised by the build, the page write or a ``print`` killed the process. Two such failures are
known on Windows: ``os.replace`` refusing to overwrite a page the browser is reading, and the
cp1252 console encoding, which cannot print "Vučević" once output is redirected.
"""

from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

LOG_DIR = Path("data/logs")


def utf8_output() -> None:
    """Make stdout and stderr UTF-8, so redirected output and scheduled runs can print any
    player name. ``errors="replace"`` because a log line is never worth crashing a job over."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def log_exception(job: str, error: BaseException, log_dir: Path = LOG_DIR) -> Path:
    """Append the full traceback of ``error`` to ``<log_dir>/<job>.log`` and return the path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{job}.log"
    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    with path.open("a", encoding="utf-8") as log:
        log.write(f"--- {datetime.now().isoformat(timespec='seconds')}\n{trace}\n")
    return path


def run_loop(
    job: str,
    step: Callable[[], None],
    *,
    interval: float,
    should_stop: Callable[[], bool],
    on_error: Callable[[Exception, Path], None],
    sleep: Callable[[float], None] = time.sleep,
    log_dir: Path = LOG_DIR,
) -> None:
    """Call ``step`` every ``interval`` seconds until ``should_stop()`` is true.

    The whole pass is guarded, not only its network calls: an exception is logged with its
    traceback, handed to ``on_error`` (which shows it to the user), and the next pass runs as
    normal. Only ``KeyboardInterrupt`` and other non-``Exception`` exits end the loop.
    ``should_stop`` is checked after failed passes too, so a ``--once`` run still exits.
    """
    while True:
        try:
            step()
        except Exception as error:
            on_error(error, log_exception(job, error, log_dir))
        if should_stop():
            return
        sleep(interval)
