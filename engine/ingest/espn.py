"""Tip-off times from ESPN's public scoreboard -- the one thing Sleeper's schedule lacks.

    site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD

Probed 2026-10-21, from this machine and from GitHub Actions (rule 6). ``dates`` is the US
date of the games; each event's ``date`` is the tip-off in UTC, so a 9:30 pm ET game on the
22nd reads ``2026-10-23T01:30Z``. ESPN abbreviates six teams differently from the NBA (and
Sleeper); ``ESPN_TO_NBA`` translates them, and anything else unrecognised is left out of the
result so the caller reports a missing tip time instead of trusting a wrong team.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import httpx

from engine.ingest.http_retry import request_with_retry

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

ESPN_TO_NBA = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
NBA_TEAMS = frozenset({
    "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC",
    "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC",
    "SAS", "TOR", "UTA", "WAS",
})


def parse_tip_times(payload: dict, day: date) -> dict[tuple[str, str], datetime]:
    """``{(team, US date): tip-off in UTC}`` for both teams of every game on ``day``."""
    tips: dict[tuple[str, str], datetime] = {}
    for event in payload.get("events", []):
        tip = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
        for competitor in event["competitions"][0]["competitors"]:
            raw = competitor["team"]["abbreviation"]
            team = ESPN_TO_NBA.get(raw, raw)
            if team in NBA_TEAMS:
                tips[(team, day.isoformat())] = tip
    return tips


def fetch_tip_times(days: Iterable[date], timeout: float = 20.0) -> dict[tuple[str, str], datetime]:
    tips: dict[tuple[str, str], datetime] = {}
    with httpx.Client(timeout=timeout) as http:
        for day in sorted(set(days)):
            response = request_with_retry(
                lambda day=day: http.get(SCOREBOARD_URL, params={"dates": day.strftime("%Y%m%d")}))
            tips.update(parse_tip_times(response.json(), day))
    return tips
