"""Probe every endpoint the cloud nightly job would use, from wherever this runs (rule 6).

    python -m engine.jobs.probe_endpoints          # locally, or in GitHub Actions

Standard library only, so the Actions workflow needs no dependency install. Prints one line
per endpoint (HTTP status, size, time, and whether the body has the shape we build against)
and, in Actions, the same table as the job summary. Exits 1 if any endpoint marked required
fails; stats.nba.com and cdn.nba.com are informational -- the plan does not depend on them.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

CONFIG_PATH = Path("config/league.json")
TIMEOUT_SECONDS = 25
# A known completed game night, so box-score probes have data: opening night.
PROBE_DATE = date(2026, 10, 20)
PROBE_PLAYER = "1711"  # Jaylen Brown, played 2026-10-20
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}


@dataclass
class Probe:
    name: str
    url: str
    required: bool
    check: Callable[[Any], str]  # returns "" when the shape is right, else what is wrong


def _sleeper_stats_per_game(data: Any) -> str:
    games = [g for week in data.values() for g in (week or [])] if isinstance(data, dict) else []
    dated = [g for g in games if g.get("date") == PROBE_DATE.isoformat()]
    return "" if dated and "pts" in dated[0].get("stats", {}) else "no dated box line for the probe date"


def _espn_tip_times(data: Any) -> str:
    events = data.get("events", []) if isinstance(data, dict) else []
    return "" if events and all("date" in e for e in events) else "no events with start times"


def probes(league_id: str, season_year: str) -> list[Probe]:
    yyyymmdd = PROBE_DATE.strftime("%Y%m%d")
    tomorrow = (date.today() + timedelta(days=1)).strftime("%Y%m%d")
    return [
        Probe("sleeper league", f"https://api.sleeper.app/v1/league/{league_id}", True,
              lambda d: "" if "settings" in d and "scoring_settings" in d else "missing settings"),
        Probe("sleeper rosters", f"https://api.sleeper.app/v1/league/{league_id}/rosters", True,
              lambda d: "" if isinstance(d, list) and len(d) == 8 else "not 8 rosters"),
        Probe("sleeper matchups wk1", f"https://api.sleeper.app/v1/league/{league_id}/matchups/1", True,
              lambda d: "" if isinstance(d, list) and d and "players_points" in d[0] else "no players_points"),
        Probe("sleeper transactions wk1", f"https://api.sleeper.app/v1/league/{league_id}/transactions/1", True,
              lambda d: "" if isinstance(d, list) else "not a list"),
        Probe("sleeper players", "https://api.sleeper.app/v1/players/nba", True,
              lambda d: "" if isinstance(d, dict) and len(d) > 1000 else "too few players"),
        Probe("sleeper stats per game",
              f"https://api.sleeper.com/stats/nba/player/{PROBE_PLAYER}?season_type=regular"
              f"&season={season_year}&grouping=week", True, _sleeper_stats_per_game),
        Probe("sleeper stats week bulk",
              f"https://api.sleeper.com/stats/nba/{season_year}?season_type=regular&week=1", True,
              lambda d: "" if isinstance(d, list) and len(d) > 100 else "too few rows"),
        Probe("sleeper schedule", f"https://api.sleeper.com/schedule/nba/regular/{season_year}", True,
              lambda d: "" if isinstance(d, list) and d and {"date", "status", "home", "away"} <= set(d[0])
              else "missing date/status/teams"),
        Probe("espn scoreboard (past)", "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
              f"scoreboard?dates={yyyymmdd}", True, _espn_tip_times),
        Probe("espn scoreboard (tomorrow)", "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
              f"scoreboard?dates={tomorrow}", True,
              lambda d: "" if isinstance(d, dict) and "events" in d else "no events key"),
        Probe("cdn.nba.com scoreboard",
              "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json", False,
              lambda d: "" if isinstance(d, dict) and "scoreboard" in d else "no scoreboard"),
        Probe("stats.nba.com scoreboard",
              f"https://stats.nba.com/stats/scoreboardv2?GameDate={PROBE_DATE.isoformat()}"
              "&LeagueID=00&DayOffset=0", False,
              lambda d: "" if isinstance(d, dict) and "resultSets" in d else "no resultSets"),
    ]


def run(probe: Probe) -> tuple[bool, str]:
    headers = BROWSER_HEADERS if "nba.com" in probe.url else {"User-Agent": "sleeper-probe/1.0"}
    start = time.monotonic()
    try:
        with urllib.request.urlopen(urllib.request.Request(probe.url, headers=headers),
                                    timeout=TIMEOUT_SECONDS) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        return False, f"HTTP {error.code} after {time.monotonic() - start:.1f}s"
    except Exception as error:  # timeouts, resets, DNS: all are an answer worth recording
        return False, f"{type(error).__name__}: {error} after {time.monotonic() - start:.1f}s"
    elapsed = time.monotonic() - start
    try:
        problem = probe.check(json.loads(body))
    except Exception as error:
        problem = f"unreadable body ({type(error).__name__})"
    detail = f"HTTP {status}, {len(body) / 1024:.0f} KB, {elapsed:.1f}s"
    return not problem, detail + (f" -- {problem}" if problem else "")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    where = "GitHub Actions" if os.environ.get("GITHUB_ACTIONS") else "this machine"
    rows, required_failed = [], 0
    print(f"Probing from {where}:")
    for probe in probes(config["league_id"], config["season"].split("-")[0]):
        ok, detail = run(probe)
        required_failed += probe.required and not ok
        mark = "OK  " if ok else ("FAIL" if probe.required else "no  ")
        print(f"  {mark} {probe.name:28} {detail}")
        rows.append(f"| {'✅' if ok else ('❌' if probe.required else '➖')} | {probe.name} | "
                    f"{'required' if probe.required else 'info'} | {detail} |")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as out:
            out.write(f"### Endpoint probe from {where}\n\n| | endpoint | | result |\n|---|---|---|---|\n")
            out.write("\n".join(rows) + "\n")
    print(f"\n{required_failed} required endpoint(s) failed.")
    return 1 if required_failed else 0


if __name__ == "__main__":
    sys.exit(main())
