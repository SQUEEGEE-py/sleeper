"""The current season's game logs, built from Sleeper's per-game stats instead of stats.nba.com.

stats.nba.com blocks cloud IPs (and this project's cloud sessions cannot reach it at all), so
until now the season being played reached the model only through a laptop refresh of
``snapshot/``. Sleeper's per-game stats answer from anywhere and were checked against
stats.nba.com on week 1 (2026-10-25, 258 player-games): 235 identical, 20 different only by
technicals/flagrants -- which Sleeper has and stats.nba.com does not -- and 3 by one steal. So
for the season being played Sleeper is the better source.

The frame has exactly the columns ``nba_stats.normalize_game_logs`` produces, so nothing
downstream changes, with two differences worth knowing:

* ``game_id`` is **Sleeper's** game id. ``distributions.team_games_played`` counts a team's games
  as distinct ``game_id``s within one season, so a season must come wholly from one source --
  this replaces the season's file, it is never appended to stats.nba.com rows;
* ``tech_foul`` and ``flagrant_foul`` are real here (``tf``/``ff``), not the zeros of
  ``GAME_LOG_MISSING_STATS``.

Only games with minutes on the clock are kept, as stats.nba.com's logs only list games played.
Pure conversion here; the fetch loop is in ``engine/jobs/refresh_current_season.py``.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from engine.ingest.sleeper import PlayerGame

# Sleeper stat key -> game-log column. Missing keys are zeros: Sleeper omits a stat it has none of.
SLEEPER_TO_LOG = {
    "pts": "pts", "reb": "reb", "ast": "ast", "stl": "stl", "blk": "blk", "to": "to",
    "tpm": "fg3m", "tf": "tech_foul", "ff": "flagrant_foul",
}
LOG_COLUMNS = ["nba_player_id", "player_name", "team", "game_id", "game_date", "minutes", "pts", "reb",
               "ast", "stl", "blk", "to", "fg3m", "season", "tech_foul", "flagrant_foul", "season_type"]
REGULAR_SEASON = "Regular Season"


def minutes_of(stats: dict) -> float:
    """Minutes played: ``sp`` is seconds and exact; ``min`` is the fallback."""
    if stats.get("sp") is not None:
        return float(stats["sp"]) / 60.0
    return float(stats.get("min") or 0.0)


def log_rows(nba_player_id: int, name: str, games: Iterable[PlayerGame], season: str) -> list[dict]:
    """One player's Sleeper games -> game-log rows. Games with no minutes are dropped."""
    rows = []
    for game in games:
        stats = game.stats or {}
        minutes = minutes_of(stats)
        if minutes <= 0 or not game.team or not game.game_id:
            continue
        row = {"nba_player_id": int(nba_player_id), "player_name": name, "team": game.team,
               "game_id": str(game.game_id), "game_date": pd.Timestamp(game.date), "minutes": minutes,
               "season": season, "season_type": REGULAR_SEASON}
        for key, column in SLEEPER_TO_LOG.items():
            row[column] = float(stats.get(key) or 0.0)
        rows.append(row)
    return rows


def logs_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=LOG_COLUMNS)
    if frame.empty:
        return frame
    return frame.sort_values(["game_date", "nba_player_id"]).reset_index(drop=True)
