"""Current-season game logs from Sleeper's per-game stats. No network."""

from __future__ import annotations

import pandas as pd

from engine.ingest.sleeper import PlayerGame
from engine.ingest.sleeper_logs import LOG_COLUMNS, log_rows, logs_frame, minutes_of
from engine.jobs.refresh_current_season import compare


def game(date: str, stats: dict, team: str | None = "PHX", game_id: str | None = "g1") -> PlayerGame:
    return PlayerGame(week=3, date=date, stats=stats, team=team, game_id=game_id)


# Jalen Green, Thu 2026-11-05, as Sleeper returned it (abridged): no 'reb' key at all.
GREEN = {"pts": 6.0, "ast": 3.0, "stl": 2.0, "to": 2.0, "sp": 1440.0, "min": 24.0, "pf": 3.0, "tpa": 1.0}


def test_a_sleeper_game_becomes_a_game_log_row() -> None:
    [row] = log_rows(2309, "Jalen Green", [game("2026-11-05", GREEN)], "2026-27")
    assert row["minutes"] == 24.0 and row["pts"] == 6.0 and row["stl"] == 2.0
    assert row["reb"] == 0.0 and row["fg3m"] == 0.0  # a stat Sleeper leaves out is a zero
    assert row["team"] == "PHX" and row["game_id"] == "g1" and row["season_type"] == "Regular Season"
    assert row["game_date"] == pd.Timestamp("2026-11-05") and row["season"] == "2026-27"


def test_technicals_flagrants_and_threes_are_carried() -> None:
    [row] = log_rows(1, "x", [game("2026-11-05", {"sp": 600, "tpm": 3, "tf": 1, "ff": 1})], "2026-27")
    assert (row["fg3m"], row["tech_foul"], row["flagrant_foul"]) == (3.0, 1.0, 1.0)


def test_minutes_prefer_seconds_and_games_without_minutes_are_dropped() -> None:
    assert minutes_of({"sp": 1930, "min": 32}) == 1930 / 60
    assert minutes_of({"min": 32}) == 32.0
    rows = log_rows(1, "x", [game("2026-11-02", {"pts": 0}), game("2026-11-04", {"sp": 0}),
                             game("2026-11-05", {"sp": 60}, team=None), game("2026-11-06", {"sp": 60})],
                    "2026-27")
    assert [r["game_date"] for r in rows] == [pd.Timestamp("2026-11-06")]


def test_the_frame_has_exactly_the_stats_nba_columns() -> None:
    frame = logs_frame(log_rows(1, "x", [game("2026-11-06", GREEN), game("2026-11-02", GREEN)], "2026-27"))
    assert list(frame.columns) == LOG_COLUMNS and list(frame["game_date"].dt.day) == [2, 6]
    assert list(logs_frame([]).columns) == LOG_COLUMNS


def test_compare_passes_known_differences_and_stops_a_bad_rebuild() -> None:
    old = logs_frame(log_rows(1, "a", [game("2026-10-20", {"sp": 600, "pts": 20}),
                                       game("2026-10-22", {"sp": 600, "pts": 10})], "2026-27"))
    # A technical on one side, and a newer game the old file never had: fine.
    new = logs_frame(log_rows(1, "a", [game("2026-10-20", {"sp": 600, "pts": 20, "tf": 1}),
                                       game("2026-10-22", {"sp": 600, "pts": 10}),
                                       game("2026-11-01", {"sp": 600, "pts": 30})], "2026-27"))
    lines, passes = compare(old, new)
    assert passes and "2 player-games in both (1 score identically)" in lines[0]
    # Half the old games missing: stop.
    _, passes = compare(old, new[new["game_date"] != pd.Timestamp("2026-10-22")])
    assert not passes
    assert compare(pd.DataFrame(), new)[1]
