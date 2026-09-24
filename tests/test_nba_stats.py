"""Tests for engine/ingest/nba_stats.py (TASKS M2).

The schedule fixtures are a real slice of the 2026-27 ScheduleLeagueV2 payload, captured
2026-09-23 -- real column names, the real US date format, the real duplicated week row and a
real TBD-team game. If the endpoint's shape changes, these fail.

No test touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from engine.ingest.nba_stats import (
    GAME_LOG_MISSING_STATS,
    games_per_team_per_week,
    normalize_game_logs,
    normalize_schedule,
    normalize_weeks,
    previous_seasons,
    regular_season_games,
    schedule_completeness,
    season_string,
    verify_weeks_match_fantasy_weeks,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SEASON = "2026-27"
SEASON_START = "2026-10-20"


@pytest.fixture(scope="module")
def raw_schedule() -> pd.DataFrame:
    return pd.DataFrame(json.loads((FIXTURES / "schedule_sample.json").read_text()))


@pytest.fixture(scope="module")
def raw_weeks() -> pd.DataFrame:
    return pd.DataFrame(json.loads((FIXTURES / "weeks_sample.json").read_text()))


@pytest.fixture(scope="module")
def schedule(raw_schedule: pd.DataFrame) -> pd.DataFrame:
    return normalize_schedule(raw_schedule, SEASON)


@pytest.fixture(scope="module")
def weeks(raw_weeks: pd.DataFrame) -> pd.DataFrame:
    return normalize_weeks(raw_weeks, SEASON)


def _raw_game_log_row(**overrides) -> dict:
    """A PlayerGameLogs row with the endpoint's real column names, plus a couple of the
    60-odd columns we do not score, to prove they get dropped."""
    row = {
        "PLAYER_ID": 201939,
        "PLAYER_NAME": "Test Player",
        "TEAM_ABBREVIATION": "GSW",
        "GAME_ID": "0022600001",
        "GAME_DATE": "2026-10-20T00:00:00",
        "MIN": 34.0,
        "PTS": 30,
        "REB": 5,
        "AST": 6,
        "STL": 2,
        "BLK": 0,
        "TOV": 3,
        "FG3M": 7,
        "FG_PCT": 0.5,
        "PLUS_MINUS": 11,
    }
    row.update(overrides)
    return row


# --- season helpers ------------------------------------------------------------------------


def test_season_string() -> None:
    assert season_string(2026) == "2026-27"
    assert season_string(1999) == "1999-00"


def test_previous_seasons() -> None:
    assert previous_seasons("2026-27", 3) == ["2023-24", "2024-25", "2025-26"]


# --- game logs -----------------------------------------------------------------------------


def test_normalize_game_logs_renames_to_box_score_fields() -> None:
    frame = normalize_game_logs(pd.DataFrame([_raw_game_log_row()]), SEASON)
    row = frame.iloc[0]

    assert row["pts"] == 30.0
    assert row["to"] == 3.0  # TOV, the name scoring.BoxScore uses
    assert row["fg3m"] == 7.0
    assert row["nba_player_id"] == 201939
    assert row["season"] == SEASON
    assert "FG_PCT" not in frame.columns and "PLUS_MINUS" not in frame.columns


def test_normalize_game_logs_marks_unavailable_stats_as_zero() -> None:
    """PlayerGameLogs has no technical or flagrant fouls. They must be present and zero, so
    downstream code sees a column rather than a KeyError -- and GAME_LOG_MISSING_STATS is what
    lets the UI say the number is unmeasured rather than observed."""
    frame = normalize_game_logs(pd.DataFrame([_raw_game_log_row()]), SEASON)
    for stat in GAME_LOG_MISSING_STATS:
        assert stat in frame.columns
        assert frame.iloc[0][stat] == 0.0


def test_normalize_game_logs_rejects_a_changed_endpoint_shape() -> None:
    raw = pd.DataFrame([_raw_game_log_row()]).drop(columns=["REB"])
    with pytest.raises(ValueError, match="missing expected columns"):
        normalize_game_logs(raw, SEASON)


def test_normalize_game_logs_sorts_by_date() -> None:
    raw = pd.DataFrame(
        [
            _raw_game_log_row(GAME_DATE="2026-11-05T00:00:00", PTS=10),
            _raw_game_log_row(GAME_DATE="2026-10-20T00:00:00", PTS=20),
        ]
    )
    frame = normalize_game_logs(raw, SEASON)
    assert list(frame["pts"]) == [20.0, 10.0]


def test_game_logs_feed_the_scoring_engine() -> None:
    """The M1/M2 seam: a normalized log row has exactly the fields BoxScore wants."""
    from engine.model.scoring import BoxScore, score_box

    frame = normalize_game_logs(pd.DataFrame([_raw_game_log_row()]), SEASON)
    fields = list(BoxScore.model_fields)
    box = BoxScore(**frame.iloc[0][fields].to_dict())

    # 1.5*30 + 5 + 6 + 2 + 0 - 3 + 0.5*7 = 58.5, no double-double, no scoring bonus.
    assert score_box(box) == pytest.approx(58.5)


# --- schedule ------------------------------------------------------------------------------


def test_normalize_schedule_parses_us_dates(schedule: pd.DataFrame) -> None:
    """gameDate arrives as '10/20/2026 00:00:00'. Sorted as text, January 2027 would come
    before December 2026 and every week assignment downstream would be wrong."""
    assert schedule["game_date"].is_monotonic_increasing
    opener = schedule[schedule["game_id"] == "0022600001"].iloc[0]
    assert opener["game_date"] == pd.Timestamp("2026-10-20")
    assert opener["game_date"] == pd.Timestamp(SEASON_START)


def test_regular_season_games_excludes_preseason(schedule: pd.DataFrame) -> None:
    regular = regular_season_games(schedule)
    assert len(regular) < len(schedule)
    assert set(regular["game_id"].str[:3]) == {"002"}
    assert (regular["week_number"] >= 1).all()


def test_games_per_team_per_week_counts_home_and_away(schedule: pd.DataFrame) -> None:
    grid = games_per_team_per_week(schedule)

    # The fixture's opener is DET vs BOS in week 1; both sides must be credited.
    assert grid.loc["DET", 1] == 1
    assert grid.loc["BOS", 1] == 1
    assert (grid.to_numpy() >= 0).all()
    # Preseason games are week 0 and must not appear as a column.
    assert 0 not in grid.columns


def test_games_per_team_per_week_drops_tbd_teams(schedule: pd.DataFrame) -> None:
    """Unscheduled NBA Cup games have a null tricode. They must not become a 'nan' team row."""
    grid = games_per_team_per_week(schedule)
    assert not any(pd.isna(team) for team in grid.index)
    assert "nan" not in {str(team) for team in grid.index}


def test_schedule_completeness_flags_missing_games(schedule: pd.DataFrame) -> None:
    completeness = schedule_completeness(schedule)
    assert completeness.placeholder_games == 2  # the fixture's two TBD-team games
    assert not completeness.is_complete
    assert "SCHEDULE INCOMPLETE" in completeness.describe()


def test_schedule_completeness_reports_complete_when_it_is() -> None:
    raw = pd.DataFrame(
        [
            {
                "gameId": f"002260000{index}",
                "gameDate": "10/20/2026 00:00:00",
                "weekNumber": 1,
                "homeTeam_teamTricode": "DET",
                "awayTeam_teamTricode": "BOS",
            }
            for index in range(2)
        ]
    )
    completeness = schedule_completeness(normalize_schedule(raw, SEASON), expected_per_team=2)
    assert completeness.is_complete
    assert "complete" in completeness.describe()


# --- week grid -----------------------------------------------------------------------------


def test_normalize_weeks_collapses_the_duplicated_row(
    raw_weeks: pd.DataFrame, weeks: pd.DataFrame
) -> None:
    """The real payload lists week 8 twice, identically."""
    assert raw_weeks["weekNumber"].duplicated().sum() == 1
    assert weeks["week_number"].duplicated().sum() == 0
    assert len(weeks) == len(raw_weeks) - 1


def test_normalize_weeks_sorts(weeks: pd.DataFrame) -> None:
    """The source arrives out of order."""
    assert list(weeks["week_number"]) == list(range(1, len(weeks) + 1))


def test_normalize_weeks_keeps_conflicting_duplicates(raw_weeks: pd.DataFrame) -> None:
    """Identical rows collapse; rows that disagree about a week's dates must NOT, because that
    is a real conflict the verifier has to fail on."""
    conflicting = pd.concat(
        [raw_weeks, pd.DataFrame([{**raw_weeks.iloc[0].to_dict(), "endDate": "2026-10-26T00:00:00Z"}])],
        ignore_index=True,
    )
    weeks = normalize_weeks(conflicting, SEASON)
    problems = verify_weeks_match_fantasy_weeks(weeks, SEASON_START)
    assert any("ambiguous" in problem for problem in problems)


def test_week_grid_matches_the_leagues_fantasy_weeks(weeks: pd.DataFrame) -> None:
    """The load-bearing assumption of M3: the NBA's own week numbering IS this league's
    fantasy week numbering. Verified, not assumed."""
    assert verify_weeks_match_fantasy_weeks(weeks, SEASON_START) == []
    assert weeks.iloc[0]["start_date"] == pd.Timestamp(SEASON_START)
    assert len(weeks) == 25


def test_week_grid_covers_the_fantasy_playoffs(weeks: pd.DataFrame) -> None:
    """config puts the fantasy playoffs at week 22; the grid has to reach it."""
    config = json.loads(
        (Path(__file__).resolve().parent.parent / "config" / "league.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(weeks) >= config["playoffs"]["start_week"]


def test_verify_weeks_catches_a_wrong_season_start(weeks: pd.DataFrame) -> None:
    problems = verify_weeks_match_fantasy_weeks(weeks, "2026-10-21")
    assert any("season_start" in problem for problem in problems)


def test_verify_weeks_catches_a_non_monday_week(raw_weeks: pd.DataFrame) -> None:
    shifted = raw_weeks.copy()
    shifted.loc[shifted["weekNumber"] == 3, "startDate"] = "2026-11-03T00:00:00Z"
    problems = verify_weeks_match_fantasy_weeks(normalize_weeks(shifted, SEASON), SEASON_START)
    assert any("not a Monday" in problem for problem in problems)


def test_verify_weeks_catches_a_gap(raw_weeks: pd.DataFrame) -> None:
    without_week_five = raw_weeks[raw_weeks["weekNumber"] != 5]
    problems = verify_weeks_match_fantasy_weeks(
        normalize_weeks(without_week_five, SEASON), SEASON_START
    )
    assert any("no gaps" in problem for problem in problems)


def test_verify_weeks_handles_an_empty_grid() -> None:
    empty = pd.DataFrame(
        {"week_number": [], "week_name": [], "start_date": [], "end_date": [], "season": []}
    )
    assert verify_weeks_match_fantasy_weeks(empty, SEASON_START) == ["week grid is empty"]
