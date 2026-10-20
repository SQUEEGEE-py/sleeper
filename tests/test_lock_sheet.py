"""Tests for engine/jobs/lock_sheet.py. Fixture schedules and analytic distributions; no network."""

from __future__ import annotations

import re
from datetime import date

import pandas as pd
import pytest

from engine.jobs.lock_sheet import (
    GameSlot,
    build_sheet,
    game_thresholds,
    lineup_changes,
    recommend_lineup,
    render_html,
    render_terminal,
    team_week_games,
    week_for,
)
from engine.model.distributions import NormalScoreDistribution

SLOTS = ["PG#1", "SG#1", "G#1", "SF#1", "PF#1", "F#1", "C#1", "C#2", "UTIL#1"]


def schedule(rows: list[tuple[str, int, str, str]], game_type: str = "002") -> pd.DataFrame:
    """(date, week, home, away) -> a frame shaped like schedule_{season}.parquet."""
    return pd.DataFrame(
        [
            {"game_id": f"{game_type}26{i:05d}", "game_date": pd.Timestamp(day), "week_number": week,
             "home_team": home, "away_team": away, "game_type": game_type, "season": "2026-27"}
            for i, (day, week, home, away) in enumerate(rows)
        ]
    )


WEEK_2 = schedule(
    [
        ("2026-10-25", 1, "PHI", "DET"),  # Sunday of week 1
        ("2026-10-26", 2, "BOS", "PHI"),  # Monday: second night, across the week boundary
        ("2026-10-28", 2, "PHI", "NYK"),
        ("2026-10-31", 2, "PHI", "MIA"),
        ("2026-11-01", 2, "ATL", "PHI"),  # Sunday: second night again, and his last game
        ("2026-10-27", 2, "DEN", "LAL"),  # someone else's game
    ]
)


# --- schedule -------------------------------------------------------------------------------


def test_week_games_are_his_regular_season_games_in_order() -> None:
    games = team_week_games(WEEK_2, "PHI", 2)
    assert [g.game_date for g in games] == [
        date(2026, 10, 26), date(2026, 10, 28), date(2026, 10, 31), date(2026, 11, 1)
    ]
    assert [g.label for g in games] == ["@ BOS", "vs NYK", "vs MIA", "@ ATL"]


def test_back_to_back_is_detected_across_the_week_boundary() -> None:
    games = team_week_games(WEEK_2, "PHI", 2)
    assert [g.b2b_second for g in games] == [True, False, False, True]


def test_preseason_games_are_ignored() -> None:
    frame = pd.concat([WEEK_2, schedule([("2026-10-27", 2, "PHI", "TOR")], game_type="001")])
    games = team_week_games(frame, "PHI", 2)
    assert date(2026, 10, 27) not in [g.game_date for g in games]
    # A preseason game the night before must not make Wednesday a back-to-back either.
    assert not next(g for g in games if g.game_date == date(2026, 10, 28)).b2b_second


def test_week_for_today() -> None:
    weeks = pd.DataFrame(
        {"week_number": [1, 2], "start_date": pd.to_datetime(["2026-10-20", "2026-10-26"]),
         "end_date": pd.to_datetime(["2026-10-25", "2026-11-01"])}
    )
    assert week_for(weeks, date(2026, 10, 19)) == 1  # before the season
    assert week_for(weeks, date(2026, 10, 25)) == 1
    assert week_for(weeks, date(2026, 10, 26)) == 2
    assert week_for(weeks, date(2027, 4, 1)) == 2  # after the last week


# --- thresholds -----------------------------------------------------------------------------------


def test_thresholds_match_the_closed_form() -> None:
    """normal(45, 10): rolling on is worth 49.0 with two games left and 45.0 with one."""
    thresholds = game_thresholds(NormalScoreDistribution(45.0, 10.0), 3)
    assert thresholds[0] == pytest.approx(48.989, abs=0.01)
    assert thresholds[1] == pytest.approx(45.0, abs=0.01)
    assert thresholds[2] == float("-inf")


def test_one_game_week_has_no_decision_and_an_empty_week_has_nothing() -> None:
    assert game_thresholds(NormalScoreDistribution(45.0, 10.0), 1) == [float("-inf")]
    assert game_thresholds(NormalScoreDistribution(45.0, 10.0), 0) == []


def test_thresholds_fall_as_the_week_runs_out() -> None:
    thresholds = game_thresholds(NormalScoreDistribution(40.0, 15.0, 0.85), 4)
    assert thresholds[0] > thresholds[1] > thresholds[2]


# --- the sheet --------------------------------------------------------------------------------------


def slot(day: str, b2b: bool = False) -> GameSlot:
    return GameSlot(date.fromisoformat(day), "BOS", True, b2b)


def sheet(name: str, positions: tuple[str, ...], games: list[GameSlot], mu: float = 45.0,
          in_lineup: bool = True, injury: str = "", born: date | None = None):
    return build_sheet(NormalScoreDistribution(mu, 12.0, 0.9), name.lower(), name, "PHI",
                       positions, injury, games, in_lineup, born)


def test_last_game_on_a_back_to_back_is_the_trap() -> None:
    trapped = sheet("A", ("SF",), [slot("2026-10-24"), slot("2026-10-25", b2b=True)])
    assert trapped.last_game_b2b
    assert any("AUTO-LOCK TRAP" in w for w in trapped.warnings)


def test_an_earlier_back_to_back_is_not_the_trap() -> None:
    fine = sheet("B", ("SF",), [slot("2026-10-24"), slot("2026-10-25", b2b=True), slot("2026-10-27")])
    assert not fine.last_game_b2b
    assert not any("AUTO-LOCK TRAP" in w for w in fine.warnings)


def test_no_games_and_injury_status_are_surfaced() -> None:
    idle = sheet("C", ("C",), [], injury="Out")
    assert idle.week_value == 0.0
    assert "no games this week" in idle.warnings and "Sleeper status: Out" in idle.warnings


def test_lineup_starts_the_best_week_values_and_reports_changes() -> None:
    games3 = [slot("2026-10-20"), slot("2026-10-22"), slot("2026-10-24")]
    sheets = [
        sheet("PG one", ("PG",), games3, mu=50.0, in_lineup=False),  # should start
        sheet("Idle guard", ("PG",), [], mu=60.0, in_lineup=True),  # no games: bench him
        *[sheet(f"Wing {i}", ("SF", "PF", "SG"), games3, mu=40.0 + i) for i in range(6)],
        *[sheet(f"Big {i}", ("C",), games3, mu=40.0 + i) for i in range(2)],
    ]
    recommend_lineup(sheets, SLOTS)
    by_name = {s.name: s for s in sheets}
    assert by_name["PG one"].recommended_slot == "PG"
    assert by_name["Idle guard"].recommended_slot is None
    start, sit = lineup_changes(sheets)
    assert start == ["PG one"] and sit == ["Idle guard"]


def test_outputs_carry_thresholds_warnings_and_stay_offline() -> None:
    sheets = recommend_lineup(
        [sheet("Trap Guy", ("SF",), [slot("2026-10-24"), slot("2026-10-25", b2b=True)])], SLOTS
    )
    page = render_html(sheets, 1, "Tue Oct 20 - Sun Oct 25", ["X: unmapped"], "now")
    text = render_terminal(sheets, 1, ["X: unmapped"])
    for output in (page, text):
        assert "AUTO-LOCK TRAP" in output and "GAP: X: unmapped" in output
    assert "final: auto" in text and "lock ≥" in text
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", page)


# --- back-to-back pricing -----------------------------------------------------------------------

OLD = date(1988, 1, 1)  # 38 in October 2026
YOUNG = date(2004, 1, 1)  # 22
MIDDLE_B2B = [slot("2026-10-21"), slot("2026-10-23"), slot("2026-10-24", b2b=True), slot("2026-10-26")]
NO_B2B = [slot("2026-10-21"), slot("2026-10-23"), slot("2026-10-25"), slot("2026-10-27")]


def test_a_34_plus_players_b2b_night_lowers_his_continuation_value() -> None:
    """The point of step 2. With game 3 a B2B second night, rolling on from games 1 and 2 is
    worth less, so he should lock earlier; the threshold after game 3 (only game 4 left, an
    ordinary night) is untouched. And an old player is charged more than a young one."""
    old = sheet("Old", ("SF",), MIDDLE_B2B, born=OLD)
    young = sheet("Young", ("SF",), MIDDLE_B2B, born=YOUNG)
    plain = sheet("Plain", ("SF",), NO_B2B, born=OLD)
    assert old.thresholds[0] < young.thresholds[0] < plain.thresholds[0]
    assert old.thresholds[1] < young.thresholds[1] < plain.thresholds[1]
    assert old.thresholds[2] == pytest.approx(plain.thresholds[2])
    assert old.week_value < young.week_value < plain.week_value
    assert old.game_p_play == pytest.approx([0.9, 0.9, 0.9 * 0.879, 0.9])


def test_a_week_without_a_back_to_back_is_unchanged_by_age() -> None:
    plain = sheet("Plain", ("SF",), NO_B2B, born=OLD)
    assert plain.thresholds == game_thresholds(NormalScoreDistribution(45.0, 12.0, 0.9), 4)


def test_b2b_is_priced_at_his_age_on_the_game_date() -> None:
    """Turns 34 on the day of the back-to-back: that game takes the 34+ ratio."""
    turning = sheet("Birthday", ("SF",), MIDDLE_B2B, born=date(1992, 10, 24))
    assert turning.age == 33
    assert turning.game_p_play[2] == pytest.approx(0.9 * 0.879)


def test_unknown_birth_date_leaves_b2b_unpriced_and_says_so() -> None:
    unknown = sheet("Unknown", ("SF",), MIDDLE_B2B)
    assert unknown.game_p_play == [0.9] * 4
    assert any("NOT priced" in w for w in unknown.warnings)


def test_trap_warning_quotes_the_priced_chance_he_plays() -> None:
    trapped = sheet("Old", ("SF",), [slot("2026-10-24"), slot("2026-10-25", b2b=True)], born=OLD)
    assert any("plays only 79%" in w for w in trapped.warnings)  # 0.9 x 0.879
    assert "B2B 2nd night, plays 79%" in render_terminal(recommend_lineup([trapped], SLOTS), 1, [])
