"""Role watch: minutes this season vs his latest earlier season. No network."""

from __future__ import annotations

from engine.model.roles import MIN_GAMES, MINUTES_FLAG, role_check


def test_nothing_is_judged_before_five_games() -> None:
    check = role_check("Jaylen Brown", "PHI", "BOS", [40.0] * (MIN_GAMES - 1), 34.4, "2025-26")
    assert not check.judged and not check.flagged and check.new_team
    assert "4 game(s) so far" in check.describe() and "was BOS" in check.describe()


def test_a_big_enough_move_is_flagged_either_way() -> None:
    up = role_check("A", "UTA", "UTA", [33.1 + MINUTES_FLAG] * MIN_GAMES, 33.1, "2025-26")
    down = role_check("B", "UTA", "UTA", [30.0] * MIN_GAMES, 30.0 + MINUTES_FLAG + 0.5, "2025-26")
    assert up.flagged and "UP 4.0 min" in up.describe()
    assert down.flagged and "DOWN 4.5 min" in down.describe()


def test_a_small_move_is_steady() -> None:
    check = role_check("C", "HOU", "HOU", [34.0, 35.0, 36.0, 34.0, 35.0], 36.4, "2025-26")
    assert check.judged and not check.flagged and "steady" in check.describe()


def test_a_team_change_alone_is_not_a_flag() -> None:
    check = role_check("Kawhi Leonard", "TOR", "LAC", [32.0] * 6, 32.1, "2025-26")
    assert check.new_team and not check.flagged


def test_comparison_names_the_season_it_uses() -> None:
    check = role_check("Kyrie Irving", "DAL", "DAL", [30.0] * 6, 36.1, "2024-25")
    assert "36.1 in 2024-25" in check.describe() and check.flagged


def test_no_earlier_season_is_said_not_guessed() -> None:
    check = role_check("Caleb Wilson", "CHI", None, [25.0] * 6, None)
    assert not check.judged and not check.new_team and "no earlier season" in check.describe()


# --- role factor (2026-11-15) --------------------------------------------------------------

import pandas as pd
import pytest

from engine.model.player_model import PlayerModel
from engine.model.roles import ROLE_ELASTICITY, ROLE_MIN_LAST_GAMES, ROLE_MIN_LAST_MINUTES, role_factor


def test_role_factor_is_the_backtested_power_faded_by_the_earlier_share() -> None:
    assert role_factor(10.0, 12, 30.0, 1.0) == pytest.approx((10 / 30) ** ROLE_ELASTICITY)
    assert role_factor(36.0, 12, 30.0, 0.5) == pytest.approx((36 / 30) ** (ROLE_ELASTICITY * 0.5))
    assert role_factor(10.0, 12, 30.0, 0.0) == 1.0  # all of his weight is this season: nothing to add


def test_role_factor_is_not_applied_outside_what_was_backtested() -> None:
    assert role_factor(10.0, MIN_GAMES - 1, 30.0, 1.0) == 1.0
    assert role_factor(30.0, 12, ROLE_MIN_LAST_MINUTES - 0.1, 1.0) == 1.0
    assert role_factor(None, 0, 30.0, 1.0) == role_factor(20.0, 12, None, 1.0) == 1.0


def role_logs(now_minutes: float, now_games: int, last_games: int = 40, last_minutes: float = 32.0) -> pd.DataFrame:
    rows = []
    for season, games, minutes, start in (("2024-25", 30, 30.0, "2025-01-01"),  # older, ignored
                                          ("2025-26", last_games, last_minutes, "2026-01-01"),
                                          ("2026-27", now_games, now_minutes, "2026-10-21")):
        for i in range(games):
            rows.append({"nba_player_id": 1, "player_name": "P1", "team": "ORL", "game_id": f"{season}-{i}",
                         "game_date": pd.Timestamp(start) + pd.Timedelta(days=i), "minutes": minutes,
                         "fp": 30.0 + (i % 5), "season": season})
    return pd.DataFrame(rows)


def test_player_model_applies_the_role_factor_only_when_switched_on() -> None:
    logs = role_logs(now_minutes=16.0, now_games=10)
    plain = PlayerModel(logs, pd.DataFrame(), "2025-26")
    on = PlayerModel(logs, pd.DataFrame(), "2025-26", season="2026-27")
    on_mean = on.distribution(1).mean()
    assert 0.5 < on.role_factors[1] < 1.0  # halved minutes, against the most recent earlier season (32)
    assert on_mean == pytest.approx(plain.distribution(1).mean() * on.role_factors[1])
    assert plain.role_factors == {}


def test_a_thin_earlier_season_is_skipped_for_the_one_before() -> None:
    # 9 games of 2025-26 at his current 16 minutes would say "no change"; the comparison is
    # with 2024-25's 30 minutes instead, so he is cut.
    logs = role_logs(16.0, 10, last_games=ROLE_MIN_LAST_GAMES - 1, last_minutes=16.0)
    model = PlayerModel(logs, pd.DataFrame(), "2025-26", season="2026-27")
    model.distribution(1)
    assert model.role_factors[1] < 0.9
