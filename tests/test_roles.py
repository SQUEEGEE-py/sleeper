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
