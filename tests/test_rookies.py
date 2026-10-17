"""Tests for engine/model/rookies.py and the board's candidate filter. Synthetic logs, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.jobs.build_board import MIN_RECENT_GAMES, candidate_players
from engine.model.rookies import (
    fit_minutes_projection,
    player_seasons,
    preseason_minutes,
    rookie_cohort_distribution,
    rookie_discount,
    rookie_seasons,
)

SEASON_STARTS = {"2023-24": "2023-10-24", "2024-25": "2024-10-22", "2025-26": "2025-10-21"}


def player_games(
    player_id: int,
    season: str,
    games: int,
    minutes: float,
    fp: float | list[float],
    team: str = "DEN",
    name: str | None = None,
) -> list[dict]:
    """``games`` consecutive team games for one player, one every two days."""
    start = pd.Timestamp(SEASON_STARTS.get(season, "2026-10-03"))
    values = fp if isinstance(fp, list) else [fp] * games
    return [
        {
            "nba_player_id": player_id,
            "player_name": name or f"Player {player_id}",
            "team": team,
            "game_id": f"{season}-{team}-{index}",
            "game_date": start + pd.Timedelta(days=2 * index),
            "minutes": minutes,
            "season": season,
            "fp": values[index],
        }
        for index in range(games)
    ]


def frame(*groups: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([row for group in groups for row in group])


# --- minutes projection -----------------------------------------------------------------


def test_minutes_projection_recovers_a_known_line() -> None:
    regular, preseason = [], []
    for player_id in range(1, 21):
        pre = 10.0 + player_id
        regular += player_games(player_id, "2025-26", 30, 0.9 * pre + 7.0, 20.0)
        preseason += player_games(player_id, "2026-27", 3, pre, 20.0)
    projection = fit_minutes_projection(frame(regular), frame(preseason))
    assert projection.slope == pytest.approx(0.9)
    assert projection.intercept == pytest.approx(7.0)
    assert projection.residual_sd == pytest.approx(0.0, abs=1e-9)
    assert projection.project(20.0) == pytest.approx(25.0)


def test_minutes_projection_refuses_a_tiny_sample() -> None:
    regular = frame(*(player_games(i, "2025-26", 30, 30.0, 20.0) for i in range(1, 4)))
    preseason = frame(*(player_games(i, "2026-27", 3, 20.0, 20.0) for i in range(1, 4)))
    with pytest.raises(ValueError, match="too few"):
        fit_minutes_projection(regular, preseason)


def test_minutes_projection_ignores_one_game_preseason_cameos() -> None:
    regular, preseason = [], []
    for player_id in range(1, 21):
        regular += player_games(player_id, "2025-26", 30, 10.0 + player_id, 20.0)
        preseason += player_games(player_id, "2026-27", 3, 10.0 + player_id, 20.0)
    # One wild outlier who only played a single preseason game must not bend the fit.
    regular += player_games(99, "2025-26", 30, 40.0, 20.0)
    preseason += player_games(99, "2026-27", 1, 2.0, 20.0)
    projection = fit_minutes_projection(frame(regular), frame(preseason))
    assert projection.slope == pytest.approx(1.0)
    assert projection.players == 20


# --- who counts as a rookie ---------------------------------------------------------------


def test_rookie_seasons_skips_players_the_window_starts_on() -> None:
    logs = frame(
        player_games(1, "2023-24", 30, 30.0, 30.0),  # first seen in the earliest season
        player_games(1, "2024-25", 30, 30.0, 30.0),
        player_games(2, "2024-25", 25, 20.0, 15.0),  # a 2024-25 rookie
        player_games(2, "2025-26", 25, 25.0, 25.0),
    )
    rookies = rookie_seasons(logs)
    assert set(rookies["nba_player_id"]) == {2}
    assert set(rookies["season"]) == {"2024-25"}  # only his rookie season, not the next


def test_rookie_seasons_needs_two_seasons() -> None:
    with pytest.raises(ValueError):
        rookie_seasons(frame(player_games(1, "2025-26", 10, 30.0, 30.0)))


def test_rookie_discount_measures_the_ratio_at_equal_minutes() -> None:
    logs = frame(
        player_games(1, "2023-24", 40, 30.0, 40.0),
        player_games(2, "2023-24", 40, 20.0, 25.0),
        player_games(1, "2024-25", 40, 30.0, 40.0),
        player_games(3, "2024-25", 40, 30.0, 32.0),  # rookie: 0.8 of the 30-minute veteran
        player_games(4, "2025-26", 40, 20.0, 20.0),  # rookie: 0.8 of the 20-minute veteran
    )
    assert rookie_discount(logs, rookie_seasons(logs)) == pytest.approx(0.8)


# --- the cohort -------------------------------------------------------------------------


def cohort_logs() -> pd.DataFrame:
    """Twelve player-seasons at each of 20, 26 and 32 minutes, scoring 1 fp per minute."""
    groups = []
    player_id = 0
    for minutes in (20.0, 26.0, 32.0):
        for _ in range(12):
            player_id += 1
            groups.append(
                player_games(
                    player_id, "2025-26", 30, minutes,
                    [minutes - 5.0, minutes + 5.0] * 15, team=f"T{player_id}",
                )
            )
    return frame(*groups)


def test_cohort_rises_with_projected_minutes() -> None:
    logs = cohort_logs()
    seasons = player_seasons(logs)
    means = [
        rookie_cohort_distribution(logs, seasons, minutes, 1.0).mean()
        for minutes in (20.0, 26.0, 32.0)
    ]
    assert means == pytest.approx([20.0, 26.0, 32.0])


def test_cohort_scales_mean_and_spread_by_the_discount() -> None:
    logs = cohort_logs()
    seasons = player_seasons(logs)
    full = rookie_cohort_distribution(logs, seasons, 26.0, 1.0)
    scaled = rookie_cohort_distribution(logs, seasons, 26.0, 0.9)
    assert scaled.mean() == pytest.approx(0.9 * full.mean())
    assert scaled.sd() == pytest.approx(0.9 * full.sd())


def test_cohort_keeps_game_to_game_variance() -> None:
    """All of a cohort player's games are used, not only those inside a minutes window."""
    logs = cohort_logs()
    distribution = rookie_cohort_distribution(logs, player_seasons(logs), 26.0, 1.0)
    assert distribution.sd() == pytest.approx(5.0)


def test_cohort_widens_until_it_has_enough_players() -> None:
    logs = cohort_logs()
    # 38 minutes is six away from the nearest cohort; it must widen to the 32-minute group
    # rather than fail or return an empty distribution.
    distribution = rookie_cohort_distribution(logs, player_seasons(logs), 38.0, 1.0)
    assert distribution.mean() == pytest.approx(32.0)


def test_cohort_weights_each_player_season_equally() -> None:
    long_season = player_games(1, "2025-26", 80, 26.0, 40.0, team="A")
    short_seasons = [
        player_games(player_id, "2025-26", 20, 26.0, 20.0, team=f"T{player_id}")
        for player_id in range(2, 9)
    ]
    logs = frame(long_season, *short_seasons)
    distribution = rookie_cohort_distribution(logs, player_seasons(logs), 26.0, 1.0)
    # Eight player-seasons, one at 40 and seven at 20: an equal-weight mean of 22.5, not the
    # game-weighted 27.3 that one 80-game season would force.
    assert distribution.mean() == pytest.approx(22.5)


def test_rookie_projection_is_marked_unreliable() -> None:
    logs = cohort_logs()
    distribution = rookie_cohort_distribution(logs, player_seasons(logs), 26.0, 1.0)
    assert distribution.history_reliable is False
    assert distribution.games_observed == 0
    assert 0.0 < distribution.p_play <= 1.0


def test_preseason_minutes_summarises_per_player() -> None:
    logs = frame(
        player_games(1, "2026-27", 3, 24.0, 20.0, team="CHI", name="A Rookie"),
    )
    summary = preseason_minutes(logs)
    row = summary.iloc[0]
    assert (row["player_name"], row["team"], row["games"]) == ("A Rookie", "CHI", 3)
    assert row["minutes"] == pytest.approx(24.0)


# --- the board's candidate filter ---------------------------------------------------------


def test_candidates_include_a_star_who_missed_last_season() -> None:
    """The bug this guards: Haliburton, Irving, Lillard and VanVleet were silently dropped."""
    logs = frame(
        player_games(1, "2024-25", 70, 34.0, 45.0, name="Injured Star"),
        player_games(2, "2025-26", 70, 30.0, 35.0, name="Healthy Starter"),
        player_games(3, "2025-26", MIN_RECENT_GAMES - 1, 10.0, 8.0, name="Cameo"),
    )
    candidates = candidate_players(logs, "2026-27").set_index("player_name")
    assert set(candidates.index) == {"Injured Star", "Healthy Starter"}
    assert bool(candidates.loc["Injured Star", "missed_last_season"]) is True
    assert bool(candidates.loc["Healthy Starter", "missed_last_season"]) is False
