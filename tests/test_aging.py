"""Age adjustment: the measured curve, the flat tail, and the fade as this season arrives."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from engine.model.aging import (
    age_adjusted,
    age_factor,
    effective_factor,
    previous_season_share,
    season_reference_date,
)
from engine.model.distributions import EmpiricalScoreDistribution
from engine.model.player_model import PlayerModel, birth_dates_by_nba_id


def test_curve_is_relative_to_27_and_flat_past_34() -> None:
    assert age_factor(27) == 1.0
    assert age_factor(30) == pytest.approx(math.exp(-0.0176 * 3))  # 0.948
    assert age_factor(34) == pytest.approx(0.884, abs=0.001)
    assert age_factor(35) == age_factor(41) == age_factor(34)
    assert age_factor(22) == pytest.approx(1.092, abs=0.001)


def test_age_is_taken_on_1_february_of_the_season() -> None:
    assert season_reference_date("2026-27") == date(2027, 2, 1)
    # Jaylen Brown, born 1996-10-24: 30 on the reference date, whatever today is.
    assert effective_factor(date(1996, 10, 24), "2026-27", 1.0) == pytest.approx(age_factor(30))


def logs_for(seasons: list[str]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(seasons), freq="D")
    return pd.DataFrame({"game_date": dates, "season": seasons})


def test_adjustment_fades_as_this_seasons_games_arrive() -> None:
    assert previous_season_share(logs_for(["2025-26"] * 50), "2026-27") == 1.0
    assert previous_season_share(logs_for(["2026-27"] * 5), "2026-27") == 0.0
    # Half-life 25: the newest 25 games carry 0.5 of the geometric total, the 50 before them
    # 0.5 x (1 - 0.25) = 0.375, so the old season's share is 0.375 / 0.875 = 3/7.
    share = previous_season_share(logs_for(["2025-26"] * 50 + ["2026-27"] * 25), "2026-27")
    assert share == pytest.approx(3 / 7, abs=0.002)
    old = date(1984, 12, 30)  # LeBron: 42 on the reference date, held at the 34 factor
    assert effective_factor(old, "2026-27", 0.5) == pytest.approx(1 + (age_factor(34) - 1) * 0.5)
    assert effective_factor(old, "2026-27", 0.0) == 1.0


def test_scaling_moves_the_whole_distribution_not_just_the_mean() -> None:
    dist = EmpiricalScoreDistribution(np.array([20.0, 40.0, 60.0]), np.ones(3), _p_play=0.8)
    aged = age_adjusted(dist, 0.9)
    assert aged.mean() == pytest.approx(36.0) and aged.sd() == pytest.approx(dist.sd() * 0.9)
    assert aged.p_play == 0.8 and age_adjusted(dist, 1.0) is dist


def model_logs() -> pd.DataFrame:
    rows = []
    for pid in (1, 2):
        for i in range(40):
            rows.append({"nba_player_id": pid, "player_name": f"P{pid}", "team": "DEN", "game_id": f"{pid}-{i}",
                         "game_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i), "minutes": 32.0,
                         "fp": 40.0 + (i % 5), "season": "2025-26"})
    return pd.DataFrame(rows)


def test_player_model_applies_age_only_when_switched_on() -> None:
    logs = model_logs()
    plain = PlayerModel(logs, pd.DataFrame(), "2025-26")
    aged = PlayerModel(logs, pd.DataFrame(), "2025-26", season="2026-27",
                       birth_dates={1: date(1985, 1, 1)})  # 42 -> held at 34
    assert aged.distribution(1).mean() == pytest.approx(plain.distribution(1).mean() * age_factor(34))
    assert aged.age_factors[1] == pytest.approx(age_factor(34))
    # No birth date: unadjusted and reported, never a guessed age.
    assert aged.distribution(2).mean() == pytest.approx(plain.distribution(2).mean())
    assert aged.unaged == {2}


def test_birth_dates_skip_missing_and_unreadable() -> None:
    class P:
        def __init__(self, birth_date):
            self.birth_date = birth_date
    births = birth_dates_by_nba_id({1: "a", 2: "b", 3: "c", 4: "zz"},
                                   {"a": P("1996-10-24"), "b": P(None), "c": P("not a date")})
    assert births == {1: date(1996, 10, 24)}
