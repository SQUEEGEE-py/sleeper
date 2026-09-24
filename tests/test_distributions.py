"""Tests for engine/model/distributions.py (TASKS M3).

Synthetic game logs throughout, shaped like the parquet ``ingest_history`` writes. No network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.model.distributions import (
    DEFAULT_HALF_LIFE_GAMES,
    MIN_GAMES_FOR_RELIABLE_HISTORY,
    EmpiricalScoreDistribution,
    NormalScoreDistribution,
    availability,
    build_distribution,
    recency_weights,
    role_prior,
    score_game_logs,
    shrink_toward_prior,
    team_games_played,
)


def make_logs(
    rows: list[dict], season: str = "2025-26", start: str = "2025-10-21"
) -> pd.DataFrame:
    """Build a game-log frame with the columns the real parquet has."""
    base = pd.Timestamp(start)
    records = []
    for index, row in enumerate(rows):
        record = {
            "nba_player_id": 1,
            "player_name": "Test Player",
            "team": "DEN",
            "game_id": f"002260{index:04d}",
            "game_date": base + pd.Timedelta(days=index * 2),
            "minutes": 32.0,
            "season": season,
            "pts": 0.0,
            "reb": 0.0,
            "ast": 0.0,
            "stl": 0.0,
            "blk": 0.0,
            "to": 0.0,
            "fg3m": 0.0,
            "tech_foul": 0.0,
            "flagrant_foul": 0.0,
        }
        record.update(row)
        records.append(record)
    return pd.DataFrame(records)


# --- NormalScoreDistribution -----------------------------------------------------------------


def test_normal_expected_max_above_and_below_the_mean() -> None:
    normal = NormalScoreDistribution(45.0, 10.0)
    # At the mean, E[max(X, mu)] = mu + sigma/sqrt(2*pi).
    assert normal.expected_max(45.0) == pytest.approx(45.0 + 10.0 * 0.3989422804, abs=1e-6)
    # Far below, the threshold never binds; far above, it always does.
    assert normal.expected_max(-1e6) == pytest.approx(45.0)
    assert normal.expected_max(1e6) == pytest.approx(1e6)


def test_normal_with_zero_spread_is_a_constant() -> None:
    flat = NormalScoreDistribution(45.0, 0.0)
    assert flat.expected_max(40.0) == pytest.approx(45.0)
    assert flat.expected_max(50.0) == pytest.approx(50.0)


def test_normal_expected_max_matches_simulation() -> None:
    rng = np.random.default_rng(7)
    sample = rng.normal(45.0, 10.0, 400_000)
    normal = NormalScoreDistribution(45.0, 10.0)
    for threshold in (30.0, 45.0, 60.0):
        assert normal.expected_max(threshold) == pytest.approx(
            float(np.maximum(sample, threshold).mean()), abs=0.1
        )


# --- EmpiricalScoreDistribution ---------------------------------------------------------------


def test_empirical_moments_respect_weights() -> None:
    distribution = EmpiricalScoreDistribution(
        values=np.array([10.0, 20.0]), weights=np.array([3.0, 1.0])
    )
    assert distribution.mean() == pytest.approx(12.5)
    assert distribution.expected_max(15.0) == pytest.approx(0.75 * 15.0 + 0.25 * 20.0)


def test_empirical_weights_need_not_be_normalized() -> None:
    values = np.array([10.0, 20.0, 30.0])
    a = EmpiricalScoreDistribution(values=values, weights=np.array([1.0, 1.0, 1.0]))
    b = EmpiricalScoreDistribution(values=values, weights=np.array([5.0, 5.0, 5.0]))
    assert a.mean() == pytest.approx(b.mean())
    assert a.expected_max(15.0) == pytest.approx(b.expected_max(15.0))


def test_empirical_keeps_the_right_tail() -> None:
    """The reason the conditional distribution is empirical rather than normal: the 40+/50+
    bonuses live in the tail, and a fitted normal would smooth it away."""
    values = np.concatenate([np.full(95, 30.0), np.full(5, 110.0)])
    distribution = EmpiricalScoreDistribution(values=values, weights=np.ones(100))
    assert distribution.quantile(0.99) == pytest.approx(110.0)
    assert distribution.expected_max(100.0) > 100.0


def test_effective_sample_size() -> None:
    equal = EmpiricalScoreDistribution(values=np.ones(100), weights=np.ones(100))
    assert equal.effective_sample_size == pytest.approx(100.0)

    lopsided = EmpiricalScoreDistribution(
        values=np.ones(100), weights=np.concatenate([[1000.0], np.ones(99)])
    )
    assert lopsided.effective_sample_size < 5.0


def test_empirical_rejects_bad_samples() -> None:
    with pytest.raises(ValueError, match="same length"):
        EmpiricalScoreDistribution(values=np.array([1.0]), weights=np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="empty"):
        EmpiricalScoreDistribution(values=np.array([]), weights=np.array([]))
    with pytest.raises(ValueError, match="non-finite"):
        EmpiricalScoreDistribution(values=np.array([np.nan]), weights=np.array([1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        EmpiricalScoreDistribution(values=np.array([1.0]), weights=np.array([-1.0]))
    with pytest.raises(ValueError, match="sum to zero"):
        EmpiricalScoreDistribution(values=np.array([1.0]), weights=np.array([0.0]))


# --- recency weighting --------------------------------------------------------------------------


def test_recency_weights_favour_recent_games() -> None:
    dates = pd.Series(pd.to_datetime(["2025-10-21", "2025-11-21", "2025-12-21"]))
    weights = recency_weights(dates, half_life_games=1.0)
    assert weights[2] == pytest.approx(1.0)   # newest
    assert weights[1] == pytest.approx(0.5)
    assert weights[0] == pytest.approx(0.25)  # oldest


def test_recency_weights_use_rank_not_elapsed_time() -> None:
    """A player who missed three weeks injured has not become staler than his last game says.
    Two samples with the same ordering must weight identically however the dates are spaced."""
    tight = pd.Series(pd.to_datetime(["2025-10-21", "2025-10-23", "2025-10-25"]))
    spread = pd.Series(pd.to_datetime(["2025-10-21", "2025-12-23", "2026-03-25"]))
    assert recency_weights(tight) == pytest.approx(recency_weights(spread))


def test_recency_half_life_is_what_it_says() -> None:
    dates = pd.Series(pd.to_datetime(["2025-10-21"]) .repeat(1).tolist()
                      + [pd.Timestamp("2025-10-21") + pd.Timedelta(days=2 * i) for i in range(1, 51)])
    weights = recency_weights(dates, half_life_games=DEFAULT_HALF_LIFE_GAMES)
    ordered = weights[np.argsort(-dates.to_numpy().astype("datetime64[ns]").view("int64"))]
    assert ordered[int(DEFAULT_HALF_LIFE_GAMES)] == pytest.approx(0.5, abs=1e-9)


def test_recency_weights_rejects_a_bad_half_life() -> None:
    with pytest.raises(ValueError, match="positive"):
        recency_weights(pd.Series(pd.to_datetime(["2025-10-21"])), half_life_games=0.0)


# --- scoring the logs ----------------------------------------------------------------------------


def test_score_game_logs_uses_the_scoring_engine() -> None:
    logs = make_logs([{"pts": 30, "reb": 5, "ast": 6, "stl": 2, "to": 3, "fg3m": 7}])
    # 1.5*30 + 5 + 6 + 2 - 3 + 0.5*7 = 58.5
    assert score_game_logs(logs).iloc[0] == pytest.approx(58.5)


def test_score_game_logs_rejects_missing_columns() -> None:
    logs = make_logs([{"pts": 10}]).drop(columns=["reb"])
    with pytest.raises(ValueError, match="missing box score columns"):
        score_game_logs(logs)


# --- availability / p_play --------------------------------------------------------------------------


def test_availability_counts_team_games_in_the_stint() -> None:
    """Player 1 plays 3 of the 4 team games inside his window."""
    logs = make_logs([{"pts": 10} for _ in range(4)])
    logs.loc[:, "nba_player_id"] = 1
    teammate = logs.copy()
    teammate["nba_player_id"] = 2
    logs = logs.drop(index=2)  # player 1 misses the third game
    combined = pd.concat([logs, teammate], ignore_index=True)

    p_play, played, available = availability(combined, 1)
    assert (played, available) == (3, 4)
    assert p_play == pytest.approx(0.75)


def test_availability_handles_a_midseason_trade() -> None:
    """A traded player must not be charged for both teams' full seasons.

    Measured per stint: he plays every DEN game in his DEN window and every MIA game in his
    MIA window, so he is fully available despite appearing for two teams.
    """
    den = make_logs([{"team": "DEN"} for _ in range(3)], start="2025-10-21")
    mia = make_logs([{"team": "MIA"} for _ in range(3)], start="2025-12-01")
    mia["game_id"] = ["0022601" + str(i) for i in range(3)]

    traded = pd.concat([den, mia], ignore_index=True)
    others = traded.copy()
    others["nba_player_id"] = 2  # teammates on both clubs, same games

    p_play, played, available = availability(pd.concat([traded, others]), 1)
    assert played == 6
    assert p_play == pytest.approx(1.0)


def test_availability_needs_the_player_to_exist() -> None:
    with pytest.raises(ValueError, match="no game logs"):
        availability(make_logs([{"pts": 1}]), 999)


def test_team_games_played_counts_distinct_games() -> None:
    logs = make_logs([{"pts": 1} for _ in range(3)])
    teammate = logs.copy()
    teammate["nba_player_id"] = 2
    counts = team_games_played(pd.concat([logs, teammate], ignore_index=True))
    assert int(counts.loc[counts["team"] == "DEN", "team_games"].iloc[0]) == 3


# --- building and shrinking -----------------------------------------------------------------------


def _logs_with_fp(n_games: int, pts: float = 20.0) -> pd.DataFrame:
    logs = make_logs([{"pts": pts} for _ in range(n_games)])
    logs["fp"] = score_game_logs(logs)
    return logs


def test_build_distribution_flags_thin_history() -> None:
    thin = build_distribution(_logs_with_fp(5), 1)
    assert thin.games_observed == 5
    assert not thin.history_reliable

    thick = build_distribution(_logs_with_fp(MIN_GAMES_FOR_RELIABLE_HISTORY + 5), 1)
    assert thick.history_reliable


def test_build_distribution_needs_the_player() -> None:
    with pytest.raises(ValueError, match="no game logs"):
        build_distribution(_logs_with_fp(3), 999)


def test_shrinkage_pulls_a_thin_sample_toward_the_prior() -> None:
    """A six-game sample must not produce a confident number."""
    player = EmpiricalScoreDistribution(
        values=np.full(6, 60.0), weights=np.ones(6), games_observed=6
    )
    prior = EmpiricalScoreDistribution(values=np.full(50, 30.0), weights=np.ones(50))

    shrunk = shrink_toward_prior(player, prior, prior_strength_games=20.0)
    assert 30.0 < shrunk.mean() < 60.0
    assert shrunk.mean() < 45.0  # the prior outweighs six games


def test_shrinkage_barely_moves_a_long_history() -> None:
    player = EmpiricalScoreDistribution(
        values=np.full(200, 60.0), weights=np.ones(200), games_observed=200
    )
    prior = EmpiricalScoreDistribution(values=np.full(50, 30.0), weights=np.ones(50))

    shrunk = shrink_toward_prior(player, prior, prior_strength_games=20.0)
    assert shrunk.mean() > 56.0


def test_shrinkage_keeps_the_players_own_availability() -> None:
    """How often he is available is about him, not about his role."""
    player = EmpiricalScoreDistribution(
        values=np.full(10, 60.0), weights=np.ones(10), _p_play=0.6, games_observed=10
    )
    prior = EmpiricalScoreDistribution(
        values=np.full(50, 30.0), weights=np.ones(50), _p_play=1.0
    )
    shrunk = shrink_toward_prior(player, prior)
    assert shrunk.p_play == pytest.approx(0.6)
    assert shrunk.games_observed == 10


def test_zero_strength_prior_is_a_no_op() -> None:
    player = EmpiricalScoreDistribution(values=np.full(10, 60.0), weights=np.ones(10))
    prior = EmpiricalScoreDistribution(values=np.full(50, 30.0), weights=np.ones(50))
    assert shrink_toward_prior(player, prior, 0.0).mean() == pytest.approx(60.0)


def test_negative_prior_strength_is_rejected() -> None:
    player = EmpiricalScoreDistribution(values=np.full(10, 60.0), weights=np.ones(10))
    with pytest.raises(ValueError, match="prior_strength_games"):
        shrink_toward_prior(player, player, -1.0)


def test_role_prior_selects_a_minutes_band() -> None:
    starters = make_logs([{"minutes": 34.0, "pts": 25.0} for _ in range(10)])
    bench = make_logs([{"minutes": 12.0, "pts": 6.0} for _ in range(10)])
    bench["nba_player_id"] = 2
    logs = pd.concat([starters, bench], ignore_index=True)
    logs["fp"] = score_game_logs(logs)

    prior = role_prior(logs, 28.0, 40.0)
    assert prior.mean() == pytest.approx(37.5)  # 1.5 * 25, starters only


def test_role_prior_on_an_empty_band() -> None:
    logs = _logs_with_fp(3)
    with pytest.raises(ValueError, match="minute band"):
        role_prior(logs, 44.0, 48.0)
