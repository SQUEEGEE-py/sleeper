"""Tests for engine/model/lockin.py (TASKS M3).

The anchor is the closed-form regression in SPEC §5.2 and CLAUDE.md: a normally distributed
player with mean 45 and sd 10 is worth 45.0 / 49.0 / 51.3 / 52.9 in a 1 / 2 / 3 / 4 game week.
It is checked both analytically and by simulation, because the analytic path alone would only
prove the normal formula matches itself.

The rest of the file pins the structural claims the whole valuation rests on: volatility is an
asset, more games help with diminishing returns, and missed games are cheap.

No test touches the network.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.model.distributions import EmpiricalScoreDistribution, NormalScoreDistribution
from engine.model.lockin import (
    lock_threshold,
    marginal_value_of_a_game,
    roll_on_values,
    season_value,
    weekly_value,
)

# SPEC §5.2 / CLAUDE.md "Testing".
REGRESSION_MEAN = 45.0
REGRESSION_SD = 10.0
REGRESSION_EXPECTED = {1: 45.0, 2: 49.0, 3: 51.3, 4: 52.9}


@pytest.fixture
def normal() -> NormalScoreDistribution:
    return NormalScoreDistribution(REGRESSION_MEAN, REGRESSION_SD)


# --- the regression -------------------------------------------------------------------------


@pytest.mark.parametrize("n_games,expected", sorted(REGRESSION_EXPECTED.items()))
def test_closed_form_regression(
    normal: NormalScoreDistribution, n_games: int, expected: float
) -> None:
    """The number this whole module exists to get right."""
    assert weekly_value(normal, n_games) == pytest.approx(expected, abs=0.05)


@pytest.mark.parametrize("n_games,expected", sorted(REGRESSION_EXPECTED.items()))
def test_simulation_reproduces_the_closed_form(n_games: int, expected: float) -> None:
    """CLAUDE.md: "Simulation must reproduce this within tolerance."

    A large sample drawn from the same normal, run through the empirical path, has to land on
    the analytic answer. This is what proves the empirical machinery -- which is what real
    players actually use -- agrees with the closed form.
    """
    rng = np.random.default_rng(20261020)
    sample = rng.normal(REGRESSION_MEAN, REGRESSION_SD, size=200_000)
    empirical = EmpiricalScoreDistribution(
        values=sample, weights=np.ones_like(sample), games_observed=len(sample)
    )
    assert weekly_value(empirical, n_games) == pytest.approx(expected, abs=0.15)


def test_thresholds_are_the_roll_on_values(normal: NormalScoreDistribution) -> None:
    """In a 4-game week the lock thresholds are W_2, W_3, W_4, then nothing left to roll to."""
    values = roll_on_values(normal, 4)
    assert values[0] == pytest.approx(REGRESSION_EXPECTED[4], abs=0.05)
    assert values[1] == pytest.approx(REGRESSION_EXPECTED[3], abs=0.05)
    assert values[2] == pytest.approx(REGRESSION_EXPECTED[2], abs=0.05)
    assert values[3] == pytest.approx(REGRESSION_EXPECTED[1], abs=0.05)
    assert values[4] == 0.0


def test_thresholds_fall_as_the_week_runs_out(normal: NormalScoreDistribution) -> None:
    """Be picky early, take what you can get late -- the bar drops with each game used up."""
    values = roll_on_values(normal, 5)
    assert values == sorted(values, reverse=True)


# --- the structural claims -------------------------------------------------------------------


@pytest.mark.parametrize("n_games", [2, 3, 4])
def test_volatility_is_an_asset(n_games: int) -> None:
    """SPEC §3: two players with equal means are not equal; the erratic one is worth more,
    because only the kept game counts. This is the single biggest way Lock-In departs from
    standard fantasy, so it gets an explicit test rather than being left implicit."""
    values = [
        weekly_value(NormalScoreDistribution(REGRESSION_MEAN, sd), n_games)
        for sd in (0.0, 5.0, 10.0, 20.0)
    ]
    assert values == sorted(values)
    assert values[0] == pytest.approx(REGRESSION_MEAN)  # a metronome is worth exactly its mean


def test_one_game_week_is_just_the_mean_times_availability(
    normal: NormalScoreDistribution,
) -> None:
    """With a single game there is no decision to make: you get what he scores, or zero."""
    assert weekly_value(normal, 1) == pytest.approx(REGRESSION_MEAN)
    half = NormalScoreDistribution(REGRESSION_MEAN, REGRESSION_SD, 0.5)
    assert weekly_value(half, 1) == pytest.approx(REGRESSION_MEAN * 0.5)


def test_more_games_help_with_diminishing_returns(normal: NormalScoreDistribution) -> None:
    gains = [marginal_value_of_a_game(normal, n) for n in range(2, 7)]
    assert all(gain > 0 for gain in gains)
    assert gains == sorted(gains, reverse=True)


def test_the_final_game_is_forced_not_optional() -> None:
    """The last game of the week has no roll-on option, so its value is E[X], not E[max(X,0)].

    It matters when a score can be negative: a line of no points and three turnovers is a real
    thing, and the auto-lock fallback does not protect you from it. Using the general
    recursion at the terminal step would quietly assume it does.
    """
    negative = EmpiricalScoreDistribution(
        values=np.array([-3.0, -1.0, 10.0]), weights=np.ones(3), games_observed=3
    )
    assert weekly_value(negative, 1) == pytest.approx(2.0)  # the plain mean, negatives included


def test_recursion_with_partial_availability_is_exact() -> None:
    """Pins both p_play branches of W_i at once, against a hand-computed value.

    Normal(45, 10), p_play = 0.85, two-game week:
        W_2 = 0.85 * 45                                    = 38.250
        E[max(X, 38.25)] = 38.25*Phi(z) + 45*(1-Phi(z)) + 10*phi(z),  z = -0.675
                         = 9.556 + 33.759 + 3.177          = 46.492
        W_1 = 0.85 * 46.492 + 0.15 * 38.25                 = 45.256

    Dropping either the ``p_play *`` or the ``(1 - p_play) *`` term changes this.
    """
    distribution = NormalScoreDistribution(45.0, 10.0, 0.85)
    assert roll_on_values(distribution, 2)[1] == pytest.approx(38.25, abs=0.005)
    assert weekly_value(distribution, 2) == pytest.approx(45.256, abs=0.005)


@pytest.mark.parametrize("n_games", [1, 2, 4])
def test_value_rises_with_availability(n_games: int) -> None:
    values = [
        weekly_value(NormalScoreDistribution(45.0, 10.0, p), n_games)
        for p in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert values == sorted(values)
    assert values[0] == pytest.approx(0.0)


def test_missed_games_are_cheap() -> None:
    """SPEC §3: rest-prone stars are systematically underpriced by standard rankings.

    Losing 15% of his games must cost a player well under 15% of his weekly value, because in
    a multi-game week the remaining games still give the stopping rule something to work with.
    """
    always = NormalScoreDistribution(REGRESSION_MEAN, REGRESSION_SD, 1.0)
    rests = NormalScoreDistribution(REGRESSION_MEAN, REGRESSION_SD, 0.85)

    loss = 1.0 - weekly_value(rests, 4) / weekly_value(always, 4)
    assert 0 < loss < 0.15, f"missing 15% of games cost {loss:.1%} of value"


def test_availability_matters_more_in_a_short_week() -> None:
    """A one-game week is all-or-nothing; a four-game week absorbs an absence. This is why the
    nightly job forces a lock when a player's next game is his last of the week."""
    always = NormalScoreDistribution(REGRESSION_MEAN, REGRESSION_SD, 1.0)
    rests = NormalScoreDistribution(REGRESSION_MEAN, REGRESSION_SD, 0.85)

    short = 1.0 - weekly_value(rests, 1) / weekly_value(always, 1)
    long = 1.0 - weekly_value(rests, 4) / weekly_value(always, 4)
    assert short > long


# --- edges and errors -------------------------------------------------------------------------


def test_zero_game_week_is_worth_nothing(normal: NormalScoreDistribution) -> None:
    """Bye weeks are real: the schedule grid has a week where some team plays nobody."""
    assert weekly_value(normal, 0) == 0.0
    assert roll_on_values(normal, 0) == [0.0]


def test_a_player_who_never_plays_is_worth_nothing() -> None:
    assert weekly_value(NormalScoreDistribution(45.0, 10.0, 0.0), 4) == pytest.approx(0.0)


def test_lock_threshold_on_the_last_game_of_the_week(normal: NormalScoreDistribution) -> None:
    """Nothing left to roll on to, so any score is worth keeping."""
    assert lock_threshold(normal, 0) == float("-inf")
    assert lock_threshold(normal, 2) == pytest.approx(weekly_value(normal, 2))


def test_rejects_invalid_inputs(normal: NormalScoreDistribution) -> None:
    with pytest.raises(ValueError, match="n_games"):
        roll_on_values(normal, -1)
    with pytest.raises(ValueError, match="p_play"):
        weekly_value(NormalScoreDistribution(45.0, 10.0, 1.5), 2)
    with pytest.raises(ValueError, match="games_remaining_after_this"):
        lock_threshold(normal, -1)
    with pytest.raises(ValueError, match="n_games"):
        marginal_value_of_a_game(normal, 0)


# --- season value -----------------------------------------------------------------------------


def test_season_value_averages_over_the_schedule(normal: NormalScoreDistribution) -> None:
    schedule = [3, 4, 2]
    expected = sum(weekly_value(normal, n) for n in schedule) / 3
    assert season_value(normal, schedule) == pytest.approx(expected)


def test_season_value_counts_bye_weeks(normal: NormalScoreDistribution) -> None:
    """A week with no games produced nothing and must drag the average down, not be skipped."""
    assert season_value(normal, [3, 0]) < season_value(normal, [3])


def test_season_value_needs_a_schedule(normal: NormalScoreDistribution) -> None:
    with pytest.raises(ValueError, match="empty"):
        season_value(normal, [])
