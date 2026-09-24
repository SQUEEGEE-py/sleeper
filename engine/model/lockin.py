"""Optimal stopping over a player's week (SPEC §5.2).

Lock-In counts exactly one game per player per week. After a game ends you may lock that
score, and you must decide before his next game tips off. If you never lock, you are given his
team's final game of the week -- which is 0 if he did not play, and never reverts to an earlier
game.

So a player's weekly value is not his average. It is the value of playing an optimal stopping
rule over the games he has that week:

    W_{n+1} = 0                                          week over, nothing locked
    W_n     = E[X_n]                                     forced: the final game, 0 if he sits
    W_i     = p_play * E[max(X_i, W_{i+1})] + (1 - p_play) * W_{i+1}

and the rule is: **lock game i if and only if the score you just watched beats W_{i+1}**, the
value of rolling on. Those thresholds are exactly what the nightly job reports in phase 3.

Two consequences the whole valuation rests on, both falling straight out of this recursion:

* **Volatility is an asset.** ``E[max(X, c)]`` rises with spread, so of two players with the
  same mean the more erratic one is worth more -- only the kept game counts.
* **More games help, with diminishing returns.** Each extra game is another draw, but it only
  helps when it beats the threshold you already had.

Everything here is a pure function of a distribution object. No I/O, no network, no globals.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ScoreDistribution(Protocol):
    """What the stopping rule needs from a player's single-game distribution.

    Implementations live in ``distributions.py``: an analytic normal (used by the regression
    test) and a weighted empirical sample (used for real players, because the 40+/50+ bonuses
    put weight in the right tail that a normal would miss).
    """

    @property
    def p_play(self) -> float:
        """Probability he appears in a given game."""

    def mean(self) -> float:
        """Mean score **conditional on playing**."""

    def expected_max(self, threshold: float) -> float:
        """``E[max(X, threshold)]`` **conditional on playing**."""


def roll_on_values(distribution: ScoreDistribution, n_games: int) -> list[float]:
    """``[W_1, ..., W_{n+1}]`` -- the value of entering each game with nothing locked yet.

    ``W_{i+1}`` is the lock threshold for game ``i``: the value of passing on it. The last
    element is 0 by definition (the week is over and nothing was locked).
    """
    if n_games < 0:
        raise ValueError(f"n_games must be >= 0, got {n_games}")
    if n_games == 0:
        return [0.0]

    p_play = distribution.p_play
    if not 0.0 <= p_play <= 1.0:
        raise ValueError(f"p_play must be in [0, 1], got {p_play}")

    # W_{n+1} = 0, and the final game is forced: there is no decision left to make, so its
    # value is E[X_n] rather than E[max(X_n, 0)]. Those differ only when a score can be
    # negative -- which it can, on a line with turnovers and little else -- and using the
    # recursion here would quietly assume the auto-lock fallback protects you from that. It
    # does not.
    values = [0.0, p_play * distribution.mean()]

    for _ in range(n_games - 1):
        roll_on = values[-1]
        hold = p_play * distribution.expected_max(roll_on) + (1.0 - p_play) * roll_on
        values.append(hold)

    return list(reversed(values))


def weekly_value(distribution: ScoreDistribution, n_games: int) -> float:
    """``W_1``: what this player is worth in a week containing ``n_games`` games."""
    return roll_on_values(distribution, n_games)[0]


def lock_threshold(distribution: ScoreDistribution, games_remaining_after_this: int) -> float:
    """The score tonight's game must beat to be worth locking.

    ``games_remaining_after_this`` is how many further games he has this week **after** the
    one just played. Zero means this was his last -- nothing to roll on to, so any score is
    worth keeping and the threshold is the smallest possible number.
    """
    if games_remaining_after_this < 0:
        raise ValueError("games_remaining_after_this must be >= 0")
    if games_remaining_after_this == 0:
        return float("-inf")
    return weekly_value(distribution, games_remaining_after_this)


def marginal_value_of_a_game(distribution: ScoreDistribution, n_games: int) -> float:
    """What the ``n_games``-th game of a week adds over having one fewer.

    Used on the board to show why a four-game week is worth less than four times a one-game
    week, and in waiver ranking where games remaining matters more than season value.
    """
    if n_games < 1:
        raise ValueError("n_games must be >= 1")
    return weekly_value(distribution, n_games) - weekly_value(distribution, n_games - 1)


def season_value(
    distribution: ScoreDistribution,
    games_per_week: list[int] | tuple[int, ...],
) -> float:
    """Mean weekly value over a real schedule (SPEC §5.2).

    ``games_per_week`` is one entry per fantasy week, straight out of the team x week grid
    that ``ingest_history`` writes. A bye week contributes 0 and still counts in the mean --
    a week with no games is a week this roster slot produced nothing.
    """
    if not games_per_week:
        raise ValueError("games_per_week is empty; need at least one week")
    return sum(weekly_value(distribution, n) for n in games_per_week) / len(games_per_week)
