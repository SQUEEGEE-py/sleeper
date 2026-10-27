"""Per-player single-game score distributions (SPEC §5.1).

A player is modelled as two separate things, because Lock-In treats them very differently:

* ``p_play`` -- the probability he appears at all. A missed game is not a bad game; under the
  auto-lock fallback it is a zero, but only if it is his *last* game of the week.
* the distribution of his score **given that he plays**.

The conditional distribution is kept as a weighted empirical sample rather than fitted to a
normal. The 40+/50+ bonuses and the double-double bonus all live in the right tail, so the
shape there is worth real money and a normal would flatten it.

Two distribution types are provided. Both satisfy ``lockin.ScoreDistribution``:

* :class:`NormalScoreDistribution` -- analytic, used for the closed-form regression test and
  for reasoning about what volatility is worth.
* :class:`EmpiricalScoreDistribution` -- a weighted sample, used for real players.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from engine.model.scoring import BoxScore, ScoringRules, score_box

# Recent games say more about a player than old ones. SPEC §5.1 suggests a half-life of about
# 25 games -- roughly a third of a season, so last season still counts but this season
# dominates.
DEFAULT_HALF_LIFE_GAMES = 25.0

# How many games' worth of evidence the role prior is worth. A player with far fewer games
# than this is mostly described by his prior; one with many more is mostly described by
# himself. Shrinkage is what stops a 3-game sample producing a confident projection.
DEFAULT_PRIOR_STRENGTH_GAMES = 20.0

# Below this many games, a player's own history cannot carry a projection on its own and the
# board must flag him (SPEC §5.3 wants "a flag for players whose history is unreliable").
MIN_GAMES_FOR_RELIABLE_HISTORY = 20

# How much less often a player appears on the second night of a back-to-back, by age at the
# game date. Measured 2026-10-19 on 2023-26 logs, rotation players only (20+ min that season),
# 57,769 player-games, same per-stint windows as ``availability``:
#
#     age    other nights          B2B 2nd night         ratio
#     <30    0.838 (n=37,314)      0.796 (n=8,096)       0.951
#     30-33  0.835 (n=6,873)       0.753 (n=1,514)       0.901
#     34+    0.867 (n=3,268)       0.761 (n=704)         0.879
#
# It is rest, not lingering injury: conditional on playing the previous game the ratios are
# 0.951 / 0.923 / 0.881. And it must be a multiplier on B2B nights, not an age penalty on
# p_play overall -- 34+ rotation players are *more* available on ordinary nights (survivors).
# Each entry is (lowest age in the band, ratio); a player's band is the last one he reaches.
B2B_AVAILABILITY_RATIO_BY_AGE: tuple[tuple[int, float], ...] = (
    (0, 0.951),
    (30, 0.901),
    (34, 0.879),
)


def age_on(birth_date: date, day: date) -> int:
    """Whole years old on ``day``."""
    return day.year - birth_date.year - ((day.month, day.day) < (birth_date.month, birth_date.day))


def b2b_availability_ratio(age: int) -> float:
    """The measured B2B second-night multiplier for a player of this age."""
    ratio = B2B_AVAILABILITY_RATIO_BY_AGE[0][1]
    for lowest_age, band_ratio in B2B_AVAILABILITY_RATIO_BY_AGE:
        if age >= lowest_age:
            ratio = band_ratio
    return ratio


def game_p_play(base_p_play: float, b2b_second: bool, age: int | None) -> float:
    """``p_play`` for one specific game.

    ``base_p_play`` is the player's own historical rate (``availability``). On the second night
    of a back-to-back it is multiplied by his age band's measured ratio; on any other night it
    is unchanged.

    ``age=None`` means his age is unknown, and then the game is left at the base rate rather
    than charged a guessed band. Callers must surface that as a gap -- an unknown age means the
    back-to-back is *not* priced, which is a different claim from "it does not matter".

    Known approximation: the base rate already averages over his B2B nights (~18% of games),
    so ordinary nights are understated by about ``0.18 x (1 - ratio)`` -- at most ~2% for 34+.
    """
    if not 0.0 <= base_p_play <= 1.0:
        raise ValueError(f"base_p_play must be in [0, 1], got {base_p_play}")
    if not b2b_second or age is None:
        return base_p_play
    return base_p_play * b2b_availability_ratio(age)


@dataclass(frozen=True)
class NormalScoreDistribution:
    """A normal distribution of scores, conditional on playing.

    Not how a real player is modelled -- real scores are right-skewed and bonus-laden -- but
    it is the case the optimal-stopping regression test is defined against, and it makes the
    "volatility is an asset" claim checkable in closed form.
    """

    mu: float
    sigma: float
    _p_play: float = 1.0

    @property
    def p_play(self) -> float:
        return self._p_play

    def mean(self) -> float:
        return self.mu

    def expected_max(self, threshold: float) -> float:
        """``E[max(X, c)]`` for X ~ N(mu, sigma), in closed form:

            c * Phi(z) + mu * (1 - Phi(z)) + sigma * phi(z),   z = (c - mu) / sigma
        """
        if self.sigma <= 0:
            return max(self.mu, threshold)
        z = (threshold - self.mu) / self.sigma
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        return threshold * cdf + self.mu * (1.0 - cdf) + self.sigma * pdf

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """``size`` scores **conditional on playing**, for the matchup simulation."""
        return rng.normal(self.mu, max(self.sigma, 0.0), size)


@dataclass(frozen=True)
class EmpiricalScoreDistribution:
    """A weighted sample of scores, conditional on playing.

    ``values`` are fantasy point totals from rescored box scores; ``weights`` are the recency
    weights (and, after shrinkage, the prior's share). Weights need not sum to 1.
    """

    values: np.ndarray
    weights: np.ndarray
    _p_play: float = 1.0
    games_observed: int = 0
    history_reliable: bool = True

    def __post_init__(self) -> None:
        if len(self.values) != len(self.weights):
            raise ValueError("values and weights must be the same length")
        if len(self.values) == 0:
            raise ValueError("cannot build a distribution from an empty sample")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("values contain non-finite entries")
        if np.any(self.weights < 0):
            raise ValueError("weights must be non-negative")
        if self.weights.sum() <= 0:
            raise ValueError("weights sum to zero")

    @property
    def p_play(self) -> float:
        return self._p_play

    def _normalized_weights(self) -> np.ndarray:
        return self.weights / self.weights.sum()

    def mean(self) -> float:
        return float(np.dot(self._normalized_weights(), self.values))

    def sd(self) -> float:
        weights = self._normalized_weights()
        mean = float(np.dot(weights, self.values))
        variance = float(np.dot(weights, (self.values - mean) ** 2))
        return math.sqrt(max(variance, 0.0))

    def expected_max(self, threshold: float) -> float:
        return float(np.dot(self._normalized_weights(), np.maximum(self.values, threshold)))

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """``size`` scores **conditional on playing**, drawn from the weighted sample itself --
        not a fitted curve, so the 40+/50+ bonus tail survives into the simulation."""
        return rng.choice(self.values, size=size, p=self._normalized_weights())

    def quantile(self, q: float) -> float:
        """Weighted quantile. Used for the board's ceiling/floor columns."""
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must be in [0, 1]")
        order = np.argsort(self.values)
        values = self.values[order]
        cumulative = np.cumsum(self._normalized_weights()[order])
        index = int(np.searchsorted(cumulative, q))
        return float(values[min(index, len(values) - 1)])

    @property
    def effective_sample_size(self) -> float:
        """Kish effective sample size. Recency weighting means 200 games of history are worth
        rather fewer than 200 independent observations."""
        weights = self._normalized_weights()
        return float(1.0 / np.sum(weights**2))


# --- building distributions from rescored game logs -----------------------------------------


def recency_weights(
    game_dates: pd.Series, half_life_games: float = DEFAULT_HALF_LIFE_GAMES
) -> np.ndarray:
    """Exponential decay by recency **rank**, not by calendar time.

    Rank, because the gap between games is mostly an artifact of the schedule -- a player who
    missed three weeks injured has not become three weeks staler than his last game suggests.
    The most recent game gets weight 1.
    """
    if half_life_games <= 0:
        raise ValueError("half_life_games must be positive")
    order = np.argsort(np.argsort(-game_dates.to_numpy().astype("datetime64[ns]").view("int64")))
    return np.power(0.5, order / half_life_games)


def score_game_logs(logs: pd.DataFrame, rules: ScoringRules | None = None) -> pd.Series:
    """Rescore box score rows into fantasy points through the one scoring implementation."""
    fields = list(BoxScore.model_fields)
    missing = [field for field in fields if field not in logs.columns]
    if missing:
        raise ValueError(f"game logs are missing box score columns {missing}")
    records = logs[fields].to_dict("records")
    return pd.Series(
        [score_box(BoxScore(**record), rules) for record in records], index=logs.index
    )


def team_games_played(logs: pd.DataFrame) -> pd.DataFrame:
    """Distinct games per (season, team), from the logs themselves.

    The denominator for ``p_play``. Derived from the logs rather than the schedule so that
    ``p_play`` needs no extra data source and no extra network call.
    """
    return (
        logs.groupby(["season", "team"])["game_id"]
        .nunique()
        .rename("team_games")
        .reset_index()
    )


def availability(logs: pd.DataFrame, player_id: int) -> tuple[float, int, int]:
    """``(p_play, games_played, games_available)`` for one player.

    Availability is measured **per team stint**: for each (season, team) the player appears
    in, the window runs from his first to his last game for that team, and the denominator is
    that team's games inside the window. Per stint, because a traded player would otherwise be
    charged for both teams' full seasons and look half-available.

    The honest limitation: because each window starts at his first appearance, games missed
    at the very start or end of a stint are invisible here. Historical ``p_play`` is a base
    rate for *in-stint* availability -- load management and mid-season injuries, which is what
    it is used for. Injury designations from Sleeper and news adjust it at decision time
    (SPEC §7.2).
    """
    player_logs = logs[logs["nba_player_id"] == player_id]
    if player_logs.empty:
        raise ValueError(f"no game logs for player {player_id}")

    played = 0
    available = 0
    for (season, team), stint in player_logs.groupby(["season", "team"]):
        first, last = stint["game_date"].min(), stint["game_date"].max()
        team_logs = logs[(logs["season"] == season) & (logs["team"] == team)]
        in_window = team_logs[
            (team_logs["game_date"] >= first) & (team_logs["game_date"] <= last)
        ]
        played += len(stint)
        available += in_window["game_id"].nunique()

    if available == 0:
        return 1.0, played, 0
    return played / available, played, available


# The share of team games a rotation player plays on an ordinary night, measured on 2023-26 for
# the back-to-back study (0.838 for under-30s, the bulk of the league; 0.835 at 30-33).
AVAILABILITY_PRIOR = 0.84
AVAILABILITY_PRIOR_GAMES = 10.0


def shrunk_availability(games_played: int, games_available: int) -> float:
    """``p_play`` shrunk toward ``AVAILABILITY_PRIOR`` as if he had played that share of
    ``AVAILABILITY_PRIOR_GAMES`` extra games.

    A stint window starts at his first game and ends at his last, so a player who has played
    every game of a short window reads 100% -- Caleb Wilson after four games (2026-10-27),
    which had the advisor starting him over Bey and LeBron. With 10 prior games 4-of-4 reads
    0.89; a veteran's 180-of-210 moves from 0.857 to 0.856.
    """
    return ((games_played + AVAILABILITY_PRIOR_GAMES * AVAILABILITY_PRIOR)
            / (games_available + AVAILABILITY_PRIOR_GAMES))


def build_distribution(
    logs: pd.DataFrame,
    player_id: int,
    rules: ScoringRules | None = None,
    half_life_games: float = DEFAULT_HALF_LIFE_GAMES,
    prior: EmpiricalScoreDistribution | None = None,
    prior_strength_games: float = DEFAULT_PRIOR_STRENGTH_GAMES,
) -> EmpiricalScoreDistribution:
    """One player's conditional score distribution, recency-weighted and optionally shrunk.

    ``logs`` must already carry a ``fp`` column (see :func:`score_game_logs`), so that three
    seasons are rescored once rather than once per player.
    """
    player_logs = logs[logs["nba_player_id"] == player_id]
    if player_logs.empty:
        raise ValueError(f"no game logs for player {player_id}")

    values = player_logs["fp"].to_numpy(dtype=float)
    weights = recency_weights(player_logs["game_date"], half_life_games)
    _, games_played, games_available = availability(logs, player_id)
    p_play = shrunk_availability(games_played, games_available)

    distribution = EmpiricalScoreDistribution(
        values=values,
        weights=weights,
        _p_play=p_play,
        games_observed=games_played,
        history_reliable=games_played >= MIN_GAMES_FOR_RELIABLE_HISTORY,
    )

    if prior is not None:
        distribution = shrink_toward_prior(distribution, prior, prior_strength_games)
    return distribution


def shrink_toward_prior(
    player: EmpiricalScoreDistribution,
    prior: EmpiricalScoreDistribution,
    prior_strength_games: float = DEFAULT_PRIOR_STRENGTH_GAMES,
) -> EmpiricalScoreDistribution:
    """Blend a player's own sample with a role prior (SPEC §5.1 step 3).

    The two weighted samples are concatenated, with the prior's total weight set to
    ``prior_strength_games`` and the player's to his effective sample size. A player with lots
    of history barely moves; a rookie or a role-changer is carried almost entirely by the
    prior. That is the point -- it is what stops a six-game sample producing a confident
    number.

    ``p_play`` and the reliability flag come from the player, not the prior: how often he is
    available is about him.
    """
    if prior_strength_games < 0:
        raise ValueError("prior_strength_games must be >= 0")
    if prior_strength_games == 0:
        return player

    player_weights = player.weights / player.weights.sum() * player.effective_sample_size
    prior_weights = prior.weights / prior.weights.sum() * prior_strength_games

    return EmpiricalScoreDistribution(
        values=np.concatenate([player.values, prior.values]),
        weights=np.concatenate([player_weights, prior_weights]),
        _p_play=player.p_play,
        games_observed=player.games_observed,
        history_reliable=player.history_reliable,
    )


def role_prior(
    logs: pd.DataFrame,
    minutes_low: float,
    minutes_high: float,
    half_life_games: float = DEFAULT_HALF_LIFE_GAMES,
) -> EmpiricalScoreDistribution:
    """A prior built from every game played inside a minutes band (SPEC §5.1 step 3).

    Minutes are the usable proxy for role here: a 32-minute player and a 14-minute player have
    different distributions whatever their listed position. For a rookie or a role-changer,
    the band comes from projected minutes rather than history -- which is what
    ``config/player_overrides.csv`` is for.
    """
    band = logs[(logs["minutes"] >= minutes_low) & (logs["minutes"] < minutes_high)]
    if band.empty:
        raise ValueError(f"no games in the {minutes_low}-{minutes_high} minute band")
    return EmpiricalScoreDistribution(
        values=band["fp"].to_numpy(dtype=float),
        weights=recency_weights(band["game_date"], half_life_games),
        _p_play=1.0,
        games_observed=len(band),
        history_reliable=True,
    )
