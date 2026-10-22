"""Age adjustment: scale a veteran's distribution by how players his age change year to year.

Measured 2026-10-22 (phase 3 step 6) as a **backtest of the model itself**, not of raw
season-to-season changes: for 2024-25 and 2025-26, ``PlayerModel`` was built from the earlier
seasons only, and each player's projected mean (conditional on playing) was compared with what
he then averaged, for every player with 20+ games in both seasons (713, ages from Sleeper
birth dates; age on 1 February of the target season). Fitting
``log(actual / projected) ~ age + last season's minutes``:

    age      -0.0176 per year (se 0.0034)   2024-25 alone -0.0159, 2025-26 alone -0.0195
    minutes  -0.0096 per minute, but U-shaped: x1.53 under 12 min, x0.87-0.89 at 18-30,
             x0.98 over 30 -- role change, not a rule. Left to current-season data.

Raw year-over-year ratios were rejected first: selecting players by last season's minutes
puts every age band below 1.0 (regression to the mean), which is not aging.

**Past 34 the line is held flat (the user's choice, 2026-10-22).** The line predicts x0.84 for
35+ relative to 25-29; measured on 29 player-seasons it was x0.98 (95% 0.89-1.07). The decline
there is concentrated in role players (Conley, Lowry, Lopez, Gordon, Batum) while stars held
(Durant 49 -> 51, 49 -> 49; Curry; Harden; LeBron 51 -> 54 at 40, 49 -> 45 at 41) -- too few to
separate survivorship from quality, so the curve does not extrapolate past the solid data.

The factor is **relative to a 27-year-old**: it reshapes the board by age, it does not move
the league's overall level. And it applies only to the weight that still comes from
**previous seasons**: as this season's games accumulate (rank-weighted, half-life 25 games)
they are the player as he is now, so the adjustment fades out by itself.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import date

import numpy as np
import pandas as pd

from engine.model.distributions import DEFAULT_HALF_LIFE_GAMES, EmpiricalScoreDistribution, age_on, recency_weights

AGE_SLOPE_PER_YEAR = -0.0176
REFERENCE_AGE = 27
OLDEST_AGE_ON_THE_LINE = 34  # held flat beyond this; see the module docstring


def age_factor(age: int) -> float:
    """The measured multiplier for a player of ``age`` (on 1 February of the season)."""
    return math.exp(AGE_SLOPE_PER_YEAR * (min(age, OLDEST_AGE_ON_THE_LINE) - REFERENCE_AGE))


def season_reference_date(season: str) -> date:
    """1 February of a season like "2026-27": the date ages were measured on."""
    return date(int(season[:4]) + 1, 2, 1)


def previous_season_share(player_logs: pd.DataFrame, season: str,
                          half_life_games: float = DEFAULT_HALF_LIFE_GAMES) -> float:
    """The share of a player's recency weight that comes from seasons before ``season``.
    1.0 on opening night; falls as this season's games arrive."""
    if player_logs.empty:
        return 0.0
    weights = recency_weights(player_logs["game_date"], half_life_games)
    earlier = (player_logs["season"] != season).to_numpy()
    return float(weights[earlier].sum() / weights.sum())


def effective_factor(birth_date: date, season: str, previous_share: float) -> float:
    """The multiplier actually applied: the age factor, faded by how much of the player's
    weight is already this season's games."""
    factor = age_factor(age_on(birth_date, season_reference_date(season)))
    return 1.0 + (factor - 1.0) * previous_share


def age_adjusted(distribution: EmpiricalScoreDistribution, factor: float) -> EmpiricalScoreDistribution:
    """Scale every score -- spread and the bonus tail with it, as ``rookies.py`` does."""
    if factor == 1.0:
        return distribution
    return dataclasses.replace(distribution, values=np.asarray(distribution.values) * factor)
