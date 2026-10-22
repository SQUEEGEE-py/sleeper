"""One place that turns a player id into his single-game score distribution.

The board, the cheat sheet and the nightly job must agree on what a player is worth, so all of
them build distributions here instead of each carrying a copy of the recipe. Moved out of
``build_board.main()`` unchanged on 2026-10-19; the board's ``pool.csv`` was diffed before and
after the move and is identical.

The recipe (each step's reasoning lives with the function it calls):

* **A player with regular-season history** -- recency-weighted logs, shrunk toward the prior
  for his minutes band (``distributions.build_distribution``). For a thin history (under 20
  games) the band comes from his preseason-projected minutes when he has any.
* **A rookie** -- past player-seasons at his projected minutes, scaled by the measured rookie
  discount (``rookies.rookie_cohort_distribution``).
* Either way, a row in ``config/player_overrides.csv`` is applied last.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from engine.ingest.player_ids import PlayerOverride
from engine.model.aging import age_adjusted, effective_factor, previous_season_share
from engine.model.distributions import (
    MIN_GAMES_FOR_RELIABLE_HISTORY,
    EmpiricalScoreDistribution,
    build_distribution,
    role_prior,
)
from engine.model.rookies import (
    MinutesProjection,
    fit_minutes_projection,
    player_seasons,
    preseason_minutes,
    rookie_cohort_distribution,
    rookie_discount,
    rookie_seasons,
)

# Minutes bands for the role prior (SPEC §5.1 step 3). Minutes are the usable proxy for role:
# a 34-minute player and a 14-minute player have different distributions whatever their listed
# position. Bands are wide enough that each holds thousands of games.
MINUTES_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 12.0),
    (12.0, 18.0),
    (18.0, 24.0),
    (24.0, 30.0),
    (30.0, 36.0),
    (36.0, 60.0),
)


def build_minutes_priors(logs: pd.DataFrame) -> dict[tuple[float, float], EmpiricalScoreDistribution]:
    """One prior per minutes band, built once and shared across players."""
    priors = {}
    for band in MINUTES_BANDS:
        try:
            priors[band] = role_prior(logs, band[0], band[1])
        except ValueError:
            continue  # an empty band is fine; nobody will map to it
    return priors


def birth_dates_by_nba_id(nba_to_sleeper: dict[int, str], sleeper_players: dict) -> dict[int, date]:
    """Sleeper ``birth_date`` per NBA id, for the age adjustment. Players with no or an
    unreadable birth date are simply absent; ``PlayerModel`` then reports them as unaged."""
    births: dict[int, date] = {}
    for nba_id, sleeper_id in nba_to_sleeper.items():
        player = sleeper_players.get(sleeper_id)
        text = getattr(player, "birth_date", None)
        try:
            if text:
                births[int(nba_id)] = date.fromisoformat(text)
        except ValueError:
            continue
    return births


def band_for(minutes: float) -> tuple[float, float]:
    for band in MINUTES_BANDS:
        if band[0] <= minutes < band[1]:
            return band
    return MINUTES_BANDS[-1]


def apply_override(
    distribution: EmpiricalScoreDistribution, multiplier: float, p_play: float | None
) -> EmpiricalScoreDistribution:
    """Scale a distribution for a role change, and/or force availability."""
    updated = distribution
    if multiplier != 1.0:
        updated = dataclasses.replace(updated, values=updated.values * multiplier)
    if p_play is not None:
        updated = dataclasses.replace(updated, _p_play=p_play)
    return updated


@dataclass
class PlayerModel:
    """Everything needed to build any player's distribution, computed once.

    ``logs`` must carry ``fp`` (``build_board.load_logs`` adds it). ``preseason`` may be empty.
    """

    logs: pd.DataFrame
    preseason: pd.DataFrame
    latest_season: str
    override_by_id: dict[int, PlayerOverride] = field(default_factory=dict)
    # The season being played, and birth dates by NBA id: together they switch on the age
    # adjustment (``aging.py``). Left out, the model is exactly the pre-step-6 one -- which is
    # what the aging backtest itself needs. A player with no birth date is left unadjusted and
    # listed in ``unaged``, never given a guessed age.
    season: str | None = None
    birth_dates: dict[int, date] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.priors = build_minutes_priors(self.logs)
        self.mean_minutes = self.logs.groupby("nba_player_id")["minutes"].mean()
        self.games_played = self.logs.groupby("nba_player_id").size()
        self.projection: MinutesProjection | None = None
        self.preseason_minutes_by_id: dict[int, float] = {}
        if len(self.preseason):
            self.projection = fit_minutes_projection(
                self.logs[self.logs["season"] == self.latest_season], self.preseason
            )
            self.preseason_minutes_by_id = (
                preseason_minutes(self.preseason).set_index("nba_player_id")["minutes"].to_dict()
            )
        # The rookie tables are slow (per-player availability), so they are built on first use.
        self._seasons_table: pd.DataFrame | None = None
        self._discount: float | None = None
        self.age_factors: dict[int, float] = {}  # effective factor applied, for reports
        self.unaged: set[int] = set()

    # --- who is who ---------------------------------------------------------------------------

    def has_history(self, nba_id: int) -> bool:
        return nba_id in self.games_played.index

    def is_rookie(self, nba_id: int) -> bool:
        """No regular-season games in the window, but a preseason role to project from."""
        return not self.has_history(nba_id) and nba_id in self.preseason_minutes_by_id

    def projected_minutes(self, nba_id: int) -> float:
        if self.projection is None or nba_id not in self.preseason_minutes_by_id:
            raise KeyError(f"no preseason minutes for player {nba_id}")
        return self.projection.project(self.preseason_minutes_by_id[nba_id])

    # --- distributions --------------------------------------------------------------------------

    def _override(self, nba_id: int, distribution: EmpiricalScoreDistribution) -> EmpiricalScoreDistribution:
        override = self.override_by_id.get(nba_id)
        if override is None:
            return distribution
        return apply_override(distribution, override.value_multiplier, override.p_play)

    def veteran(self, nba_id: int) -> EmpiricalScoreDistribution:
        """Recency-weighted, shrunk toward the prior for his minutes band, then overridden.

        Without the shrinkage a volatile 14-minute rookie outranks established starters:
        Lock-In rewards variance, and a thin sample is almost all variance. The prior is what
        makes a small sample behave like a small sample.

        For a thin history the prior's band comes from his **preseason-projected** minutes
        when he has any. Otherwise the band is picked by the very sample it is meant to
        temper: late-season call-ups on tanking teams (Kennedy Chandler, 11 games at 32
        minutes for Utah in March-April 2026) got a 30-36 minute prior and ranked 59th.
        """
        minutes = float(self.mean_minutes.get(nba_id, 0.0))
        if (
            self.projection is not None
            and self.games_played.get(nba_id, 0) < MIN_GAMES_FOR_RELIABLE_HISTORY
            and nba_id in self.preseason_minutes_by_id
        ):
            minutes = self.projection.project(self.preseason_minutes_by_id[nba_id])
        prior = self.priors.get(band_for(minutes))
        distribution = build_distribution(self.logs, nba_id, prior=prior)
        return self._override(nba_id, self._aged(nba_id, distribution))

    def _aged(self, nba_id: int, distribution: EmpiricalScoreDistribution) -> EmpiricalScoreDistribution:
        """The age adjustment, for veterans only: the rookie cohort is calibrated on its own.
        Applied before an override, which is a deliberate human correction and has the last
        word."""
        if self.season is None:
            return distribution
        birth = self.birth_dates.get(nba_id)
        if birth is None:
            self.unaged.add(nba_id)
            return distribution
        share = previous_season_share(self.logs[self.logs["nba_player_id"] == nba_id], self.season)
        factor = effective_factor(birth, self.season, share)
        self.age_factors[nba_id] = factor
        return age_adjusted(distribution, factor)

    @property
    def rookie_discount(self) -> float:
        if self._discount is None:
            self._discount = rookie_discount(self.logs, rookie_seasons(self.logs))
        return self._discount

    def rookie_at(self, nba_id: int, minutes: float) -> EmpiricalScoreDistribution:
        """A rookie's distribution at a given projected minutes (the board also asks for the
        +/- one-sd minutes to show a range)."""
        if self._seasons_table is None:
            self._seasons_table = player_seasons(self.logs)
        distribution = rookie_cohort_distribution(
            self.logs, self._seasons_table, minutes, self.rookie_discount
        )
        return self._override(nba_id, distribution)

    def distribution(self, nba_id: int) -> EmpiricalScoreDistribution:
        """The distribution for anyone: history if he has it, else the rookie projection."""
        if self.has_history(nba_id):
            return self.veteran(nba_id)
        if self.is_rookie(nba_id):
            return self.rookie_at(nba_id, self.projected_minutes(nba_id))
        raise KeyError(f"player {nba_id} has neither regular-season logs nor preseason minutes")
