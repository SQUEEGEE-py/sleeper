"""Valuing players with no NBA regular-season history: rookies (SPEC §5.1 step 3).

The board's normal path builds a player from his own game logs. A rookie has none, and
leaving him off is the worst option -- the top rookies sit inside Sleeper's top 60 and the
other seven managers will draft them. So a rookie is valued as **the players who played
his projected minutes, scaled down by a measured rookie discount**, and the board says plainly
that this is what it is.

Two things measured on 2023-26 data shaped this, so it does not get re-litigated:

* **Rookies score 10-15% less than veterans at the same minutes**, and with a lower spread
  by almost the same factor (30-36 min: rookies 32.4 mean / 11.8 sd, veterans 37.6 / 13.8).
  So the cohort is scaled by a measured discount rather than used as-is. A cohort of past
  rookies alone was tried first and failed: only three 2024-26 rookies averaged 30+ minutes,
  so it stopped rising with minutes.
* **Preseason minutes are a weak predictor of regular-season minutes** (r = 0.47, residual
  sd ~6.8 minutes; starters are capped in preseason, so the fit has slope ~0.9 and intercept
  ~7). One number would claim far more than that supports, so every rookie also carries a
  low/high value at the projection +/- one residual sd. Elite picks live at the high end:
  Flagg's rookie season was 43.8 fp at 33.5 minutes, above veterans at those minutes.

The cohort is built per *player-season*, selecting seasons whose average minutes are near the
projection and then using all of their games. Selecting individual games inside a minutes
window instead would throw away exactly the game-to-game variance Lock-In pays for.

What this cannot see is draft pedigree: it values the #1 pick and a second-rounder at the
same projected minutes identically. The board shows Sleeper's own rank beside ours, and the
high end of the range is where the elite picks have historically landed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.model.distributions import EmpiricalScoreDistribution, availability

# A preseason sample shorter than this is not a role signal.
MIN_PRESEASON_GAMES = 1
# Players needed in the past-rookie cohort before it is used.
MIN_COHORT_PLAYERS = 8
# Past rookies need this many games for their rookie season to describe a role.
MIN_COHORT_GAMES = 20
COHORT_HALF_WIDTH_MINUTES = 3.0


@dataclass(frozen=True)
class MinutesProjection:
    """Linear map from preseason minutes per game to regular-season minutes per game."""

    slope: float
    intercept: float
    residual_sd: float
    players: int

    def project(self, preseason_minutes: float) -> float:
        return float(self.slope * preseason_minutes + self.intercept)

    def describe(self) -> str:
        return (
            f"regular min = {self.slope:.2f} x preseason + {self.intercept:.1f} "
            f"(+/- {self.residual_sd:.1f}, n={self.players})"
        )


def fit_minutes_projection(
    regular_logs: pd.DataFrame,
    preseason_logs: pd.DataFrame,
    min_regular_games: int = 20,
    min_preseason_games: int = 2,
) -> MinutesProjection:
    """Fit regular-season minutes on the following preseason's minutes, across veterans.

    ``regular_logs`` is the last completed season; ``preseason_logs`` is the preseason that
    followed it. Only players with a real sample in both are used.
    """
    regular = regular_logs.groupby("nba_player_id")["minutes"].agg(["mean", "size"])
    regular = regular[regular["size"] >= min_regular_games]
    preseason = preseason_logs.groupby("nba_player_id")["minutes"].agg(["mean", "size"])
    preseason = preseason[preseason["size"] >= min_preseason_games]
    joined = regular.join(preseason, lsuffix="_reg", rsuffix="_pre", how="inner")
    if len(joined) < 10:
        raise ValueError(
            f"only {len(joined)} players have both a regular season and a preseason sample; "
            "too few to fit a minutes projection"
        )

    slope, intercept = np.polyfit(joined["mean_pre"], joined["mean_reg"], 1)
    residuals = joined["mean_reg"] - (slope * joined["mean_pre"] + intercept)
    return MinutesProjection(
        slope=float(slope),
        intercept=float(intercept),
        residual_sd=float(residuals.std(ddof=1)),
        players=len(joined),
    )


def rookie_seasons(logs: pd.DataFrame) -> pd.DataFrame:
    """Each past rookie's rookie-season games.

    A rookie is a player whose first appearance in ``logs`` is *after* its earliest season:
    someone first seen in the earliest season may simply be a veteran the window starts on.
    """
    seasons = sorted(logs["season"].unique())
    if len(seasons) < 2:
        raise ValueError("need at least two seasons of logs to identify rookies")
    first_season = logs.groupby("nba_player_id")["season"].min()
    rookies = first_season[first_season != seasons[0]]
    keyed = logs.merge(
        rookies.rename("rookie_season"), left_on="nba_player_id", right_index=True
    )
    return keyed[keyed["season"] == keyed["rookie_season"]].drop(columns="rookie_season")


def rookie_discount(logs: pd.DataFrame, rookie_logs: pd.DataFrame) -> float:
    """How much a rookie scores relative to a veteran playing the same minutes.

    For every rookie game, the veteran expectation is the veteran mean fp in the same
    whole-minute bin; the discount is total rookie fp over total expected fp. Measured at
    **0.908** on 2023-26 data. (Coarse 6-minute bands suggest ~0.86, but that overstates it:
    inside each band rookies sit at the low-minutes end.) The rookie *spread* is lower by
    about the same factor as the mean, which is why scaling the whole distribution is the
    right shape of adjustment.
    """
    rookie_keys = set(zip(rookie_logs["nba_player_id"], rookie_logs["season"]))
    is_rookie = pd.Series(
        [key in rookie_keys for key in zip(logs["nba_player_id"], logs["season"])],
        index=logs.index,
    )
    bins = logs["minutes"].round().astype(int)
    veteran_by_bin = logs[~is_rookie].groupby(bins[~is_rookie])["fp"].mean()
    expected = bins[is_rookie].map(veteran_by_bin)
    usable = expected.notna()
    actual = logs.loc[is_rookie, "fp"][usable].sum()
    return float(actual / expected[usable].sum())


def player_seasons(logs: pd.DataFrame, min_games: int = MIN_COHORT_GAMES) -> pd.DataFrame:
    """One row per (player, season) with ``min_games``+ games: mean minutes, games, p_play.

    Built once and shared, because ``availability`` is the slow part of the cohort.
    """
    sizes = logs.groupby(["nba_player_id", "season"])["minutes"].agg(["mean", "size"])
    sizes = sizes[sizes["size"] >= min_games]
    by_season = {season: frame for season, frame in logs.groupby("season")}
    p_plays = [
        availability(by_season[season], int(player_id))[0] for player_id, season in sizes.index
    ]
    return pd.DataFrame(
        {"minutes": sizes["mean"], "games": sizes["size"], "p_play": p_plays}, index=sizes.index
    )


def rookie_cohort_distribution(
    logs: pd.DataFrame,
    seasons: pd.DataFrame,
    projected_minutes: float,
    discount: float,
    half_width: float = COHORT_HALF_WIDTH_MINUTES,
    min_players: int = MIN_COHORT_PLAYERS,
) -> EmpiricalScoreDistribution:
    """A distribution for a rookie projected to play ``projected_minutes``.

    The cohort is every **player-season** (``min_games``+ games) whose average minutes are
    within ``half_width`` of the projection -- all of those players' games, scaled by the
    rookie ``discount``. Player-seasons rather than past rookies alone, because only three
    rookies in 2024-26 averaged 30+ minutes; a rookies-only cohort ran out at the top and
    stopped rising with minutes. Each player-season gets equal total weight. ``p_play`` is
    the cohort median, measured per stint like everyone else's.

    ``logs`` must carry ``fp``; ``seasons`` comes from :func:`player_seasons`.
    """
    if len(seasons) < min_players:
        raise ValueError(f"only {len(seasons)} player-seasons to build a cohort from")

    width = half_width
    while True:
        near = seasons[(seasons["minutes"] - projected_minutes).abs() <= width]
        if len(near) >= min_players:
            break
        width += 1.0

    keys = pd.MultiIndex.from_frame(logs[["nba_player_id", "season"]])
    cohort = logs[keys.isin(near.index)]
    games_per_season = cohort.groupby(["nba_player_id", "season"])["fp"].transform("size")

    return EmpiricalScoreDistribution(
        values=cohort["fp"].to_numpy(dtype=float) * discount,
        weights=(1.0 / games_per_season).to_numpy(dtype=float),
        _p_play=float(near["p_play"].median()),
        games_observed=0,
        history_reliable=False,
    )


def preseason_minutes(preseason_logs: pd.DataFrame) -> pd.DataFrame:
    """Per player: name, team, games and mean minutes this preseason."""
    ordered = preseason_logs.sort_values("game_date")
    summary = ordered.groupby("nba_player_id").agg(
        player_name=("player_name", "last"),
        team=("team", "last"),
        games=("minutes", "size"),
        minutes=("minutes", "mean"),
    )
    return summary[summary["games"] >= MIN_PRESEASON_GAMES].reset_index()
