"""This week's head-to-head: win probability, and the lock-or-roll call that maximises it.

``lockin.py`` answers "what is this player worth?" by maximising his *expected* score. In a
matchup that is the wrong objective at the margin: a manager who is well behind should take
risks (roll on a decent score, hoping for a big one) and a manager who is well ahead should
bank them. So the nightly call is made here, by simulation:

* every starting slot of both teams is simulated through the rest of the week, each player
  following the expected-value threshold policy from ``lockin.lock_threshold``;
* for one of my players with an open decision (a finished game, not yet locked, more games to
  come), the week is simulated twice -- forced LOCK and forced ROLL -- and the call is
  whichever wins more often.

Both runs use **common random numbers**: each slot draws from its own seeded stream, so forcing
one slot leaves every other slot's week identical and the difference is all signal. When even
that difference is within two standard errors, the matchup does not change anything and the
expected-value call stands.

The opponent is assumed to follow the same threshold policy. They may not; it is the honest
default, and is said so in the report. Pure: no I/O, no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import numpy as np

from engine.model.lockin import ScoreDistribution, lock_threshold

DEFAULT_SIMS = 20_000
Force = Literal["lock", "roll"]


class SampleableDistribution(ScoreDistribution, Protocol):
    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Scores conditional on playing."""


@dataclass(frozen=True)
class Slot:
    """One starting slot, mid-week.

    ``locked``: a locked score -- fixed, nothing left to decide.
    ``pending``: the score of his latest finished game, not locked; the decision is still open
    until his next game tips off. 0.0 if he sat that game.
    ``remaining_p_play``: one probability per game this week that has not started, in order.
    """

    name: str
    distribution: SampleableDistribution | None
    remaining_p_play: tuple[float, ...] = ()
    locked: float | None = None
    pending: float | None = None

    def __post_init__(self) -> None:
        if self.locked is not None and self.pending is not None:
            raise ValueError(f"{self.name}: a slot is either locked or pending, not both")
        needs_model = self.locked is None and bool(self.remaining_p_play)
        if needs_model and self.distribution is None:
            raise ValueError(f"{self.name}: games left to simulate but no distribution")

    @property
    def decision_open(self) -> bool:
        """A finished, unlocked game with more games to come: LOCK or ROLL is a real choice."""
        return self.locked is None and self.pending is not None and bool(self.remaining_p_play)

    def threshold(self, games_after: int) -> float:
        """The expected-value lock threshold with ``games_after`` games still to come."""
        if games_after == 0:
            return float("-inf")
        later = self.remaining_p_play[len(self.remaining_p_play) - games_after:]
        return lock_threshold(self.distribution, games_after, later)  # type: ignore[arg-type]


def policy_lock(distribution: ScoreDistribution, completed_scores: Sequence[float],
                p_play_by_game: Sequence[float]) -> float | None:
    """The score an expected-value manager would have locked by now, or None if he would
    still be waiting.

    Replays the finished games in order: after game ``i``, lock if its score reaches the
    threshold for the games that were still to come then (``p_play_by_game[i + 1:]``, one
    probability per game of his whole week). Used for the **opponent**, whose locks Sleeper
    never shows (``locks.py``): the user asked to assume he locks optimally (2026-10-25).
    """
    if len(p_play_by_game) < len(completed_scores):
        raise ValueError("p_play_by_game must cover every game of the week")
    for index, score in enumerate(completed_scores):
        later = list(p_play_by_game[index + 1:])
        if not later:
            return None  # his last game: nothing to decide, it simply counts
        if score >= lock_threshold(distribution, len(later), later):
            return score
    return None


def slot_outcomes(slot: Slot, rng: np.random.Generator, n_sims: int, force: Force | None = None) -> np.ndarray:
    """The score this slot ends the week with, in each of ``n_sims`` simulated weeks."""
    if slot.locked is not None:
        return np.full(n_sims, slot.locked)
    remaining = slot.remaining_p_play
    if slot.pending is not None:
        if not remaining:
            if force == "roll":
                raise ValueError(f"{slot.name}: no games left to roll on to")
            return np.full(n_sims, slot.pending)
        if force == "lock" or (force is None and slot.pending >= slot.threshold(len(remaining))):
            return np.full(n_sims, slot.pending)
    elif force is not None:
        raise ValueError(f"{slot.name}: no open decision to force")
    if not remaining:
        return np.zeros(n_sims)  # no games this week at all

    result = np.zeros(n_sims)
    undecided = np.ones(n_sims, dtype=bool)
    for index, p_play in enumerate(remaining):
        plays = rng.random(n_sims) < p_play
        scores = np.where(plays, slot.distribution.sample(rng, n_sims), 0.0)  # type: ignore[union-attr]
        games_after = len(remaining) - index - 1
        if games_after == 0:
            # The final game is forced, and 0 if he sits -- never a fallback to an earlier one.
            result[undecided] = scores[undecided]
        else:
            lock = undecided & plays & (scores >= slot.threshold(games_after))
            result[lock] = scores[lock]
            undecided &= ~lock
    return result


def team_outcomes(
    slots: Sequence[Slot], n_sims: int, seed: int, team: int, force: dict[int, Force] | None = None
) -> np.ndarray:
    """Team totals per simulated week. Each slot has its own random stream (common random
    numbers), keyed by ``(seed, team, slot index)``."""
    force = force or {}
    streams = np.random.SeedSequence([seed, team]).spawn(len(slots))
    total = np.zeros(n_sims)
    for index, (slot, stream) in enumerate(zip(slots, streams)):
        total += slot_outcomes(slot, np.random.default_rng(stream), n_sims, force.get(index))
    return total


@dataclass(frozen=True)
class Odds:
    p_win: float
    p_tie: float
    my_mean: float
    their_mean: float
    margin_p10: float  # my total minus theirs, 10th percentile
    margin_p90: float


def _wins(mine: np.ndarray, theirs: np.ndarray) -> np.ndarray:
    """1 per simulated win, 0.5 per tie."""
    return (mine > theirs) + 0.5 * (mine == theirs)


def win_probability(
    mine: Sequence[Slot], theirs: Sequence[Slot], n_sims: int = DEFAULT_SIMS, seed: int = 0
) -> Odds:
    my_totals = team_outcomes(mine, n_sims, seed, team=0)
    their_totals = team_outcomes(theirs, n_sims, seed, team=1)
    margin = my_totals - their_totals
    return Odds(
        p_win=float(_wins(my_totals, their_totals).mean()),
        p_tie=float((margin == 0).mean()),
        my_mean=float(my_totals.mean()),
        their_mean=float(their_totals.mean()),
        margin_p10=float(np.percentile(margin, 10)),
        margin_p90=float(np.percentile(margin, 90)),
    )


@dataclass(frozen=True)
class Decision:
    slot: str
    pending: float
    threshold: float          # the expected-value threshold for this decision
    ev_call: Force            # what maximising his expected score says
    p_win_lock: float
    p_win_roll: float
    standard_error: float     # of p_win_lock - p_win_roll, paired
    call: Force               # what maximising the chance of winning the week says

    @property
    def matchup_changed_it(self) -> bool:
        return self.call != self.ev_call


def decide(
    mine: Sequence[Slot], theirs: Sequence[Slot], index: int, n_sims: int = DEFAULT_SIMS, seed: int = 0
) -> Decision:
    """LOCK or ROLL for ``mine[index]``, by the chance of winning the week."""
    slot = mine[index]
    if not slot.decision_open:
        raise ValueError(f"{slot.name}: no open decision")
    threshold = slot.threshold(len(slot.remaining_p_play))
    ev_call: Force = "lock" if slot.pending >= threshold else "roll"  # type: ignore[operator]
    their_totals = team_outcomes(theirs, n_sims, seed, team=1)
    wins = {
        force: _wins(team_outcomes(mine, n_sims, seed, team=0, force={index: force}), their_totals)
        for force in ("lock", "roll")
    }
    difference = wins["lock"] - wins["roll"]
    standard_error = float(difference.std(ddof=1) / np.sqrt(n_sims))
    gap = float(difference.mean())
    # <=, so a call that cannot change the result (gap and error both 0) keeps the EV call.
    call: Force = ev_call if abs(gap) <= 2 * standard_error else ("lock" if gap > 0 else "roll")
    return Decision(slot.name, float(slot.pending), threshold, ev_call,  # type: ignore[arg-type]
                    float(wins["lock"].mean()), float(wins["roll"].mean()), standard_error, call)
