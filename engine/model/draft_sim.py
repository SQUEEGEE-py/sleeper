"""Live draft advice: who survives to your next pick, and who to take now (SPEC §6.2, cut down).

This is the version built for the 2026-10-18 draft, not the full SPEC §6.2 simulator. What it
does, and why each simplification is acceptable:

**Opponent model: ADP with lognormal noise.** Each simulation draws, for every available
player, a noisy draft position ``ADP x exp(sigma * z)``; the opponents collectively take the
players with the smallest draws. Calibrated on this league's own 2025 draft against Sleeper's
2025 ADP: Spearman 0.948 between ADP and actual pick, and ``sd(log(pick / ADP))`` = 0.21,
nearly constant across rounds (0.21 / 0.24 / 0.19 for picks 1-24 / 25-56 / 57-104) -- which is
exactly what a multiplicative noise model assumes.

**sigma is 0.30, not 0.21**, and that was measured, not guessed. Replaying the 2025 draft and
predicting, at every pick, which players last to that team's next pick: sigma 0.21 is
overconfident (players given 3% actually lasted 11% of the time), while 0.30 is calibrated in
every bin -- predicted 0.04 / 0.21 / 0.41 / 0.61 / 0.82 / 0.97 against actual 0.05 / 0.23 /
0.36 / 0.60 / 0.83 / 0.98 -- and has the best Brier score of 0.15-0.40 (0.0978). The spread of
drafted players' picks understates the noise in the *order*, because it never sees the
players who went undrafted or the ones who slid. One draft, in-sample, one parameter.

No positional-need term and no per-coach updating: the SPEC wants both, and both are deferred.
(cyapp1 reached ~25% ahead of ADP all draft in 2025 -- the largest per-coach effect, and the
first thing to add.)

**Value to the user's roster, in VOR units.** A player's worth to *this* roster is how much he
raises the best starting lineup's total value over replacement, with an empty slot worth 0
(it can be filled at replacement level). While a matching slot is open that is just his board
VOR; once his positions are full of better players it shrinks to nothing. Working in VOR units
is deliberate: modelling empty slots as replacement-level phantom players instead lets every
phantom chain into UTIL, and value collapses to "season value minus the UTIL bar" -- the flat
cutoff M4 fixed (see ``valuation.replacement_for``). VOR is clipped at 0 inside the lineup so
a below-replacement player is never *forced* into a slot and charged for it.

**Recommendation: one-step lookahead.** For each leading candidate, the score is his gain now
plus the expected gain of the best player still there at the user's next pick, averaged over
the simulations. That is what turns "who is best" into "who is best to take *now*": a player
likely to come back is worth less to take now than his raw value says.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.model.lineup import lineup_value

# Best-calibrated noise on this league's 2025 draft. See the module docstring.
DEFAULT_SIGMA = 0.30
DEFAULT_SIMS = 2000
# A player Sleeper gives no ADP to is, as far as the other managers' draft room goes, not on
# the list. He still gets drafted occasionally, so he is placed far down rather than removed.
# This number never touches the user's own valuation, only the opponent model.
UNRANKED_ADP = 400.0
# Candidates given the full lookahead, and how deep the "best next pick" search goes.
LOOKAHEAD_CANDIDATES = 12
LOOKAHEAD_DEPTH = 60


# --- snake arithmetic ---------------------------------------------------------------------


def slot_for_pick(pick_no: int, teams: int) -> int:
    """Draft slot (1-based) on the clock at overall pick ``pick_no`` (1-based), snake order."""
    if pick_no < 1:
        raise ValueError("pick_no is 1-based")
    round_index, position = divmod(pick_no - 1, teams)
    return position + 1 if round_index % 2 == 0 else teams - position


def snake_pick_numbers(slot: int, teams: int, rounds: int) -> list[int]:
    """Every overall pick number belonging to ``slot``."""
    if not 1 <= slot <= teams:
        raise ValueError(f"slot {slot} outside 1..{teams}")
    return [
        pick
        for pick in range(1, teams * rounds + 1)
        if slot_for_pick(pick, teams) == slot
    ]


@dataclass(frozen=True)
class DraftClock:
    """Where the user stands, given how many picks have been made."""

    picks_made: int
    on_clock: int | None  # overall pick now on the clock; None when the draft is over
    my_turn: bool
    my_next_pick: int | None  # the user's next pick (the current one if it is his turn)
    my_following_pick: int | None  # the one after that
    picks_until_next: int  # opponents' picks before ``my_next_pick``
    picks_between: int  # opponents' picks between ``my_next_pick`` and the following one


def draft_clock(slot: int, teams: int, rounds: int, picks_made: int) -> DraftClock:
    total = teams * rounds
    mine = [pick for pick in snake_pick_numbers(slot, teams, rounds) if pick > picks_made]
    on_clock = picks_made + 1 if picks_made < total else None
    next_pick = mine[0] if mine else None
    following = mine[1] if len(mine) > 1 else None
    return DraftClock(
        picks_made=picks_made,
        on_clock=on_clock,
        my_turn=on_clock is not None and on_clock == next_pick,
        my_next_pick=next_pick,
        my_following_pick=following,
        picks_until_next=(next_pick - picks_made - 1) if next_pick else 0,
        picks_between=(following - next_pick - 1) if following else 0,
    )


# --- survival -----------------------------------------------------------------------------


def simulate_ranks(
    adp: np.ndarray, sigma: float, n_sims: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_sims, players)`` rank of each player in the opponents' collective order, 0 first."""
    adp = np.asarray(adp, dtype=float)
    if np.any(adp <= 0):
        raise ValueError("ADP must be positive")
    noisy = np.log(adp)[None, :] + sigma * rng.standard_normal((n_sims, len(adp)))
    return np.argsort(np.argsort(noisy, axis=1), axis=1)


def survival(ranks: np.ndarray, picks_before: int) -> np.ndarray:
    """P(player is still there after ``picks_before`` opponent picks)."""
    return (ranks >= picks_before).mean(axis=0)


def survives_if_taken(ranks: np.ndarray, taken: int, picks_before: int) -> np.ndarray:
    """``(n_sims, players)`` survival given the user removes player ``taken`` first.

    Opponents walk the same noisy order with ``taken`` gone, so anyone ranked behind him moves
    up a place. ``taken`` himself is marked as not surviving.
    """
    shifted = ranks - (ranks[:, [taken]] < ranks).astype(int)
    alive = shifted >= picks_before
    alive[:, taken] = False
    return alive


# --- roster value ---------------------------------------------------------------------------


def roster_gain(
    roster_vor: list[float],
    roster_positions: list[tuple[str, ...]],
    slots: list[str],
    candidate_vor: float,
    candidate_positions: tuple[str, ...],
) -> float:
    """How much the candidate raises the roster's best starting lineup, in VOR units."""
    clipped = [max(value, 0.0) for value in roster_vor]
    before = lineup_value(clipped, roster_positions, slots)
    after = lineup_value(
        [*clipped, max(candidate_vor, 0.0)], [*roster_positions, candidate_positions], slots
    )
    return after - before


@dataclass(frozen=True)
class Advice:
    index: int  # into the available arrays
    gain: float  # lineup VOR gain if taken now
    p_survive: float  # to the user's next pick (or following pick, when it is his turn)
    score: float  # gain now + expected gain at the following pick


def advise(
    vor: np.ndarray,
    positions: list[tuple[str, ...]],
    adp: np.ndarray,
    roster_vor: list[float],
    roster_positions: list[tuple[str, ...]],
    slots: list[str],
    clock: DraftClock,
    sigma: float = DEFAULT_SIGMA,
    n_sims: int = DEFAULT_SIMS,
    rng: np.random.Generator | None = None,
    candidates: int = LOOKAHEAD_CANDIDATES,
    depth: int = LOOKAHEAD_DEPTH,
) -> list[Advice]:
    """Rank the available players for the user. Arrays describe available players only.

    When it is the user's turn, ``p_survive`` is to his *following* pick and the lookahead
    scores what he would have then. When it is not, ``p_survive`` is to his *next* pick --
    "will he still be there when I am up" -- and the score is the gain alone.
    """
    rng = rng or np.random.default_rng()
    vor = np.asarray(vor, dtype=float)
    adp = np.asarray(adp, dtype=float)
    n = len(vor)
    if n == 0:
        return []

    gains = np.array(
        [roster_gain(roster_vor, roster_positions, slots, vor[i], positions[i]) for i in range(n)]
    )
    # Ties (everyone 0 once the lineup is full) fall back to board VOR: best player available.
    order = np.lexsort((-vor, -gains))

    if clock.my_turn:
        horizon = clock.picks_between
        has_following = clock.my_following_pick is not None
    else:
        horizon = clock.picks_until_next
        has_following = False

    ranks = simulate_ranks(adp, sigma, n_sims, rng)
    p_survive = survival(ranks, horizon)

    scores = gains.copy()
    if has_following:
        pool = order[: max(depth, candidates)]
        for i in order[:candidates]:
            # Gains next time, given he is on the roster.
            next_roster_vor = [*roster_vor, vor[i]]
            next_roster_positions = [*roster_positions, positions[i]]
            next_gains = np.array(
                [
                    roster_gain(next_roster_vor, next_roster_positions, slots, vor[j], positions[j])
                    if j != i
                    else -np.inf
                    for j in pool
                ]
            )
            alive = survives_if_taken(ranks, int(i), horizon)[:, pool]
            best_first = np.argsort(-next_gains)
            alive_sorted = alive[:, best_first]
            first_alive = np.argmax(alive_sorted, axis=1)
            any_alive = alive_sorted.any(axis=1)
            best_next = np.where(any_alive, next_gains[best_first][first_alive], 0.0)
            scores[i] = gains[i] + float(best_next.mean())
        # Players outside the lookahead keep their bare gain, which can only rank them lower.
        outside = np.setdiff1d(np.arange(n), order[:candidates])
        scores[outside] = -np.inf

    ranked = sorted(range(n), key=lambda i: (-scores[i], -gains[i], -vor[i]))
    return [
        Advice(index=i, gain=float(gains[i]), p_survive=float(p_survive[i]), score=float(scores[i]))
        for i in ranked
    ]
