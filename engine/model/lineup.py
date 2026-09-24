"""Optimal slot assignment (SPEC §5.4).

Given a roster and a value per player, decide who starts where. This is a max-weight bipartite
matching, solved exactly with ``scipy.optimize.linear_sum_assignment`` -- greedy assignment
gets it wrong in exactly the cases that matter, like a player who is the best option at two
slots while someone else can only fill one of them.

Used for roster valuation in the draft simulator (SPEC §6.2) and for weekly lineup advice.

Slot *names* come from ``config/league.json``. What each slot accepts is a property of the
Sleeper platform rather than of this league, so it lives here as a constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

# Which listed positions each starting slot accepts. G is any guard, F any forward, UTIL
# anyone. Bench and IR are not starting slots and never appear here.
SLOT_ELIGIBILITY: dict[str, frozenset[str] | None] = {
    "PG": frozenset({"PG"}),
    "SG": frozenset({"SG"}),
    "G": frozenset({"PG", "SG"}),
    "SF": frozenset({"SF"}),
    "PF": frozenset({"PF"}),
    "F": frozenset({"SF", "PF"}),
    "C": frozenset({"C"}),
    "UTIL": None,  # None means "any position"
}

NON_STARTING_SLOTS = frozenset({"BN", "IR", "TAXI"})

# Large enough that the optimiser never prefers an ineligible pairing, small enough to stay
# far from floating point trouble.
_INELIGIBLE_COST = 1e9


@dataclass(frozen=True)
class LineupAssignment:
    """Who starts where, and what it is worth."""

    assignments: dict[str, int]  # slot label -> player index
    total_value: float
    unfilled_slots: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return not self.unfilled_slots


def starting_slots(roster_positions: list[str]) -> list[str]:
    """The starting slots from ``config/league.json``'s roster_positions, bench dropped.

    Returned with duplicates intact and labelled, so a two-centre league yields ``C#1`` and
    ``C#2`` rather than one slot that silently swallows both.
    """
    slots = []
    seen: dict[str, int] = {}
    for position in roster_positions:
        if position in NON_STARTING_SLOTS:
            continue
        if position not in SLOT_ELIGIBILITY:
            raise ValueError(
                f"roster position {position!r} has no eligibility rule; add it to "
                "SLOT_ELIGIBILITY deliberately rather than letting it be ignored"
            )
        seen[position] = seen.get(position, 0) + 1
        slots.append(f"{position}#{seen[position]}")
    return slots


def slot_position(slot: str) -> str:
    """'C#2' -> 'C'."""
    return slot.split("#")[0]


def is_eligible(positions: list[str] | tuple[str, ...] | None, slot: str) -> bool:
    """Can a player with these listed positions fill this slot?"""
    accepted = SLOT_ELIGIBILITY[slot_position(slot)]
    if accepted is None:
        return True
    if not positions:
        return False
    return any(position in accepted for position in positions)


def assign_lineup(
    values: list[float] | np.ndarray,
    positions: list[list[str] | tuple[str, ...] | None],
    slots: list[str],
) -> LineupAssignment:
    """Max-weight assignment of players to starting slots.

    ``values[i]`` and ``positions[i]`` describe player ``i``. Players not assigned are on the
    bench. A slot with nobody eligible is reported in ``unfilled_slots`` rather than being
    filled by an ineligible player -- an incomplete lineup is a real state in this league
    (two centre slots, eight teams, sixteen centres) and the caller needs to know.
    """
    if len(values) != len(positions):
        raise ValueError("values and positions must be the same length")
    for slot in slots:
        if slot_position(slot) not in SLOT_ELIGIBILITY:
            raise ValueError(f"unknown slot {slot!r}")

    if not slots:
        return LineupAssignment({}, 0.0, ())
    if len(values) == 0:
        return LineupAssignment({}, 0.0, tuple(slots))

    # linear_sum_assignment minimises, so cost is negated value. Ineligible pairings get a
    # large positive cost instead of infinity, which would make the problem unsolvable
    # whenever a slot has no eligible player at all.
    cost = np.full((len(values), len(slots)), _INELIGIBLE_COST, dtype=float)
    for player_index, player_positions in enumerate(positions):
        for slot_index, slot in enumerate(slots):
            if is_eligible(player_positions, slot):
                cost[player_index, slot_index] = -float(values[player_index])

    rows, columns = linear_sum_assignment(cost)

    assignments: dict[str, int] = {}
    filled: set[str] = set()
    total = 0.0
    for player_index, slot_index in zip(rows, columns):
        if cost[player_index, slot_index] >= _INELIGIBLE_COST:
            continue  # the optimiser had to pad; this pairing is not real
        slot = slots[slot_index]
        assignments[slot] = int(player_index)
        filled.add(slot)
        total += float(values[player_index])

    unfilled = tuple(slot for slot in slots if slot not in filled)
    return LineupAssignment(assignments, total, unfilled)


def lineup_value(
    values: list[float] | np.ndarray,
    positions: list[list[str] | tuple[str, ...] | None],
    slots: list[str],
) -> float:
    """The value of the best legal starting lineup. The objective the draft simulator
    maximises (``EV_roster`` in SPEC §6.2)."""
    return assign_lineup(values, positions, slots).total_value


def marginal_value_to_roster(
    values: list[float] | np.ndarray,
    positions: list[list[str] | tuple[str, ...] | None],
    slots: list[str],
    candidate_value: float,
    candidate_positions: list[str] | tuple[str, ...] | None,
) -> float:
    """How much adding this player would improve the best starting lineup.

    This, not raw value, is what a pick is worth to a roster: a fourth good centre adds almost
    nothing once both centre slots and UTIL are full. It is also the quantity SPEC §6.2's
    ``denial`` is built from -- the drop an opponent suffers when a player is taken from them.
    """
    before = lineup_value(values, positions, slots)
    after = lineup_value(
        [*list(values), candidate_value], [*positions, candidate_positions], slots
    )
    return after - before
