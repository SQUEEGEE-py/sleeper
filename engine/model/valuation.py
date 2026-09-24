"""Replacement level and value over replacement (SPEC §5.3).

A player's season lock-in value is not what he is worth at the draft. What matters is how much
better he is than whoever you could have had instead *in the slot he would occupy*.

Replacement level is therefore **slot-specific**, computed by simulating a greedy league-wide
draft of the best players into the real slot structure and taking the marginal starter at each
slot type. SPEC §5.3 is explicit that a flat "player number 72" cutoff is wrong because it
undervalues centres: this league starts two of them on each of eight teams, so sixteen centres
hold starting jobs, and the sixteenth-best centre is usually a worse player than the
seventy-second-best player overall.

The consequence, which is the whole point: a centre is measured against a low bar and so earns
more value over replacement than a guard with the same raw production.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.model.lineup import SLOT_ELIGIBILITY, is_eligible, slot_position


@dataclass(frozen=True)
class PlayerValue:
    """One player as the board sees him."""

    player_id: int
    name: str
    team: str
    positions: tuple[str, ...]
    season_value: float
    weekly_value: float = 0.0
    mean: float = 0.0
    sd: float = 0.0
    p_play: float = 1.0
    games_observed: int = 0
    history_reliable: bool = True
    value_multiplier: float = 1.0
    note: str = ""


@dataclass(frozen=True)
class ReplacementLevels:
    """Replacement value per slot type, plus the draft that produced them."""

    by_slot: dict[str, float]
    drafted: dict[str, list[int]]  # slot type -> player ids that filled it
    starters_filled: int
    starter_slots: int

    @property
    def is_complete(self) -> bool:
        return self.starters_filled == self.starter_slots


def _slot_restrictiveness(position: str) -> int:
    """How few positions a slot accepts. Used to fill the pickiest slots first."""
    accepted = SLOT_ELIGIBILITY[position]
    return 99 if accepted is None else len(accepted)


def compute_replacement_levels(
    players: list[PlayerValue],
    starting_slots: list[str],
    teams: int,
) -> ReplacementLevels:
    """Greedily draft the whole league into its slots, then read off the marginal starter.

    Players are taken in value order. Each is placed in the **most restrictive** open slot he
    is eligible for, on the first team that has one -- a centre takes a C slot before he takes
    UTIL, which is what keeps UTIL from absorbing scarce positions and flattening the very
    differences this function exists to measure.

    Replacement for a slot type is the lowest value among the players who ended up starting
    there: the last man to hold a job, and therefore the bar a draftee must clear.
    """
    if teams < 1:
        raise ValueError("teams must be >= 1")

    slot_types = [slot_position(slot) for slot in starting_slots]
    # Open slots per type across the whole league.
    remaining = {
        position: slot_types.count(position) * teams for position in set(slot_types)
    }
    order = sorted(remaining, key=lambda position: (_slot_restrictiveness(position), position))

    filled: dict[str, list[float]] = {position: [] for position in remaining}
    drafted: dict[str, list[int]] = {position: [] for position in remaining}
    total_slots = sum(remaining.values())
    placed = 0

    for player in sorted(players, key=lambda p: -p.season_value):
        if placed >= total_slots:
            break
        for position in order:
            if remaining[position] <= 0:
                continue
            if not is_eligible(player.positions, position):
                continue
            remaining[position] -= 1
            filled[position].append(player.season_value)
            drafted[position].append(player.player_id)
            placed += 1
            break

    by_slot = {
        position: (min(values) if values else 0.0) for position, values in filled.items()
    }
    return ReplacementLevels(by_slot, drafted, placed, total_slots)


def replacement_for(
    positions: tuple[str, ...] | list[str] | None, levels: ReplacementLevels
) -> float:
    """The bar a given player has to clear.

    Measured against the **lowest replacement among his dedicated positional slots**, because
    that is where starting him displaces the weakest alternative and a rational manager slots
    him exactly there. Scarcity at centre then shows up as value: sixteen centre jobs exist
    leaguewide, so the marginal starting centre is a weaker player than the marginal starting
    point guard, and a centre's surplus over replacement is correspondingly larger.

    **UTIL is deliberately excluded from that minimum.** Everyone is UTIL-eligible and the
    greedy fills UTIL last, so the marginal UTIL starter is structurally the weakest starter
    in the league. Including it would hand every player the same number and reduce value over
    replacement to "season value minus a constant" -- precisely the flat player-number-72
    cutoff SPEC §5.3 says not to use. UTIL is still computed and reported, because it is the
    right yardstick for the last bench spot; it is just not what scarcity is measured against.
    """
    positional = [
        value
        for position, value in levels.by_slot.items()
        if position != "UTIL" and is_eligible(positions, position)
    ]
    if positional:
        return min(positional)

    # No dedicated slot fits him, so UTIL really is his ceiling -- fall back to it, or to the
    # hardest bar available if this roster structure has no UTIL at all.
    if "UTIL" in levels.by_slot:
        return levels.by_slot["UTIL"]
    return max(levels.by_slot.values(), default=0.0)


def value_over_replacement(player: PlayerValue, levels: ReplacementLevels) -> float:
    return player.season_value - replacement_for(player.positions, levels)


def build_board(
    players: list[PlayerValue],
    starting_slots: list[str],
    teams: int,
) -> tuple[list[tuple[PlayerValue, float, float]], ReplacementLevels]:
    """Rank every player by value over replacement.

    Returns ``[(player, vor, replacement), ...]`` sorted best first, plus the replacement
    levels so the UI can show what each player was measured against.
    """
    levels = compute_replacement_levels(players, starting_slots, teams)
    rows = [
        (player, value_over_replacement(player, levels), replacement_for(player.positions, levels))
        for player in players
    ]
    rows.sort(key=lambda row: -row[1])
    return rows, levels
