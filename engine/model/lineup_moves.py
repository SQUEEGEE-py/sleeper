"""Mid-week lineup changes: who should start for the rest of the week.

Lock-In (``docs/SPEC.md`` §3) decides what a player is worth to a slot from now on:

* a game counts only if he was **starting when it tipped**, so a bench player moved in is
  worth only the games he has left (``weekly_value`` over them; his earlier games are gone);
* a starter left in place keeps his open decision: he is worth ``max(pending, W_left)`` --
  lock the finished game he is showing, or roll on. Benching him throws that pending score
  away, which is why a starter who has already scored well is not benched for a bench player
  with a slightly higher roll-on value;
* a **locked** starter is frozen (locks freeze the slot), a starter whose week is over
  counts his final game automatically, and a player **in a game right now** cannot be moved
  until it ends -- all three are fixed;
* an Out player's next game is already priced at 0 in his ``remaining_p_play``
  (``nightly.build_starter``), so an injured starter is replaced through the same values.

The best lineup is an exact assignment over the movable slots (``scipy``'s
``linear_sum_assignment``, as ``lineup.assign_lineup``), with a small bonus for leaving a
player where he is so equal-value reshuffles are never suggested. A change is suggested only if
the whole lineup gains at least ``MIN_LINEUP_GAIN`` expected points. Expected points, not win
probability, picks the lineup; the caller shows what the change does to the chance of winning.

Pure: no network, no Sleeper types. ``nightly.py`` builds the options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import numpy as np
from scipy.optimize import linear_sum_assignment

from engine.model.lineup import is_eligible, slot_position
from engine.model.lockin import weekly_value
from engine.model.matchup import Slot

# Smallest gain, in expected fantasy points for the week, worth a lineup change. Below this the
# model cannot tell the lineups apart (a lock threshold moves more than that between runs).
MIN_LINEUP_GAIN = 1.0

# Bonus for keeping a player in the slot he holds. Far below any real difference, it only
# breaks ties, so two equal lineups never produce a "change".
_STAY_BONUS = 1e-3
_INELIGIBLE = 1e9


def stay_value(slot: Slot) -> float:
    """What a starter left in his slot is worth from now: lock what he shows, or roll on."""
    if slot.locked is not None:
        return slot.locked
    if not slot.remaining_p_play:
        return slot.pending or 0.0
    roll = weekly_value(slot.distribution, len(slot.remaining_p_play), slot.remaining_p_play)  # type: ignore[arg-type]
    return max(roll, slot.pending) if slot.pending is not None else roll


def move_in_value(slot: Slot) -> float:
    """What a bench player moved into the lineup now is worth: his games left, nothing earlier."""
    if not slot.remaining_p_play:
        return 0.0
    return weekly_value(slot.distribution, len(slot.remaining_p_play), slot.remaining_p_play)  # type: ignore[arg-type]


@dataclass(frozen=True)
class LineupOption:
    """One player who could fill a slot. ``value`` is his worth for the rest of the week if he
    starts from now (``stay_value`` for a starter, ``move_in_value`` for a bench player).
    ``movable`` False with a ``fixed_reason`` keeps him where he is."""

    player_id: str
    name: str
    positions: tuple[str, ...]
    value: float
    movable: bool = True
    fixed_reason: str = ""
    next_game: date | None = None
    next_tip: datetime | None = None  # UTC
    days: tuple[str, ...] = ()


@dataclass(frozen=True)
class SlotChange:
    slot: str
    out: LineupOption | None  # None: the slot was empty
    into: LineupOption
    from_bench: bool

    @property
    def deadline(self) -> tuple[datetime | None, date | None]:
        """The change must be made before either player's next game tips: the incoming one
        must be starting when his tips, the outgoing one must not be."""
        players = [p for p in (self.into, self.out) if p is not None]
        tips = [p.next_tip for p in players if p.next_tip is not None]
        days = [p.next_game for p in players if p.next_game is not None]
        return (min(tips) if tips else None, min(days) if days else None)

    def describe(self) -> str:
        position = slot_position(self.slot)
        days = f" ({', '.join(self.into.days)})" if self.into.days else ""
        if self.out is None:
            return f"start {self.into.name} at {position}{days} -- the slot is empty"
        if self.from_bench:
            return f"start {self.into.name} at {position}{days}, in place of {self.out.name}"
        return f"move {self.into.name} to {position}{days}, in place of {self.out.name}"


@dataclass
class LineupPlan:
    current_value: float
    best_value: float
    changes: list[SlotChange] = field(default_factory=list)
    to_bench: list[LineupOption] = field(default_factory=list)
    fixed: list[tuple[str, LineupOption]] = field(default_factory=list)  # (slot, player)
    assignment: dict[str, LineupOption | None] = field(default_factory=dict)  # every slot, after

    @property
    def gain(self) -> float:
        return self.best_value - self.current_value


def plan_lineup(slots: list[str], starters: list[LineupOption | None], bench: list[LineupOption],
                min_gain: float = MIN_LINEUP_GAIN) -> LineupPlan:
    """The best lineup for the rest of the week, and the changes from the current one.

    ``starters[i]`` holds ``slots[i]`` now (None: the slot is empty). Fixed starters keep their
    slot; every other slot is re-filled from the movable starters and bench players. Changes
    are returned only if the lineup gains ``min_gain`` or more.
    """
    if len(slots) != len(starters):
        raise ValueError("one starter (or None) per slot")
    fixed = [(slot, s) for slot, s in zip(slots, starters) if s is not None and not s.movable]
    open_slots = [i for i, s in enumerate(starters) if s is None or s.movable]
    current = {slots[i]: starters[i] for i in open_slots}
    current_value = sum(s.value for s in current.values() if s is not None)
    candidates = [s for s in current.values() if s is not None] + [b for b in bench if b.movable]

    plan = LineupPlan(current_value, current_value, fixed=fixed,
                      assignment={slot: s for slot, s in zip(slots, starters)})
    if not open_slots or not candidates:
        return plan

    cost = np.full((len(candidates), len(open_slots)), _INELIGIBLE)
    for row, player in enumerate(candidates):
        for column, index in enumerate(open_slots):
            if is_eligible(player.positions, slots[index]):
                stays = current[slots[index]] is player
                cost[row, column] = -(player.value + (_STAY_BONUS if stays else 0.0))
    rows, columns = linear_sum_assignment(cost)
    best: dict[str, LineupOption | None] = {slots[i]: None for i in open_slots}
    for row, column in zip(rows, columns):
        if cost[row, column] < _INELIGIBLE:
            best[slots[open_slots[column]]] = candidates[row]
    best_value = sum(p.value for p in best.values() if p is not None)
    plan.best_value = best_value
    if best_value - current_value < min_gain:
        return plan

    starting_now = {s.player_id for s in current.values() if s is not None}
    starting_after = {p.player_id for p in best.values() if p is not None}
    for slot, player in best.items():
        before = current[slot]
        if player is None or (before is not None and before.player_id == player.player_id):
            continue
        plan.changes.append(SlotChange(slot, before, player, from_bench=player.player_id not in starting_now))
    plan.to_bench = [s for s in current.values() if s is not None and s.player_id not in starting_after]
    plan.assignment.update(best)
    never = datetime.max.replace(tzinfo=timezone.utc)
    plan.changes.sort(key=lambda c: (c.deadline[0] or never, c.deadline[1] or date.max, c.slot))
    return plan
