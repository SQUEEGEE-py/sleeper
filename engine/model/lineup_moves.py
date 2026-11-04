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

**Auditions** (``plan_auditions``, user request 2026-11-04). A bench player whose game comes
*before* a starter's next game can take that starter's slot just for it: if he scores well,
lock him; if not, put the starter back before the starter tips off (or keep the bench player
in, if his own games left are worth more). Because a game counts only for whoever was starting
when it tipped, the starter loses nothing but any pending score he is showing -- so with no
pending score this is pure upside. Its value is one step of the stopping recursion with the
starter as the fallback: ``p * E[max(X, c)] + (1 - p) * c``, ``c = max(W_starter, W_rest)``.
Suggested at ``MIN_AUDITION_GAIN``. The swap back is the next report's lineup change (with its
own ``MIN_LINEUP_GAIN``, so within a point of the threshold it may just say lock or roll).

Pure: no network, no Sleeper types. ``nightly.py`` builds the options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

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

# Smallest gain worth an audition. Lower than a lineup change: with no pending score lost it
# cannot cost points, only the two taps in the app.
MIN_AUDITION_GAIN = 0.5

# On the same day, the bench player's game must be over before the starter's tips: a game
# takes about 2.5 hours, so the starter must tip at least this long after him.
SAME_DAY_GAP = timedelta(hours=3)
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
    sim: Slot | None = None  # his games left (and pending score), for auditions


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


def audition_value(bench: Slot, fallback: float) -> float:
    """A bench player started for his next game only. After it: lock his score, keep him in
    for his games left, or bring the starter back (``fallback``, the starter's value moved in
    again) -- whichever is worth most. One step of the stopping recursion."""
    first, rest = bench.remaining_p_play[0], bench.remaining_p_play[1:]
    rest_value = weekly_value(bench.distribution, len(rest), rest) if rest else 0.0  # type: ignore[arg-type]
    carry = max(fallback, rest_value)
    return first * bench.distribution.expected_max(carry) + (1.0 - first) * carry  # type: ignore[union-attr]


def plays_first(bench: LineupOption, starter: LineupOption) -> bool:
    """Is the bench player's next game over before the starter's next one tips?"""
    if bench.next_game is None or starter.next_game is None:
        return False
    if bench.next_game < starter.next_game:
        return True
    if bench.next_game == starter.next_game and bench.next_tip and starter.next_tip:
        return starter.next_tip - bench.next_tip >= SAME_DAY_GAP
    return False


@dataclass(frozen=True)
class Audition:
    slot: str
    starter: LineupOption
    bench: LineupOption
    value: float
    gain: float
    lock_at: float  # lock the bench player's score at this or more
    keep_bench: bool  # below it: keep him in (his games left beat the starter), else swap back

    def describe(self, when_bench: str, when_starter: str) -> str:
        position = slot_position(self.slot)
        text = (f"start {self.bench.name} at {position} for his {when_bench} game, in place of "
                f"{self.starter.name} (next game {when_starter}): lock him if he scores "
                f"{self.lock_at:.1f}+, ")
        if self.keep_bench:
            text += "otherwise keep him in -- his games left beat the starter's"
        else:
            text += f"otherwise put {self.starter.name} back before {when_starter}"
        pending = self.starter.sim.pending if self.starter.sim is not None else None
        cost = f"; gives up the {pending:g} {self.starter.name} is showing" if pending else "; nothing to lose"
        return text + f" (+{self.gain:.1f} expected{cost})"


def plan_auditions(plan: LineupPlan, bench: list[LineupOption],
                   min_gain: float = MIN_AUDITION_GAIN) -> list[Audition]:
    """Bench players worth starting for one game ahead of a starter who plays later, on top
    of ``plan`` (the lineup after its changes). Each bench player and slot used once, best
    gain first."""
    starting = {p.player_id for p in plan.assignment.values() if p is not None}
    fixed = {slot for slot, _ in plan.fixed}
    options = []
    for slot, starter in plan.assignment.items():
        if starter is None or slot in fixed or not starter.movable or starter.sim is None:
            continue
        fallback = move_in_value(starter.sim)
        for b in bench:
            if (b.player_id in starting or not b.movable or b.sim is None or not b.sim.remaining_p_play
                    or b.sim.distribution is None or not is_eligible(b.positions, slot)
                    or not plays_first(b, starter)):
                continue
            value = audition_value(b.sim, fallback)
            gain = value - starter.value
            if gain >= min_gain:
                rest = b.sim.remaining_p_play[1:]
                rest_value = weekly_value(b.sim.distribution, len(rest), rest) if rest else 0.0
                options.append(Audition(slot, starter, b, value, gain, max(fallback, rest_value),
                                        keep_bench=rest_value > fallback))
    chosen, used_slots, used_bench = [], set(), set()
    for a in sorted(options, key=lambda a: -a.gain):
        if a.slot not in used_slots and a.bench.player_id not in used_bench:
            chosen.append(a)
            used_slots.add(a.slot)
            used_bench.add(a.bench.player_id)
    never = datetime.max.replace(tzinfo=timezone.utc)
    return sorted(chosen, key=lambda a: (a.bench.next_tip or never, a.bench.next_game or date.max, a.slot))


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
