"""Roster management (phase 3 step 7): IR moves and free-agent pickups. Pure: no I/O.

The tool never makes a move -- Sleeper's API is read-only -- so everything here is an alert the
user acts on in the app.

**IR.** The league has 2 IR slots (``config["ir"]``), open to the statuses its
``reserve_allow_*`` settings allow: OUT, SUS and DNR. Probed 2026-10-22, Sleeper's NBA
``injury_status`` values in use were ``Out``, ``GTD``, ``DTD`` and ``IR``:

* ``Out`` -> OUT (eligible);
* ``GTD`` (game-time decision) and ``DTD`` (day-to-day) are not eligible, matching the league's
  ``reserve_allow_dtd = 0``;
* ``IR`` is Sleeper's own long-term designation. Treated as eligible -- Sleeper's IR slot is
  for exactly these players -- but not confirmable from the settings, so alerts say so;
* suspension and DNR spellings had no live example; ``Sus``/``Suspended`` and ``DNR`` are
  mapped, and **any other value is reported as unrecognised, never guessed**.

A roster's ``players`` list includes its IR players (roster 7: 14 players, one in
``reserve``), so active players are ``players`` minus ``reserve``.

**Pickups (reworked 2026-11-15).** A pickup is valued by what it does to the user's
*expected* starting lineup over a week in which anyone might be out, in lock-in points per week
(``season_value``, the board's weekly value). The earlier rule -- the free agent's VOR minus the
weakest bench player's -- had two blind spots, both live on 2026-11-15: it assumed every add
needs a drop, so with an open roster spot (12 active, two on IR) it reported "nothing beats
your bench"; and it valued bench players only as starters, so the one thing a bench player is
for -- cover when a starter is out -- counted for nothing. That roster had two C slots and
exactly two C-eligible players.

* Each active player misses a whole week with probability ``week_out_probability(p_play)``,
  measured on 2023-26 rotation players (20+ min), per team stint: 8.4% of 17,922
  player-weeks, steeply by availability (table below). Absences are treated as independent,
  and sets of up to ``MAX_ABSENT`` absent players are enumerated exactly (more than two absent
  at once is about 1% of weeks and is renormalised away). Deterministic, no sampling.
* The lineup each week is the best legal one (exact assignment); an empty slot scores 0.
* An add into an open roster spot is compared with the roster as it is; with no open spot,
  with dropping each bench player in turn, and the best drop is named. Starters are never
  offered as drops: the matchup advice is built on them.
* With an open spot an add costs nothing but a later drop, so its bar is
  ``OPEN_SPOT_MIN_GAIN``; a swap must clear ``PICKUP_MIN_GAIN``.

**Position cover.** ``cover_alerts`` reports a starting slot that would sit empty in
``COVER_ALERT_P`` or more of weeks -- no eligible backup -- with the players who can fill it.

Reverse-standings priority means a winning claim costs nothing later, so a claim is only ever
worth making; the open question per player is whether he is on waivers (claim, decided by
priority) or a free agent (add now, before his next game tips, after which he goes back on
waivers).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations
from typing import Literal, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from engine.model.lineup import is_eligible, slot_position

SLEEPER_TO_IR_CODE = {"Out": "OUT", "Sus": "SUS", "Suspended": "SUS", "DNR": "DNR"}
NOT_IR_ELIGIBLE = frozenset({"GTD", "DTD", "Questionable", "Doubtful", "Probable"})
SLEEPER_IR_DESIGNATION = "IR"
PICKUP_MIN_GAIN = 1.0  # expected lineup points per week for a swap (add + drop), as trades.MIN_GAIN
OPEN_SPOT_MIN_GAIN = 0.25  # an add into an empty roster spot costs only a later drop
COVER_ALERT_P = 0.02  # a slot empty in 1 week in 50 is worth saying
MAX_ABSENT = 2
# (p_play, share of weeks he played none of his team's games): 2023-26 rotation players, by
# p_play bin (<0.7, 0.7-0.8, 0.8-0.9, 0.9+; 80/84/149/83 players), x = the bin's mean p_play.
# An absent week is followed by another 55% of the time, against 4% after a week he played.
WEEK_OUT_BY_P_PLAY = ((0.573, 0.204), (0.758, 0.099), (0.851, 0.051), (0.948, 0.010))

IREligible = Literal["yes", "no", "assumed", "unknown"]


def unavailable(injury_status: str | None) -> bool:
    """Out, suspended, DNR or on Sleeper's IR: not playing, whatever the league's IR rules."""
    return injury_status in SLEEPER_TO_IR_CODE or injury_status == SLEEPER_IR_DESIGNATION


def ir_eligibility(injury_status: str | None, allowed: Sequence[str]) -> IREligible:
    """Whether a player with this Sleeper ``injury_status`` can sit in an IR slot."""
    if injury_status is None or injury_status in NOT_IR_ELIGIBLE:
        return "no"
    if injury_status == SLEEPER_IR_DESIGNATION:
        return "assumed"
    code = SLEEPER_TO_IR_CODE.get(injury_status)
    if code is None:
        return "unknown"
    return "yes" if code in allowed else "no"


@dataclass(frozen=True)
class RosterPlayer:
    player_id: str
    name: str
    injury_status: str | None


@dataclass(frozen=True)
class IRAlert:
    kind: Literal["stash", "activate", "full", "unrecognised"]
    player: str
    text: str


def ir_alerts(active: Sequence[RosterPlayer], reserve: Sequence[RosterPlayer], slots: int,
              allowed: Sequence[str]) -> list[IRAlert]:
    """What to do with IR right now. ``active`` excludes players already in ``reserve``."""
    alerts: list[IRAlert] = []
    for player in reserve:
        if ir_eligibility(player.injury_status, allowed) == "no":
            status = player.injury_status or "healthy"
            alerts.append(IRAlert("activate", player.name,
                f"Activate {player.name} from IR: his status is {status}, so he no longer "
                "qualifies. Sleeper restricts roster moves while an ineligible player sits in IR."))
    free = max(slots - len(reserve), 0)
    for player in active:
        eligible = ir_eligibility(player.injury_status, allowed)
        if eligible == "unknown":
            alerts.append(IRAlert("unrecognised", player.name,
                f"{player.name}: Sleeper status '{player.injury_status}' is not one this tool "
                "knows -- check in the app whether he can go to IR."))
            continue
        if eligible == "no":
            continue
        caveat = (" (Sleeper's 'IR' designation -- assumed eligible; the app will refuse if not)"
                  if eligible == "assumed" else "")
        if free > 0:
            free -= 1
            alerts.append(IRAlert("stash", player.name,
                f"Move {player.name} ({player.injury_status}) to IR and add a free agent -- "
                f"no drop needed{caveat}."))
        else:
            alerts.append(IRAlert("full", player.name,
                f"{player.name} ({player.injury_status}) could go to IR, but both IR slots are "
                f"full{caveat}."))
    return alerts


def waiver_reason(player_id: str, team: str | None, dropped_at: dict[str, datetime], now: datetime,
                  clear_days: int, teams_started_today: set[str], after_game_start: bool) -> str | None:
    """Why a player is on waivers right now, or None for a free agent.

    Two ways onto waivers, both from ``config["waivers"]``: dropped within ``clear_days``, and
    -- with ``after_game_start`` -- his team's game today has tipped, until the next waiver run
    (about 3 am ET, one observation). Sleeper exposes neither directly; both are inferred from
    transactions and the schedule, so the alert says "claim" and the app has the last word.
    """
    dropped = dropped_at.get(player_id)
    if dropped is not None and now - dropped < timedelta(days=clear_days):
        return f"dropped {dropped:%a %H:%M} UTC, on waivers for {clear_days} day(s)"
    if after_game_start and team in teams_started_today:
        return "his game today has started, so he is on waivers until they next run"
    return None


def week_out_probability(p_play: float) -> float:
    """Chance he misses a whole week, from his per-game availability (measured table above;
    flat beyond its ends)."""
    xs, ys = zip(*WEEK_OUT_BY_P_PLAY)
    return float(np.interp(p_play, xs, ys))


@dataclass(frozen=True)
class DepthPlayer:
    player_id: str
    name: str
    positions: tuple[str, ...]
    value: float  # weekly lock-in value as a starter (``season_value``)
    week_out: float


@dataclass(frozen=True)
class LineupOutlook:
    expected: float  # expected best-lineup value over a week, absences included
    healthy: float  # with everyone available
    p_unfilled: dict[str, float]  # by slot position ("C", "G"...), the chance any such slot sits empty


def lineup_outlook(players: Sequence[DepthPlayer], slots: Sequence[str],
                   max_absent: int = MAX_ABSENT) -> LineupOutlook:
    """Expected best lineup when each player is out for the week with his own probability."""
    n = len(players)
    if n == 0 or not slots:
        return LineupOutlook(0.0, 0.0, {slot_position(slot): 1.0 for slot in slots})
    eligible = np.array([[is_eligible(p.positions, slot) for slot in slots] for p in players])
    values = np.array([p.value for p in players])
    q = np.clip(np.array([p.week_out for p in players]), 0.0, 1.0)

    def best(available: np.ndarray) -> tuple[float, np.ndarray]:
        idx = np.flatnonzero(available)
        filled = np.zeros(len(slots), dtype=bool)
        if len(idx) == 0:
            return 0.0, filled
        gain = np.where(eligible[idx], values[idx, None], 0.0)
        ok = eligible[idx]
        rows, cols = linear_sum_assignment(np.where(ok, -gain, 1e9))
        real = ok[rows, cols]
        filled[cols[real]] = True
        return float(gain[rows[real], cols[real]].sum()), filled

    positions = sorted({slot_position(slot) for slot in slots})
    of_position = np.array([[slot_position(slot) == pos for slot in slots] for pos in positions])
    total = weight = 0.0
    unfilled = np.zeros(len(positions))
    healthy = None
    for k in range(min(max_absent, n) + 1):
        for out in combinations(range(n), k):
            mask = np.ones(n, dtype=bool)
            mask[list(out)] = False
            w = float(np.prod(np.where(mask, 1.0 - q, q)))
            value, filled = best(mask)
            if k == 0:
                healthy = value
            total += w * value
            unfilled += w * (of_position & ~filled).any(axis=1)
            weight += w
    return LineupOutlook(total / weight, float(healthy), dict(zip(positions, unfilled / weight)))


@dataclass(frozen=True)
class Candidate:
    player_id: str
    name: str
    team: str
    positions: tuple[str, ...]
    value: float  # weekly lock-in value (``season_value``)
    p_play: float
    injury_status: str | None = None
    on_waivers: bool = False
    waiver_note: str = ""
    # This season so far, shown beside the value so a stale number can be seen for what it is.
    games_now: int = 0
    minutes_now: float | None = None
    fp_now: float | None = None

    def depth_player(self) -> DepthPlayer:
        return DepthPlayer(self.player_id, self.name, self.positions, self.value,
                           week_out_probability(self.p_play))

    def season_line(self) -> str:
        if not self.games_now or self.minutes_now is None or self.fp_now is None:
            return "no games this season"
        return f"this season {self.games_now} g, {self.minutes_now:.1f} min, {self.fp_now:.1f} fp"


@dataclass(frozen=True)
class Pickup:
    candidate: Candidate
    drop: str | None  # None: into an open roster spot
    gain: float  # expected lineup points per week, absences included
    healthy_gain: float  # with everyone available

    def describe(self) -> str:
        c = self.candidate
        if c.on_waivers:
            how = f"claim on waivers ({c.waiver_note})" if c.waiver_note else "claim on waivers"
        else:
            how = "add now (free agent), before his next game tips"
        injury = f", {c.injury_status}" if c.injury_status else ""
        where = "into your open roster spot" if self.drop is None else f"dropping {self.drop}"
        why = ("as a starter" if self.healthy_gain >= self.gain - 1e-9 else
               f"+{self.healthy_gain:.1f} with everyone healthy, the rest as cover")
        return (f"{c.name} ({c.team}, {'/'.join(c.positions)}{injury}): +{self.gain:.1f}/week {where} "
                f"({why}; {c.season_line()}) -- {how}")


def rank_pickups(candidates: Sequence[Candidate], active: Sequence[DepthPlayer], droppable: set[str],
                 slots: Sequence[str], open_spots: int, min_gain: float = PICKUP_MIN_GAIN,
                 open_min_gain: float = OPEN_SPOT_MIN_GAIN, limit: int | None = 5) -> list[Pickup]:
    """Free agents that raise the expected lineup, best first.

    ``active``: the user's active players (IR excluded). ``droppable``: player ids he could
    release -- his bench. With ``open_spots`` > 0 an add needs no drop; otherwise each bench
    player is tried as the drop and the best is named. OUT free agents are left out: a pickup
    is for games, and he has none to give.
    """
    base = lineup_outlook(active, slots)
    pickups = []
    for c in candidates:
        if unavailable(c.injury_status):
            continue
        new = c.depth_player()
        if open_spots > 0:
            after = lineup_outlook([*active, new], slots)
            options = [(None, after, open_min_gain)]
        else:
            options = []
            for drop in active:
                if drop.player_id in droppable:
                    rest = [p for p in active if p.player_id != drop.player_id]
                    options.append((drop.name, lineup_outlook([*rest, new], slots), min_gain))
        scored = [(after.expected - base.expected, after.healthy - base.healthy, drop, bar)
                  for drop, after, bar in options]
        if not scored:
            continue
        gain, healthy, drop, bar = max(scored, key=lambda t: t[0])
        if gain >= bar:
            pickups.append(Pickup(c, drop, gain, healthy))
    return sorted(pickups, key=lambda p: -p.gain)[:limit]  # limit=None: all of them


@dataclass(frozen=True)
class CoverAlert:
    position: str  # the slot's position, e.g. "C"
    p_empty: float  # chance in a given week that one of these slots sits empty
    eligible: tuple[str, ...]  # active players who can fill it
    fix: Pickup | None = None

    def describe(self) -> str:
        who = " and ".join(self.eligible) if self.eligible else "nobody"
        text = (f"No backup at {self.position}: only {who} can play it, so a {self.position} slot "
                f"sits empty in about {self.p_empty:.0%} of weeks (0 points)")
        if self.fix is not None:
            text += f". Best cover: {self.fix.candidate.name}, +{self.fix.gain:.1f}/week"
        return text


def cover_alerts(active: Sequence[DepthPlayer], slots: Sequence[str], pickups: Sequence[Pickup] = (),
                 threshold: float = COVER_ALERT_P) -> list[CoverAlert]:
    """Starting positions with no backup: a slot empty in ``threshold`` or more of weeks."""
    outlook = lineup_outlook(active, slots)
    alerts = []
    for position, p in outlook.p_unfilled.items():
        if p < threshold:
            continue
        slot = next(s for s in slots if slot_position(s) == position)
        eligible = tuple(a.name for a in active if is_eligible(a.positions, slot))
        fix = next((pk for pk in pickups if is_eligible(pk.candidate.positions, slot)), None)
        alerts.append(CoverAlert(position, p, eligible, fix))
    return alerts
