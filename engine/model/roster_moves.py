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

**Pickups.** A free agent is worth adding when his value beats the user's weakest droppable
player's by ``PICKUP_MIN_GAIN`` or more, in VOR units (lock-in points per week above the
replacement starter at his positions, from the board's ``pool.csv``). Reverse-standings
priority means a winning claim costs nothing later, so a claim is only ever worth making; the
open question per player is whether he is on waivers (claim, decided by priority) or a free
agent (add now, before his next game tips, after which he goes back on waivers).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Sequence

SLEEPER_TO_IR_CODE = {"Out": "OUT", "Sus": "SUS", "Suspended": "SUS", "DNR": "DNR"}
NOT_IR_ELIGIBLE = frozenset({"GTD", "DTD", "Questionable", "Doubtful", "Probable"})
SLEEPER_IR_DESIGNATION = "IR"
PICKUP_MIN_GAIN = 2.0  # VOR points per week; below this a move is noise against model error

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


@dataclass(frozen=True)
class Candidate:
    player_id: str
    name: str
    team: str
    positions: tuple[str, ...]
    vor: float
    injury_status: str | None = None
    on_waivers: bool = False
    waiver_note: str = ""


@dataclass(frozen=True)
class Pickup:
    candidate: Candidate
    drop: str
    gain: float

    def describe(self) -> str:
        c = self.candidate
        if c.on_waivers:
            how = f"claim on waivers ({c.waiver_note})" if c.waiver_note else "claim on waivers"
        else:
            how = "add now (free agent), before his next game tips"
        injury = f", {c.injury_status}" if c.injury_status else ""
        return (f"{c.name} ({c.team}, {'/'.join(c.positions)}{injury}): +{self.gain:.1f}/week over "
                f"{self.drop} -- {how}")


def rank_pickups(candidates: Sequence[Candidate], droppable: Sequence[tuple[str, float]],
                 min_gain: float = PICKUP_MIN_GAIN, limit: int = 5) -> list[Pickup]:
    """Free agents worth more than the user's weakest droppable player, best first.

    ``droppable``: (name, vor) for players the user could release -- his bench, since his
    starters are the lineup the matchup advice is built on. Players listed OUT are left out of
    the candidates: a pickup is for games, and an OUT free agent has none to give.
    """
    if not droppable:
        return []
    drop_name, drop_vor = min(droppable, key=lambda pair: pair[1])
    pickups = [Pickup(c, drop_name, c.vor - drop_vor) for c in candidates if not unavailable(c.injury_status)]
    pickups = [p for p in pickups if p.gain >= min_gain]
    return sorted(pickups, key=lambda p: -p.gain)[:limit]
