"""Nightly matchup advisor: win probability this week, and LOCK or ROLL for each open decision.

    uv run python -m engine.jobs.nightly                 # the week containing today
    uv run python -m engine.jobs.nightly --week 2 --sims 40000

For both teams in the user's matchup it reads Sleeper's live counted score per starter, each
player's finished games this week (Sleeper per-game stats, rescored through ``scoring.py``) and
the games still to come (Sleeper's schedule), then:

* infers each starter's lock status (``engine/model/locks.py`` -- Sleeper exposes no lock
  field, so a lock on tonight's game cannot be told apart from no lock until his next game);
* simulates the rest of the week for both lineups (``engine/model/matchup.py``) to get the
  chance of winning;
* for each of the user's players with an open decision, compares the chance of winning if he
  locks tonight's score with the chance if he rolls on, ordered by the earliest next game.

Every in-season input here comes from Sleeper, which answers from cloud IPs (probed from
GitHub Actions 2026-10-21). The player model still needs ``data/board/pool.csv`` and the
game-log parquet on disk; moving those to the cloud is the rest of step 4, as are tip-off times
(ESPN) and the Discord post. Read-only: lock in the Sleeper app.

Stated limits:

* the opponent is assumed to follow the expected-value lock policy;
* a game in progress is simulated as a fresh game -- its partial score is ignored -- and the
  decision before it is treated as gone;
* lineups are taken as set now; bench swaps later in the week are not modelled.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from engine.ingest.espn import fetch_tip_times
from engine.ingest.nba_stats import DATA_DIR, GAME_TYPE_REGULAR, previous_seasons
from engine.ingest.sleeper import INJURY_STATUS_TTL_SECONDS, Matchup, Player, ScheduledGame, SleeperClient
from engine.jobs.build_board import load_logs
from engine.jobs.discord import WEBHOOK_ENV, post, webhook_url
from engine.jobs.lock_sheet import GameSlot, games_p_play, my_roster, parse_birth_date, team_week_games
from engine.jobs.runtime import log_exception, utf8_output
from engine.model.locks import LockStatus, infer_lock
from engine.model.matchup import DEFAULT_SIMS, Decision, Odds, Slot, decide, policy_lock, win_probability
from engine.model.player_model import PlayerModel, birth_dates_by_nba_id
from engine.model.roles import MIN_GAMES, RoleCheck, role_check
from engine.model.lineup import is_eligible, slot_position, starting_slots
from engine.model.lockin import weekly_value
from engine.model.trades import Deadline, Trade, TradePlayer, find_trades, post_trades_today
from engine.model.trades import deadline as trade_deadline
from engine.model.roster_moves import (
    Candidate,
    IRAlert,
    Pickup,
    RosterPlayer,
    ir_alerts,
    rank_pickups,
    unavailable,
    waiver_reason,
)
from engine.model.scoring import box_from_sleeper_stats, load_rules, score_box

CONFIG_PATH = Path("config/league.json")
POOL_PATH = Path("data/board/pool.csv")
LOCKS_PATH = Path("state/locks.json")


def reported_locks(path: Path, week: int) -> dict[str, float]:
    """Locks the user reported for ``week``, by Sleeper player id. Another week's file counts
    for nothing: locks reset every week."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("week") != week:
        return {}
    return {pid: float(entry["score"]) for pid, entry in (data.get("locked") or {}).items()}
DEFAULT_USERNAME = "Squ33gee"
EMPTY_SLOT = "0"
SEED = 20261020
EASTERN = ZoneInfo("America/New_York")
SNAPSHOT_STALE_DAYS = 8


# --- schedule -------------------------------------------------------------------------------


def schedule_frame(games: list[ScheduledGame]) -> pd.DataFrame:
    """Sleeper's schedule in the shape ``lock_sheet.team_week_games`` reads. Sleeper lists
    regular-season games only, and its weeks match ours exactly (checked 2026-10-21: all 1200
    games agree on date, teams and week)."""
    return pd.DataFrame(
        [{"game_date": pd.Timestamp(g.date), "home_team": g.home, "away_team": g.away,
          "week_number": g.week, "game_type": GAME_TYPE_REGULAR} for g in games]
    )


def week_containing(games: list[ScheduledGame], today: date) -> int:
    """The fantasy week of today's games, else of the next game day."""
    upcoming = sorted((g.date, g.week) for g in games if g.date >= today.isoformat())
    if not upcoming:
        raise SystemExit("no games left in Sleeper's schedule")
    return upcoming[0][1]


# --- one starter ----------------------------------------------------------------------------


@dataclass
class Starter:
    player_id: str
    name: str
    team: str
    counted: float
    games: list[GameSlot] = field(default_factory=list)
    completed: list[tuple[date, float]] = field(default_factory=list)
    in_progress: bool = False
    upcoming: list[GameSlot] = field(default_factory=list)
    lock: LockStatus | None = None
    slot: Slot | None = None
    gap: str = ""

    next_tip: datetime | None = None  # UTC, from ESPN

    @property
    def next_game(self) -> date | None:
        return self.upcoming[0].game_date if self.upcoming else None


def build_starter(
    player_id: str, counted: float, player: Player | None, schedule: pd.DataFrame,
    status: dict[tuple[str, str], str], scores_by_date: dict[str, float], week: int,
    distribution, base_p_play: float | None, mine: bool = True,
    reported_lock: float | None = None,
) -> Starter:
    """Lock status and simulation slot for one starter. ``distribution`` is None when the
    player has no model; that is a gap only if he still has games to simulate."""
    name = (player.full_name if player else None) or player_id
    team = (player.team if player else None) or ""
    starter = Starter(player_id, name, team, counted)
    if not team:
        starter.gap = "no NBA team in Sleeper"
    else:
        starter.games = team_week_games(schedule, team, week)
    for game in starter.games:
        state = status.get((team, game.game_date.isoformat()), "pre_game")
        if state == "complete":
            starter.completed.append((game.game_date, scores_by_date.get(game.game_date.isoformat(), 0.0)))
        elif state == "in_game":
            starter.in_progress = True
            starter.upcoming.append(game)  # simulated as a fresh game; see the module docstring
        else:
            starter.upcoming.append(game)

    starter.lock = infer_lock(starter.completed, counted, starter.in_progress)
    remaining: tuple[float, ...] = ()
    if starter.upcoming:
        if distribution is None:
            starter.gap = starter.gap or "no player model (not in pool.csv)"
            return starter
        birth = parse_birth_date(player.birth_date if player else None)
        remaining = tuple(games_p_play(base_p_play, starter.upcoming, birth))
        if birth is None:
            starter.gap = "no birth date: back-to-back nights not priced"
        status = player.injury_status if player else None
        if unavailable(status) and not starter.in_progress:
            # Out (or suspended, DNR, Sleeper's IR): his next game is priced at 0. Later games
            # keep their usual chance -- Sleeper gives no return date, and the status is
            # re-read every run, so this rolls forward a game at a time instead of guessing.
            remaining = (0.0,) + remaining[1:]
            starter.gap = f"{status}: his next game is priced at 0"

    if reported_lock is not None and mine:
        # The user told us. Sleeper's own numbers cannot show a lock (state/locks.json).
        starter.lock = LockStatus("locked", None, reported_lock, note="reported by you")
        starter.gap = ""
        starter.slot = Slot(name, distribution, remaining, locked=reported_lock)
        return starter
    if not mine and distribution is not None and starter.completed:
        # The opponent's locks are never visible (locks.py). The user's assumption (2026-10-25):
        # he locks optimally -- replay his finished games through the same thresholds.
        birth = parse_birth_date(player.birth_date if player else None)
        week_p = games_p_play(base_p_play, starter.games, birth)
        scores = [score for _, score in starter.completed]
        locked_at = policy_lock(distribution, scores, week_p)
        if locked_at is not None:
            day = next(d for d, s in starter.completed if s == locked_at)
            starter.lock = LockStatus("locked", day, locked_at, note="assumed: locks optimally")
            starter.slot = Slot(name, distribution, remaining, locked=locked_at)
        elif starter.in_progress or not starter.upcoming:
            # Mid-game, the earlier decision is gone; with no games left, the last one counts.
            final = None if starter.upcoming else scores[-1]
            starter.lock = LockStatus("open", note="assumed: locks optimally; not locked yet")
            starter.slot = Slot(name, distribution, remaining, pending=final)
        else:
            starter.lock = LockStatus("open", note="assumed: locks optimally; not locked yet")
            starter.slot = Slot(name, distribution, remaining, pending=scores[-1])
        return starter
    lock = starter.lock
    if lock.state == "unclear" and mine:
        # Sleeper still shows an earlier game: locked, or not caught up (locks.py). For the
        # user's own player assume NOT locked -- he knows if he did, and the advice says so --
        # and work from his latest finished game, the one a lock now would bank.
        starter.gap = f"{lock.note}; advice below assumes you have not locked him"
        starter.slot = Slot(name, distribution, remaining, pending=starter.completed[-1][1])
    elif lock.state in ("locked", "unclear"):
        # The opponent: Sleeper's number is the best there is; treat it as banked.
        starter.slot = Slot(name, distribution, remaining, locked=lock.locked_score)
    elif lock.state == "no_game_yet" or (lock.state == "open" and starter.in_progress):
        starter.slot = Slot(name, distribution, remaining)
    else:
        # open, or unexplained: what Sleeper counts now is real, and it stays counted unless a
        # later game replaces it -- the simulation decides that, nothing is assumed.
        if lock.state == "unexplained":
            starter.gap = f"lock status unclear: {lock.note}"
        starter.slot = Slot(name, distribution, remaining, pending=counted)
    return starter


# --- the report -----------------------------------------------------------------------------


def _day(day: date | None) -> str:
    return f"{day:%a %m/%d}" if day else "--"


def deadline(starter: Starter) -> str:
    """When the decision closes: his next tip-off in Eastern time, else just the date."""
    if starter.next_tip is not None:
        tip = starter.next_tip.astimezone(EASTERN)
        return f"{tip:%a} {tip:%I:%M %p}".replace(" 0", " ") + " ET"
    return f"{_day(starter.next_game)} (tip time unknown)"


def render_discord(week: int, their_name: str, mine: list[Starter], my_matchup: Matchup,
                   their_matchup: Matchup, odds: Odds | None,
                   decisions: list[tuple[Starter, Decision]], gaps: list[str],
                   roles: list[RoleCheck] | None = None, moves: RosterMoves | None = None,
                   trades: TradeReport | None = None, swaps: list[Swap] | None = None) -> str:
    """The phone version: short lines, no columns. Injured starters first, then decisions,
    then every other starter -- the user asked that no one in the lineup go unmentioned."""
    if odds is None:
        head = f"**Week {week} vs {their_name}** -- win probability not computed (see gaps)"
    else:
        head = (f"**Week {week} vs {their_name}: {odds.p_win:.0%} to win**\n"
                f"Sleeper now {my_matchup.points or 0:g}-{their_matchup.points or 0:g} · "
                f"projected {odds.my_mean:.0f}-{odds.their_mean:.0f}")
    lines = [head, ""]
    for swap in swaps or []:
        lines += [f"⚠️ **{swap.describe()}**", ""]
    if decisions:
        lines.append("Only if you have **not** already locked him:")
        for starter, decision in decisions:
            lines.append(f"**{decision.call.upper()} {starter.name}** -- before {deadline(starter)}")
            why = (f"counting {decision.pending:g} · lock at {decision.threshold:.1f}+ · "
                   f"win {decision.p_win_lock:.0%} if locked / {decision.p_win_roll:.0%} if rolled")
            if decision.matchup_changed_it:
                why += " · *matchup overrides the points call*"
            lines.append(why)
    else:
        lines.append("No open lock decisions.")
    decided = {s.player_id for s, _ in decisions}
    others = [s for s in mine if s.player_id not in decided and s.player_id != EMPTY_SLOT]
    if others:
        lines += ["", "**The rest of your lineup**"] + [f"- {starter_line(s)}" for s in others]
    if moves is not None and (moves.ir or moves.pickups):
        priority = f" (waiver priority #{moves.waiver_position})" if moves.waiver_position else ""
        lines += ["", f"**Roster moves**{priority}"]
        lines += [f"- {a.text}" for a in moves.ir] + [f"- Pick up {pk.describe()}" for pk in moves.pickups]
    flagged = [r for r in roles or [] if r.flagged]
    if flagged:
        lines += ["", "Role change:"] + [f"- {r.describe()}" for r in flagged]
    if trades is not None and trades.post:
        lines += ["", f"**Trade ideas** ({trades.deadline.describe()})"]
        lines += [f"- {t.describe()}" for t in trades.trades] or ["- none that improve both lineups"]
    if gaps:
        lines += ["", "Gaps:"] + [f"- {gap}" for gap in gaps]
    return "\n".join(lines)


def starter_line(s: Starter) -> str:
    """Where a starter with no open decision stands -- so every slot is in the report."""
    lock = s.lock
    if lock is not None and lock.state == "locked":
        return f"{s.name}: locked {lock.locked_score:g}"
    if s.slot is None:
        return f"{s.name}: {s.gap or 'no model'}"
    if not s.upcoming:
        return f"{s.name}: done for the week, {s.counted:g} counts" if s.completed else f"{s.name}: no games this week"
    games_after = len(s.slot.remaining_p_play) - 1
    nxt = deadline(s) if s.next_tip else _day(s.next_game)
    if s.in_progress:
        return f"{s.name}: playing now"
    if games_after == 0:
        return f"{s.name}: last game {nxt} -- it counts as is"
    return f"{s.name}: next game {nxt} -- lock it if he scores {s.slot.threshold(games_after):.1f}+"


def decision_line(starter: Starter, decision: Decision) -> str:
    call = decision.call.upper()
    why = (f"counting {decision.pending:g}; lock is worth it at {decision.threshold:.1f}+ on expected "
           f"points; win chance {decision.p_win_lock:.0%} if locked vs {decision.p_win_roll:.0%} if rolled")
    if decision.matchup_changed_it:
        why += " -- the matchup overrides the expected-points call"
    return f"  {deadline(starter):18} {call:5} {starter.name:24} {why}"


def render(week: int, my_name: str, their_name: str, mine: list[Starter], theirs: list[Starter],
           my_matchup: Matchup, their_matchup: Matchup, odds: Odds | None,
           decisions: list[tuple[Starter, Decision]], gaps: list[str],
           roles: list[RoleCheck] | None = None, moves: RosterMoves | None = None,
           trades: TradeReport | None = None, swaps: list[Swap] | None = None) -> str:
    lines = [f"WEEK {week}: {my_name} vs {their_name}"]
    lines.append(f"Sleeper now: {my_matchup.points or 0:g} - {their_matchup.points or 0:g}")
    if odds is None:
        lines.append("WIN PROBABILITY: not computed -- see gaps below")
    else:
        lines.append(f"WIN PROBABILITY: {odds.p_win:.0%}   projected {odds.my_mean:.1f} - {odds.their_mean:.1f}"
                     f"   (your margin, 10th-90th pct: {odds.margin_p10:+.0f} to {odds.margin_p90:+.0f})")
    lines.append("")
    if decisions:
        lines.append("DECIDE before his next game -- only if you have NOT already locked him:")
        lines += [decision_line(s, d) for s, d in decisions]
    else:
        lines.append("No open lock decisions right now.")
    lines.append("")
    lines.append("YOUR STARTERS")
    for s in mine:
        lines.append(f"  {s.name:24} {_state(s)}")
    lines.append(f"\n{their_name.upper()}'S STARTERS")
    for s in theirs:
        lines.append(f"  {s.name:24} {_state(s)}")
    if swaps:
        lines.append("\nINJURED STARTERS")
        lines += [f"  !! {swap.describe()}" for swap in swaps]
    if moves is not None:
        priority = f" (you are #{moves.waiver_position} in waiver priority)" if moves.waiver_position else ""
        lines.append(f"\nROSTER MOVES{priority}")
        lines += [f"  IR: {a.text}" for a in moves.ir] or ["  IR: nothing to do"]
        lines += ([f"  PICKUP: {pk.describe()}" for pk in moves.pickups]
                  or ["  PICKUP: no free agent beats your bench by the margin"])
    if trades is not None:
        lines.append(f"\nTRADES ({trades.deadline.describe()}; both lineups must improve, our model;"
                     " Slpr = Sleeper's rank, the market view)")
        lines += [f"  {t.describe()}" for t in trades.trades] or ["  none that improve both lineups"]
    if roles:
        lines.append(f"\nROLE WATCH (your roster; minutes this season vs his latest earlier season, judged from {MIN_GAMES} games)")
        lines += [f"  {'!! ' if r.flagged else '   '}{r.describe()}" for r in roles]
    lines.append("\nAssumes the opponent locks by the expected-points thresholds. Locks are inferred "
                 "from Sleeper's counted scores; a lock on the latest game cannot be seen until the next one.")
    if gaps:
        lines.append("\nGAPS:")
        lines += [f"  - {gap}" for gap in gaps]
    return "\n".join(lines)


def _state(s: Starter) -> str:
    done = ", ".join(f"{d:%a} {score:g}" for d, score in s.completed) or "no games yet"
    left = ", ".join(f"{g.game_date:%a}" + (" (B2B)" if g.b2b_second else "") for g in s.upcoming)
    lock = s.lock
    if lock is None:
        state = s.gap
    elif lock.state == "locked":
        when = f"{lock.locked_on:%a}" if lock.locked_on else lock.note
        state = f"LOCKED {lock.locked_score:g} ({when})"
    elif lock.state == "unclear":
        state = f"Sleeper shows {lock.locked_score:g} ({lock.locked_on:%a}) -- locked?"
    elif s.in_progress and lock.state in ("no_game_yet", "open"):
        state = f"IN GAME, live {s.counted:g}"
    elif lock.state == "no_game_yet":
        state = "no game yet"
    else:
        state = f"counting {s.counted:g}" + (" (in game)" if s.in_progress else "")
    return f"{state:30} played: {done:24} left: {left or 'none'}"


# --- main -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Swap:
    out: str
    status: str
    slot: str
    options: list[tuple[str, float, list[str]]]  # (bench player, value of his games left, days)

    def describe(self) -> str:
        if not self.options:
            return (f"{self.out} is {self.status}: no healthy bench player with games left can play "
                    f"{slot_position(self.slot)} -- look at free agents")
        (name, value, days), *rest = self.options
        text = (f"{self.out} is {self.status}: bench him and start {name} at {slot_position(self.slot)} "
                f"({', '.join(days)}; worth {value:.1f})")
        if rest:
            text += " -- next best " + ", ".join(f"{n} {v:.1f}" for n, v, _ in rest[:2])
        return text


def injury_swaps(mine: list[Starter], slots: list[str], bench: list[tuple[Starter, tuple[str, ...]]],
                 players: dict) -> list[Swap]:
    """For each of the user's starters who is Out (or SUS, DNR, Sleeper's IR) and has not locked
    a score, the bench players who could take his slot, ranked by what their games left this
    week are worth (``weekly_value``; a bench player's earlier games do not count for you)."""
    swaps = []
    for starter, slot in zip(mine, slots):
        player = players.get(starter.player_id)
        status = player.injury_status if player else None
        if not unavailable(status) or (starter.lock and starter.lock.state == "locked"):
            continue
        options = []
        for candidate, positions in bench:
            slot_obj = candidate.slot
            healthy = not unavailable(getattr(players.get(candidate.player_id), "injury_status", None))
            if (slot_obj is None or slot_obj.distribution is None or not slot_obj.remaining_p_play
                    or not healthy or not is_eligible(positions, slot)):
                continue
            value = weekly_value(slot_obj.distribution, len(slot_obj.remaining_p_play), slot_obj.remaining_p_play)
            days = [f"{g.game_date:%a}" + (" B2B" if g.b2b_second else "") for g in candidate.upcoming]
            options.append((candidate.name, value, days))
        options.sort(key=lambda o: -o[1])
        swaps.append(Swap(starter.name, status, slot, options))
    return swaps


@dataclass
class RosterMoves:
    ir: list[IRAlert]
    pickups: list[Pickup]
    waiver_position: int | None
    gaps: list[str]


def roster_moves(config: dict, roster, rosters: list, players: dict, pool: pd.DataFrame,
                 status: dict[tuple[str, str], str], today: date, transactions: list,
                 now: datetime) -> RosterMoves:
    """IR alerts and free-agent pickups for the user's roster (``roster_moves.py`` has the
    rules). Values are the board's VOR from ``pool.csv``; a bench player missing from it is a
    gap, not a zero."""
    extra = roster.model_extra or {}
    reserve_ids = list(extra.get("reserve") or [])
    roster_ids = list(roster.players or [])
    gaps: list[str] = []

    def name_of(pid: str) -> str:
        player = players.get(pid)
        return (player.full_name if player else None) or pid

    def status_of(pid: str) -> str | None:
        player = players.get(pid)
        return player.injury_status if player else None

    active = [RosterPlayer(pid, name_of(pid), status_of(pid)) for pid in roster_ids if pid not in reserve_ids]
    reserve = [RosterPlayer(pid, name_of(pid), status_of(pid)) for pid in reserve_ids]
    ir = ir_alerts(active, reserve, config["ir"]["slots"], config["ir"]["eligible_statuses"])

    dropped_at: dict[str, datetime] = {}
    for t in transactions:
        if t.status == "complete" and t.drops and t.status_updated:
            when = datetime.fromtimestamp(t.status_updated / 1000, timezone.utc)
            for pid in t.drops:
                dropped_at[pid] = max(when, dropped_at.get(pid, when))
    started = {team for (team, day), state in status.items() if day == today.isoformat() and state != "pre_game"}
    rostered = {pid for r in rosters for pid in (r.players or [])}
    waivers = config["waivers"]
    candidates = []
    for sid, row in pool.iterrows():
        player = players.get(sid)
        if sid in rostered or player is None or not player.team:
            continue
        reason = waiver_reason(sid, player.team, dropped_at, now, waivers["clear_days"], started,
                               waivers["after_game_start"])
        candidates.append(Candidate(sid, str(row["name"]), player.team, tuple(ast.literal_eval(row["positions"])),
                                    float(row["vor"]), player.injury_status, reason is not None, reason or ""))

    droppable = []
    for pid in roster_ids:
        if pid in (roster.starters or []) or pid in reserve_ids:
            continue
        if pid in pool.index:
            droppable.append((name_of(pid), float(pool.loc[pid, "vor"])))
        else:
            gaps.append(f"{name_of(pid)}: not in pool.csv, so not compared against free agents")
    position = (extra.get("settings") or {}).get("waiver_position")
    return RosterMoves(ir, rank_pickups(candidates, droppable), position, gaps)


@dataclass
class TradeReport:
    deadline: Deadline
    trades: list[Trade]
    post: bool  # include in today's Discord post


def trade_report(config: dict, my_roster_id: int, rosters: list, owners: dict[int, str],
                 pool: pd.DataFrame, slots: list[str], games: list[ScheduledGame], week: int,
                 today: date) -> TradeReport:
    """Trades improving both lineups (``trades.py``), and the deadline countdown. Players in
    IR are left out of both sides; a player missing from ``pool.csv`` is left out too, and
    ``build_board``'s own gap list already reports him."""
    deadline_week = config["trade_deadline_week"]
    days = [g.date for g in games if g.week == deadline_week]
    last_day = date.fromisoformat(max(days)) if days else date.max
    dl = trade_deadline(deadline_week, week, last_day)
    if dl.passed:
        return TradeReport(dl, [], False)

    def side(r) -> list[TradePlayer]:
        reserve = set((r.model_extra or {}).get("reserve") or [])
        out = []
        for pid in r.players or []:
            if pid in reserve or pid not in pool.index:
                continue
            row = pool.loc[pid]
            rank = None if pd.isna(row["sleeper_rank"]) else float(row["sleeper_rank"])
            out.append(TradePlayer(pid, str(row["name"]), tuple(ast.literal_eval(row["positions"])),
                                   float(row["season_value"]), rank))
        return out

    mine = next(side(r) for r in rosters if r.roster_id == my_roster_id)
    partners = {owners[r.roster_id]: side(r) for r in rosters if r.roster_id != my_roster_id}
    return TradeReport(dl, find_trades(mine, partners, slots), post_trades_today(dl, today))


def role_watch(roster: list[str], pool: pd.DataFrame, model: PlayerModel, players: dict,
               season_games) -> list[RoleCheck]:
    """Minutes this season (Sleeper per-game ``min``, so it works in Actions) against his most
    recent earlier season in the model's logs -- not always last season: Lillard and Kyrie
    missed 2025-26 injured. Every player on the user's roster, flagged first."""
    earlier = model.logs[model.logs["season"] != model.season].sort_values("game_date")
    latest = earlier.groupby("nba_player_id")["season"].last()
    earlier = earlier[earlier["season"] == earlier["nba_player_id"].map(latest)]
    last_minutes = earlier.groupby("nba_player_id")["minutes"].mean()
    last_team = earlier.groupby("nba_player_id")["team"].last()
    checks = []
    for player_id in roster:
        player = players.get(player_id)
        name = (player.full_name if player else None) or player_id
        nba_id = int(pool.loc[player_id, "nba_player_id"]) if player_id in pool.index else None
        minutes = [float(g.stats["min"]) for g in season_games(player_id) if g.stats.get("min")]
        known = nba_id is not None and nba_id in last_minutes.index
        checks.append(role_check(
            name, (player.team if player else None) or "?",
            last_team[nba_id] if known else None,
            minutes,
            float(last_minutes[nba_id]) if known else None,
            latest[nba_id] if known else None,
        ))
    return sorted(checks, key=lambda c: (not c.flagged, not c.new_team, c.name))


def today_eastern() -> date:
    """Game dates are US dates; a run at 3 am UTC is still the previous evening in New York."""
    return datetime.now(EASTERN).date()


def run(args: argparse.Namespace) -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    season, season_year, league_id = config["season"], config["season"].split("-")[0], config["league_id"]
    if not args.pool.exists():
        raise SystemExit(f"missing {args.pool}. Run: uv run python -m engine.jobs.build_board")
    pool = pd.read_csv(args.pool, dtype={"sleeper_id": str}).set_index("sleeper_id")
    preseason_path = args.data_dir / f"preseason_logs_{season}.parquet"
    preseason = pd.read_parquet(preseason_path) if preseason_path.exists() else pd.DataFrame()
    model = PlayerModel(load_logs(args.data_dir, season), preseason, previous_seasons(season, 1)[0],
                        season=season)
    rules = load_rules()
    today = today_eastern()
    gaps: list[str] = []
    manifest = args.data_dir / "manifest.json"
    if manifest.exists():  # running from snapshot/ (Actions): say if the model is going stale
        published = datetime.fromisoformat(json.loads(manifest.read_text(encoding="utf-8"))["published"])
        age = (datetime.now(timezone.utc) - published).days
        if age > SNAPSHOT_STALE_DAYS:
            gaps.append(f"model snapshot is {age} days old -- refresh it on the laptop "
                        "(ingest_history, build_board, publish_snapshot, push)")

    with SleeperClient() as client:
        games = client.get_nba_schedule(season_year)
        week = args.week or week_containing(games, today)
        reported = reported_locks(args.locks, week)
        roster = my_roster(client, league_id, args.user)
        matchups = client.get_matchups_live(league_id, week)
        users = {u.user_id: u.display_name for u in client.get_users(league_id)}
        rosters = client.get_rosters_live(league_id)
        owners = {r.roster_id: users.get(r.owner_id or "", f"roster {r.roster_id}") for r in rosters}
        # This week's drops, and last week's too early in a week, decide who is on waivers.
        transactions = client.get_transactions_live(league_id, week)
        if week > 1:
            transactions += client.get_transactions_live(league_id, week - 1)
        players = client.get_players(ttl_seconds=INJURY_STATUS_TTL_SECONDS)
        # Read lazily by the model, so filling them here, before any distribution, is enough.
        model.birth_dates = birth_dates_by_nba_id(
            {int(nba_id): sid for sid, nba_id in pool["nba_player_id"].items()}, players)
        mine_m = next(m for m in matchups if m.roster_id == roster.roster_id)
        theirs_m = next(m for m in matchups if m.matchup_id == mine_m.matchup_id and m.roster_id != roster.roster_id)

        schedule = schedule_frame(games)
        status = {}
        for g in games:
            status[(g.home, g.date)] = status[(g.away, g.date)] = g.status

        games_cache: dict[str, list] = {}

        def season_games(player_id: str) -> list:
            if player_id not in games_cache:
                games_cache[player_id] = client.get_player_games(player_id, season_year)
            return games_cache[player_id]

        def starters_of(m: Matchup) -> list[Starter]:
            built = []
            for player_id, counted in zip(m.starters or [], m.starters_points or []):
                if player_id == EMPTY_SLOT:
                    built.append(Starter(EMPTY_SLOT, "(empty slot)", "", 0.0, slot=Slot("(empty)", None)))
                    continue
                scores = {g.date: score_box(box_from_sleeper_stats(g.stats), rules)
                          for g in season_games(player_id) if g.week == week}
                distribution, base = None, None
                if player_id in pool.index:
                    try:
                        distribution = model.distribution(int(pool.loc[player_id, "nba_player_id"]))
                        base = distribution.p_play
                    except KeyError:
                        pass
                starter = build_starter(player_id, float(counted), players.get(player_id), schedule,
                                        status, scores, week, distribution, base,
                                        mine=m.roster_id == roster.roster_id,
                                        reported_lock=reported.get(player_id))
                if starter.gap:
                    gaps.append(f"{owners[m.roster_id]}: {starter.name}: {starter.gap}")
                built.append(starter)
            return built

        mine, theirs = starters_of(mine_m), starters_of(theirs_m)
        roles = role_watch(mine_m.model_extra.get("players") or [], pool, model, players, season_games)
        reserve_ids = set((roster.model_extra or {}).get("reserve") or [])
        bench = []
        for pid in roster.players or []:
            if pid in (mine_m.starters or []) or pid in reserve_ids:
                continue
            player = players.get(pid)
            dist = None
            if pid in pool.index:
                try:
                    dist = model.distribution(int(pool.loc[pid, "nba_player_id"]))
                except KeyError:
                    pass
            built = build_starter(pid, 0.0, player, schedule, status, {}, week, dist,
                                  dist.p_play if dist else None)
            bench.append((built, tuple((player.fantasy_positions or []) if player else [])))
        swaps = injury_swaps(mine, starting_slots(config["roster_positions"]), bench, players)
        moves = roster_moves(config, roster, rosters, players, pool, status, today, transactions,
                             datetime.now(timezone.utc))
        gaps += moves.gaps
        trades = trade_report(config, roster.roster_id, rosters, owners, pool,
                              starting_slots(config["roster_positions"]), games, week, today)

    # Deadlines need tip-off times; Sleeper has none, so they come from ESPN. A failure here
    # costs only the clock time, so it is a gap, not a crash.
    next_days = {s.next_game for s in mine if s.next_game}
    try:
        tips = fetch_tip_times(next_days)
    except Exception as error:
        tips = {}
        gaps.append(f"tip-off times unavailable from ESPN ({type(error).__name__}); deadlines by date only")
    for starter in mine:
        if starter.next_game:
            starter.next_tip = tips.get((starter.team, starter.next_game.isoformat()))

    my_slots = [s.slot for s in mine]
    their_slots = [s.slot for s in theirs]
    odds, decisions = None, []
    if all(slot is not None for slot in my_slots + their_slots):
        odds = win_probability(my_slots, their_slots, args.sims, SEED)
        for index, starter in enumerate(mine):
            if starter.slot.decision_open:
                decisions.append((starter, decide(my_slots, their_slots, index, args.sims, SEED)))
        decisions.sort(key=lambda pair: (pair[0].next_tip or datetime.max.replace(tzinfo=timezone.utc),
                                         pair[0].next_game or date.max, pair[0].name))

    my_name, their_name = owners[mine_m.roster_id], owners[theirs_m.roster_id]
    print(render(week, my_name, their_name, mine, theirs, mine_m, theirs_m, odds, decisions, gaps, roles, moves, trades, swaps))

    if args.post:
        due_today = [d for d in decisions if d[0].next_game == today]
        if args.only_if_due_today and not due_today:
            print("\nnot posted: no decision is due today")
            return 0
        shown = due_today if args.only_if_due_today else decisions
        message = render_discord(week, their_name, mine, mine_m, theirs_m, odds, shown, gaps, roles, moves, trades, swaps)
        if args.reason:
            message = rerun_banner(args.reason) + "\n" + message
        post_or_say(message)
    return 0


def rerun_banner(reason: str) -> str:
    """Heads a report posted because the model changed, not because the night ended: a call
    that differs from the last post is then explained by the change, not by the player (the
    Kawhi flip on 2026-10-22 came from the age adjustment, and arrived unexplained)."""
    first_line = reason.strip().splitlines()[0][:200] if reason.strip() else "model updated"
    return (f"🔄 **Re-run after a model change:** {first_line}\n"
            "Calls below may differ from earlier posts -- this version supersedes them.")


def post_or_say(message: str) -> None:
    url = webhook_url()
    if url is None:
        print(f"\nnot posted: {WEBHOOK_ENV} is not set (.env locally, a repository secret in Actions)")
        return
    post(message, url)
    print("\nposted to Discord")


def main() -> int:
    utf8_output()
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--user", default=DEFAULT_USERNAME)
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--pool", type=Path, default=POOL_PATH)
    parser.add_argument("--post", action="store_true", help="also post the report to Discord")
    parser.add_argument("--only-if-due-today", action="store_true",
                        help="with --post: post only decisions whose next game is today (the afternoon reminder)")
    parser.add_argument("--locks", type=Path, default=LOCKS_PATH,
                        help="locks the user reported (Sleeper does not expose them)")
    parser.add_argument("--reason", default="",
                        help="with --post: why this run happened off-schedule (the commit that changed the model)")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as error:
        # A scheduled run that dies quietly is the 18:31 incident again: log it, and say so in
        # Discord too when posting, then fail the run so Actions shows it red.
        log_path = log_exception("nightly", error)
        if args.post:
            try:
                post_or_say(f"**Nightly advisor failed**: {type(error).__name__}: {error}"[:1900])
            except Exception:
                pass
        print(f"nightly failed; traceback in {log_path}", file=sys.stderr)
        if os.environ.get("GITHUB_ACTIONS"):
            # An annotation is public on the run page; step logs need a GitHub login, and
            # without this the 10/26 failure showed only "exit code 1".
            print(actions_error_annotation(error))
        raise


def actions_error_annotation(error: BaseException) -> str:
    """A GitHub Actions ``::error`` line naming the failure, with its innermost frame."""
    frame = traceback.extract_tb(error.__traceback__)[-1] if error.__traceback__ else None
    where = f" at {Path(frame.filename).name}:{frame.lineno} in {frame.name}" if frame else ""
    text = f"{type(error).__name__}: {error}{where}"
    escaped = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return f"::error title=Nightly advisor failed::{escaped[:1000]}"


if __name__ == "__main__":
    sys.exit(main())
