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
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from engine.ingest.nba_stats import DATA_DIR, GAME_TYPE_REGULAR, previous_seasons
from engine.ingest.sleeper import Matchup, Player, ScheduledGame, SleeperClient
from engine.jobs.build_board import load_logs
from engine.jobs.lock_sheet import GameSlot, games_p_play, my_roster, parse_birth_date, team_week_games
from engine.jobs.runtime import utf8_output
from engine.model.locks import LockStatus, infer_lock
from engine.model.matchup import DEFAULT_SIMS, Decision, Odds, Slot, decide, win_probability
from engine.model.player_model import PlayerModel
from engine.model.scoring import box_from_sleeper_stats, load_rules, score_box

CONFIG_PATH = Path("config/league.json")
POOL_PATH = Path("data/board/pool.csv")
DEFAULT_USERNAME = "Squ33gee"
EMPTY_SLOT = "0"
SEED = 20261020


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

    @property
    def next_game(self) -> date | None:
        return self.upcoming[0].game_date if self.upcoming else None


def build_starter(
    player_id: str, counted: float, player: Player | None, schedule: pd.DataFrame,
    status: dict[tuple[str, str], str], scores_by_date: dict[str, float], week: int,
    distribution, base_p_play: float | None,
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

    lock = starter.lock
    if lock.state == "locked":
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


def decision_line(starter: Starter, decision: Decision) -> str:
    call = decision.call.upper()
    why = (f"counting {decision.pending:g}; lock is worth it at {decision.threshold:.1f}+ on expected "
           f"points; win chance {decision.p_win_lock:.0%} if locked vs {decision.p_win_roll:.0%} if rolled")
    if decision.matchup_changed_it:
        why += " -- the matchup overrides the expected-points call"
    return f"  {_day(starter.next_game):10} {call:5} {starter.name:24} {why}"


def render(week: int, my_name: str, their_name: str, mine: list[Starter], theirs: list[Starter],
           my_matchup: Matchup, their_matchup: Matchup, odds: Odds | None,
           decisions: list[tuple[Starter, Decision]], gaps: list[str]) -> str:
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
        state = f"LOCKED {lock.locked_score:g} ({lock.locked_on:%a})"
    elif s.in_progress and lock.state in ("no_game_yet", "open"):
        state = f"IN GAME, live {s.counted:g}"
    elif lock.state == "no_game_yet":
        state = "no game yet"
    else:
        state = f"counting {s.counted:g}" + (" (in game)" if s.in_progress else "")
    return f"{state:30} played: {done:24} left: {left or 'none'}"


# --- main -----------------------------------------------------------------------------------


def main() -> int:
    utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--user", default=DEFAULT_USERNAME)
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    season, season_year, league_id = config["season"], config["season"].split("-")[0], config["league_id"]
    if not POOL_PATH.exists():
        raise SystemExit(f"missing {POOL_PATH}. Run: uv run python -m engine.jobs.build_board")
    pool = pd.read_csv(POOL_PATH, dtype={"sleeper_id": str}).set_index("sleeper_id")
    preseason_path = args.data_dir / f"preseason_logs_{season}.parquet"
    preseason = pd.read_parquet(preseason_path) if preseason_path.exists() else pd.DataFrame()
    model = PlayerModel(load_logs(args.data_dir, season), preseason, previous_seasons(season, 1)[0])
    rules = load_rules()

    with SleeperClient() as client:
        games = client.get_nba_schedule(season_year)
        week = args.week or week_containing(games, date.today())
        roster = my_roster(client, league_id, args.user)
        matchups = client.get_matchups_live(league_id, week)
        users = {u.user_id: u.display_name for u in client.get_users(league_id)}
        owners = {r.roster_id: users.get(r.owner_id or "", f"roster {r.roster_id}")
                  for r in client.get_rosters_live(league_id)}
        players = client.get_players()
        mine_m = next(m for m in matchups if m.roster_id == roster.roster_id)
        theirs_m = next(m for m in matchups if m.matchup_id == mine_m.matchup_id and m.roster_id != roster.roster_id)

        schedule = schedule_frame(games)
        status = {}
        for g in games:
            status[(g.home, g.date)] = status[(g.away, g.date)] = g.status

        gaps: list[str] = []

        def starters_of(m: Matchup) -> list[Starter]:
            built = []
            for player_id, counted in zip(m.starters or [], m.starters_points or []):
                if player_id == EMPTY_SLOT:
                    built.append(Starter(EMPTY_SLOT, "(empty slot)", "", 0.0, slot=Slot("(empty)", None)))
                    continue
                scores = {g.date: score_box(box_from_sleeper_stats(g.stats), rules)
                          for g in client.get_player_games(player_id, season_year) if g.week == week}
                distribution, base = None, None
                if player_id in pool.index:
                    try:
                        distribution = model.distribution(int(pool.loc[player_id, "nba_player_id"]))
                        base = distribution.p_play
                    except KeyError:
                        pass
                starter = build_starter(player_id, float(counted), players.get(player_id), schedule,
                                        status, scores, week, distribution, base)
                if starter.gap:
                    gaps.append(f"{owners[m.roster_id]}: {starter.name}: {starter.gap}")
                built.append(starter)
            return built

        mine, theirs = starters_of(mine_m), starters_of(theirs_m)

    my_slots = [s.slot for s in mine]
    their_slots = [s.slot for s in theirs]
    odds, decisions = None, []
    if all(slot is not None for slot in my_slots + their_slots):
        odds = win_probability(my_slots, their_slots, args.sims, SEED)
        for index, starter in enumerate(mine):
            if starter.slot.decision_open:
                decisions.append((starter, decide(my_slots, their_slots, index, args.sims, SEED)))
        decisions.sort(key=lambda pair: (pair[0].next_game or date.max, pair[0].name))

    print(render(week, owners[mine_m.roster_id], owners[theirs_m.roster_id], mine, theirs,
                 mine_m, theirs_m, odds, decisions, gaps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
