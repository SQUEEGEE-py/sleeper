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
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from engine.ingest.espn import fetch_tip_times
from engine.ingest.nba_stats import DATA_DIR, GAME_TYPE_REGULAR, previous_seasons
from engine.ingest.sleeper import Matchup, Player, ScheduledGame, SleeperClient
from engine.jobs.build_board import load_logs
from engine.jobs.discord import WEBHOOK_ENV, post, webhook_url
from engine.jobs.lock_sheet import GameSlot, games_p_play, my_roster, parse_birth_date, team_week_games
from engine.jobs.runtime import log_exception, utf8_output
from engine.model.locks import LockStatus, infer_lock
from engine.model.matchup import DEFAULT_SIMS, Decision, Odds, Slot, decide, win_probability
from engine.model.player_model import PlayerModel, birth_dates_by_nba_id
from engine.model.roles import MIN_GAMES, RoleCheck, role_check
from engine.model.scoring import box_from_sleeper_stats, load_rules, score_box

CONFIG_PATH = Path("config/league.json")
POOL_PATH = Path("data/board/pool.csv")
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


def deadline(starter: Starter) -> str:
    """When the decision closes: his next tip-off in Eastern time, else just the date."""
    if starter.next_tip is not None:
        tip = starter.next_tip.astimezone(EASTERN)
        return f"{tip:%a} {tip:%I:%M %p}".replace(" 0", " ") + " ET"
    return f"{_day(starter.next_game)} (tip time unknown)"


def render_discord(week: int, their_name: str, mine: list[Starter], my_matchup: Matchup,
                   their_matchup: Matchup, odds: Odds | None,
                   decisions: list[tuple[Starter, Decision]], gaps: list[str],
                   roles: list[RoleCheck] | None = None) -> str:
    """The phone version: short lines, no columns, decisions first."""
    if odds is None:
        head = f"**Week {week} vs {their_name}** -- win probability not computed (see gaps)"
    else:
        head = (f"**Week {week} vs {their_name}: {odds.p_win:.0%} to win**\n"
                f"Sleeper now {my_matchup.points or 0:g}-{their_matchup.points or 0:g} · "
                f"projected {odds.my_mean:.0f}-{odds.their_mean:.0f}")
    lines = [head, ""]
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
    locked = [f"{s.name} {s.lock.locked_score:g}" for s in mine if s.lock and s.lock.state == "locked"]
    if locked:
        lines += ["", "Locked: " + ", ".join(locked)]
    flagged = [r for r in roles or [] if r.flagged]
    if flagged:
        lines += ["", "Role change:"] + [f"- {r.describe()}" for r in flagged]
    if gaps:
        lines += ["", "Gaps:"] + [f"- {gap}" for gap in gaps]
    return "\n".join(lines)


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
           roles: list[RoleCheck] | None = None) -> str:
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
        state = f"LOCKED {lock.locked_score:g} ({lock.locked_on:%a})"
    elif s.in_progress and lock.state in ("no_game_yet", "open"):
        state = f"IN GAME, live {s.counted:g}"
    elif lock.state == "no_game_yet":
        state = "no game yet"
    else:
        state = f"counting {s.counted:g}" + (" (in game)" if s.in_progress else "")
    return f"{state:30} played: {done:24} left: {left or 'none'}"


# --- main -----------------------------------------------------------------------------------


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
        roster = my_roster(client, league_id, args.user)
        matchups = client.get_matchups_live(league_id, week)
        users = {u.user_id: u.display_name for u in client.get_users(league_id)}
        owners = {r.roster_id: users.get(r.owner_id or "", f"roster {r.roster_id}")
                  for r in client.get_rosters_live(league_id)}
        players = client.get_players()
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
                                        status, scores, week, distribution, base)
                if starter.gap:
                    gaps.append(f"{owners[m.roster_id]}: {starter.name}: {starter.gap}")
                built.append(starter)
            return built

        mine, theirs = starters_of(mine_m), starters_of(theirs_m)
        roles = role_watch(mine_m.model_extra.get("players") or [], pool, model, players, season_games)

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
    print(render(week, my_name, their_name, mine, theirs, mine_m, theirs_m, odds, decisions, gaps, roles))

    if args.post:
        due_today = [d for d in decisions if d[0].next_game == today]
        if args.only_if_due_today and not due_today:
            print("\nnot posted: no decision is due today")
            return 0
        shown = due_today if args.only_if_due_today else decisions
        message = render_discord(week, their_name, mine, mine_m, theirs_m, odds, shown, gaps, roles)
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
        raise


if __name__ == "__main__":
    sys.exit(main())
