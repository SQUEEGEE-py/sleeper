"""Weekly lock cheat sheet: the score each of your players must beat, game by game.

    uv run python -m engine.jobs.lock_sheet            # the week containing today (week 1 pre-season)
    uv run python -m engine.jobs.lock_sheet --week 3

A stopgap until the nightly advisor (TASKS M6) exists. For every player on the user's roster it
lists this week's games and, for each one, **the lock threshold**: after that game, lock the
score if it is at least this number, because that is what rolling on to his remaining games is
worth on average (``lockin.lock_threshold``). The final game has no threshold -- if nothing is
locked by then Sleeper takes it automatically, and it scores **0 if he sits**.

That fallback is the trap this sheet exists to flag. A player whose last game of the week is
the **second night of a back-to-back** is the one most likely to sit that game, so the sheet
marks it: lock him earlier than usual, or accept the risk knowingly.

It also says which nine to start. Only starters can be locked, so a player on the bench all
week contributes nothing; the recommended lineup maximises this week's expected lock-in value
(``lockin.weekly_value`` at his game count) through the same exact slot assignment the board
uses, and is compared with the lineup currently set in Sleeper.

Writes ``data/lock/lock_sheet.html`` (self-contained, works offline, like ``board.html``) and
prints the same to the terminal. Read-only: change lineups and lock scores in the Sleeper app.

Known limits, stated rather than hidden:

* A back-to-back second night is priced by **age band only** (``distributions.game_p_play``:
  x0.951 under 30, x0.901 at 30-33, x0.879 at 34+). A player's own B2B habits are not used
  yet; a player with no birth date in Sleeper gets no B2B adjustment and is flagged.
* The schedule has game dates, not tip-off times, so deadlines are by date: lock before his
  next game *date*. Exact tip times are the nightly job's problem.
* Sleeper injury designations are shown next to the name but do not change the numbers.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from engine.ingest.nba_stats import DATA_DIR, previous_seasons, regular_season_games
from engine.ingest.sleeper import Roster, SleeperClient
from engine.jobs.build_board import load_logs
from engine.jobs.runtime import utf8_output
from engine.model.lineup import assign_lineup, slot_position, starting_slots
from engine.model.distributions import age_on, game_p_play
from engine.model.lockin import ScoreDistribution, lock_threshold, weekly_value
from engine.model.player_model import PlayerModel, birth_dates_by_nba_id

CONFIG_PATH = Path("config/league.json")
POOL_PATH = Path("data/board/pool.csv")
OUT_DIR = Path("data/lock")
DEFAULT_USERNAME = "Squ33gee"


# --- the schedule -------------------------------------------------------------------------------


@dataclass(frozen=True)
class GameSlot:
    game_date: date
    opponent: str
    home: bool
    b2b_second: bool  # his team also played the previous calendar day

    @property
    def label(self) -> str:
        return f"{'vs' if self.home else '@'} {self.opponent}"


def team_week_games(schedule: pd.DataFrame, team: str, week: int) -> list[GameSlot]:
    """A team's regular-season games in one fantasy week, in date order.

    ``b2b_second`` looks across the whole schedule, so a Monday game after a Sunday game in
    the *previous* week still counts as the second night.
    """
    regular = regular_season_games(schedule)
    mine = regular[(regular["home_team"] == team) | (regular["away_team"] == team)]
    played_on = set(pd.to_datetime(mine["game_date"]).dt.date)
    games = []
    for row in mine[mine["week_number"] == week].sort_values("game_date").itertuples():
        day = pd.Timestamp(row.game_date).date()
        home = row.home_team == team
        games.append(
            GameSlot(
                game_date=day,
                opponent=row.away_team if home else row.home_team,
                home=home,
                b2b_second=(day - timedelta(days=1)) in played_on,
            )
        )
    return games


def week_for(weeks: pd.DataFrame, today: date) -> int:
    """The fantasy week containing ``today``; before the season, week 1; after it, the last."""
    ordered = weeks.sort_values("week_number")
    for row in ordered.itertuples():
        if pd.Timestamp(row.start_date).date() <= today <= pd.Timestamp(row.end_date).date():
            return int(row.week_number)
    if today < pd.Timestamp(ordered.iloc[0]["start_date"]).date():
        return int(ordered.iloc[0]["week_number"])
    return int(ordered.iloc[-1]["week_number"])


# --- the numbers ----------------------------------------------------------------------------------


def game_thresholds(
    distribution: ScoreDistribution, n_games: int, p_play_by_game: list[float] | None = None
) -> list[float]:
    """Lock threshold after each of ``n_games`` games; the last is ``-inf`` (nothing to roll to).

    Game ``i`` (1-based) leaves ``n_games - i`` games to roll on to, and the threshold is the
    value of those: ``lock_threshold(distribution, n_games - i)``, with the per-game ``p_play``
    of exactly those remaining games.
    """
    return [
        lock_threshold(
            distribution,
            n_games - i,
            None if p_play_by_game is None else p_play_by_game[i:],
        )
        for i in range(1, n_games + 1)
    ]


def games_p_play(base_p_play: float, games: list[GameSlot], birth_date: date | None) -> list[float]:
    """``p_play`` for each of his games this week: lower on a B2B second night, by his age then."""
    return [
        game_p_play(
            base_p_play,
            game.b2b_second,
            None if birth_date is None else age_on(birth_date, game.game_date),
        )
        for game in games
    ]


def parse_birth_date(text: str | None) -> date | None:
    try:
        return date.fromisoformat(text) if text else None
    except ValueError:
        return None


@dataclass
class PlayerSheet:
    sleeper_id: str
    name: str
    team: str
    positions: tuple[str, ...]
    injury: str
    p_play: float
    mean: float
    games: list[GameSlot]
    game_p_play: list[float]  # per game; below p_play on a B2B second night
    thresholds: list[float]
    week_value: float  # expected lock-in value this week if started
    in_sleeper_lineup: bool
    age: int | None  # at his first game this week; None = no birth date, B2B not priced
    recommended_slot: str | None = None  # slot type if in the recommended nine, else None

    @property
    def last_game_b2b(self) -> bool:
        return bool(self.games) and self.games[-1].b2b_second

    @property
    def warnings(self) -> list[str]:
        out = []
        if not self.games:
            out.append("no games this week")
        if self.last_game_b2b:
            out.append(
                "AUTO-LOCK TRAP: his last game is the 2nd night of a back-to-back -- "
                f"plays only {self.game_p_play[-1]:.0%}, and it scores 0 if he sits"
            )
        if self.age is None and any(g.b2b_second for g in self.games):
            out.append("no birth date in Sleeper: back-to-back NOT priced into these numbers")
        if self.injury:
            out.append(f"Sleeper status: {self.injury}")
        return out


def build_sheet(
    distribution: ScoreDistribution,
    sleeper_id: str,
    name: str,
    team: str,
    positions: tuple[str, ...],
    injury: str,
    games: list[GameSlot],
    in_sleeper_lineup: bool,
    birth_date: date | None,
) -> PlayerSheet:
    p_plays = games_p_play(distribution.p_play, games, birth_date)
    return PlayerSheet(
        sleeper_id=sleeper_id,
        name=name,
        team=team,
        positions=positions,
        injury=injury,
        p_play=distribution.p_play,
        mean=distribution.mean(),
        games=games,
        game_p_play=p_plays,
        thresholds=game_thresholds(distribution, len(games), p_plays),
        week_value=weekly_value(distribution, len(games), p_plays),
        in_sleeper_lineup=in_sleeper_lineup,
        age=None if birth_date is None or not games else age_on(birth_date, games[0].game_date),
    )


def recommend_lineup(sheets: list[PlayerSheet], slots: list[str]) -> list[PlayerSheet]:
    """Mark the nine starters that maximise this week's expected lock-in total."""
    lineup = assign_lineup([s.week_value for s in sheets], [s.positions for s in sheets], slots)
    for sheet in sheets:
        sheet.recommended_slot = None
    for slot, index in lineup.assignments.items():
        sheets[index].recommended_slot = slot_position(slot)
    return sheets


# --- rendering ------------------------------------------------------------------------------------


def _day(day: date) -> str:
    return f"{day:%a} {day.month}/{day.day}"


def _threshold_text(value: float) -> str:
    return "final: auto" if value == float("-inf") else f"lock ≥ {value:.1f}"


def ordered(sheets: list[PlayerSheet], slots: list[str]) -> list[PlayerSheet]:
    order = [slot_position(s) for s in slots]
    return sorted(
        sheets,
        key=lambda s: (
            s.recommended_slot is None,
            order.index(s.recommended_slot) if s.recommended_slot else 0,
            -s.week_value,
        ),
    )


def lineup_changes(sheets: list[PlayerSheet]) -> tuple[list[str], list[str]]:
    start = [s.name for s in sheets if s.recommended_slot and not s.in_sleeper_lineup]
    sit = [s.name for s in sheets if not s.recommended_slot and s.in_sleeper_lineup]
    return start, sit


# The lineup is chosen from scratch for the whole week. Mid-week that is wrong in two ways it
# cannot see: a locked player's slot is fixed, and games already played are not re-chosen.
# Found 2026-10-22, when the age adjustment had it bench Lillard, locked at 49.5 the night before.
MIDWEEK_CAVEAT = ("pre-week advice: it ignores games already played and locks -- never bench a "
                  "locked player; mid-week, follow the nightly report")


def render_terminal(sheets: list[PlayerSheet], week: int, gaps: list[str]) -> str:
    lines = [f"WEEK {week} LOCK SHEET -- lock a score if it is at least the number shown"]
    start, sit = lineup_changes(sheets)
    if start or sit:
        lines.append(f"LINEUP: start {', '.join(start) or '-'}; bench {', '.join(sit) or '-'}")
        lines.append(f"  ({MIDWEEK_CAVEAT})")
    for sheet in sheets:
        role = sheet.recommended_slot or "BN"
        lines.append(
            f"\n{role:4} {sheet.name} ({sheet.team}, {'/'.join(sheet.positions)})  "
            f"week value {sheet.week_value:.1f}, plays {sheet.p_play:.0%}"
            f"{'' if sheet.age is None else f', age {sheet.age}'}"
        )
        for game, threshold, p_play in zip(sheet.games, sheet.thresholds, sheet.game_p_play):
            b2b = f"  B2B 2nd night, plays {p_play:.0%}" if game.b2b_second else ""
            lines.append(f"     {_day(game.game_date):9} {game.label:7} {_threshold_text(threshold)}{b2b}")
        for warning in sheet.warnings:
            lines.append(f"     !! {warning}")
    for gap in gaps:
        lines.append(f"\nGAP: {gap}")
    return "\n".join(lines)


def render_html(sheets: list[PlayerSheet], week: int, week_dates: str, gaps: list[str], generated: str) -> str:
    esc = html.escape
    start, sit = lineup_changes(sheets)
    lineup_block = ""
    if start or sit:
        lineup_block = (
            "<div class=warnbox><b>Set your lineup in Sleeper.</b> Only starters can be locked. "
            f"Start: {esc(', '.join(start) or '-')}. Bench: {esc(', '.join(sit) or '-')}. "
            f"<i>{esc(MIDWEEK_CAVEAT)}</i></div>"
        )
    cards = []
    for sheet in sheets:
        rows = "".join(
            f"<tr class='{'b2b' if g.b2b_second else ''}'><td>{esc(_day(g.game_date))}</td>"
            f"<td>{esc(g.label)}{f' <span class=tag>B2B</span> <span class=mut>plays {p:.0%}</span>' if g.b2b_second else ''}</td>"
            f"<td class=num><b>{esc(_threshold_text(t))}</b></td></tr>"
            for g, t, p in zip(sheet.games, sheet.thresholds, sheet.game_p_play)
        ) or "<tr><td colspan=3 class=mut>no games this week</td></tr>"
        warns = "".join(f"<div class=warn>{esc(w)}</div>" for w in sheet.warnings)
        role = sheet.recommended_slot or "BN"
        sleeper_note = "" if bool(sheet.recommended_slot) == sheet.in_sleeper_lineup else (
            " <span class=tag>change in Sleeper</span>"
        )
        cards.append(
            f"<div class='card{' bench' if not sheet.recommended_slot else ''}'>"
            f"<div class=hd><span class=slot>{esc(role)}</span> <b>{esc(sheet.name)}</b> "
            f"<span class=mut>{esc(sheet.team)} &middot; {esc('/'.join(sheet.positions))}</span>{sleeper_note}</div>"
            f"<div class=mut>week value {sheet.week_value:.1f} &middot; {sheet.mean:.0f} fp/g when he plays "
            f"&middot; plays {sheet.p_play:.0%} of games"
            f"{'' if sheet.age is None else f' &middot; age {sheet.age}'}</div>"
            f"<table>{rows}</table>{warns}</div>"
        )
    gap_block = "".join(f"<div class=warn>GAP: {esc(g)}</div>" for g in gaps)
    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Lock Sheet</title>
<style>
:root{{--bg:#fff;--fg:#111;--mut:#666;--line:#e3e3e3;--accent:#0b5;--warn:#c60;--head:#f7f7f7}}
@media(prefers-color-scheme:dark){{:root{{--bg:#14161a;--fg:#e8e8e8;--mut:#9aa0a6;--line:#2a2e35;--accent:#3d8;--warn:#f93;--head:#1c1f25}}}}
*{{box-sizing:border-box}}
body{{margin:0;padding:12px;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
h1{{font-size:18px;margin:0 0 2px}}
.mut{{color:var(--mut);font-size:12px}}
.how{{font-size:13px;margin:6px 0 10px}}
.card{{border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:8px 0}}
.card.bench{{opacity:.75}}
.hd{{font-size:15px}}
.slot{{display:inline-block;min-width:38px;font-size:11px;font-weight:700;color:var(--accent)}}
table{{border-collapse:collapse;width:100%;margin-top:4px;font-variant-numeric:tabular-nums}}
td{{padding:4px 6px;border-bottom:1px solid var(--line)}}
td.num{{text-align:right}}
tr.b2b td{{color:var(--warn)}}
.tag{{font-size:10px;color:var(--warn);border:1px solid var(--warn);border-radius:3px;padding:0 3px}}
.warn{{color:var(--warn);font-size:12px;margin-top:4px}}
.warnbox{{border-left:3px solid var(--warn);background:var(--head);padding:6px 10px;margin:8px 0}}
@media(min-width:900px){{.grid{{display:grid;grid-template-columns:1fr 1fr;gap:0 12px}}}}
</style></head><body>
<h1>Week {week} lock sheet</h1>
<div class=mut>{esc(week_dates)} &middot; generated {esc(generated)} &middot; works offline</div>
<div class=how>After each game, <b>lock the score if it is at least the number shown</b> &mdash; that is
what rolling on to his remaining games is worth. The final game is taken automatically if nothing is
locked, and scores <b>0 if he sits</b>. <span class=tag>B2B</span> = second night of a back-to-back,
priced in: less likely to play, more so with age.</div>
{lineup_block}{gap_block}
<div class=grid>{''.join(cards)}</div>
</body></html>"""


# --- main -------------------------------------------------------------------------------------------


def my_roster(client: SleeperClient, league_id: str, username: str) -> Roster:
    user_id = client.get_user(username).user_id
    for roster in client.get_rosters_live(league_id):
        if roster.owner_id == user_id:
            return roster
    raise SystemExit(f"{username} owns no roster in league {league_id}")


def main() -> int:
    utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--user", default=DEFAULT_USERNAME)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    season = config["season"]
    slots = starting_slots(config["roster_positions"])
    if not POOL_PATH.exists():
        raise SystemExit(f"missing {POOL_PATH}. Run: uv run python -m engine.jobs.build_board")

    schedule = pd.read_parquet(args.data_dir / f"schedule_{season}.parquet")
    weeks = pd.read_parquet(args.data_dir / f"weeks_{season}.parquet")
    week = args.week or week_for(weeks, date.today())
    week_row = weeks[weeks["week_number"] == week].iloc[0]
    week_dates = f"{pd.Timestamp(week_row['start_date']):%a %b %d} - {pd.Timestamp(week_row['end_date']):%a %b %d}"

    pool = pd.read_csv(POOL_PATH, dtype={"sleeper_id": str}).set_index("sleeper_id")
    with SleeperClient() as client:
        roster = my_roster(client, config["league_id"], args.user)
        sleeper_players = client.get_players()

    preseason_path = args.data_dir / f"preseason_logs_{season}.parquet"
    preseason = pd.read_parquet(preseason_path) if preseason_path.exists() else pd.DataFrame()
    nba_to_sleeper = {int(nba_id): sleeper_id for sleeper_id, nba_id in pool["nba_player_id"].items()}
    model = PlayerModel(load_logs(args.data_dir, season), preseason, previous_seasons(season, 1)[0],
                        season=season, birth_dates=birth_dates_by_nba_id(nba_to_sleeper, sleeper_players))

    gaps: list[str] = []
    sheets: list[PlayerSheet] = []
    starters = set(roster.starters or [])
    for sleeper_id in roster.players or []:
        player = sleeper_players.get(sleeper_id)
        name = player.full_name if player else sleeper_id
        if sleeper_id not in pool.index:
            gaps.append(f"{name}: not in the valued pool (rebuild the board?); no thresholds")
            continue
        row = pool.loc[sleeper_id]
        team = (player.team if player and player.team else row["team"])
        try:
            distribution = model.distribution(int(row["nba_player_id"]))
        except KeyError as error:
            gaps.append(f"{name}: {error}")
            continue
        sheets.append(
            build_sheet(
                distribution,
                sleeper_id,
                row["name"],
                team,
                tuple(ast.literal_eval(row["positions"])),
                (player.injury_status or "") if player else "",
                team_week_games(schedule, team, week),
                sleeper_id in starters,
                parse_birth_date(player.birth_date if player else None),
            )
        )

    sheets = ordered(recommend_lineup(sheets, slots), slots)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    args.out.mkdir(parents=True, exist_ok=True)
    page = args.out / "lock_sheet.html"
    page.write_text(render_html(sheets, week, week_dates, gaps, generated), encoding="utf-8")
    (args.out / f"lock_sheet_week{week}.html").write_text(page.read_text(encoding="utf-8"), encoding="utf-8")
    print(render_terminal(sheets, week, gaps))
    print(f"\nwrote {page.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
