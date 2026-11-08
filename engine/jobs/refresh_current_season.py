"""Rebuild the season being played from Sleeper, so the snapshot refresh needs no laptop.

    uv run python -m engine.jobs.refresh_current_season   # -> data/game_logs_{season}.parquet + week grid
    uv run python -m engine.jobs.build_board
    uv run python -m engine.jobs.publish_snapshot          # then commit and push snapshot/

Completed seasons never change, so they stay as ingested from stats.nba.com (copy them from
``snapshot/`` into ``data/`` on a fresh machine). Only the current season's logs and the team x
week grid are rebuilt here, both from Sleeper, which answers from cloud IPs
(``engine/ingest/sleeper_logs.py`` has why it is the better source).

Who is fetched: every Sleeper player on an NBA team. Each is joined to an NBA id by name, the
same join the board uses (``player_ids.map_players``), against every player in the logs and the
preseason logs we hold. A Sleeper player with games this season but no NBA id -- someone who
had never appeared in any log before -- is listed as a gap and left out, never given an
invented id.

Before writing, the new season is compared with the file it replaces over the dates both
cover: player-games and fantasy points per game should agree to within the known differences
(technicals and flagrants, the odd stat correction). A large disagreement stops the job.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from engine.ingest.nba_stats import DATA_DIR, games_per_team_per_week, previous_seasons
from engine.ingest.player_ids import load_overrides, map_players
from engine.ingest.sleeper import SleeperClient
from engine.ingest.sleeper_logs import log_rows, logs_frame
from engine.jobs.build_board import HISTORY_SEASONS
from engine.jobs.runtime import utf8_output
from engine.model.distributions import score_game_logs as score_logs
from engine.model.scoring import load_rules

CONFIG_PATH = Path("config/league.json")
WORKERS = 6  # a few hundred requests, a handful at a time: polite, and done in about a minute
# Week 1 check: 235/258 identical, the rest technicals/flagrants or one steal. Allow for that.
MAX_MISSING_SHARE = 0.03
MAX_MEAN_FP_GAP = 0.5


def known_players(data_dir: Path, season: str) -> list[tuple[int, str, str]]:
    """(nba id, latest name, latest team) for everyone in the logs and preseason logs we hold."""
    frames = []
    for name in ([f"game_logs_{s}.parquet" for s in previous_seasons(season, HISTORY_SEASONS)]
                 + [f"game_logs_{season}.parquet", f"preseason_logs_{season}.parquet"]):
        path = data_dir / name
        if path.exists():
            frames.append(pd.read_parquet(path, columns=["nba_player_id", "player_name", "team", "game_date"]))
    if not frames:
        raise SystemExit(f"no game logs in {data_dir}: copy snapshot/*.parquet there first")
    latest = pd.concat(frames).sort_values("game_date").groupby("nba_player_id").last()
    return [(int(i), r.player_name, r.team) for i, r in latest.iterrows()]


def compare(old: pd.DataFrame, new: pd.DataFrame) -> tuple[list[str], bool]:
    """Agreement over the dates the old file covers. Returns report lines and whether it passes."""
    if old.empty:
        return ["no earlier file to compare with"], True
    end = old["game_date"].max()
    new_part = new[new["game_date"] <= end]
    rules = load_rules()
    key = ["nba_player_id", "game_date"]
    merged = old.assign(fp=score_logs(old, rules))[key + ["fp"]].merge(
        new_part.assign(fp=score_logs(new_part, rules))[key + ["fp"]], on=key, how="outer",
        suffixes=("_old", "_new"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    only_old = int((merged["_merge"] == "left_only").sum())
    only_new = int((merged["_merge"] == "right_only").sum())
    same = int((abs(both["fp_old"] - both["fp_new"]) < 0.01).sum())
    gap = float((both["fp_new"] - both["fp_old"]).mean()) if len(both) else 0.0
    lines = [f"vs the file it replaces, through {end:%Y-%m-%d}: {len(both)} player-games in both "
             f"({same} score identically), {only_old} only in the old, {only_new} only in the new; "
             f"mean fp/game new - old {gap:+.2f}"]
    passes = only_old <= MAX_MISSING_SHARE * len(old) and abs(gap) <= MAX_MEAN_FP_GAP
    return lines, passes


def main() -> int:
    utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--force", action="store_true", help="write even if the comparison fails")
    args = parser.parse_args()
    season = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["season"]
    year = season.split("-")[0]

    known = known_players(args.data_dir, season)
    with SleeperClient() as client:
        players = client.get_players()
        on_team = {sid: p for sid, p in players.items() if p.team}
        mapping = map_players(known, on_team, load_overrides())
        sleeper_to_nba = {sid: nba for nba, sid in mapping.nba_to_sleeper.items()}
        print(f"{len(on_team)} Sleeper players on an NBA team; {len(sleeper_to_nba)} joined to an NBA id")

        with ThreadPoolExecutor(WORKERS) as pool:
            games = dict(zip(on_team, pool.map(lambda sid: client.get_player_games(sid, year), on_team)))
        schedule = client.get_nba_schedule(year)

    rows, gaps = [], []
    for sid, player_games in games.items():
        played = [g for g in player_games if (g.stats or {}).get("sp") or (g.stats or {}).get("min")]
        if not played:
            continue
        if sid not in sleeper_to_nba:
            gaps.append(f"{on_team[sid].full_name} ({on_team[sid].team}): {len(played)} game(s), no NBA id -- left out")
            continue
        rows += log_rows(sleeper_to_nba[sid], on_team[sid].full_name or sid, player_games, season)
    new = logs_frame(rows)
    teams = new.groupby("team")["game_id"].nunique()
    print(f"{len(new)} player-games, {new['game_date'].min():%Y-%m-%d} to {new['game_date'].max():%Y-%m-%d}; "
          f"games per team {teams.min()}-{teams.max()} ({len(teams)} teams)")

    target = args.data_dir / f"game_logs_{season}.parquet"
    old = pd.read_parquet(target) if target.exists() else pd.DataFrame()
    lines, passes = compare(old, new)
    print("\n".join(lines))
    if gaps:
        print(f"\nGAPS ({len(gaps)}):\n" + "\n".join(f"  - {g}" for g in gaps))
    if not passes and not args.force:
        print("\nNOT WRITTEN: the rebuild disagrees with the file it replaces by more than the known "
              "differences. Look before overriding with --force.")
        return 1

    new.to_parquet(target)
    # Imported here: nightly is heavy, and only its schedule shape is needed.
    from engine.jobs.nightly import schedule_frame
    grid = games_per_team_per_week(schedule_frame(schedule))
    grid_path = args.data_dir / f"games_per_team_per_week_{season}.parquet"
    grid.to_parquet(grid_path)
    print(f"\nwrote {target} and {grid_path} ({grid.shape[0]} teams x {grid.shape[1]} weeks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
