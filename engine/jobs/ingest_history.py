"""Pull historical game logs and the upcoming schedule to parquet (TASKS M2).

RUN LOCALLY: ``uv run python -m engine.jobs.ingest_history [--force]``

stats.nba.com blocks datacenter IPs, so this is a by-hand job on the dev machine, not
something a GitHub Actions cron can do. It is incremental -- seasons already on disk are not
refetched -- so re-running it is cheap.

It prints what it fetched and, more importantly, what is missing: unscheduled games and the
stats this source cannot provide. A gap that is visible is a gap someone can decide about; a
gap that is silently zero-filled turns into a wrong recommendation (CLAUDE.md rule 4).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from engine.ingest.nba_stats import (
    DATA_DIR,
    GAME_LOG_MISSING_STATS,
    REQUEST_SPACING_SECONDS,
    games_per_team_per_week,
    ingest_schedule,
    ingest_season_logs,
    previous_seasons,
    schedule_completeness,
    verify_weeks_match_fantasy_weeks,
)

CONFIG_PATH = Path("config/league.json")
HISTORY_SEASONS = 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="refetch even if parquet already exists"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    season = config["season"]
    season_start = config["season_start"]

    print(f"Target season {season}; history = {HISTORY_SEASONS} prior seasons.\n")

    total_rows = 0
    for index, past_season in enumerate(previous_seasons(season, HISTORY_SEASONS)):
        if index:
            time.sleep(REQUEST_SPACING_SECONDS)
        frame, fetched = ingest_season_logs(past_season, args.data_dir, force=args.force)
        total_rows += len(frame)
        source = "fetched" if fetched else "cached"
        players = frame["nba_player_id"].nunique()
        print(
            f"  {past_season}  {len(frame):>6,} rows  {players:>4} players  "
            f"{frame['game_date'].min().date()} -> {frame['game_date'].max().date()}  ({source})"
        )

    print(f"\n  total {total_rows:,} game-log rows across {HISTORY_SEASONS} seasons")

    time.sleep(REQUEST_SPACING_SECONDS)
    games, weeks, fetched = ingest_schedule(season, args.data_dir, force=args.force)
    print(f"\nSchedule {season}: {len(games):,} games, {len(weeks)} weeks "
          f"({'fetched' if fetched else 'cached'})")

    problems = verify_weeks_match_fantasy_weeks(weeks, season_start)
    if problems:
        print(
            "\nERROR: the NBA week grid does not match this league's fantasy weeks. "
            "Lock-in value is computed per fantasy week, so this must be resolved before "
            "M3 -- do not paper over it:"
        )
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"  week grid verified against config season_start {season_start}: "
          f"weeks 1-{len(weeks)}, Monday-Sunday")

    completeness = schedule_completeness(games)
    print(f"\n{completeness.describe()}")

    grid = games_per_team_per_week(games)
    grid_path = args.data_dir / f"games_per_team_per_week_{season}.parquet"
    grid.to_parquet(grid_path)
    print(f"  wrote {grid_path} ({grid.shape[0]} teams x {grid.shape[1]} weeks)")

    counts = grid.to_numpy()
    print(f"  games per team-week: min {counts.min()}, max {counts.max()}, "
          f"mean {counts.mean():.2f}")
    playoff_start = config["playoffs"]["start_week"]
    print(f"  weeks with a bye for some team: "
          f"{int((grid == 0).any(axis=0).sum())} of {grid.shape[1]} "
          f"(fantasy playoffs start week {playoff_start})")

    print("\nKnown gaps in this data:")
    print(
        f"  - {', '.join(GAME_LOG_MISSING_STATS)} are not in PlayerGameLogs and are stored as "
        f"0.0. The league scores them, so rescored history runs very slightly high."
    )
    if not completeness.is_complete:
        print(
            f"  - {completeness.expected_per_team - completeness.games_per_team_min} games per "
            f"team are unscheduled (NBA Cup window). Games-per-week in those weeks is a floor."
        )

    supabase_configured = bool(config.get("_supabase")) or "SUPABASE_URL" in _env_keys()
    if not supabase_configured:
        print(
            "  - Supabase is NOT configured (no .env), so nothing was uploaded. Parquet on "
            "disk is the only copy. TASKS M2 wants both."
        )

    return 0


def _env_keys() -> set[str]:
    env_path = Path(".env")
    if not env_path.exists():
        return set()
    keys = set()
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, _, value = line.partition("=")
            if value.strip():
                keys.add(key.strip())
    return keys


if __name__ == "__main__":
    sys.exit(main())
