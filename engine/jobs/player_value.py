"""Print a player's lock-in value: weekly by game count, and season on the real schedule.

    uv run python -m engine.jobs.player_value "Nikola Jokic"
    uv run python -m engine.jobs.player_value --compare "Luka Doncic" "Rudy Gobert"

This is TASKS M3's acceptance test made runnable. It reads the parquet written by
``ingest_history`` and never touches the network.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import pandas as pd

from engine.ingest.nba_stats import DATA_DIR, previous_seasons
from engine.model.distributions import (
    MIN_GAMES_FOR_RELIABLE_HISTORY,
    EmpiricalScoreDistribution,
    build_distribution,
    score_game_logs,
)
from engine.model.lockin import marginal_value_of_a_game, season_value, weekly_value

CONFIG_PATH = Path("config/league.json")
HISTORY_SEASONS = 3


def _fold_accents(text: str) -> str:
    """'Jokić' -> 'Jokic', so a name can be typed from a US keyboard."""
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(char)
    ).casefold()


def load_logs(data_dir: Path, season: str) -> pd.DataFrame:
    frames = []
    for past in previous_seasons(season, HISTORY_SEASONS):
        path = data_dir / f"game_logs_{past}.parquet"
        if not path.exists():
            raise SystemExit(
                f"missing {path}. Run: uv run python -m engine.jobs.ingest_history"
            )
        frames.append(pd.read_parquet(path))
    # The season being played, if it has started. Recency weighting then does the work of
    # learning new roles: about 50% of a player's weight comes from the current season after
    # 25 games, 75% after 50. Preseason is deliberately NOT included -- its minutes are not
    # representative -- so it informs role flags, not the distribution.
    current = data_dir / f"game_logs_{season}.parquet"
    if current.exists():
        current_logs = pd.read_parquet(current)
        if len(current_logs):
            frames.append(current_logs)

    logs = pd.concat(frames, ignore_index=True)
    logs["fp"] = score_game_logs(logs)
    return logs


def find_player(logs: pd.DataFrame, query: str) -> tuple[int, str]:
    names = logs.drop_duplicates("nba_player_id")[["nba_player_id", "player_name"]]
    folded = names["player_name"].map(_fold_accents)
    target = _fold_accents(query)

    exact = names[folded == target]
    if len(exact) == 1:
        return int(exact.iloc[0]["nba_player_id"]), exact.iloc[0]["player_name"]

    partial = names[folded.str.contains(target, regex=False)]
    if partial.empty:
        raise SystemExit(f"no player matching {query!r} in the last {HISTORY_SEASONS} seasons")
    if len(partial) > 1:
        options = ", ".join(sorted(partial["player_name"]))
        raise SystemExit(f"{query!r} is ambiguous: {options}")
    return int(partial.iloc[0]["nba_player_id"]), partial.iloc[0]["player_name"]


def player_team(logs: pd.DataFrame, player_id: int) -> str:
    """His most recent team. Offseason moves are handled by player_overrides.csv, not here."""
    played = logs[logs["nba_player_id"] == player_id].sort_values("game_date")
    return str(played.iloc[-1]["team"])


def schedule_for_team(grid: pd.DataFrame, team: str) -> list[int]:
    if team not in grid.index:
        raise SystemExit(f"team {team!r} is not in the schedule grid")
    return [int(games) for games in grid.loc[team]]


def describe(
    name: str,
    distribution: EmpiricalScoreDistribution,
    games_per_week: list[int],
    playoff_start: int,
    verbose: bool = True,
) -> dict:
    regular = games_per_week[: playoff_start - 1]
    playoffs = games_per_week[playoff_start - 1 :]

    summary = {
        "name": name,
        "p_play": distribution.p_play,
        "mean": distribution.mean(),
        "sd": distribution.sd(),
        "season_value": season_value(distribution, games_per_week),
        "regular_season_value": season_value(distribution, regular) if regular else 0.0,
        "playoff_value": season_value(distribution, playoffs) if playoffs else 0.0,
    }

    if not verbose:
        return summary

    print(f"\n{name}")
    print(
        f"  conditional on playing: mean {distribution.mean():.1f}, sd {distribution.sd():.1f}, "
        f"floor(p10) {distribution.quantile(0.10):.1f}, ceiling(p90) {distribution.quantile(0.90):.1f}"
    )
    print(
        f"  availability: p_play {distribution.p_play:.3f} over {distribution.games_observed} "
        f"games ({distribution.effective_sample_size:.0f} effective after recency weighting)"
    )
    if not distribution.history_reliable:
        print(
            f"  ** HISTORY UNRELIABLE: only {distribution.games_observed} games, below the "
            f"{MIN_GAMES_FOR_RELIABLE_HISTORY}-game bar. Treat this projection as a prior, "
            f"not a measurement. **"
        )

    print("\n  weekly value by game count (W_1), and what each added game is worth:")
    for n in range(1, 5):
        print(
            f"    {n} game week  W = {weekly_value(distribution, n):6.2f}"
            f"   (+{marginal_value_of_a_game(distribution, n):5.2f})"
        )

    counts = pd.Series(games_per_week).value_counts().sort_index()
    profile = ", ".join(f"{int(weeks)}x {int(games)}-game" for games, weeks in counts.items())
    print(f"\n  schedule: {profile}")
    print(f"  season value (mean over {len(games_per_week)} weeks): "
          f"{summary['season_value']:.2f}")
    print(f"    weeks 1-{playoff_start - 1}: {summary['regular_season_value']:.2f}   "
          f"fantasy playoffs (week {playoff_start}+): {summary['playoff_value']:.2f}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("players", nargs="+", help="player name(s), accents optional")
    parser.add_argument("--compare", action="store_true", help="one summary line per player")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    season = config["season"]
    playoff_start = config["playoffs"]["start_week"]

    grid_path = args.data_dir / f"games_per_team_per_week_{season}.parquet"
    if not grid_path.exists():
        raise SystemExit(f"missing {grid_path}. Run: uv run python -m engine.jobs.ingest_history")
    grid = pd.read_parquet(grid_path)
    grid.columns = [int(column) for column in grid.columns]

    logs = load_logs(args.data_dir, season)

    summaries = []
    for query in args.players:
        player_id, name = find_player(logs, query)
        team = player_team(logs, player_id)
        distribution = build_distribution(logs, player_id)
        summaries.append(
            describe(
                f"{name} ({team})",
                distribution,
                schedule_for_team(grid, team),
                playoff_start,
                verbose=not args.compare,
            )
        )

    if args.compare:
        print(f"{'player':<28}{'p_play':>7}{'mean':>7}{'sd':>6}{'season':>9}{'playoffs':>10}")
        for summary in sorted(summaries, key=lambda s: -s["season_value"]):
            print(
                f"{summary['name']:<28}{summary['p_play']:>7.3f}{summary['mean']:>7.1f}"
                f"{summary['sd']:>6.1f}{summary['season_value']:>9.2f}"
                f"{summary['playoff_value']:>10.2f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
