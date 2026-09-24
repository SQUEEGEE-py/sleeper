"""Historical game logs and the season schedule, from stats.nba.com (TASKS M2).

RUN THIS LOCALLY. stats.nba.com blocks datacenter IPs, so this will work on the dev machine
and fail from a GitHub-hosted runner. Everything it fetches is written to parquet under
``data/nba/`` so the rest of the engine -- and every test -- reads from disk and never needs
the network.

Two sources, both on stats.nba.com:

* ``PlayerGameLogs``   -- one row per player per game, rescored later by ``model/scoring.py``.
* ``ScheduleLeagueV2`` -- the season's games plus the league's own week grid.

SPEC §2.2 suggests cdn.nba.com for the schedule as a lighter-weight alternative. **It is
blocked from this machine** (Akamai "Access Denied", verified 2026-09-23 with and without
browser headers), so the schedule comes from stats.nba.com too.

Known gaps, surfaced rather than papered over (CLAUDE.md rule 4):

* **Technical and flagrant fouls are not in PlayerGameLogs.** The league scores them (`tf`
  -1.0, `ff` -2.0) but the box score endpoint only carries personal fouls. Rescored history
  therefore treats every game as having zero technicals, which biases historical fantasy
  points very slightly high for players who collect them. Getting them needs play-by-play.
  See ``GAME_LOG_MISSING_STATS``.
* **Two games per team are unscheduled.** See :func:`schedule_completeness`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/nba")

# Seconds to wait between stats.nba.com calls. The endpoint is undocumented and rate limited;
# a handful of calls a minute is plenty for a job that runs by hand a few times a season.
# Never retry into a rate limit (CLAUDE.md, known environment traps).
REQUEST_SPACING_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 60

# gameId prefixes. The first three characters encode the game type.
GAME_TYPE_PRESEASON = "001"
GAME_TYPE_REGULAR = "002"
GAME_TYPE_NBA_CUP_FINAL = "006"

# stats.nba.com's season_type values. "Pre Season" really does have the space.
SEASON_TYPE_REGULAR = "Regular Season"
SEASON_TYPE_PRESEASON = "Pre Season"

# Box-score columns we keep, mapped to the field names scoring.BoxScore uses. Anything not
# listed is dropped; PlayerGameLogs returns 70+ columns and we need nine of them.
GAME_LOG_STAT_COLUMNS: dict[str, str] = {
    "PTS": "pts",
    "REB": "reb",
    "AST": "ast",
    "STL": "stl",
    "BLK": "blk",
    "TOV": "to",
    "FG3M": "fg3m",
}

GAME_LOG_ID_COLUMNS: dict[str, str] = {
    "PLAYER_ID": "nba_player_id",
    "PLAYER_NAME": "player_name",
    "TEAM_ABBREVIATION": "team",
    "GAME_ID": "game_id",
    "GAME_DATE": "game_date",
    "MIN": "minutes",
}

# Scored by the league but absent from PlayerGameLogs. Written as 0.0 and listed here so the
# board can show it as a known gap instead of implying the number is measured.
GAME_LOG_MISSING_STATS: tuple[str, ...] = ("tech_foul", "flagrant_foul")


@dataclass(frozen=True)
class ScheduleCompleteness:
    """How much of the season is actually scheduled.

    Games per week is the main driver of lock-in value (SPEC §3), so an unscheduled game is
    not a cosmetic gap -- it understates a player's week.
    """

    teams: int
    games_scheduled: int
    games_per_team_min: int
    games_per_team_max: int
    expected_per_team: int
    placeholder_games: int

    @property
    def is_complete(self) -> bool:
        return (
            self.games_per_team_min == self.expected_per_team
            and self.placeholder_games == 0
        )

    def describe(self) -> str:
        if self.is_complete:
            return f"schedule complete: {self.teams} teams x {self.expected_per_team} games"
        return (
            f"SCHEDULE INCOMPLETE: {self.teams} teams have "
            f"{self.games_per_team_min}-{self.games_per_team_max} of "
            f"{self.expected_per_team} games scheduled; {self.placeholder_games} games still "
            f"have a TBD team. Games-per-week in the affected weeks is a floor, not a count."
        )


def season_string(start_year: int) -> str:
    """2026 -> '2026-27', the format stats.nba.com expects."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def previous_seasons(season: str, count: int) -> list[str]:
    """The ``count`` seasons before ``season``, oldest first. '2026-27', 3 ->
    ['2023-24', '2024-25', '2025-26']."""
    start_year = int(season.split("-")[0])
    return [season_string(year) for year in range(start_year - count, start_year)]


# --- pure transforms (no network; unit tested against fixtures) -----------------------------


def normalize_game_logs(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """PlayerGameLogs output -> the tidy frame the rest of the engine reads.

    Renames to box-score field names, drops the 60-odd columns we do not score, and adds
    explicit zero columns for the stats this source cannot provide.
    """
    missing = [
        column
        for column in (*GAME_LOG_ID_COLUMNS, *GAME_LOG_STAT_COLUMNS)
        if column not in raw.columns
    ]
    if missing:
        raise ValueError(
            f"PlayerGameLogs response is missing expected columns {missing}. "
            "The endpoint's shape changed; fix the mapping rather than filling defaults"
        )

    columns = {**GAME_LOG_ID_COLUMNS, **GAME_LOG_STAT_COLUMNS}
    frame = raw[list(columns)].rename(columns=columns).copy()
    frame["season"] = season
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.tz_localize(None)

    for stat in GAME_LOG_STAT_COLUMNS.values():
        frame[stat] = pd.to_numeric(frame[stat], errors="coerce").fillna(0.0).astype(float)

    # Not measured by this source. Zero, and advertised as a gap.
    for stat in GAME_LOG_MISSING_STATS:
        frame[stat] = 0.0

    return frame.sort_values(["game_date", "nba_player_id"]).reset_index(drop=True)


def normalize_schedule(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """ScheduleLeagueV2's game frame -> one row per scheduled game.

    ``gameDate`` arrives as a US-format string ('10/20/2026 00:00:00'), which sorts wrong as
    text -- parse it rather than comparing strings.
    """
    frame = pd.DataFrame(
        {
            "game_id": raw["gameId"],
            "game_date": pd.to_datetime(raw["gameDate"], format="%m/%d/%Y %H:%M:%S"),
            "week_number": pd.to_numeric(raw["weekNumber"], errors="coerce"),
            "home_team": raw["homeTeam_teamTricode"],
            "away_team": raw["awayTeam_teamTricode"],
        }
    )
    frame["game_type"] = frame["game_id"].str[:3]
    frame["season"] = season
    return frame.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def regular_season_games(schedule: pd.DataFrame) -> pd.DataFrame:
    return schedule[schedule["game_type"] == GAME_TYPE_REGULAR].copy()


def normalize_weeks(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """ScheduleLeagueV2's week frame.

    Two quirks of the source, both observed on the real 2026-27 payload:

    * It is **not sorted** -- a week arrives out of order -- so sorting here is load-bearing.
    * It contains an **exactly duplicated row** (week 8 appears twice, identical in every
      field). Identical rows are collapsed. Rows that share a week number but disagree about
      its dates are NOT collapsed: that would be a real conflict about when a week starts, and
      :func:`verify_weeks_match_fantasy_weeks` is left to fail on it.
    """
    frame = pd.DataFrame(
        {
            "week_number": pd.to_numeric(raw["weekNumber"]),
            "week_name": raw["weekName"],
            "start_date": pd.to_datetime(raw["startDate"]).dt.tz_localize(None),
            "end_date": pd.to_datetime(raw["endDate"]).dt.tz_localize(None),
        }
    )
    frame["season"] = season
    frame = frame.drop_duplicates()
    return frame.sort_values("week_number").reset_index(drop=True)


def verify_weeks_match_fantasy_weeks(weeks: pd.DataFrame, season_start: str) -> list[str]:
    """Check the NBA's week grid really is the league's fantasy week grid before relying on it.

    Lock-in value is computed per fantasy week, so using the wrong week boundaries would
    silently corrupt every number downstream. Verified rather than assumed: week 1 must begin
    on the configured season start, weeks must be numbered 1..n with no gaps, and each week
    after the first must run Monday to Sunday.

    Returns a list of problems; empty means the grids agree.
    """
    problems: list[str] = []
    expected_start = pd.Timestamp(season_start)

    if weeks.empty:
        return ["week grid is empty"]

    first = weeks.iloc[0]
    if first["start_date"] != expected_start:
        problems.append(
            f"week 1 starts {first['start_date'].date()}, but config season_start is "
            f"{expected_start.date()}"
        )

    conflicting = weeks[weeks["week_number"].duplicated(keep=False)]
    if not conflicting.empty:
        problems.append(
            "week numbers appear more than once with differing dates, so the week grid is "
            f"ambiguous: {sorted(set(conflicting['week_number']))}"
        )

    numbers = list(weeks["week_number"])
    if numbers != list(range(1, len(numbers) + 1)):
        problems.append(f"week numbers are not 1..n with no gaps: {numbers}")

    for _, week in weeks.iloc[1:].iterrows():
        if week["start_date"].dayofweek != 0:  # Monday
            problems.append(
                f"week {int(week['week_number'])} starts on a "
                f"{week['start_date'].day_name()}, not a Monday"
            )
        if (week["end_date"] - week["start_date"]).days != 6:
            problems.append(
                f"week {int(week['week_number'])} spans "
                f"{(week['end_date'] - week['start_date']).days + 1} days, not 7"
            )

    return problems


def games_per_team_per_week(schedule: pd.DataFrame) -> pd.DataFrame:
    """The M2 deliverable: a team x fantasy-week grid of game counts.

    This is the input the lock-in model cares about most -- a week with four games is four
    draws from a player's distribution, a week with one is a forced single.
    """
    regular = regular_season_games(schedule)
    home = regular[["home_team", "week_number"]].rename(columns={"home_team": "team"})
    away = regular[["away_team", "week_number"]].rename(columns={"away_team": "team"})
    appearances = pd.concat([home, away], ignore_index=True)
    appearances = appearances[appearances["team"].notna()]

    grid = (
        appearances.groupby(["team", "week_number"]).size().rename("games").reset_index()
    )
    return grid.pivot(index="team", columns="week_number", values="games").fillna(0).astype(int)


def schedule_completeness(
    schedule: pd.DataFrame, expected_per_team: int = 82
) -> ScheduleCompleteness:
    """Quantify what is missing from the schedule.

    The NBA Cup leaves two games per team unassigned until the group stage resolves, so a
    freshly published schedule is legitimately short. Downstream code must know that, because
    games-per-week for those weeks is a lower bound.
    """
    regular = regular_season_games(schedule)
    teams = pd.concat([regular["home_team"], regular["away_team"]], ignore_index=True)
    # A placeholder game has BOTH sides unknown, so count games, not null team slots.
    placeholder = int(
        (regular["home_team"].isna() | regular["away_team"].isna()).sum()
    )
    counts = teams.dropna().value_counts()

    return ScheduleCompleteness(
        teams=int(len(counts)),
        games_scheduled=int(len(regular)),
        games_per_team_min=int(counts.min()) if len(counts) else 0,
        games_per_team_max=int(counts.max()) if len(counts) else 0,
        expected_per_team=expected_per_team,
        placeholder_games=placeholder,
    )


# --- fetch + persist (network; run locally) ------------------------------------------------


def _parquet_path(name: str, season: str, data_dir: Path) -> Path:
    return data_dir / f"{name}_{season}.parquet"


def fetch_game_logs(
    season: str,
    season_type: str = SEASON_TYPE_REGULAR,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """One season of player game logs. Network call; local only.

    Returns an empty frame with the right columns when the season has not started, which is
    the normal state for the current season in September.
    """
    from nba_api.stats.endpoints import playergamelogs

    response = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable=season_type,
        timeout=timeout,
    )
    raw = response.get_data_frames()[0]
    if raw.empty:
        columns = [
            *GAME_LOG_ID_COLUMNS.values(),
            *GAME_LOG_STAT_COLUMNS.values(),
            *GAME_LOG_MISSING_STATS,
            "season",
            "season_type",
        ]
        return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})
    frame = normalize_game_logs(raw, season)
    frame["season_type"] = season_type
    return frame


def fetch_schedule(
    season: str, timeout: int = REQUEST_TIMEOUT_SECONDS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(games, weeks) for a season. Network call; local only."""
    from nba_api.stats.endpoints import scheduleleaguev2

    response = scheduleleaguev2.ScheduleLeagueV2(season=season, league_id="00", timeout=timeout)
    games_raw, weeks_raw = response.get_data_frames()
    return normalize_schedule(games_raw, season), normalize_weeks(weeks_raw, season)


def ingest_season_logs(
    season: str,
    data_dir: Path = DATA_DIR,
    force: bool = False,
    season_type: str = SEASON_TYPE_REGULAR,
) -> tuple[pd.DataFrame, bool]:
    """Fetch and persist one season of logs. Returns (frame, fetched).

    Incremental: a season already on disk is not refetched unless ``force``. Completed
    seasons never change, so re-running this job costs one call per missing season and
    nothing for the rest.
    """
    name = "game_logs" if season_type == SEASON_TYPE_REGULAR else "preseason_logs"
    path = _parquet_path(name, season, data_dir)
    if path.exists() and not force:
        return pd.read_parquet(path), False

    frame = fetch_game_logs(season, season_type)
    data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame, True


def ingest_current_season(
    season: str, data_dir: Path = DATA_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regular-season and preseason logs for the season now being played.

    **Always refetched**, unlike completed seasons -- the whole point is that it changes. Once
    the season starts this is what lets the model learn a player's new role on its own:
    recency weighting reaches 50% new-role weight after about 25 games (roughly 8 weeks) and
    75% after 50, with no manual override needed.

    Before opening night both come back empty, which is a normal state and not an error.
    Preseason is kept in a separate file and tagged, because preseason minutes are not
    representative -- starters play limited minutes and deep bench players play a lot -- so it
    is a role-change *signal*, not distribution data (SPEC §2.2).
    """
    regular, _ = ingest_season_logs(season, data_dir, force=True)
    time.sleep(REQUEST_SPACING_SECONDS)
    preseason, _ = ingest_season_logs(
        season, data_dir, force=True, season_type=SEASON_TYPE_PRESEASON
    )
    return regular, preseason


def ingest_schedule(
    season: str, data_dir: Path = DATA_DIR, force: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Fetch and persist the schedule and week grid. Returns (games, weeks, fetched).

    Unlike completed seasons' logs, the upcoming schedule DOES change -- unscheduled NBA Cup
    games get filled in -- so this is worth re-running with ``force`` closer to the season.
    """
    games_path = _parquet_path("schedule", season, data_dir)
    weeks_path = _parquet_path("weeks", season, data_dir)
    if games_path.exists() and weeks_path.exists() and not force:
        return pd.read_parquet(games_path), pd.read_parquet(weeks_path), False

    games, weeks = fetch_schedule(season)
    data_dir.mkdir(parents=True, exist_ok=True)
    games.to_parquet(games_path, index=False)
    weeks.to_parquet(weeks_path, index=False)
    return games, weeks, True
