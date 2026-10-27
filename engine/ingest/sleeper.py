"""Typed, read-only client for the Sleeper API endpoints listed in docs/SPEC.md §2.1.

Every response is cached to disk (JSON, alongside a fetch timestamp) so tests and repeated
runs never need the network. Sleeper is public and unauthenticated; we still stay well under
their documented rate limit by caching aggressively rather than by throttling.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from engine.ingest.http_retry import request_with_retry
from pydantic import BaseModel, ConfigDict

BASE_URL = "https://api.sleeper.app/v1"
# Undocumented, but the source of Sleeper's own ADP. Probed 2026-10-16 (rule 6):
# GET /projections/nba/{season}?season_type=regular -> list of ~2130 rows, each with
# ``player_id`` and a ``stats`` dict whose ``adp_std`` is redraft ADP (``adp_dynasty`` is the
# dynasty one). The 2026 rows were updated that day; the 2025 rows were last touched
# June-July 2026, after the season, so treat historical ADP as a close but unproven snapshot.
# **A past season's stat "projections" are actuals, overwritten after the season**: 2025 rows
# match real 2025-26 per-game output at r = 1.000 for rookies and 0.992 for veterans. Never
# backtest anything against them. ADP looks genuinely pre-season (Knueppel 132 before his
# breakout, unranked players at 999), which is what the sigma calibration relies on.
PROJECTIONS_URL = "https://api.sleeper.com/projections/nba"
SLEEPER_STATS_URL = "https://api.sleeper.com/stats/nba"
SLEEPER_SCHEDULE_URL = "https://api.sleeper.com/schedule/nba/regular"
ADP_CACHE_TTL_SECONDS = 6 * 60 * 60
DEFAULT_CACHE_DIR = Path("data/cache/sleeper")

# players/nba is a multi-MB payload Sleeper asks callers to fetch at most once a day.
PLAYERS_CACHE_TTL_SECONDS = 24 * 60 * 60
# For injury statuses: the nightly job runs twice a day, so this keeps it to about two
# fetches a day while never acting on a status more than an hour old.
INJURY_STATUS_TTL_SECONDS = 60 * 60


class League(BaseModel):
    model_config = ConfigDict(extra="allow")

    league_id: str
    name: str
    season: str
    sport: str
    total_rosters: int
    roster_positions: list[str]
    scoring_settings: dict[str, float]
    # League rules outside scoring and the active slots: reserve_slots (IR), waiver_*,
    # playoff_*, trade_deadline, divisions. IR is here, never in roster_positions.
    settings: dict[str, Any] = {}


class Roster(BaseModel):
    model_config = ConfigDict(extra="allow")

    roster_id: int
    owner_id: str | None
    league_id: str
    players: list[str] | None
    starters: list[str] | None


class User(BaseModel):
    model_config = ConfigDict(extra="allow")

    user_id: str
    display_name: str
    league_id: str | None = None


class Matchup(BaseModel):
    model_config = ConfigDict(extra="allow")

    roster_id: int
    matchup_id: int | None
    points: float | None
    starters: list[str] | None
    players_points: dict[str, float] | None = None
    starters_points: list[float] | None = None


class Transaction(BaseModel):
    """One add/drop/claim/trade. Probed 2026-10-22: a waiver claim carries ``settings.seq``
    (priority order) and no bid; ``status_updated`` (ms) is when waivers processed it -- the
    one claim so far was processed Wed 2026-10-21 03:08 ET, just after midnight Pacific."""

    model_config = ConfigDict(extra="allow")

    type: str
    status: str
    status_updated: int | None = None
    adds: dict[str, int] | None = None
    drops: dict[str, int] | None = None


class PlayerGame(BaseModel):
    """One game of a player's Sleeper stats. Keys are Sleeper's scoring keys."""

    week: int
    date: str  # "YYYY-MM-DD", US date of the game
    stats: dict[str, float | None]


class ScheduledGame(BaseModel):
    game_id: str
    date: str
    week: int
    status: str  # pre_game | in_game | complete
    home: str
    away: str


class Draft(BaseModel):
    model_config = ConfigDict(extra="allow")

    draft_id: str
    league_id: str | None = None
    status: str
    draft_order: dict[str, int] | None = None


class DraftPick(BaseModel):
    model_config = ConfigDict(extra="allow")

    pick_no: int
    round: int
    roster_id: int | None
    player_id: str
    draft_id: str


class ProjectionRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    player_id: str
    stats: dict[str, float | None] = {}


class Player(BaseModel):
    model_config = ConfigDict(extra="allow")

    player_id: str
    full_name: str | None = None
    position: str | None = None
    fantasy_positions: list[str] | None = None
    team: str | None = None
    status: str | None = None
    injury_status: str | None = None
    active: bool | None = None
    birth_date: str | None = None  # "YYYY-MM-DD"; absent for some players


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def _read_cache(cache_dir: Path, key: str, ttl_seconds: float | None) -> Any | None:
    path = _cache_path(cache_dir, key)
    if not path.exists():
        return None
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if ttl_seconds is not None and (time.time() - envelope["fetched_at"]) > ttl_seconds:
        return None
    return envelope["data"]


def _write_cache(cache_dir: Path, key: str, data: Any) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, key)
    envelope = {"fetched_at": time.time(), "data": data}
    path.write_text(json.dumps(envelope), encoding="utf-8")


class SleeperClient:
    """Read-only Sleeper API client. Sleeper has no write endpoints for league data;
    this tool never mutates league state regardless."""

    def __init__(
        self,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        base_url: str = BASE_URL,
        timeout: float = 15.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "SleeperClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _get(self, path: str, cache_key: str, ttl_seconds: float | None = None) -> Any:
        cached = _read_cache(self.cache_dir, cache_key, ttl_seconds)
        if cached is not None:
            return cached
        response = request_with_retry(lambda: self._http.get(path))
        data = response.json()
        _write_cache(self.cache_dir, cache_key, data)
        return data

    def get_user(self, username: str) -> User:
        """A Sleeper user by username. Probed 2026-10-16: ``/user/{username}`` returns
        ``user_id``, ``username`` and ``display_name``; no ``league_id``."""
        data = self._get(f"/user/{username}", f"user_{username.lower()}")
        return User.model_validate(data)

    def get_league(self, league_id: str) -> League:
        data = self._get(f"/league/{league_id}", f"league_{league_id}")
        return League.model_validate(data)

    def get_league_live(self, league_id: str) -> League:
        """The league, never from cache. ``get_league`` caches forever, so verify_league would
        otherwise keep checking the config against whatever was fetched first."""
        return League.model_validate(self._get_fresh(f"/league/{league_id}", f"league_{league_id}"))

    def get_rosters(self, league_id: str) -> list[Roster]:
        data = self._get(f"/league/{league_id}/rosters", f"rosters_{league_id}")
        return [Roster.model_validate(r) for r in data]

    def get_rosters_live(self, league_id: str) -> list[Roster]:
        """Rosters, never from cache. ``get_rosters`` caches forever, which was fine before the
        draft and is wrong in-season: waivers, trades and lineup changes move them daily.
        ``starters`` is ordered like ``roster_positions`` (PG, SG, G, SF, PF, F, C, C, UTIL)."""
        data = self._get_fresh(f"/league/{league_id}/rosters", f"rosters_{league_id}")
        return [Roster.model_validate(r) for r in data]

    def get_users(self, league_id: str) -> list[User]:
        data = self._get(f"/league/{league_id}/users", f"users_{league_id}")
        return [User.model_validate(u) for u in data]

    def get_matchups(self, league_id: str, week: int) -> list[Matchup]:
        data = self._get(
            f"/league/{league_id}/matchups/{week}", f"matchups_{league_id}_{week}"
        )
        return [Matchup.model_validate(m) for m in data]

    def get_drafts(self, league_id: str) -> list[Draft]:
        data = self._get(f"/league/{league_id}/drafts", f"drafts_{league_id}")
        return [Draft.model_validate(d) for d in data]

    def get_draft(self, draft_id: str) -> Draft:
        data = self._get(f"/draft/{draft_id}", f"draft_{draft_id}")
        return Draft.model_validate(data)

    def get_draft_picks(self, draft_id: str) -> list[DraftPick]:
        data = self._get(f"/draft/{draft_id}/picks", f"draft_picks_{draft_id}")
        return [DraftPick.model_validate(p) for p in data]

    def _get_fresh(self, path: str, cache_key: str) -> Any:
        """Always hit the network, then cache. For state that changes by the second."""
        response = request_with_retry(lambda: self._http.get(path))
        data = response.json()
        _write_cache(self.cache_dir, cache_key, data)
        return data

    def get_draft_live(self, draft_id: str) -> Draft:
        """The draft object, never from cache: draft_order appears when the draft starts."""
        return Draft.model_validate(self._get_fresh(f"/draft/{draft_id}", f"draft_{draft_id}"))

    def get_draft_picks_live(self, draft_id: str) -> list[DraftPick]:
        """Picks so far, never from cache. ``get_draft_picks`` caches forever, which is right
        for a completed draft and wrong for one in progress."""
        data = self._get_fresh(f"/draft/{draft_id}/picks", f"draft_picks_{draft_id}")
        return [DraftPick.model_validate(p) for p in data]

    def get_matchups_live(self, league_id: str, week: int) -> list[Matchup]:
        """This week's matchups, never from cache: ``starters_points`` moves with every game.
        There is no lock field (probed 2026-10-21); see ``engine/model/locks.py``."""
        data = self._get_fresh(f"/league/{league_id}/matchups/{week}", f"matchups_{league_id}_{week}")
        return [Matchup.model_validate(m) for m in data]

    def get_transactions_live(self, league_id: str, week: int) -> list[Transaction]:
        """A week's transactions, never from cache: who was dropped, and so is on waivers."""
        data = self._get_fresh(f"/league/{league_id}/transactions/{week}", f"transactions_{league_id}_{week}")
        return [Transaction.model_validate(t) for t in data]

    def get_player_games(self, player_id: str, season: str) -> list[PlayerGame]:
        """Every regular-season game a player has stats for, dated, never from cache.

        ``api.sleeper.com/stats/nba/player/{id}?grouping=week`` (probed 2026-10-21) returns
        ``{week: [game, ...]}``. A game he did not play has no entry. Works from cloud IPs,
        unlike stats.nba.com, and carries ``tf``/``ff``, which stats.nba.com's logs lack.
        """
        data = self._get_fresh(
            f"{SLEEPER_STATS_URL}/player/{player_id}?season_type=regular&season={season}&grouping=week",
            f"player_games_{season}_{player_id}",
        )
        return [
            PlayerGame(week=int(week), date=game["date"], stats=game.get("stats") or {})
            for week, games in (data or {}).items()
            for game in (games or [])
            if game and game.get("date")
        ]

    def get_nba_schedule(self, season: str) -> list[ScheduledGame]:
        """The regular-season schedule with live status (``pre_game`` / ``in_game`` /
        ``complete``), never from cache. NBA tricodes; ``week`` is the fantasy week. No tip-off
        times -- those come from ESPN."""
        data = self._get_fresh(f"{SLEEPER_SCHEDULE_URL}/{season}", f"nba_schedule_{season}")
        return [
            ScheduledGame(game_id=g["game_id"], date=g["date"], week=int(g["week"]), status=g["status"],
                          home=g["home"]["team"], away=g["away"]["team"])
            for g in data
        ]

    def get_adp(self, season: str) -> dict[str, float]:
        """Sleeper redraft ADP (``adp_std``) by Sleeper player id. Players without one are
        absent, not defaulted -- the caller decides what an unranked player means."""
        key = f"projections_{season}"
        data = _read_cache(self.cache_dir, key, ADP_CACHE_TTL_SECONDS)
        if data is None:
            response = request_with_retry(lambda: self._http.get(
                f"{PROJECTIONS_URL}/{season}", params={"season_type": "regular"}
            ))
            data = response.json()
            _write_cache(self.cache_dir, key, data)
        rows = [ProjectionRow.model_validate(row) for row in data]
        return {
            row.player_id: float(row.stats["adp_std"])
            for row in rows
            if row.stats.get("adp_std") is not None
        }

    def get_players(self, ttl_seconds: float = PLAYERS_CACHE_TTL_SECONDS) -> dict[str, Player]:
        """The player dictionary. Cached a day by default -- fine for names and teams, too stale
        for injury statuses: on 2026-10-27 a day-old copy still had Kawhi as DTD after he was
        ruled Out, so the IR alert stayed silent. Callers that act on statuses pass
        ``INJURY_STATUS_TTL_SECONDS``."""
        data = self._get("/players/nba", "players_nba", ttl_seconds=ttl_seconds)
        return {pid: Player.model_validate(p) for pid, p in data.items()}
