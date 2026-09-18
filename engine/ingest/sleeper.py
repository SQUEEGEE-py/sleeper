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
from pydantic import BaseModel, ConfigDict

BASE_URL = "https://api.sleeper.app/v1"
DEFAULT_CACHE_DIR = Path("data/cache/sleeper")

# players/nba is a multi-MB payload Sleeper asks callers to fetch at most once a day.
PLAYERS_CACHE_TTL_SECONDS = 24 * 60 * 60


class League(BaseModel):
    model_config = ConfigDict(extra="allow")

    league_id: str
    name: str
    season: str
    sport: str
    total_rosters: int
    roster_positions: list[str]
    scoring_settings: dict[str, float]


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
        response = self._http.get(path)
        response.raise_for_status()
        data = response.json()
        _write_cache(self.cache_dir, cache_key, data)
        return data

    def get_league(self, league_id: str) -> League:
        data = self._get(f"/league/{league_id}", f"league_{league_id}")
        return League.model_validate(data)

    def get_rosters(self, league_id: str) -> list[Roster]:
        data = self._get(f"/league/{league_id}/rosters", f"rosters_{league_id}")
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

    def get_players(self) -> dict[str, Player]:
        data = self._get(
            "/players/nba", "players_nba", ttl_seconds=PLAYERS_CACHE_TTL_SECONDS
        )
        return {pid: Player.model_validate(p) for pid, p in data.items()}
