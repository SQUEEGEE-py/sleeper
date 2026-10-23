"""nightly.roster_moves: live-shaped Sleeper data -> IR alerts and pickups. No network."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from engine.ingest.sleeper import Player, Roster, Transaction
from engine.jobs.nightly import roster_moves

CONFIG = json.loads(Path("config/league.json").read_text(encoding="utf-8"))
NOW = datetime(2026, 10, 22, 20, 0, tzinfo=timezone.utc)
TODAY = date(2026, 10, 22)


def roster(rid: int, players: list[str], starters: list[str], reserve: list[str] | None = None) -> Roster:
    return Roster.model_validate({"roster_id": rid, "owner_id": f"u{rid}", "league_id": "L", "players": players,
                                  "starters": starters, "reserve": reserve, "settings": {"waiver_position": 7}})


def player(pid: str, team: str | None = "UTA", status: str | None = None) -> Player:
    return Player(player_id=pid, full_name=f"P{pid}", team=team, injury_status=status)


POOL = pd.DataFrame([
    {"sleeper_id": "s1", "name": "Starter", "positions": "('C',)", "vor": 20.0, "nba_player_id": 1},
    {"sleeper_id": "b1", "name": "Bench", "positions": "('C',)", "vor": -3.0, "nba_player_id": 2},
    {"sleeper_id": "ir1", "name": "Injured", "positions": "('PF',)", "vor": 9.0, "nba_player_id": 3},
    {"sleeper_id": "fa1", "name": "Free", "positions": "('PG',)", "vor": 1.0, "nba_player_id": 4},
    {"sleeper_id": "fa2", "name": "Dropped", "positions": "('SF',)", "vor": 2.0, "nba_player_id": 5},
    {"sleeper_id": "theirs", "name": "Taken", "positions": "('SF',)", "vor": 30.0, "nba_player_id": 6},
    {"sleeper_id": "fa3", "name": "Teamless", "positions": "('SF',)", "vor": 30.0, "nba_player_id": 7},
]).set_index("sleeper_id")


def run(mine: Roster, players: dict, transactions: list | None = None):
    other = roster(1, ["theirs"], ["theirs"])
    return roster_moves(CONFIG, mine, [mine, other], players, POOL, {}, TODAY, transactions or [], NOW)


def test_rostered_and_teamless_players_are_never_pickups() -> None:
    mine = roster(4, ["s1", "b1"], ["s1"])
    players = {pid: player(pid) for pid in POOL.index} | {"fa3": player("fa3", team=None)}
    moves = run(mine, players)
    names = [p.candidate.name for p in moves.pickups]
    # "Injured" is on nobody's roster and healthy here, so he is a pickup too; "Taken" and
    # "Teamless" (vor 30) never are.
    assert names == ["Injured", "Dropped", "Free"] and all(p.drop == "Pb1" for p in moves.pickups)
    assert moves.waiver_position == 7 and not moves.ir


def test_ir_players_are_not_active_and_a_recovered_one_must_come_off() -> None:
    mine = roster(4, ["s1", "b1", "ir1"], ["s1"], reserve=["ir1"])
    players = {pid: player(pid) for pid in POOL.index} | {"s1": player("s1", status="Out")}
    moves = run(mine, players)
    assert sorted((a.kind, a.player) for a in moves.ir) == [("activate", "Pir1"), ("stash", "Ps1")]
    # The IR player is not "bench" to drop either.
    assert all(p.drop == "Pb1" for p in moves.pickups)


def test_a_player_already_in_ir_is_not_told_to_go_there() -> None:
    mine = roster(4, ["s1", "b1", "ir1"], ["s1"], reserve=["ir1"])
    players = {pid: player(pid) for pid in POOL.index} | {"ir1": player("ir1", status="Out")}
    assert run(mine, players).ir == []


def test_a_fresh_drop_is_a_waiver_claim() -> None:
    mine = roster(4, ["s1", "b1"], ["s1"])
    players = {pid: player(pid) for pid in POOL.index}
    dropped = Transaction(type="free_agent", status="complete", drops={"fa2": 2},
                          status_updated=int(datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc).timestamp() * 1000))
    moves = run(mine, players, [dropped])
    by_name = {p.candidate.name: p.candidate for p in moves.pickups}
    assert by_name["Dropped"].on_waivers and not by_name["Free"].on_waivers


def test_a_bench_player_missing_from_the_pool_is_a_gap() -> None:
    mine = roster(4, ["s1", "ghost"], ["s1"])
    players = {pid: player(pid) for pid in POOL.index} | {"ghost": player("ghost")}
    moves = run(mine, players)
    assert moves.pickups == [] and any("Pghost" in gap for gap in moves.gaps)
