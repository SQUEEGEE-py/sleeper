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
    {"sleeper_id": sid, "name": name, "positions": pos, "vor": 0.0, "season_value": value, "p_play": 0.9,
     "nba_player_id": nba}
    for sid, name, pos, value, nba in [
        ("s1", "Starter", "('C',)", 50.0, 1), ("b1", "Bench", "('C',)", 30.0, 2),
        ("ir1", "Injured", "('PF',)", 45.0, 3), ("fa1", "Free", "('PG',)", 40.0, 4),
        ("fa2", "Dropped", "('SF',)", 42.0, 5), ("theirs", "Taken", "('SF',)", 60.0, 6),
        ("fa3", "Teamless", "('SF',)", 60.0, 7)]
]).set_index("sleeper_id")
SEASON_LOGS = pd.DataFrame({"nba_player_id": [4, 4, 4], "minutes": [30.0, 32.0, 34.0], "fp": [30.0, 40.0, 35.0]})


def run(mine: Roster, players: dict, transactions: list | None = None, config: dict = CONFIG):
    other = roster(1, ["theirs"], ["theirs"])
    return roster_moves(config, mine, [mine, other], players, POOL, {}, TODAY, transactions or [], NOW,
                        SEASON_LOGS)


def test_rostered_and_teamless_players_are_never_pickups() -> None:
    mine = roster(4, ["s1", "b1"], ["s1"])
    players = {pid: player(pid) for pid in POOL.index} | {"fa3": player("fa3", team=None)}
    moves = run(mine, players)
    names = [p.candidate.name for p in moves.pickups]
    # "Injured" is on nobody's roster and healthy here, so he is a pickup too; "Taken" and
    # "Teamless" (value 60) never are. Two players on a 13-spot roster: no drops.
    assert names == ["Injured", "Dropped", "Free"] and all(p.drop is None for p in moves.pickups)
    assert moves.open_spots == len(CONFIG["roster_positions"]) - 2
    assert moves.waiver_position == 7 and not moves.ir
    free = next(p for p in moves.pickups if p.candidate.name == "Free")
    assert "this season 3 g, 32.0 min, 35.0 fp" in free.describe()


def test_ir_players_take_no_roster_spot_and_a_full_roster_means_a_drop() -> None:
    # A three-spot league: two active plus one in IR leaves a spot open; three active fill it.
    small = CONFIG | {"roster_positions": ["C", "UTIL", "BN"]}
    players = {pid: player(pid) for pid in POOL.index} | {"ir1": player("ir1", status="Out")}
    with_ir = run(roster(4, ["s1", "b1", "ir1"], ["s1"], reserve=["ir1"]), players, config=small)
    assert with_ir.open_spots == 1 and all(p.drop is None for p in with_ir.pickups)
    full = run(roster(4, ["s1", "b1", "fa1"], ["s1"]), players, config=small)
    assert full.open_spots == 0 and full.pickups and all(p.drop in ("Pb1", "Pfa1") for p in full.pickups)


def test_no_backup_at_a_position_is_a_cover_alert() -> None:
    one_c = CONFIG | {"roster_positions": ["C", "C", "BN"]}
    players = {pid: player(pid) for pid in POOL.index}
    moves = run(roster(4, ["s1", "b1"], ["s1", "b1"]), players, config=one_c)
    (alert,) = moves.cover
    assert alert.position == "C" and set(alert.eligible) == {"Ps1", "Pb1"}


def test_ir_players_are_not_active_and_a_recovered_one_must_come_off() -> None:
    mine = roster(4, ["s1", "b1", "ir1"], ["s1"], reserve=["ir1"])
    players = {pid: player(pid) for pid in POOL.index} | {"s1": player("s1", status="Out")}
    moves = run(mine, players)
    assert sorted((a.kind, a.player) for a in moves.ir) == [("activate", "Pir1"), ("stash", "Ps1")]
    # The IR player takes no roster spot and is no cover: the out starter counts as absent.
    assert moves.open_spots == len(CONFIG["roster_positions"]) - 2
    assert all(p.drop is None for p in moves.pickups)


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
    assert any("Pghost" in gap for gap in moves.gaps)


def test_trade_report_leaves_ir_players_out_and_stops_after_the_deadline() -> None:
    from engine.ingest.sleeper import ScheduledGame
    from engine.jobs.nightly import trade_report
    pool = pd.DataFrame([
        {"sleeper_id": sid, "name": sid, "positions": pos, "season_value": v, "sleeper_rank": 50.0}
        for sid, pos, v in [("g", "('PG',)", 40), ("f1", "('SF',)", 45), ("f2", "('PF',)", 44),
                            ("c", "('C',)", 30), ("star", "('C',)", 60), ("G", "('PG',)", 40),
                            ("F", "('SF',)", 31), ("C1", "('C',)", 43), ("C2", "('C',)", 42)]
    ]).set_index("sleeper_id")
    slots = ["G#1", "F#1", "C#1"]
    games = [ScheduledGame(game_id="x", date="2027-02-28", week=19, status="pre_game", home="PHI", away="BOS")]
    mine = roster(4, ["g", "f1", "f2", "c", "star"], ["g"], reserve=["star"])
    theirs = roster(1, ["G", "F", "C1", "C2"], ["G"])
    report = trade_report(CONFIG, 4, [mine, theirs], {4: "me", 1: "rival"}, pool, slots, games, 1, TODAY)
    assert report.trades and all("star" not in {p.player_id for p in t.give} for t in report.trades)
    assert report.deadline.last_day == date(2027, 2, 28) and report.deadline.weeks_left == 19
    after = trade_report(CONFIG, 4, [mine, theirs], {4: "me", 1: "rival"}, pool, slots, games, 20, TODAY)
    assert after.deadline.passed and after.trades == [] and not after.post


def test_an_out_active_player_is_no_cover() -> None:
    # He has not been moved to IR yet; until then he still blocks a spot but plays no games.
    one_c = CONFIG | {"roster_positions": ["C", "C", "BN"]}
    players = {pid: player(pid) for pid in POOL.index} | {"s1": player("s1", status="Out")}
    (alert,) = run(roster(4, ["s1", "b1"], ["s1", "b1"]), players, config=one_c).cover
    assert alert.p_empty > 0.9
