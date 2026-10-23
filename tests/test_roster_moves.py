"""IR alerts and pickup ranking. No network. ALLOWED mirrors config/league.json's ir statuses."""

from __future__ import annotations

import json
from pathlib import Path

from engine.model.roster_moves import (
    Candidate,
    RosterPlayer,
    ir_alerts,
    ir_eligibility,
    rank_pickups,
    unavailable,
)

CONFIG = json.loads(Path("config/league.json").read_text(encoding="utf-8"))
ALLOWED = CONFIG["ir"]["eligible_statuses"]
SLOTS = CONFIG["ir"]["slots"]


def p(name: str, status: str | None = None) -> RosterPlayer:
    return RosterPlayer(name.lower(), name, status)


def test_eligibility_follows_the_league_settings() -> None:
    assert ALLOWED == ["OUT", "SUS", "DNR"] and SLOTS == 2
    assert ir_eligibility("Out", ALLOWED) == "yes"
    assert ir_eligibility("Sus", ALLOWED) == ir_eligibility("DNR", ALLOWED) == "yes"
    assert ir_eligibility("GTD", ALLOWED) == ir_eligibility("DTD", ALLOWED) == "no"
    assert ir_eligibility(None, ALLOWED) == "no"
    assert ir_eligibility("Out", ["SUS"]) == "no"  # a league that did not allow OUT
    assert ir_eligibility("IR", ALLOWED) == "assumed"
    assert ir_eligibility("Covid", ALLOWED) == "unknown"


def test_an_out_starter_is_stashed_while_a_slot_is_free() -> None:
    alerts = ir_alerts([p("Durant", "Out"), p("Jokic")], [], SLOTS, ALLOWED)
    assert [(a.kind, a.player) for a in alerts] == [("stash", "Durant")]
    assert "no drop needed" in alerts[0].text


def test_slots_run_out() -> None:
    alerts = ir_alerts([p("A", "Out"), p("B", "Out")], [p("C", "Out")], SLOTS, ALLOWED)
    assert [(a.kind, a.player) for a in alerts] == [("stash", "A"), ("full", "B")]


def test_a_recovered_player_must_come_off_ir() -> None:
    alerts = ir_alerts([], [p("Kawhi", None), p("Lillard", "DTD"), p("Kyrie", "Out")], SLOTS, ALLOWED)
    assert [(a.kind, a.player) for a in alerts] == [("activate", "Kawhi"), ("activate", "Lillard")]
    assert "healthy" in alerts[0].text and "DTD" in alerts[1].text


def test_unrecognised_status_is_reported_not_guessed() -> None:
    (alert,) = ir_alerts([p("X", "Covid")], [], SLOTS, ALLOWED)
    assert alert.kind == "unrecognised" and "Covid" in alert.text


def test_sleeper_ir_designation_is_flagged_as_assumed() -> None:
    (alert,) = ir_alerts([p("X", "IR")], [], SLOTS, ALLOWED)
    assert alert.kind == "stash" and "assumed eligible" in alert.text


def c(name: str, vor: float, status: str | None = None, waivers: bool = False) -> Candidate:
    return Candidate(name.lower(), name, "UTA", ("SF",), vor, status, waivers)


def test_pickups_beat_the_weakest_droppable_player_by_the_margin() -> None:
    bench = [("DeRozan", -1.9), ("Vucevic", -3.6), ("Bey", 2.1)]
    picks = rank_pickups([c("Small", -2.0), c("Big", 1.5), c("Huge", 4.0, waivers=True)], bench)
    assert [(x.candidate.name, x.drop, round(x.gain, 1)) for x in picks] == [("Huge", "Vucevic", 7.6), ("Big", "Vucevic", 5.1)]
    assert "claim on waivers" in picks[0].describe() and "add now" in picks[1].describe()


def test_unavailable_free_agents_are_not_pickups() -> None:
    assert unavailable("Out") and unavailable("IR") and not unavailable("GTD") and not unavailable(None)
    assert rank_pickups([c("Hurt", 9.0, "Out")], [("Bey", 0.0)]) == []
    assert rank_pickups([c("Q", 9.0, "GTD")], [("Bey", 0.0)])[0].candidate.name == "Q"


def test_no_bench_no_pickups() -> None:
    assert rank_pickups([c("Huge", 9.0)], []) == []


def test_waiver_reason() -> None:
    from datetime import datetime, timezone
    from engine.model.roster_moves import waiver_reason
    now = datetime(2026, 10, 22, 20, 0, tzinfo=timezone.utc)
    dropped = {"a": datetime(2026, 10, 22, 2, 0, tzinfo=timezone.utc), "b": datetime(2026, 10, 20, 2, 0, tzinfo=timezone.utc)}
    clear = CONFIG["waivers"]["clear_days"]
    assert "dropped" in waiver_reason("a", "UTA", dropped, now, clear, set(), True)
    assert waiver_reason("b", "UTA", dropped, now, clear, set(), True) is None  # cleared
    assert "game today has started" in waiver_reason("c", "PHI", {}, now, clear, {"PHI"}, True)
    assert waiver_reason("c", "PHI", {}, now, clear, {"PHI"}, False) is None
    assert waiver_reason("c", None, {}, now, clear, {"PHI"}, True) is None
