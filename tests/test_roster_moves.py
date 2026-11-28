"""IR alerts and pickup ranking. No network. ALLOWED mirrors config/league.json's ir statuses."""

from __future__ import annotations

import json
from pathlib import Path

from itertools import product

import pytest

from engine.model.lineup import is_eligible
from engine.model.roster_moves import (
    OPEN_SPOT_MIN_GAIN,
    PICKUP_MIN_GAIN,
    WEEK_OUT_BY_P_PLAY,
    Candidate,
    DepthPlayer,
    RosterPlayer,
    cover_alerts,
    ir_alerts,
    ir_eligibility,
    lineup_outlook,
    rank_pickups,
    unavailable,
    week_out_probability,
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


SLOTS_SMALL = ["G#1", "F#1", "C#1", "C#2"]


def d(name: str, pos: str, value: float, out: float = 0.05) -> DepthPlayer:
    return DepthPlayer(name.lower(), name, (pos,), value, out)


def c(name: str, pos: str, value: float, status: str | None = None, waivers: bool = False,
      p_play: float = 0.95) -> Candidate:
    return Candidate(name.lower(), name, "UTA", (pos,), value, p_play, status, waivers)


# Two C slots, exactly two centres -- the user's roster on 2026-11-15.
ROSTER = [d("Guard", "PG", 45), d("Guard2", "SG", 41), d("Wing", "SF", 44), d("Jokic", "C", 65), d("Bam", "C", 52), d("Bench", "SF", 39)]


def test_week_out_follows_the_measured_table() -> None:
    for p_play, rate in WEEK_OUT_BY_P_PLAY:
        assert week_out_probability(p_play) == pytest.approx(rate)
    assert week_out_probability(0.80) == pytest.approx(0.099 + (0.80 - 0.758) / (0.851 - 0.758) * (0.051 - 0.099))
    assert week_out_probability(0.30) == 0.204 and week_out_probability(1.0) == 0.010


def test_outlook_equals_brute_force_over_every_absence() -> None:
    from engine.model.lineup import lineup_value
    players = [d("A", "PG", 40, 0.2), d("B", "C", 50, 0.1), d("C2", "C", 30, 0.3), d("F", "SF", 35, 0.05)]
    slots = ["G#1", "C#1", "UTIL#1"]
    expected = p_c_empty = 0.0
    for outs in product([False, True], repeat=len(players)):
        w = 1.0
        for player, out in zip(players, outs):
            w *= player.week_out if out else 1 - player.week_out
        here = [p for p, out in zip(players, outs) if not out]
        expected += w * lineup_value([p.value for p in here], [p.positions for p in here], slots)
        p_c_empty += w * (not any(is_eligible(p.positions, "C#1") for p in here))
    outlook = lineup_outlook(players, slots, max_absent=len(players))
    assert outlook.expected == pytest.approx(expected) and outlook.p_unfilled["C"] == pytest.approx(p_c_empty)
    assert outlook.healthy == pytest.approx(50 + 40 + 35)
    # Truncated at two absent (the default) and renormalised: under 0.1% of the lineup at
    # realistic week-out rates, and nearly the same for every candidate compared.
    real = [DepthPlayer(p.player_id, p.name, p.positions, p.value, 0.05) for p in players] * 2
    assert lineup_outlook(real, slots).expected == pytest.approx(
        lineup_outlook(real, slots, max_absent=len(real)).expected, rel=0.001)


def test_players_certain_to_be_out_do_not_use_up_the_absence_budget() -> None:
    # 2026-11-28: three active Out players (week_out 1.0) with max_absent 2 left every
    # enumerated outcome at probability 0 and the report crashed dividing by zero.
    out = [d("Out1", "PG", 50, 1.0), d("Out2", "SF", 48, 1.0), d("Out3", "C", 47, 1.0)]
    rest = [d("A", "PG", 40, 0.2), d("B", "C", 45, 0.1), d("F", "SF", 35, 0.05)]
    slots = ["G#1", "C#1", "UTIL#1"]
    with_out = lineup_outlook(out + rest, slots)  # the default max_absent, which crashed
    without = lineup_outlook(rest, slots)
    assert with_out.expected == pytest.approx(without.expected)
    assert with_out.p_unfilled == pytest.approx(without.p_unfilled)
    exact = lineup_outlook(rest, slots, max_absent=len(rest))
    assert lineup_outlook(out + rest, slots, max_absent=len(rest)).expected == pytest.approx(exact.expected)
    assert with_out.healthy == pytest.approx(50 + 48 + 47)  # "with everyone available"
    everyone_out = lineup_outlook(out, slots)
    assert everyone_out.expected == 0.0 and everyone_out.p_unfilled["C"] == 1.0


def test_a_backup_at_the_uncovered_position_beats_a_better_player_elsewhere() -> None:
    # Neither improves the healthy lineup; the centre covers a slot that would otherwise sit
    # empty, the wing only improves on a 39 backup.
    picks = rank_pickups([c("Wing2", "SF", 43), c("Gobert", "C", 34)], ROSTER, {"bench"}, SLOTS_SMALL, open_spots=1,
                         open_min_gain=0.0)
    assert [p.candidate.name for p in picks] == ["Gobert", "Wing2"] and picks[0].gain > 5 * picks[1].gain
    pick = picks[0]
    assert pick.drop is None and pick.healthy_gain == pytest.approx(0.0)
    assert pick.gain == pytest.approx(34 * 0.05 * 0.95 * 2, rel=0.05)  # either centre out, the other in
    assert "into your open roster spot" in pick.describe() and "as cover" in pick.describe()


def test_with_no_open_spot_the_best_bench_drop_is_named_and_starters_are_never_dropped() -> None:
    star = c("Star", "PG", 60)
    (pick,) = rank_pickups([star], ROSTER, {"bench"}, SLOTS_SMALL, open_spots=0)
    assert pick.drop == "Bench" and pick.gain > PICKUP_MIN_GAIN
    assert "dropping Bench" in pick.describe()
    assert rank_pickups([star], ROSTER, set(), SLOTS_SMALL, open_spots=0) == []  # no bench: nothing to drop


def test_a_swap_has_a_higher_bar_than_an_add_into_an_open_spot() -> None:
    # A shade above the open-spot bar: listed with a spot free, not as a swap.
    small = [c("Backup", "C", 4.0)]
    open_gain = rank_pickups(small, ROSTER, {"bench"}, SLOTS_SMALL, open_spots=1)[0].gain
    assert OPEN_SPOT_MIN_GAIN <= open_gain < PICKUP_MIN_GAIN
    assert rank_pickups(small, ROSTER, {"bench"}, SLOTS_SMALL, open_spots=0) == []


def test_unavailable_free_agents_are_not_pickups() -> None:
    assert unavailable("Out") and unavailable("IR") and not unavailable("GTD") and not unavailable(None)
    assert rank_pickups([c("Hurt", "PG", 90, "Out")], ROSTER, {"bench"}, SLOTS_SMALL, 1) == []
    assert rank_pickups([c("Q", "PG", 90, "GTD")], ROSTER, {"bench"}, SLOTS_SMALL, 1)[0].candidate.name == "Q"


def test_cover_alert_for_a_position_with_no_backup() -> None:
    picks = rank_pickups([c("Gobert", "C", 34)], ROSTER, {"bench"}, SLOTS_SMALL, 1)
    (alert,) = cover_alerts(ROSTER, SLOTS_SMALL, picks)
    assert alert.position == "C" and alert.eligible == ("Jokic", "Bam")
    assert alert.p_empty == pytest.approx(1 - 0.95 ** 2, rel=0.03)  # truncated at two absent
    assert "No backup at C" in alert.describe() and "Best cover: Gobert" in alert.describe()
    covered = [*ROSTER, d("Gobert", "C", 34)]
    assert cover_alerts(covered, SLOTS_SMALL) == []  # empty only when two of three centres are out


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
