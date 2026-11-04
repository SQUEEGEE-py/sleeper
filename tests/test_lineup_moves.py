"""lineup_moves: mid-week lineup changes under Lock-In. Closed-form normal(45, 10) values:
45.0 / 48.989 / 51.297 for 1 / 2 / 3 games. No network."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from engine.jobs.lock_sheet import GameSlot
from engine.jobs.nightly import Starter, lineup_after, lineup_options
from engine.ingest.sleeper import Player
from engine.model.distributions import NormalScoreDistribution
from engine.model.lineup_moves import LineupOption, move_in_value, plan_lineup, stay_value
from engine.model.locks import LockStatus
from engine.model.matchup import Slot

N45 = NormalScoreDistribution(45.0, 10.0, 1.0)
WED = date(2026, 11, 4)
THU = date(2026, 11, 5)


def tip(day: date, hour: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, 0, tzinfo=timezone.utc)


def opt(pid: str, value: float, positions=("SF",), **kw) -> LineupOption:
    return LineupOption(pid, pid.upper(), tuple(positions), value, **kw)


# --- values ---------------------------------------------------------------------------------


def test_stay_value_keeps_the_pending_score_and_the_roll_on() -> None:
    assert stay_value(Slot("a", N45, (1.0,), pending=60.0)) == 60.0  # lock it: beats 45
    assert stay_value(Slot("a", N45, (1.0,), pending=30.0)) == pytest.approx(45.0)  # roll on
    assert stay_value(Slot("a", N45, (1.0, 1.0, 1.0))) == pytest.approx(51.297, abs=0.01)
    assert stay_value(Slot("a", N45, (1.0,), locked=38.0)) == 38.0
    assert stay_value(Slot("a", N45, (), pending=22.0)) == 22.0  # week over: the last game counts


def test_move_in_value_counts_only_games_left() -> None:
    assert move_in_value(Slot("b", N45, (1.0, 1.0, 1.0))) == pytest.approx(51.297, abs=0.01)
    assert move_in_value(Slot("b", N45, (1.0, 1.0))) == pytest.approx(48.989, abs=0.01)
    assert move_in_value(Slot("b", N45, (0.0, 1.0))) == pytest.approx(45.0)  # Out next game
    assert move_in_value(Slot("b", None, ())) == 0.0


# --- the plan -------------------------------------------------------------------------------


def test_bench_player_with_more_games_replaces_a_starter() -> None:
    starter = opt("s", 45.0, next_game=THU, next_tip=tip(THU, 23))
    bench = opt("b", 51.3, next_game=WED, next_tip=tip(WED, 23), days=("Wed", "Fri", "Sun"))
    plan = plan_lineup(["F#1"], [starter], [bench])
    (change,) = plan.changes
    assert change.into is bench and change.out is starter and change.from_bench
    assert plan.to_bench == [starter] and plan.gain == pytest.approx(6.3)
    # Before whichever game tips first -- here the bench player's, Wednesday.
    assert change.deadline == (tip(WED, 23), WED)
    assert change.describe() == "start B at F (Wed, Fri, Sun), in place of S"


def test_a_good_pending_score_is_not_thrown_away() -> None:
    # 55 showing with one game left: staying is worth 55, more than the bench player's 51.3.
    starter = opt("s", stay_value(Slot("s", N45, (1.0,), pending=55.0)))
    plan = plan_lineup(["F#1"], [starter], [opt("b", move_in_value(Slot("b", N45, (1.0, 1.0, 1.0))))])
    assert plan.changes == [] and plan.gain == pytest.approx(0.0)


def test_fixed_starters_are_never_moved() -> None:
    locked = opt("l", 20.0, movable=False, fixed_reason="locked")
    playing = opt("p", 10.0, movable=False, fixed_reason="in a game now")
    plan = plan_lineup(["F#1", "UTIL#1"], [locked, playing], [opt("b", 50.0)])
    assert plan.changes == [] and plan.assignment == {"F#1": locked, "UTIL#1": playing}
    # A bench player in a game cannot come in either.
    plan = plan_lineup(["F#1"], [opt("s", 30.0)], [opt("b", 50.0, movable=False)])
    assert plan.changes == []


def test_small_gains_and_ties_are_not_changes() -> None:
    plan = plan_lineup(["F#1"], [opt("s", 45.0)], [opt("b", 45.6)])
    assert plan.changes == [] and plan.gain == pytest.approx(0.6)
    # Two flexible starters of equal value: no reshuffle between F and UTIL.
    a, b = opt("a", 40.0, ("SF", "PF")), opt("b", 40.0, ("SF", "PF"))
    assert plan_lineup(["F#1", "UTIL#1"], [a, b], [opt("c", 39.0)]).changes == []
    # A real gain makes exactly the one change it needs, not a reshuffle of equal starters.
    slots = ["G#1", "F#1", "C#1", "UTIL#1"]
    starters = [opt("g", 30.0, ("SG", "SF")), opt("f", 30.0, ("SF", "PF")), opt("c", 30.0, ("PF", "C")),
                opt("u", 30.0, ("C",))]
    plan = plan_lineup(slots, starters, [opt("b", 45.0, ("SG", "SF"))])
    assert [(c.slot, c.out.player_id, c.into.player_id) for c in plan.changes] == [("UTIL#1", "u", "b")]


def test_a_starter_moves_slots_to_make_room() -> None:
    # A guard comes off the bench; the flexible starter shifts G -> F; the pure forward sits.
    flex = opt("a", 40.0, ("SG", "SF"))
    forward = opt("f", 30.0, ("PF",))
    guard = opt("g", 45.0, ("PG",))
    plan = plan_lineup(["G#1", "F#1"], [flex, forward], [guard])
    assert {(c.slot, c.into.player_id, c.from_bench) for c in plan.changes} == {("G#1", "g", True), ("F#1", "a", False)}
    assert plan.to_bench == [forward] and plan.gain == pytest.approx(15.0)
    assert any(c.describe().startswith("move A to F") for c in plan.changes)


def test_empty_slots_are_filled_and_ineligible_players_are_not_used() -> None:
    plan = plan_lineup(["C#1", "G#1"], [None, None], [opt("c", 30.0, ("C",)), opt("f", 50.0, ("SF",))])
    assert [(c.slot, c.into.player_id, c.out) for c in plan.changes] == [("C#1", "c", None)]
    assert plan.assignment["G#1"] is None  # the forward cannot play G


# --- nightly glue ---------------------------------------------------------------------------


def starter(pid: str, slot: Slot | None, **kw) -> Starter:
    return Starter(pid, pid.upper(), "PHI", kw.pop("counted", 0.0), slot=slot, **kw)


def test_lineup_options_fix_locked_playing_and_finished_starters() -> None:
    players = {p: Player(player_id=p, fantasy_positions=["SF"]) for p in "lpdo"}
    mine = [
        starter("l", Slot("l", N45, (1.0,), locked=48.5), lock=LockStatus("locked", None, 48.5)),
        starter("p", Slot("p", N45, (1.0, 1.0)), in_progress=True, counted=12.0),
        starter("d", Slot("d", N45, (), pending=31.0), completed=[(WED, 31.0)], counted=31.0),
        starter("o", Slot("o", N45, (1.0, 1.0), pending=20.0), completed=[(WED, 20.0)]),
        Starter("0", "(empty slot)", "", 0.0, slot=Slot("(empty)", None)),
    ]
    mine[3].upcoming = [GameSlot(THU, "BOS", True, False), GameSlot(date(2026, 11, 6), "NYK", False, False)]
    starters, _ = lineup_options(mine, [], players)
    assert [(s.movable, s.fixed_reason) for s in starters[:4]] == [
        (False, "locked"), (False, "in a game now"), (False, "his week is over; his last game counts"), (True, "")]
    assert starters[0].value == 48.5 and starters[4] is None
    assert starters[3].next_game == THU and starters[3].days == ("Thu", "Fri")
    assert starters[3].value == pytest.approx(48.989, abs=0.01)  # 2 games to roll on beat 20


def test_lineup_options_bench_counts_games_left_and_skips_players_mid_game() -> None:
    bench = [
        (starter("b", Slot("b", N45, (1.0, 1.0, 1.0), pending=70.0)), ("SF",)),  # pending ignored
        (starter("m", Slot("m", N45, (1.0,)), in_progress=True), ("SF",)),
    ]
    _, options = lineup_options([], bench, {})
    (only,) = options
    assert only.player_id == "b" and only.value == pytest.approx(51.297, abs=0.01)


def test_lineup_after_brings_bench_players_in_fresh() -> None:
    mine = [starter("s", Slot("s", N45, (1.0,), pending=10.0))]
    bench = [(starter("b", Slot("b", N45, (1.0, 1.0), pending=70.0)), ("SF",))]
    starters, options = lineup_options(mine, bench, {"s": Player(player_id="s", fantasy_positions=["SF"])})
    plan = plan_lineup(["F#1"], starters, options)
    (slot,) = lineup_after(plan, ["F#1"], mine, bench)
    assert slot.name == "B" and slot.pending is None and slot.remaining_p_play == (1.0, 1.0)


# --- auditions: a bench player who plays first, for one game ----------------------------------

from scipy.stats import norm  # noqa: E402

from engine.model.lineup_moves import audition_value, plan_auditions, plays_first  # noqa: E402

FRI = date(2026, 11, 6)


def e_max(c: float, mu: float = 45.0, sigma: float = 10.0) -> float:
    """E[max(X, c)] for X ~ normal(mu, sigma), in closed form."""
    z = (c - mu) / sigma
    return c + sigma * (norm.pdf(z) - z * (1 - norm.cdf(z)))


def sim_opt(pid: str, games: tuple[float, ...], positions=("PG",), day=None, hour=23, pending=None) -> LineupOption:
    sim = Slot(pid, N45, games, pending=pending)
    value = stay_value(sim) if pending is not None else move_in_value(sim)
    return LineupOption(pid, pid.upper(), positions, value, next_game=day,
                        next_tip=tip(day, hour) if day else None, sim=sim)


def no_changes(slots, starters, bench):
    return plan_lineup(slots, starters, bench, min_gain=1e9)  # auditions on the lineup as it is


def test_a_bench_guard_tonight_stands_in_for_a_guard_who_plays_later() -> None:
    starter = sim_opt("s", (1.0, 1.0), day=FRI)  # two games from Friday: 48.989
    bench = sim_opt("b", (1.0,), day=THU)  # one game, tonight
    (a,) = plan_auditions(no_changes(["G#1"], [starter], [bench]), [bench])
    w2 = 48.989
    assert a.value == pytest.approx(e_max(w2), abs=0.01) and a.gain == pytest.approx(e_max(w2) - w2, abs=0.01)
    assert a.lock_at == pytest.approx(w2, abs=0.01) and not a.keep_bench
    text = a.describe("Thu 7:00 PM ET", "Fri 7:00 PM ET")
    assert "lock him if he scores 49.0+" in text and "put S back before Fri" in text and "nothing to lose" in text


def test_a_bench_player_who_might_sit_is_worth_less_and_never_negative() -> None:
    starter = sim_opt("s", (1.0, 1.0), day=FRI)
    for p, expected in [(0.5, 0.5 * e_max(48.989) + 0.5 * 48.989), (0.0, None)]:
        bench = sim_opt("b", (p,), day=THU)
        found = plan_auditions(no_changes(["G#1"], [starter], [bench]), [bench], min_gain=0.0)
        if expected is None:
            assert audition_value(bench.sim, starter.value) == pytest.approx(starter.value)
            assert [a.gain for a in found] == [pytest.approx(0.0)]  # Out tonight: worth exactly the starter
        else:
            assert found[0].value == pytest.approx(expected, abs=0.01)


def test_the_starter_must_play_after_the_bench_players_game_is_over() -> None:
    later = sim_opt("s", (1.0, 1.0), day=THU, hour=23)
    assert not plays_first(sim_opt("b", (1.0,), day=FRI), later)  # starter plays first
    assert not plays_first(sim_opt("b", (1.0,), day=THU, hour=21), later)  # same night, 2 h apart
    assert plays_first(sim_opt("b", (1.0,), day=THU, hour=20), later)  # 3 h: his game is over
    assert not plays_first(sim_opt("b", (1.0,), day=None), later)


def test_no_audition_that_throws_away_a_good_pending_score() -> None:
    starter = sim_opt("s", (1.0, 1.0), day=FRI, pending=60.0)  # stay: 60 showing
    bench = sim_opt("b", (1.0,), day=THU)
    assert plan_auditions(no_changes(["G#1"], [starter], [bench]), [bench]) == []
    # A small pending score: the audition is still worth it, and the cost is said out loud.
    starter = sim_opt("s", (1.0, 1.0), day=FRI, pending=10.0)
    (a,) = plan_auditions(no_changes(["G#1"], [starter], [bench]), [bench])
    assert "gives up the 10 S is showing" in a.describe("Thu", "Fri")


def test_auditions_skip_fixed_ineligible_and_already_starting_players() -> None:
    bench = sim_opt("b", (1.0,), day=THU)
    locked = LineupOption("l", "L", ("PG",), 30.0, movable=False, next_game=FRI, sim=Slot("l", N45, (1.0,), locked=30.0))
    assert plan_auditions(no_changes(["G#1"], [locked], [bench]), [bench]) == []
    forward_slot = sim_opt("s", (1.0, 1.0), positions=("PF",), day=FRI)
    assert plan_auditions(no_changes(["F#1"], [forward_slot], [bench]), [bench]) == []  # a PG cannot play F
    starter = sim_opt("s", (1.0, 1.0), day=FRI)
    plan = no_changes(["G#1", "UTIL#1"], [starter, bench], [])  # b is starting already
    assert plan_auditions(plan, [bench]) == []


def test_each_bench_player_auditions_once_for_the_slot_where_it_gains_most() -> None:
    one = sim_opt("one", (1.0,), day=FRI)  # one game from Friday: 45 -- the cheaper slot to fill
    two = sim_opt("two", (1.0, 1.0), day=FRI)
    bench = sim_opt("b", (1.0,), day=THU)
    (a,) = plan_auditions(no_changes(["G#1", "G#2"], [two, one], [bench]), [bench])
    assert a.starter.player_id == "one" and a.gain == pytest.approx(e_max(45.0) - 45.0, abs=0.01)


def test_keep_the_bench_player_in_when_his_games_left_beat_the_starter() -> None:
    starter = sim_opt("s", (0.9,), day=FRI)  # 40.5
    bench = sim_opt("b", (1.0, 1.0), day=THU)  # tonight, then one more (45)
    (a,) = plan_auditions(no_changes(["G#1"], [starter], [bench]), [bench])
    assert a.keep_bench and a.lock_at == pytest.approx(45.0) and "otherwise keep him in" in a.describe("Thu", "Fri")


def test_lineup_options_carry_the_simulation_slot_and_bench_has_no_pending() -> None:
    mine = [starter("s", Slot("s", N45, (1.0,), pending=12.0))]
    mine[0].upcoming = [GameSlot(FRI, "BOS", True, False)]
    bench = [(starter("b", Slot("b", N45, (1.0,), pending=70.0)), ("PG",))]
    starters, options = lineup_options(mine, bench, {"s": Player(player_id="s", fantasy_positions=["PG"])})
    assert starters[0].sim.pending == 12.0 and options[0].sim.pending is None
