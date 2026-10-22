"""Matchup simulation: win probability and the matchup-aware lock-or-roll call. No network."""

from __future__ import annotations

import numpy as np
import pytest

from engine.model.distributions import NormalScoreDistribution
from engine.model.lockin import weekly_value
from engine.model.matchup import Slot, decide, slot_outcomes, team_outcomes, win_probability

N45 = NormalScoreDistribution(45.0, 10.0)


def locked(name: str, score: float) -> Slot:
    return Slot(name, None, locked=score)


def rng() -> np.random.Generator:
    return np.random.default_rng(7)


# --- one slot -----------------------------------------------------------------------------


def test_threshold_policy_reproduces_the_closed_form_week() -> None:
    # The M3 regression, by simulation: normal(45, 10) over 3 games is worth 51.297.
    outcomes = slot_outcomes(Slot("p", N45, (1.0, 1.0, 1.0)), rng(), 200_000)
    assert outcomes.mean() == pytest.approx(51.297, abs=0.15)


def test_per_game_p_play_matches_lockin() -> None:
    p_plays = (0.9, 0.9, 0.67)  # a B2B final night
    outcomes = slot_outcomes(Slot("p", N45, p_plays), rng(), 200_000)
    assert outcomes.mean() == pytest.approx(weekly_value(N45, 3, p_plays), abs=0.2)


def test_final_game_is_forced_and_zero_if_he_sits() -> None:
    sits_half = slot_outcomes(Slot("p", N45, (0.5,)), rng(), 200_000)
    assert sits_half.mean() == pytest.approx(22.5, abs=0.15)
    assert (sits_half == 0).mean() == pytest.approx(0.5, abs=0.01)
    # A negative final game counts as negative: no silent max(X, 0).
    negative = slot_outcomes(Slot("p", NormalScoreDistribution(-5.0, 1.0), (1.0,)), rng(), 50_000)
    assert negative.mean() == pytest.approx(-5.0, abs=0.05)


def test_pending_score_is_locked_when_it_beats_the_threshold() -> None:
    # One game left: the threshold is E[X] = 45.
    assert set(slot_outcomes(Slot("p", N45, (1.0,), pending=46.0), rng(), 1000)) == {46.0}
    rolled = slot_outcomes(Slot("p", N45, (1.0,), pending=44.0), rng(), 1000)
    assert len(set(rolled)) > 1


def test_forcing_and_its_errors() -> None:
    slot = Slot("p", N45, (1.0,), pending=60.0)
    assert len(set(slot_outcomes(slot, rng(), 1000, force="roll"))) > 1
    assert set(slot_outcomes(Slot("p", N45, (1.0,), pending=10.0), rng(), 10, force="lock")) == {10.0}
    with pytest.raises(ValueError, match="no games left"):
        slot_outcomes(Slot("p", N45, (), pending=10.0), rng(), 10, force="roll")
    with pytest.raises(ValueError, match="no open decision"):
        slot_outcomes(Slot("p", N45, (1.0,)), rng(), 10, force="lock")


def test_slot_validation() -> None:
    with pytest.raises(ValueError, match="not both"):
        Slot("p", N45, (1.0,), locked=40.0, pending=40.0)
    with pytest.raises(ValueError, match="no distribution"):
        Slot("p", None, (1.0,))
    assert set(slot_outcomes(Slot("p", None), rng(), 5)) == {0.0}  # no games this week


# --- teams and odds ---------------------------------------------------------------------------


def test_fully_decided_weeks() -> None:
    assert win_probability([locked("a", 50)], [locked("b", 40)]).p_win == 1.0
    assert win_probability([locked("a", 40)], [locked("b", 50)]).p_win == 0.0
    tie = win_probability([locked("a", 45)], [locked("b", 45)])
    assert tie.p_win == 0.5 and tie.p_tie == 1.0


def test_identical_teams_are_a_coin_flip() -> None:
    team = [Slot(f"p{i}", N45, (0.9, 0.9, 0.9)) for i in range(9)]
    assert win_probability(team, team, n_sims=40_000).p_win == pytest.approx(0.5, abs=0.02)


def test_forcing_one_slot_leaves_every_other_slot_identical() -> None:
    # Common random numbers: the lock-vs-roll difference must be that slot's alone.
    mine = [Slot("open", N45, (1.0, 1.0), pending=47.0), Slot("other", N45, (0.9, 0.9, 0.9))]
    as_lock = team_outcomes(mine, 5000, seed=3, team=0, force={0: "lock"})
    as_roll = team_outcomes(mine, 5000, seed=3, team=0, force={0: "roll"})
    streams = np.random.SeedSequence([3, 0]).spawn(2)
    open_roll = slot_outcomes(mine[0], np.random.default_rng(streams[0]), 5000, force="roll")
    other = slot_outcomes(mine[1], np.random.default_rng(streams[1]), 5000)
    assert np.allclose(as_lock - 47.0, other)
    assert np.allclose(as_roll - open_roll, other)


# --- the matchup-aware call --------------------------------------------------------------------


def test_behind_rolls_a_score_expected_value_would_lock() -> None:
    # 50 beats the 45 threshold, but locking it loses for sure (needs 61+ from this slot).
    mine = [Slot("open", N45, (1.0,), pending=50.0), locked("rest", 100.0)]
    decision = decide(mine, [locked("them", 160.0)], 0, n_sims=20_000)
    assert decision.ev_call == "lock" and decision.call == "roll" and decision.matchup_changed_it
    # Rolling wins iff the last game beats 60: P(N(45, 10) > 60) = 1 - Phi(1.5) = 0.0668.
    assert decision.p_win_lock == 0.0 and decision.p_win_roll == pytest.approx(0.0668, abs=0.006)


def test_ahead_locks_a_score_expected_value_would_roll() -> None:
    # 40 is under the 45 threshold, but locking it wins for sure; rolling risks a low game.
    mine = [Slot("open", N45, (1.0,), pending=40.0), locked("rest", 100.0)]
    decision = decide(mine, [locked("them", 135.0)], 0, n_sims=20_000)
    assert decision.ev_call == "roll" and decision.call == "lock"
    assert decision.p_win_lock == 1.0 and decision.p_win_roll < 0.95


def test_a_call_that_cannot_change_the_result_keeps_the_ev_call() -> None:
    mine = [Slot("open", N45, (1.0,), pending=40.0), locked("rest", 500.0)]
    decision = decide(mine, [locked("them", 10.0)], 0, n_sims=2000)
    assert decision.p_win_lock == decision.p_win_roll == 1.0
    assert decision.call == decision.ev_call == "roll" and not decision.matchup_changed_it


def test_decide_needs_an_open_decision() -> None:
    with pytest.raises(ValueError, match="no open decision"):
        decide([locked("a", 40.0)], [locked("b", 40.0)], 0)
