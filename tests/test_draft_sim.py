"""Tests for engine/model/draft_sim.py. Pure, no network."""

from __future__ import annotations

import numpy as np
import pytest

from engine.model.draft_sim import (
    advise,
    draft_clock,
    roster_gain,
    simulate_ranks,
    slot_for_pick,
    snake_pick_numbers,
    survival,
    survives_if_taken,
)

SLOTS = ["PG#1", "SG#1", "G#1", "SF#1", "PF#1", "F#1", "C#1", "C#2", "UTIL#1"]


# --- snake arithmetic ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pick", "slot"), [(1, 1), (8, 8), (9, 8), (16, 1), (17, 1), (24, 8), (97, 1), (104, 8)]
)
def test_slot_for_pick_snakes(pick: int, slot: int) -> None:
    assert slot_for_pick(pick, 8) == slot


def test_snake_pick_numbers_for_slot_five() -> None:
    picks = snake_pick_numbers(5, 8, 13)
    assert picks[:4] == [5, 12, 21, 28]
    assert len(picks) == 13


def test_clock_before_my_first_pick() -> None:
    clock = draft_clock(slot=5, teams=8, rounds=13, picks_made=0)
    assert (clock.on_clock, clock.my_turn, clock.my_next_pick) == (1, False, 5)
    assert clock.picks_until_next == 4


def test_clock_on_my_turn_counts_picks_to_the_following_one() -> None:
    clock = draft_clock(slot=5, teams=8, rounds=13, picks_made=4)
    assert clock.my_turn
    assert (clock.my_next_pick, clock.my_following_pick, clock.picks_between) == (5, 12, 6)


def test_clock_at_the_turn_has_back_to_back_picks() -> None:
    clock = draft_clock(slot=8, teams=8, rounds=13, picks_made=7)
    assert clock.my_turn
    assert (clock.my_next_pick, clock.my_following_pick, clock.picks_between) == (8, 9, 0)


def test_clock_on_my_last_pick_and_after_the_draft() -> None:
    last = draft_clock(slot=1, teams=8, rounds=2, picks_made=15)
    assert last.my_turn and last.my_following_pick is None and last.picks_between == 0
    done = draft_clock(slot=1, teams=8, rounds=2, picks_made=16)
    assert done.on_clock is None and not done.my_turn and done.my_next_pick is None


# --- survival -----------------------------------------------------------------------------


def test_zero_noise_takes_players_in_adp_order() -> None:
    ranks = simulate_ranks(np.arange(1.0, 11.0), 0.0, 5, np.random.default_rng(0))
    assert list(survival(ranks, 3)) == [0, 0, 0] + [1] * 7


def test_taking_a_player_moves_everyone_behind_him_up() -> None:
    ranks = simulate_ranks(np.arange(1.0, 11.0), 0.0, 5, np.random.default_rng(0))
    alive = survives_if_taken(ranks, taken=0, picks_before=3)[0]
    # The user takes the ADP-1 player; opponents then take ADP 2, 3 and 4.
    assert list(alive) == [False, False, False, False] + [True] * 6


def test_taking_a_player_nobody_wanted_changes_nothing_for_the_rest() -> None:
    ranks = simulate_ranks(np.arange(1.0, 11.0), 0.0, 5, np.random.default_rng(0))
    alive = survives_if_taken(ranks, taken=9, picks_before=3)[0]
    assert list(alive) == [False, False, False] + [True] * 6 + [False]


def test_noise_lets_a_due_player_slip_sometimes() -> None:
    adp = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    tight = survival(simulate_ranks(adp, 0.01, 4000, np.random.default_rng(1)), 4)
    loose = survival(simulate_ranks(adp, 0.5, 4000, np.random.default_rng(1)), 4)
    assert tight[3] == pytest.approx(0.0, abs=0.01)
    assert 0.1 < loose[3] < 0.9
    assert np.all(np.diff(loose) >= -0.03)  # later ADP never meaningfully less likely to last


def test_survival_rejects_non_positive_adp() -> None:
    with pytest.raises(ValueError):
        simulate_ranks(np.array([0.0, 1.0]), 0.2, 10, np.random.default_rng(0))


# --- roster value ---------------------------------------------------------------------------


def test_gain_on_an_empty_roster_is_board_vor() -> None:
    assert roster_gain([], [], SLOTS, 12.5, ("C",)) == pytest.approx(12.5)


def test_gain_is_zero_once_his_slots_hold_better_players() -> None:
    roster_vor = [20.0, 18.0, 15.0]
    roster_positions = [("C",), ("C",), ("C",)]  # both C slots and UTIL are full of centres
    assert roster_gain(roster_vor, roster_positions, SLOTS, 10.0, ("C",)) == 0.0


def test_gain_counts_only_the_upgrade_over_who_he_displaces() -> None:
    roster_vor = [20.0, 18.0, 6.0]
    roster_positions = [("C",), ("C",), ("C",)]
    assert roster_gain(roster_vor, roster_positions, SLOTS, 10.0, ("C",)) == pytest.approx(4.0)


def test_below_replacement_player_is_never_charged_for_starting() -> None:
    assert roster_gain([], [], SLOTS, -3.0, ("PG",)) == 0.0
    assert roster_gain([5.0], [("PG",)], SLOTS, 4.0, ("PG",)) == pytest.approx(4.0)


def test_gain_is_not_a_flat_cutoff() -> None:
    """The phantom-player trap: a PG must still be worth more to a PG-less roster than a
    third centre is to a roster with two centres and a filled UTIL."""
    roster_vor = [15.0, 14.0, 13.0]
    roster_positions = [("C",), ("C",), ("SF",)]
    pg = roster_gain(roster_vor, roster_positions, SLOTS, 8.0, ("PG",))
    third_c = roster_gain(roster_vor, roster_positions, SLOTS, 8.0, ("C",))
    assert pg == pytest.approx(8.0)
    assert third_c == pytest.approx(8.0)  # UTIL still open, so he starts there
    roster_vor.append(12.0)
    roster_positions.append(("PF",))
    fourth_big = roster_gain(roster_vor, roster_positions, SLOTS, 8.0, ("C",))
    assert fourth_big == pytest.approx(8.0)  # UTIL still open: PF went to PF, not UTIL


# --- the recommendation -----------------------------------------------------------------------


def fillers(count: int, start_adp: float) -> tuple[list[float], list[tuple[str, ...]], list[float]]:
    return [1.0] * count, [("SF",)] * count, [start_adp + i for i in range(count)]


def test_takes_the_player_who_will_be_gone_over_the_one_who_will_last() -> None:
    # A is worth a little more but nobody else wants him (ADP 300); B is about to go (ADP 1).
    vor = [10.0, 9.0]
    positions: list[tuple[str, ...]] = [("PG",), ("SG",)]
    adp = [300.0, 1.0]
    filler_vor, filler_pos, filler_adp = fillers(10, 2.0)
    clock = draft_clock(slot=5, teams=8, rounds=13, picks_made=4)  # my turn, 6 picks between
    advice = advise(
        np.array(vor + filler_vor), positions + filler_pos, np.array(adp + filler_adp),
        [], [], SLOTS, clock, sigma=0.05, n_sims=500, rng=np.random.default_rng(0),
    )
    assert advice[0].index == 1  # B now, A will still be there at pick 12
    a = next(item for item in advice if item.index == 0)
    assert a.p_survive > 0.95
    assert advice[0].score == pytest.approx(9.0 + 10.0, abs=0.1)


def test_takes_the_better_player_when_both_will_last() -> None:
    vor = [10.0, 9.0]
    positions: list[tuple[str, ...]] = [("PG",), ("SG",)]
    adp = [300.0, 301.0]
    filler_vor, filler_pos, filler_adp = fillers(10, 2.0)
    clock = draft_clock(slot=5, teams=8, rounds=13, picks_made=4)
    advice = advise(
        np.array(vor + filler_vor), positions + filler_pos, np.array(adp + filler_adp),
        [], [], SLOTS, clock, sigma=0.05, n_sims=500, rng=np.random.default_rng(0),
    )
    assert advice[0].index == 0


def test_off_turn_advice_is_gain_with_survival_to_my_next_pick() -> None:
    vor = [10.0, 9.0, 8.0]
    positions: list[tuple[str, ...]] = [("PG",), ("SG",), ("C",)]
    adp = [1.0, 2.0, 300.0]
    filler_vor, filler_pos, filler_adp = fillers(5, 3.0)
    clock = draft_clock(slot=5, teams=8, rounds=13, picks_made=0)  # 4 picks before mine
    advice = advise(
        np.array(vor + filler_vor), positions + filler_pos, np.array(adp + filler_adp),
        [], [], SLOTS, clock, sigma=0.05, n_sims=300, rng=np.random.default_rng(0),
    )
    assert [item.index for item in advice[:3]] == [0, 1, 2]
    assert [item.score for item in advice[:3]] == [10.0, 9.0, 8.0]
    assert advice[0].p_survive == 0.0 and advice[2].p_survive == 1.0


def test_last_pick_takes_the_best_gain() -> None:
    clock = draft_clock(slot=1, teams=8, rounds=2, picks_made=15)
    advice = advise(
        np.array([3.0, 7.0]), [("PG",), ("C",)], np.array([1.0, 50.0]), [], [], SLOTS, clock,
        n_sims=100, rng=np.random.default_rng(0),
    )
    assert advice[0].index == 1


def test_full_lineup_falls_back_to_best_player_available() -> None:
    roster_vor = [20.0] * 9
    roster_positions = [("PG",), ("SG",), ("PG",), ("SF",), ("PF",), ("SF",), ("C",), ("C",), ("C",)]
    clock = draft_clock(slot=1, teams=8, rounds=13, picks_made=96)
    advice = advise(
        np.array([2.0, 5.0, 3.0]), [("PG",), ("SF",), ("C",)], np.array([90.0, 95.0, 99.0]),
        roster_vor, roster_positions, SLOTS, clock, n_sims=100, rng=np.random.default_rng(0),
    )
    assert all(item.gain == 0.0 for item in advice)
    assert advice[0].index == 1
