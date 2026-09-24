"""Tests for engine/model/valuation.py (TASKS M4).

The property that matters most is the one SPEC §5.3 is emphatic about: replacement level must
be slot-specific, and a flat cutoff undervalues centres. There is an explicit regression test
for the way that can silently break.
"""

from __future__ import annotations

import pytest

from engine.model.valuation import (
    PlayerValue,
    build_board,
    compute_replacement_levels,
    replacement_for,
    value_over_replacement,
)

SLOTS = ["PG#1", "SG#1", "G#1", "SF#1", "PF#1", "F#1", "C#1", "C#2", "UTIL#1"]


def player(index: int, value: float, positions: tuple[str, ...]) -> PlayerValue:
    return PlayerValue(
        player_id=index,
        name=f"P{index}",
        team="DEN",
        positions=positions,
        season_value=value,
    )


def make_pool(guards: int, forwards: int, centers: int, top: float = 100.0) -> list[PlayerValue]:
    """A pool where guards and forwards are deep and centres are thin, with values descending
    so the ordering is unambiguous."""
    pool: list[PlayerValue] = []
    index = 0
    for count, positions in ((guards, ("PG", "SG")), (forwards, ("SF", "PF")), (centers, ("C",))):
        for offset in range(count):
            pool.append(player(index, top - index, positions))
            index += 1
    return pool


def test_replacement_is_computed_per_slot() -> None:
    levels = compute_replacement_levels(make_pool(40, 40, 30), SLOTS, teams=8)
    assert set(levels.by_slot) == {"PG", "SG", "G", "SF", "PF", "F", "C", "UTIL"}
    assert levels.is_complete
    assert levels.starter_slots == 9 * 8


def test_scarce_positions_get_a_lower_bar() -> None:
    """Sixteen centre jobs but few centres: the marginal starting centre is weaker than the
    marginal starting guard, so centres are measured against a lower bar."""
    # 60 guards and 60 forwards but only 20 centres, all interleaved by value.
    pool = []
    for index in range(140):
        if index % 7 == 0:
            positions = ("C",)
        elif index % 2 == 0:
            positions = ("PG", "SG")
        else:
            positions = ("SF", "PF")
        pool.append(player(index, 100.0 - index * 0.5, positions))

    levels = compute_replacement_levels(pool, SLOTS, teams=8)
    assert levels.by_slot["C"] < levels.by_slot["G"]


def test_vor_is_not_a_flat_cutoff() -> None:
    """The regression that matters.

    Everyone is UTIL-eligible and the greedy fills UTIL last, so the marginal UTIL starter is
    structurally the weakest starter in the league. If UTIL is included when picking a
    player's bar, every player gets the same number and value over replacement collapses to
    "season value minus a constant" -- exactly the flat player-number-72 cutoff SPEC §5.3
    forbids. This asserts the board still distinguishes positions.
    """
    pool = []
    for index in range(140):
        positions = ("C",) if index % 7 == 0 else ("PG", "SG")
        pool.append(player(index, 100.0 - index * 0.5, positions))

    rows, levels = build_board(pool, SLOTS, teams=8)
    replacements = {round(replacement, 6) for _, _, replacement in rows}
    assert len(replacements) > 1, "every player got the same replacement level"


def test_a_center_beats_an_equal_guard() -> None:
    """Same season value, different scarcity: the centre is worth more at the draft."""
    pool = []
    for index in range(140):
        positions = ("C",) if index % 7 == 0 else ("PG", "SG")
        pool.append(player(index, 100.0 - index * 0.5, positions))

    levels = compute_replacement_levels(pool, SLOTS, teams=8)
    center = player(999, 70.0, ("C",))
    guard = player(998, 70.0, ("PG",))

    assert value_over_replacement(center, levels) > value_over_replacement(guard, levels)


def test_util_is_excluded_from_a_players_bar() -> None:
    pool = make_pool(40, 40, 30)
    levels = compute_replacement_levels(pool, SLOTS, teams=8)
    assert replacement_for(("C",), levels) == pytest.approx(levels.by_slot["C"])
    assert replacement_for(("C",), levels) != pytest.approx(levels.by_slot["UTIL"]) or (
        levels.by_slot["C"] == levels.by_slot["UTIL"]
    )


def test_multi_position_players_take_the_lowest_positional_bar() -> None:
    levels = compute_replacement_levels(make_pool(40, 40, 30), SLOTS, teams=8)
    both = replacement_for(("C", "PF"), levels)
    assert both == pytest.approx(min(levels.by_slot["C"], levels.by_slot["PF"], levels.by_slot["F"]))


def test_player_eligible_for_nothing_falls_back_to_util() -> None:
    levels = compute_replacement_levels(make_pool(40, 40, 30), SLOTS, teams=8)
    assert replacement_for((), levels) == pytest.approx(levels.by_slot["UTIL"])


def test_board_is_sorted_by_vor() -> None:
    rows, _ = build_board(make_pool(40, 40, 30), SLOTS, teams=8)
    vors = [vor for _, vor, _ in rows]
    assert vors == sorted(vors, reverse=True)


def test_incomplete_pool_is_flagged_not_hidden() -> None:
    """Fewer players than starting slots: replacement levels are approximate and the caller
    has to be told, not handed a confident-looking number."""
    levels = compute_replacement_levels(make_pool(3, 3, 2), SLOTS, teams=8)
    assert not levels.is_complete
    assert levels.starters_filled < levels.starter_slots


def test_centers_fill_center_slots_before_util() -> None:
    """If UTIL could absorb scarce positions the whole measurement flattens."""
    levels = compute_replacement_levels(make_pool(40, 40, 30), SLOTS, teams=8)
    center_ids = set(levels.drafted["C"])
    assert len(center_ids) == 16  # 2 per team x 8 teams


def test_teams_must_be_positive() -> None:
    with pytest.raises(ValueError, match="teams"):
        compute_replacement_levels(make_pool(5, 5, 5), SLOTS, teams=0)
