"""Tests for engine/model/lineup.py (TASKS M4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.model.lineup import (
    assign_lineup,
    is_eligible,
    lineup_value,
    marginal_value_to_roster,
    slot_position,
    starting_slots,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "league.json"


@pytest.fixture(scope="module")
def league_slots() -> list[str]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return starting_slots(config["roster_positions"])


def test_starting_slots_from_the_real_config(league_slots: list[str]) -> None:
    """Nine starters, two of them centres, bench and IR excluded."""
    assert len(league_slots) == 9
    assert league_slots.count("C#1") == 1 and league_slots.count("C#2") == 1
    assert not any(slot.startswith("BN") or slot.startswith("IR") for slot in league_slots)


def test_duplicate_slots_are_labelled_separately() -> None:
    assert starting_slots(["C", "C", "BN"]) == ["C#1", "C#2"]


def test_unknown_roster_position_is_rejected() -> None:
    with pytest.raises(ValueError, match="no eligibility rule"):
        starting_slots(["PG", "WIZARD"])


def test_slot_position_strips_the_label() -> None:
    assert slot_position("C#2") == "C"


@pytest.mark.parametrize(
    "positions,slot,expected",
    [
        (["PG"], "PG#1", True),
        (["PG"], "SG#1", False),
        (["PG"], "G#1", True),
        (["SG"], "G#1", True),
        (["SF"], "F#1", True),
        (["PF"], "F#1", True),
        (["C"], "F#1", False),
        (["C"], "C#1", True),
        (["PG"], "UTIL#1", True),
        (["C"], "UTIL#1", True),
        ([], "UTIL#1", True),
        ([], "PG#1", False),
        (None, "C#1", False),
    ],
)
def test_eligibility_rules(positions, slot, expected) -> None:
    assert is_eligible(positions, slot) is expected


def test_assignment_beats_greedy() -> None:
    """The case greedy gets wrong.

    Player A (30) is a pure centre. Player B (28) can play centre or forward. Greedy takes the
    highest value first and puts B in the only C slot, stranding A on the bench for a total of
    28 + nothing. The optimum is A at C and B at F, worth 58.
    """
    values = [30.0, 28.0]
    positions = [["C"], ["C", "PF"]]
    result = assign_lineup(values, positions, ["C#1", "F#1"])

    assert result.total_value == pytest.approx(58.0)
    assert result.assignments["C#1"] == 0
    assert result.assignments["F#1"] == 1
    assert result.is_complete


def test_unfillable_slot_is_reported_not_faked() -> None:
    """Two centre slots and one centre. The second must come back unfilled rather than being
    filled by an ineligible guard -- an incomplete lineup is a real state in this league."""
    result = assign_lineup([40.0, 35.0], [["C"], ["PG"]], ["C#1", "C#2"])

    assert result.assignments == {"C#1": 0}
    assert result.unfilled_slots == ("C#2",)
    assert not result.is_complete
    assert result.total_value == pytest.approx(40.0)


def test_best_players_start(league_slots: list[str]) -> None:
    """Thirteen players, nine slots: the bench holds the worst, subject to eligibility."""
    values = [float(50 - index) for index in range(13)]
    positions = [["PG", "SG", "SF", "PF", "C"]] * 13
    result = assign_lineup(values, positions, league_slots)

    assert len(result.assignments) == 9
    assert sorted(result.assignments.values()) == list(range(9))


def test_empty_inputs() -> None:
    assert assign_lineup([], [], ["PG#1"]).unfilled_slots == ("PG#1",)
    assert assign_lineup([10.0], [["PG"]], []).total_value == 0.0


def test_mismatched_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        assign_lineup([1.0, 2.0], [["PG"]], ["PG#1"])


def test_lineup_value_matches_assignment(league_slots: list[str]) -> None:
    values = [float(index) for index in range(12)]
    positions = [["PG", "SG", "SF", "PF", "C"]] * 12
    assert lineup_value(values, positions, league_slots) == pytest.approx(
        assign_lineup(values, positions, league_slots).total_value
    )


def test_marginal_value_diminishes_at_a_filled_position() -> None:
    """A fourth good centre adds almost nothing once both C slots and UTIL are taken. This is
    the quantity the draft simulator's `denial` is built from."""
    slots = ["C#1", "C#2", "UTIL#1"]
    values = [50.0, 48.0, 46.0]
    positions = [["C"], ["C"], ["C"]]

    added = marginal_value_to_roster(values, positions, slots, 47.0, ["C"])
    assert added == pytest.approx(1.0)  # 47 replaces the 46 at UTIL

    empty_roster = marginal_value_to_roster([], [], slots, 47.0, ["C"])
    assert empty_roster == pytest.approx(47.0)


def test_marginal_value_is_never_negative() -> None:
    """Adding a player can only help: worst case he sits on the bench."""
    slots = ["C#1", "UTIL#1"]
    added = marginal_value_to_roster([50.0, 48.0], [["C"], ["C"]], slots, 1.0, ["C"])
    assert added == pytest.approx(0.0)
