"""Tests for engine/ingest/player_ids.py (TASKS M4, SPEC §6.3).

The rule under test throughout: an unmapped player is a visible gap, never a dropped row
(CLAUDE.md rule 4). A silently missing player is a star who never appears on the board.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from engine.ingest.player_ids import (
    PlayerOverride,
    load_overrides,
    map_players,
    normalize_name,
    overrides_by_nba_id,
)


@dataclass
class FakePlayer:
    """Stands in for engine.ingest.sleeper.Player."""

    full_name: str
    team: str | None = None
    fantasy_positions: list[str] | None = None

    @property
    def model_extra(self) -> dict:
        return {"search_full_name": self.full_name}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Nikola Jokić", "nikolajokic"),
        ("Luka Dončić", "lukadoncic"),
        ("Jaren Jackson Jr.", "jarenjackson"),
        ("Gary Payton II", "garypayton"),
        ("Marvin Bagley III", "marvinbagley"),
        ("D'Angelo Russell", "dangelorussell"),
        ("Karl-Anthony Towns", "karlanthonytowns"),
        ("  Trae   Young ", "traeyoung"),
    ],
)
def test_normalize_name(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


def test_suffix_variants_collide_on_purpose() -> None:
    """Sleeper and NBA.com disagree about suffixes constantly; once first and last name match
    the suffix carries no identifying information."""
    assert normalize_name("Jaren Jackson Jr.") == normalize_name("Jaren Jackson")
    assert normalize_name("Gary Payton II") == normalize_name("Gary Payton")


def test_maps_across_accent_and_suffix_differences() -> None:
    sleeper = {
        "100": FakePlayer("Nikola Jokic"),
        "200": FakePlayer("Jaren Jackson"),
    }
    result = map_players(
        [(1, "Nikola Jokić", "DEN"), (2, "Jaren Jackson Jr.", "MEM")], sleeper
    )

    assert result.nba_to_sleeper == {1: "100", 2: "200"}
    assert result.gap_count == 0
    assert "no gaps" in result.describe()


def test_unmapped_players_are_reported_not_dropped() -> None:
    result = map_players([(1, "Nobody At All", "DEN")], {"100": FakePlayer("Someone Else")})

    assert result.nba_to_sleeper == {}
    assert result.unmapped == [(1, "Nobody At All", "DEN")]
    assert result.gap_count == 1
    assert "unmapped" in result.describe()


def test_same_name_is_broken_by_team() -> None:
    sleeper = {
        "100": FakePlayer("Jalen Williams", team="OKC"),
        "200": FakePlayer("Jalen Williams", team="WAS"),
    }
    result = map_players([(1, "Jalen Williams", "OKC")], sleeper)

    assert result.nba_to_sleeper == {1: "100"}
    assert not result.ambiguous


def test_unresolvable_ambiguity_is_surfaced() -> None:
    """Two players, same name, neither on his team: a human resolves this, not a guess."""
    sleeper = {
        "100": FakePlayer("Jalen Williams", team="OKC"),
        "200": FakePlayer("Jalen Williams", team="WAS"),
    }
    result = map_players([(1, "Jalen Williams", "MEM")], sleeper)

    assert result.nba_to_sleeper == {}
    assert result.ambiguous[0][0] == 1
    assert sorted(result.ambiguous[0][2]) == ["100", "200"]
    assert "ambiguous" in result.describe()


def test_override_forces_a_mapping_by_nba_id() -> None:
    sleeper = {"100": FakePlayer("Someone Else")}
    override = PlayerOverride(name="Whoever", nba_player_id=1, sleeper_id="100")
    result = map_players([(1, "Unmatchable Name", "DEN")], sleeper, [override])

    assert result.nba_to_sleeper == {1: "100"}
    assert result.gap_count == 0


def test_override_forces_a_mapping_by_name() -> None:
    sleeper = {"100": FakePlayer("Someone Else")}
    override = PlayerOverride(name="Unmatchable Name", sleeper_id="100")
    result = map_players([(1, "Unmatchable Name", "DEN")], sleeper, [override])
    assert result.nba_to_sleeper == {1: "100"}


def test_override_beats_a_wrong_name_match() -> None:
    """Overrides win outright -- that is the point of having them."""
    sleeper = {"100": FakePlayer("Jalen Williams"), "200": FakePlayer("Someone Else")}
    override = PlayerOverride(name="Jalen Williams", nba_player_id=1, sleeper_id="200")
    result = map_players([(1, "Jalen Williams", "OKC")], sleeper, [override])
    assert result.nba_to_sleeper == {1: "200"}


def test_players_without_a_name_are_skipped_safely() -> None:
    sleeper = {"100": FakePlayer(""), "200": FakePlayer("Real Player")}
    result = map_players([(1, "Real Player", "DEN")], sleeper)
    assert result.nba_to_sleeper == {1: "200"}


# --- the overrides file ---------------------------------------------------------------------


def test_load_overrides_skips_comments(tmp_path) -> None:
    path = tmp_path / "overrides.csv"
    path.write_text(
        "# a comment line before the header\n"
        "#\n"
        "name,nba_player_id,sleeper_id,value_multiplier,p_play,note\n"
        "LeBron James,,,1.10,,moved to PHI\n",
        encoding="utf-8",
    )
    overrides = load_overrides(path)

    assert len(overrides) == 1
    assert overrides[0].name == "LeBron James"
    assert overrides[0].value_multiplier == pytest.approx(1.10)
    assert overrides[0].p_play is None
    assert overrides[0].note == "moved to PHI"


def test_load_overrides_defaults_blank_columns(tmp_path) -> None:
    path = tmp_path / "overrides.csv"
    path.write_text(
        "name,nba_player_id,sleeper_id,value_multiplier,p_play,note\nSomeone,,,,,\n",
        encoding="utf-8",
    )
    override = load_overrides(path)[0]
    assert override.value_multiplier == 1.0
    assert override.sleeper_id is None
    assert override.nba_player_id is None


def test_load_overrides_on_a_missing_file() -> None:
    assert load_overrides("does/not/exist.csv") == []


def test_the_real_overrides_file_parses() -> None:
    """config/player_overrides.csv ships heavily commented; it must still load."""
    assert isinstance(load_overrides(), list)


def test_overrides_by_nba_id_resolves_names() -> None:
    overrides = [
        PlayerOverride(name="Nikola Jokić", value_multiplier=1.2),
        PlayerOverride(name="Ghost Player", value_multiplier=2.0),
    ]
    resolved = overrides_by_nba_id(overrides, {normalize_name("Nikola Jokic"): 42})

    assert set(resolved) == {42}
    assert resolved[42].value_multiplier == pytest.approx(1.2)
