"""Golden tests for engine/model/scoring.py (TASKS M1).

The expected values in tests/fixtures/box_scores.json were computed by hand from SPEC §4 and
reviewed before this file existed. They are the reference; nothing here recomputes a total
from the weights, because a test that reimplements the formula only proves the formula equals
itself.

Every fixture runs through both settings of both ambiguity flags. No test touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.model.scoring import (
    STAT_FIELD_TO_SCORING_KEY,
    BoxScore,
    ScoringRules,
    default_rules,
    double_digit_categories,
    load_rules,
    score_box,
    score_breakdown,
    with_flags,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "league.json"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "box_scores.json"

FLAG_COMBINATIONS = [(False, False), (False, True), (True, False), (True, True)]


def _load_fixtures() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]


CASES = _load_fixtures()
CASE_IDS = [case["name"] for case in CASES]


@pytest.fixture(scope="module")
def league_rules() -> ScoringRules:
    """The real league rules, loaded from config/league.json by absolute path so the suite
    passes regardless of the working directory pytest was started from."""
    return load_rules(CONFIG_PATH)


def _flag_key(bonuses_stack: bool, td_includes_dd: bool) -> str:
    return f"{str(bonuses_stack).lower()}/{str(td_includes_dd).lower()}"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("bonuses_stack,td_includes_dd", FLAG_COMBINATIONS)
def test_total_matches_hand_scored_fixture(
    case: dict, bonuses_stack: bool, td_includes_dd: bool, league_rules: ScoringRules
) -> None:
    """Every fixture, under every combination of the two unresolved rules."""
    rules = with_flags(
        league_rules, bonuses_stack=bonuses_stack, td_includes_dd=td_includes_dd
    )
    expected = case["expected"][_flag_key(bonuses_stack, td_includes_dd)]

    actual = score_box(BoxScore(**case["box"]), rules)

    assert actual == pytest.approx(expected), (
        f"{case['name']} under bonuses_stack={bonuses_stack} "
        f"td_includes_dd={td_includes_dd}: {case['hand_arithmetic']}"
    )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_base_is_flag_independent(case: dict, league_rules: ScoringRules) -> None:
    """The dot-product part of the score never depends on an ambiguity flag. If this breaks,
    a bonus has leaked into the base."""
    for bonuses_stack, td_includes_dd in FLAG_COMBINATIONS:
        rules = with_flags(
            league_rules, bonuses_stack=bonuses_stack, td_includes_dd=td_includes_dd
        )
        breakdown = score_breakdown(BoxScore(**case["box"]), rules)
        assert breakdown.base == pytest.approx(case["base"]), case["hand_arithmetic"]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_double_digit_categories(case: dict) -> None:
    """Which categories reached double figures, and in the canonical order."""
    box = BoxScore(**case["box"])
    assert list(double_digit_categories(box)) == case["double_digit_categories"]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_breakdown_parts_sum_to_total(case: dict, league_rules: ScoringRules) -> None:
    """The itemisation shown in the UI has to add up to the number it explains."""
    breakdown = score_breakdown(BoxScore(**case["box"]), league_rules)
    parts = (
        breakdown.base
        + breakdown.double_double
        + breakdown.triple_double
        + breakdown.points_bonus
    )
    assert parts == pytest.approx(breakdown.total)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_score_box_agrees_with_breakdown(case: dict, league_rules: ScoringRules) -> None:
    box = BoxScore(**case["box"])
    assert score_box(box, league_rules) == pytest.approx(
        score_breakdown(box, league_rules).total
    )


def test_fixtures_actually_discriminate_both_flags() -> None:
    """Guards the suite against itself.

    A fixture set where every case scores the same under all four flag combinations would pass
    every test above while proving nothing about the ambiguity branches. Assert that each flag
    independently changes at least one fixture's total.
    """
    stack_matters = any(
        case["expected"]["false/false"] != case["expected"]["true/false"] for case in CASES
    )
    td_matters = any(
        case["expected"]["false/false"] != case["expected"]["false/true"] for case in CASES
    )
    assert stack_matters, "no fixture distinguishes bonuses_stack"
    assert td_matters, "no fixture distinguishes td_includes_dd"


def test_quadruple_double_does_not_crash(league_rules: ScoringRules) -> None:
    """SPEC §4: rare enough to ignore, but it must not blow up a live nightly job."""
    box = BoxScore(pts=20, reb=11, ast=10, stl=10, blk=10, to=6, fg3m=2)
    assert len(double_digit_categories(box)) == 5
    assert score_box(box, league_rules) > 0


# --- rules loading: fail loudly, never default silently (CLAUDE.md rule 4) ------------------


def _config_with(tmp_path: Path, mutate) -> Path:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(config)
    path = tmp_path / "league.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_default_rules_match_config(league_rules: ScoringRules) -> None:
    """default_rules() is the league's real configuration, not a built-in default."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    ambiguities = config["scoring_ambiguities"]

    assert default_rules() == league_rules
    assert league_rules.bonuses_stack is bool(ambiguities["bonuses_stack"]["value"])
    assert league_rules.td_includes_dd is bool(ambiguities["td_includes_dd"]["value"])

    # Weights are keyed by box-score field name; config uses Sleeper's spelling.
    for field, scoring_key in STAT_FIELD_TO_SCORING_KEY.items():
        assert league_rules.weights[field] == pytest.approx(config["scoring"][scoring_key])
    for key in ("dd", "td"):
        assert league_rules.weights[key] == pytest.approx(config["scoring"][key])


def test_scoring_keys_cover_the_live_payload_exactly() -> None:
    """config/league.json mirrors Sleeper's scoring_settings verbatim, so the set of keys this
    module knows how to apply must equal the set the config declares -- no key understood but
    absent, none present but unapplied. This is what caught tpm/tf/ff on the first real run."""
    scoring = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["scoring"]
    bonus_keys = {key for key in scoring if key.startswith("bonus_pt_")}
    applied = set(STAT_FIELD_TO_SCORING_KEY.values()) | {"dd", "td"} | bonus_keys
    assert applied == set(scoring)


def test_missing_weight_raises(tmp_path: Path) -> None:
    def drop_rebounds(config: dict) -> None:
        del config["scoring"]["reb"]

    with pytest.raises(ValueError, match="missing weights"):
        load_rules(_config_with(tmp_path, drop_rebounds))


def test_unknown_scoring_key_raises(tmp_path: Path) -> None:
    """A scoring key we do not apply is an error, not something to quietly ignore -- it would
    mean the league scores something this engine does not."""

    def add_unknown(config: dict) -> None:
        config["scoring"]["dunk"] = 2.0

    with pytest.raises(ValueError, match="does not know how to apply"):
        load_rules(_config_with(tmp_path, add_unknown))


def test_unknown_quadruple_double_policy_raises(tmp_path: Path) -> None:
    """Validated at load time rather than at score time, so a quadruple-double can never take
    down a live job."""

    def bad_policy(config: dict) -> None:
        config["scoring_ambiguities"]["quadruple_double"]["value"] = "pay_double_td"

    with pytest.raises(ValueError, match="unknown quadruple_double policy"):
        load_rules(_config_with(tmp_path, bad_policy))


def test_new_points_bonus_threshold_is_picked_up_from_config(tmp_path: Path) -> None:
    """Bonus thresholds are parsed from config key names, so the commissioner adding a 60-point
    bonus needs no code change."""

    def add_sixty(config: dict) -> None:
        config["scoring"]["bonus_pt_60p"] = 5.0
        config["scoring_ambiguities"]["bonuses_stack"]["value"] = False

    rules = load_rules(_config_with(tmp_path, add_sixty))
    assert (60, 5.0) in rules.points_bonuses

    sixty_point_game = BoxScore(pts=60)
    # Highest threshold cleared only: 1.5*60 = 90, plus the 60-point bonus.
    assert score_box(sixty_point_game, rules) == pytest.approx(90.0 + 5.0)
    assert score_box(sixty_point_game, with_flags(rules, bonuses_stack=True)) == pytest.approx(
        90.0 + 3.0 + 4.0 + 5.0
    )


def test_extra_box_score_field_is_rejected() -> None:
    """Box scores come from NBA game logs. A field we do not score is a mapping bug worth
    surfacing, not a column to drop on the floor."""
    with pytest.raises(Exception):
        BoxScore(pts=10, minutes=34)
