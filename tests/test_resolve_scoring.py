"""resolve_scoring: which flag combinations reproduce what Sleeper counted. No network.

Expected scores are hand-computed: 20 pts / 10 reb / 10 ast is 30 + 10 + 10 = 50 base, plus 7
(dd 3 + td 4) if td_includes_dd else 4. 50 pts alone is 75, plus 7 (40+ 3, 50+ 4) if
bonuses_stack else 4.
"""

from __future__ import annotations

from engine.jobs.resolve_scoring import PlayerWeek, tally
from engine.model.scoring import BoxScore, load_rules

TD = BoxScore(pts=20, reb=10, ast=10)
FIFTY = BoxScore(pts=50)
PLAIN = BoxScore(pts=10, reb=4)  # 19, the same under every combination


def week(counted: float, *boxes: BoxScore) -> PlayerWeek:
    return PlayerWeek("p", 1, counted, [(f"d{i}", box) for i, box in enumerate(boxes)])


def test_triple_double_week_decides_td_includes_dd() -> None:
    result = tally([week(57.0, TD, PLAIN)], load_rules())
    (label, matching), = result.evidence
    assert matching == {(False, True), (True, True)}
    assert tally([week(54.0, TD)], load_rules()).evidence[0][1] == {(False, False), (True, False)}


def test_fifty_point_week_decides_bonuses_stack() -> None:
    assert tally([week(82.0, FIFTY)], load_rules()).evidence[0][1] == {(True, False), (True, True)}
    assert tally([week(79.0, FIFTY)], load_rules()).evidence[0][1] == {(False, False), (False, True)}


def test_ordinary_weeks_and_zeros_are_undecided_not_evidence() -> None:
    result = tally([week(19.0, PLAIN), week(0.0, TD), week(0.0)], load_rules())
    assert result.undecided == 3 and not result.evidence and not result.unexplained


def test_a_score_no_combination_explains_is_reported() -> None:
    result = tally([week(23.0, PLAIN)], load_rules())
    assert len(result.unexplained) == 1 and "counted 23" in result.unexplained[0]


def test_consistency_counts() -> None:
    result = tally([week(57.0, TD), week(82.0, FIFTY), week(19.0, PLAIN)], load_rules())
    assert result.consistent((True, True)) == 2
    assert result.consistent((False, False)) == 0
    assert result.consistent((True, False)) == 1 and result.consistent((False, True)) == 1
