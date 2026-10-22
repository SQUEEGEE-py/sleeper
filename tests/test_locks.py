"""Lock-status inference from Sleeper's counted score, and Sleeper box lines. No network."""

from __future__ import annotations

from datetime import date

from engine.model.locks import infer_lock
from engine.model.scoring import box_from_sleeper_stats, score_box

TUE, THU, SAT = date(2026, 10, 20), date(2026, 10, 22), date(2026, 10, 24)


def test_no_game_yet() -> None:
    assert infer_lock([], 0.0).state == "no_game_yet"
    assert infer_lock([], 12.0, game_in_progress=True).state == "no_game_yet"
    assert infer_lock([], 12.0).state == "unexplained"


def test_counting_the_latest_game_is_open_not_locked() -> None:
    status = infer_lock([(TUE, 43.5)], 43.5)
    assert status.state == "open" and status.locked_on is None
    assert infer_lock([(TUE, 43.5), (THU, 30.0)], 30.0).state == "open"


def test_counting_an_earlier_game_means_locked_on_it() -> None:
    status = infer_lock([(TUE, 43.5), (THU, 30.0), (SAT, 51.0)], 43.5)
    assert (status.state, status.locked_on, status.locked_score) == ("locked", TUE, 43.5)


def test_a_dnp_counts_as_zero() -> None:
    assert infer_lock([(TUE, 43.5), (THU, 0.0)], 0.0).state == "open"
    assert infer_lock([(TUE, 43.5), (THU, 0.0)], 43.5).state == "locked"


def test_a_score_matching_no_game_is_a_gap_not_a_guess() -> None:
    status = infer_lock([(TUE, 43.5), (THU, 30.0)], 46.5)  # e.g. a scoring flag disagreement
    assert status.state == "unexplained" and "46.5" in status.note and "43.5" in status.note


def test_game_in_progress() -> None:
    assert infer_lock([(TUE, 43.5)], 43.5, game_in_progress=True).state == "locked"
    assert infer_lock([(TUE, 43.5)], 12.0, game_in_progress=True).state == "open"


def test_equal_scores_take_the_latest_earlier_match() -> None:
    status = infer_lock([(TUE, 30.0), (THU, 30.0), (SAT, 20.0)], 30.0)
    assert status.state == "locked" and status.locked_on == THU


def test_sleeper_box_lines_rescore_to_sleepers_own_totals() -> None:
    # Opening night 2026-10-20, Sleeper's per-game stats; Sleeper counted 43.5 and 38.0.
    brown = {"pts": 19.0, "reb": 8.0, "ast": 6.0, "stl": 1.0, "blk": 1.0, "to": 2.0, "tpm": 2.0, "min": 34.9}
    lebron = {"pts": 14.0, "reb": 9.0, "ast": 7.0, "stl": 1.0, "blk": 1.0, "to": 1.0, "min": 33.75}
    assert score_box(box_from_sleeper_stats(brown)) == 43.5
    assert score_box(box_from_sleeper_stats(lebron)) == 38.0


def test_sleeper_technicals_and_flagrants_are_scored() -> None:
    box = box_from_sleeper_stats({"pts": 10.0, "tf": 1.0, "ff": 1.0, "tpm": None})
    assert (box.tech_foul, box.flagrant_foul, box.fg3m) == (1.0, 1.0, 0.0)
    assert score_box(box) == 15.0 - 1.0 - 2.0
