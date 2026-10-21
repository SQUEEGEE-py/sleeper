"""verify_league's settings check: IR, waivers, playoffs, trade deadline. No network.

LIVE_SETTINGS is the relevant part of the real league's settings as fetched 2026-10-21.
"""

import copy
import json
from pathlib import Path

from engine.jobs.verify_league import diff_settings

CONFIG = json.loads(Path("config/league.json").read_text(encoding="utf-8"))

LIVE_SETTINGS = {
    "num_teams": 8,
    "reserve_slots": 2,
    "reserve_allow_out": 1,
    "reserve_allow_sus": 1,
    "reserve_allow_dnr": 1,
    "reserve_allow_dtd": 0,
    "reserve_allow_doubtful": 0,
    "reserve_allow_na": 0,
    "reserve_allow_cov": 0,
    "waiver_type": 1,
    "waiver_budget": 100,
    "waiver_clear_days": 1,
    "waiver_after_game_start": 1,
    "playoff_teams": 4,
    "playoff_week_start": 22,
    "trade_deadline": 19,
    "divisions": 2,
}


def live(**changes: object) -> dict:
    settings = dict(LIVE_SETTINGS)
    settings.update(changes)
    return settings


def test_config_matches_the_live_league():
    assert diff_settings(CONFIG, LIVE_SETTINGS) == []


def test_the_no_ir_mistake_is_caught():
    # 2026-09-23 recorded 0 IR slots from roster_positions; reserve_slots says 2.
    config = copy.deepcopy(CONFIG)
    config["ir"]["slots"] = 0
    assert diff_settings(config, LIVE_SETTINGS) == ["  ir.slots: config=0 live reserve_slots=2"]


def test_a_changed_waiver_type_is_caught():
    lines = diff_settings(CONFIG, live(waiver_type=2))
    assert lines == ["  waivers.sleeper_waiver_type: config=1 live waiver_type=2"]


def test_waiver_budget_is_not_evidence_of_faab():
    # It reads 100 in a league that does not bid; changing it alone must not matter.
    assert diff_settings(CONFIG, live(waiver_budget=200)) == []


def test_config_boolean_matches_sleeper_integer():
    assert CONFIG["waivers"]["after_game_start"] is True
    assert diff_settings(CONFIG, live(waiver_after_game_start=0)) == [
        "  waivers.after_game_start: config=1 live waiver_after_game_start=0"
    ]


def test_ir_allowing_day_to_day_is_caught():
    lines = diff_settings(CONFIG, live(reserve_allow_dtd=1))
    assert lines == [
        "  ir.eligible_statuses: config=['DNR', 'OUT', 'SUS'] live=['DNR', 'DTD', 'OUT', 'SUS']"
    ]


def test_a_status_sleeper_adds_later_is_caught():
    lines = diff_settings(CONFIG, live(reserve_allow_inj=1))
    assert len(lines) == 1 and "INJ" in lines[0]


def test_a_missing_settings_key_is_reported_not_skipped():
    settings = dict(LIVE_SETTINGS)
    del settings["trade_deadline"]
    lines = diff_settings(CONFIG, settings)
    assert len(lines) == 1 and "trade_deadline" in lines[0] and "no " in lines[0]
