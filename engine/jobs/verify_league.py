"""Fetch the live league from Sleeper and diff it against config/league.json.

config/league.json is the only source of truth for league rules (CLAUDE.md, rule 1). This job
is the loud-failure check: it never silently accepts a mismatch between what we've hardcoded
in config and what Sleeper actually reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from engine.ingest.sleeper import SleeperClient

CONFIG_PATH = Path("config/league.json")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def diff_scoring(config_scoring: dict, live_scoring: dict) -> list[str]:
    """Compare config/league.json's scoring keys against Sleeper's live scoring_settings.

    This assumes config's keys (pts, reb, ast, ...) already match Sleeper's own field names,
    since config/league.json was transcribed from the commissioner's real Sleeper settings.
    We have not independently verified Sleeper's NBA scoring_settings key names against a real
    NBA league (only confirmed the endpoint's shape via an NFL example league), so a config key
    missing entirely from the live payload is reported as its own case rather than silently
    treated as "0 vs expected" -- it likely means the real Sleeper key name differs and needs
    a human to confirm it, not a guessed translation table.
    """
    lines = []
    for config_key, expected in config_scoring.items():
        if config_key not in live_scoring:
            lines.append(
                f"  scoring[{config_key}]: key not found in live scoring_settings at all "
                f"(expected {expected!r}) -- Sleeper's real field name may differ; confirm "
                f"against the live payload rather than assuming"
            )
            continue
        actual = live_scoring[config_key]
        if expected != actual:
            lines.append(f"  scoring[{config_key}]: config={expected!r} live={actual!r}")
    return lines


def diff_roster_positions(config_positions: list[str], live_positions: list[str]) -> list[str]:
    if config_positions == live_positions:
        return []
    return [
        f"  roster_positions: config={config_positions!r} live={live_positions!r}"
    ]


# Readable config field -> Sleeper's settings key. Values are compared after int(), so a config
# true matches Sleeper's 1. waiver_budget is deliberately absent: it reads 100 whether or not
# the league bids, which is how FAAB was wrongly recorded in 2026-09.
SETTINGS_KEYS: dict[tuple[str, ...], str] = {
    ("teams",): "num_teams",
    ("ir", "slots"): "reserve_slots",
    ("waivers", "sleeper_waiver_type"): "waiver_type",
    ("waivers", "clear_days"): "waiver_clear_days",
    ("waivers", "after_game_start"): "waiver_after_game_start",
    ("playoffs", "teams"): "playoff_teams",
    ("playoffs", "start_week"): "playoff_week_start",
    ("trade_deadline_week",): "trade_deadline",
    ("divisions",): "divisions",
}

IR_STATUS_PREFIX = "reserve_allow_"


def _lookup(config: dict, path: tuple[str, ...]) -> object:
    value: object = config
    for key in path:
        value = value[key]  # type: ignore[index]
    return value


def diff_settings(config: dict, live_settings: dict) -> list[str]:
    """Compare the league rules that live in Sleeper's ``settings`` rather than in scoring or
    roster_positions: IR, waivers, playoffs, trade deadline, divisions.

    IR eligibility is compared as a set of statuses: every ``reserve_allow_<status>`` set to 1
    must be in ``config["ir"]["eligible_statuses"]`` and nothing else may be, so a status Sleeper
    adds later shows up here instead of being ignored.
    """
    lines = []
    for path, sleeper_key in SETTINGS_KEYS.items():
        name = ".".join(path)
        if sleeper_key not in live_settings:
            lines.append(f"  {name}: Sleeper settings has no {sleeper_key!r} -- confirm the key name")
            continue
        expected, actual = int(_lookup(config, path)), int(live_settings[sleeper_key])
        if expected != actual:
            lines.append(f"  {name}: config={expected!r} live {sleeper_key}={actual!r}")

    live_ir = {
        key.removeprefix(IR_STATUS_PREFIX).upper()
        for key, value in live_settings.items()
        if key.startswith(IR_STATUS_PREFIX) and value
    }
    config_ir = set(config["ir"]["eligible_statuses"])
    if live_ir != config_ir:
        lines.append(
            f"  ir.eligible_statuses: config={sorted(config_ir)!r} live={sorted(live_ir)!r}"
        )
    return lines


def main() -> int:
    load_dotenv()
    config = load_config()
    league_id = config.get("league_id")

    if not league_id or league_id == "TODO_FILL_IN":
        print(
            "ERROR: config/league.json has league_id = 'TODO_FILL_IN'. "
            "Fill in the real Sleeper league_id before running verify_league.",
            file=sys.stderr,
        )
        return 1

    with SleeperClient() as client:
        league = client.get_league_live(league_id)

    mismatches: list[str] = []
    mismatches += diff_scoring(config["scoring"], league.scoring_settings)
    mismatches += diff_roster_positions(config["roster_positions"], league.roster_positions)
    mismatches += diff_settings(config, league.settings)

    print(f"League: {league.name} ({league.season})")
    print(f"Teams: {league.total_rosters}")
    print(f"Roster slots: {league.roster_positions}")
    print(f"IR slots: {league.settings.get('reserve_slots')}")

    if mismatches:
        print(f"\nMISMATCH between config/league.json and live Sleeper league {league_id}:")
        for line in mismatches:
            print(line)
        return 1

    print("\nScoring, roster positions, IR, waivers, playoffs and trade deadline match config/league.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
