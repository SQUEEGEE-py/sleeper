"""Settle the two open scoring flags from what Sleeper actually counted (phase 3 step 5).

    uv run python -m engine.jobs.resolve_scoring                      # last season's league
    uv run python -m engine.jobs.resolve_scoring --league <id> --season 2026

``config/league.json`` leaves two rules open (SPEC §4): does a 50-point game pay both points
bonuses (``bonuses_stack``), and does a triple-double also pay the double-double bonus
(``td_includes_dd``)? Sleeper's matchups record, for every rostered player and every week, the
score it counted -- in Lock-In, one of his games that week (or 0). So for each player-week we
rescore each of his games under all four flag combinations through ``scoring.py`` and ask
which combinations reproduce Sleeper's number.

Most player-weeks cannot tell the combinations apart: without a triple-double or a 50-point
game all four give the same scores. The ones that can are the evidence, and are listed. Last
season's league (``previous_league_id``) had identical scoring settings and was Lock-In too, so
a whole season is available instead of one week.

Read-only; prints the tally. Changing the config is a separate, deliberate edit.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

from engine.ingest.sleeper import PlayerGame, SleeperClient
from engine.jobs.runtime import utf8_output
from engine.model.scoring import BoxScore, ScoringRules, box_from_sleeper_stats, load_rules, score_box, with_flags

CONFIG_PATH = Path("config/league.json")
MATCH_TOLERANCE = 0.05
Combo = tuple[bool, bool]  # (bonuses_stack, td_includes_dd)
COMBOS: tuple[Combo, ...] = tuple(product((False, True), repeat=2))


@dataclass
class PlayerWeek:
    player: str
    week: int
    counted: float
    games: list[tuple[str, BoxScore]]  # (date, box)


@dataclass
class Tally:
    weeks: int = 0
    undecided: int = 0                      # every combination matches equally
    unexplained: list[str] = field(default_factory=list)   # no combination matches
    evidence: list[tuple[str, frozenset[Combo]]] = field(default_factory=list)

    def consistent(self, combo: Combo) -> int:
        return sum(combo in matching for _, matching in self.evidence)


def matching_combos(week: PlayerWeek, rules_by_combo: dict[Combo, ScoringRules]) -> frozenset[Combo]:
    """The flag combinations under which Sleeper's counted score is one of his games' scores,
    or 0 (a week whose counted game he sat)."""
    matching = set()
    for combo, rules in rules_by_combo.items():
        scores = [score_box(box, rules) for _, box in week.games] + [0.0]
        if any(abs(score - week.counted) < MATCH_TOLERANCE for score in scores):
            matching.add(combo)
    return frozenset(matching)


def tally(weeks: list[PlayerWeek], base_rules: ScoringRules) -> Tally:
    rules_by_combo = {combo: with_flags(base_rules, bonuses_stack=combo[0], td_includes_dd=combo[1])
                      for combo in COMBOS}
    result = Tally()
    for week in weeks:
        result.weeks += 1
        matching = matching_combos(week, rules_by_combo)
        label = f"week {week.week} {week.player}: Sleeper counted {week.counted:g}"
        if not matching:
            scores = ", ".join(f"{day} {score_box(box, base_rules):g}" for day, box in week.games)
            result.unexplained.append(f"{label}; games {scores or 'none'}")
        elif len(matching) == len(COMBOS):
            result.undecided += 1
        else:
            result.evidence.append((label, matching))
    return result


def describe(combo: Combo) -> str:
    return f"bonuses_stack={str(combo[0]).lower()}, td_includes_dd={str(combo[1]).lower()}"


def main() -> int:
    utf8_output()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--league", default=config["keepers"]["previous_league_id"])
    parser.add_argument("--season", default=None, help="stats season, e.g. 2025; default the league's own")
    args = parser.parse_args()

    with SleeperClient() as client:
        league = client.get_league_live(args.league)
        if league.scoring_settings != config["scoring"]:
            raise SystemExit(f"league {args.league} scores differently from config; its evidence does not apply")
        season = args.season or league.season
        last_week = int(league.settings.get("last_scored_leg") or league.settings.get("leg") or 0)
        players = client.get_players()
        games_by_player: dict[str, list[PlayerGame]] = {}
        weeks: list[PlayerWeek] = []
        for week in range(1, last_week + 1):
            for matchup in client.get_matchups_live(args.league, week):
                for player_id, counted in (matchup.players_points or {}).items():
                    if player_id not in games_by_player:
                        games_by_player[player_id] = client.get_player_games(player_id, season)
                    player = players.get(player_id)
                    games = [(g.date, box_from_sleeper_stats(g.stats))
                             for g in games_by_player[player_id] if g.week == week]
                    weeks.append(PlayerWeek((player.full_name if player else None) or player_id,
                                            week, float(counted), games))

    result = tally(weeks, load_rules())
    print(f"{league.name} {league.season}: {result.weeks} player-weeks, weeks 1-{last_week}, "
          f"{len(games_by_player)} players")
    print(f"  all four combinations agree: {result.undecided}")
    print(f"  no combination matches:      {len(result.unexplained)}")
    print(f"  can tell them apart:         {len(result.evidence)}")
    print("\nConsistent with Sleeper, out of the player-weeks that can tell them apart:")
    for combo in COMBOS:
        print(f"  {describe(combo):45} {result.consistent(combo)} / {len(result.evidence)}")
    print("\nThe deciding player-weeks:")
    for label, matching in result.evidence:
        print(f"  {label}  -> matches {'; '.join(describe(c) for c in sorted(matching))}")
    if result.unexplained:
        print(f"\nUnexplained ({len(result.unexplained)}; first 15):")
        for line in result.unexplained[:15]:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
