"""Box score -> fantasy points. The single implementation (CLAUDE.md, conventions).

Every weight and every bonus threshold is read from ``config/league.json``; nothing about the
scoring system is hardcoded here. What *is* hardcoded is basketball: which five categories can
form a double-double, and that "double" means ten. Those are rules of the sport, not settings
the commissioner can change.

Two readings of the rules are unresolved (SPEC §4) and live behind flags in
``config/league.json``:

* ``bonuses_stack``   -- does a 50-point game pay the 40-point bonus as well?
* ``td_includes_dd``  -- does a triple-double also pay the double-double bonus?

Both branches are implemented. Which one is active is a property of the loaded rules, so the
UI can surface it; it is never decided inside this module.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict

CONFIG_PATH = Path("config/league.json")

# Rules of basketball, not league settings.
DOUBLE_DIGIT_THRESHOLD = 10
COUNTABLE_CATEGORIES: tuple[str, ...] = ("pts", "reb", "ast", "stl", "blk")

# Box score field -> the key Sleeper uses for it in scoring_settings.
#
# Sleeper's names are abbreviated and a few are not obvious (tpm, tf, ff), so box scores keep
# readable field names and this table does the translation. It is NOT guessed: it was read off
# the live league payload on 2026-09-23, where all 13 keys were present and every weight
# matched what had been transcribed by hand. config/league.json stores Sleeper's spelling, so
# verify_league can diff the two with no translation of its own.
STAT_FIELD_TO_SCORING_KEY: dict[str, str] = {
    "pts": "pts",
    "reb": "reb",
    "ast": "ast",
    "stl": "stl",
    "blk": "blk",
    "to": "to",
    "fg3m": "tpm",
    "tech_foul": "tf",
    "flagrant_foul": "ff",
}

# Those weights are signed in the config (turnovers and fouls are negative), so scoring them is
# a dot product -- this module never applies a sign of its own.
STAT_FIELDS: tuple[str, ...] = tuple(STAT_FIELD_TO_SCORING_KEY)

# config["scoring"] keys that are bonuses rather than per-stat weights.
DD_KEY = "dd"
TD_KEY = "td"
_POINTS_BONUS_KEY = re.compile(r"^bonus_pt_(\d+)p$")

_QUADRUPLE_DOUBLE_POLICIES = frozenset({"treat_as_td"})


class BoxScore(BaseModel):
    """One player's line in one game."""

    model_config = ConfigDict(extra="forbid")

    pts: float = 0.0
    reb: float = 0.0
    ast: float = 0.0
    stl: float = 0.0
    blk: float = 0.0
    to: float = 0.0
    fg3m: float = 0.0
    tech_foul: float = 0.0
    flagrant_foul: float = 0.0


@dataclass(frozen=True)
class ScoringRules:
    """Resolved league scoring, built from config/league.json."""

    weights: dict[str, float]
    points_bonuses: tuple[tuple[int, float], ...]  # (threshold, value), ascending
    bonuses_stack: bool
    td_includes_dd: bool
    quadruple_double: str


@dataclass(frozen=True)
class ScoreBreakdown:
    """Why a line scored what it scored. The board and the nightly alert both need to show
    this, and the golden tests assert against the parts, not just the total."""

    base: float
    double_double: float
    triple_double: float
    points_bonus: float
    total: float
    double_digit_categories: tuple[str, ...]


def load_rules(config_path: Path | str = CONFIG_PATH) -> ScoringRules:
    """Read scoring rules out of config/league.json.

    Fails loudly on anything missing or unrecognised. A silently-defaulted weight would put a
    wrong number into every downstream projection (CLAUDE.md, rule 4).
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    scoring = config["scoring"]

    missing = [
        f"{field} ({key})"
        for field, key in STAT_FIELD_TO_SCORING_KEY.items()
        if key not in scoring
    ]
    if missing:
        raise ValueError(
            f"config scoring is missing weights for {missing}; "
            "refusing to score with defaulted values"
        )

    points_bonuses = []
    for key, value in scoring.items():
        match = _POINTS_BONUS_KEY.match(key)
        if match:
            points_bonuses.append((int(match.group(1)), float(value)))
    points_bonuses.sort()

    known = (
        set(STAT_FIELD_TO_SCORING_KEY.values())
        | {DD_KEY, TD_KEY}
        | {f"bonus_pt_{threshold}p" for threshold, _ in points_bonuses}
    )
    unknown = sorted(key for key in scoring if key not in known)
    if unknown:
        raise ValueError(
            f"config scoring has keys this module does not know how to apply: {unknown}. "
            "Add them here deliberately rather than letting them be silently ignored"
        )

    for key in (DD_KEY, TD_KEY):
        if key not in scoring:
            raise ValueError(f"config scoring is missing the {key!r} bonus")

    ambiguities = config["scoring_ambiguities"]
    quadruple_double = ambiguities["quadruple_double"]["value"]
    if quadruple_double not in _QUADRUPLE_DOUBLE_POLICIES:
        raise ValueError(
            f"unknown quadruple_double policy {quadruple_double!r}; "
            f"expected one of {sorted(_QUADRUPLE_DOUBLE_POLICIES)}"
        )

    # Keyed by box-score field name, not by Sleeper's spelling: past this point the rest of
    # the engine only ever speaks in readable field names.
    weights = {
        field: float(scoring[key]) for field, key in STAT_FIELD_TO_SCORING_KEY.items()
    }
    weights[DD_KEY] = float(scoring[DD_KEY])
    weights[TD_KEY] = float(scoring[TD_KEY])

    return ScoringRules(
        weights=weights,
        points_bonuses=tuple(points_bonuses),
        bonuses_stack=bool(ambiguities["bonuses_stack"]["value"]),
        td_includes_dd=bool(ambiguities["td_includes_dd"]["value"]),
        quadruple_double=quadruple_double,
    )


@lru_cache(maxsize=None)
def default_rules() -> ScoringRules:
    """The league's own rules, read once. Callers needing a variant (both branches of an
    ambiguity flag, say) build it with :func:`with_flags`."""
    return load_rules()


def with_flags(
    rules: ScoringRules,
    *,
    bonuses_stack: bool | None = None,
    td_includes_dd: bool | None = None,
) -> ScoringRules:
    """A copy of ``rules`` with one or both ambiguity flags overridden.

    Used to show the user what a player is worth under each reading, and by the tests to run
    the same fixtures through both branches.
    """
    return replace(
        rules,
        bonuses_stack=rules.bonuses_stack if bonuses_stack is None else bonuses_stack,
        td_includes_dd=rules.td_includes_dd if td_includes_dd is None else td_includes_dd,
    )


def double_digit_categories(box: BoxScore) -> tuple[str, ...]:
    """The countable categories in which this line reached double figures."""
    return tuple(
        category
        for category in COUNTABLE_CATEGORIES
        if getattr(box, category) >= DOUBLE_DIGIT_THRESHOLD
    )


def _double_bonus(n_categories: int, rules: ScoringRules) -> tuple[float, float]:
    """(double_double, triple_double) bonus for a line with ``n_categories`` doubles.

    Four or more categories is a quadruple-double: vanishingly rare, and the config says to
    pay it as a triple-double rather than crash on it (SPEC §4).
    """
    if n_categories < 2:
        return 0.0, 0.0
    if n_categories == 2:
        return rules.weights[DD_KEY], 0.0
    # Three, or -- under the quadruple_double policy -- more.
    dd = rules.weights[DD_KEY] if rules.td_includes_dd else 0.0
    return dd, rules.weights[TD_KEY]


def _points_bonus(points: float, rules: ScoringRules) -> float:
    """Scoring-volume bonus. Either every threshold cleared pays, or only the highest does."""
    cleared = [value for threshold, value in rules.points_bonuses if points >= threshold]
    if not cleared:
        return 0.0
    if rules.bonuses_stack:
        return sum(cleared)
    return cleared[-1]  # points_bonuses is ascending, so this is the highest threshold cleared


def score_breakdown(box: BoxScore, rules: ScoringRules | None = None) -> ScoreBreakdown:
    """Score one line, itemised. Pure: no I/O, no network, no global state."""
    rules = default_rules() if rules is None else rules

    base = sum(rules.weights[field] * getattr(box, field) for field in STAT_FIELDS)
    doubles = double_digit_categories(box)
    dd, td = _double_bonus(len(doubles), rules)
    pts_bonus = _points_bonus(box.pts, rules)

    return ScoreBreakdown(
        base=base,
        double_double=dd,
        triple_double=td,
        points_bonus=pts_bonus,
        total=base + dd + td + pts_bonus,
        double_digit_categories=doubles,
    )


def score_box(box: BoxScore, rules: ScoringRules | None = None) -> float:
    """Fantasy points for one line. Every fantasy-point number in this codebase comes from
    here."""
    return score_breakdown(box, rules).total
