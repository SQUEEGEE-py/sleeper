"""Mapping NBA stats player IDs to Sleeper player IDs (SPEC §6.3).

Sleeper's NBA player dictionary carries no NBA.com id -- it has `sportradar_id`, `swish_id`,
`rotowire_id`, `kalshi_id` and others, but not the one we need -- so the join is by name, and
names do not agree across sources. Normalising accents, punctuation and generational suffixes
("Jaren Jackson Jr." vs "Jaren Jackson"), then breaking the remaining ties by team, matched
**582 of 582** players who appeared in 2025-26 when this was written.

That will not stay true. Rookies arrive, players change teams, and Sleeper occasionally lists
someone under a different name. So:

**An unmapped player is a visible gap, never a dropped row** (CLAUDE.md rule 4). Mapping
returns the failures alongside the successes and every caller is expected to surface them.
Manual fixes go in ``config/player_overrides.csv``, which is also where role changes live.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

OVERRIDES_PATH = Path("config/player_overrides.csv")

# Generational suffixes. Sleeper and NBA.com disagree about these constantly, and they carry
# no identifying information once first and last name match.
_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def normalize_name(name: str) -> str:
    """A name reduced to a comparable key.

    Strips accents ("Dončić" -> "doncic"), punctuation ("D'Angelo" -> "dangelo",
    "Karl-Anthony" -> "karl anthony"), generational suffixes, and finally all non-alphanumerics.
    """
    folded = "".join(
        char
        for char in unicodedata.normalize("NFKD", str(name))
        if not unicodedata.combining(char)
    )
    folded = folded.lower().replace(".", " ").replace("'", "").replace("-", " ")
    folded = _SUFFIXES.sub(" ", folded)
    return "".join(char for char in folded if char.isalnum())


@dataclass(frozen=True)
class PlayerOverride:
    """One row of ``config/player_overrides.csv``.

    Two jobs in one file, because both are "the automated answer is wrong for this player":

    * **Identity** -- ``sleeper_id`` forces a mapping the name match could not make.
    * **Projection** -- ``value_multiplier`` and ``p_play`` adjust for a role change that
      history cannot know about. SPEC §5.1 calls for this because of the heavy 2026 offseason
      movement: a player whose usage is about to jump has a historical distribution that
      understates him, and no amount of clever weighting fixes that.
    """

    name: str
    sleeper_id: str | None = None
    nba_player_id: int | None = None
    value_multiplier: float = 1.0
    p_play: float | None = None
    note: str = ""


@dataclass
class MappingResult:
    """Successful mappings plus every failure, so callers can show the gaps."""

    nba_to_sleeper: dict[int, str] = field(default_factory=dict)
    unmapped: list[tuple[int, str, str]] = field(default_factory=list)  # (id, name, team)
    ambiguous: list[tuple[int, str, list[str]]] = field(default_factory=list)

    @property
    def mapped_count(self) -> int:
        return len(self.nba_to_sleeper)

    @property
    def gap_count(self) -> int:
        return len(self.unmapped) + len(self.ambiguous)

    def describe(self) -> str:
        total = self.mapped_count + self.gap_count
        if not self.gap_count:
            return f"player identity: {self.mapped_count}/{total} mapped, no gaps"
        return (
            f"player identity: {self.mapped_count}/{total} mapped, "
            f"{len(self.unmapped)} unmapped, {len(self.ambiguous)} ambiguous "
            f"-- add them to {OVERRIDES_PATH} rather than ignoring them"
        )


def load_overrides(path: Path | str = OVERRIDES_PATH) -> list[PlayerOverride]:
    """Read the manual override table. A missing file is fine and means no overrides."""
    path = Path(path)
    if not path.exists():
        return []

    # The file is heavily commented for whoever edits it by hand, so strip comment lines
    # before the reader sees them -- otherwise the first one becomes the header row.
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        return []

    def optional(row: dict, column: str) -> str | None:
        value = (row.get(column) or "").strip()
        return value or None

    overrides = []
    for row in csv.DictReader(lines):
        if not (row.get("name") or "").strip():
            continue
        nba_player_id = optional(row, "nba_player_id")
        multiplier = optional(row, "value_multiplier")
        p_play = optional(row, "p_play")
        overrides.append(
            PlayerOverride(
                name=row["name"].strip(),
                sleeper_id=optional(row, "sleeper_id"),
                nba_player_id=int(nba_player_id) if nba_player_id else None,
                value_multiplier=float(multiplier) if multiplier else 1.0,
                p_play=float(p_play) if p_play else None,
                note=(row.get("note") or "").strip(),
            )
        )
    return overrides


def overrides_by_nba_id(
    overrides: list[PlayerOverride], name_to_id: dict[str, int]
) -> dict[int, PlayerOverride]:
    """Key the override rows by NBA player id, resolving those given only by name."""
    resolved: dict[int, PlayerOverride] = {}
    for override in overrides:
        player_id = override.nba_player_id
        if player_id is None:
            player_id = name_to_id.get(normalize_name(override.name))
        if player_id is not None:
            resolved[player_id] = override
    return resolved


def map_players(
    nba_players: list[tuple[int, str, str]],
    sleeper_players: dict[str, object],
    overrides: list[PlayerOverride] | None = None,
) -> MappingResult:
    """Join NBA players to Sleeper ids.

    ``nba_players`` is ``(nba_player_id, name, team)``; ``sleeper_players`` is the dict from
    ``SleeperClient.get_players()``. Overrides win outright over any name match.
    """
    by_name: dict[str, list[tuple[str, object]]] = {}
    for sleeper_id, player in sleeper_players.items():
        extra = getattr(player, "model_extra", None) or {}
        raw_name = extra.get("search_full_name") or getattr(player, "full_name", None)
        if not raw_name:
            continue
        by_name.setdefault(normalize_name(raw_name), []).append((sleeper_id, player))

    forced: dict[int, str] = {}
    forced_by_name: dict[str, str] = {}
    for override in overrides or []:
        if override.sleeper_id:
            if override.nba_player_id is not None:
                forced[override.nba_player_id] = override.sleeper_id
            else:
                forced_by_name[normalize_name(override.name)] = override.sleeper_id

    result = MappingResult()
    for nba_id, name, team in nba_players:
        key = normalize_name(name)

        if nba_id in forced:
            result.nba_to_sleeper[nba_id] = forced[nba_id]
            continue
        if key in forced_by_name:
            result.nba_to_sleeper[nba_id] = forced_by_name[key]
            continue

        candidates = by_name.get(key, [])
        if not candidates:
            result.unmapped.append((nba_id, name, team))
            continue
        if len(candidates) == 1:
            result.nba_to_sleeper[nba_id] = candidates[0][0]
            continue

        # Same name, more than one player. Team breaks it; anything left is a real ambiguity
        # a human has to resolve, not something to guess at.
        same_team = [
            (sleeper_id, player)
            for sleeper_id, player in candidates
            if getattr(player, "team", None) == team
        ]
        if len(same_team) == 1:
            result.nba_to_sleeper[nba_id] = same_team[0][0]
        else:
            result.ambiguous.append((nba_id, name, [sid for sid, _ in candidates]))

    return result
