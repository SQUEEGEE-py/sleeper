"""Role watch: has a player's job changed this season? (phase 3 step 6, part 2)

The model learns a new role by itself through recency weighting, but slowly -- after 10 games
this season carries about 24% of a player's weight. Minutes move first, so they are the early
warning: a player whose minutes have jumped or collapsed is worth a look before the model has
caught up (a waiver target, a sell, a lineup question).

Rules, deliberately simple and stated:

* nothing is judged before ``MIN_GAMES`` this season -- one or two games are noise;
* flagged when minutes per game this season differ from his most recent earlier season's by ``MINUTES_FLAG`` or
  more (4 minutes is about 12% of a 32-minute starter, and two-thirds of one of the model's
  six-minute role bands);
* a team change is reported alongside, not as a flag by itself: Brown and LeBron to PHI or
  Kawhi to TOR matter only if their minutes move.

No hand-entered overrides. Pure: no I/O.

**Role factor (2026-11-15).** The report alone left the numbers lagging: twelve games into
2026-27, Vucevic (10 min a night, 8 fp) still valued 35.9 and DeRozan (23.6 min, 19.6 fp) 39.1,
so the pickup list could recommend players who had lost their jobs. ``role_factor`` scales a
veteran's distribution by

    (minutes this season / minutes in his most recent earlier season) ** (ROLE_ELASTICITY * share)

where ``share`` is the part of his recency weight still from earlier seasons, the same fade as
``aging.py``: it vanishes as this season's games take over the sample, so it never double
counts. Backtested on 2024-25 and 2025-26 (``PlayerModel`` from logs up to each team's game
8/12/20/35/50, predicted mean vs actual rest-of-season fp per game, 545-620 players with 5+
games before and 15+ after, 10+ minutes the season before): mean absolute error

    game  8: 4.80 -> 4.44     game 12: 4.71 -> 4.30     game 20: 4.43 -> 4.14
    game 35: 4.27 -> 4.06     game 50: 4.42 -> 4.35

and for players whose minutes moved 4+ (n ~ 210-235) 5.33 -> ~4.3 at game 12. 0.65 is within
0.02 of the best exponent at every checkpoint, and the same in both seasons. An unfaded
exponent (0.45) was as good early but worse than no adjustment by game 50. Nothing is applied
before ``MIN_GAMES`` this season or against an earlier season under ``ROLE_MIN_LAST_MINUTES``
(both outside what was backtested), and the earlier season is the most recent with
``ROLE_MIN_LAST_GAMES`` or more (Lillard, who missed 2025-26, is compared with 2024-25); rookies have no earlier season and are left alone.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_GAMES = 5
MINUTES_FLAG = 4.0
ROLE_ELASTICITY = 0.65  # backtested, see the module docstring
ROLE_MIN_LAST_MINUTES = 10.0
ROLE_MIN_LAST_GAMES = 10  # an earlier season of a handful of games says little about his role


def role_factor(minutes_now: float | None, games_now: int, minutes_last: float | None,
                previous_share: float) -> float:
    """Multiplier on a veteran's scores for a role change this season; 1.0 when not judged."""
    if (minutes_now is None or minutes_last is None or games_now < MIN_GAMES
            or minutes_last < ROLE_MIN_LAST_MINUTES or minutes_now <= 0):
        return 1.0
    return float((minutes_now / minutes_last) ** (ROLE_ELASTICITY * previous_share))


@dataclass(frozen=True)
class RoleCheck:
    name: str
    team: str
    last_team: str | None
    games: int
    minutes_now: float | None
    minutes_last: float | None
    last_season: str | None = None  # the season minutes_last is from

    @property
    def new_team(self) -> bool:
        return self.last_team is not None and self.last_team != self.team

    @property
    def delta(self) -> float | None:
        if self.minutes_now is None or self.minutes_last is None:
            return None
        return self.minutes_now - self.minutes_last

    @property
    def judged(self) -> bool:
        return self.games >= MIN_GAMES and self.delta is not None

    @property
    def flagged(self) -> bool:
        return self.judged and abs(self.delta) >= MINUTES_FLAG  # type: ignore[arg-type]

    def describe(self) -> str:
        team = f"{self.team} (new, was {self.last_team})" if self.new_team else self.team
        if self.minutes_last is None:
            return f"{self.name}, {team}: no earlier season to compare"
        if not self.judged:
            return (f"{self.name}, {team}: {self.games} game(s) so far, judged from {MIN_GAMES}; "
                    f"{self.last_season} {self.minutes_last:.1f} min")
        direction = "UP" if self.delta > 0 else "DOWN"  # type: ignore[operator]
        verdict = f"{direction} {abs(self.delta):.1f} min" if self.flagged else "steady"  # type: ignore[arg-type]
        return (f"{self.name}, {team}: {self.minutes_now:.1f} min over {self.games} games vs "
                f"{self.minutes_last:.1f} in {self.last_season} -- {verdict}")


def role_check(name: str, team: str, last_team: str | None, minutes_this_season: list[float],
               minutes_last: float | None, last_season: str | None = None) -> RoleCheck:
    """``minutes_this_season``: one entry per game he played this season. ``minutes_last``
    is from ``last_season``, his most recent earlier season (not always the previous one)."""
    games = len(minutes_this_season)
    now = sum(minutes_this_season) / games if games else None
    return RoleCheck(name, team, last_team, games, now, minutes_last, last_season)
