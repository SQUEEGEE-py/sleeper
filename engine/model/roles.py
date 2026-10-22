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

No hand-entered overrides; this only reports. Pure: no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_GAMES = 5
MINUTES_FLAG = 4.0


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
