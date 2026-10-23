"""Infer whether a starter is locked, from the one number Sleeper does expose.

Probed 2026-10-21 (rule 6): Sleeper's public API has **no lock field** -- not in matchups,
rosters, roster settings or the league. What it has is ``starters_points``, the score each slot
currently counts. So:

* **during** his team's game: an unlocked starter's counted score follows the live game (Jalen
  Williams, 10/22), so counted == an earlier finished game means **locked** on it;
* **after** a game: counted == an **earlier** game's score is **"unclear"**, not locked.
  Corrected 2026-10-22: after the 76ers' Thursday game every 76ers *starter* in the league
  still counted Tuesday's score -- LeBron 38 (Thu 43), Brown 43.5 (Thu 41), Embiid 41.5 (Thu
  43.5), Maxey 56 (Thu 58), on three different rosters -- and the user confirmed he had locked
  neither of his. The one 76ers bench player counted Thursday's. So either Sleeper refreshes
  starters' counted scores after a delay, or it holds the earlier game for another reason;
  both look exactly like a lock. The caller decides what "unclear" means for its side;
* counted == the latest game  ->  "open": unlocked, *or* locked on that very game. The two are
  indistinguishable until his next game is played. For the user's own players the advice is
  therefore phrased "if you have not locked him"; for the opponent the simulation treats the
  decision as still open;
* counted matches no game  ->  "unexplained", shown as a gap. Either a game is still in
  progress or our rescoring disagrees with Sleeper's (the unresolved scoring flags), and a
  guess would be worse than saying so.

Known limit: Sleeper only counts games he *started*; a game he sat on your bench is invisible
here and treated like any other. Pure: no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Sequence

MATCH_TOLERANCE = 0.05  # scores are in half points; this only absorbs float noise
LockState = Literal["no_game_yet", "open", "locked", "unclear", "unexplained"]


@dataclass(frozen=True)
class LockStatus:
    state: LockState
    locked_on: date | None = None
    locked_score: float | None = None
    note: str = ""


def infer_lock(
    completed: Sequence[tuple[date, float]], counted: float, game_in_progress: bool = False
) -> LockStatus:
    """``completed``: this week's finished games for his team, in date order, with his score in
    each (0.0 if he did not play). ``counted``: Sleeper's ``starters_points`` for his slot.
    ``game_in_progress``: his team is playing right now, so ``counted`` may be a live score."""
    def matches(score: float) -> bool:
        return abs(score - counted) < MATCH_TOLERANCE

    if not completed:
        if game_in_progress or matches(0.0):
            return LockStatus("no_game_yet")
        return LockStatus("unexplained", note=f"Sleeper counts {counted:g} before any finished game")

    if game_in_progress:
        # Every finished game is in the past; a match means it was locked. No match means the
        # counted score is the live one, so he is not locked. A live score that happens to
        # equal an old one would read as locked -- rare, and fixed by the next run.
        earlier = [(day, score) for day, score in completed if matches(score)]
        if earlier:
            day, score = earlier[-1]
            return LockStatus("locked", day, score)
        return LockStatus("open", note="game in progress; not locked on any finished game")

    latest_day, latest_score = completed[-1]
    if matches(latest_score):
        return LockStatus("open", note=f"counting {latest_day:%a}'s {latest_score:g}; locked or not "
                                       "cannot be told apart until his next game")
    earlier = [(day, score) for day, score in completed[:-1] if matches(score)]
    if earlier:
        day, score = earlier[-1]
        return LockStatus("unclear", day, score,
                          note=f"Sleeper still shows {day:%a}'s {score:g} although {latest_day:%a}'s "
                               f"{latest_score:g} has finished: locked on {day:%a}, or Sleeper has not "
                               "caught up -- the public API cannot tell")
    scores = ", ".join(f"{day:%a} {score:g}" for day, score in completed)
    return LockStatus("unexplained", note=f"Sleeper counts {counted:g}; our rescoring has {scores}")
