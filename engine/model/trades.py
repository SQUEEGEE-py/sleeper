"""Trade finder (phase 3 step 7): trades that make both starting lineups better. Pure: no I/O.

A team's strength is its best starting lineup -- the exact slot assignment of
``lineup.assign_lineup`` -- valued by each player's ``season_value`` (expected lock-in points
per week over his remaining schedule, from the board's ``pool.csv``). A trade is listed only
if it raises **both** teams' lineup values by at least ``MIN_GAIN``: a trade only one side
gains from is a trade the other side refuses, so showing it would be noise.

What the other manager actually sees is the market, not our model, so each trade also carries
both sides' **Sleeper rank** (``search_rank``, what the app shows by default). Our model rates
old players lower than the market does (``aging.py``); a trade the market calls even that our
model calls a win is the edge worth pitching. The tool never proposes anything in Sleeper --
the user decides and sends it.

Stated limits:

* lineup value only. In Lock-In a bench player matters as cover for a starter who sits,
  which this does not price, so 2-for-1 trades favour the side receiving two slightly;
* in a 2-for-1 the side receiving two must release someone; the lineup value assumes it is
  his weakest non-starter, so it is unchanged by the release;
* our values for both teams. The partner's own view of his roster is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import combinations
from typing import Sequence

from engine.model.lineup import assign_lineup

MIN_GAIN = 1.0  # lock-in points per week, each side
MAX_PLAYERS_PER_SIDE = 2
DAILY_WITHIN_WEEKS = 2  # Discord: trades every day this close to the deadline, else Mondays


@dataclass(frozen=True)
class Deadline:
    """The trade deadline is the end of ``config["trade_deadline_week"]``: its last game day."""

    week: int
    last_day: date
    weeks_left: int  # fantasy weeks from the current one to the deadline week, inclusive

    @property
    def passed(self) -> bool:
        return self.weeks_left <= 0

    def describe(self) -> str:
        if self.passed:
            return f"trade deadline passed (end of week {self.week}, {self.last_day:%a %b %d})"
        return (f"trade deadline: end of week {self.week}, {self.last_day:%a %b %d} -- "
                f"{self.weeks_left} week(s) left including this one")


def deadline(deadline_week: int, current_week: int, last_game_day: date) -> Deadline:
    return Deadline(deadline_week, last_game_day, deadline_week - current_week + 1)


def post_trades_today(dl: Deadline, today: date) -> bool:
    """Discord gets trade ideas on Mondays, and every day in the deadline's last two weeks."""
    return not dl.passed and (today.weekday() == 0 or dl.weeks_left <= DAILY_WITHIN_WEEKS)


@dataclass(frozen=True)
class TradePlayer:
    player_id: str
    name: str
    positions: tuple[str, ...]
    value: float                 # season_value: lock-in points per week
    market_rank: float | None = None  # Sleeper search_rank; lower is better


def lineup_value(players: Sequence[TradePlayer], slots: list[str]) -> float:
    return assign_lineup([p.value for p in players], [list(p.positions) for p in players], slots).total_value


@dataclass(frozen=True)
class Trade:
    partner: str
    give: tuple[TradePlayer, ...]
    get: tuple[TradePlayer, ...]
    my_gain: float
    their_gain: float

    def describe(self) -> str:
        def side(players: tuple[TradePlayer, ...]) -> str:
            return " + ".join(f"{p.name} (Slpr #{p.market_rank:.0f})" if p.market_rank else p.name
                              for p in players)
        return (f"with {self.partner}: give {side(self.give)}, get {side(self.get)} -- "
                f"you +{self.my_gain:.1f}/week, them +{self.their_gain:.1f}/week")


PADDING_TOLERANCE = 0.1  # points per week: a throw-in changing less than this changes nothing


def _padded(trade: Trade, smaller: Trade) -> bool:
    """``trade`` is ``smaller`` plus extra players, for (about) the same gains."""
    ids = lambda players: {p.player_id for p in players}
    return (ids(smaller.give) <= ids(trade.give) and ids(smaller.get) <= ids(trade.get)
            and (ids(smaller.give), ids(smaller.get)) != (ids(trade.give), ids(trade.get))
            and abs(trade.my_gain - smaller.my_gain) < PADDING_TOLERANCE
            and abs(trade.their_gain - smaller.their_gain) < PADDING_TOLERANCE)


def _packages(players: Sequence[TradePlayer], size: int) -> list[tuple[TradePlayer, ...]]:
    return [combo for n in range(1, size + 1) for combo in combinations(players, n)]


def find_trades(mine: Sequence[TradePlayer], partners: dict[str, Sequence[TradePlayer]],
                slots: list[str], min_gain: float = MIN_GAIN,
                max_per_side: int = MAX_PLAYERS_PER_SIDE, per_partner: int = 2,
                limit: int = 5) -> list[Trade]:
    """Trades of up to ``max_per_side`` players each way that improve both lineups, best for
    the user first, at most ``per_partner`` per manager."""
    my_base = lineup_value(mine, slots)
    found: list[Trade] = []
    for partner, theirs in partners.items():
        their_base = lineup_value(theirs, slots)
        best: list[Trade] = []
        for give in _packages(mine, max_per_side):
            give_ids = {p.player_id for p in give}
            mine_after_give = [p for p in mine if p.player_id not in give_ids]
            for get in _packages(theirs, max_per_side):
                my_gain = lineup_value(mine_after_give + list(get), slots) - my_base
                if my_gain < min_gain:
                    continue
                get_ids = {p.player_id for p in get}
                their_gain = lineup_value([p for p in theirs if p.player_id not in get_ids] + list(give),
                                          slots) - their_base
                if their_gain >= min_gain:
                    best.append(Trade(partner, give, get, my_gain, their_gain))
        # Fewest players first among equals, then drop any trade that is a kept one plus
        # throw-ins changing nothing (Haliburton vs Haliburton + Jarrett Allen, both +1.7).
        best.sort(key=lambda t: (-round(t.my_gain, 1), -round(t.their_gain, 1), len(t.give) + len(t.get)))
        kept: list[Trade] = []
        for trade in best:
            if not any(_padded(trade, smaller) for smaller in kept):
                kept.append(trade)
        found += kept[:per_partner]
    return sorted(found, key=lambda t: (-t.my_gain, -t.their_gain))[:limit]
