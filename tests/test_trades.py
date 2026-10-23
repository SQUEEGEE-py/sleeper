"""Trade finder: only trades that improve both lineups. No network."""

from __future__ import annotations

from engine.model.trades import TradePlayer, find_trades, lineup_value

SLOTS = ["G#1", "F#1", "C#1"]


def tp(pid: str, pos: str, value: float, rank: float | None = None) -> TradePlayer:
    return TradePlayer(pid, pid, (pos,), value, rank)


def test_lineup_value_is_the_best_starting_three() -> None:
    team = [tp("g", "PG", 40), tp("f1", "SF", 45), tp("f2", "PF", 44), tp("c", "C", 30)]
    assert lineup_value(team, SLOTS) == 40 + 45 + 30  # f2 is surplus: one F slot


def test_surplus_for_need_is_found_both_ways() -> None:
    # I have two good forwards and a weak centre; they have two good centres and a weak forward.
    mine = [tp("g", "PG", 40), tp("f1", "SF", 45), tp("f2", "PF", 44), tp("c", "C", 30)]
    theirs = [tp("G", "PG", 40), tp("F", "SF", 31), tp("C1", "C", 43), tp("C2", "C", 42)]
    trades = find_trades(mine, {"rival": theirs}, SLOTS)
    best = trades[0]
    # f2 for C1: I gain 43 - 30 = 13; they lose C1 but C2 (42) starts, and f2 (44) replaces
    # F (31): 13 - 1 = 12. Better for me than f2 for C2 (+12 / +13), so it ranks first.
    assert [p.player_id for p in best.give] == ["f2"] and [p.player_id for p in best.get] == ["C1"]
    assert best.my_gain == 13.0 and best.their_gain == 12.0


def test_a_trade_only_one_side_gains_from_is_not_listed() -> None:
    mine = [tp("g", "PG", 40), tp("f", "SF", 45), tp("c", "C", 30)]
    theirs = [tp("G", "PG", 40), tp("F", "SF", 31), tp("C", "C", 43)]
    # Swapping centres helps me +13 and hurts them -13: never shown.
    assert find_trades(mine, {"rival": theirs}, SLOTS) == []


def test_minimum_gain_applies_to_each_side() -> None:
    mine = [tp("g", "PG", 40), tp("f1", "SF", 45), tp("f2", "PF", 44), tp("c", "C", 43.5)]
    theirs = [tp("G", "PG", 40), tp("F", "SF", 43.6), tp("C1", "C", 44), tp("C2", "C", 44)]
    # f2 for C2: me +0.5, them +0.4 -- both positive, both under 1.0.
    assert find_trades(mine, {"rival": theirs}, SLOTS) == []
    assert find_trades(mine, {"rival": theirs}, SLOTS, min_gain=0.3) != []


def test_two_for_one_consolidation() -> None:
    # Two decent forwards for their star: I start the star and my bench player moves up to
    # UTIL (40 + 50 + 40 + 30 = 160 vs 155); they fill an empty UTIL (20 + 38 + 40 + 37 = 135
    # vs 110).
    mine = [tp("g", "PG", 40), tp("f1", "SF", 38), tp("f2", "PF", 37), tp("c", "C", 40), tp("b", "PG", 30)]
    theirs = [tp("G", "PG", 20), tp("F", "SF", 50), tp("C", "C", 40)]
    slots = ["G#1", "F#1", "C#1", "UTIL#1"]
    trades = find_trades(mine, {"rival": theirs}, slots, per_partner=50, limit=50)
    consolidation =[t for t in trades if {p.player_id for p in t.give} == {"f1", "f2"}
                     and [p.player_id for p in t.get] == ["F"]]
    assert consolidation and consolidation[0].my_gain == 5.0 and consolidation[0].their_gain == 25.0
    assert all(t.my_gain >= 1.0 and t.their_gain >= 1.0 for t in trades)


def test_describe_shows_market_ranks() -> None:
    mine = [tp("g", "PG", 40), tp("f1", "SF", 45, 9), tp("f2", "PF", 44, 40), tp("c", "C", 30, 120)]
    theirs = [tp("G", "PG", 40), tp("F", "SF", 31, 90), tp("C1", "C", 43, 20), tp("C2", "C", 42, 35)]
    text = find_trades(mine, {"rival": theirs}, SLOTS)[0].describe()
    assert "give f2 (Slpr #40)" in text and "get C1 (Slpr #20)" in text and "you +13.0/week" in text


def test_a_throw_in_that_changes_nothing_is_not_a_second_trade() -> None:
    # Live case 2026-10-22: Kawhi + LeBron for Haliburton, and the same plus Jarrett Allen
    # (who starts for neither side) at identical gains. Only the smaller trade is shown.
    mine = [tp("g", "PG", 40), tp("f1", "SF", 45), tp("f2", "PF", 44), tp("c", "C", 30)]
    theirs = [tp("G", "PG", 40), tp("F", "SF", 31), tp("C1", "C", 43), tp("C2", "C", 42), tp("junk", "PG", 1)]
    trades = find_trades(mine, {"rival": theirs}, SLOTS, per_partner=50, limit=50)
    shapes = [({p.player_id for p in t.give}, {p.player_id for p in t.get}) for t in trades]
    assert ({"f2"}, {"C1"}) in shapes and ({"f2"}, {"C1", "junk"}) not in shapes


def test_deadline_countdown_and_posting_days() -> None:
    import json
    from datetime import date
    from pathlib import Path
    from engine.model.trades import deadline, post_trades_today
    week = json.loads(Path("config/league.json").read_text(encoding="utf-8"))["trade_deadline_week"]
    last = date(2027, 2, 28)
    assert week == 19
    early = deadline(week, 1, last)
    assert early.weeks_left == 19 and "19 week(s) left" in early.describe()
    assert post_trades_today(early, date(2026, 10, 26))      # a Monday
    assert not post_trades_today(early, date(2026, 10, 22))  # a Thursday, far from the deadline
    assert post_trades_today(deadline(week, 18, last), date(2027, 2, 18))  # two weeks out: daily
    gone = deadline(week, 20, last)
    assert gone.passed and not post_trades_today(gone, date(2027, 3, 1)) and "passed" in gone.describe()
