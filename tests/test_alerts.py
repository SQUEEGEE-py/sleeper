"""ESPN tip times, Discord chunking and the phone report. No network."""

from __future__ import annotations

from datetime import date, datetime, timezone

from engine.ingest.espn import parse_tip_times
from engine.ingest.sleeper import Matchup
from engine.jobs.discord import chunk
from engine.jobs.nightly import Starter, deadline, render_discord
from engine.model.locks import LockStatus
from engine.model.matchup import Decision, Odds

THU = date(2026, 10, 22)


def event(tip: str, home: str, away: str) -> dict:
    return {"date": tip, "competitions": [{"competitors": [
        {"homeAway": "home", "team": {"abbreviation": home}},
        {"homeAway": "away", "team": {"abbreviation": away}}]}]}


def test_espn_tips_are_utc_and_team_codes_are_translated() -> None:
    payload = {"events": [event("2026-10-22T23:00Z", "PHI", "CLE"),
                          event("2026-10-23T01:30Z", "GS", "NY")]}
    tips = parse_tip_times(payload, THU)
    assert tips[("PHI", "2026-10-22")] == datetime(2026, 10, 22, 23, 0, tzinfo=timezone.utc)
    # A 9:30 pm ET game is the next day in UTC but keyed by its US date.
    assert tips[("GSW", "2026-10-22")] == tips[("NYK", "2026-10-22")]
    assert ("GS", "2026-10-22") not in tips


def test_unknown_espn_team_is_left_out_not_guessed() -> None:
    tips = parse_tip_times({"events": [event("2026-10-22T23:00Z", "XYZ", "PHI")]}, THU)
    assert list(tips) == [("PHI", "2026-10-22")]


def test_deadline_is_eastern_time() -> None:
    s = Starter("1", "Jaylen Brown", "PHI", 43.5)
    s.next_tip = datetime(2026, 10, 22, 23, 0, tzinfo=timezone.utc)
    assert deadline(s) == "Thu 7:00 PM ET"
    s.next_tip = datetime(2026, 11, 3, 0, 30, tzinfo=timezone.utc)  # after DST ends: EST
    assert deadline(s) == "Mon 7:30 PM ET"


def test_chunk_respects_the_limit_and_keeps_every_line() -> None:
    text = "\n".join(f"line {i:03d} " + "x" * 40 for i in range(100))
    parts = chunk(text, limit=500)
    assert all(len(p) <= 500 for p in parts)
    assert "\n".join(parts) == text
    long = chunk("y" * 1200, limit=500)
    assert [len(p) for p in long] == [500, 500, 200]


def test_phone_report_leads_with_odds_and_the_calls() -> None:
    s = Starter("1", "Kevin Durant", "HOU", 51.0)
    s.next_tip = datetime(2026, 10, 24, 1, 30, tzinfo=timezone.utc)
    d = Decision("Kevin Durant", 51.0, 49.3, "lock", 0.74, 0.73, 0.004, "lock")
    locked = Starter("2", "Nikola Jokic", "DEN", 60.0, lock=LockStatus("locked", THU, 60.0))
    odds = Odds(0.74, 0.0, 452.6, 415.3, -39, 112)
    mine_m = Matchup(roster_id=4, matchup_id=1, points=333.5, starters=[])
    their_m = Matchup(roster_id=3, matchup_id=1, points=387.0, starters=[])
    text = render_discord(1, "2jake4davis6", [s, locked], mine_m, their_m, odds, [(s, d)], ["a gap"])
    lines = text.splitlines()
    assert lines[0] == "**Week 1 vs 2jake4davis6: 74% to win**"
    assert "**LOCK Kevin Durant** -- before Fri 9:30 PM ET" in text
    assert "win 74% if locked / 73% if rolled" in text
    assert "- Nikola Jokic: locked 60" in text and "- a gap" in text
    assert "not** already locked" in text


def test_rerun_banner_names_the_change_and_supersedes() -> None:
    from engine.jobs.nightly import rerun_banner
    banner = rerun_banner("phase3: age curve from a backtest\n\nlong body that is not shown")
    assert "Re-run after a model change" in banner and "phase3: age curve from a backtest" in banner
    assert "long body" not in banner and "supersedes" in banner
    assert "model updated" in rerun_banner("   ")


def test_phone_report_mentions_every_starter() -> None:
    # User, 2026-10-27: "the nightly advisor didn't address everyone in my lineup".
    from engine.jobs.lock_sheet import GameSlot
    from engine.jobs.nightly import Swap
    from engine.model.distributions import NormalScoreDistribution
    from engine.model.matchup import Slot
    n45 = NormalScoreDistribution(45.0, 10.0)
    wed = GameSlot(date(2026, 10, 28), "IND", False, False)
    fri = GameSlot(date(2026, 10, 30), "NYK", True, False)
    decided = Starter("1", "Kyrie Irving", "DAL", 40.0)
    waiting = Starter("2", "Jaylen Brown", "PHI", 0.0, upcoming=[wed, fri], slot=Slot("Brown", n45, (1.0, 1.0)))
    waiting.next_tip = datetime(2026, 10, 28, 23, 0, tzinfo=timezone.utc)
    locked = Starter("3", "Nikola Jokic", "DEN", 53.0, lock=LockStatus("locked", None, 53.0, note="reported by you"))
    finished = Starter("4", "Bam Adebayo", "MIA", 41.0, completed=[(THU, 41.0)], slot=Slot("Bam", n45, ()))
    d = Decision("Kyrie Irving", 40.0, 39.5, "lock", 0.56, 0.56, 0.0, "lock")
    m = Matchup(roster_id=4, matchup_id=1, points=0.0, starters=[])
    swap = Swap("Kawhi Leonard", "Out", "F#1", [("Saddiq Bey", 43.1, ["Wed", "Thu B2B", "Sat"])])
    text = render_discord(2, "tristantro", [decided, waiting, locked, finished], m, m, None, [(decided, d)], [],
                          swaps=[swap])
    assert "**LOCK Kyrie Irving**" in text
    assert "- Jaylen Brown: next game Wed 7:00 PM ET -- lock it if he scores 45.0+" in text
    assert "- Nikola Jokic: locked 53" in text and "- Bam Adebayo: done for the week, 41 counts" in text
    assert text.index("Kawhi Leonard is Out") < text.index("LOCK Kyrie")  # injuries first
    assert "start Saddiq Bey at F" in text


def test_a_stand_in_is_one_instruction_not_a_lock_and_an_audition() -> None:
    # User, 2026-11-05: the report said LOCK LeBron *and* start Jalen Green in his slot.
    from engine.jobs.nightly import LineupReport, decision_line
    from engine.model.lineup_moves import Audition, LineupOption, LineupPlan
    lebron = Starter("1362", "LeBron James", "PHI", 40.0)
    lebron.next_tip = datetime(2026, 11, 7, 0, 30, tzinfo=timezone.utc)   # Fri 7:30 PM ET
    lebron.upcoming = []
    green = LineupOption("2309", "Jalen Green", ("SG", "SF"), 40.0, next_game=date(2026, 11, 5),
                         next_tip=datetime(2026, 11, 6, 2, 0, tzinfo=timezone.utc))  # Thu 9:00 PM ET
    start = LineupOption("1362", "LeBron James", ("SF", "PF"), 40.9)
    lebron.stand_in = Audition("SF#1", start, green, 45.0, 4.1, 40.9, keep_bench=False)
    d = Decision("LeBron James", 40.0, 40.9, "stand_in", 0.95, 0.94, 0.0, "stand_in", 0.95, 45.0)
    m = Matchup(roster_id=4, matchup_id=1, points=0.0, starters=[])
    report = LineupReport(LineupPlan(0.0, 0.0), auditions=[])  # folded into the decision
    text = render_discord(3, "ChickenJoe2107", [lebron], m, m, None, [(lebron, d)], [], lineup=report)
    assert "**START Jalen Green at SF for his Thu 9:00 PM ET game instead of LeBron James" in text
    assert "lock Jalen Green at 40.9+, otherwise put LeBron James back before Fri 7:30 PM ET" in text
    assert "before Thu 9:00 PM ET" in text  # the deadline is Green's tip, not LeBron's
    assert "95% starting Jalen Green Thu" in text
    assert "LOCK LeBron" not in text and "Start for one game" not in text
    assert "The rest of your lineup" not in text  # LeBron is covered by the decision
    terminal = decision_line(lebron, d)
    assert terminal.startswith("  Thu 9:00 PM ET") and "starting Jalen Green first 45.0" in terminal


def test_auditions_on_a_starter_with_a_decision_are_folded_into_it() -> None:
    from engine.jobs.nightly import attach_stand_ins
    from engine.model.distributions import NormalScoreDistribution
    from engine.model.lineup_moves import Audition, LineupOption
    from engine.model.matchup import Slot
    n45 = NormalScoreDistribution(45.0, 10.0)
    open_ = Starter("1", "LeBron James", "PHI", 40.0, slot=Slot("LeBron", n45, (1.0, 1.0), pending=40.0))
    fresh = Starter("2", "Kevin Durant", "HOU", 0.0, slot=Slot("Durant", n45, (1.0,)))  # nothing pending
    def audition(pid: str) -> Audition:
        return Audition("F#1", LineupOption(pid, "x", ("SF",), 40.0), LineupOption("9", "y", ("SF",), 40.0),
                        45.0, 4.0, 41.0, keep_bench=False)
    left = attach_stand_ins([open_, fresh], [audition("1"), audition("2")])
    assert open_.stand_in is not None and fresh.stand_in is None
    assert [a.starter.player_id for a in left] == ["2"]  # no decision to weigh it in: shown alone
