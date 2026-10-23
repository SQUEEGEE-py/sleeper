"""nightly.py's assembly: Sleeper schedule + counted score + box scores -> lock status + slot."""

from __future__ import annotations

from datetime import date

from engine.ingest.sleeper import Player, ScheduledGame
from engine.jobs.nightly import build_starter, schedule_frame, week_containing
from engine.model.distributions import NormalScoreDistribution

N45 = NormalScoreDistribution(45.0, 10.0, 0.9)
# PHI in week 1: Tue, Thu, Sat, Sun (Sun is a B2B second night). Week 2 starts Mon.
GAMES = [
    ScheduledGame(game_id="1", date="2026-10-20", week=1, status="complete", home="NYK", away="PHI"),
    ScheduledGame(game_id="2", date="2026-10-22", week=1, status="pre_game", home="PHI", away="CLE"),
    ScheduledGame(game_id="3", date="2026-10-24", week=1, status="pre_game", home="PHI", away="MIL"),
    ScheduledGame(game_id="4", date="2026-10-25", week=1, status="pre_game", home="PHI", away="DET"),
    ScheduledGame(game_id="5", date="2026-10-27", week=2, status="pre_game", home="BOS", away="PHI"),
]
BROWN = Player(player_id="1711", full_name="Jaylen Brown", team="PHI", birth_date="1996-10-24")


def build(games, counted, scores, distribution=N45, player=BROWN):
    status = {}
    for g in games:
        status[(g.home, g.date)] = status[(g.away, g.date)] = g.status
    return build_starter(player.player_id if player else "x", counted, player, schedule_frame(games),
                         status, scores, 1, distribution, distribution.p_play if distribution else None)


def with_status(**by_date: str) -> list[ScheduledGame]:
    return [g.model_copy(update={"status": by_date.get(g.date, g.status)}) for g in GAMES]


def test_open_decision_after_one_game() -> None:
    s = build(GAMES, 43.5, {"2026-10-20": 43.5})
    assert s.lock.state == "open" and s.slot.pending == 43.5 and s.slot.decision_open
    assert s.next_game == date(2026, 10, 22)
    # Three games left; the Sunday B2B is priced by his age that day (30 from 2026-10-24).
    assert s.slot.remaining_p_play == (0.9, 0.9, 0.9 * 0.901)


def test_an_earlier_game_shown_after_a_newer_one_is_unclear() -> None:
    # Brown on 10/22: Sleeper still showed Tue 43.5 after Thu 41 finished, and he was not
    # locked. For the user's own player: assume not locked, decide on Thursday's score.
    games = with_status(**{"2026-10-22": "complete"})
    s = build(games, 43.5, {"2026-10-20": 43.5, "2026-10-22": 41.0})
    assert s.lock.state == "unclear" and s.slot.pending == 41.0 and s.slot.decision_open
    assert "assumes you have not locked him" in s.gap
    # For the opponent, Sleeper's number is treated as banked.
    status = {}
    for g in games:
        status[(g.home, g.date)] = status[(g.away, g.date)] = g.status
    theirs = build_starter("1711", 43.5, BROWN, schedule_frame(games), status,
                           {"2026-10-20": 43.5, "2026-10-22": 41.0}, 1, N45, 0.9, mine=False)
    assert theirs.slot.locked == 43.5 and not theirs.slot.decision_open


def test_a_real_lock_is_seen_during_a_live_game() -> None:
    games = with_status(**{"2026-10-22": "in_game"})
    s = build(games, 43.5, {"2026-10-20": 43.5, "2026-10-22": 12.0})
    assert s.lock.state == "locked" and s.slot.locked == 43.5 and not s.slot.decision_open


def test_a_dnp_is_a_zero_not_a_missing_game() -> None:
    games = with_status(**{"2026-10-22": "complete"})
    s = build(games, 0.0, {"2026-10-20": 43.5})  # no stat line Thursday: he sat
    assert s.completed[-1] == (date(2026, 10, 22), 0.0) and s.slot.pending == 0.0


def test_game_in_progress_is_simulated_fresh_and_closes_the_decision() -> None:
    games = with_status(**{"2026-10-22": "in_game"})
    s = build(games, 12.0, {"2026-10-20": 43.5, "2026-10-22": 12.0})
    assert s.in_progress and s.lock.state == "open"
    assert s.slot.pending is None and len(s.slot.remaining_p_play) == 3


def test_unexplained_score_is_kept_as_counted_and_flagged() -> None:
    s = build(GAMES, 46.5, {"2026-10-20": 43.5})
    assert s.lock.state == "unexplained" and s.slot.pending == 46.5 and "lock status unclear" in s.gap


def test_no_model_is_a_gap_only_with_games_left() -> None:
    s = build(GAMES, 43.5, {"2026-10-20": 43.5}, distribution=None)
    assert s.slot is None and "no player model" in s.gap
    done = with_status(**{d: "complete" for d in ("2026-10-22", "2026-10-24", "2026-10-25")})
    s = build(done, 20.0, {"2026-10-25": 20.0}, distribution=None)
    assert s.slot is not None and s.slot.pending == 20.0 and not s.gap


def test_no_birth_date_is_flagged_not_guessed() -> None:
    s = build(GAMES, 43.5, {"2026-10-20": 43.5},
              player=Player(player_id="1711", full_name="Jaylen Brown", team="PHI"))
    assert s.slot.remaining_p_play == (0.9, 0.9, 0.9) and "no birth date" in s.gap


def test_week_containing() -> None:
    assert week_containing(GAMES, date(2026, 10, 21)) == 1  # no games today: next game's week
    assert week_containing(GAMES, date(2026, 10, 26)) == 2
