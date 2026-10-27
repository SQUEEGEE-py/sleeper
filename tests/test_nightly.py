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
    # For the opponent, Sleeper's number means nothing; he is assumed to lock optimally.
    # 43.5 and 41 are both under the mean-45 thresholds with games to come: still waiting.
    status = {}
    for g in games:
        status[(g.home, g.date)] = status[(g.away, g.date)] = g.status
    theirs = build_starter("1711", 43.5, BROWN, schedule_frame(games), status,
                           {"2026-10-20": 43.5, "2026-10-22": 41.0}, 1, N45, 0.9, mine=False)
    assert theirs.slot.locked is None and theirs.slot.pending == 41.0
    # A big first game is locked by the policy, whatever Sleeper shows.
    big = build_starter("1711", 41.0, BROWN, schedule_frame(games), status,
                        {"2026-10-20": 62.0, "2026-10-22": 41.0}, 1, N45, 0.9, mine=False)
    assert big.slot.locked == 62.0 and big.lock.note == "assumed: locks optimally"


def test_a_reported_lock_wins_over_whatever_sleeper_shows() -> None:
    # Kawhi 10/23: locked at 48.5, Sleeper showing his next game's 34.
    games = with_status(**{"2026-10-22": "complete"})
    status = {}
    for g in games:
        status[(g.home, g.date)] = status[(g.away, g.date)] = g.status
    s = build_starter("1711", 34.0, BROWN, schedule_frame(games), status,
                      {"2026-10-20": 48.5, "2026-10-22": 34.0}, 1, N45, 0.9, reported_lock=48.5)
    assert s.lock.state == "locked" and s.slot.locked == 48.5 and not s.slot.decision_open
    assert "reported by you" in s.lock.note and not s.gap
    # Only the user's own players: the opponent's locks are never reported.
    theirs = build_starter("1711", 34.0, BROWN, schedule_frame(games), status,
                           {"2026-10-20": 48.5, "2026-10-22": 34.0}, 1, N45, 0.9, mine=False, reported_lock=48.5)
    assert theirs.slot.locked is None


def test_reported_locks_count_only_for_their_week(tmp_path) -> None:
    import json
    from engine.jobs.nightly import reported_locks
    path = tmp_path / "locks.json"
    path.write_text(json.dumps({"week": 1, "locked": {"1433": {"name": "Durant", "score": 51.0}}}), encoding="utf-8")
    assert reported_locks(path, 1) == {"1433": 51.0}
    assert reported_locks(path, 2) == {}
    assert reported_locks(tmp_path / "missing.json", 1) == {}


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


def test_injury_swap_picks_the_best_healthy_eligible_bench_player() -> None:
    from engine.jobs.nightly import Starter, injury_swaps
    from engine.model.matchup import Slot
    from engine.model.locks import LockStatus
    hurt = Starter("k", "Kawhi Leonard", "TOR", 0.0)
    bench = [
        (Starter("b", "Saddiq Bey", "NOP", 0.0, slot=Slot("Bey", N45, (0.9, 0.9, 0.9))), ("PF", "SF")),
        (Starter("w", "Caleb Wilson", "CHI", 0.0, slot=Slot("Wilson", N45, (0.9, 0.9))), ("PF", "SF")),
        (Starter("g", "Guard Only", "LAC", 0.0, slot=Slot("G", N45, (1.0, 1.0, 1.0, 1.0))), ("PG",)),
        (Starter("x", "Also Hurt", "LAL", 0.0, slot=Slot("X", N45, (0.9, 0.9, 0.9))), ("SF",)),
    ]
    players = {"k": Player(player_id="k", injury_status="Out"), "x": Player(player_id="x", injury_status="Out"),
               "b": Player(player_id="b"), "w": Player(player_id="w"), "g": Player(player_id="g")}
    (swap,) = injury_swaps([hurt], ["F#1"], bench, players)
    assert [name for name, _, _ in swap.options] == ["Saddiq Bey", "Caleb Wilson"]  # 3 games beat 2
    assert "start Saddiq Bey at F" in swap.describe()
    # A locked starter's score is banked: nothing to swap.
    hurt.lock = LockStatus("locked", None, 48.5)
    assert injury_swaps([hurt], ["F#1"], bench, players) == []
