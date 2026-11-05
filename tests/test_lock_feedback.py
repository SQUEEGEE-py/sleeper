"""Lock feedback through Discord: the message format, reading 🔒 back, and what gets posted. No
network -- the Discord client runs against httpx.MockTransport."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import httpx
import pytest

from engine.jobs.lock_feedback import (
    Candidate,
    FeedbackLock,
    LockFeedback,
    format_message,
    lock_reacted,
    locks_from_messages,
    parse_tag,
    posted_keys,
    to_post,
)
from engine.jobs.lock_sheet import GameSlot
from engine.jobs.nightly import Starter, feedback_candidates, lock_slot_gaps, merge_feedback_locks
from engine.model.distributions import NormalScoreDistribution
from engine.model.locks import LockStatus
from engine.model.matchup import Decision, Slot

MON, TUE, WED = date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 4)
HOOK = "900"
ME = "555"
N45 = NormalScoreDistribution(45.0, 10.0, 0.9)
BROWN = Candidate(3, MON, "1711", "Jaylen Brown", "F#1", 59.5, "LOCK (win 74% if locked / 70% if rolled)")


def msg(mid: str, c: Candidate, reactions=None, webhook_id: str | None = HOOK, content: str | None = None) -> dict:
    m = {"id": mid, "content": format_message(c) if content is None else content,
         "timestamp": "2026-11-03T08:00:00+00:00"}
    if webhook_id:
        m["webhook_id"] = webhook_id
    if reactions is not None:
        m["reactions"] = reactions
    return m


def lock_reaction(count: int, me: bool) -> list[dict]:
    return [{"emoji": {"id": None, "name": "🔒"}, "count": count, "me": me}]


def test_message_round_trips_through_its_tag() -> None:
    text = format_message(BROWN)
    assert "**Jaylen Brown** (F) -- **59.5** on Mon 11/02" in text and "React 🔒" in text
    assert "Advisor: LOCK" in text
    assert parse_tag(text) == (3, MON, "1711", "F#1", 59.5)
    assert parse_tag("just chatting") is None
    neg = Candidate(3, MON, "7", "X", "UTIL#1", -1.5)
    assert parse_tag(format_message(neg))[4] == -1.5


def test_the_bots_own_seeded_reaction_is_not_a_lock() -> None:
    assert not lock_reacted(msg("1", BROWN, lock_reaction(1, me=True)))
    assert lock_reacted(msg("1", BROWN, lock_reaction(2, me=True)))
    assert lock_reacted(msg("1", BROWN, lock_reaction(1, me=False)))
    assert not lock_reacted(msg("1", BROWN, [{"emoji": {"name": "👍"}, "count": 3, "me": False}]))
    assert not lock_reacted(msg("1", BROWN))


def test_a_lock_is_any_lock_emoji_on_our_message_this_week() -> None:
    wilson = Candidate(3, MON, "4862", "Caleb Wilson", "UTIL#1", 51.0)
    old = Candidate(2, date(2026, 10, 28), "1711", "Jaylen Brown", "F#1", 62.0)
    fake = Candidate(3, MON, "99", "Typed By Hand", "C#1", 70.0)
    seeded = Candidate(3, MON, "8", "Bot Only", "G#1", 40.0)
    rolled = Candidate(3, MON, "9", "Not Locked", "C#2", 30.0)
    user = lock_reaction(1, me=False)
    messages = [msg("a", BROWN, user), msg("b", wilson, user), msg("c", old, user),
                msg("d", fake, user, webhook_id=None), msg("e", seeded, lock_reaction(1, me=True)),
                msg("f", rolled)]
    locks, gaps = locks_from_messages(messages, 3, HOOK)
    assert set(locks) == {"1711", "4862"} and not gaps
    assert locks["1711"] == FeedbackLock("1711", "Jaylen Brown", 59.5, MON, "F#1", "a")
    # With the user's id set, only his 🔒 counts.
    locks, _ = locks_from_messages(messages, 3, HOOK, {"a": [{"id": ME}], "b": [{"id": "777"}]}, user_id=ME)
    assert set(locks) == {"1711"}


def test_two_locks_on_one_player_keep_the_first_and_say_so() -> None:
    later = Candidate(3, WED, "1711", "Jaylen Brown", "F#1", 61.0)
    user = lock_reaction(1, me=False)
    messages = [msg("b", later, user), msg("a", BROWN, user)]  # Discord lists newest first
    locks, gaps = locks_from_messages(messages, 3, HOOK)
    assert locks["1711"].score == 59.5 and locks["1711"].game_date == MON
    assert len(gaps) == 1 and "two games" in gaps[0]


def test_unreadable_content_fails_loudly_instead_of_meaning_no_locks() -> None:
    with pytest.raises(RuntimeError, match="Message Content intent"):
        locks_from_messages([msg("a", BROWN, content="")], 3, HOOK, {}, None)


def test_posting_is_idempotent() -> None:
    posted = posted_keys([msg("a", BROWN), msg("z", BROWN, webhook_id="other")], HOOK)
    assert posted == {(3, "2026-11-02", "1711")}
    wilson = Candidate(3, MON, "4862", "Caleb Wilson", "UTIL#1", 51.0)
    assert to_post([BROWN, wilson], posted) == [wilson]


def starter(pid, name, completed, upcoming=True, in_progress=False, lock=None, played=None) -> Starter:
    s = Starter(pid, name, "PHI", completed[-1][1] if completed else 0.0)
    s.completed = list(completed)
    s.upcoming = [GameSlot(WED, "BOS", True, False)] if upcoming else []
    s.in_progress = in_progress
    s.lock = lock or LockStatus("open")
    s.played = frozenset(played if played is not None else [d.isoformat() for d, _ in completed])
    return s


def test_feedback_candidates() -> None:
    slots = ["PG#1", "F#1", "C#1", "C#2", "UTIL#1", "G#1", "SG#1"]
    final = {"2026-11-02": {"complete"}, "2026-11-03": {"complete", "in_game"}}
    mine = [
        starter("1", "Open Mon", [(MON, 50.0)]),
        starter("2", "Locked", [(MON, 59.5)], lock=LockStatus("locked", MON, 59.5)),
        starter("3", "Week over", [(MON, 40.0)], upcoming=False),
        starter("4", "Tue not final", [(MON, 30.0), (TUE, 44.0)]),
        starter("5", "Sat out Mon", [(MON, 0.0)], played=[]),
        starter("6", "Playing now", [(MON, 33.0)], in_progress=True),
        starter("7", "No game yet", []),
        starter("8", "Next game tipped", [(MON, 52.0)]),
    ]
    mine[-1].next_tip = datetime(2026, 11, 4, 23, 0, tzinfo=timezone.utc)
    mine[0].next_tip = datetime(2026, 11, 5, 0, 30, tzinfo=timezone.utc)
    d = Decision("Open Mon", 50.0, 48.0, "lock", 0.74, 0.70, 0.004, "lock")
    got = feedback_candidates(3, mine, slots + ["G#2"], final, [(mine[0], d)],
                              now=datetime(2026, 11, 4, 23, 5, tzinfo=timezone.utc))
    assert [(c.player_id, c.slot, c.score, c.game_date) for c in got] == [("1", "PG#1", 50.0, MON)]
    assert got[0].advice.startswith("LOCK (win 74% if locked / 70% if rolled)")


def test_lock_slot_gaps_and_merge() -> None:
    mine = [starter("1711", "Jaylen Brown", [(MON, 59.5)]), starter("4862", "Caleb Wilson", [(MON, 51.0)])]
    feedback = {"1711": FeedbackLock("1711", "Jaylen Brown", 59.5, MON, "F#1", "a"),
                "4862": FeedbackLock("4862", "Caleb Wilson", 51.0, MON, "C#2", "b")}
    assert lock_slot_gaps(mine, ["F#1", "C#1"], feedback) == []  # C#1 vs C#2 is the same position
    gaps = lock_slot_gaps(mine, ["UTIL#1", "C#1"], feedback)
    assert len(gaps) == 1 and "Sleeper now has him at UTIL" in gaps[0]
    gaps = lock_slot_gaps(mine[1:], ["C#1"], feedback)
    assert "not in Sleeper's starting lineup" in gaps[0]

    merge_gaps: list[str] = []
    merged = merge_feedback_locks({"1711": 58.0, "1": 30.0}, feedback, merge_gaps)
    assert merged == {"1711": 58.0, "1": 30.0, "4862": 51.0}
    assert len(merge_gaps) == 1 and "using the file" in merge_gaps[0]


def test_client_posts_without_reacting_and_clears_its_old_reactions() -> None:
    calls: list[tuple[str, str]] = []
    posted_body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode()  # still percent-encoded, as sent
        calls.append((request.method, path))
        if request.method == "GET" and path == "/api/webhooks/900/tok":
            assert "authorization" not in request.headers  # the webhook needs no bot token
            return httpx.Response(200, json={"id": HOOK, "channel_id": "42", "name": "locks"})
        if request.method == "POST" and path.startswith("/api/webhooks/900/tok"):
            assert request.url.params["wait"] == "true"
            posted_body.update(json.loads(request.content))
            return httpx.Response(200, json={"id": "m1"})
        assert request.headers["authorization"] == "Bot sekrit"
        if request.method == "DELETE":
            return httpx.Response(204)
        if path.startswith("/api/v10/channels/42/messages?"):
            return httpx.Response(200, json=[
                msg("m2", Candidate(3, TUE, "4862", "Caleb Wilson", "UTIL#1", 51.0), lock_reaction(1, me=False)),
                msg("m1", BROWN, lock_reaction(1, me=True)),  # only the bot's seed: not a lock
                msg("m0", BROWN, lock_reaction(1, me=False)) | {"timestamp": "2026-10-20T08:00:00+00:00"},
            ])
        return httpx.Response(404)

    client = LockFeedback("https://discord.com/api/webhooks/900/tok", "sekrit",
                          http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.post(BROWN) == "m1"
    assert parse_tag(posted_body["content"]) == (3, MON, "1711", "F#1", 59.5)
    assert posted_body["allowed_mentions"] == {"parse": []}
    assert not any(method == "PUT" for method, _ in calls)  # posting never reacts

    locks, gaps, posted = client.read_week(3, datetime(2026, 11, 2, 5, tzinfo=timezone.utc))
    assert set(locks) == {"4862"} and not gaps
    assert posted == {(3, "2026-11-02", "1711"), (3, "2026-11-03", "4862")}
    # The bot's own 🔒 on m1 is taken back; nothing before the week is touched.
    deletes = [path for method, path in calls if method == "DELETE"]
    assert deletes == ["/api/v10/channels/42/messages/m1/reactions/%F0%9F%94%92/@me"]
    assert not any("m0" in path for _, path in calls)
