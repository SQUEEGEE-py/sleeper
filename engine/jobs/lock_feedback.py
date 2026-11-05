"""Lock feedback through Discord: one message per open starter after his day's games are final,
and the user reacts 🔒 to each one he locked in Sleeper.

Sleeper's public API never shows a lock (``engine/model/locks.py``), so the only source is the
user. Before this he edited ``state/locks.json`` and pushed; now:

1. After every game of a day is ``complete`` in Sleeper's schedule, the next ``nightly --post``
   run posts to the ``#daily-locks`` channel one message per starter whose latest finished game
   was that day, who still has games left this week (a final game counts by itself) and who is
   not already locked: his name, slot, score and the advisor's call. Each message ends in a
   small machine-readable tag (``lock w3 2026-11-02 1711 F#1 59.5``).
2. The user reacts 🔒 on the ones he locked. The bot never reacts (user's call, 2026-11-05: a
   🔒 on a message means he locked that player, nothing else).
3. Every later run reads this week's messages back, and a 🔒 on one is a lock at the tagged
   score, game and slot. ``state/locks.json`` still works and is merged in.

The channel is the state: nothing is stored in the repo, and posting is idempotent (a message
already in the channel for that week, day and player is not posted again), so a late,
repeated or push-triggered run is harmless.

Why a bot as well as a webhook (decided 2026-11-05): a webhook can post, but cannot list a
channel, so it could only read back messages whose ids were stored somewhere between runs. A
bot token can list the channel and read its reactions. Posting stays on the webhook, because a bot must open a gateway connection once before
Discord lets it post, and Actions runs are plain HTTP.

Secrets (``.env`` locally, repository secrets in Actions), all optional -- without the first
two the feature is off and the report says so:

* ``DISCORD_LOCKS_WEBHOOK_URL`` -- a webhook in ``#daily-locks``. Its channel id comes from the
  webhook itself.
* ``DISCORD_BOT_TOKEN`` -- a bot in the same server with View Channel and Read Message History
  there (plus Add Reactions, only to clear the 🔒 it seeded on the first day), and the **Message Content** intent switched on (without it Discord
  returns webhook messages with empty content; that is detected and fails loudly, never read
  as "no locks").
* ``DISCORD_LOCKS_USER_ID`` -- the user's Discord user id. When set, only his 🔒 counts. Unset
  (the user's channel is private to him), any 🔒 but the bot's own counts.

    uv run python -m engine.jobs.lock_feedback          # check the setup; shows recent 🔒 reactions
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

from engine.ingest.http_retry import request_with_retry
from engine.model.lineup import slot_position

LOCKS_WEBHOOK_ENV = "DISCORD_LOCKS_WEBHOOK_URL"
BOT_TOKEN_ENV = "DISCORD_BOT_TOKEN"
LOCKS_USER_ENV = "DISCORD_LOCKS_USER_ID"
LOCK_EMOJI = "🔒"
API = "https://discord.com/api/v10"
USER_AGENT = "DiscordBot (https://github.com/SQUEEGEE-py/sleeper, 1.0)"
PAGE = 100
MAX_PAGES = 10  # a week is ~30-60 messages; 1000 is a runaway guard, not a limit in practice

TAG = re.compile(r"lock w(?P<week>\d+) (?P<day>\d{4}-\d{2}-\d{2}) (?P<pid>\S+) (?P<slot>\S+) "
                 r"(?P<score>-?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class Candidate:
    """A starter whose lock decision is open after his day's games went final."""

    week: int
    game_date: date
    player_id: str
    name: str
    slot: str           # "F#1": the lineup slot label, as lineup.starting_slots names it
    score: float
    advice: str = ""    # the advisor's call, e.g. "LOCK (win 74% if locked / 70% if rolled)"

    @property
    def key(self) -> tuple[int, str, str]:
        return (self.week, self.game_date.isoformat(), self.player_id)


@dataclass(frozen=True)
class FeedbackLock:
    """A lock the user confirmed with 🔒."""

    player_id: str
    name: str
    score: float
    game_date: date
    slot: str
    message_id: str


def tag(c: Candidate) -> str:
    return f"lock w{c.week} {c.game_date.isoformat()} {c.player_id} {c.slot} {c.score:g}"


def format_message(c: Candidate) -> str:
    lines = [f"**{c.name}** ({slot_position(c.slot)}) -- **{c.score:g}** on {c.game_date:%a %m/%d}"]
    if c.advice:
        lines.append(f"Advisor: {c.advice}")
    lines.append(f"React {LOCK_EMOJI} if you locked him.")
    lines.append(f"-# {tag(c)}")  # Discord subtext: small and grey, read back by parse_tag
    return "\n".join(lines)


def parse_tag(content: str) -> tuple[int, date, str, str, float] | None:
    match = TAG.search(content or "")
    if match is None:
        return None
    return (int(match["week"]), date.fromisoformat(match["day"]), match["pid"], match["slot"],
            float(match["score"]))


def _name_from(content: str) -> str:
    match = re.match(r"\*\*(.+?)\*\*", content or "")
    return match.group(1) if match else ""


def lock_reacted(message: dict) -> bool:
    """Is there a 🔒 on it? The bot's own (``me``; it seeded them on 2026-11-05 only, and
    ``read_week`` removes them) never counts."""
    for reaction in message.get("reactions") or []:
        if (reaction.get("emoji") or {}).get("name") == LOCK_EMOJI:
            return int(reaction.get("count") or 0) - (1 if reaction.get("me") else 0) > 0
    return False


def posted_keys(messages: Iterable[dict], webhook_id: str) -> set[tuple[int, str, str]]:
    keys = set()
    for m in messages:
        if m.get("webhook_id") != webhook_id:
            continue
        parsed = parse_tag(m.get("content", ""))
        if parsed:
            keys.add((parsed[0], parsed[1].isoformat(), parsed[2]))
    return keys


def locks_from_messages(messages: Iterable[dict], week: int, webhook_id: str,
                        reactors: dict[str, list[dict]] | None = None, user_id: str | None = None
                        ) -> tuple[dict[str, FeedbackLock], list[str]]:
    """This week's locks from the channel: any message of ours with a 🔒 on it. With
    ``user_id`` set, the 🔒 must be that user's -- ``reactors`` maps message id -> the users who
    reacted (fetched only then, only where ``lock_reacted``). A message from our webhook with no content means the
    bot cannot read content -- that raises, because reading it as "no locks" would be a
    silently invented answer."""
    locks: dict[str, FeedbackLock] = {}
    gaps: list[str] = []
    for m in messages:
        if m.get("webhook_id") != webhook_id:
            continue  # only our own posts: a typed message cannot fake a lock
        content = m.get("content", "")
        if not content and not m.get("embeds"):
            raise RuntimeError("Discord returned a #daily-locks message with no content: switch on the "
                               "bot's Message Content intent (Developer Portal -> Bot)")
        parsed = parse_tag(content)
        if parsed is None or parsed[0] != week:
            continue
        if not lock_reacted(m):
            continue
        if user_id and not any(u.get("id") == user_id for u in (reactors or {}).get(m["id"], [])):
            continue
        _, day, pid, slot, score = parsed
        lock = FeedbackLock(pid, _name_from(content) or pid, score, day, slot, m["id"])
        if pid in locks and locks[pid].game_date != day:
            first, second = sorted([locks[pid], lock], key=lambda x: x.game_date)
            gaps.append(f"{first.name}: {LOCK_EMOJI} on two games ({first.game_date:%a} {first.score:g}, "
                        f"{second.game_date:%a} {second.score:g}); a player locks once -- using "
                        f"{first.game_date:%a}'s. Remove the wrong reaction.")
            lock = first
        locks[pid] = lock
    return locks, gaps


class LockFeedback:
    """The Discord side: post through the webhook, read with the bot."""

    def __init__(self, webhook_url: str, bot_token: str, user_id: str | None = None,
                 http: httpx.Client | None = None) -> None:
        self.webhook_url = webhook_url.split("?")[0]
        self.user_id = user_id or None
        self._http = http or httpx.Client(timeout=15.0)
        self._bot = {"Authorization": f"Bot {bot_token}", "User-Agent": USER_AGENT}
        self._webhook: dict | None = None

    @classmethod
    def from_env(cls) -> "LockFeedback | None":
        url, token = os.environ.get(LOCKS_WEBHOOK_ENV), os.environ.get(BOT_TOKEN_ENV)
        if not url or not token:
            return None
        return cls(url, token, os.environ.get(LOCKS_USER_ENV))

    def close(self) -> None:
        self._http.close()

    def _get(self, url: str, **params: Any) -> Any:
        return request_with_retry(lambda: self._http.get(url, params=params or None, headers=self._bot)).json()

    @property
    def webhook(self) -> dict:
        """The webhook object: its own id (to recognise its posts) and its channel."""
        if self._webhook is None:
            self._webhook = request_with_retry(
                lambda: self._http.get(self.webhook_url, headers={"User-Agent": USER_AGENT})).json()
        return self._webhook

    @property
    def channel_id(self) -> str:
        return str(self.webhook["channel_id"])

    @property
    def webhook_id(self) -> str:
        return str(self.webhook["id"])

    def messages_since(self, since: datetime) -> list[dict]:
        """The channel's messages created at or after ``since``, newest first."""
        out: list[dict] = []
        before = None
        for _ in range(MAX_PAGES):
            params: dict[str, Any] = {"limit": PAGE}
            if before:
                params["before"] = before
            page = self._get(f"{API}/channels/{self.channel_id}/messages", **params)
            if not page:
                break
            for m in page:
                if datetime.fromisoformat(m["timestamp"]) < since:
                    return out
                out.append(m)
            before = page[-1]["id"]
            if len(page) < PAGE:
                break
        return out

    def _reactors(self, message_id: str) -> list[dict]:
        return self._get(f"{API}/channels/{self.channel_id}/messages/{message_id}/reactions/"
                         f"{quote(LOCK_EMOJI)}", limit=100)

    def read_week(self, week: int, since: datetime
                  ) -> tuple[dict[str, FeedbackLock], list[str], set[tuple[int, str, str]]]:
        """This week's locks, any gaps, and the (week, day, player) keys already posted."""
        messages = self.messages_since(since)
        ours = [m for m in messages if m.get("webhook_id") == self.webhook_id]
        self.remove_own_reactions(ours)
        reactors = ({m["id"]: self._reactors(m["id"]) for m in ours if lock_reacted(m)}
                    if self.user_id else None)
        locks, gaps = locks_from_messages(messages, week, self.webhook_id, reactors, self.user_id)
        return locks, gaps, posted_keys(messages, self.webhook_id)

    def remove_own_reactions(self, messages: Iterable[dict]) -> int:
        """Take back any 🔒 the bot added itself (it seeded them on 2026-11-05), so the only
        reactions in the channel are the user's. Best effort: a failure costs nothing, since
        the bot's own reaction never counts as a lock anyway."""
        removed = 0
        for m in messages:
            if not any((r.get("emoji") or {}).get("name") == LOCK_EMOJI and r.get("me")
                       for r in m.get("reactions") or []):
                continue
            try:
                request_with_retry(lambda m=m: self._http.delete(
                    f"{API}/channels/{self.channel_id}/messages/{m['id']}/reactions/{quote(LOCK_EMOJI)}/@me",
                    headers=self._bot))
                removed += 1
            except httpx.HTTPError as error:
                print(f"lock feedback: could not remove the bot's own {LOCK_EMOJI} ({error})", file=sys.stderr)
        return removed

    def post(self, c: Candidate) -> str:
        """Post one candidate, with no reaction: the user adds 🔒 himself. Returns the id."""
        body = {"content": format_message(c), "allowed_mentions": {"parse": []}}
        message = request_with_retry(lambda: self._http.post(
            self.webhook_url, params={"wait": "true"}, json=body, headers={"User-Agent": USER_AGENT})).json()
        return str(message["id"])


def to_post(candidates: Iterable[Candidate], posted: set[tuple[int, str, str]]) -> list[Candidate]:
    return [c for c in candidates if c.key not in posted]


def check(feedback: LockFeedback) -> int:
    """Rule 6 for Discord: prove each piece works before relying on it."""
    me = feedback._get(f"{API}/users/@me")
    print(f"bot: {me.get('username')} (id {me.get('id')})")
    print(f"webhook: {feedback.webhook.get('name')} -> channel {feedback.channel_id}")
    recent = feedback._get(f"{API}/channels/{feedback.channel_id}/messages", limit=20)
    print(f"read {len(recent)} recent messages from the channel")
    ours = [m for m in recent if m.get("webhook_id") == feedback.webhook_id]
    if ours and not any(m.get("content") for m in ours):
        print("FAIL: our messages come back empty -- switch on the bot's Message Content intent")
        return 1
    tagged = [m for m in ours if parse_tag(m.get("content", ""))]
    for m in tagged:
        if lock_reacted(m):
            users = [u.get("username") for u in feedback._reactors(m["id"])]
            print(f"{LOCK_EMOJI} on {_name_from(m['content'])}: {', '.join(users)}")
    print(f"{len(tagged)} lock messages in the last 20; only-this-user filter: {feedback.user_id or 'off'}")
    return 0


TEST_WEEK = 0  # no fantasy week is 0, so a test message can never count as a real lock


def post_test(feedback: LockFeedback) -> int:
    """Post a week-0 test message through the webhook and read it back with the bot."""
    c = Candidate(TEST_WEEK, date.today(), "test", "Lock feedback test", "UTIL#1", 0.0,
                  "this is a setup test -- react to it to check your reactions are read; it never counts")
    message_id = feedback.post(c)
    print(f"posted test message {message_id}")
    back = feedback._get(f"{API}/channels/{feedback.channel_id}/messages/{message_id}")
    if not back.get("content"):
        print("FAIL: the bot reads the test message as empty -- switch on its Message Content intent")
        return 1
    if parse_tag(back["content"]) is None:
        print("FAIL: the tag did not survive the round trip")
        return 1
    print("OK: content readable, tag parsed")
    return 0


def main() -> int:
    from engine.jobs.runtime import utf8_output

    utf8_output()
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--post-test", action="store_true",
                        help="also post a week-0 test message and read it back")
    args = parser.parse_args()
    feedback = LockFeedback.from_env()
    if feedback is None:
        print(f"lock feedback is off: set {LOCKS_WEBHOOK_ENV} and {BOT_TOKEN_ENV}")
        return 1
    try:
        result = check(feedback)
        if result == 0 and args.post_test:
            result = post_test(feedback)
        return result
    finally:
        feedback.close()


if __name__ == "__main__":
    sys.exit(main())
