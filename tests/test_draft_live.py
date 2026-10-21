"""Tests for engine/jobs/draft_live.py's pure parts: state building and rendering. No network."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from engine.ingest.sleeper import Draft, DraftPick
from engine.jobs.draft_live import FetchStatus, build_view, render_html, resolve_slot

SLOTS = ["PG#1", "SG#1", "G#1", "SF#1", "PF#1", "F#1", "C#1", "C#2", "UTIL#1"]
ME = "u-me"


def pool() -> pd.DataFrame:
    rows = [
        ("s1", "Big One", ("C",), 20.0, 1.0),
        ("s2", "Guard One", ("PG",), 18.0, 2.0),
        ("s3", "Wing One", ("SF",), 15.0, 3.0),
        ("s4", "Big Two", ("C",), 12.0, 4.0),
        ("s5", "Guard Two", ("SG",), 10.0, 5.0),
    ] + [(f"f{i}", f"Filler {i}", ("PF",), 1.0, 10.0 + i) for i in range(20)]
    frame = pd.DataFrame(
        [
            {"sleeper_id": sid, "name": name, "team": "DEN", "positions": positions,
             "vor": vor, "mean": 40.0, "sd": 12.0, "p_play": 0.9, "flags": ""}
            for sid, name, positions, vor, _ in rows
        ]
    )
    return frame.set_index("sleeper_id", drop=False)


ADP = {"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.0, "s5": 5.0, **{f"f{i}": 10.0 + i for i in range(20)}}


def pick(pick_no: int, player_id: str, slot: int, by: str = "someone", name: str = "") -> DraftPick:
    first, _, last = (name or player_id).partition(" ")
    return DraftPick.model_validate(
        {"pick_no": pick_no, "round": 1, "roster_id": slot, "player_id": player_id,
         "draft_id": "d", "draft_slot": slot, "picked_by": by,
         "metadata": {"first_name": first, "last_name": last, "position": "SF"}}
    )


def draft(order: dict[str, int] | None) -> Draft:
    return Draft.model_validate({"draft_id": "d", "status": "drafting", "draft_order": order})


def view_for(picks: list[DraftPick], order: dict[str, int] | None = None, override: int | None = None):
    return build_view(
        pool(), ADP, draft(order), picks, ME, SLOTS, teams=4, rounds=4,
        manager_names={}, slot_override=override, n_sims=200, rng=np.random.default_rng(0),
    )


def test_slot_comes_from_draft_order_then_own_picks_then_nothing() -> None:
    assert resolve_slot(draft({ME: 3}), [], ME, None) == 3
    assert resolve_slot(draft(None), [pick(2, "s1", 2, by=ME)], ME, None) == 2
    assert resolve_slot(draft(None), [], ME, None) is None
    assert resolve_slot(draft({ME: 3}), [], ME, 1) == 1  # explicit override wins


def test_unknown_slot_still_ranks_by_value() -> None:
    view = view_for([])
    assert view.clock is None
    assert [rec.name for rec in view.recommendations[:3]] == ["Big One", "Guard One", "Wing One"]
    assert all(np.isnan(rec.p_survive) for rec in view.recommendations)


def test_taken_players_leave_the_pool_and_mine_join_my_roster() -> None:
    picks = [pick(1, "s1", 1), pick(2, "s2", 2, by=ME), pick(3, "s3", 3)]
    view = view_for(picks, order={ME: 2})
    names = [rec.name for rec in view.recommendations]
    assert "Big One" not in names and "Guard One" not in names and "Wing One" not in names
    assert [(slot, name) for slot, name, _, _ in view.roster] == [("PG", "Guard One")]
    assert "PG" not in view.open_slots


def test_need_changes_the_recommendation() -> None:
    # With one centre already rostered, the second still fills C; a PG-holder is not offered PG.
    picks = [pick(1, "s2", 1, by=ME)]
    view = view_for(picks, order={ME: 1})
    top = view.recommendations[0]
    assert top.name == "Big One" and top.fills == "C"


def test_an_unvalued_pick_is_shown_not_dropped() -> None:
    picks = [pick(1, "zz", 1, by=ME, name="Mystery Rookie")]
    view = view_for(picks, order={ME: 1})
    assert view.unvalued_taken == ["Mystery Rookie"]
    assert view.roster[0][1] == "Mystery Rookie (unvalued)"


def test_page_shows_failure_and_stays_self_contained() -> None:
    view = view_for([pick(1, "s1", 1)], order={ME: 2})
    page = render_html(
        view, FetchStatus(last_ok=1000.0, last_error="ConnectError: down", failures=4), 3, 1047.0
    )
    assert "Sleeper fetch failing (4x)" in page and "47s old" in page
    assert "board.html" in page  # tells the user where to go
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", page)


def test_page_marks_my_turn() -> None:
    view = view_for([pick(1, "s1", 1)], order={ME: 2})
    page = render_html(view, FetchStatus(last_ok=1.0), 3, 2.0)
    assert "YOUR PICK" in page and "myturn" in page
    assert view.recommendations[0].why.startswith("starts at")


@pytest.mark.parametrize("picks_made", [0, 5, 15])
def test_view_never_recommends_a_taken_player(picks_made: int) -> None:
    ids = list(pool().index)
    picks = [pick(n + 1, ids[n], (n % 4) + 1) for n in range(picks_made)]
    view = view_for(picks, order={ME: 4})
    taken = {p.player_id for p in picks}
    assert not taken & {rec.sleeper_id for rec in view.recommendations}


class FakeClient:
    """Serves one more pick per poll; the last poll reports the draft complete."""

    def __init__(self, picks: list[DraftPick]) -> None:
        self.picks, self.polls = picks, 0

    def get_draft_live(self, draft_id: str) -> Draft:
        self.polls += 1
        status = "complete" if self.polls > len(self.picks) else "drafting"
        return Draft.model_validate({"draft_id": draft_id, "status": status, "draft_order": {ME: 1}})

    def get_draft_picks_live(self, draft_id: str) -> list[DraftPick]:
        return self.picks[: self.polls - 1]


def test_an_exception_inside_build_view_does_not_stop_the_advisor(tmp_path, monkeypatch) -> None:
    # The 2026-10-18 incident: only the fetch was guarded, so anything raised after it killed
    # the advisor mid-draft. Pass 2 raises inside build_view; passes 3-4 must still run.
    import engine.jobs.draft_live as live
    from engine.jobs.runtime import run_loop

    real_build, calls = live.build_view, []

    def flaky_build(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ZeroDivisionError("boom in build_view")
        return real_build(*args, **kwargs)

    monkeypatch.setattr(live, "build_view", flaky_build)
    client = FakeClient([pick(1, "s1", 1, by=ME), pick(2, "s2", 2), pick(3, "s3", 3)])
    page = tmp_path / "live.html"
    poller = live.LivePoller(client, {"teams": 4, "draft_settings_live": {"rounds": 4}}, pool(),
                             ADP, ME, SLOTS, "d", page, interval=0, n_sims=50)
    pages: list[str] = []

    def failed(error, path):
        poller.failed(error, path)
        pages.append(page.read_text(encoding="utf-8"))

    run_loop("draft_live", poller.poll, interval=0, should_stop=poller.done, on_error=failed,
             sleep=lambda _: None, log_dir=tmp_path / "logs")

    assert client.polls == 4 and poller.done()
    assert len(pages) == 1 and "boom in build_view" in pages[0]  # shown on the page, not lost
    assert "ZeroDivisionError" in (tmp_path / "logs" / "draft_live.log").read_text(encoding="utf-8")
    final = page.read_text(encoding="utf-8")
    assert "boom in build_view" not in final  # a good pass clears the banner
    assert all(sid not in [r.sleeper_id for r in poller.view.recommendations] for sid in ("s1", "s2", "s3"))
