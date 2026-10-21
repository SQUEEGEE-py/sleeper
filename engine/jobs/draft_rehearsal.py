"""Dress rehearsal for the live draft advisor (TASKS M5 acceptance), with no live draft needed.

    uv run python -m engine.jobs.draft_rehearsal [--slot 5] [--drafts 3]   # synthetic drafts
    uv run python -m engine.jobs.draft_rehearsal --calibrate                # 2025 replay

**Synthetic drafts** run the exact code path ``draft_live`` uses (``build_view``) for every
one of the 104 picks. Opponents draft from Sleeper's current ADP with the calibrated noise;
the user takes the advisor's top recommendation. It reports the slowest pick against the
10-second budget, the final roster, and whether the lineup filled. It also renders the page
once with a simulated fetch failure, so the red banner is seen before draft night.

**Calibration** replays this league's completed 2025 draft against Sleeper's 2025 ADP: at
every pick, for every player in play, predict whether he lasts to that team's next pick, then
compare with what happened. This is the evidence behind ``draft_sim.DEFAULT_SIGMA``.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from engine.ingest.sleeper import Draft, DraftPick, SleeperClient
from engine.jobs.draft_live import (
    OUT_DIR,
    FetchStatus,
    build_view,
    load_config,
    load_pool,
    render_html,
    render_terminal,
    write_atomic,
)
from engine.jobs.runtime import utf8_output
from engine.model.draft_sim import (
    DEFAULT_SIGMA,
    UNRANKED_ADP,
    simulate_ranks,
    slot_for_pick,
    snake_pick_numbers,
    survival,
)
from engine.model.lineup import starting_slots

PICK_BUDGET_SECONDS = 10.0
REHEARSAL_DRAFT_ID = "rehearsal"
PREVIOUS_DRAFT_ID = "1275683545217503232"  # the league's completed 2025 draft
PREVIOUS_SEASON = "2025"


def make_pick(pick_no: int, teams: int, player_id: str, name: str, user_id: str) -> DraftPick:
    first, _, last = name.partition(" ")
    return DraftPick.model_validate(
        {
            "pick_no": pick_no,
            "round": (pick_no - 1) // teams + 1,
            "roster_id": slot_for_pick(pick_no, teams),
            "player_id": player_id,
            "draft_id": REHEARSAL_DRAFT_ID,
            "draft_slot": slot_for_pick(pick_no, teams),
            "picked_by": user_id,
            "metadata": {"first_name": first, "last_name": last},
        }
    )


def synthetic_draft(slot: int, seed: int, sims: int, show: bool) -> tuple[float, bool]:
    config = load_config()
    teams, rounds = config["teams"], config["draft_settings_live"]["rounds"]
    slots = starting_slots(config["roster_positions"])
    pool = load_pool()
    with SleeperClient() as client:
        adp = client.get_adp(config["season"].split("-")[0])

    rng = np.random.default_rng(seed)
    me = "me"
    order = {f"user{s}": s for s in range(1, teams + 1)}
    order[me] = order.pop(f"user{slot}")
    draft = Draft.model_validate(
        {"draft_id": REHEARSAL_DRAFT_ID, "status": "drafting", "draft_order": order,
         "settings": {"teams": teams, "rounds": rounds}}
    )
    names = {s: ("you" if s == slot else f"opp{s}") for s in range(1, teams + 1)}

    picks: list[DraftPick] = []
    slowest = 0.0
    view = None
    for pick_no in range(1, teams * rounds + 1):
        started = time.perf_counter()
        view = build_view(pool, adp, draft, picks, me, slots, teams, rounds, names,
                          n_sims=sims, rng=rng)
        elapsed = time.perf_counter() - started
        slowest = max(slowest, elapsed)

        on_clock = slot_for_pick(pick_no, teams)
        if on_clock == slot:
            choice = view.recommendations[0]
            picks.append(make_pick(pick_no, teams, choice.sleeper_id, choice.name, me))
            if show:
                print(f"#{pick_no:3d} you   {choice.name:26} {choice.fills:5} "
                      f"gain {choice.gain:5.1f}  ({elapsed:.1f}s)")
        else:
            taken = {pick.player_id for pick in picks}
            available = [sid for sid in pool.index if sid not in taken]
            noisy = [adp.get(sid, UNRANKED_ADP) * np.exp(DEFAULT_SIGMA * rng.standard_normal())
                     for sid in available]
            sid = available[int(np.argmin(noisy))]
            picks.append(make_pick(pick_no, teams, sid, pool.loc[sid, "name"], f"user{on_clock}"))

    final = build_view(pool, adp, draft, picks, me, slots, teams, rounds, names, n_sims=sims, rng=rng)
    filled = not final.open_slots
    if show:
        print(f"\nfinal roster (slot {slot}):")
        for slot_name, name, positions, vor in final.roster:
            print(f"  {slot_name:5} {name:26} {positions:10} VOR {vor:6.1f}")
        starters = sum(max(vor, 0) for slot_name, _, _, vor in final.roster if slot_name != "BN")
        print(f"  starting lineup VOR {starters:.1f}; open slots: {final.open_slots or 'none'}")

        # The failure banner, rendered once on purpose so it has been seen before draft night.
        failing = FetchStatus(last_ok=time.time() - 47, last_error="ConnectError: [rehearsal] network pulled", failures=9)
        write_atomic(OUT_DIR / "rehearsal_failure.html", render_html(view, failing, 3, time.time()))
        print("\n" + render_terminal(view, failing).splitlines()[1])
        print(f"wrote {OUT_DIR / 'rehearsal_failure.html'} (what a dead network looks like)")
    return slowest, filled


def calibrate(sigmas: list[float], sims: int) -> None:
    with SleeperClient() as client:
        adp = client.get_adp(PREVIOUS_SEASON)
        picks = sorted(client.get_draft_picks(PREVIOUS_DRAFT_ID), key=lambda pick: pick.pick_no)
    teams, rounds = 8, len(picks) // 8
    pick_at = {pick.player_id: pick.pick_no for pick in picks}
    pool = sorted(set(adp) | set(pick_at))
    print(f"2025 draft: {len(picks)} picks; ADP for {sum(p in adp for p in pick_at)} of the drafted")

    for sigma in sigmas:
        rng = np.random.default_rng(0)
        predicted, actual = [], []
        for made in range(len(picks)):
            slot = slot_for_pick(made + 1, teams)
            later = [p for p in snake_pick_numbers(slot, teams, rounds) if p > made + 1]
            if not later:
                continue
            # The team on the clock takes one, then the others pick until its next turn.
            picks_before = later[0] - made - 1
            in_play = sorted(
                (p for p in pool if pick_at.get(p, 10**6) > made),
                key=lambda p: adp.get(p, UNRANKED_ADP),
            )[:40]
            ranks = simulate_ranks(np.array([adp.get(p, UNRANKED_ADP) for p in in_play]), sigma, sims, rng)
            for player, p in zip(in_play, survival(ranks, picks_before)):
                predicted.append(p)
                actual.append(pick_at.get(player, 10**6) >= later[0])
        predicted, actual = np.array(predicted), np.array(actual, dtype=float)
        bins = np.digitize(predicted, [0.1, 0.3, 0.5, 0.7, 0.9])
        table = "  ".join(
            f"{predicted[bins == b].mean():.2f}->{actual[bins == b].mean():.2f}"
            for b in range(6) if (bins == b).any()
        )
        print(f"sigma {sigma:.2f}: Brier {np.mean((predicted - actual) ** 2):.4f}   predicted->actual {table}")


def main() -> int:
    utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slot", type=int, default=None, help="default: every slot once")
    parser.add_argument("--drafts", type=int, default=1, help="drafts per slot")
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    if args.calibrate:
        calibrate([0.15, 0.21, 0.25, 0.30, 0.35, 0.40], sims=1000)
        return 0

    teams = load_config()["teams"]
    slots_to_run = [args.slot] if args.slot else list(range(1, teams + 1))
    worst, failures = 0.0, []
    for slot in slots_to_run:
        for draft_index in range(args.drafts):
            show = slot == slots_to_run[0] and draft_index == 0
            slowest, filled = synthetic_draft(slot, seed=1000 * slot + draft_index, sims=args.sims, show=show)
            worst = max(worst, slowest)
            if not filled:
                failures.append((slot, draft_index))
            print(f"slot {slot} draft {draft_index}: slowest pick {slowest:.2f}s, lineup "
                  f"{'filled' if filled else 'NOT FILLED'}")
    verdict = "PASS" if worst < PICK_BUDGET_SECONDS and not failures else "FAIL"
    print(f"\n{verdict}: slowest pick {worst:.2f}s against a {PICK_BUDGET_SECONDS:.0f}s budget; "
          f"unfilled lineups: {failures or 'none'}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
