"""Live draft advisor: poll the Sleeper draft, recommend a pick, survive network failure.

    uv run python -m engine.jobs.draft_live                      # the league's real draft
    uv run python -m engine.jobs.draft_live --draft-id <mock id> # a Sleeper mock draft
    uv run python -m engine.jobs.draft_live --once               # one pass, then exit

Writes ``data/draft/live.html`` -- open it in a browser; it reloads itself every few seconds
-- and prints the same advice to the terminal. Read-only: it never touches Sleeper state.

**Failure behaviour is the point of half of this file.** On any fetch error it keeps showing
the last good state with a red banner saying how stale it is, and the page itself turns red
if the advisor process stops updating it at all. Either way the fallback is the static board,
``data/board/board.html``, which needs no network.

Needs ``data/board/pool.csv`` from ``build_board``. The model is ``engine/model/draft_sim.py``.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from engine.ingest.sleeper import Draft, DraftPick, SleeperClient
from engine.model.draft_sim import (
    DEFAULT_SIGMA,
    DEFAULT_SIMS,
    UNRANKED_ADP,
    DraftClock,
    advise,
    draft_clock,
)
from engine.jobs.runtime import run_loop, utf8_output
from engine.model.lineup import assign_lineup, slot_position, starting_slots

CONFIG_PATH = Path("config/league.json")
POOL_PATH = Path("data/board/pool.csv")
OUT_DIR = Path("data/draft")
DEFAULT_USERNAME = "Squ33gee"
DEFAULT_INTERVAL_SECONDS = 3.0
ROWS_SHOWN = 15
# The page goes red if it has not been rewritten for this long: the advisor has died.
STALE_AFTER_SECONDS = 20


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_pool(path: Path = POOL_PATH) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing {path}. Run: uv run python -m engine.jobs.build_board")
    pool = pd.read_csv(path, dtype={"sleeper_id": str})
    pool["positions"] = pool["positions"].map(lambda text: tuple(ast.literal_eval(text)))
    pool["flags"] = pool["flags"].fillna("")
    return pool.set_index("sleeper_id", drop=False)


# --- state ------------------------------------------------------------------------------------


@dataclass
class Recommendation:
    sleeper_id: str
    name: str
    team: str
    positions: tuple[str, ...]
    vor: float
    gain: float
    p_survive: float
    adp: float | None
    score: float
    fills: str  # slot he would start in, or "bench"
    why: str
    flags: str


@dataclass
class View:
    draft_status: str
    teams: int
    rounds: int
    my_slot: int | None
    clock: DraftClock | None
    recommendations: list[Recommendation]
    roster: list[tuple[str, str, str, float]]  # (slot or BN, name, positions, vor)
    open_slots: list[str]
    recent: list[tuple[int, str, str]]  # (pick_no, who, player)
    unvalued_taken: list[str] = field(default_factory=list)
    compute_seconds: float = 0.0


def pick_name(pick: DraftPick) -> str:
    meta = pick.model_extra.get("metadata") or {}
    return f"{meta.get('first_name', '')} {meta.get('last_name', '')}".strip() or pick.player_id


def resolve_slot(draft: Draft, picks: list[DraftPick], user_id: str, override: int | None) -> int | None:
    """The user's draft slot: from the draft order, else his own picks, else the override."""
    if override is not None:
        return override
    if draft.draft_order and user_id in draft.draft_order:
        return int(draft.draft_order[user_id])
    for pick in picks:
        if pick.model_extra.get("picked_by") == user_id:
            return int(pick.model_extra["draft_slot"])
    return None


def why_line(row: pd.Series, fills: str, p_survive: float, clock: DraftClock | None, gain: float) -> str:
    parts = []
    if fills != "bench":
        parts.append(f"starts at {fills}")
    else:
        parts.append("bench depth")
    parts.append(f"VOR {row['vor']:+.1f}" + (f", +{gain:.1f} to your lineup" if gain > 0 else ""))
    parts.append(f"{row['mean']:.0f}±{row['sd']:.0f} fp/g, plays {row['p_play']:.0%}")
    if clock is not None:
        target = clock.my_following_pick if clock.my_turn else clock.my_next_pick
        if target:
            parts.append(f"{p_survive:.0%} still there at #{target}")
    return "; ".join(parts)


def build_view(
    pool: pd.DataFrame,
    adp: dict[str, float],
    draft: Draft,
    picks: list[DraftPick],
    user_id: str,
    slots: list[str],
    teams: int,
    rounds: int,
    manager_names: dict[int, str],
    slot_override: int | None = None,
    n_sims: int = DEFAULT_SIMS,
    sigma: float = DEFAULT_SIGMA,
    rng: np.random.Generator | None = None,
) -> View:
    started = time.perf_counter()
    picks = sorted(picks, key=lambda pick: pick.pick_no)
    my_slot = resolve_slot(draft, picks, user_id, slot_override)
    clock = draft_clock(my_slot, teams, rounds, len(picks)) if my_slot else None

    taken = {pick.player_id for pick in picks}
    mine = [
        pick for pick in picks
        if my_slot is not None and int(pick.model_extra.get("draft_slot") or 0) == my_slot
    ]
    unvalued_taken = [pick_name(pick) for pick in picks if pick.player_id not in pool.index]

    roster_names, roster_vor, roster_positions = [], [], []
    for pick in mine:
        if pick.player_id in pool.index:
            row = pool.loc[pick.player_id]
            roster_names.append(row["name"])
            roster_vor.append(float(row["vor"]))
            roster_positions.append(row["positions"])
        else:  # drafted someone the board never valued: he counts, at replacement level
            meta = pick.model_extra.get("metadata") or {}
            roster_names.append(pick_name(pick) + " (unvalued)")
            roster_vor.append(0.0)
            roster_positions.append(tuple(filter(None, [meta.get("position")])))

    lineup = assign_lineup([max(v, 0.0) for v in roster_vor], roster_positions, slots)
    slot_of = {index: slot for slot, index in lineup.assignments.items()}
    roster = [
        (slot_position(slot_of[i]) if i in slot_of else "BN", roster_names[i],
         "/".join(roster_positions[i]), roster_vor[i])
        for i in range(len(roster_names))
    ]
    roster.sort(key=lambda row: (slots.index(next((s for s in slots if slot_position(s) == row[0]), slots[-1])) if row[0] != "BN" else 99))
    open_slots = [slot_position(slot) for slot in lineup.unfilled_slots]

    available = pool[~pool.index.isin(taken)]
    recommendations: list[Recommendation] = []
    if len(available):
        available_adp = np.array([adp.get(sid, UNRANKED_ADP) for sid in available.index])
        working_clock = clock or draft_clock(1, teams, rounds, len(picks))
        advice = advise(
            available["vor"].to_numpy(dtype=float),
            list(available["positions"]),
            available_adp,
            roster_vor,
            roster_positions,
            slots,
            working_clock,
            sigma=sigma,
            n_sims=n_sims,
            rng=rng,
        )
        for item in advice[:ROWS_SHOWN]:
            row = available.iloc[item.index]
            with_him = assign_lineup(
                [*[max(v, 0.0) for v in roster_vor], max(float(row["vor"]), 0.0)],
                [*roster_positions, row["positions"]],
                slots,
            )
            new_index = len(roster_vor)
            fills = next(
                (slot_position(slot) for slot, index in with_him.assignments.items() if index == new_index),
                "bench",
            )
            sid = row["sleeper_id"]
            recommendations.append(
                Recommendation(
                    sleeper_id=sid,
                    name=row["name"],
                    team=row["team"],
                    positions=row["positions"],
                    vor=float(row["vor"]),
                    gain=item.gain,
                    p_survive=item.p_survive if clock else float("nan"),
                    adp=adp.get(sid),
                    score=item.score,
                    fills=fills,
                    why=why_line(row, fills, item.p_survive, clock, item.gain),
                    flags=row["flags"],
                )
            )

    recent = [
        (pick.pick_no, manager_names.get(int(pick.model_extra.get("draft_slot") or 0),
                                         f"slot {pick.model_extra.get('draft_slot')}"),
         pick_name(pick))
        for pick in picks[-8:]
    ][::-1]

    return View(
        draft_status=draft.status,
        teams=teams,
        rounds=rounds,
        my_slot=my_slot,
        clock=clock,
        recommendations=recommendations,
        roster=roster,
        open_slots=open_slots,
        recent=recent,
        unvalued_taken=unvalued_taken,
        compute_seconds=time.perf_counter() - started,
    )


# --- rendering --------------------------------------------------------------------------------


@dataclass
class FetchStatus:
    last_ok: float | None = None
    last_error: str | None = None
    failures: int = 0


def headline(view: View) -> str:
    clock = view.clock
    if view.draft_status == "complete":
        return "Draft complete"
    if clock is None:
        return "Your draft slot is not known yet (draft order not set). Advice ranks by value only."
    if clock.on_clock is None:
        return "Draft complete"
    if clock.my_turn:
        return f"YOUR PICK — #{clock.on_clock} (round {(clock.on_clock - 1) // view.teams + 1})"
    if clock.my_next_pick is None:
        return f"Pick #{clock.on_clock} on the clock. You have no picks left."
    return (
        f"Pick #{clock.on_clock} on the clock. You pick at #{clock.my_next_pick} "
        f"({clock.picks_until_next} picks away)"
    )


def render_terminal(view: View, status: FetchStatus) -> str:
    lines = [headline(view)]
    if status.last_error:
        age = time.time() - status.last_ok if status.last_ok else float("nan")
        lines.append(f"!! FETCH FAILING ({status.failures}x): {status.last_error} -- showing data {age:.0f}s old")
    lines.append(f"slot {view.my_slot}  open: {', '.join(view.open_slots) or 'none'}  "
                 f"(computed in {view.compute_seconds:.1f}s)")
    lines.append("")
    lines.append(f"{'':3}{'player':25}{'pos':9}{'fills':6}{'gain':>6}{'VOR':>6}{'surv':>6}{'ADP':>6}")
    for index, rec in enumerate(view.recommendations[:10], 1):
        surv = "" if np.isnan(rec.p_survive) else f"{rec.p_survive:.0%}"
        adp = "" if rec.adp is None else f"{rec.adp:.0f}"
        lines.append(f"{index:<3}{rec.name[:24]:25}{'/'.join(rec.positions):9}{rec.fills:6}"
                     f"{rec.gain:6.1f}{rec.vor:6.1f}{surv:>6}{adp:>6}  {rec.flags.replace('|', ' ')}")
    if view.recommendations:
        lines.append(f"\n-> {view.recommendations[0].name}: {view.recommendations[0].why}")
    return "\n".join(lines)


def render_html(view: View, status: FetchStatus, refresh_seconds: float, now: float) -> str:
    esc = html.escape
    clock = view.clock
    error_block = ""
    if status.last_error:
        age = now - status.last_ok if status.last_ok else None
        age_text = f"data is {age:.0f}s old" if age is not None else "no data yet"
        error_block = (
            f"<div class=err>Sleeper fetch failing ({status.failures}x): {esc(status.last_error)} "
            f"&mdash; {age_text}. If this persists, switch to board.html.</div>"
        )
    top = ""
    if view.recommendations:
        rec = view.recommendations[0]
        verb = "Take" if clock and clock.my_turn else "Best for you now"
        top = (
            f"<div class=top><div class=lbl>{verb}</div><div class=pname>{esc(rec.name)} "
            f"<span class=mut>{esc(rec.team)} &middot; {esc('/'.join(rec.positions))}</span></div>"
            f"<div class=why>{esc(rec.why)}</div></div>"
        )
    rows = []
    for index, rec in enumerate(view.recommendations, 1):
        surv = "&ndash;" if np.isnan(rec.p_survive) else f"{rec.p_survive:.0%}"
        surv_class = "" if np.isnan(rec.p_survive) else ("lo" if rec.p_survive < 0.35 else ("hi" if rec.p_survive > 0.8 else ""))
        flags = "".join(f"<span class=flag>{esc(flag)}</span>" for flag in rec.flags.split("|") if flag)
        rows.append(
            f"<tr><td class=mut>{index}</td><td><b>{esc(rec.name)}</b> <span class=mut>{esc(rec.team)}</span>"
            f"{flags}<div class=whyrow>{esc(rec.why)}</div></td><td>{esc('/'.join(rec.positions))}</td>"
            f"<td>{esc(rec.fills)}</td><td class=num><b>{rec.gain:.1f}</b></td><td class=num>{rec.vor:.1f}</td>"
            f"<td class='num {surv_class}'>{surv}</td>"
            f"<td class=num>{'' if rec.adp is None else f'{rec.adp:.0f}'}</td></tr>"
        )
    roster_rows = "".join(
        f"<tr><td class=mut>{esc(slot)}</td><td>{esc(name)}</td><td>{esc(pos)}</td><td class=num>{vor:.1f}</td></tr>"
        for slot, name, pos, vor in view.roster
    ) or "<tr><td colspan=4 class=mut>no picks yet</td></tr>"
    recent_rows = "".join(
        f"<li><span class=mut>#{pick_no} {esc(who)}</span> {esc(player)}</li>" for pick_no, who, player in view.recent
    ) or "<li class=mut>no picks yet</li>"
    unvalued = ""
    if view.unvalued_taken:
        unvalued = (
            "<div class=gap>Drafted but not on our board (no valuation): "
            + esc(", ".join(view.unvalued_taken)) + "</div>"
        )
    updated = datetime.fromtimestamp(now).strftime("%H:%M:%S")
    turn_class = " myturn" if clock and clock.my_turn else ""
    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<meta http-equiv=refresh content="{refresh_seconds:.0f}">
<title>Draft Live</title>
<style>
:root{{--bg:#fff;--fg:#111;--mut:#666;--line:#e3e3e3;--accent:#0b5;--warn:#c60;--bad:#c22;--head:#f7f7f7}}
@media(prefers-color-scheme:dark){{:root{{--bg:#14161a;--fg:#e8e8e8;--mut:#9aa0a6;--line:#2a2e35;--accent:#3d8;--warn:#f93;--bad:#f55;--head:#1c1f25}}}}
*{{box-sizing:border-box}}
body{{margin:0;padding:12px;background:var(--bg);color:var(--fg);font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
.hdr{{font-size:20px;font-weight:700;padding:8px 10px;border-radius:6px;background:var(--head)}}
.hdr.myturn{{background:var(--accent);color:#000}}
.meta{{color:var(--mut);font-size:12px;margin:4px 0 8px}}
.err,.stale{{background:var(--bad);color:#fff;padding:8px 10px;border-radius:6px;margin:8px 0;font-weight:600}}
.stale{{display:none}}
.gap{{border-left:3px solid var(--warn);padding:4px 8px;margin:8px 0;font-size:12px}}
.top{{border:2px solid var(--accent);border-radius:8px;padding:10px;margin:8px 0}}
.lbl{{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut)}}
.pname{{font-size:22px;font-weight:700}}
.why{{margin-top:4px}}
.mut{{color:var(--mut)}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:5px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{font-size:11px;text-transform:uppercase;color:var(--mut)}}
td.num{{text-align:right;white-space:nowrap}}
td.lo{{color:var(--bad);font-weight:700}} td.hi{{color:var(--accent)}}
.whyrow{{font-size:11px;color:var(--mut)}}
.flag{{font-size:10px;color:var(--warn);border:1px solid var(--warn);border-radius:3px;padding:0 3px;margin-left:4px}}
.cols{{display:grid;grid-template-columns:1fr;gap:12px}}
@media(min-width:900px){{.cols{{grid-template-columns:2fr 1fr}}}}
h2{{font-size:13px;text-transform:uppercase;color:var(--mut);margin:12px 0 4px}}
ul{{margin:0;padding-left:18px}}
.wrap{{overflow-x:auto}}
</style></head><body>
<div class="hdr{turn_class}">{esc(headline(view))}</div>
<div class=meta>slot {view.my_slot or '?'} &middot; open: {esc(', '.join(view.open_slots) or 'none')}
&middot; updated <span id=upd data-t="{now:.0f}">{updated}</span> &middot; computed in {view.compute_seconds:.1f}s</div>
<div class=stale id=stale>This page has stopped updating &mdash; the advisor is not running. Use board.html.</div>
{error_block}{unvalued}{top}
<div class=cols><div><h2>Best for you</h2><div class=wrap><table>
<tr><th>#</th><th>Player</th><th>Pos</th><th>Fills</th><th title="gain to your starting lineup, VOR units">Gain</th>
<th>VOR</th><th title="chance he is still there at your next pick">Back?</th><th>ADP</th></tr>
{''.join(rows)}
</table></div></div>
<div><h2>Your roster</h2><table>{roster_rows}</table>
<h2>Recent picks</h2><ul>{recent_rows}</ul></div></div>
<script>
(function(){{var t=+document.getElementById('upd').dataset.t;
setInterval(function(){{if(Date.now()/1000-t>{STALE_AFTER_SECONDS})document.getElementById('stale').style.display='block'}},1000)}})();
</script>
</body></html>"""


def write_atomic(path: Path, text: str) -> None:
    """Write via a temp file so a browser reload never sees half a page."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


# --- main loop --------------------------------------------------------------------------------


def manager_names_by_slot(client: SleeperClient, draft: Draft) -> dict[int, str]:
    if not draft.draft_order or not draft.league_id:
        return {}
    try:
        users = {user.user_id: user.display_name for user in client.get_users(draft.league_id)}
    except Exception:  # names are cosmetic; never let them stop the advisor
        return {}
    return {slot: users.get(user_id, f"slot {slot}") for user_id, slot in draft.draft_order.items()}


@dataclass
class LivePoller:
    """One pass of the advisor: fetch the draft, rebuild the view if anything changed, write
    the page. ``main`` runs ``poll`` through ``runtime.run_loop``, so an exception anywhere in a
    pass -- not only in the fetch -- is logged and the next pass still runs."""

    client: SleeperClient
    config: dict
    pool: pd.DataFrame
    adp: dict[str, float]
    user_id: str
    slots: list[str]
    draft_id: str
    page: Path
    interval: float
    slot_override: int | None = None
    n_sims: int = DEFAULT_SIMS
    status: FetchStatus = field(default_factory=FetchStatus)
    draft: Draft | None = None
    picks: list[DraftPick] = field(default_factory=list)
    names: dict[int, str] = field(default_factory=dict)
    view: View | None = None
    signature: tuple | None = None

    def poll(self) -> None:
        status = self.status
        try:
            self.draft = self.client.get_draft_live(self.draft_id)
            self.picks = self.client.get_draft_picks_live(self.draft_id)
            status.last_ok, status.last_error, status.failures = time.time(), None, 0
        except Exception as error:
            status.last_error = f"{type(error).__name__}: {error}"[:200]
            status.failures += 1

        draft = self.draft
        if draft is not None:
            settings = draft.model_extra.get("settings") or {}
            teams = int(settings.get("teams") or self.config["teams"])
            rounds = int(settings.get("rounds") or self.config["draft_settings_live"]["rounds"])
            if not self.names:
                self.names = manager_names_by_slot(self.client, draft)
            signature = (len(self.picks), json.dumps(draft.draft_order, sort_keys=True), draft.status)
            if signature != self.signature:
                self.view = build_view(self.pool, self.adp, draft, self.picks, self.user_id,
                                       self.slots, teams, rounds, self.names, self.slot_override,
                                       n_sims=self.n_sims)
                self.signature = signature
                print("\033[2J\033[H" + render_terminal(self.view, status), flush=True)
            elif status.last_error:
                print(f"!! fetch failing ({status.failures}x): {status.last_error}", flush=True)

        if self.view is not None:
            write_atomic(self.page, render_html(self.view, status, self.interval, time.time()))
        elif status.last_error:
            print(f"!! cannot reach Sleeper yet: {status.last_error}", flush=True)

    def done(self) -> bool:
        return self.draft is not None and self.draft.status == "complete"

    def failed(self, error: Exception, log_path: Path) -> None:
        """A pass raised. Say so in the terminal and on the page, keep the last good view."""
        self.status.last_error = f"advisor error {type(error).__name__}: {error}"[:200]
        self.status.failures += 1
        print(f"!! poll failed ({self.status.failures}x): {self.status.last_error} "
              f"-- traceback in {log_path}", flush=True)
        if self.view is not None:
            try:
                write_atomic(self.page, render_html(self.view, self.status, self.interval, time.time()))
            except Exception:  # the page write may be what failed; the next pass retries it
                pass


def main() -> int:
    utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config = load_config()
    parser.add_argument("--draft-id", default=config["draft_settings_live"]["draft_id"])
    parser.add_argument("--user", default=DEFAULT_USERNAME, help="your Sleeper username")
    parser.add_argument("--slot", type=int, default=None, help="force your draft slot")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--once", action="store_true", help="one pass, then exit")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    slots = starting_slots(config["roster_positions"])
    pool = load_pool()
    page = args.out / "live.html"

    with SleeperClient() as client:
        user_id = client.get_user(args.user).user_id
        season = config["season"].split("-")[0]
        try:
            adp = client.get_adp(season)
        except Exception as error:  # ADP only feeds the opponent model; say so and go on
            print(f"!! no ADP ({error}); every player treated as unranked by opponents")
            adp = {}
        print(f"pool {len(pool)} players, ADP for {sum(sid in adp for sid in pool.index)} of them; "
              f"user {args.user} = {user_id}; page {page.resolve()}")

        poller = LivePoller(client, config, pool, adp, user_id, slots, args.draft_id, page,
                            args.interval, slot_override=args.slot, n_sims=args.sims)
        run_loop("draft_live", poller.poll, interval=args.interval,
                 should_stop=lambda: args.once or poller.done(), on_error=poller.failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
