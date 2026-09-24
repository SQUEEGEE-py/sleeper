"""Build the draft board: board.csv plus a static HTML page that works with no network.

    uv run python -m engine.jobs.build_board [--top 120] [--out data/board]

This is the phase 1 deliverable (TASKS M4). The HTML page is deliberately self-contained --
no CDN, no fonts, no fetch -- because its job is to still work on a phone in a draft room
when the live advisor or the wifi has fallen over.

Reads the parquet written by ``ingest_history`` and the cached Sleeper player dictionary.
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from engine.ingest.nba_stats import DATA_DIR, previous_seasons
from engine.ingest.player_ids import (
    load_overrides,
    map_players,
    normalize_name,
    overrides_by_nba_id,
)
from engine.ingest.sleeper import SleeperClient
from engine.model.distributions import (
    EmpiricalScoreDistribution,
    build_distribution,
    role_prior,
    score_game_logs,
)
from engine.model.lineup import starting_slots
from engine.model.lockin import season_value, weekly_value
from engine.model.valuation import PlayerValue, build_board as rank_board

CONFIG_PATH = Path("config/league.json")
HISTORY_SEASONS = 3
DEFAULT_TOP = 120

# A player needs some recent evidence to appear at all. Someone who last played two seasons
# ago and not since is not a draft candidate; he is noise on a 120-row board.
MIN_RECENT_GAMES = 10


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_logs(data_dir: Path, season: str) -> pd.DataFrame:
    frames = []
    for past in previous_seasons(season, HISTORY_SEASONS):
        path = data_dir / f"game_logs_{past}.parquet"
        if not path.exists():
            raise SystemExit(
                f"missing {path}. Run: uv run python -m engine.jobs.ingest_history"
            )
        frames.append(pd.read_parquet(path))
    logs = pd.concat(frames, ignore_index=True)
    logs["fp"] = score_game_logs(logs)
    return logs


def candidate_players(logs: pd.DataFrame, season: str) -> pd.DataFrame:
    """Players with enough recent evidence to be worth ranking."""
    latest = previous_seasons(season, 1)[0]
    recent = logs[logs["season"] == latest]
    counts = recent.groupby("nba_player_id").size()
    keep = set(counts[counts >= MIN_RECENT_GAMES].index)
    return (
        recent[recent["nba_player_id"].isin(keep)]
        .sort_values("game_date")
        .drop_duplicates("nba_player_id", keep="last")[
            ["nba_player_id", "player_name", "team"]
        ]
        .reset_index(drop=True)
    )


# Minutes bands for the role prior (SPEC §5.1 step 3). Minutes are the usable proxy for role:
# a 34-minute player and a 14-minute player have different distributions whatever their listed
# position. Bands are wide enough that each holds thousands of games.
MINUTES_BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 12.0),
    (12.0, 18.0),
    (18.0, 24.0),
    (24.0, 30.0),
    (30.0, 36.0),
    (36.0, 60.0),
)


def build_minutes_priors(logs: pd.DataFrame) -> dict[tuple[float, float], EmpiricalScoreDistribution]:
    """One prior per minutes band, built once and shared across players."""
    priors = {}
    for band in MINUTES_BANDS:
        try:
            priors[band] = role_prior(logs, band[0], band[1])
        except ValueError:
            continue  # an empty band is fine; nobody will map to it
    return priors


def band_for(minutes: float) -> tuple[float, float]:
    for band in MINUTES_BANDS:
        if band[0] <= minutes < band[1]:
            return band
    return MINUTES_BANDS[-1]


def apply_override(
    distribution: EmpiricalScoreDistribution, multiplier: float, p_play: float | None
) -> EmpiricalScoreDistribution:
    """Scale a distribution for a role change, and/or force availability."""
    updated = distribution
    if multiplier != 1.0:
        updated = dataclasses.replace(updated, values=updated.values * multiplier)
    if p_play is not None:
        updated = dataclasses.replace(updated, _p_play=p_play)
    return updated


def render_html(
    rows: list[dict], levels: dict, config: dict, generated: str, gaps: list[str]
) -> str:
    """A single self-contained page. No network of any kind."""
    flags = config["scoring_ambiguities"]
    ambiguity = (
        f"bonuses_stack={str(flags['bonuses_stack']['value']).lower()}, "
        f"td_includes_dd={str(flags['td_includes_dd']['value']).lower()}"
    )
    replacement_cells = "".join(
        f"<span class=rep><b>{html.escape(slot)}</b> {value:.1f}</span>"
        for slot, value in sorted(levels.items(), key=lambda item: -item[1])
    )
    gap_block = ""
    if gaps:
        items = "".join(f"<li>{html.escape(gap)}</li>" for gap in gaps)
        gap_block = f"<div class=gaps><b>Known gaps</b><ul>{items}</ul></div>"

    body_rows = []
    for index, row in enumerate(rows, start=1):
        classes = []
        if not row["history_reliable"]:
            classes.append("thin")
        if row["value_multiplier"] != 1.0:
            classes.append("adj")
        flag_bits = []
        if not row["history_reliable"]:
            flag_bits.append(f"<span class=warn title='only {row['games_observed']} games'>thin</span>")
        if row["value_multiplier"] != 1.0:
            flag_bits.append(
                f"<span class=adjm title='{html.escape(row['note'])}'>x{row['value_multiplier']:.2f}</span>"
            )
        body_rows.append(
            "<tr class='{cls}'><td class=rank>{rank}</td><td class=name>{name}"
            "<span class=team>{team}</span></td><td>{pos}</td>"
            "<td class=num><b>{vor:.1f}</b></td><td class=num>{season:.1f}</td>"
            "<td class=num>{rep:.1f}</td><td class=num>{mean:.1f}</td>"
            "<td class=num>{sd:.1f}</td><td class=num>{pplay:.0%}</td>"
            "<td class=num>{w1:.0f}/{w2:.0f}/{w3:.0f}/{w4:.0f}</td><td>{flags}</td></tr>".format(
                cls=" ".join(classes),
                rank=index,
                name=html.escape(row["name"]),
                team=html.escape(row["team"]),
                pos=html.escape("/".join(row["positions"])),
                vor=row["vor"],
                season=row["season_value"],
                rep=row["replacement"],
                mean=row["mean"],
                sd=row["sd"],
                pplay=row["p_play"],
                w1=row["w1"], w2=row["w2"], w3=row["w3"], w4=row["w4"],
                flags="".join(flag_bits),
            )
        )

    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Draft Board</title>
<style>
:root{{--bg:#fff;--fg:#111;--mut:#666;--line:#e3e3e3;--accent:#0b5;--warn:#c60;--head:#f7f7f7}}
@media(prefers-color-scheme:dark){{:root{{--bg:#14161a;--fg:#e8e8e8;--mut:#9aa0a6;--line:#2a2e35;--accent:#3d8;--warn:#f93;--head:#1c1f25}}}}
*{{box-sizing:border-box}}
body{{margin:0;padding:12px;background:var(--bg);color:var(--fg);
font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
h1{{font-size:18px;margin:0 0 2px}}
.meta{{color:var(--mut);font-size:12px;margin-bottom:8px}}
.rep{{display:inline-block;margin:0 8px 4px 0;padding:2px 6px;background:var(--head);
border:1px solid var(--line);border-radius:4px;font-size:12px}}
.rep b{{color:var(--accent)}}
.gaps{{border-left:3px solid var(--warn);padding:6px 10px;margin:8px 0;background:var(--head);font-size:12px}}
.gaps ul{{margin:4px 0 0;padding-left:18px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:5px 6px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{position:sticky;top:0;background:var(--head);font-size:11px;text-transform:uppercase;
letter-spacing:.04em;color:var(--mut);cursor:pointer;user-select:none}}
td.num{{text-align:right}}
td.rank{{color:var(--mut);width:34px}}
.name{{font-weight:600}}
.team{{color:var(--mut);font-weight:400;font-size:11px;margin-left:6px}}
tr.thin{{background:rgba(255,153,51,.07)}}
.warn{{color:var(--warn);font-size:11px;border:1px solid var(--warn);border-radius:3px;padding:0 4px}}
.adjm{{color:var(--accent);font-size:11px;border:1px solid var(--accent);border-radius:3px;padding:0 4px;margin-left:4px}}
#q{{width:100%;padding:8px;margin-bottom:8px;border:1px solid var(--line);border-radius:6px;
background:var(--bg);color:var(--fg);font-size:16px}}
.wrap{{overflow-x:auto}}
</style></head><body>
<h1>Draft Board &mdash; {html.escape(config.get('_league_name', 'Lock-In'))}</h1>
<div class=meta>{len(rows)} players &middot; {config['teams']} teams &middot; generated {generated}
&middot; scoring flags: {ambiguity} &middot; <b>works offline</b></div>
<div>{replacement_cells}</div>
{gap_block}
<input id=q placeholder="filter by name, team or position" autocomplete=off>
<div class=wrap><table id=b><thead><tr>
<th>#</th><th>Player</th><th>Pos</th><th title="value over replacement">VOR</th>
<th title="mean weekly lock-in value over the real schedule">Season</th>
<th title="replacement level for his best slot">Repl</th>
<th title="mean fantasy points per game played">Mean</th>
<th title="standard deviation; higher is better in Lock-In">SD</th>
<th title="probability he plays a given game">Play</th>
<th title="weekly value in a 1/2/3/4 game week">W 1/2/3/4</th><th></th>
</tr></thead><tbody>
{''.join(body_rows)}
</tbody></table></div>
<script>
var q=document.getElementById('q'),rows=[].slice.call(document.querySelectorAll('#b tbody tr'));
q.addEventListener('input',function(){{var t=q.value.toLowerCase();
rows.forEach(function(r){{r.style.display=r.textContent.toLowerCase().indexOf(t)<0?'none':''}})}});
document.querySelectorAll('#b th').forEach(function(th,i){{th.addEventListener('click',function(){{
var tb=document.querySelector('#b tbody'),rs=[].slice.call(tb.querySelectorAll('tr'));
var asc=th.dataset.asc==='1';th.dataset.asc=asc?'0':'1';
rs.sort(function(a,b){{var x=a.cells[i].textContent.trim(),y=b.cells[i].textContent.trim();
var nx=parseFloat(x.replace('%','')),ny=parseFloat(y.replace('%',''));
if(!isNaN(nx)&&!isNaN(ny))return asc?nx-ny:ny-nx;
return asc?x.localeCompare(y):y.localeCompare(x)}});
rs.forEach(function(r){{tb.appendChild(r)}})}})}});
</script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=Path("data/board"))
    args = parser.parse_args()

    config = load_config()
    season = config["season"]
    slots = starting_slots(config["roster_positions"])

    grid_path = args.data_dir / f"games_per_team_per_week_{season}.parquet"
    if not grid_path.exists():
        raise SystemExit(f"missing {grid_path}. Run: uv run python -m engine.jobs.ingest_history")
    grid = pd.read_parquet(grid_path)
    grid.columns = [int(column) for column in grid.columns]

    logs = load_logs(args.data_dir, season)
    candidates = candidate_players(logs, season)
    print(f"{len(candidates)} candidates with >= {MIN_RECENT_GAMES} recent games")

    with SleeperClient() as client:
        sleeper_players = client.get_players()

    overrides = load_overrides()
    mapping = map_players(
        [
            (int(row.nba_player_id), row.player_name, row.team)
            for row in candidates.itertuples()
        ],
        sleeper_players,
        overrides,
    )
    print(mapping.describe())

    name_to_id = {
        normalize_name(row.player_name): int(row.nba_player_id)
        for row in candidates.itertuples()
    }
    override_by_id = overrides_by_nba_id(overrides, name_to_id)

    priors = build_minutes_priors(logs)
    mean_minutes = logs.groupby("nba_player_id")["minutes"].mean()

    def distribution_for(nba_id: int) -> EmpiricalScoreDistribution:
        """Recency-weighted, shrunk toward the prior for his minutes band, then overridden.

        Without the shrinkage a volatile 14-minute rookie outranks established starters:
        Lock-In rewards variance, and a thin sample is almost all variance. The prior is what
        makes a small sample behave like a small sample.
        """
        prior = priors.get(band_for(float(mean_minutes.get(nba_id, 0.0))))
        distribution = build_distribution(logs, nba_id, prior=prior)
        override = override_by_id.get(nba_id)
        if override is not None:
            distribution = apply_override(distribution, override.value_multiplier, override.p_play)
        return distribution

    gaps: list[str] = []
    for nba_id, name, team in mapping.unmapped:
        gaps.append(f"unmapped to Sleeper: {name} ({team}, nba id {nba_id})")
    for nba_id, name, _ in mapping.ambiguous:
        gaps.append(f"ambiguous Sleeper match: {name} (nba id {nba_id})")

    players: list[PlayerValue] = []
    missing_schedule: set[str] = set()
    for row in candidates.itertuples():
        nba_id = int(row.nba_player_id)
        sleeper_id = mapping.nba_to_sleeper.get(nba_id)
        if sleeper_id is None:
            continue  # already reported as a gap above

        sleeper_player = sleeper_players[sleeper_id]
        positions = tuple(sleeper_player.fantasy_positions or [])
        if not positions:
            gaps.append(f"no fantasy positions in Sleeper: {row.player_name}")
            continue

        team = sleeper_player.team or row.team
        if team not in grid.index:
            missing_schedule.add(team)
            continue

        override = override_by_id.get(nba_id)
        distribution = distribution_for(nba_id)

        schedule = [int(games) for games in grid.loc[team]]
        players.append(
            PlayerValue(
                player_id=nba_id,
                name=row.player_name,
                team=team,
                positions=positions,
                season_value=season_value(distribution, schedule),
                weekly_value=weekly_value(distribution, 3),
                mean=distribution.mean(),
                sd=distribution.sd(),
                p_play=distribution.p_play,
                games_observed=distribution.games_observed,
                history_reliable=distribution.history_reliable,
                value_multiplier=override.value_multiplier if override else 1.0,
                note=override.note if override else "",
            )
        )

    for team in sorted(missing_schedule):
        gaps.append(f"no 2026-27 schedule for team {team}; its players are omitted")

    if not players:
        raise SystemExit("no players could be valued; nothing to rank")

    ranked, levels = rank_board(players, slots, config["teams"])
    if not levels.is_complete:
        gaps.append(
            f"only {levels.starters_filled} of {levels.starter_slots} league starting slots "
            "could be filled from the candidate pool; replacement levels are approximate"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for player, vor, replacement in ranked[: args.top]:
        distribution_row = {
            "name": player.name,
            "team": player.team,
            "positions": list(player.positions),
            "vor": vor,
            "season_value": player.season_value,
            "replacement": replacement,
            "mean": player.mean,
            "sd": player.sd,
            "p_play": player.p_play,
            "games_observed": player.games_observed,
            "history_reliable": player.history_reliable,
            "value_multiplier": player.value_multiplier,
            "note": player.note,
            "nba_player_id": player.player_id,
        }
        rows.append(distribution_row)

    # Weekly value by game count, for the board's W 1/2/3/4 column.
    for row, (player, _, _) in zip(rows, ranked[: args.top]):
        schedule_distribution = distribution_for(player.player_id)
        for n in (1, 2, 3, 4):
            row[f"w{n}"] = weekly_value(schedule_distribution, n)

    frame = pd.DataFrame(rows)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    csv_path = args.out / "board.csv"
    frame.to_csv(csv_path, index=False)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    config_with_name = {**config, "_league_name": "National Logan League"}
    html_path = args.out / "board.html"
    html_path.write_text(
        render_html(rows, levels.by_slot, config_with_name, generated, gaps), encoding="utf-8"
    )

    print(f"\nreplacement levels: "
          + ", ".join(f"{slot} {value:.1f}" for slot, value in sorted(levels.by_slot.items())))
    print(f"\nwrote {csv_path} and {html_path} ({len(frame)} players)")
    if gaps:
        print(f"\n{len(gaps)} gap(s) surfaced on the board:")
        for gap in gaps[:10]:
            print(f"  - {gap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
