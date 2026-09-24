# Sleeper Lock-In Assistant

Decision-support for one 8-team Sleeper NBA fantasy league that plays **Lock-In**: only one
game per player per week counts, and you choose which one by locking it after the game ends.
That single rule makes standard fantasy basketball rankings wrong here — volatility becomes an
asset, and missed games are cheap.

The tool never writes to Sleeper. Every recommendation ends with you doing something in the app.

- Design: [`docs/SPEC.md`](docs/SPEC.md)
- Milestones: [`docs/TASKS.md`](docs/TASKS.md)
- Working notes and current state: [`CLAUDE.md`](CLAUDE.md)

## Setup

```bash
uv sync
```

Copy `.env.example` to `.env` if you have Supabase or Anthropic credentials. Neither is
required for the draft board — the engine degrades to local parquet and a paste-into-chat
briefing.

## Running things

```bash
uv run python -m engine.jobs.verify_league     # check config/league.json against live Sleeper
uv run python -m engine.jobs.ingest_history    # pull game logs + schedule (LOCAL ONLY, see below)
uv run pytest                                  # no test needs the network
```

`verify_league` exits non-zero if the league's live scoring or roster slots drift from
`config/league.json`, which is the only source of truth for league rules.

## Data sources and their traps

### stats.nba.com is local-only

`engine/jobs/ingest_history.py` pulls player game logs and the season schedule from
stats.nba.com. **stats.nba.com blocks datacenter IPs**, so this works from a laptop and fails
from a GitHub-hosted Actions runner. It is deliberately a by-hand job:

- Output is written to parquet under `data/nba/` (gitignored, regenerable).
- It is incremental. A season already on disk is not refetched, so re-running costs nothing.
- Calls are spaced by `REQUEST_SPACING_SECONDS`. It never retries into a rate limit.
- Everything downstream, including every test, reads the parquet rather than the network.

Pass `--force` to refetch. Worth doing for the *schedule* closer to the season, since
unscheduled games get filled in; completed seasons' logs never change.

### cdn.nba.com is blocked

`docs/SPEC.md` §2.2 suggests cdn.nba.com's static schedule JSON as a lighter-weight
alternative. It returns Akamai `403 Access Denied` from this machine, with and without browser
headers (verified 2026-09-23). The schedule comes from stats.nba.com's `ScheduleLeagueV2`
instead, which also supplies the league's week grid.

### Known gaps in the historical data

Surfaced rather than silently defaulted, per rule 4 in `CLAUDE.md`:

- **Technical and flagrant fouls are not in `PlayerGameLogs`.** This league scores them
  (`tf` −1.0, `ff` −2.0) but the box score endpoint only carries personal fouls. Rescored
  history stores them as `0.0` and lists them in `GAME_LOG_MISSING_STATS`, so historical
  fantasy points run very slightly high for players who collect technicals. Recovering them
  needs play-by-play.
- **Two games per team are not yet scheduled.** The NBA Cup leaves them unassigned until the
  group stage resolves. `schedule_completeness()` quantifies this, and games-per-week for the
  affected weeks is a floor rather than a count — which matters, because games per week is a
  direct input to lock-in value.

## Layout

```
config/league.json     source of truth for league rules; verified against Sleeper on every run
engine/ingest/         Sleeper, NBA stats, markets, news clients
engine/model/          scoring, distributions, lock-in optimal stopping, valuation, draft sim
engine/jobs/           entrypoints
tests/                 fixtures only, never the network
```
