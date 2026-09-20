# CLAUDE.md

Context for Claude Code working in this repo. Read `docs/SPEC.md` for the full design and
`docs/TASKS.md` for the milestone list. **Current state and next steps are in the "Current
state" section below** — read it before starting work, it says what is actually built.

## What this is

A decision-support tool for a single 8-team Sleeper NBA fantasy league that uses **Lock-In**
scoring. Two products share one valuation engine:

1. **Draft assistant** — a ranked board plus a live snake-draft advisor.
2. **In-season coach** — a nightly lock-or-roll advisor, waiver and trade suggestions.

The tool **never writes to Sleeper**. Sleeper's public API is read-only. Every recommendation
ends with the user taking the action in the Sleeper app.

## Non-negotiable rules

1. **`config/league.json` is the only source of truth for league rules.** Never hardcode
   scoring weights, roster slots, or team counts anywhere else. On startup, fetch the league
   from Sleeper and assert the config matches; fail loudly on mismatch.
2. **The math is deterministic code, not an LLM.** Projections, distributions, optimal stopping,
   draft simulation, and win probability are all computed in Python and unit tested. The LLM
   layer only explains results, ingests unstructured news, and drafts trade messages. If you
   find yourself asking a model to compute a number, that is a bug.
3. **Lock-In is not standard fantasy basketball.** Only one game per player per week counts.
   Any ranking, projection, or piece of advice copied from standard fantasy logic is wrong here.
   See `docs/SPEC.md` §3.
4. **Never invent data.** If an API is unreachable or a player is unmapped, surface it as an
   explicit gap in the UI. A recommendation built on a silently-defaulted number is worse than
   no recommendation.
5. **Secrets stay in `.env`** (gitignored). No keys, league IDs excepted, in committed code.
6. **Verify third-party endpoints before building against them.** The URLs in the spec were
   accurate when written but these are mostly undocumented or fast-moving APIs. Probe first,
   then write the client.

## Current state (2026-09-19)

**M0 complete** (commit `1d14dc4`, pushed to `origin/main`). **M1 not started.**

What exists and works:

- `uv` project on Python 3.11 with the M0 deps. Run anything with `uv run python -m <module>`.
- Repo skeleton from SPEC §1. Modules belonging to later milestones are **empty placeholder
  files, not stubs** — `scoring.py`, `lockin.py`, `distributions.py` and friends are 0 bytes
  on purpose. Don't mistake them for work in progress.
- `engine/ingest/sleeper.py` — typed pydantic client for the SPEC §2.1 endpoints. Every
  response is cached to disk under `data/cache/sleeper/` with a fetch timestamp; `players/nba`
  is TTL-capped at 24h. Reruns and tests never need the network.
- `engine/jobs/verify_league.py` — diffs live `scoring_settings` and `roster_positions`
  against `config/league.json`, prints a readable diff, exits non-zero on mismatch.

Endpoints probed against the live API on 2026-09-17 per rule 6, using a public league:
`/league/{id}` (`scoring_settings` dict, `roster_positions` list), `/rosters`, `/users`,
`/drafts`, and `/players/nba` (dict keyed by player_id, ~2100 players, 2.4 MB). All match the
shapes the spec describes.

### Blockers, in priority order

1. **`config/league.json` still has `league_id: "TODO_FILL_IN"`.** `verify_league` exits 1 with
   an explicit message until it is filled in, which means the M0 acceptance test — print the
   league name, 8 teams, the roster slots — **has never actually run against the real league.**
   Highest-value thing to unblock; everything downstream assumes config matches reality.
2. **Sleeper's NBA `scoring_settings` key names are unconfirmed.** The endpoint *shapes* were
   verified with an NFL league; the NBA field names were not. So `diff_scoring` compares
   `config/league.json`'s own key names verbatim, and reports a config key that is absent from
   the live payload as its own distinct case rather than guessing a translation table
   (`flagrant_foul` vs `pf_flag`, `bonus_40_pts` vs `bonus_pts_40`, etc). Expect the first real
   run to surface name mismatches. Resolve them by reading the live payload and telling the
   user, never by inventing a mapping.
3. **Both `scoring_ambiguities` remain unresolved** (`bonuses_stack`, `td_includes_dd`). Per
   SPEC §4 the empirical fix is to recompute real week-1 box scores and diff against Sleeper's
   own totals, which is impossible before the season opens. Both branches stay behind flags.

### Next step — M1 (scoring engine)

`engine/model/scoring.py` per SPEC §4, then `tests/test_scoring.py` over 8–10 hand-scored
fixtures in `tests/fixtures/`. The parts that are easy to get wrong:

- Every weight comes from `config/league.json`. No hardcoded numbers anywhere, including in
  the tests.
- Both ambiguity flags honored, and the suite must pass under **both** settings of each.
- `quadruple_double: "treat_as_td"` must not crash.
- No network in any test.

A fixture set with hand-computed expected values has been drafted and is **awaiting the user's
review** before the test file is written; the user asked to check the arithmetic first. The
fixtures that actually discriminate between flag settings are the triple-double, the 50-point
game, and a 50-point triple-double that exercises both flags at once. The remaining ones
(plain line, double-double, 40-point game, tech, high-turnover, flagrant, quadruple-double)
are flag-independent except where noted.

Do not start M2 until M1's acceptance test passes.

## Stack

- **Engine:** Python 3.11+, `uv` for deps. numpy, pandas, scipy, httpx, pydantic, pytest.
- **Storage:** Supabase (Postgres). Migrations in `supabase/migrations/`.
- **Web:** React + Vite + TypeScript + Tailwind, deployed on Vercel at `fantasy.ferf.gg`
  (DNS in Route 53, same pattern as the househunt tracker).
- **Scheduling:** GitHub Actions cron for anything that talks to Sleeper/Kalshi/Polymarket.
- **Notifications:** ntfy.sh topic (free, no account) for nightly lock alerts.
- **LLM:** Anthropic API, optional. Must degrade to a "copy this briefing into a chat" text
  blob if no key is set. See `docs/SPEC.md` §8.

## Known environment traps

- **stats.nba.com blocks datacenter IPs.** Anything using `nba_api` will work locally and
  likely fail on GitHub-hosted runners. Design ingestion as a local-first script that writes
  parquet to Supabase, and keep it runnable by hand. Do not silently retry into a rate limit.
- **Player identity differs across sources.** Sleeper IDs, NBA.com IDs, and the free-text
  names in Kalshi/Polymarket markets do not agree. There is a mapping table with manual
  overrides; see spec §6.3. Unmapped players are a visible gap, never a dropped row.
- **Prediction market coverage is thin and uneven.** Most nights only a handful of players have
  props. Treat market data as a bonus signal that improves a projection when present, never as
  a required input.

## Conventions

- Type hints everywhere; pydantic models at every I/O boundary.
- Pure functions for anything mathematical, so it can be tested without network access.
- Cache all external API responses to disk with timestamps; never hammer an endpoint in a loop.
- Every fantasy-point calculation flows through `engine/model/scoring.py`. One implementation.
- Commit messages: `phase1: <what>`. Keep phases separable, see `docs/TASKS.md`.

## Testing

- Golden tests: hand-scored box scores in `tests/fixtures/` verify `scoring.py` including the
  bonus edge cases and both branches of the ambiguity flags.
- The optimal-stopping module has a closed-form regression test: for a normally distributed
  player with mean 45 and sd 10, optimal weekly lock-in value is ≈45.0 / 49.0 / 51.3 / 52.9
  for 1 / 2 / 3 / 4 game weeks. Simulation must reproduce this within tolerance.
- No test may require network access. Fixtures only.

## What "done" looks like for the user

He is on his phone, mid-draft, with 90 seconds on the clock. He needs one screen: who to take,
how likely each is to survive to his next pick, and one line of why. Everything else is
secondary to that.
