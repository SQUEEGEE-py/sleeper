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

## Current state (2026-09-23)

**M0, M1 and M2 complete.** M3 not started.

What exists and works:

- `uv` project on Python 3.11 with the M0 deps. Run anything with `uv run python -m <module>`.
- Repo skeleton from SPEC §1. Modules belonging to later milestones are **empty placeholder
  files, not stubs** — `lockin.py`, `distributions.py`, `valuation.py`, `lineup.py`,
  `draft_sim.py`, the other `ingest/` modules and the `jobs/` entrypoints are 0 bytes on
  purpose. Don't mistake them for work in progress. (`scoring.py` is real as of M1.)
- `engine/ingest/sleeper.py` — typed pydantic client for the SPEC §2.1 endpoints. Every
  response is cached to disk under `data/cache/sleeper/` with a fetch timestamp; `players/nba`
  is TTL-capped at 24h. Reruns and tests never need the network.
- `engine/jobs/verify_league.py` — diffs live `scoring_settings` and `roster_positions`
  against `config/league.json`, prints a readable diff, exits non-zero on mismatch.

Endpoints probed against the live API on 2026-09-17 per rule 6, using a public league:
`/league/{id}` (`scoring_settings` dict, `roster_positions` list), `/rosters`, `/users`,
`/drafts`, and `/players/nba` (dict keyed by player_id, ~2100 players, 2.4 MB). All match the
shapes the spec describes.

### M0 acceptance test: PASSING as of 2026-09-23

`uv run python -m engine.jobs.verify_league` exits 0 against the real league —
**National Logan League**, `league_id` `1405604996254339072`, 8 teams, 13 roster slots,
scoring confirmed. The first run exited 1 on six mismatches; all six are now resolved and the
resolutions are recorded in `config/league.json`'s `_*_comment` fields:

- **Five scoring keys were named wrong** in the original hand transcription. Sleeper's real
  names are `tpm`, `tf`, `ff`, `bonus_pt_40p`, `bonus_pt_50p`. **Every weight was already
  correct** — only the spellings differed, and the live payload has exactly 13 keys with no
  extras. `config/league.json` now stores Sleeper's spelling verbatim so `verify_league`
  compares with no translation; `scoring.py` holds the one verified
  `STAT_FIELD_TO_SCORING_KEY` table and the rest of the engine speaks readable field names.
- **The league has no IR slots.** The transcription assumed 2. Confirmed twice (league
  `roster_positions` and the draft object's `slots_*`). See "What the live league changed".

### The league itself

**National Logan League**, `league_id` `1405604996254339072`, draft `1405604996266901504`
(snake, 13 rounds, 90s pick timer, `pre_draft`). The user is **`Squ33gee`, roster_id 4**. The
other seven managers, who are the opponent models in SPEC §6.2: `tristantro`, `amancito`,
`2jake4davis6`, `aala1aala2`, `cyapp1`, `gilbobo`, `ChickenJoe2107`.

`previous_league_id` `1275683545209122816` is last season's league with the same managers.
Nothing carries over (redraft), but its **completed draft is replayable data** — it answers
SPEC §9.6's dress-rehearsal question and could seed per-coach priors with observed behaviour
instead of a generic prior. Not fetched yet.

### Blockers, in priority order

1. **Draft date and the user's slot are still unknown.** `start_time` is null and
   `draft_order` is null on draft `1405604996266901504` (status `pre_draft`). Both appear as
   soon as the commissioner schedules it. `pick_timer` is 90s, which is the budget the live
   advisor has to beat.
2. **Both `scoring_ambiguities` remain unresolved** (`bonuses_stack`, `td_includes_dd`). Per
   SPEC §4 the empirical fix is to recompute real week-1 box scores and diff against Sleeper's
   own totals, which is impossible before the season opens. Both branches stay behind flags.
   Resolving them is a one-line config change, not a code change.

### What the live league changed vs docs/SPEC.md

The spec was written from screenshots and is wrong in three places, plus one thing it never
mentioned. `config/league.json` has
been corrected; **the spec has not been rewritten yet.**

- **No IR slots.** SPEC §7.2's IR-stash strategy (flagging injured stars who cost no active
  roster spot) does not apply. Delete it rather than implementing it.
- **Waivers are FAAB**, `waiver_budget: 100`, not rolling priority. SPEC §7.3 ranks waiver
  targets but says nothing about bid sizing, which is now a real modelling question.
- **Two divisions**, which the spec never mentions. Check how they affect the top-4 playoff
  cut that the draft simulator's `threat[team]` is built around.
- **No keepers, despite `max_keepers: 1`.** That field is an inert Sleeper default here;
  `settings.type` is 0 (redraft) and all 8 pre-draft rosters are empty. Every player is
  draftable. Recorded in `config/league.json` so it does not get reopened from `max_keepers`
  alone.
- Non-issue, but do not be alarmed by it: the league-level `settings.draft_rounds` is `3`. It
  is a stale default. The draft object's `rounds: 13` is the real value, giving the 104 picks
  the spec assumes.

### M1 — scoring engine (done)

`engine/model/scoring.py` implements SPEC §4. `uv run pytest tests/test_scoring.py` is green:
89 tests, 10 hand-scored fixtures × both settings of both ambiguity flags, no network.

How it is put together, so it does not get re-litigated:

- Weights are a **signed dot product** over `config["scoring"]`. The config already carries the
  minus signs on turnovers and fouls, so the module applies no sign of its own and hardcodes no
  number. The only hardcoded things are basketball: the five countable categories, and that a
  "double" is 10.
- Points-bonus thresholds are **parsed from the config key names** (`bonus_pt_(\d+)p`), so a
  commissioner adding `bonus_pt_60p` needs no code change. There is a test for that.
- `load_rules` fails loudly on a missing weight, an **unrecognised** scoring key, or an unknown
  `quadruple_double` policy. The last is validated at load time specifically so a quad-double
  can never crash a live nightly job.
- `score_breakdown()` returns the itemised parts for the UI's one-line "why"; `score_box()` is
  the total. Everything downstream calls one of those two.
- The expected values in `tests/fixtures/box_scores.json` were hand-computed and user-reviewed
  before the test file was written. **They are the reference — if scoring.py disagrees,
  scoring.py is wrong. Never regenerate them from code output.** No test recomputes a total
  from the weights, since a test that reimplements the formula only proves it equals itself.
- Verified by mutation: flipping non-stacking to pay the lowest threshold, and disabling the dd
  bonus, each fail 5 and 8 tests respectively.

### M2 — historical ingest (done)

`engine/ingest/nba_stats.py` + `engine/jobs/ingest_history.py`. Run it by hand:
`uv run python -m engine.jobs.ingest_history [--force]`. 22 tests, no network.

On disk after a run (all gitignored and regenerable): **79,358 game-log rows** across 2023-24,
2024-25 and 2025-26, plus the 2026-27 schedule, week grid, and the team x week games grid that
M2's acceptance test asks for.

What was learned building it, so it is not rediscovered:

- **cdn.nba.com is blocked from this machine** (Akamai 403, with and without browser headers),
  so SPEC §2.2's lighter-weight schedule source is unavailable. stats.nba.com works fine and
  supplies both the logs and, via `ScheduleLeagueV2`, the schedule and week grid.
- **The NBA's own week numbering IS this league's fantasy week numbering** — week 1 starts
  2026-10-20 matching config `season_start`, every later week runs Monday to Sunday, 25 weeks
  covering the week-22 fantasy playoffs. This is load-bearing for M3, so it is *verified at
  ingest time* by `verify_weeks_match_fantasy_weeks`, which fails the job rather than
  assuming. Do not replace it with a hand-rolled week calculation.
- **Two real data quirks, both fixture-tested.** `gameDate` is a US-format string, so sorting
  it as text puts January before December and corrupts every week assignment. And the week
  frame arrives unsorted with week 8 listed twice, identically — identical rows are collapsed,
  but rows that disagree about a week's dates are deliberately left to fail the verifier.
- **Two games per team are unscheduled** (6 placeholder rows with both sides TBD, plus 24
  games not yet listed; every team has 80 of 82). NBA Cup. `schedule_completeness()` reports
  it and the job prints it. Games-per-week in that window is a floor, not a count — which
  matters because games per week feeds lock-in value directly.
- **Technicals and flagrants are not in `PlayerGameLogs`.** The league scores them; the
  endpoint only has personal fouls. Stored as 0.0 and listed in `GAME_LOG_MISSING_STATS` so
  the UI can mark them unmeasured. Recovering them needs play-by-play.

Sanity check, rescoring real 2025-26 logs through `scoring.py`: the top of the board is
Dončić 68.9, Jokić 67.8, SGA 59.1, Wembanyama 57.0 mean fp/game, league mean 23.3 — exactly
the shape this scoring system should produce. And **Jokić is the player most affected by the
two unresolved ambiguity flags** (+1.66 fp/game between the extremes), which is precisely what
SPEC §4 predicted.

### Next step — M3 (distributions and lock-in value)

Per `docs/TASKS.md`. The regression target is in "Testing" below. Everything M3 needs is now
on disk; it should not need the network at all.

**Supabase is not set up.** There is no `.env`, so `ingest_history` writes parquet only and
says so. TASKS M2 asks for parquet *and* Supabase; the Supabase half is deferred, not done.

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
- **Windows Smart App Control is ON in enforcement mode on the dev machine.** It blocks
  compiled `.pyd`/`.dll` files that are neither signed nor known-good to Microsoft's
  reputation service, and the failure looks like
  `ImportError: DLL load failed ... An Application Control policy has blocked this file`.
  Reputation is **per file hash**, so a brand-new release of a package can be blocked while
  the previous release and every other package load fine — that is exactly what happened with
  pandas 3.0.6 (blocked) vs 2.3.3 (fine), which is why `pandas` is pinned `<3` in
  `pyproject.toml`. If a new dependency fails to import this way, check
  `Microsoft-Windows-CodeIntegrity/Operational` in Event Viewer for the verdict, then try a
  slightly older release before anything drastic. **Do not suggest turning Smart App Control
  off:** it cannot be turned back on without reinstalling Windows.
  - Known-blocked and accepted: `_hashlib.pyd` in uv's own unsigned CPython build
    (`AppData\Roaming\uv\python\...`). Cosmetic — `hashlib` falls back to Python's builtin
    SHA implementations, `_ssl.pyd` loads, and HTTPS to Sleeper works. It only produces a
    Windows toast. The clean fix, if it ever becomes worth it, is a python.org 3.11 install
    (PSF-signed) plus `tool.uv.python-preference = "only-system"`.

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
